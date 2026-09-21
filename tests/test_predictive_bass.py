"""Agendamento do baixo pela cifra antes da confirmação do áudio."""
import unittest

import numpy as np

from app.instruments.bass_model import BassPatternType
from app.instruments.bass_player import BassPlayer
from app.instruments.registry import VirtualPlayerRegistry
from app.music.musical_context import MusicalContext
from app.music.harmonic_rhythm import HarmonicRhythmEvent
from app.music.position_estimator import TrackingState
from app.song.song import Song
from app.song.song_session import SongSession


class TestPredictiveBass(unittest.TestCase):
    def session(self, text="C\nAm\nF\nG"):
        return SongSession(Song(title="Predictive bass", chart_text=text, bpm=120))

    def registry(self):
        bass = BassPlayer(sample_rate=1000, pattern=BassPatternType.ROOT)
        return bass, VirtualPlayerRegistry(bass_player=bass)

    def test_chart_predicts_all_downbeats_without_detected_chords(self):
        session = self.session()
        bass, registry = self.registry()
        for timestamp, chord, note, next_bar in [(1.75, "C", "A", 2),
                                                  (3.75, "Am", "F", 3),
                                                  (5.75, "F", "G", 4)]:
            session.update_audio_tick(timestamp, detected_chord="--")
            self.assertEqual(session.current_chord, chord)
            self.assertEqual(session.context.next_change_bar, next_bar)
            event = registry.dispatch_context(session.context)["bass"]
            self.assertEqual((event.bar, event.beat), (next_bar, 1))
            self.assertTrue(event.note.startswith(note))
            self.assertAlmostEqual(event.start_time, next_bar * 2 - 2)
            prepared = bass.synthesizer.scheduled_events[-1]
            self.assertEqual(prepared.source, "chart")
            self.assertEqual(prepared.scheduled_beat, (next_bar, 1))
        self.assertEqual(bass.lookahead["next_change_position"], (4, 1))

    def test_wrong_audio_observation_does_not_override_next_chart_chord(self):
        session = self.session()
        bass, registry = self.registry()
        session.update_audio_tick(1.75, detected_chord="D", detected_confidence=.95)
        event = registry.dispatch_context(session.context)["bass"]
        self.assertTrue(event.note.startswith("A"))
        self.assertEqual(bass.synthesizer.scheduled_events[0].source, "chart")

    def test_sustained_chord_does_not_change_one_bar_early(self):
        original = self.session("C\nAm")
        chart = original.chart
        chart.sections[0].chords[0].duration_bars = 2
        session = SongSession(Song(title="Sustain", chart_data=chart.to_dict(), bpm=120))
        bass, registry = self.registry()
        session.update_audio_tick(1.75)
        self.assertEqual(session.context.next_change_bar, 3)
        event = registry.dispatch_context(session.context)["bass"]
        self.assertEqual((event.bar, event.beat), (2, 1))
        self.assertTrue(event.note.startswith("C"))
        session.update_audio_tick(3.75)
        event = registry.dispatch_context(session.context)["bass"]
        self.assertEqual((event.bar, event.beat), (3, 1))
        self.assertTrue(event.note.startswith("A"))

    def test_tempo_change_retimes_the_same_future_event(self):
        bass, registry = self.registry()
        ctx = MusicalContext(timestamp=1.75, chord="C", next_expected_chord="Am",
                             next_change_bar=2, bar=1, beat=4, beat_position=.5,
                             bpm=120, tempo_tracking_state="TRACKING")
        registry.dispatch_context(ctx)
        first = bass.synthesizer.scheduled_events[0]
        ctx.timestamp = 1.78
        ctx.bpm = 100
        event = registry.dispatch_context(ctx)["bass"]
        self.assertIsNotNone(event)
        self.assertEqual(len(bass.synthesizer.scheduled_events), 1)
        self.assertGreater(bass.synthesizer.scheduled_events[0].scheduled_time,
                           first.scheduled_time + .05)

    def test_reanchor_invalidates_previous_generation(self):
        session = self.session()
        bass, registry = self.registry()
        session.update_audio_tick(1.75)
        registry.dispatch_context(session.context)
        old = bass.synthesizer.scheduled_events[0]
        session.seek_to_bar(4)
        registry.dispatch_context(session.context)
        events = bass.synthesizer.scheduled_events
        self.assertTrue(events)
        self.assertTrue(all(ev.generation_id == session.context.position_generation for ev in events))
        self.assertNotIn(old, events)

    def test_tracking_loss_is_silent_until_position_is_confirmed(self):
        bass, registry = self.registry()
        ctx = MusicalContext(timestamp=1.75, chord="C", next_expected_chord="Am",
                             next_change_bar=2, bar=1, beat=4, beat_position=.5,
                             bpm=120, tempo_tracking_state="HOLDOVER",
                             tracking_state="LOST", position_confidence=.95,
                             confidence=.95, follow_confidence_level="HIGH")
        events = registry.dispatch_context(ctx)
        self.assertNotIn("bass", events)
        self.assertEqual(bass.synthesizer.scheduled_events, [])

    def test_confirmed_variation_adapts_future_decision_without_single_hit_switch(self):
        bass, registry = self.registry()
        ctx = MusicalContext(timestamp=.25, chord="C", bar=1, beat=1,
                             beat_position=.5, bpm=120, tempo_tracking_state="TRACKING")
        first = registry.dispatch_context(ctx)["bass"]
        self.assertTrue(first.note.startswith("C"))
        ctx.confirmed_variation_chord = "Dm"
        second = registry.dispatch_context(ctx)["bass"]
        self.assertTrue(second.note.startswith("D"))
        self.assertEqual(bass.synthesizer.scheduled_events[0].source, "audio-confirmed")

    def test_fusion_confirms_variation_only_after_sustained_evidence(self):
        session = self.session("C\nC\nC\nC\nC")
        bass, registry = self.registry()
        for timestamp in (.1, 1.1):
            session.update_audio_tick(timestamp, detected_chord="Dm", detected_confidence=.95)
        self.assertEqual(session.context.confirmed_variation_chord, "--")
        session.update_audio_tick(3.4, detected_chord="Dm", detected_confidence=.95)
        self.assertEqual(session.context.confirmed_variation_chord, "Dm")
        event = registry.dispatch_context(session.context)["bass"]
        self.assertTrue(event.note.startswith("D"))

    def test_audio_reanchor_cancels_a_prepared_chord(self):
        chart = "C\n" * 29 + "Am\nF\nG\nC\nDm\nBb\nE7\n" + "C\n" * 12
        session = self.session(chart)
        bass, registry = self.registry()
        session.update_audio_tick(38.0, detected_chord="C", detected_confidence=.9)
        registry.dispatch_context(session.context)
        old_generation = session.context.position_generation
        session.position_estimator._tracking_state = TrackingState.LOST
        session.position_estimator._startup_lock = False
        for event_index, symbol in enumerate(("C", "Dm", "Bb", "E7")):
            rhythm_event = HarmonicRhythmEvent(
                symbol, event_index * 4.0, 4.0, 4.0, .95, "TEST")
            session.position_estimator._recent_harmonic_rhythm.append(
                ((rhythm_event.symbol, rhythm_event.start_beat), rhythm_event))
        for index, chord in enumerate(("Dm", "Bb", "E7")):
            session.update_audio_tick(38.1 + index * .05,
                                      detected_chord=chord, detected_confidence=.9)
        self.assertEqual(session.current_bar, 36)
        self.assertGreater(session.context.position_generation, old_generation)
        registry.dispatch_context(session.context)
        self.assertTrue(all(e.generation_id == session.context.position_generation
                            for e in bass.synthesizer.scheduled_events))

    def test_transport_seek_invalidates_prepared_time(self):
        session = self.session()
        bass, registry = self.registry()
        session.update_audio_tick(1.75)
        registry.dispatch_context(session.context)
        previous = bass.synthesizer.scheduled_events[0]
        session.update_audio_tick(.5)
        registry.dispatch_context(session.context)
        self.assertGreater(session.context.position_generation, previous.generation_id)
        self.assertTrue(all(e.generation_id == session.context.position_generation
                            for e in bass.synthesizer.scheduled_events))

    def test_queue_is_bounded_and_rejects_old_generation(self):
        bass, _ = self.registry()
        synth = bass.synthesizer
        synth.set_generation(3)
        for n in range(12):
            synth.schedule_note((n + 1, 1), n + 1.0, 48, 100, .3,
                                generation_id=3, note="C2")
        self.assertLessEqual(len(synth.scheduled_events), 8)
        synth.set_generation(4)
        synth.set_generation(3)
        synth.schedule_note((1, 1), 1.0, 48, 100, .3, generation_id=3)
        self.assertEqual(synth.scheduled_events, [])

    def test_downbeat_is_rendered_at_scheduled_sample(self):
        session = self.session()
        bass, registry = self.registry()
        session.update_audio_tick(1.75)
        registry.dispatch_context(session.context)
        audio = bass.synthesizer.render_chunk(200, 1000, 1.9)
        self.assertTrue(np.allclose(audio[:99], 0))
        self.assertGreater(np.max(abs(audio[110:])), 0)


if __name__ == "__main__":
    unittest.main()
