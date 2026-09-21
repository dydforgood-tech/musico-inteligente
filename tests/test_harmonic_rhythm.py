"""Ritmo harmônico em beats, aprendizado intra-sessão e localização temporal."""
import unittest
from collections import deque

from app.instruments.bass_model import BassPatternType
from app.instruments.bass_player import BassPlayer
from app.music.chart_alignment import ChartAlignment
from app.music.chord_chart import ChartChord, ChordChart, ChartSection, parse_chord
from app.music.harmonic_rhythm import HarmonicRhythmEvent, HarmonicRhythmTracker, quantize_beats
from app.music.musical_clock import MusicalClock
from app.music.musical_context import MusicalContext
from app.music.pattern_memory import PatternMemory
from app.music.position_estimator import PositionEstimator
from app.music.prediction_engine import PredictionEngine
from app.music.music_structure import MusicPosition


def section(identifier, name, chords, first_bar):
    return ChartSection(id=identifier, name=name, section_type=name.upper(), chords=[
        ChartChord(symbol=parse_chord(chord), bar=first_bar + i) for i, chord in enumerate(chords)
    ])


class TestHarmonicRhythm(unittest.TestCase):
    def record_verse(self, tracker, start=0.0, variant=(4.0, 2.0, 2.0, 8.0)):
        for chord, offset in zip(("G", "D", "Em", "C"),
                                 (0.0, variant[0], variant[0] + variant[1], variant[0] + variant[1] + variant[2])):
            tracker.observe(start + offset, "Verse 1", chord, .92)
        tracker.observe(start + sum(variant), "Chorus", "G", .92)

    def test_represents_unequal_chord_durations_in_beats(self):
        memory = PatternMemory()
        tracker = HarmonicRhythmTracker(memory)
        self.record_verse(tracker)
        pattern = memory.get_harmonic_rhythm_patterns()[0]
        self.assertEqual(pattern.chord_sequence, ["G", "D", "Em", "C"])
        self.assertEqual(pattern.duration_sequence, [4.0, 2.0, 2.0, 8.0])
        self.assertEqual(pattern.total_beats, 16.0)

    def test_second_verse_reuses_learned_duration_and_quantizes_human_variation(self):
        memory = PatternMemory()
        tracker = HarmonicRhythmTracker(memory)
        self.record_verse(tracker, variant=(4.0, 2.0, 2.0, 8.0))
        tracker.observe(20.0, "Verse 2", "G", .90)
        learned = tracker.observe(23.85, "Verse 2", "G", .90)
        self.assertEqual(learned.pattern_id.startswith("VERSE_RHYTHM"), True)
        self.assertAlmostEqual(learned.expected_chord_duration_beats, 4.0)
        self.assertGreater(learned.pattern_confidence, .5)
        self.assertEqual(quantize_beats(3.86), 4.0)
        self.assertEqual(quantize_beats(1.94), 2.0)

    def test_unknown_frames_keep_one_sustained_event(self):
        memory = PatternMemory()
        tracker = HarmonicRhythmTracker(memory)
        for beat, chord in ((0, "G"), (1, "G"), (2, "--"), (2.5, "UNKNOWN"), (3, "G"), (4, "D")):
            tracker.observe(beat, "Verse", chord, .90 if chord not in ("--", "UNKNOWN") else 0.0)
        event = tracker.timeline[0]
        self.assertEqual(event.symbol, "G")
        self.assertEqual(event.quantized_duration_beats, 4.0)

    def test_duration_disambiguates_identical_intro_and_verse_progressions(self):
        chart = ChordChart(sections=[
            section("intro", "Intro", ["G", "D", "Em", "C"], 1),
            section("verse", "Verse", ["G", "D", "Em", "C"], 5),
        ])
        memory = PatternMemory()
        memory.register_harmonic_rhythm("INTRO", ["G", "D", "Em", "C"], [8, 8, 8, 8], .92)
        memory.register_harmonic_rhythm("VERSE", ["G", "D", "Em", "C"], [4, 2, 2, 4], .92)

        class Structure:
            pattern_memory = memory
        estimator = PositionEstimator(ChartAlignment(chart), MusicalClock(120), structure_analyzer=Structure())
        estimator._recent_detected = deque(["G", "D", "Em"], maxlen=8)
        estimator._recent_harmonic_rhythm = deque([
            (("G", 0.0), HarmonicRhythmEvent("G", 0, 4, 4, .9, "VERSE")),
            (("D", 4.0), HarmonicRhythmEvent("D", 4, 2, 2, .9, "VERSE")),
        ], maxlen=4)
        location = estimator.localize(near_bar=2)
        self.assertEqual(location[0], 7)  # Em do Verse; não permanece na Intro.
        self.assertGreater(location[1], .9)

    def test_prediction_uses_beats_until_learned_change(self):
        ctx = MusicalContext(current_chord_elapsed_beats=3.4,
                             expected_chord_duration_beats=4.0,
                             beats_until_chord_change=.6,
                             rhythmic_next_chord="D", duration_confidence=.9,
                             pattern_confidence=.8, harmonic_rhythm_pattern="VERSE_RHYTHM_01")
        prediction = PredictionEngine().predict_next(MusicPosition(current_bar=1, current_beat=4),
                                                      None, PatternMemory(), harmonic_rhythm=ctx)
        self.assertEqual(prediction.predicted_chords, ["D"])
        self.assertAlmostEqual(prediction.beats_until_change, .6)

    def test_bass_prepares_learned_intra_bar_change(self):
        bass = BassPlayer(sample_rate=1000, pattern=BassPatternType.ROOT)
        ctx = MusicalContext(timestamp=1.25, chord="G", bar=1, beat=3, beat_position=.5,
                             bpm=120, meter="4/4", chart_available=True,
                             tempo_tracking_state="TRACKING", position_confidence=.95,
                             follow_confidence_level="MEDIUM", rhythmic_next_chord="D",
                             beats_until_chord_change=.5, duration_confidence=.9,
                             pattern_confidence=.9)
        event = bass.on_musical_context(ctx)
        self.assertTrue(event.note.startswith("D"))
        self.assertEqual(bass.synthesizer.scheduled_events[0].source, "chart-pattern")

    def test_beats_remain_stable_while_bpm_changes(self):
        clock = MusicalClock(120)
        beats = []
        timestamp = 0.0
        for bpm in (120, 124, 128, 123, 118):
            timestamp += 60.0 / bpm
            clock.update(timestamp, bpm=bpm)
            beats.append(clock.total_beats)
        self.assertGreater(beats[-1], 4.7)
        self.assertLess(beats[-1], 5.3)


if __name__ == "__main__":
    unittest.main()
