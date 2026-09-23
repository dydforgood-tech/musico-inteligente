"""Regressões do ouvido polifônico e da duração observada dos acordes."""
import unittest

import numpy as np

from app.analysis.active_notes import ActiveNoteTracker
from app.analysis.audio_analyzer import AudioAnalyzer
from app.analysis.chord_detector import ChordDetector
from app.instruments.bass_model import BassPatternType
from app.instruments.bass_player import BassPlayer
from app.music.harmonic_rhythm import HarmonicRhythmTracker
from app.music.musical_context import MusicalContext
from app.music.pattern_memory import PatternMemory
from app.music.theory import PITCH_CLASSES
from app.song.song import Song
from app.song.song_session import SongSession
from app.utils.audio_generator import generate_chord, generate_tone


def chroma(**notes):
    vector = np.zeros(12, dtype=np.float32)
    for note, strength in notes.items():
        vector[PITCH_CLASSES.index(note)] = strength
    return vector


class TestPolyphonicHearing(unittest.TestCase):
    def setUp(self):
        self.notes = ActiveNoteTracker()
        self.chords = ChordDetector()
        self.silent_audio = np.zeros(1024, dtype=np.float32)

    def hear(self, values, at, key="--", expected=None):
        state = self.notes.update(chroma(**values), at, key=key)
        chord = self.chords.detect(state.chroma, self.silent_audio, 44100,
                                   timestamp=at, expected_chord=expected, key=key)
        return state, chord

    def test_single_note_remains_note_not_fabricated_chord(self):
        state, chord = self.hear({"A": 1.0}, 0.0, key="C Major", expected="Am")
        self.assertEqual(list(state.notes), ["A"])
        self.assertEqual(chord.symbol, "--")

    def test_real_audio_pipeline_does_not_promote_monophonic_pitch_to_chord(self):
        analyzer = AudioAnalyzer()
        duration = 4096 / 44100
        chord_audio = generate_chord([164.81, 196.0, 261.63], duration)
        chord = analyzer.analyze_chunk(chord_audio, 44100, .1)
        self.assertEqual(chord.raw_detected_chord.split("/")[0], "C")
        analyzer.reset_musical_history()
        single = analyzer.analyze_chunk(generate_tone(220.0, duration), 44100, .1)
        self.assertEqual(single.raw_note, "A3")
        self.assertEqual(set(single.active_notes), {"A"})
        self.assertEqual(single.raw_detected_chord, "--")

    def test_chords_and_inversion_preserve_harmonic_root(self):
        for values, expected in (({"C": 1, "E": .8, "G": .9}, "C"),
                                 ({"D": 1, "F#": .8, "A": .9}, "D"),
                                 ({"A": 1, "C": .8, "E": .9}, "Am")):
            self.notes.reset()
            self.chords = ChordDetector()
            state, chord = self.hear(values, 0.0)
            self.assertTrue(set(values).issubset(state.notes))
            self.assertEqual(chord.root, expected.rstrip("m"))
        self.notes.reset()
        self.chords = ChordDetector()
        _, chord = self.hear({"E": 1, "G": .9, "C": .8}, 0.0)
        self.assertEqual(chord.root, "C")

    def test_arpeggio_accumulates_notes_without_waiting_for_bar(self):
        self.hear({"C": 1}, 0.0)
        self.hear({"E": 1}, .10)
        state, chord = self.hear({"G": 1}, .20)
        self.assertEqual(set(state.notes), {"C", "E", "G"})
        self.assertEqual(chord.root, "C")

    def test_wrong_chart_expectation_cannot_override_strong_audio(self):
        _, chord = self.hear({"C": 1, "E": .9, "G": .9}, 0.0,
                             key="G Major", expected="F#m")
        self.assertEqual(chord.root, "C")

    def test_percussive_broadband_frame_does_not_create_twelve_notes(self):
        state, chord = self.hear(dict.fromkeys(PITCH_CLASSES, 1.0), 0.0)
        self.assertFalse(state.notes)
        self.assertEqual(chord.symbol, "--")

    def test_unknown_expires_old_harmonic_event(self):
        rhythm = HarmonicRhythmTracker(PatternMemory())
        rhythm.observe(0, "Verse", "F#m", .9, observation_available=True)
        rhythm.observe(.5, "Verse", "F#m", .9, observation_available=True)
        state = rhythm.observe(1.6, "Verse", "--", 0, observation_available=True)
        self.assertEqual(state.current_chord, "--")
        self.assertEqual(rhythm.timeline[-1].end_reason, "UNSUPPORTED")
        self.assertLess(rhythm.timeline[-1].raw_duration_beats, 1.6)
        state = rhythm.observe(1.7, "Verse", "A", .9, observation_available=True)
        self.assertEqual(state.current_root, "A")

    def test_confirmed_onset_preserves_real_duration_in_beats(self):
        rhythm = HarmonicRhythmTracker(PatternMemory())
        for now, chord, onset in ((.2, "G", 0), (4.2, "D", 4),
                                   (6.2, "Em", 6), (8.2, "C", 8)):
            rhythm.observe(now, "Verse", chord, .9, observation_available=True,
                           transition_beat=onset)
        rhythm.observe(16, "Chorus", "G", .9, observation_available=True)
        self.assertEqual([ev.quantized_duration_beats for ev in rhythm.timeline],
                         [4, 2, 2, 8])

    def test_live_chart_waits_for_next_chord_even_with_capo_metadata(self):
        session = SongSession(Song(title="Capo domain", bpm=120,
                                   chart_text="[Intro]\nD#m C#4 B9 F#"))
        session.position_estimator.set_capo(2)
        for time, chord in ((.1, "D#m"), (.3, "D#m"), (1, "F#"), (1.2, "F#")):
            source = MusicalContext(audio_activity=.1, smoothed_detected_chord=chord,
                                    stable_chord_confidence=.9, chord_start_time=time)
            session.update_audio_tick(time, detected_chord=chord,
                                      detected_confidence=.9, source_context=source)
        self.assertEqual(session.chart_position.current_chord, "D#m")
        for time in (2.0, 2.2):
            source = MusicalContext(audio_activity=.1, smoothed_detected_chord="C#",
                                    stable_chord_confidence=.9, chord_start_time=2.0)
            session.update_audio_tick(time, detected_chord="C#",
                                      detected_confidence=.9, source_context=source)
        self.assertEqual(session.chart_position.current_chord, "C#4")
        self.assertEqual(session.position_estimator.chart_cursor_index, 1)

    def test_bass_waits_when_live_harmony_is_unknown_and_position_is_weak(self):
        bass = BassPlayer(sample_rate=1000, pattern=BassPatternType.ROOT)
        context = MusicalContext(timestamp=1.0, chord="F#", audio_activity=.1,
                                 smoothed_detected_chord="--", position_confidence=.4,
                                 tempo_tracking_state="TRACKING", bar=1, beat=3,
                                 bpm=120, meter="4/4")
        self.assertIsNone(bass.on_musical_context(context))
        self.assertFalse(bass.synthesizer.scheduled_events)


if __name__ == "__main__":
    unittest.main()
