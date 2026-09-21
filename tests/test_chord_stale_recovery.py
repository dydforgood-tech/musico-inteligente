"""Regressões do acorde estabilizado: suporte recente, votação e prior da cifra."""

import unittest
from unittest.mock import Mock

import numpy as np

from app.analysis.audio_analyzer import AudioAnalyzer
from app.analysis.chord_detector import Chord, ChordDetector
from app.analysis.chord_history import ChordStabilizer
from app.music.context_manager import MusicalContextManager
from app.music.theory import PITCH_CLASSES


def chroma_for(root: str, quality: str = "major") -> np.ndarray:
    chroma = np.zeros(12, dtype=np.float32)
    root_index = PITCH_CLASSES.index(root)
    intervals = (0, 3, 7) if quality == "minor" else (0, 4, 7)
    for strength, interval in zip((1.0, .9, .8), intervals):
        chroma[(root_index + interval) % 12] = strength
    return chroma


class TestChordStaleRecovery(unittest.TestCase):
    def make_stabilizer(self):
        return ChordStabilizer(min_confidence=.4, confirmation_time=.15)

    def seed_f_sharp(self, stabilizer):
        for timestamp in (0.0, .05, .10, .15):
            stabilizer.process("F#", .90, timestamp, root="F#", quality="major",
                               chroma_vector=chroma_for("F#"), audio_activity=.08)

    def test_wrong_f_sharp_cannot_survive_a_family_oscillation(self):
        stabilizer = self.make_stabilizer()
        self.seed_f_sharp(stabilizer)
        variants = (
            ("A", "A", "major", "A"),
            ("A7", "A", "major", "A"),
            ("A", "A", "major", "A"),
            ("E/A", "E", "major", "A"),
            ("A", "A", "major", "A"),
            ("Asus4", "A", "sus4", "A"),
            ("A", "A", "major", "A"),
        )
        trace = []
        for index in range(50):
            symbol, root, quality, bass = variants[index % len(variants)]
            timestamp = .20 + index * .10
            stable = stabilizer.process(
                symbol, .62, timestamp, root=root, quality=quality,
                bass_note=bass, chroma_vector=chroma_for("A"), audio_activity=.08)
            trace.append(stable)
        self.assertNotEqual(trace[-1], "F#")
        self.assertTrue(trace[-1] == "--" or stabilizer.chord_root == "A")
        self.assertLess(stabilizer.time_since_current_chord_support, 1.0)

    def test_single_wrong_frame_keeps_hysteresis(self):
        stabilizer = self.make_stabilizer()
        self.seed_f_sharp(stabilizer)
        stabilizer.process("A", .72, .20, root="A", quality="major",
                           chroma_vector=chroma_for("A"), audio_activity=.08)
        for timestamp in (.25, .30):
            stabilizer.process("F#", .90, timestamp, root="F#", quality="major",
                               chroma_vector=chroma_for("F#"), audio_activity=.08)
        self.assertEqual(stabilizer.current_chord, "F#")
        self.assertGreater(stabilizer.chord_confidence, .70)

    def test_active_audio_without_current_support_expires_old_chord(self):
        stabilizer = self.make_stabilizer()
        self.seed_f_sharp(stabilizer)
        unrelated = (("A", "A"), ("E", "E"), ("C#m", "C#"), ("D", "D"))
        for index in range(20):
            symbol, root = unrelated[index % len(unrelated)]
            stabilizer.process(
                symbol, .46, .20 + index * .10, root=root,
                quality="minor" if symbol.endswith("m") else "major",
                chroma_vector=chroma_for(root, "minor" if symbol.endswith("m") else "major"),
                audio_activity=.08)
        self.assertEqual(stabilizer.current_chord, "--")
        self.assertEqual(stabilizer.chord_confidence, 0.0)
        self.assertTrue(stabilizer.is_current_chord_stale)

    def test_context_never_publishes_expired_stable_chord(self):
        manager = MusicalContextManager()
        for timestamp in (0.0, .05, .10):
            ctx = manager.update(
                timestamp, 44100,
                chord_res=Chord(root="F#", quality="major", symbol="F#", confidence=.9),
                chroma_vector=chroma_for("F#"), audio_activity=.08)
        for index, (symbol, root) in enumerate((("A", "A"), ("E", "E"),
                                                ("C#m", "C#"), ("D", "D")) * 5):
            ctx = manager.update(
                .20 + index * .10, 44100,
                chord_res=Chord(root=root, quality="major", symbol=symbol, confidence=.46),
                chroma_vector=chroma_for(root), audio_activity=.08)
        self.assertEqual(ctx.chord, "--")
        self.assertEqual(ctx.smoothed_detected_chord, "--")

    def test_strong_a_audio_beats_wrong_f_sharp_chart_prior(self):
        detector = ChordDetector()
        chord = detector.detect(chroma_for("A"), np.zeros(1024, dtype=np.float32),
                                44100, timestamp=1.0, expected_chord="F#")
        self.assertEqual(chord.root, "A")

    def test_chart_prior_is_disabled_when_position_is_not_reliable(self):
        analyzer = AudioAnalyzer()
        analyzer.set_harmonic_expectation("F#", "F# Major")
        expected, enabled = analyzer._harmonic_prior_for(
            Mock(tracking_state="UNCERTAIN", position_confidence=.45))
        self.assertIsNone(expected)
        self.assertFalse(enabled)
        expected, enabled = analyzer._harmonic_prior_for(
            Mock(tracking_state="TRACKING", position_confidence=.92))
        self.assertEqual(expected, "F#")
        self.assertTrue(enabled)


if __name__ == "__main__":
    unittest.main()
