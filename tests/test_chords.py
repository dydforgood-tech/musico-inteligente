"""Testes automatizados da camada de análise harmônica e detecção de acordes (v0.1-C)."""

import unittest
import numpy as np

from app.analysis.chord_detector import ChordDetector, ChordHistory, Chord
from app.analysis.harmonic_analyzer import DefaultHarmonicAnalyzer
from app.analysis.chroma_extractor import ChromaExtractor
from app.utils.audio_generator import generate_tone, generate_chord


class TestChordAnalysis(unittest.TestCase):
    """Validação da detecção de tríades maiores, menores, inversões e histórico."""

    @classmethod
    def setUpClass(cls):
        cls.sr = 44100
        cls.detector = ChordDetector()
        cls.chroma = ChromaExtractor()
        cls.harmonic_analyzer = DefaultHarmonicAnalyzer()

    def _create_chord_audio(self, freqs: list, duration: float = 0.5) -> np.ndarray:
        return generate_chord(freqs, duration=duration, sr=self.sr)

    def test_detect_c_major(self):
        """Valida identificação do acorde de C Maior (C-E-G)."""
        # C4 (261.63), E4 (329.63), G4 (392.00)
        audio = self._create_chord_audio([261.63, 329.63, 392.00])
        chr_vec = self.chroma.extract(audio, self.sr)
        chord = self.detector.detect(chr_vec, audio, self.sr, timestamp=1.0)

        self.assertEqual(chord.root, "C")
        self.assertEqual(chord.quality, "major")
        self.assertTrue(chord.symbol.startswith("C"))
        self.assertGreater(chord.confidence, 0.70)
        self.assertIn("C", chord.detected_notes)
        self.assertIn("E", chord.detected_notes)
        self.assertIn("G", chord.detected_notes)

    def test_detect_a_minor(self):
        """Valida identificação do acorde de A Menor (A-C-E)."""
        # A3 (220.00), C4 (261.63), E4 (329.63)
        audio = self._create_chord_audio([220.00, 261.63, 329.63])
        chr_vec = self.chroma.extract(audio, self.sr)
        chord = self.detector.detect(chr_vec, audio, self.sr, timestamp=2.0)

        self.assertEqual(chord.root, "A")
        self.assertEqual(chord.quality, "minor")
        self.assertTrue(chord.symbol.startswith("Am"))
        self.assertGreater(chord.confidence, 0.70)

    def test_detect_g_major(self):
        """Valida identificação do acorde de G Maior (G-B-D)."""
        # G3 (196.00), B3 (246.94), D4 (293.66)
        audio = self._create_chord_audio([196.00, 246.94, 293.66])
        chr_vec = self.chroma.extract(audio, self.sr)
        chord = self.detector.detect(chr_vec, audio, self.sr, timestamp=3.0)

        self.assertEqual(chord.root, "G")
        self.assertEqual(chord.quality, "major")
        self.assertTrue(chord.symbol.startswith("G"))
        self.assertGreater(chord.confidence, 0.70)

    def test_chord_history_progression(self):
        """Valida armazenamento e encadeamento da progressão harmônica no ChordHistory."""
        history = ChordHistory(max_entries=5)

        history.update(Chord(symbol="C", root="C", timestamp=0.0, duration=0.5))
        history.update(Chord(symbol="C", root="C", timestamp=0.5, duration=0.5))
        history.update(Chord(symbol="G", root="G", timestamp=1.0, duration=1.0))
        history.update(Chord(symbol="Am", root="A", timestamp=2.0, duration=1.0))

        recent = history.get_recent()
        self.assertEqual(len(recent), 3)  # C, G, Am
        self.assertEqual(recent[0].symbol, "C")
        self.assertEqual(recent[1].symbol, "G")
        self.assertEqual(recent[2].symbol, "Am")

        summary = history.get_summary_text()
        self.assertIn("C", summary)
        self.assertIn("G", summary)
        self.assertIn("Am", summary)


if __name__ == "__main__":
    unittest.main()
