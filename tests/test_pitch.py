"""Testes automatizados da camada de análise de pitch (f0) e cromagrama (v0.1-B)."""

import unittest
import numpy as np

from app.analysis.pitch_detector import AutocorrelationPitchDetector, HPSPitchDetector, PitchResult
from app.analysis.chroma_extractor import ChromaExtractor
from app.analysis.audio_analyzer import AudioAnalyzer
from app.utils.audio_generator import generate_tone


class TestPitchAnalysis(unittest.TestCase):
    """Validação de precisão de pitch para senóides e instrumentos harmônicos."""

    @classmethod
    def setUpClass(cls):
        cls.sr = 44100
        cls.chunk_size = 4096
        cls.autocorr = AutocorrelationPitchDetector(fmin=35.0, fmax=1800.0)
        cls.hps = HPSPitchDetector(fmin=35.0, fmax=1800.0)
        cls.chroma = ChromaExtractor()
        cls.analyzer = AudioAnalyzer(sample_rate=cls.sr, chunk_size=cls.chunk_size)

    def _generate_test_chunk(self, freq_hz: float, harmonics: int = 1) -> np.ndarray:
        return generate_tone(freq_hz, duration=self.chunk_size / self.sr, sr=self.sr, harmonics=harmonics)

    def test_pitch_autocorr_a4(self):
        """Valida detecção de A4 (440 Hz) padrão internacional."""
        chunk = self._generate_test_chunk(440.0, harmonics=1)
        res = self.autocorr.detect(chunk, self.sr)
        self.assertTrue(res.is_voiced)
        self.assertEqual(res.note_name, "A")
        self.assertEqual(res.octave, 4)
        self.assertAlmostEqual(res.frequency_hz, 440.0, delta=2.0)
        self.assertGreater(res.confidence, 0.85)

    def test_pitch_autocorr_e2(self):
        """Valida detecção de nota grave E2 (82.41 Hz) - bordão de violão / guitarra."""
        chunk = self._generate_test_chunk(82.41, harmonics=3)
        res = self.autocorr.detect(chunk, self.sr)
        self.assertTrue(res.is_voiced)
        self.assertEqual(res.note_name, "E")
        self.assertEqual(res.octave, 2)
        self.assertAlmostEqual(res.frequency_hz, 82.41, delta=2.0)
        self.assertGreater(res.confidence, 0.75)

    def test_pitch_autocorr_a2(self):
        """Valida detecção de A2 (110 Hz)."""
        chunk = self._generate_test_chunk(110.0, harmonics=2)
        res = self.autocorr.detect(chunk, self.sr)
        self.assertTrue(res.is_voiced)
        self.assertEqual(res.note_name, "A")
        self.assertEqual(res.octave, 2)
        self.assertAlmostEqual(res.frequency_hz, 110.0, delta=2.0)

    def test_pitch_autocorr_g2(self):
        """Valida detecção de G2 (98.0 Hz)."""
        chunk = self._generate_test_chunk(98.0, harmonics=3)
        res = self.autocorr.detect(chunk, self.sr)
        self.assertTrue(res.is_voiced)
        self.assertEqual(res.note_name, "G")
        self.assertEqual(res.octave, 2)
        self.assertAlmostEqual(res.frequency_hz, 98.0, delta=2.0)

    def test_pitch_autocorr_c4(self):
        """Valida detecção de C4 (Dó Central, 261.63 Hz)."""
        chunk = self._generate_test_chunk(261.63, harmonics=1)
        res = self.autocorr.detect(chunk, self.sr)
        self.assertTrue(res.is_voiced)
        self.assertEqual(res.note_name, "C")
        self.assertEqual(res.octave, 4)
        self.assertAlmostEqual(res.frequency_hz, 261.63, delta=2.0)

    def test_pitch_silence_handling(self):
        """Valida que silêncio não gera nota falsa."""
        silence = np.zeros(self.chunk_size, dtype=np.float32)
        res = self.autocorr.detect(silence, self.sr)
        self.assertFalse(res.is_voiced)
        self.assertEqual(res.note_name, "--")
        self.assertEqual(res.frequency_hz, 0.0)

    def test_chroma_extraction_c_major(self):
        """Valida que o cromagrama destaca C, E e G em uma tríade de C Maior."""
        # Somar C4 (261.63), E4 (329.63), G4 (392.00)
        c = self._generate_test_chunk(261.63)
        e = self._generate_test_chunk(329.63)
        g = self._generate_test_chunk(392.00)
        chord_chunk = (c + e + g) / 3.0

        chroma_vec = self.chroma.extract(chord_chunk, self.sr)
        self.assertEqual(len(chroma_vec), 12)

        # C=0, E=4, G=7 no vetor de classes
        self.assertGreater(chroma_vec[0], 0.70)  # C
        self.assertGreater(chroma_vec[4], 0.70)  # E
        self.assertGreater(chroma_vec[7], 0.70)  # G

    def test_audio_analyzer_orchestrator(self):
        """Valida integração do AudioAnalyzer com MusicalContext e medição de latência."""
        chunk = self._generate_test_chunk(440.0, harmonics=1)
        ctx = self.analyzer.analyze_chunk(chunk, self.sr, timestamp=1.5)

        self.assertEqual(ctx.current_note, "A4")
        self.assertAlmostEqual(ctx.current_frequency, 440.0, delta=2.0)
        self.assertGreater(ctx.note_confidence, 0.85)
        self.assertEqual(len(ctx.chroma_vector), 12)
        self.assertGreater(ctx.processing_latency, 0.0)
        self.assertLess(ctx.processing_latency, 20.0)  # Menos de 20ms de CPU


if __name__ == "__main__":
    unittest.main()
