"""Teste de integração ponta a ponta da cadeia de análise (v0.1-C, D, E)."""

import os
import unittest
import numpy as np
from app.audio.audio_loader import FileAudioSource
from app.analysis.audio_analyzer import AudioAnalyzer


class TestPipelineIntegration(unittest.TestCase):
    """Valida a integração completa: Nota, Acorde, Tonalidade e Andamento."""

    @classmethod
    def setUpClass(cls):
        cls.test_wav = os.path.abspath("test_song_120bpm.wav")

    def test_full_analysis_pipeline(self):
        if not os.path.exists(self.test_wav):
            self.skipTest("Arquivo test_song_120bpm.wav não encontrado.")

        source = FileAudioSource(self.test_wav)
        analyzer = AudioAnalyzer()
        sr = source.get_sample_rate()

        # 1. Pré-análise de Andamento
        bpm = analyzer.pre_analyze_track(source.mono_data, sr)
        self.assertAlmostEqual(bpm, 120.0, delta=4.0)

        # 2. Amostragem nos CENTROS de cada acorde (C:0-2, G:2-4, Am:4-6, F:6-8 a 120 BPM)
        # 1.5s: Acorde C
        chunk_c = source.get_chunk_at(int(1.5 * sr), 4096)
        ctx_c = analyzer.analyze_chunk(chunk_c, sr, 1.5)
        self.assertEqual(ctx_c.current_chord, "C")

        # 5.0s: Acorde Am (centro da 3ª seção)
        chunk_am = source.get_chunk_at(int(5.0 * sr), 4096)
        ctx_am = analyzer.analyze_chunk(chunk_am, sr, 5.0)
        self.assertEqual(ctx_am.current_chord, "Am")

        # 2. Amostragem contínua simulando a reprodução real (~10 frames/s)
        for t in np.arange(0.5, 11.5, 0.5):
            chunk = source.get_chunk_at(int(t * sr), 4096)
            ctx = analyzer.analyze_chunk(chunk, sr, float(t))

        # Após 11 segundos de reprodução, a tonalidade de Dó Maior já convergiu
        self.assertIn(ctx.current_key, ["C Major", "A Minor"])
        self.assertGreater(ctx.key_confidence, 0.70)

        source.close()


if __name__ == "__main__":
    unittest.main()
