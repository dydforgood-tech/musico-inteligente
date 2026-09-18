"""Validação com Áudio de Violão Acústico Real / Modelado (Fases 15 e 16).

Testa a resposta dos algoritmos sobre o timbre complexo do violão:
- Múltiplas cordas soando simultaneamente
- Ataque percussivo de palheta / unha
- Harmônicos superiores e decaimento acústico
- Transições de acordes (C -> G -> Am -> F)
- Rastreamento rítmico de compasso, beat e andamento
- Formação do ChordHistory consolidado
"""

import os
import unittest
import numpy as np

from app.audio.audio_loader import FileAudioSource
from app.analysis.audio_analyzer import AudioAnalyzer
from app.utils.audio_generator import generate_acoustic_guitar_sample


class TestAcousticGuitarAnalysis(unittest.TestCase):
    """Testes sobre áudio de violão acústico."""

    @classmethod
    def setUpClass(cls):
        cls.wav_path = os.path.abspath("violao_teste_120bpm.wav")
        if not os.path.exists(cls.wav_path):
            generate_acoustic_guitar_sample(cls.wav_path, bpm=120.0)

    def test_guitar_chord_and_context_pipeline(self):
        """Valida que o pipeline processa violão sem quebrar e extrai o contexto musical temporal."""
        source = FileAudioSource(self.wav_path)
        analyzer = AudioAnalyzer()
        sr = source.get_sample_rate()

        # 1. Pré-análise de andamento
        bpm = analyzer.pre_analyze_track(source.mono_data, sr)
        self.assertGreater(bpm, 0.0)
        self.assertAlmostEqual(bpm, 120.0, delta=6.0)

        # 2. Simular reprodução contínua ao longo de 12 segundos (C -> G -> Am)
        # Amostragem em passos de 50 ms
        detected_chords = set()
        for t in np.arange(0.2, 12.0, 0.05):
            chunk = source.get_chunk_at(int(t * sr), 4096)
            ctx = analyzer.analyze_chunk(chunk, sr, float(t))
            if ctx.chord != "--":
                detected_chords.add(ctx.chord)

        # Deve identificar os acordes principais tocados pelo violão
        self.assertTrue(any("C" in c for c in detected_chords), f"Acorde C não encontrado em {detected_chords}")
        self.assertTrue(any("G" in c for c in detected_chords), f"Acorde G não encontrado em {detected_chords}")
        self.assertTrue(any("Am" in c for c in detected_chords), f"Acorde Am não encontrado em {detected_chords}")

        # 3. Validar métricas do MusicalContext
        final_ctx = analyzer.context
        self.assertGreater(final_ctx.bar, 1)
        self.assertIn(final_ctx.beat, [1, 2, 3, 4])
        self.assertGreater(final_ctx.chord_duration, 0.0)
        self.assertNotEqual(final_ctx.previous_chord, "--")
        self.assertGreater(final_ctx.analysis_window, 0.0)
        self.assertGreater(final_ctx.stabilization_delay, 0.0)
        self.assertGreater(final_ctx.estimated_musical_latency, 0.0)

        # 4. Validar ChordHistory: eventos consolidados gerados
        events = analyzer.chord_history.get_events()
        self.assertGreaterEqual(len(events), 2)
        for ev in events:
            self.assertGreater(ev.duration, 0.1)
            self.assertGreater(ev.confidence, 0.3)

        source.close()


if __name__ == "__main__":
    unittest.main()
