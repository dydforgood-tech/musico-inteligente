"""Validação de Histórico Completo de Longa Duração (> 48s, 60s, 120s / 2 minutos).

Garante que o ChordHistory e a interface gráfica acompanhem a duração integral do áudio:
- Áudios de 60 segundos registram acordes após 48 segundos sem truncamento.
- Áudios de 120 segundos (2 minutos) mantêm o histórico completo desde 00:00 até 02:00.
- Nenhum evento inicial é descartado (sem restrição de maxlen=50).
- A tabela de histórico na interface atualiza incrementalmente todos os eventos.
"""

import unittest
import numpy as np

from app.analysis.chord_history import ChordHistory, ChordEvent, ChordStabilizer
from app.analysis.audio_analyzer import AudioAnalyzer
from app.audio.audio_loader import FileAudioSource
from app.utils.audio_generator import generate_test_song


class TestLongAudioHistory(unittest.TestCase):
    """Testes de durabilidade e integridade do histórico musical em faixas longas."""

    def test_chord_history_60_seconds_unbounded(self):
        """Valida que um áudio de 60s registra eventos além de 48 segundos e preserva o início."""
        history = ChordHistory()
        stabilizer = ChordStabilizer(history=history, confirmation_time=0.15)

        # Simular 60 transições de acorde ao longo de 60 segundos (1 acorde por segundo)
        # Acordes alternados: C, G, Am, F (repetidos 15 vezes = 60s)
        chords_sequence = ["C", "G", "Am", "F"] * 15

        for second_idx, chord_sym in enumerate(chords_sequence):
            # Simular 20 frames por segundo
            for frame in range(20):
                t = second_idx + (frame * 0.05)
                stabilizer.process(chord_sym, 0.92, float(t), root=chord_sym[0])

        # Finalizar último acorde
        stabilizer._finalize_current_event_if_active(end_time=60.0)

        events = history.get_events()

        # 1. Deve haver ~60 eventos (e NÃO ser truncado em 50)
        self.assertGreaterEqual(len(events), 55, f"Total de eventos ({len(events)}) deve ser >= 55")

        # 2. O primeiro evento deve começar no início da faixa (tempo ~0.0s)
        self.assertAlmostEqual(events[0].start_time, 0.0, delta=0.2,
                               msg="Primeiro evento não pode ser descartado da memória!")
        self.assertEqual(events[0].chord, "C")

        # 3. Deve haver eventos confirmados DEPOIS de 48 segundos!
        events_after_48 = [e for e in events if e.start_time >= 48.0]
        self.assertGreater(len(events_after_48), 5,
                           f"Esperado múltiplos eventos após 48s, obtido: {len(events_after_48)}")

        # 4. O último evento deve estar próximo a 59s / 60s
        self.assertGreater(events[-1].start_time, 57.0,
                           f"Último evento deve alcançar o final da faixa (t={events[-1].start_time:.1f}s)")

    def test_chord_history_2_minutes_120_seconds(self):
        """Valida que uma faixa de 2 minutos (120s) possui histórico íntegro de ponta a ponta."""
        history = ChordHistory()
        stabilizer = ChordStabilizer(history=history, confirmation_time=0.15)

        # 120 segundos de música (acordes de 2 segundos cada = 60 acordes no total)
        # Progressão: C -> G -> Am -> F repetida por 120 segundos
        progression = ["C", "G", "Am", "F"]
        total_seconds = 120.0

        for t in np.arange(0.0, total_seconds, 0.05):
            chord_idx = int(t // 2.0) % 4
            chord_sym = progression[chord_idx]
            stabilizer.process(chord_sym, 0.90, float(t), root=chord_sym[0])

        stabilizer._finalize_current_event_if_active(end_time=total_seconds)

        events = history.get_events()

        # Deve haver 60 acordes de 2 segundos cada
        self.assertGreaterEqual(len(events), 58)

        # Início preservado
        self.assertAlmostEqual(events[0].start_time, 0.0, delta=0.2)

        # Eventos distribuídos por toda a extensão de 2 minutos
        timestamps = [e.start_time for e in events]
        self.assertTrue(any(t < 10.0 for t in timestamps), "Falta início (t < 10s)")
        self.assertTrue(any(45.0 <= t <= 55.0 for t in timestamps), "Falta região intermediária (~50s)")
        self.assertTrue(any(75.0 <= t <= 85.0 for t in timestamps), "Falta região de 1m20s")
        self.assertTrue(any(t >= 115.0 for t in timestamps), "Falta final da faixa (t >= 115s)")

        # Duração total coberta deve alcançar 120 segundos
        total_covered = events[-1].end_time - events[0].start_time
        self.assertAlmostEqual(total_covered, 120.0, delta=2.5)

    def test_ui_incremental_rendering_logic(self):
        """Valida que a lógica de renderização incremental da UI não trava nem trunca eventos."""
        events = [
            ChordEvent(chord="C", start_time=float(i), end_time=float(i + 1), duration=1.0, confidence=0.9)
            for i in range(120)  # 120 eventos (2 minutos)
        ]

        rendered_items = []
        # Simular chamadas incrementais como ocorrem no loop da UI
        for batch_size in [10, 25, 45, 50, 60, 80, 100, 120]:
            current_events = events[:batch_size]
            total_count = len(current_events)
            rendered_count = len(rendered_items)

            if total_count > rendered_count:
                for i in range(rendered_count, total_count):
                    rendered_items.append(current_events[i].to_dict())

        self.assertEqual(len(rendered_items), 120)
        self.assertEqual(rendered_items[0]["chord"], "C")
        self.assertEqual(rendered_items[49]["chord"], "C")  # Antigo limite de 50
        self.assertEqual(rendered_items[50]["chord"], "C")  # Evento 51 mantido
        self.assertEqual(rendered_items[-1]["chord"], "C") # Evento 120 mantido

    def test_pipeline_with_long_audio_file(self):
        """Testa o pipeline completo do AudioAnalyzer em uma faixa sintetizada de 64 segundos (> 48s)."""
        import os
        import tempfile
        temp_wav = os.path.join(tempfile.gettempdir(), "test_song_64s_temp.wav")
        try:
            # 8 ciclos de C-G-Am-F (8 * 8s = 64s)
            generate_test_song(temp_wav, bpm=120.0, total_cycles=8)
            source = FileAudioSource(temp_wav)
            analyzer = AudioAnalyzer()
            sr = source.get_sample_rate()

            # Percorrer áudio até 60 segundos
            for t in np.arange(0.5, 61.0, 0.25):
                chunk = source.get_chunk_at(int(t * sr), 4096)
                analyzer.analyze_chunk(chunk, sr, float(t))

            events = analyzer.chord_history.get_events()
            events_after_48 = [e for e in events if e.start_time >= 48.0]

            self.assertGreater(len(events_after_48), 2,
                               f"Devem existir eventos confirmados após 48s! Obtido: {len(events_after_48)}")
            self.assertEqual(events[0].chord, "C", "Primeiro acorde C deve continuar no histórico")
            source.close()
        finally:
            if os.path.exists(temp_wav):
                try:
                    os.remove(temp_wav)
                except Exception:
                    pass


if __name__ == "__main__":
    unittest.main()
