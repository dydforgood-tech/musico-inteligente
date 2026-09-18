"""Bateria de testes automatizados da Camada de Contexto Musical, Memória e Relógio (Fase 14).

Cobre rigorosamente:
  TESTE 1 — MusicalContext (criação, campos, consistência, latência e aliases)
  TESTE 2 — ChordHistory (eventos discretos confirmados: 3 acordes = 3 eventos, não centenas)
  TESTE 3 — ChordStabilizer (transição confirmada: C, C, C -> G, G, G)
  TESTE 4 — Rejeição de Ruído/Oscilação (C, C, G, C, C, C -> G isolado ignorado)
  TESTE 5 — MusicalClock 120 BPM 4/4 (0s Beat 1, 0.5s Beat 2, 1.0s Beat 3, 1.5s Beat 4, 2.0s Compasso 2 Beat 1)
  TESTE 6 — MusicalClock 100 BPM (validação de 600 ms por batida)
  TESTE 7 — Suporte a 3/4 e Métricas Variadas
  TESTE 8 — KeyHistory e Estabilidade Tonal (rejeição de falsas modulações)
"""

import unittest
import numpy as np

from app.music.musical_context import MusicalContext
from app.music.musical_clock import MusicalClock
from app.analysis.chord_history import ChordHistory, ChordEvent, ChordStabilizer
from app.analysis.key_history import KeyHistory, KeyEvent, KeyStabilizer
from app.music.context_manager import MusicalContextManager


class TestMusicalContextAndMemory(unittest.TestCase):
    """Testes formais dos requisitos da Fase 14."""

    def test_1_musical_context_creation_and_fields(self):
        """TESTE 1 — Valida criação, atualização, campos temporais e latência do MusicalContext."""
        ctx = MusicalContext(timestamp=12.5, sample_rate=44100)
        ctx.note = "C3"
        ctx.frequency = 130.81
        ctx.note_confidence = 0.94

        ctx.chord = "C"
        ctx.previous_chord = "F"
        ctx.chord_duration = 1.42
        ctx.chord_confidence = 0.91

        ctx.key = "G Major"
        ctx.key_duration = 18.3
        ctx.key_confidence = 0.87

        ctx.bpm = 118.0
        ctx.meter = "4/4"
        ctx.beat = 3
        ctx.bar = 12

        ctx.processing_latency = 1.2
        ctx.analysis_window = 92.8
        ctx.stabilization_delay = 200.0
        ctx.estimated_musical_latency = 247.6

        # Sincroniza aliases
        ctx.sync_aliases()

        # Validação dos campos canônicos e aliases
        self.assertEqual(ctx.current_note, "C3")
        self.assertEqual(ctx.current_chord, "C")
        self.assertEqual(ctx.previous_chord, "F")
        self.assertAlmostEqual(ctx.chord_duration, 1.42)
        self.assertEqual(ctx.current_key, "G Major")
        self.assertAlmostEqual(ctx.key_duration, 18.3)
        self.assertEqual(ctx.time_signature, "4/4")
        self.assertEqual(ctx.bar, 12)
        self.assertEqual(ctx.beat, 3)

        # Validação do dicionário estruturado
        summary = ctx.get_summary_dict()
        self.assertEqual(summary["chord"], "C")
        self.assertEqual(summary["previous_chord"], "F")
        self.assertEqual(summary["bar"], 12)
        self.assertEqual(summary["beat"], 3)

    def test_2_chord_history_discrete_events(self):
        """TESTE 2 — Simula C (2s) -> G (1s) -> Am (2s) e valida que gera exatamente 3 eventos (não centenas)."""
        history = ChordHistory()
        stabilizer = ChordStabilizer(
            min_confidence=0.4,
            min_stability_time=0.15,
            confirmation_time=0.20,
            history=history
        )

        # Simular amostragem a 30 FPS (~33 ms por frame)
        # 1. Tocar C de 0.0s até 2.0s (60 frames de "C")
        for t in np.arange(0.0, 2.0, 0.033):
            stabilizer.process(raw_symbol="C", confidence=0.95, timestamp=float(t), root="C")

        # 2. Tocar G de 2.0s até 3.0s (30 frames de "G")
        for t in np.arange(2.0, 3.0, 0.033):
            stabilizer.process(raw_symbol="G", confidence=0.92, timestamp=float(t), root="G")

        # 3. Tocar Am de 3.0s até 5.0s (60 frames de "Am")
        for t in np.arange(3.0, 5.0, 0.033):
            stabilizer.process(raw_symbol="Am", confidence=0.90, timestamp=float(t), root="A", quality="minor")

        # Finalizar o último acorde ativo
        stabilizer._finalize_current_event_if_active(end_time=5.0)

        events = history.get_events()

        # Deve haver EXATAMENTE 3 eventos consolidados, JAMAIS 150 eventos!
        self.assertEqual(len(events), 3, f"Esperado 3 eventos no histórico, obtido: {len(events)}")
        self.assertEqual(events[0].chord, "C")
        self.assertAlmostEqual(events[0].duration, 2.0, delta=0.25)

        self.assertEqual(events[1].chord, "G")
        self.assertAlmostEqual(events[1].duration, 1.0, delta=0.25)

        self.assertEqual(events[2].chord, "Am")
        self.assertAlmostEqual(events[2].duration, 2.0, delta=0.25)

    def test_3_chord_stabilization_transition(self):
        """TESTE 3 — Entrada C, C, C -> G, G, G deve confirmar transição C -> G."""
        history = ChordHistory()
        stabilizer = ChordStabilizer(
            min_confidence=0.4,
            confirmation_time=0.15,
            history=history
        )

        # C por 3 frames (0.0s, 0.05s, 0.10s)
        stabilizer.process("C", 0.9, 0.00, root="C")
        stabilizer.process("C", 0.9, 0.05, root="C")
        stabilizer.process("C", 0.9, 0.10, root="C")
        self.assertEqual(stabilizer.current_chord, "C")

        # G por 3 frames consecutivos com tempo suficiente para transição (0.15s, 0.25s, 0.35s)
        stabilizer.process("G", 0.9, 0.15, root="G")
        stabilizer.process("G", 0.9, 0.25, root="G")
        stabilizer.process("G", 0.9, 0.35, root="G")

        # A transição para G deve ter sido confirmada
        self.assertEqual(stabilizer.current_chord, "G")
        self.assertEqual(stabilizer.previous_chord, "C")

    def test_4_noise_and_oscillation_rejection(self):
        """TESTE 4 — Sequência C, C, G, C, C, C, C não deve criar mudança de acorde com G isolado."""
        history = ChordHistory()
        stabilizer = ChordStabilizer(
            min_confidence=0.4,
            confirmation_time=0.20,
            history=history
        )

        # C estável
        stabilizer.process("C", 0.95, 0.00)
        stabilizer.process("C", 0.95, 0.05)
        self.assertEqual(stabilizer.current_chord, "C")

        # G isolado (ruído ou transiente transitório)
        stabilizer.process("G", 0.70, 0.10)
        # O acorde estável deve continuar C!
        self.assertEqual(stabilizer.current_chord, "C")

        # C retorna nos frames seguintes
        stabilizer.process("C", 0.95, 0.15)
        stabilizer.process("C", 0.95, 0.20)
        stabilizer.process("C", 0.95, 0.25)
        stabilizer.process("C", 0.95, 0.30)

        # Nenhum evento falso de G deve ter sido gerado
        self.assertEqual(stabilizer.current_chord, "C")
        self.assertEqual(len(history.get_events()), 0)

    def test_5_musical_clock_120_bpm_4_4(self):
        """TESTE 5 — BPM 120 em 4/4 (500 ms por batida)."""
        clock = MusicalClock(bpm=120.0, meter="4/4")

        # 0.0s -> Compasso 1, Beat 1
        clock.update(0.000)
        self.assertEqual(clock.bar, 1)
        self.assertEqual(clock.beat, 1)

        # 0.5s -> Compasso 1, Beat 2
        clock.update(0.500)
        self.assertEqual(clock.bar, 1)
        self.assertEqual(clock.beat, 2)

        # 1.0s -> Compasso 1, Beat 3
        clock.update(1.000)
        self.assertEqual(clock.bar, 1)
        self.assertEqual(clock.beat, 3)

        # 1.5s -> Compasso 1, Beat 4
        clock.update(1.500)
        self.assertEqual(clock.bar, 1)
        self.assertEqual(clock.beat, 4)

        # 2.0s -> Novo compasso (Compasso 2), Beat 1
        clock.update(2.000)
        self.assertEqual(clock.bar, 2)
        self.assertEqual(clock.beat, 1)

    def test_6_musical_clock_100_bpm(self):
        """TESTE 6 — BPM 100 em 4/4: valida aproximadamente 600 ms (0.6s) por batida."""
        clock = MusicalClock(bpm=100.0, meter="4/4")
        self.assertAlmostEqual(clock.beat_duration, 0.600, places=3)

        # 0.0s -> Beat 1
        clock.update(0.000)
        self.assertEqual(clock.beat, 1)

        # 0.600s -> Beat 2
        clock.update(0.600)
        self.assertEqual(clock.beat, 2)

        # 1.200s -> Beat 3
        clock.update(1.200)
        self.assertEqual(clock.beat, 3)

        # 1.800s -> Beat 4
        clock.update(1.800)
        self.assertEqual(clock.beat, 4)

        # 2.400s -> Compasso 2, Beat 1
        clock.update(2.400)
        self.assertEqual(clock.bar, 2)
        self.assertEqual(clock.beat, 1)

    def test_7_musical_clock_3_4_meter(self):
        """TESTE 7 — Garante que o MusicalClock suporta 3/4 (não limitado a 4/4)."""
        clock = MusicalClock(bpm=120.0, meter="3/4")
        self.assertEqual(clock.beats_per_bar, 3)
        self.assertAlmostEqual(clock.bar_duration, 1.500, places=3)

        # 0.0s -> Compasso 1, Beat 1
        clock.update(0.000)
        self.assertEqual(clock.bar, 1)
        self.assertEqual(clock.beat, 1)

        # 0.5s -> Compasso 1, Beat 2
        clock.update(0.500)
        self.assertEqual(clock.bar, 1)
        self.assertEqual(clock.beat, 2)

        # 1.0s -> Compasso 1, Beat 3
        clock.update(1.000)
        self.assertEqual(clock.bar, 1)
        self.assertEqual(clock.beat, 3)

        # 1.5s -> Compasso 2, Beat 1 (porque em 3/4 o compasso tem 3 tempos!)
        clock.update(1.500)
        self.assertEqual(clock.bar, 2)
        self.assertEqual(clock.beat, 1)

    def test_8_key_history_and_stability(self):
        """TESTE 8 — Tonalidade estável e rejeição de oscilações transitórias em KeyHistory."""
        history = KeyHistory()
        stabilizer = KeyStabilizer(
            min_confidence=0.60,
            confirmation_time=2.00,
            history=history
        )

        # 1. Tonalidade C Major estável por 4 segundos
        for t in np.arange(0.0, 4.0, 0.2):
            stabilizer.process("C Major", 0.90, float(t), root="C", scale_type="Major")

        self.assertEqual(stabilizer.current_key, "C Major")

        # 2. Oscilação transitória isolada (ex: G Major por apenas 0.4s durante um acorde dominante)
        for t in [4.2, 4.4]:
            stabilizer.process("G Major", 0.75, float(t), root="G", scale_type="Major")

        # Não deve mudar a tonalidade para G Major porque não persistiu por 2.0s
        self.assertEqual(stabilizer.current_key, "C Major")

        # 3. Retorno a C Major
        for t in np.arange(4.6, 7.0, 0.2):
            stabilizer.process("C Major", 0.92, float(t), root="C", scale_type="Major")

        self.assertEqual(stabilizer.current_key, "C Major")
        # Nenhuma falsa modulação deve ter sido gravada no histórico
        self.assertEqual(len(history.get_events()), 0)


if __name__ == "__main__":
    unittest.main()
