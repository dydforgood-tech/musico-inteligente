"""Bateria de testes automatizados para estabilidade de tonalidade e detecção de modulação harmônica real.

Cobre rigorosamente:
  TESTE 1 — Sequência em G Major (G -> Em -> C -> D) com resultado G Major estável.
  TESTE 2 — Sequência (G -> Em -> C -> D -> G) sem gerar Em, C ou D como mudanças de tom.
  TESTE 3 — Áudio real de violão em tonalidade única mantendo estabilidade sem falsas modulações.
  TESTE 4 — Modulação artificial progressiva G Major -> A Major detectada e confirmada no KeyHistory.
"""

import os
import tempfile
import unittest
import numpy as np

from app.analysis.chord_history import ChordHistory, ChordEvent
from app.analysis.key_history import KeyHistory, KeyEvent, KeyStabilizer
from app.analysis.key_detector import KrumhanslSchmucklerKeyDetector, KeyResult
from app.analysis.audio_analyzer import AudioAnalyzer
from app.audio.audio_loader import FileAudioSource
from app.utils.audio_generator import (
    generate_chord,
    generate_acoustic_guitar_sample,
    generate_modulating_song
)


class TestKeyStability(unittest.TestCase):
    """Testes dos requisitos de estabilidade tonal e modulação."""

    def test_1_g_major_sequence_stability(self):
        """TESTE 1 — Sequência em G Major (G -> Em -> C -> D) deve manter G Major estável."""
        history = KeyHistory()
        stabilizer = KeyStabilizer(history=history)

        # Adicionar acordes de G Major ao histórico
        chords = [
            ChordEvent(chord="G", root="G", quality="major", start_time=0.0, end_time=2.0, duration=2.0, confidence=0.95),
            ChordEvent(chord="Em", root="E", quality="minor", start_time=2.0, end_time=4.0, duration=2.0, confidence=0.92),
            ChordEvent(chord="C", root="C", quality="major", start_time=4.0, end_time=6.0, duration=2.0, confidence=0.94),
            ChordEvent(chord="D", root="D", quality="major", start_time=6.0, end_time=8.0, duration=2.0, confidence=0.93),
        ]

        # Inicializa em G Major
        stabilizer.process("G Major", 0.90, 0.0, root="G", scale_type="Major", recent_chords=[chords[0]])
        self.assertEqual(stabilizer.current_key, "G Major")

        # Simula o fluxo temporal durante a progressão
        # Mesmo quando o detector instantâneo emitir "E Minor" durante o acorde Em
        for t in np.arange(2.0, 4.0, 0.2):
            stabilizer.process("E Minor", 0.82, float(t), root="E", scale_type="Minor", recent_chords=chords[:2])
            self.assertEqual(stabilizer.current_key, "G Major", f"Tom oscilou para {stabilizer.current_key} no acorde Em")

        # Mesmo quando o detector instantâneo emitir "C Major" durante o acorde C
        for t in np.arange(4.0, 6.0, 0.2):
            stabilizer.process("C Major", 0.80, float(t), root="C", scale_type="Major", recent_chords=chords[:3])
            self.assertEqual(stabilizer.current_key, "G Major", f"Tom oscilou para {stabilizer.current_key} no acorde C")

        # Mesmo quando o detector instantâneo emitir "D Major" durante o acorde D
        for t in np.arange(6.0, 8.0, 0.2):
            stabilizer.process("D Major", 0.82, float(t), root="D", scale_type="Major", recent_chords=chords)
            self.assertEqual(stabilizer.current_key, "G Major", f"Tom oscilou para {stabilizer.current_key} no acorde D")

        # Nenhuma modulação deve ter sido gravada no histórico
        self.assertEqual(len(history.get_events()), 0)
        self.assertEqual(stabilizer.current_key, "G Major")

    def test_2_sequence_no_spurious_key_changes(self):
        """TESTE 2 — G -> Em -> C -> D -> G não deve gerar Em, C ou D como mudanças de tom."""
        history = KeyHistory()
        stabilizer = KeyStabilizer(history=history)

        chords = [
            ChordEvent("G", 0.0, 2.0, 2.0, 0.95, "G", "major"),
            ChordEvent("Em", 2.0, 4.0, 2.0, 0.92, "E", "minor"),
            ChordEvent("C", 4.0, 6.0, 2.0, 0.94, "C", "major"),
            ChordEvent("D", 6.0, 8.0, 2.0, 0.93, "D", "major"),
            ChordEvent("G", 8.0, 10.0, 2.0, 0.95, "G", "major"),
        ]

        time_points = [
            (0.5, "G Major", "G", "Major", chords[:1]),
            (2.5, "E Minor", "E", "Minor", chords[:2]),
            (4.5, "C Major", "C", "Major", chords[:3]),
            (6.5, "D Major", "D", "Major", chords[:4]),
            (8.5, "G Major", "G", "Major", chords[:5]),
        ]

        for t, raw_k, r, s, rec_ch in time_points:
            stabilizer.process(raw_k, 0.85, t, root=r, scale_type=s, recent_chords=rec_ch)

        # O tom consolidado deve ser G Major
        self.assertEqual(stabilizer.current_key, "G Major")
        # Nenhuma falsa modulação gerada
        recorded_keys = [ev.key for ev in history.get_events()]
        self.assertNotIn("E Minor", recorded_keys)
        self.assertNotIn("C Major", recorded_keys)
        self.assertNotIn("D Major", recorded_keys)
        self.assertEqual(len(recorded_keys), 0)

    def test_3_acoustic_guitar_audio_single_key_stability(self):
        """TESTE 3 — Áudio real/modelado de violão em C Major deve manter C Major sem oscilações."""
        wav_path = os.path.abspath("violao_teste_120bpm.wav")
        if not os.path.exists(wav_path):
            generate_acoustic_guitar_sample(wav_path, bpm=120.0)

        source = FileAudioSource(wav_path)
        sr = source.get_sample_rate()
        analyzer = AudioAnalyzer(sample_rate=sr)

        # Pré-analisar
        analyzer.pre_analyze_track(source.mono_data[:sr * 8], sr)

        observed_keys = []
        # Analisar os primeiros 14 segundos (C -> G -> Am -> F)
        for t in np.arange(0.2, 14.0, 0.1):
            chunk = source.get_chunk_at(int(t * sr), 4096)
            ctx = analyzer.analyze_chunk(chunk, sr, float(t))
            if ctx.key != "--":
                observed_keys.append(ctx.key)

        source.close()

        self.assertGreater(len(observed_keys), 0)
        # Deve consolidar C Major
        final_key = analyzer.context.key
        self.assertEqual(final_key, "C Major")

        # Nenhuma falsa modulação gravada
        modulations = analyzer.key_history.get_events()
        mod_keys = [m.key for m in modulations]
        self.assertEqual(len(modulations), 0, f"Modulações espúrias detectadas no violão: {mod_keys}")

    def test_4_artificial_modulation_g_to_a_detection(self):
        """TESTE 4 — Modulação artificial simples G Major -> A Major com tempo suficiente deve ser detectada."""
        wav_path = os.path.join(tempfile.gettempdir(), "test_modulation_g_to_a_temp.wav")
        generate_modulating_song(wav_path, bpm=120.0, cycles_per_key=2)

        self.assertTrue(os.path.exists(wav_path))
        source = FileAudioSource(wav_path)
        sr = source.get_sample_rate()
        analyzer = AudioAnalyzer(sample_rate=sr)

        # Analisar a primeira seção (G Major: 0s a 15s)
        for t in np.arange(0.2, 15.0, 0.1):
            chunk = source.get_chunk_at(int(t * sr), 4096)
            analyzer.analyze_chunk(chunk, sr, float(t))

        key_section_1 = analyzer.context.key
        self.assertEqual(key_section_1, "G Major", f"Seção 1 deveria ser G Major, obtido: {key_section_1}")

        # Analisar a segunda seção (A Major: 16s a 31s)
        for t in np.arange(15.0, 31.0, 0.1):
            chunk = source.get_chunk_at(int(t * sr), 4096)
            analyzer.analyze_chunk(chunk, sr, float(t))

        key_section_2 = analyzer.context.key
        source.close()

        # Limpar arquivo temporário
        try:
            if os.path.exists(wav_path):
                os.remove(wav_path)
        except Exception:
            pass

        # Na segunda seção, o sistema deve detectar e confirmar a modulação para A Major
        self.assertEqual(key_section_2, "A Major", f"Seção 2 deveria modular para A Major, obtido: {key_section_2}")

        # O KeyHistory deve registrar a modulação com sucesso
        events = analyzer.key_history.get_events()
        self.assertGreaterEqual(len(events), 1, "Modulação para A Major não foi registrada no KeyHistory")
        self.assertEqual(events[0].key, "G Major")
        self.assertEqual(analyzer.context.previous_key, "G Major")


if __name__ == "__main__":
    unittest.main()
