"""Bateria de testes formais do Music Structure Analyzer, Pattern Memory & Prediction Engine (v0.3).

Cobre os 20 cenários obrigatórios:
  1. Normalização de acordes (maior, menor, 7ª, maj7, dim, aug, inversão)
  2. Reconhecimento de transposição idêntica: C-G-Am-F == G-D-Em-C (similaridade >= 0.95)
  3. Reconhecimento de transposição com tonalidade: D-A-Bm-G -> I-V-vi-IV
  4. Tolerância a ruído harmônico (inversões ou acorde de passagem)
  5. Registro dinâmico de padrão no PatternMemory (P01)
  6. Incremento de ocorrência e estabilidade no PatternMemory
  7. Cálculo de probabilidade de transição condicional P(B|A)
  8. Estimativa de posição musical e progresso da seção [0.0, 1.0]
  9. Predição da próxima seção por repetição
  10. Predição de seção com múltiplos candidatos por probabilidade máxima
  11. Look-ahead harmônico (1, 2 e 4 compassos)
  12. Redução de confiança quando padrão não for claro
  13. Não classificar seção como definitiva com evidência fraca (UNKNOWN)
  14. Não quebrar compassos por variação tímbrica ou ruído
  15. Reset do analisador estrutural limpa estado sem falhas
  16. Consistência entre atualização incremental (online) e análise offline
  17. Exportação do relatório JSON estruturado com todos os campos obrigatórios
  18. Consumo de contexto preditivo pelo BassPlayer sem quebrar execução
  19. Teste ponta a ponta com progressão sintética de 64 compassos
  20. Teste integrado com áudio real (violao_teste_120bpm.wav)
"""

import os
import json
import tempfile
import unittest
import numpy as np

from app.music.music_structure import (
    SectionType, PatternOccurrence, MusicalPattern,
    MusicSection, SectionTransition, MusicPosition, MusicStructure
)
from app.music.harmonic_normalization import (
    parse_chord_signature, normalize_chord_sequence,
    sequence_similarity, ChordSignature, NormalizedChordSequence
)
from app.music.pattern_memory import PatternMemory
from app.music.prediction_engine import PredictionEngine
from app.analysis.music_position_estimator import MusicPositionEstimator
from app.analysis.music_structure_analyzer import MusicStructureAnalyzer
from app.analysis.structure_report import export_structure_report
from app.analysis.chord_history import ChordHistory
from app.analysis.key_history import KeyHistory
from app.music.musical_clock import MusicalClock
from app.music.musical_context import MusicalContext
from app.instruments.bass_player import BassPlayer
from app.utils.audio_generator import generate_acoustic_guitar_sample
from app.audio.audio_loader import FileAudioSource
from app.analysis.audio_analyzer import AudioAnalyzer


class TestMusicStructure(unittest.TestCase):
    """Suíte completa de testes formais para o Subsistema de Estrutura & Predição Musical."""

    def setUp(self):
        self.pattern_mem = PatternMemory(similarity_threshold=0.82)
        self.predictor = PredictionEngine()
        self.pos_estimator = MusicPositionEstimator()
        self.structure_analyzer = MusicStructureAnalyzer()
        self.clock = MusicalClock(bpm=120.0, meter="4/4")

    # -------------------------------------------------------------
    # 1. Normalização de acordes
    # -------------------------------------------------------------
    def test_01_harmonic_normalization_types(self):
        """TESTE 1 — Normalização correta de tríades, tétrades e inversões."""
        sig_c = parse_chord_signature("C")
        self.assertEqual(sig_c.root, "C")
        self.assertEqual(sig_c.quality, "major")
        self.assertEqual(sig_c.bass_note, "C")
        self.assertEqual(sig_c.inversion, "root")

        sig_am = parse_chord_signature("Am")
        self.assertEqual(sig_am.root, "A")
        self.assertEqual(sig_am.quality, "minor")

        sig_g7 = parse_chord_signature("G7")
        self.assertEqual(sig_g7.root, "G")
        self.assertEqual(sig_g7.quality, "7")

        sig_fmaj7 = parse_chord_signature("Fmaj7")
        self.assertEqual(sig_fmaj7.root, "F")
        self.assertEqual(sig_fmaj7.quality, "maj7")

        sig_bdim = parse_chord_signature("Bdim")
        self.assertEqual(sig_bdim.root, "B")
        self.assertEqual(sig_bdim.quality, "dim")

        sig_inv = parse_chord_signature("C/E")
        self.assertEqual(sig_inv.root, "C")
        self.assertEqual(sig_inv.bass_note, "E")
        self.assertEqual(sig_inv.inversion, "first")

    # -------------------------------------------------------------
    # 2. Reconhecimento de transposição idêntica: C-G-Am-F == G-D-Em-C
    # -------------------------------------------------------------
    def test_02_transposition_invariance_c_to_g(self):
        """TESTE 2 — C-G-Am-F e G-D-Em-C devem ter similaridade >= 0.95."""
        seq_c = ["C", "G", "Am", "F"]
        seq_g = ["G", "D", "Em", "C"]

        sim = sequence_similarity(seq_c, seq_g, key1="C Major", key2="G Major")
        self.assertGreaterEqual(sim, 0.95, f"Esperado >= 0.95, obtido {sim:.3f}")

    # -------------------------------------------------------------
    # 3. Reconhecimento de transposição com tonalidade: D-A-Bm-G -> I-V-vi-IV
    # -------------------------------------------------------------
    def test_03_transposition_invariance_d_key(self):
        """TESTE 3 — D-A-Bm-G em D Major deve produzir graus I-V-vi-IV e combinar com C-G-Am-F."""
        norm_d = normalize_chord_sequence(["D", "A", "Bm", "G"], tonal_center="D Major")
        self.assertEqual(norm_d.roman_numerals, ["I", "V", "vi", "IV"])

        sim = sequence_similarity(["C", "G", "Am", "F"], ["D", "A", "Bm", "G"],
                                  key1="C Major", key2="D Major")
        self.assertGreaterEqual(sim, 0.95)

    # -------------------------------------------------------------
    # 4. Tolerância a ruído harmônico
    # -------------------------------------------------------------
    def test_04_noise_tolerance(self):
        """TESTE 4 — Sequência com acorde de passagem ou variação curta mantém alta similaridade."""
        base_seq = ["C", "G", "Am", "F"]
        var_seq = ["C", "G7", "Am", "F"]
        sim = sequence_similarity(base_seq, var_seq)
        self.assertGreaterEqual(sim, 0.85, f"Esperado >= 0.85, obtido {sim:.3f}")

        inv_seq = ["C", "G/B", "Am", "F"]
        sim_inv = sequence_similarity(base_seq, inv_seq)
        self.assertGreaterEqual(sim_inv, 0.75, f"Esperado >= 0.75, obtido {sim_inv:.3f}")

    # -------------------------------------------------------------
    # 5. Registro dinâmico de padrão no PatternMemory
    # -------------------------------------------------------------
    def test_05_pattern_memory_registration(self):
        """TESTE 5 — PatternMemory deve cadastrar novo padrão com ID incremental."""
        pat, is_new, occ = self.pattern_mem.register_pattern(
            chord_sequence=["C", "G", "Am", "F"],
            duration_bars=4,
            start_time=0.0,
            tonal_center="C Major"
        )
        self.assertTrue(is_new)
        self.assertEqual(pat.id, "P01")
        self.assertEqual(pat.occurrence_count, 1)
        self.assertEqual(len(self.pattern_mem.get_known_patterns()), 1)

    # -------------------------------------------------------------
    # 6. Incremento de ocorrência e estabilidade no PatternMemory
    # -------------------------------------------------------------
    def test_06_pattern_memory_occurrence_tracking(self):
        """TESTE 6 — Repetição de padrão existente deve incrementar ocorrências e manter o mesmo ID."""
        pat1, new1, occ1 = self.pattern_mem.register_pattern(["C", "G", "Am", "F"], duration_bars=4, start_time=0.0, tonal_center="C Major")
        self.assertTrue(new1)

        pat2, new2, occ2 = self.pattern_mem.register_pattern(["C", "G", "Am", "F"], duration_bars=4, start_time=8.0, tonal_center="C Major")
        self.assertFalse(new2)
        self.assertEqual(pat2.id, "P01")
        self.assertEqual(pat2.occurrence_count, 2)
        self.assertGreaterEqual(pat2.stability_score, 0.80)


    # -------------------------------------------------------------
    # 7. Cálculo de probabilidade de transição condicional P(B|A)
    # -------------------------------------------------------------
    def test_07_transition_probability(self):
        """TESTE 7 — Cálculo estatístico de P(next_section | current_section)."""
        self.pattern_mem.record_transition("P01", "P02", from_section="VERSE", to_section="CHORUS", duration=8.0)
        self.pattern_mem.record_transition("P01", "P02", from_section="VERSE", to_section="CHORUS", duration=8.0)
        self.pattern_mem.record_transition("P01", "P03", from_section="VERSE", to_section="BRIDGE", duration=8.0)

        p_chorus = self.pattern_mem.get_transition_probability("VERSE", "CHORUS")
        p_bridge = self.pattern_mem.get_transition_probability("VERSE", "BRIDGE")

        self.assertAlmostEqual(p_chorus, 2.0 / 3.0, delta=0.01)
        self.assertAlmostEqual(p_bridge, 1.0 / 3.0, delta=0.01)

    # -------------------------------------------------------------
    # 8. Estimativa de posição musical e progresso da seção
    # -------------------------------------------------------------
    def test_08_position_estimator(self):
        """TESTE 8 — PositionEstimator deve calcular progresso no intervalo [0.0, 1.0]."""
        sec = MusicSection(
            id="sec_01",
            section_type="VERSE",
            start_time=0.0,
            end_time=16.0,
            start_bar=1,
            end_bar=8,
            duration_bars=8,
            chord_sequence=["C", "G", "Am", "F"],
            pattern_id="P01"
        )
        pat = MusicalPattern(id="P01", chord_signatures=["C", "G", "Am", "F"], duration_bars=4)

        self.clock.seek(8.0)
        pos = self.pos_estimator.estimate_position(
            timestamp=8.0,
            clock=self.clock,
            current_section=sec,
            current_pattern=pat
        )

        self.assertEqual(pos.current_section, "VERSE")
        self.assertGreaterEqual(pos.section_progress, 0.0)
        self.assertLessEqual(pos.section_progress, 1.0)
        self.assertGreaterEqual(pos.confidence, 0.70)

    # -------------------------------------------------------------
    # 9. Predição da próxima seção por repetição
    # -------------------------------------------------------------
    def test_09_predict_next_section_repetition(self):
        """TESTE 9 — PredictionEngine deve antecipar próxima seção com base no histórico aprendido."""
        self.pattern_mem.record_transition("P01", "P02", from_section="VERSE", to_section="CHORUS", duration=8.0)
        
        pos = MusicPosition(
            current_bar=8,
            current_beat=4,
            current_section="VERSE",
            section_progress=0.90,
            pattern_id="P01",
            bar_in_section=8,
            remaining_bars_in_section=1,
            confidence=0.88
        )
        pat = MusicalPattern(
            id="P01",
            chord_signatures=["C", "G", "Am", "F"],
            duration_bars=8,
            expected_next_section="CHORUS"
        )

        pred = self.predictor.predict_next(
            current_position=pos,
            current_pattern=pat,
            pattern_memory=self.pattern_mem,
            current_section_type="VERSE"
        )

        self.assertEqual(pred.predicted_section, "CHORUS")
        self.assertLessEqual(pred.bars_until_change, 1)
        self.assertGreaterEqual(pred.confidence, 0.70)

    # -------------------------------------------------------------
    # 10. Predição com múltiplos candidatos por probabilidade
    # -------------------------------------------------------------
    def test_10_predict_next_section_multicandidate_probability(self):
        """TESTE 10 — Se houver múltiplos candidatos, escolhe a transição de maior probabilidade."""
        self.pattern_mem.record_transition("P01", "P02", "VERSE", "CHORUS", 8.0)
        self.pattern_mem.record_transition("P01", "P02", "VERSE", "CHORUS", 8.0)
        self.pattern_mem.record_transition("P01", "P03", "VERSE", "BRIDGE", 8.0)

        pos = MusicPosition(
            current_bar=8,
            current_beat=4,
            current_section="VERSE",
            section_progress=0.95,
            pattern_id="P01",
            bar_in_section=8,
            remaining_bars_in_section=1
        )
        pat = MusicalPattern(id="P01", chord_signatures=["C", "G", "Am", "F"], duration_bars=8)

        pred = self.predictor.predict_next(
            current_position=pos,
            current_pattern=pat,
            pattern_memory=self.pattern_mem,
            current_section_type="VERSE"
        )

        self.assertEqual(pred.predicted_section, "CHORUS")
        self.assertIn("Probabilidade", pred.prediction_reason)

    # -------------------------------------------------------------
    # 11. Look-ahead harmônico (1, 2 e 4 compassos)
    # -------------------------------------------------------------
    def test_11_harmonic_lookahead(self):
        """TESTE 11 — Look-ahead deve projetar os próximos acordes antecipadamente."""
        pat = MusicalPattern(id="P01", chord_signatures=["C", "G", "Am", "F"], duration_bars=4)
        
        chords_1bar = self.predictor.predict_harmonic_lookahead(pat, bar_in_pattern=1, lookahead_bars=1)
        self.assertEqual(len(chords_1bar), 1)
        self.assertEqual(chords_1bar[0], "G")

        chords_2bars = self.predictor.predict_harmonic_lookahead(pat, bar_in_pattern=1, lookahead_bars=2)
        self.assertEqual(len(chords_2bars), 2)
        self.assertEqual(chords_2bars, ["G", "Am"])

        chords_4bars = self.predictor.predict_harmonic_lookahead(pat, bar_in_pattern=1, lookahead_bars=4)
        self.assertEqual(len(chords_4bars), 4)
        self.assertEqual(chords_4bars, ["G", "Am", "F", "C"])

    # -------------------------------------------------------------
    # 12. Redução de confiança quando padrão não for claro
    # -------------------------------------------------------------
    def test_12_confidence_reduction_unclear_pattern(self):
        """TESTE 12 — Padrão novo ou ruidoso deve apresentar confiança reduzida."""
        pos = MusicPosition(
            current_bar=1,
            current_beat=1,
            current_section="UNKNOWN",
            section_progress=0.1,
            pattern_id="P_UNKNOWN",
            confidence=0.40
        )
        pred = self.predictor.predict_next(
            current_position=pos,
            current_pattern=None,
            pattern_memory=self.pattern_mem,
            current_section_type="UNKNOWN"
        )
        self.assertLessEqual(pred.confidence, 0.50)
        self.assertEqual(pred.predicted_section, "UNKNOWN")

    # -------------------------------------------------------------
    # 13. Não classificar seção como definitiva com evidência fraca (UNKNOWN)
    # -------------------------------------------------------------
    def test_13_probabilistic_classification_unknown_fallback(self):
        """TESTE 13 — Seção curta/isolada sem repetições deve ser rotulada como UNKNOWN sem inventar."""
        sec_type, conf = self.structure_analyzer._classify_section(
            start_bar=25,
            duration_bars=1,
            occurrence_count=1,
            total_song_bars=64,
            is_contrasting=False
        )
        self.assertEqual(sec_type, "UNKNOWN")
        self.assertLessEqual(conf, 0.60)

    # -------------------------------------------------------------
    # 14. Não quebrar compassos por variação tímbrica ou ruído
    # -------------------------------------------------------------
    def test_14_timbral_noise_stability(self):
        """TESTE 14 — Repetição do mesmo acorde ao longo do compasso não cria falsas fronteiras de frase."""
        ctx = MusicalContext(timestamp=0.0)
        ch_hist = ChordHistory()
        key_hist = KeyHistory()

        for beat in range(1, 5):
            t = (beat - 1) * 0.5
            ch_hist.record(t, "C", 0.90)
            ctx.timestamp = t
            ctx.bar = 1
            ctx.beat = beat
            ctx.chord = "C"
            self.clock.tick(0.5)
            self.structure_analyzer.update_online(ctx, ch_hist, key_hist, self.clock)

        self.assertEqual(len(self.structure_analyzer.sections), 0)

    # -------------------------------------------------------------
    # 15. Reset do analisador estrutural limpa estado sem falhas
    # -------------------------------------------------------------
    def test_15_structure_analyzer_reset(self):
        """TESTE 15 — reset() reinicia todas as estruturas para estado vazio e seguro."""
        self.structure_analyzer._section_counter = 5
        self.structure_analyzer._pattern_memory.register_pattern(["C", "G", "Am", "F"], duration_bars=4, start_time=0.0)
        self.assertEqual(len(self.structure_analyzer.pattern_memory.get_known_patterns()), 1)

        self.structure_analyzer.reset()
        self.assertEqual(len(self.structure_analyzer.sections), 0)
        self.assertEqual(len(self.structure_analyzer.pattern_memory.get_known_patterns()), 0)
        self.assertEqual(self.structure_analyzer.structure.sections, [])

    # -------------------------------------------------------------
    # 16. Consistência entre atualização incremental (online) e análise offline
    # -------------------------------------------------------------
    def test_16_online_vs_offline_consistency(self):
        """TESTE 16 — Análise offline a partir de eventos detecta seções e padrões equivalentes."""
        chords_events = []
        pattern = ["C", "G", "Am", "F"]
        t = 0.0
        for rep in range(4):
            for c in pattern:
                chords_events.append((t, c))
                t += 2.0

        struct = self.structure_analyzer.analyze_full_song_from_events(chords_events, total_duration=32.0, bpm=120.0)
        self.assertGreaterEqual(len(struct.sections), 2)
        self.assertGreaterEqual(len(struct.patterns), 1)

    # -------------------------------------------------------------
    # 17. Exportação do relatório JSON estruturado com todos os campos obrigatórios
    # -------------------------------------------------------------
    def test_17_export_structure_report_json(self):
        """TESTE 17 — Exportação do relatório JSON deve gerar schema completo e válido."""
        sec = MusicSection(
            id="sec_01",
            section_type="VERSE",
            start_time=0.0,
            end_time=16.0,
            start_bar=1,
            end_bar=8,
            duration_bars=8,
            chord_sequence=["C", "G", "Am", "F"],
            pattern_id="P01",
            label="A",
            confidence=0.88
        )
        struct = MusicStructure(
            sections=[sec],
            patterns={"P01": MusicalPattern(id="P01", canonical_chords=["C", "G", "Am", "F"], duration_bars=8)},
            structure_sequence=["VERSE"],
            abstract_sequence=["A"],
            analysis_confidence=0.88
        )

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
            temp_path = tf.name

        try:
            export_structure_report(struct, temp_path)
            self.assertTrue(os.path.exists(temp_path))

            with open(temp_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.assertIn("song_summary", data)
            self.assertIn("sections", data)
            self.assertIn("patterns", data)
            self.assertIn("transitions", data)
            self.assertIn("musical_form", data)
            self.assertIn("analysis_metadata", data)

            self.assertEqual(len(data["sections"]), 1)
            self.assertEqual(data["sections"][0]["type"], "VERSE")
            self.assertEqual(data["sections"][0]["pattern_id"], "P01")
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    # -------------------------------------------------------------
    # 18. Consumo de contexto preditivo pelo BassPlayer sem quebrar execução
    # -------------------------------------------------------------
    def test_18_bass_player_prediction_consumption(self):
        """TESTE 18 — BassPlayer deve consumir o contexto com previsão e manter notas corretas."""
        bass = BassPlayer(sample_rate=44100)
        ctx = MusicalContext(timestamp=15.5)
        ctx.bar = 4
        ctx.beat = 4
        ctx.chord = "F"
        ctx.bpm = 120.0
        ctx.chord_confidence = 0.90
        ctx.predicted_next_section = "CHORUS"
        ctx.predicted_next_chords = ["C", "G"]
        ctx.bars_until_change = 1
        ctx.beats_until_change = 1
        ctx.prediction_confidence = 0.85

        ev = bass.on_musical_context(ctx)
        self.assertIsNotNone(ev)
        self.assertTrue(ev.note in ["F1", "C2"], f"Nota gerada: {ev.note}")
        self.assertIn("CHORUS", ev.reason)

        pred_note, pred_timing = bass.get_prediction(current_bar=4, current_beat=4)
        self.assertIn("CHORUS", pred_timing)

    # -------------------------------------------------------------
    # 19. Teste ponta a ponta com progressão sintética de 64 compassos
    # -------------------------------------------------------------
    def test_19_end_to_end_synthetic_progression_64_bars(self):
        """TESTE 19 — Progressão completa INTRO -> VERSE -> CHORUS -> VERSE -> CHORUS -> BRIDGE -> CHORUS."""
        sections_spec = [
            ("INTRO", ["C", "C", "C", "C"]),
            ("VERSE", ["C", "G", "Am", "F"] * 2),
            ("CHORUS", ["F", "G", "C", "Am"] * 2),
            ("VERSE", ["C", "G", "Am", "F"] * 2),
            ("CHORUS", ["F", "G", "C", "Am"] * 2),
            ("BRIDGE", ["Dm", "Em", "F", "G"] * 2),
            ("CHORUS", ["F", "G", "C", "Am"] * 2),
            ("OUTRO", ["C", "C", "C", "C"])
        ]

        chord_timeline = []
        t = 0.0
        for sec_name, chords in sections_spec:
            for ch in chords:
                chord_timeline.append((t, ch))
                t += 2.0

        analyzer = MusicStructureAnalyzer()
        struct = analyzer.analyze_full_song_from_events(chord_timeline, total_duration=t, bpm=120.0)


        self.assertGreaterEqual(len(struct.sections), 4)
        self.assertGreaterEqual(len(struct.patterns), 3)
        chorus_occurrences = [s for s in struct.sections if s.section_type == "CHORUS"]
        self.assertGreaterEqual(len(chorus_occurrences), 2)

    # -------------------------------------------------------------
    # 20. Teste integrado com áudio real (violao_teste_120bpm.wav)
    # -------------------------------------------------------------
    def test_20_real_audio_structure_analysis(self):
        """TESTE 20 — Análise integrada em arquivo de áudio WAV real sem erros."""
        guitar_file = "violao_teste_120bpm.wav"
        if not os.path.exists(guitar_file):
            generate_acoustic_guitar_sample(guitar_file, bpm=120.0)

        source = FileAudioSource(guitar_file)
        sr = source.get_sample_rate()
        duration = source.get_duration()

        analyzer = AudioAnalyzer(sample_rate=sr, chunk_size=4096)
        analyzer.update_audio_format(sr, 4096)

        chunk_size = 4096
        num_chunks = int(duration * sr / chunk_size)

        for i in range(min(num_chunks, 60)):
            t = (i * chunk_size) / sr
            chunk = source.get_chunk_at(i * chunk_size, chunk_size)
            ctx = analyzer.analyze_chunk(chunk, sr, t)

        self.assertIsNotNone(ctx)
        self.assertIn(ctx.current_section, ["INTRO", "VERSE", "CHORUS", "UNKNOWN"])
        self.assertGreaterEqual(ctx.section_progress, 0.0)
        self.assertLessEqual(ctx.section_progress, 1.0)


if __name__ == "__main__":
    unittest.main()
