"""Testes Abrangentes para Correção do Follow Mode e Evolução do Position Tracking.

Cobre todos os requisitos críticos especificados:
1. Não-congelamento quando a detecção de acordes falha (1s, 5s, 10s, 20s).
2. Continuidade temporal monotônica com degradação suave de confiança.
3. Recuperação por janela de busca local e correspondência por progressão.
4. Proteção contra saltos para trás em acordes repetidos.
5. Tolerância a acordes errados (variação sem salto brusco).
6. Linhas de letra pura sem acordes.
7. Blocos de tablatura (sem geração de acordes falsos).
8. Metadados isolados (Tom, BPM, Compasso sem poluição harmônica).
9. Mapeamento de estrutura (Português e Inglês).
10. Cifra completa realista.
"""

import unittest

from app.input.chart_semantic_classifier import (
    ChartSemanticClassifier,
    SemanticLineType
)
from app.input.chart_parser import ChartParser
from app.music.chord_chart import ChordChart
from app.music.chart_alignment import ChartAlignment, ChartPosition
from app.music.musical_clock import MusicalClock
from app.music.position_estimator import (
    PositionEstimator,
    TrackingState,
    ProgressionMatch
)
from app.song.song import Song
from app.song.song_session import SongSession


class TestPositionTracking(unittest.TestCase):

    def setUp(self):
        self.realistic_chart_text = """Título: Ousado Amor
Artista: Isaías Saad
Tom: A (com forma de G)
Afinação: E A D G B E
Capotraste: 2ª casa
[Ritmo Padrão] 165 bpm
Compasso: 4/4

[Intro]
C9       G/B

[Verso 1]
C9                      G/B
Antes de eu falar, Tu cantavas sobre mim
C9             G/B
Tu tens sido tão, tão bom pra mim

[Tablatura Solo]
E|-3-3-3-3------------------5-5-5-5-3-3----------------|
B|---------7-7-7-7-5-5-3----------------7-7-5-5-3------|
G|------------------------4-----------------------4----|
D|-----------------------------------------------------|
A|-----------------------------------------------------|
E|-----------------------------------------------------|

[Refrão]
C9                  G/B
Oh, impressionante, infinito e ousado amor de Deus
Am7                 G
Que deixa as noventa e nove só pra me encontrar

[Ponte]
C9
Traz luz para as sombras
G/B
Escala montanhas
Am7                 G x2
Pra me encontrar
"""
        self.parsed_chart = ChartParser.parse(self.realistic_chart_text)
        self.song = Song(
            title="Ousado Amor",
            artist="Isaías Saad",
            key="A Major",
            bpm=165.0,
            meter="4/4",
            chart_text=self.realistic_chart_text,
            chart_data=self.parsed_chart.to_dict()
        )

    def test_semantic_classifier_tablature_not_parsed_as_chords(self):
        """PARTE 21 & 39: Tablaturas não devem gerar falsos acordes (E, B, G, D, A)."""
        tab_line = "E|-3-3-3-3------------------5-5-5-5-3-3----------------|"
        classified = ChartSemanticClassifier.classify_line(tab_line)
        self.assertEqual(classified.line_type, SemanticLineType.TABLATURE)
        self.assertEqual(len(classified.chord_tokens), 0)

    def test_semantic_classifier_metadata_not_in_harmonics(self):
        """PARTE 18, 19, 20 & 40: Tom, BPM, Compasso não devem poluir a harmonia."""
        lines = [
            ("Tom: G", "key", "G Major"),
            ("Key: F#m", "key", "F# Minor"),
            ("Tonalidade: C#m", "key", "C# Minor"),
            ("165 bpm", "bpm", 165.0),
            ("BPM: 120", "bpm", 120.0),
            ("Tempo: 90", "bpm", 90.0),
            ("Andamento: 140 bpm", "bpm", 140.0),
            ("Compasso: 3/4", "meter", "3/4"),
            ("6/8", "meter", "6/8"),
            ("Afinação: E A D G B E", "tuning", "E A D G B E"),
            ("Capotraste: 2ª casa", "capo", "2ª casa")
        ]
        for raw, key, expected in lines:
            classified = ChartSemanticClassifier.classify_line(raw)
            self.assertEqual(classified.line_type, SemanticLineType.METADATA, f"Falha na linha: {raw}")
            self.assertEqual(classified.extracted_metadata.get(key), expected, f"Valor incorreto para {key}")

    def test_semantic_classifier_section_headers(self):
        """PARTE 17 & 41: Reconhecimento de seções em Português e Inglês."""
        headers = [
            ("[Intro]", "INTRO"),
            ("[Introdução]", "INTRO"),
            ("[Primeira Parte]", "VERSE"),
            ("[Verso 2]", "VERSE"),
            ("[Pré-Refrão]", "PRE_CHORUS"),
            ("[Refrão]", "CHORUS"),
            ("[Coro]", "CHORUS"),
            ("[Chorus]", "CHORUS"),
            ("[Ponte]", "BRIDGE"),
            ("[Bridge]", "BRIDGE"),
            ("[Solo]", "SOLO"),
            ("[Instrumental]", "INSTRUMENTAL"),
            ("[Interlúdio]", "INSTRUMENTAL"),
            ("[Outro]", "OUTRO"),
            ("[Final]", "OUTRO")
        ]
        for raw, expected_type in headers:
            classified = ChartSemanticClassifier.classify_line(raw)
            self.assertEqual(classified.line_type, SemanticLineType.SECTION, f"Falha ao reconhecer: {raw}")
            self.assertEqual(classified.section_type, expected_type, f"Tipo incorreto para: {raw}")

    def test_semantic_classifier_words_not_chords(self):
        """PARTE 23: Palavras comuns da letra nunca devem virar acordes."""
        lyric_samples = [
            "Amor da minha vida",
            "Do seu lado eu fico em paz",
            "Eu estava pensando em você",
            "A graça me alcançou",
            "Ele é o Senhor dos senhores"
        ]
        for line in lyric_samples:
            classified = ChartSemanticClassifier.classify_line(line)
            self.assertEqual(classified.line_type, SemanticLineType.LYRIC)

    def test_complex_chords_preservation(self):
        """PARTE 24: Preservação de acordes complexos sem simplificação indevida."""
        complex_line = "C9   G/B   F#m7(b5)   Bbmaj7   C7M   Dm/F   C7(b9)   Cadd9"
        classified = ChartSemanticClassifier.classify_line(complex_line)
        self.assertEqual(classified.line_type, SemanticLineType.CHORD)
        extracted = [tok for tok, col in classified.chord_tokens]
        self.assertIn("C9", extracted)
        self.assertIn("G/B", extracted)
        self.assertIn("F#m7(b5)", extracted)
        self.assertIn("Bbmaj7", extracted)
        self.assertIn("C7M", extracted)
        self.assertIn("Dm/F", extracted)
        self.assertIn("C7(b9)", extracted)
        self.assertIn("Cadd9", extracted)

    def test_critical_bug_chord_loss_does_not_freeze_position(self):
        """PARTE 3, 4, 34 & 35: Simulação crítica de perda de detecção por 1s, 5s, 10s, 20s.
        
        CRITÉRIO OBRIGATÓRIO: O sistema NÃO PODE congelar na mesma linha ou no mesmo compasso.
        O relógio avança e a posição na cifra avança monotonicamente.
        """
        session = SongSession(self.song)
        clock = session.clock
        estimator = session.position_estimator

        # Início na posição confirmada Bar 1
        pos1 = estimator.update(timestamp=0.0, detected_chord="C9", detected_confidence=0.90)
        self.assertEqual(pos1.current_bar, 1)
        self.assertEqual(estimator.tracking_state, TrackingState.TRACKING)
        initial_line = pos1.line_index

        # Simula reprodução contínua com detecção PERDIDA (UNKNOWN / 0.0 conf)
        # Avança de t=0.0 até t=20.0 segundos a 165 BPM (1 beat ~= 0.363s, 1 bar ~= 1.454s)
        time_step = 0.5
        current_time = 0.0
        observed_bars = [pos1.current_bar]
        observed_lines = [pos1.line_index]

        while current_time < 20.0:
            current_time += time_step
            clock.update(current_time)
            pos = estimator.update(timestamp=current_time, detected_chord="--", detected_confidence=0.0)
            observed_bars.append(pos.current_bar)
            observed_lines.append(pos.line_index)

        # 1. Os compassos DEVEM ter avançado monotonicamente (sem congelar)
        self.assertGreater(observed_bars[-1], 10, "Compasso congelou durante perda de detecção!")
        for i in range(len(observed_bars) - 1):
            self.assertLessEqual(observed_bars[i], observed_bars[i + 1], "Compassos não foram monotônicos!")

        # 2. A linha da cifra DEVE ter avançado (não pode ter ficado presa na linha inicial)
        self.assertGreater(observed_lines[-1], initial_line, "Follow Mode congelou na mesma linha da cifra!")

        # 3. A confiança deve ter decaído suavemente, mas sem zerar o sistema
        self.assertGreater(estimator.position_confidence, 0.35)
        self.assertIn(estimator.tracking_state, (TrackingState.UNCERTAIN, TrackingState.LOST))

        # 4. Quando a detecção retorna (ex: no Refrão com C9), o sistema deve recuperar o estado TRACKING
        current_time += time_step
        clock.update(current_time)
        recovered_pos = estimator.update(timestamp=current_time, detected_chord="C9", detected_confidence=0.95)
        self.assertEqual(estimator.tracking_state, TrackingState.TRACKING)
        self.assertGreater(estimator.position_confidence, 0.85)

    def test_local_search_window_prevents_jumping_across_song(self):
        """PARTE 8 & 11: Busca de recuperação em janela local impede saltos aleatórios."""
        clock = MusicalClock(bpm=120.0, meter="4/4")
        alignment = ChartAlignment(self.parsed_chart)
        estimator = PositionEstimator(alignment, clock, local_window_bars=4)

        # Posiciona no Compasso 12 (Refrão)
        clock.seek(24.0) # 12 compassos a 2s por compasso
        pos = estimator.update(timestamp=24.0, detected_chord="--", detected_confidence=0.0)

        # Se detectar G/B, deve mapear para o compasso local do Refrão e NÃO para a Intro (Comp. 2)
        local_match = estimator._probe_local_search_window(
            center_bar=clock.bar,
            detected_chord="G/B",
            window_bars=4
        )
        self.assertIsNotNone(local_match)
        matched_bar, matched_chord = local_match
        self.assertGreaterEqual(matched_bar, 8, "Saltou para a Introdução no início da música!")

    def test_wrong_chord_performance_variation(self):
        """PARTE 36: Acorde errado / divergente não causa salto imediato para longe."""
        clock = MusicalClock(bpm=120.0, meter="4/4")
        alignment = ChartAlignment(self.parsed_chart)
        estimator = PositionEstimator(alignment, clock)

        # Compasso 1 espera C9
        clock.update(0.5)
        pos = estimator.update(timestamp=0.5, detected_chord="F#m", detected_confidence=0.80)

        # O estimador mantém a cifra soberana e não salta para longe
        self.assertEqual(pos.current_bar, 1)
        self.assertEqual(pos.current_chord, "C9")

    def test_repeated_chords_progression_continuity(self):
        """PARTE 37: Progressões repetidas em seções diferentes não causam recuo ao início."""
        session = SongSession(self.song)
        clock = session.clock
        estimator = session.position_estimator

        # Avança até a Ponte (compasso avançado)
        clock.seek(40.0)
        pos_bridge = estimator.update(timestamp=40.0, detected_chord="C9", detected_confidence=0.90)

        # Mesmo detectando C9 (que também existe no início), o compasso deve se manter na Ponte
        self.assertGreaterEqual(pos_bridge.current_bar, 15)

    def test_lyric_only_lines_advancement(self):
        """PARTE 38: Linhas somente com letra avançam no tempo musical."""
        text_lyrics_only = """[Verso]
C
Eu comecei a cantar
Eu continuei cantando a noite toda
Mesmo sem outro acorde nesta linha
"""
        chart = ChartParser.parse(text_lyrics_only)
        alignment = ChartAlignment(chart)

        # Compasso 1 (C na linha 2)
        pos1 = alignment.get_position_at(bar=1)
        self.assertEqual(pos1.line_index, 2)

        # Compasso 2 (linha de letra sem acorde)
        pos2 = alignment.get_position_at(bar=2)
        self.assertEqual(pos2.line_index, 4)

        # Compasso 3 (terceira linha de letra)
        pos3 = alignment.get_position_at(bar=3)
        self.assertEqual(pos3.line_index, 5)

    def test_progression_matching(self):
        """PARTE 10: Correspondência robusta de sequências de múltiplos acordes."""
        clock = MusicalClock(bpm=120.0, meter="4/4")
        alignment = ChartAlignment(self.parsed_chart)
        estimator = PositionEstimator(alignment, clock)

        # Progressão do Refrão: C9 -> G/B -> Am7 -> G
        match = estimator.find_progression_match(
            recent_chords=["C9", "G/B", "Am7"],
            start_bar_hint=10,
            window_bars=8
        )
        self.assertIsNotNone(match)
        self.assertIsInstance(match, ProgressionMatch)
        self.assertGreaterEqual(match.confidence, 0.75)


if __name__ == "__main__":
    unittest.main()
