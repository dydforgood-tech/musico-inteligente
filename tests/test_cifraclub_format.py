"""Testes do reconhecimento do padrão CifraClub (v0.6).

Cobre: cifragem CifraClub (C4=sus4, C2=sus2, Cm5-=dim), rótulos de seção reais,
capotraste ("na Nª casa"), BPM/andamento, e tratamento de TABLATURA no meio da cifra
(ignorada na harmonia e separando corretamente os blocos).
"""

import unittest

from app.input.chart_semantic_classifier import ChartSemanticClassifier, SemanticLineType
from app.input.chart_parser import ChartParser
from app.music.chord_chart import parse_chord


class TestCifraClubChordNotation(unittest.TestCase):
    def test_sus_and_dim_notation(self):
        self.assertEqual(parse_chord("C4").quality, "sus4")
        self.assertEqual(parse_chord("D2").quality, "sus2")
        self.assertEqual(parse_chord("Cm5-").quality, "dim")
        self.assertEqual(parse_chord("C5-").quality, "dim")
        # extensões numéricas normais não são confundidas com sus
        self.assertEqual(parse_chord("C9").extension, "9")
        self.assertEqual(parse_chord("C7M").extension, "maj7")


class TestCifraClubSections(unittest.TestCase):
    def test_section_labels(self):
        chart = ChartParser.parse(
            "[Primeira Parte]\nC G\n[Pré-Refrão]\nD E\n[Refrão]\nA B\n"
            "[Pós-Refrão]\nF C\n[Solo]\nG D\n[Final]\nC\n"
        )
        types = [s.section_type for s in chart.sections]
        self.assertEqual(types, ["VERSE", "PRE_CHORUS", "CHORUS", "CHORUS", "SOLO", "OUTRO"])

    def test_capo_metadata_variants(self):
        meta = ChartSemanticClassifier.parse_metadata_line("Capotraste na 2ª casa")
        self.assertEqual(meta.get("capo"), "2ª casa")
        chart = ChartParser.parse("Tom: Em\nCapotraste na 2ª casa\n[Intro]\nEm G\n")
        self.assertEqual(chart.capo_semitones, 2)

    def test_no_capo(self):
        chart = ChartParser.parse("Capotraste: Sem capotraste\n[Intro]\nG D\n")
        self.assertEqual(chart.capo_semitones, 0)

    def test_bpm_recognition(self):
        for line, expected in [
            ("BPM: 120", 120.0),
            ("Tempo: 92", 92.0),
            ("Andamento: 138 bpm", 138.0),
            ("165 bpm", 165.0),
        ]:
            meta = ChartSemanticClassifier.parse_metadata_line(line)
            self.assertIsNotNone(meta, f"não reconheceu: {line}")
            self.assertEqual(meta.get("bpm"), expected)


class TestCifraClubCorpusPatterns(unittest.TestCase):
    """Regressões vindas da análise de cifras reais do CifraClub (vários gêneros)."""

    def test_sharp_major_chords_preserved(self):
        # BUG corrigido: "G#" era truncado para "G". Cobre G# C# D# F# A#.
        toks = [t for t, _ in ChartSemanticClassifier.classify_line("G#  C#  D#  F#  A#").chord_tokens]
        self.assertEqual(toks, ["G#", "C#", "D#", "F#", "A#"])

    def test_inline_intro_header_with_chords(self):
        # Padrão CifraClub: "[Intro]  G#  C  Fm  C#9" (acordes na mesma linha do rótulo)
        chart = ChartParser.parse("Tom: Ab\n[Intro]  G#  C  Fm  C#9\n[Primeira Parte]\nG#  C\nletra\n")
        self.assertEqual(chart.sections[0].name, "Intro")
        self.assertEqual(
            [c.symbol.original_symbol for c in chart.sections[0].chords],
            ["G#", "C", "Fm", "C#9"],
        )

    def test_key_with_shape_annotation(self):
        # "Tom: F#m (com forma de Em)" / "Tom: A (com forma de G)"
        chart = ChartParser.parse("Tom: F#m (com forma de Em)\n[Intro]\nEm7 G D4 A7\n")
        self.assertTrue(chart.key.startswith("F#"))

    def test_complex_chords_roundtrip_transpose(self):
        # Acordes complexos de MPB/gospel preservam a estrutura ao transpor
        from app.music.transposition import transpose_chord_symbol as t
        self.assertEqual(t("F7M(9)", 2), "G7M(9)")
        self.assertEqual(t("A7(4)", 2), "B7(4)")
        self.assertEqual(t("D11/F#", 2), "E11/G#")
        self.assertEqual(t("G4(6)", 2), "A4(6)")

    def test_tab_uppercase_prefix_section(self):
        # "[TAB - Primeira Parte]" (maiúsculo) é reconhecido como seção
        chart = ChartParser.parse("[TAB - Intro]\nE|--3--5--|\n[Refrão]\nC G\n")
        self.assertEqual(len(chart.sections), 2)

    def test_real_chord_vocabulary_recognized(self):
        # Vocabulário observado em cifras reais (MPB, bossa, gospel, rock, sertanejo).
        C = ChartSemanticClassifier
        real = [
            "A#°", "D#º", "C#º7", "Bm7M", "Em7(5-)", "Dm7(b5)", "A7(13-)", "C7(4/9)",
            "Ab7(11+)", "G7(13)", "Am7(11)", "F7M(2/4+)", "B6(9)", "C7M(6)",
            "E/G#", "Db/F", "A7M/E", "G5/F#", "G4/B", "D9(11)", "F5(9)", "E5", "C6",
        ]
        for ch in real:
            self.assertTrue(C.is_chord_shaped(ch), f"não reconhecido como acorde: {ch}")

    def test_ordinal_and_degree_both_diminished(self):
        self.assertEqual(parse_chord("A#°").quality, "dim")   # sinal de grau U+00B0
        self.assertEqual(parse_chord("D#º").quality, "dim")   # indicador ordinal U+00BA

    def test_transpose_preserves_complex_suffixes(self):
        from app.music.transposition import transpose_chord_symbol as t
        self.assertEqual(t("A7(13-)", 2), "B7(13-)")
        self.assertEqual(t("A7M/E", 2), "B7M/F#")
        self.assertEqual(t("G5/F#", 2), "A5/G#")
        self.assertEqual(t("D#º", 1), "Eº")

    def test_bar_separators_in_chord_line(self):
        C = ChartSemanticClassifier
        toks = [x for x, _ in C.classify_line("C G Am F | C G F F6 C").chord_tokens]
        self.assertEqual(toks, ["C", "G", "Am", "F", "C", "G", "F", "F6", "C"])

    def test_wide_spacing_heuristic(self):
        # Acordes espaçados/alinhados sobre a letra = linha de acordes; letra permanece letra.
        C = ChartSemanticClassifier
        self.assertEqual(C.classify_line("C7M(9)              Am").line_type, SemanticLineType.CHORD)
        self.assertEqual(C.classify_line("G      D      N.C.").line_type, SemanticLineType.CHORD)
        # Letra com minúsculas NÃO vira acorde, mesmo espaçada
        self.assertEqual(C.classify_line("Eu   te   amo   muito").line_type, SemanticLineType.LYRIC)
        self.assertEqual(C.classify_line("A vida e bela demais").line_type, SemanticLineType.LYRIC)


class TestTablatureIgnored(unittest.TestCase):
    def test_tab_lines_detected(self):
        for tab in [
            "E|-3-3-3-3--5-5-3--|",
            "e|---0h2p0---3---|",
            "G|--7/9\\7--|",
            "B|---(5)---|",
            "|---0---2---3---|",
            "--3--5--7--5--3--",
            "Eb|--1--3--|",
        ]:
            self.assertTrue(ChartSemanticClassifier.is_tablature_line(tab), f"tab não detectada: {tab}")

    def test_lyrics_and_chords_not_tab(self):
        for not_tab in ["C  G  Am  F", "Luz do sol - a-ha", "Cantando a plena voz"]:
            self.assertFalse(ChartSemanticClassifier.is_tablature_line(not_tab))

    def test_tab_block_separates_sections(self):
        # Solo só com tablatura NÃO deve engolir o Verso seguinte, nem gerar acordes falsos.
        txt = ("[Intro]\nC  G  Am  F\n"
               "[Solo]\nE|-3-3-3---5-5-3--|\nB|---7-7-5-5------|\n"
               "[Verso]\nC            G\nCantando aqui\n")
        chart = ChartParser.parse(txt)
        names = [s.name for s in chart.sections]
        self.assertEqual(names, ["Intro", "Solo", "Verso"])
        solo = chart.sections[1]
        self.assertEqual(solo.chords, [])   # tablatura não vira acorde
        verso = chart.sections[2]
        self.assertEqual([c.symbol.original_symbol for c in verso.chords], ["C", "G"])


if __name__ == "__main__":
    unittest.main()
