"""Testes de Transposição de Cifras (v0.6)."""

import unittest

from app.music.transposition import (
    transpose_chord_symbol,
    transpose_text_block,
    key_uses_flats,
    semitones_between_keys,
)
from app.input.chart_parser import ChartParser
from app.music.chord_chart import ChordChart
from app.project.project_manager import ProjectManager


class TestChordSymbolTransposition(unittest.TestCase):
    def test_preserves_quality_extension_inversion(self):
        self.assertEqual(transpose_chord_symbol("G", 2), "A")
        self.assertEqual(transpose_chord_symbol("Em7", 2), "F#m7")
        self.assertEqual(transpose_chord_symbol("F#m7(b5)", 1), "Gm7(b5)")
        self.assertEqual(transpose_chord_symbol("C/E", 5), "F/A")
        self.assertEqual(transpose_chord_symbol("Dsus4", -2), "Csus4")
        self.assertEqual(transpose_chord_symbol("Cadd9", 2), "Dadd9")

    def test_enharmonic_spelling_follows_flag(self):
        self.assertEqual(transpose_chord_symbol("A", 1, use_flats=False), "A#")
        self.assertEqual(transpose_chord_symbol("A", 1, use_flats=True), "Bb")

    def test_placeholder_is_untouched(self):
        self.assertEqual(transpose_chord_symbol("--", 3), "--")


class TestTextBlockTransposition(unittest.TestCase):
    def test_only_chord_lines_change(self):
        text = "Tom: G\n[Intro]\nG  D  Em  C\nLuz do sol e a lua\n"
        out = transpose_text_block(text, 2, use_flats=False)
        self.assertIn("A  E  F#m  D", out)
        self.assertIn("Luz do sol e a lua", out)   # letra intacta
        self.assertIn("Tom: A", out)               # metadados atualizados
        self.assertIn("[Intro]", out)              # seção intacta

    def test_key_helpers(self):
        self.assertEqual(semitones_between_keys("G Major", "A Major"), 2)
        self.assertTrue(key_uses_flats("F Major"))
        self.assertFalse(key_uses_flats("D Major"))
        self.assertTrue(key_uses_flats("Bb Major"))


class TestChartTransposition(unittest.TestCase):
    def test_transpose_to_key(self):
        chart = ChartParser.parse("Tom: G\n[Intro]\nG  D  Em  C\n[Verso]\nG  C  D  G\n")
        chart.transpose_to_key("A Major")
        self.assertEqual(chart.key, "A Major")
        self.assertEqual(
            [c.symbol.original_symbol for c in chart.sections[0].chords],
            ["A", "E", "F#m", "D"],
        )
        # round-trip de volta ao tom original
        chart.transpose_to_key("G Major")
        self.assertEqual(
            [c.symbol.original_symbol for c in chart.sections[0].chords],
            ["G", "D", "Em", "C"],
        )

    def test_project_manager_transpose_song_persists(self):
        import tempfile
        import os
        tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(tmp_dir.cleanup)
        pm = ProjectManager(os.path.join(tmp_dir.name, "project.json"))
        song, _ = pm.import_from_source("Tom: G\n[Intro]\nG  D  Em  C\n", title="X")
        pm.transpose_song(song, "B Major")
        self.assertEqual(song.key, "B Major")
        syms = [c["symbol"]["original_symbol"] for c in song.chart_data["sections"][0]["chords"]]
        self.assertEqual(syms, ["B", "F#", "G#m", "E"])


if __name__ == "__main__":
    unittest.main()
