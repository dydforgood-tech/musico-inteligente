"""Testes de auto-detecção de seções e operações de bloco (v0.6)."""

import unittest

from app.music.sectionizer import (
    auto_sectionize_text,
    text_has_section_headers,
    section_color,
    SECTION_COLORS,
)
from app.input.chart_parser import ChartParser
from app.project.project_manager import ProjectManager


UNMARKED_SONG = """Tom: G

G  D  Em  C

G              D
Andando pela estrada
Em             C
Sinto o vento a soprar

C  G  D  D
Refrao que se repete forte
C  G  D  D
Cantando a plena voz

G              D
Outra estrofe diferente
Em             C
Com nova melodia aqui

C  G  D  D
Refrao que se repete forte
C  G  D  D
Cantando a plena voz

Am  F  C  G
Ponte contrastante agora
Am  F  C  G
Mudando o clima aqui
"""


class TestSectionizer(unittest.TestCase):
    def test_detects_form(self):
        out = auto_sectionize_text(UNMARKED_SONG)
        headers = [l for l in out.split("\n") if l.startswith("[")]
        self.assertEqual(headers, ["[Intro]", "[Verso 1]", "[Refrão]", "[Verso 2]", "[Refrão]", "[Ponte]"])

    def test_marked_chart_is_untouched(self):
        marked = "[Intro]\nG D\n[Refrão]\nC G\n"
        self.assertTrue(text_has_section_headers(marked))
        self.assertEqual(auto_sectionize_text(marked), marked)

    def test_colors_defined_for_all_types(self):
        for t in ("INTRO", "VERSE", "CHORUS", "BRIDGE", "OUTRO"):
            self.assertIn(t, SECTION_COLORS)
            self.assertTrue(section_color(t).startswith("#"))
        # tipos desconhecidos caem na cor neutra
        self.assertEqual(section_color("ZZZ"), SECTION_COLORS["UNKNOWN"])

    def test_parses_into_typed_sections(self):
        chart = ChartParser.parse(auto_sectionize_text(UNMARKED_SONG))
        types = [s.section_type for s in chart.sections]
        self.assertIn("CHORUS", types)
        self.assertIn("VERSE", types)
        self.assertIn("BRIDGE", types)


class TestBlockOperations(unittest.TestCase):
    def _chart(self):
        return ChartParser.parse("[Intro]\nG  D\n[Verso]\nC  G\n[Refrão]\nAm  F\n")

    def test_duplicate_section(self):
        chart = self._chart()
        n = len(chart.sections)
        chart.duplicate_section(2)  # duplica o Refrão
        self.assertEqual(len(chart.sections), n + 1)
        self.assertEqual(chart.sections[3].section_type, chart.sections[2].section_type)

    def test_move_section(self):
        chart = self._chart()
        names = [s.name for s in chart.sections]
        chart.move_section(2, 0)  # move o Refrão para o início
        self.assertEqual(chart.sections[0].name, names[2])

    def test_remove_section(self):
        chart = self._chart()
        chart.remove_section(1)
        self.assertEqual(len(chart.sections), 2)

    def test_section_color(self):
        chart = self._chart()
        self.assertTrue(chart.section_color(0).startswith("#"))


class TestImportAutoSections(unittest.TestCase):
    def test_import_auto_detects_when_unmarked(self):
        pm = ProjectManager()
        song, _ = pm.import_from_source(UNMARKED_SONG, title="Auto")
        self.assertTrue(song.metadata.get("sections_auto_detected"))
        types = [s["section_type"] for s in song.chart_data["sections"]]
        self.assertIn("CHORUS", types)

    def test_import_keeps_existing_markup(self):
        pm = ProjectManager()
        song, _ = pm.import_from_source("[Intro]\nG D\n[Refrão]\nC G\n", title="Marcada")
        self.assertFalse(song.metadata.get("sections_auto_detected"))


if __name__ == "__main__":
    unittest.main()
