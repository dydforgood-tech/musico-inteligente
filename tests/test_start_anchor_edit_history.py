"""StartAnchor, Intro editável e histórico estrutural do editor."""

import json
import os
import tempfile
import tkinter as tk
import unittest

from app.input.chart_parser import ChartParser
from app.music.chart_alignment import ChartAlignment
from app.music.chord_chart import ChordChart
from app.music.musical_clock import MusicalClock
from app.music.sectionizer import text_has_section_headers
from app.music.position_estimator import PositionEstimator, TrackingState
from app.song.song import Song
from app.song.song_session import SongSession
from app.ui.edit_history import ChartEditState, EditHistory
from app.ui.main_window import MainWindow


INTRO_CHART = """Minha Música
12345
Tom: G
BPM: 120

[Intro]

G    D
Em   C

[Verse]
G C D
"""


class TestStartAnchor(unittest.TestCase):
    def test_inline_intro_uses_physical_first_line_and_first_chord(self):
        text = "Título\nTom: G\n\n[Intro] G D\n    Em C\n\n[Verse]\nAm F\n"
        chart = ChartParser.parse(text)
        self.assertEqual(chart.sections[0].name, "Intro")
        self.assertEqual(chart.sections[0].chords[0].line_number, 4)
        self.assertEqual(chart.sections[0].chords[2].line_number, 5)
        session = SongSession(Song(chart_text=text))
        self.assertEqual(session.start_anchor.line_index, 4)
        self.assertEqual(session.chart_position.current_chord, "G")
        self.assertEqual(session.chart_position.line_index, 4)
        self.assertTrue(text_has_section_headers(text))
        self.assertIsNone(ChartParser.parse_section_header("[Ritmo Padrão] 165 bpm"))

    def test_text_repairs_stale_stored_intro_coordinates_on_load(self):
        text = "Título\nTom: G\n\n[Intro] G D\n    Em C\n\n[Verse]\nAm F\n"
        stale = ChartParser.parse("[Verse]\nC\n[Intro]\nG D\nEm C").to_dict()
        song = Song(chart_text=text, chart_data=stale)
        session = SongSession(song)
        self.assertEqual(session.start_anchor.section_name, "Intro")
        self.assertEqual(session.start_anchor.line_index, 4)
        session.start()
        self.assertEqual(session.chart_position.event_index, 0)
        self.assertEqual(session.chart_position.line_index, 4)

    def test_wrong_next_chord_frames_cannot_skip_first_intro_line_at_start(self):
        from app.music.musical_context import MusicalContext
        session = SongSession(Song(chart_text="[Intro] G D\nEm C\n[Verse]\nAm F"))
        for timestamp in (0.0, .12, .25, .38, .51):
            session.update_audio_tick(timestamp, detected_chord="D",
                                      detected_confidence=.95,
                                      source_context=MusicalContext(audio_activity=.2))
        self.assertEqual(session.chart_position.event_index, 0)
        self.assertEqual(session.chart_position.line_index, 1)
        self.assertEqual(session.chart_position.current_chord, "G")

    def test_first_playable_event_ignores_preamble_and_uses_intro(self):
        alignment = ChartAlignment(ChartParser.parse(INTRO_CHART))
        anchor = alignment.start_anchor
        self.assertIsNotNone(anchor)
        self.assertEqual(anchor.section_index, 0)
        self.assertEqual(anchor.section_occurrence, 1)
        self.assertEqual(anchor.event_index, 0)
        self.assertEqual(anchor.line_index, 8)
        self.assertEqual(anchor.chord, "G")
        self.assertEqual(anchor.section_name, "Intro")

    def test_empty_intro_falls_through_to_first_playable_section(self):
        chart = ChartParser.parse("[Intro]\n(anotação)\n[Verse]\nAm F G\n")
        anchor = ChartAlignment(chart).start_anchor
        self.assertIsNotNone(anchor)
        self.assertEqual(anchor.section_name, "Verse")
        self.assertEqual(anchor.chord, "Am")

    def test_no_intro_uses_first_playable_verse_event(self):
        anchor = ChartAlignment(ChartParser.parse("[Verso]\nC G\n")).start_anchor
        self.assertEqual(anchor.section_name, "Verso")
        self.assertEqual(anchor.chord, "C")

    def test_unknown_and_isolated_next_chord_do_not_leave_start_window(self):
        alignment = ChartAlignment(ChartParser.parse(INTRO_CHART))
        clock = MusicalClock(120.0, "4/4")
        estimator = PositionEstimator(alignment, clock)

        clock.update(0.25)
        unknown = estimator.update(0.25, "--", 0.0)
        self.assertEqual(unknown.event_index, 0)
        self.assertEqual(unknown.current_chord, "G")
        self.assertEqual(estimator.tracking_state, TrackingState.UNCERTAIN)

        clock.update(0.40)
        isolated = estimator.update(0.40, "D", 0.98)
        self.assertEqual(isolated.event_index, 0)
        self.assertFalse(estimator.start_anchor_confirmed)

    def test_audio_confirms_anchor_then_sequence_can_continue(self):
        alignment = ChartAlignment(ChartParser.parse(INTRO_CHART))
        clock = MusicalClock(120.0, "4/4")
        estimator = PositionEstimator(alignment, clock)
        clock.update(0.0)
        confirmed = estimator.update(0.0, "G", 0.95)
        self.assertEqual(confirmed.event_index, 0)
        self.assertTrue(estimator.start_anchor_confirmed)

        clock.update(2.0)
        following = estimator.update(2.0, "D", 0.95)
        self.assertEqual(following.event_index, 1)
        self.assertEqual(following.current_chord, "D")

    def test_restart_uses_anchor_but_resume_preserves_position(self):
        session = SongSession(Song(title="Anchor", chart_text=INTRO_CHART, bpm=120.0))
        session.seek_to_bar(3)
        before_pause = session.chart_position.event_index
        session.pause()
        session.resume()
        self.assertEqual(session.chart_position.event_index, before_pause)

        restarted = session.restart()
        self.assertEqual(restarted.event_index, session.start_anchor.event_index)
        self.assertEqual(restarted.current_chord, "G")
        self.assertEqual(session.position_estimator.consumed_chart_events, set())

    def test_duplicate_section_ids_remain_unique(self):
        chart = ChartParser.parse("[Intro]\nC G\n[Verse]\nAm F\n")
        chart.duplicate_section(0)
        chart.duplicate_section(0)
        ids = [section.id for section in chart.sections]
        self.assertEqual(len(ids), len(set(ids)))


class TestEditHistory(unittest.TestCase):
    def test_linear_undo_redo_and_new_action_clears_redo(self):
        history = EditHistory()
        a = ChartEditState("A")
        b = ChartEditState("B")
        c = ChartEditState("C")
        history.reset(a)
        history.record(a, b)
        self.assertEqual(history.undo(), a)
        self.assertEqual(history.redo(), b)
        self.assertEqual(history.undo(), a)
        history.record(a, c)
        self.assertFalse(history.can_redo)
        self.assertEqual(history.undo(), a)


class TestIntroEditorIntegration(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = tk.Tk()
        self.root.withdraw()
        self.project_path = os.path.join(self.tmp.name, "project.json")
        self.app = MainWindow(self.root, project_file_path=self.project_path)
        self.song = self.app._active_song()
        self.app.project_manager.update_song_chart(self.song.id, "[Intro]\nG D\n[Verse]\nAm F\n")
        self.app._render_chart_text(self.song)

    def tearDown(self):
        self.root.destroy()
        self.tmp.cleanup()

    def test_intro_is_block_highlighted_at_start_and_duplicate_is_atomic(self):
        _pre, ranges = self.app._block_line_ranges(
            self.app.text_chart_view.get("1.0", "end-1c"))
        self.assertEqual(ranges[0][2], "INTRO")

        session = self.app.project_manager.active_session
        session.start()
        self.app._highlight_chart_position(session.chart_position)
        active = self.app.text_chart_view.tag_ranges("active_chord")
        self.assertTrue(active)
        self.assertEqual(str(active[0]).split(".")[0], "2")

        self.app._selected_block_index = 0
        self.app._on_copy_block()
        self.assertEqual(len(self.song.chart_data["sections"]), 3)
        ids = [section["id"] for section in self.song.chart_data["sections"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(self.song.chart_data["sections"][0]["section_type"], "INTRO")
        self.assertEqual(self.song.chart_data["sections"][1]["section_type"], "INTRO")
        alignment = self.app.project_manager.active_session.alignment
        first_intro = alignment.get_position_for_event(0)
        second_intro = alignment.get_position_for_event(2)
        self.assertEqual(first_intro.section_occurrence, 1)
        self.assertEqual(second_intro.section_occurrence, 2)

        self.app.root.focus_set()
        self.assertEqual(self.app._on_undo_shortcut(), "break")
        self.assertEqual(len(self.song.chart_data["sections"]), 2)
        self.assertEqual(self.song.chart_data["sections"][0]["section_type"], "INTRO")

        self.assertEqual(self.app._on_redo_shortcut(), "break")
        self.assertEqual(len(self.song.chart_data["sections"]), 3)

    def test_remove_intro_undo_redo_restores_anchor_and_autosave(self):
        self.app._selected_block_index = 0
        self.app._on_delete_block()
        session = self.app.project_manager.active_session
        self.assertEqual(session.start_anchor.section_name, "Verse")
        self.assertEqual(session.start_anchor.chord, "Am")

        self.app.root.focus_set()
        self.app._on_undo_shortcut()
        self.assertEqual(session.start_anchor.section_name, "Intro")
        self.assertEqual(session.start_anchor.chord, "G")

        self.app._on_redo_shortcut()
        self.assertEqual(session.start_anchor.section_name, "Verse")
        with open(self.project_path, "r", encoding="utf-8") as stream:
            saved = json.load(stream)
        saved_song = saved["setlists"][0]["songs"][0]
        self.assertNotIn("[Intro]", saved_song["chart_text"])

    def test_inline_intro_is_editable_block_and_play_starts_on_its_line(self):
        text = "Título\nTom: G\n\n[Intro] G D\n    Em C\n\n[Verse]\nAm F\n"
        self.app.project_manager.update_song_chart(self.song.id, text)
        self.app._render_chart_text(self.song)
        preamble, ranges = self.app._block_line_ranges(text)
        self.assertEqual(preamble, 3)
        self.assertEqual(ranges[0], (4, 6, "INTRO"))
        session = self.app.project_manager.active_session
        session.start()
        self.app._highlight_chart_position(session.chart_position)
        self.assertEqual(str(self.app.text_chart_view.tag_ranges("active_chord")[0]).split(".")[0], "4")

        self.app._selected_block_index = 0
        self.app._on_copy_block()
        self.assertEqual(len(self.song.chart_data["sections"]), 3)
        self.assertEqual([s["section_type"] for s in self.song.chart_data["sections"][:2]],
                         ["INTRO", "INTRO"])
        self.app._selected_block_index = 0
        self.app._on_delete_block()
        self.assertEqual(self.app.project_manager.active_session.start_anchor.section_name, "Intro")
        self.assertEqual(len(self.song.chart_data["sections"]), 2)


if __name__ == "__main__":
    unittest.main()
