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


if __name__ == "__main__":
    unittest.main()
