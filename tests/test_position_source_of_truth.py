"""Fonte única de posição: relógio bruto, cifra reancorada e consumidores."""
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from app.analysis.audio_analyzer import AudioAnalyzer
from app.instruments.bass_model import BassPatternType
from app.instruments.bass_player import BassPlayer
from app.instruments.registry import VirtualPlayerRegistry
from app.music.chart_alignment import ChartPosition
from app.music.harmonic_rhythm import HarmonicRhythmEvent
from app.music.musical_context import MusicalContext
from app.music.position_estimator import ProgressionMatch, TrackingState
from app.project.project_manager import ProjectManager
from app.song.song import Song
from app.song.song_session import SongSession
from app.ui.main_window import MainWindow


CHART = "[Intro]\n" + "\n".join(["C"] * 29) + "\n[Refrao]\nAm F G\n[Ponte]\nC Dm Bb E7\n[Final]\n" + "\n".join(["C"] * 32)


class TestPositionSourceOfTruth(unittest.TestCase):
    def session(self):
        return SongSession(Song(title="Position test", chart_text=CHART, bpm=120.0))

    def assert_consistent(self, session, raw_bar, musical_bar):
        self.assertEqual(session.clock_bar, raw_bar)
        self.assertEqual(session.current_bar, musical_bar)
        self.assertEqual(session.chart_position.current_bar, musical_bar)
        self.assertEqual(session.position_estimator.current_estimated_position.bar, musical_bar)
        self.assertEqual(session.context.bar, musical_bar)
        self.assertEqual(session.context.current_bar, musical_bar)
        self.assertEqual(session.context.clock_bar, raw_bar)
        self.assertEqual(session.current_beat, int(session.chart_position.current_beat))
        self.assertEqual(session.context.beat, session.current_beat)
        self.assertEqual(session.context.clock_beat, session.clock_beat)
        self.assertAlmostEqual(session.context.beat_position, session.chart_position.current_beat - session.current_beat)
        self.assertEqual(session.context.line_index, session.chart_position.line_index)
        self.assertEqual(session.context.tracking_state, session.chart_position.tracking_state)
        self.assertEqual(session.structure_analyzer.structure.current_position.current_bar, musical_bar)
        self.assertEqual(session.prediction.source_bar, musical_bar)
        self.assertEqual(session.prediction.source_beat, session.current_beat)

    def confirm_recovery_progression(self, session, chords):
        for index, chord in enumerate(chords):
            event = HarmonicRhythmEvent(chord, index * 4.0, 4.0, 4.0, .95, "TEST")
            session.position_estimator._recent_harmonic_rhythm.append(
                ((event.symbol, event.start_beat), event))

    def anchor_forward(self, session):
        session.update_audio_tick(38.0, detected_chord="C", detected_confidence=0.9)
        self.assert_consistent(session, 20, 20)
        # Relocalização global só tem autoridade durante recuperação explícita.
        session.position_estimator._tracking_state = TrackingState.LOST
        session.position_estimator._startup_lock = False
        self.confirm_recovery_progression(session, ("C", "Dm", "Bb", "E7"))
        for i, chord in enumerate(["Dm", "Bb", "E7"]):
            session.update_audio_tick(38.1 + i * 0.05, detected_chord=chord, detected_confidence=0.9)
        self.assert_consistent(session, 20, 36)
        return session

    def test_no_offset_clock_and_musical_position_match(self):
        session = self.session()
        session.update_audio_tick(38.75)
        self.assert_consistent(session, 20, 20)
        self.assertEqual(session.clock_beat, 2)
        self.assertEqual(session.current_beat, 2)
        self.assertEqual(session.position_estimator.bar_offset, 0)

    def test_audio_reanchors_forward_then_advances_with_offset(self):
        session = self.anchor_forward(self.session())
        self.assertEqual(session.position_estimator.bar_offset, -16)
        session.update_audio_tick(40.0)
        self.assert_consistent(session, 21, 37)

    def test_audio_reanchors_backward(self):
        session = self.session()
        session.update_audio_tick(94.0, detected_chord="C", detected_confidence=0.9)
        session.position_estimator._tracking_state = TrackingState.LOST
        session.position_estimator._startup_lock = False
        self.confirm_recovery_progression(session, ("C", "Am", "F", "G"))
        for i, chord in enumerate(["Am", "F", "G"]):
            session.update_audio_tick(94.1 + i * 0.05, detected_chord=chord, detected_confidence=0.9)
        self.assert_consistent(session, 48, 32)
        self.assertEqual(session.position_estimator.bar_offset, 16)
        session.update_audio_tick(96.0)
        self.assert_consistent(session, 49, 33)

    def test_detection_loss_preserves_offset_and_recovery_confirms_position(self):
        session = self.anchor_forward(self.session())
        for raw_bar in range(21, 29):
            session.update_audio_tick((raw_bar - 1) * 2.0, detected_chord="UNKNOWN", detected_confidence=0.0)
            self.assert_consistent(session, raw_bar, raw_bar + 16)
        self.assertEqual(session.tracking_state, "LOST")
        session.update_audio_tick(56.25, detected_chord="C", detected_confidence=0.9)
        self.assert_consistent(session, 29, 45)
        self.assertEqual(session.tracking_state, "TRACKING")

    def test_sustained_chord_does_not_replay_old_localization(self):
        session = self.anchor_forward(self.session())
        transition_count = len(session.position_estimator.position_transition_log)
        global_count = session.position_estimator.position_stability_metrics["global_relocations"]
        session.update_audio_tick(40.0, detected_chord="E7", detected_confidence=0.9)
        self.assertEqual(session.current_bar, 36)
        self.assertEqual(len(session.position_estimator.position_transition_log), transition_count)
        self.assertEqual(
            session.position_estimator.position_stability_metrics["global_relocations"],
            global_count)

    def test_local_recovery_persists_offset_next_tick(self):
        session = self.anchor_forward(self.session())
        with patch.object(session.position_estimator, "_global_localize", return_value=None), patch.object(session.position_estimator, "_probe_local_search_window", return_value=(39, "C")):
            session.update_audio_tick(40.0, detected_chord="Dm", detected_confidence=0.9)
            session.update_audio_tick(42.0, detected_chord="Dm", detected_confidence=0.9)
        self.assert_consistent(session, 22, 39)
        session.update_audio_tick(44.0)
        self.assert_consistent(session, 23, 40)

    def test_progression_recovery_persists_offset_next_tick(self):
        session = self.anchor_forward(self.session())
        match = ProgressionMatch(["Am", "F", "G"], ["Am", "F", "G"], 37, 39, 0.9)
        with patch.object(session.position_estimator, "_global_localize", return_value=None), patch.object(session.position_estimator, "_probe_local_search_window", return_value=None), patch.object(session.position_estimator, "find_progression_match", return_value=match):
            session.update_audio_tick(40.0, detected_chord="Dm", detected_confidence=0.9)
            session.update_audio_tick(42.0, detected_chord="Dm", detected_confidence=0.9)
        self.assert_consistent(session, 22, 39)
        session.update_audio_tick(44.0)
        self.assert_consistent(session, 23, 40)

    def test_song_switch_discards_previous_offset(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = ProjectManager(str(Path(directory) / "project.json"))
            first = manager.import_song(title="A", chart_text=CHART)
            second = manager.import_song(title="B", chart_text=CHART)
            self.anchor_forward(manager.open_song(first))
            session = manager.open_song(second)
            self.assertEqual(session.clock_bar, 1)
            self.assertEqual(session.current_bar, 1)
            self.assertEqual(session.position_estimator.bar_offset, 0)
            self.assertEqual(session.tracking_state, "TRACKING")
            self.assertIsNone(session.prediction)
            session.update_audio_tick(2.0)
            self.assert_consistent(session, 2, 2)

    def test_reset_restores_clock_chart_context_and_tracking(self):
        session = self.anchor_forward(self.session())
        session.reset()
        self.assertEqual(session.clock_bar, 1)
        self.assertEqual(session.current_bar, 1)
        self.assertEqual(session.current_beat, 1)
        self.assertEqual(session.context.bar, 1)
        self.assertEqual(session.context.bar_offset, 0)
        self.assertEqual(session.chart_position.line_index, session.context.line_index)
        self.assertEqual(session.tracking_state, "TRACKING")
        self.assertIsNone(session.prediction)
        session.update_audio_tick(0.0)
        self.assert_consistent(session, 1, 1)

    def test_chart_transposition_and_edit_keep_musical_position(self):
        session = self.anchor_forward(self.session())
        clock_time = session.clock.elapsed_time
        session.transpose_to("D Major")
        self.assert_consistent(session, 20, 36)
        self.assertEqual(session.current_chord, "F#7")
        self.assertEqual(session.clock.elapsed_time, clock_time)
        session.update_chart_text(CHART)
        self.assert_consistent(session, 20, 36)
        self.assertEqual(session.current_chord, "E7")

    def test_manual_chart_navigation_reanchors_without_moving_audio_time(self):
        session = self.anchor_forward(self.session())
        clock_time = session.clock.elapsed_time
        session.next_bar()
        self.assert_consistent(session, 20, 37)
        self.assertEqual(session.clock.elapsed_time, clock_time)
        session.update_audio_tick(40.0)
        self.assert_consistent(session, 21, 38)

    def test_bass_and_registry_receive_musical_coordinates(self):
        session = self.anchor_forward(self.session())
        bass = BassPlayer(pattern=BassPatternType.ROOT)
        registry = VirtualPlayerRegistry(bass_player=bass)
        observer = Mock()
        registry.register_player("observer", observer)
        events = registry.dispatch_context(session.context)
        self.assertEqual(events["bass"].bar, 36)
        self.assertEqual(events["bass"].beat,
                         session.current_beat % session.clock.beats_per_bar + 1)
        self.assertIs(observer.on_musical_context.call_args.args[0], session.context)
        self.assertEqual(bass.current_decision.chord, session.current_chord)

    def test_pattern_memory_and_prediction_use_reanchored_phrase(self):
        session = self.anchor_forward(self.session())
        for raw_bar in range(21, 25):
            session.update_audio_tick((raw_bar - 1) * 2.0)
        patterns = session.structure_analyzer.pattern_memory.get_known_patterns()
        self.assertTrue(patterns)
        occurrences = [occ for pattern in patterns for occ in pattern.occurrences]
        self.assertTrue(any(occ.start_bar == 36 and occ.end_bar == 39 for occ in occurrences))
        self.assertEqual(session.prediction.source_bar, 40)
        self.assertEqual(session.structure_analyzer.structure.current_position.current_section, session.current_section)

    def test_prediction_input_and_follow_hud_use_same_snapshot(self):
        session = self.anchor_forward(self.session())
        window = MainWindow.__new__(MainWindow)
        for name in ("lbl_playalong_section", "lbl_playalong_chord", "lbl_playalong_next_chord", "lbl_playalong_bar_beat", "lbl_playalong_bpm", "lbl_playalong_fusion", "lbl_playalong_fusion_sub"):
            setattr(window, name, Mock())
        window._follow_mode_enabled = True
        window._highlight_chart_position = Mock()
        window._update_playalong_hud_from_session(session)
        window._highlight_chart_position.assert_called_once_with(session.chart_position)
        text = window.lbl_playalong_bar_beat.config.call_args.kwargs["text"]
        self.assertIn("Comp. 36", text)
        self.assertNotIn("Comp. 20", text)
        window._follow_mode_enabled = False
        window._highlight_chart_position.reset_mock()
        window._update_playalong_hud_from_session(session)
        window._highlight_chart_position.assert_not_called()

    def test_debug_reports_both_coordinates_and_offset(self):
        session = self.anchor_forward(self.session())
        diagnostic = session.get_position_diagnostics()
        self.assertEqual(diagnostic["clock_bar"], 20)
        self.assertEqual(diagnostic["current_bar"], 36)
        self.assertEqual(diagnostic["bar_offset"], -16)
        self.assertIn("CLOCK: Bar 20", session.format_position_diagnostics())
        self.assertIn("MUSICAL POSITION: Bar 36", session.format_position_diagnostics())

    def test_audio_pipeline_localizes_before_dispatching_once(self):
        session = self.anchor_forward(self.session())
        analyzer = AudioAnalyzer()
        registry = VirtualPlayerRegistry(bass_player=analyzer.bass_player)
        analyzer.bass_player.pattern = BassPatternType.ROOT
        original = session.update_audio_tick
        calls = []
        def capture(*args, **kwargs):
            calls.append(kwargs)
            return original(*args, **kwargs)
        with patch.object(session, "update_audio_tick", side_effect=capture), patch.object(analyzer.band, "dispatch_context") as old_dispatch:
            ctx = analyzer.analyze_chunk(np.zeros(4096, dtype=np.float32), 44100, 40.0, session=session, players=registry)
        self.assertEqual(len(calls), 1)
        old_dispatch.assert_not_called()
        self.assertIs(ctx, session.context)
        self.assertIs(analyzer.context, ctx)
        self.assertIs(analyzer.structure_analyzer, session.structure_analyzer)
        self.assertEqual(ctx.bar, 36)
        self.assertEqual(ctx.clock_bar, 21)
        self.assertEqual(analyzer.bass_player.current_event.bar, 36)
        self.assertEqual(analyzer.context_manager.context.bar, 21)
        self.assertEqual(session.prediction.source_bar, 36)
        self.assertEqual(len(analyzer.bass_player.get_recent_events()), 1)

    def test_follow_scroll_uses_official_line_even_if_chord_is_on_neighbor(self):
        window = MainWindow.__new__(MainWindow)
        window.text_chart_view = Mock()
        window.text_chart_view.search.side_effect = ["", "70.2"]
        window._follow_mode_enabled = True
        window.root = Mock()
        window.root.focus_get.return_value = None
        position = ChartPosition(current_bar=36, line_index=71, current_chord="E7")
        window._highlight_chart_position(position)
        window.text_chart_view.see.assert_called_once_with("71.0")
