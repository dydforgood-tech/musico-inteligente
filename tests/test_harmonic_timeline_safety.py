"""Barreira semântica: documento visual nunca pode contaminar o motor harmônico."""

import unittest

from app.input.chart_parser import ChartParser
from app.input.chart_semantic_classifier import ChartSemanticClassifier, SemanticLineType
from app.instruments.bass_player import BassPlayer
from app.music.chart_alignment import ChartAlignment
from app.music.chart_audio_fusion import ChartAudioFusion
from app.music.musical_clock import MusicalClock
from app.music.musical_context import MusicalContext
from app.music.position_estimator import PositionEstimator, TrackingState
from app.song.song import Song
from app.song.song_session import SongSession


NOISY_CHART = """Minha Música Favorita
12345
2026
A E I O U
Bruno
Tom: G
BPM: 120
Capotraste na 2ª casa

[Intro]
G    D    Em    C
essa é uma letra
1 2 3 4

[Verso]
G    D    Em    C
E|--3-5-7---
B|----------

[Refrão]
C G D Em
"""


class TestHarmonicTimelineSafety(unittest.TestCase):
    def test_numbers_have_explicit_non_harmonic_type(self):
        self.assertEqual(ChartSemanticClassifier.classify_line("12345").line_type, SemanticLineType.NUMBER)
        self.assertEqual(ChartSemanticClassifier.classify_line("1 2 3 4").line_type, SemanticLineType.NUMBER)

    def test_only_chord_lines_build_the_alignment_timeline(self):
        chart = ChartParser.parse(NOISY_CHART)
        timeline = ChartAlignment(chart)
        symbols = [timeline.get_position_for_event(i).current_chord
                   for i in range(timeline.event_count)]
        self.assertEqual(symbols, ["G", "D", "Em", "C", "G", "D", "Em", "C", "C", "G", "D", "Em"])
        self.assertNotIn("12345", symbols)
        self.assertNotIn("Bruno", symbols)

    def test_non_musical_input_does_not_move_position_or_search(self):
        chart = ChartParser.parse("[Intro]\nG D Em C\n[Verso]\nG D Em C")
        clock = MusicalClock(bpm=120)
        estimator = PositionEstimator(ChartAlignment(chart), clock)
        clock.update(0.0)
        before = estimator.update(0.0, "G", .95)
        clock.update(.1)
        after = estimator.update(.1, "12345", .99)
        self.assertEqual(after.event_index, before.event_index)
        self.assertEqual(after.section_name, before.section_name)
        self.assertEqual(estimator.tracking_state, TrackingState.TRACKING)
        self.assertEqual(estimator.last_search_mode, "LOCAL")
        self.assertEqual(estimator.current_estimated_position.action_note, "Ignored non-musical input")

    def test_session_rejects_title_and_number_before_rhythm_or_bass(self):
        session = SongSession(Song(chart_text="Minha Música\n12345\n[Intro]\nG D"))
        session.update_audio_tick(0.0, "G", .95)
        before = session.chart_position.event_index
        session.update_audio_tick(.1, "Minha Música", .99)
        self.assertEqual(session.chart_position.event_index, before)
        self.assertFalse(session.get_harmonic_input_diagnostic()["harmonic_eligible"])
        self.assertEqual(session.get_harmonic_input_diagnostic()["action"], "IGNORED")
        session.update_audio_tick(.5, "D", .95)
        self.assertEqual(session.current_chord, "D")

    def test_fusion_rejects_non_harmonic_detector_token(self):
        chart = ChartParser.parse("[Intro]\nG D")
        pos = ChartAlignment(chart).get_position_for_event(0)
        state = ChartAudioFusion().fuse(pos, "12345", .99, .1)
        self.assertEqual(state.detected_chord, "--")
        self.assertEqual(state.effective_chord, "G")

    def test_bass_holds_for_lost_low_confidence_or_invalid_chord(self):
        bass = BassPlayer(sample_rate=1000)
        lost = MusicalContext(chord="G", bpm=120, tracking_state="LOST",
                              follow_confidence_level="LOW", performance_state="PLAYING")
        self.assertIsNone(bass.on_musical_context(lost))
        invalid = MusicalContext(chord="12345", bpm=120, performance_state="PLAYING")
        self.assertIsNone(bass.on_musical_context(invalid))


if __name__ == "__main__":
    unittest.main()
