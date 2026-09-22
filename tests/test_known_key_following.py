"""Tom informado e continuidade do cursor após a introdução."""

import unittest
from unittest.mock import patch

import numpy as np

from app.analysis.audio_analyzer import AudioAnalyzer
from app.input.chart_parser import ChartParser
from app.music.chart_alignment import ChartAlignment
from app.music.musical_clock import MusicalClock
from app.music.musical_context import MusicalContext
from app.music.position_estimator import PositionEstimator
from app.song.song import Song
from app.song.song_session import SongSession


class TestKnownKeyFollowing(unittest.TestCase):
    def test_chart_key_overrides_wrong_audio_key(self):
        session = SongSession(Song(chart_text="Tom: G\n[Intro]\nG D\n[Verse]\nEm C", key="G Major"))
        heard = MusicalContext(key="C Major", key_confidence=.94)
        session.update_audio_tick(0.0, detected_key="C Major", source_context=heard)
        self.assertEqual(session.context.key, "G Major")
        self.assertEqual(session.context.key_confidence, 1.0)

    def test_analyzer_does_not_estimate_key_with_active_chart(self):
        session = SongSession(Song(chart_text="Tom: G\n[Intro]\nG D", key="G Major"))
        analyzer = AudioAnalyzer()
        with patch.object(analyzer._key_detector, "estimate_key", side_effect=AssertionError("key guessed")):
            analyzer.analyze_chunk(np.zeros(4096, dtype=np.float32), 44100, 0.0, session=session)
        self.assertEqual(session.context.key, "G Major")

    def test_selected_key_remains_authoritative_after_transposition(self):
        session = SongSession(Song(chart_text="[Intro]\nG D\n[Verse]\nEm C", key="G Major"))
        session.transpose_to("A Major")
        session.update_audio_tick(0.0, detected_key="C Major",
                                  source_context=MusicalContext(key="C Major"))
        self.assertEqual(session.context.key, "A Major")

    def test_wrong_candidate_disables_current_chord_prior(self):
        session = SongSession(Song(chart_text="[Intro]\nG D", key="G Major"))
        session.context.chord_candidate_root = "D"
        session.context.chord_candidate_frames = 3
        session.context.chord_candidate_confidence = .8
        expected, enabled = AudioAnalyzer()._harmonic_prior_for(session)
        self.assertIsNone(expected)
        self.assertFalse(enabled)

    @staticmethod
    def make_estimator(chart):
        clock = MusicalClock(120.0, "4/4")
        return PositionEstimator(ChartAlignment(ChartParser.parse(chart)), clock), clock

    @staticmethod
    def tick(estimator, clock, timestamp, chord="--", confidence=0.0):
        clock.update(timestamp)
        return estimator.update(timestamp, chord, confidence)

    def test_sustained_chord_does_not_shift_clock_alignment_every_frame(self):
        estimator, clock = self.make_estimator("[Intro]\nG D Em C\n[Verse]\nAm F")
        self.tick(estimator, clock, 0.0, "G", .95)
        self.tick(estimator, clock, 2.0, "G", .95)
        self.assertEqual(estimator.bar_offset, 0)
        pos = self.tick(estimator, clock, 4.0)
        self.assertEqual(pos.current_chord, "Em")
        self.assertEqual(pos.section_name, "Intro")

    def test_repeated_chart_chord_advances_on_clock(self):
        estimator, clock = self.make_estimator("[Intro]\nG G D\n[Verse]\nEm C")
        self.tick(estimator, clock, 0.0, "G", .95)
        pos = self.tick(estimator, clock, 2.0, "G", .95)
        self.assertEqual(pos.event_index, 1)
        self.assertEqual(pos.current_chord, "G")

    def test_supported_next_chord_can_confirm_after_two_frames(self):
        estimator, clock = self.make_estimator("[Intro]\nG D Em C\n[Verse]\nAm F")
        self.tick(estimator, clock, 0.0, "G", .95)
        first = self.tick(estimator, clock, .40, "D", .78)
        second = self.tick(estimator, clock, .55, "D", .78)
        self.assertEqual(first.event_index, 0)
        self.assertEqual(second.event_index, 1)

    def test_two_frame_confirmation_crosses_intro_into_verse(self):
        estimator, clock = self.make_estimator("[Intro]\nG D\n[Verse]\nEm C")
        for time, chord in ((0.0, "G"), (.40, "D"), (.55, "D"),
                            (1.0, "Em"), (1.15, "Em")):
            position = self.tick(estimator, clock, time, chord, .78)
        self.assertEqual(position.section_name, "Verse")
        self.assertEqual(position.event_index, 2)

    def test_local_candidate_prefers_forward_nearby_match(self):
        estimator, _ = self.make_estimator("[Intro]\nD C Am F\n[Verse]\nG D Em C")
        match = estimator._probe_local_search_window(5, "D", 4)
        self.assertEqual(match[0], 6)


if __name__ == "__main__":
    unittest.main()
