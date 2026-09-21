"""Regressões: beat do relógio não pode consumir eventos harmônicos sozinho."""

import unittest

from app.input.chart_parser import ChartParser
from app.music.chart_alignment import ChartAlignment
from app.music.musical_clock import MusicalClock
from app.music.position_estimator import PositionEstimator


class TestChartCursorStability(unittest.TestCase):
    def make_estimator(self, text, bpm=120.0):
        clock = MusicalClock(bpm=bpm, meter="4/4")
        return PositionEstimator(ChartAlignment(ChartParser.parse(text)), clock), clock

    @staticmethod
    def tick(estimator, clock, beat, chord, confidence=.95):
        timestamp = beat * clock.beat_duration
        clock.update(timestamp)
        return estimator.update(timestamp, chord, confidence)

    def test_cursor_follows_harmonic_durations_not_every_beat(self):
        estimator, clock = self.make_estimator("[Intro]\nG D Em C")
        played = (("G", 4), ("D", 2), ("Em", 2), ("C", 8))
        indices = []
        beat = 0
        for chord, duration in played:
            for _ in range(duration):
                position = self.tick(estimator, clock, beat, chord)
                indices.append(position.event_index)
                beat += 1

        self.assertEqual(indices, [0] * 4 + [1] * 2 + [2] * 2 + [3] * 8)
        self.assertEqual(estimator.last_position_change_reason, "confirmed expected next chord")
        self.assertEqual([entry["reason"] for entry in estimator.position_transition_log], [
            "confirmed expected next chord", "confirmed expected next chord", "confirmed expected next chord",
        ])

    def test_brief_unexpected_chord_keeps_current_event(self):
        estimator, clock = self.make_estimator("[Intro]\nG D Em C\n[Chorus]\nF# B C# D")
        for beat, chord in enumerate(("G", "G", "F#", "G")):
            position = self.tick(estimator, clock, beat, chord)
        self.assertEqual(position.current_chord, "G")
        self.assertEqual(position.event_index, 0)
        self.assertEqual(estimator.position_transition_log, [])

    def test_long_chord_remains_active_for_sixteen_beats(self):
        estimator, clock = self.make_estimator("[Intro]\nC F")
        for beat in range(16):
            position = self.tick(estimator, clock, beat, "C")
            self.assertEqual(position.event_index, 0)
            self.assertEqual(position.current_chord, "C")
        position = self.tick(estimator, clock, 16, "F")
        self.assertEqual(position.event_index, 1)
        self.assertEqual(position.current_chord, "F")

    def test_fast_chords_can_still_change_each_beat(self):
        estimator, clock = self.make_estimator("[Intro]\nC Am F G")
        positions = [self.tick(estimator, clock, beat, chord)
                     for beat, chord in enumerate(("C", "Am", "F", "G"))]
        self.assertEqual([position.event_index for position in positions], [0, 1, 2, 3])

    def test_duration_is_measured_in_beats_at_multiple_tempi(self):
        for bpm in (80.0, 120.0, 160.0):
            estimator, clock = self.make_estimator("[Intro]\nG D", bpm)
            for beat in range(4):
                position = self.tick(estimator, clock, beat, "G")
                self.assertEqual(position.event_index, 0)
            position = self.tick(estimator, clock, 4, "D")
            self.assertEqual(position.event_index, 1, f"BPM {bpm}")


if __name__ == "__main__":
    unittest.main()
