"""Cursor de consumo: cifras com progressões repetidas devem avançar em ordem."""

import unittest

from app.input.chart_parser import ChartParser
from app.music.chart_alignment import ChartAlignment
from app.music.musical_clock import MusicalClock
from app.music.position_estimator import PositionEstimator, TrackingState


REPEATED_PROGRESSIONS = """[Intro]
C Am F G
[Verso]
C Am F G
[Refrão]
C Am F G
"""


class TestChartCursor(unittest.TestCase):
    def make_estimator(self, chart_text=REPEATED_PROGRESSIONS):
        chart = ChartParser.parse(chart_text)
        clock = MusicalClock(bpm=120.0, meter="4/4")
        return PositionEstimator(ChartAlignment(chart), clock), clock

    def tick(self, estimator, clock, time, chord="--", confidence=0.0):
        clock.update(time)
        return estimator.update(time, chord, confidence)

    def test_identical_intro_verse_chorus_consumes_forward(self):
        estimator, clock = self.make_estimator()
        sections = []
        for index, chord in enumerate(("C", "Am", "F", "G") * 3):
            pos = self.tick(estimator, clock, index * 2.0, chord, .92)
            sections.append(pos.section_name)
            self.assertEqual(pos.current_chord, chord)
            self.assertEqual(pos.event_index, index)

        self.assertEqual(sections[:4], ["Intro"] * 4)
        self.assertEqual(sections[4:8], ["Verso"] * 4)
        self.assertEqual(sections[8:], ["Refrão"] * 4)
        self.assertEqual(estimator.chart_cursor_index, 11)
        self.assertEqual(estimator.consumed_chart_events, set(range(11)))

    def test_unknown_detection_does_not_block_chart_consumption(self):
        estimator, clock = self.make_estimator()
        self.tick(estimator, clock, 0.0, "C", .9)
        positions = [self.tick(estimator, clock, time) for time in (2.0, 4.0, 6.0, 8.0)]
        self.assertEqual([pos.current_chord for pos in positions], ["Am", "F", "G", "C"])
        self.assertEqual(positions[-1].section_name, "Verso")
        self.assertTrue(all(a.event_index < b.event_index for a, b in zip(positions, positions[1:])))

    def test_audio_cannot_backslide_to_consumed_identical_intro(self):
        estimator, clock = self.make_estimator()
        # A posição temporal chega ao Verso; os mesmos acordes não podem puxar
        # o cursor de volta para a Intro já consumida.
        self.tick(estimator, clock, 8.0, "C", .9)
        self.assertEqual(estimator.chart_cursor_index, 4)
        for offset, chord in enumerate(("C", "Am", "F", "G")):
            pos = self.tick(estimator, clock, 8.1 + offset * .05, chord, .9)
        self.assertEqual(pos.section_name, "Verso")
        self.assertEqual(pos.event_index, 7)
        self.assertTrue(set(range(4)).issubset(estimator.consumed_chart_events))

    def test_declared_repeat_creates_distinct_event_occurrences(self):
        estimator, clock = self.make_estimator("""[Intro]
C Am x2
[Verso]
F G
""")
        alignment = estimator._alignment
        self.assertEqual(alignment.event_count, 6)
        first = alignment.get_position_for_event(0)
        repeated = alignment.get_position_for_event(2)
        self.assertEqual(first.section_name, repeated.section_name)
        self.assertEqual(first.section_occurrence, 1)
        self.assertEqual(repeated.section_occurrence, 2)
        pos = self.tick(estimator, clock, 4.0, "C", .9)
        self.assertEqual(pos.section_occurrence, 2)
        self.assertEqual(pos.section_event_index, 0)

    def test_strong_lost_progression_can_reopen_a_real_previous_section(self):
        chart = """[Intro]
C Am F G
[Verso]
Dm G C F
[Refrão]
Em C G D
"""
        estimator, clock = self.make_estimator(chart)
        # A cifra estava no Refrão. Uma recuperação real é explicitamente LOST
        # e traz uma sequência completa e distinta do Verso.
        self.tick(estimator, clock, 16.0, "Em", .9)
        self.assertEqual(estimator._cursor_position(1, 16).section_name, "Refrão")
        estimator._tracking_state = TrackingState.LOST
        for offset, chord in enumerate(("Dm", "G", "C", "F")):
            pos = self.tick(estimator, clock, 16.1 + offset * .05, chord, .95)
        self.assertEqual(pos.section_name, "Verso")
        self.assertEqual(pos.current_chord, "F")
        self.assertLess(pos.event_index, 8)


if __name__ == "__main__":
    unittest.main()
