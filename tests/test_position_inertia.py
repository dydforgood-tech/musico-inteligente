"""Posição musical tem inércia: notas isoladas nunca são autoridade de localização."""

import unittest

from app.input.chart_parser import ChartParser
from app.music.chart_alignment import ChartAlignment
from app.music.musical_clock import MusicalClock
from app.music.harmonic_rhythm import HarmonicRhythmEvent
from app.music.position_estimator import PositionEstimator, TrackingState
from app.song.song import Song
from app.song.song_session import SongSession


class TestPositionInertia(unittest.TestCase):
    def test_intro_notes_do_not_jump_across_identical_sections(self):
        session = SongSession(Song(chart_text="""[Intro]
G D Em C
[Verse]
G D Em C
[Chorus]
G D Em C
""", bpm=120))
        for note, timestamp in (("G4", .1), ("B4", .3), ("D4", .5)):
            from app.music.musical_context import MusicalContext
            session.update_audio_tick(timestamp,
                                      source_context=MusicalContext(note=note, note_confidence=.95))
        self.assertEqual(session.current_section, "Intro")
        self.assertEqual(session.chart_position.event_index, 0)
        self.assertEqual(session.position_estimator.position_stability_metrics["global_relocations"], 0)

    def test_arpeggio_and_passing_notes_do_not_move_cursor(self):
        chart = ChartParser.parse("[Intro]\nC G\n[Chorus]\nAm F")
        clock = MusicalClock(bpm=120)
        estimator = PositionEstimator(ChartAlignment(chart), clock)
        for index, note in enumerate(("C4", "E4", "G4", "E4", "C4", "D4", "A4", "B4")):
            timestamp = index * .1
            clock.update(timestamp)
            position = estimator.update(timestamp, detected_note=note, note_confidence=.95)
        self.assertEqual(position.event_index, 0)
        self.assertEqual(position.section_name, "Intro")
        self.assertEqual(estimator.position_transition_log, [])

    def test_visual_header_is_not_a_musical_line(self):
        chart = ChartParser.parse("""Minha Música Favorita
Artista qualquer
Tom: G

[Intro]
G D Em C
""")
        self.assertEqual(chart.line_map[0].line_type, "HEADER")
        self.assertEqual(chart.line_map[0].start_bar, 0)
        position = ChartAlignment(chart).get_position_for_event(0)
        self.assertEqual(position.section_name, "Intro")
        self.assertEqual(position.current_chord, "G")
        self.assertEqual(position.line_index, 6)

    def test_normal_sequence_has_no_global_or_large_jumps(self):
        chart = ChartParser.parse("""[Intro]
G D Em C
[Verse]
G D Em C
""")
        clock = MusicalClock(bpm=120)
        estimator = PositionEstimator(ChartAlignment(chart), clock)
        sections = []
        for beat, chord in enumerate(("G", "D", "Em", "C", "G", "D", "Em", "C")):
            timestamp = beat * .5
            clock.update(timestamp)
            position = estimator.update(timestamp, chord, .95)
            sections.append(f"{position.section_name} {position.section_event_index + 1}")
        self.assertEqual(sections, ["Intro 1", "Intro 2", "Intro 3", "Intro 4",
                                    "Verse 1", "Verse 2", "Verse 3", "Verse 4"])
        self.assertEqual(estimator.position_stability_metrics, {
            "cursor_changes": 7, "global_relocations": 0,
            "backward_jumps": 0, "large_forward_jumps": 0,
        })
        for entry in estimator.position_transition_log:
            self.assertTrue({"from_event", "to_event", "distance", "state",
                             "reason", "evidence"}.issubset(entry))

    def test_global_recovery_requires_explicit_lost_state(self):
        chart = ChartParser.parse("""[Intro]
G D Em C
[Verse]
Am F C G
[Chorus]
Dm Bb F C
""")
        clock = MusicalClock(bpm=120)
        estimator = PositionEstimator(ChartAlignment(chart), clock)
        clock.update(8.0)
        estimator.seek_to_bar(5)  # Verse: libera startup lock por navegação real.
        estimator._tracking_state = TrackingState.LOST
        for event_index, symbol in enumerate(("Dm", "Bb", "F", "C")):
            event = HarmonicRhythmEvent(
                symbol, event_index * 4.0, 4.0, 4.0, .95, "TEST")
            estimator._recent_harmonic_rhythm.append(
                ((event.symbol, event.start_beat), event))
        states = []
        for offset, chord in enumerate(("Dm", "Bb", "F", "C")):
            timestamp = 8.1 + offset * .5
            clock.update(timestamp)
            position = estimator.update(timestamp, chord, .95)
            states.append(estimator.tracking_state)
        self.assertEqual(position.section_name, "Chorus")
        self.assertIn(TrackingState.RECOVERING, states)
        self.assertEqual(states[-1], TrackingState.TRACKING)
        self.assertGreaterEqual(estimator.position_stability_metrics["global_relocations"], 1)
        self.assertTrue(all(item["reason"] != "note_sequence"
                            for item in estimator.position_transition_log))

    def test_lost_raw_chord_changes_do_not_authorize_global_search(self):
        chart = ChartParser.parse("""[Intro]
G D Em C
[Verse]
Am F C G
[Chorus]
Dm Bb F C
""")
        clock = MusicalClock(bpm=120)
        estimator = PositionEstimator(ChartAlignment(chart), clock)
        clock.update(8.0)
        estimator.seek_to_bar(5)
        estimator._tracking_state = TrackingState.LOST
        for offset, chord in enumerate(("Dm", "Bb", "F", "C")):
            timestamp = 8.1 + offset * .1
            clock.update(timestamp)
            position = estimator.update(timestamp, chord, .95)
        self.assertEqual(position.section_name, "Verse")
        self.assertEqual(estimator.position_stability_metrics["global_relocations"], 0)


if __name__ == "__main__":
    unittest.main()
