"""Seguimento por notas do instrumento de referência."""
import unittest

from app.music.musical_context import MusicalContext
from app.song.song import Song
from app.song.song_session import SongSession


class TestNoteFollowing(unittest.TestCase):
    def session(self, chart="C\nG\nAm\nF\nDm\nBb\nE7"):
        return SongSession(Song(title="Note following", chart_text=chart, bpm=120))

    def hear(self, session, note, timestamp, confidence=0.95):
        source = MusicalContext(note=note, note_confidence=confidence)
        session.update_audio_tick(timestamp, source_context=source)

    def test_three_notes_are_auxiliary_and_do_not_reanchor(self):
        session = self.session()
        self.hear(session, "D4", 0.1)
        self.hear(session, "F4", 0.3)
        self.assertEqual(session.current_bar, 1)
        self.hear(session, "A4", 0.5)
        self.assertEqual(session.clock_bar, 1)
        self.assertEqual(session.current_bar, 1)
        self.assertEqual(session.chart_position.current_bar, 1)
        self.assertEqual(session.context.bar, 1)
        self.assertEqual(session.prediction.source_bar, 1)
        self.assertEqual(session.position_estimator.last_note_evidence["action"], "CONFIDENCE_ONLY")
        session.update_audio_tick(2.0)
        self.assertEqual(session.current_bar, 2)

    def test_low_confidence_and_repeated_frames_do_not_count_as_three_notes(self):
        session = self.session()
        self.hear(session, "D4", 0.1)
        self.hear(session, "F4", 0.2, confidence=0.3)
        self.hear(session, "D4", 0.3)
        self.hear(session, "A4", 0.5)
        self.assertEqual(session.current_bar, 1)

    def test_repeated_section_is_ambiguous_and_does_not_jump(self):
        session = self.session("C\nG\nDm\nBb\nC\nG\nDm\nBb")
        for note, t in [("D4", 0.1), ("F4", 0.3), ("A4", 0.5)]:
            self.hear(session, note, t)
        self.assertEqual(session.current_bar, 1)
        self.assertEqual(session.position_estimator.bar_offset, 0)

    def test_reset_clears_old_note_sequence(self):
        session = self.session()
        self.hear(session, "D4", 0.1)
        self.hear(session, "F4", 0.3)
        session.reset()
        self.hear(session, "A4", 0.5)
        self.assertEqual(session.current_bar, 1)

    def test_notes_are_compared_with_sounding_chords_under_capo(self):
        session = self.session()
        session.position_estimator.set_capo(2)
        for note, t in [("E4", 0.1), ("G4", 0.3), ("B4", 0.5)]:
            self.hear(session, note, t)
        self.assertEqual(session.current_bar, 1)
        self.assertEqual(session.position_estimator.last_note_evidence["action"], "CONFIDENCE_ONLY")

    def test_old_notes_expire_before_new_phrase(self):
        session = self.session()
        self.hear(session, "D4", 0.1)
        self.hear(session, "F4", 0.3)
        self.hear(session, "A4", 4.0)
        self.assertEqual(session.current_bar, 3)


if __name__ == "__main__":
    unittest.main()
