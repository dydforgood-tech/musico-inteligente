"""Remoção de músicas da setlist, sessão ativa e persistência."""
import json
import tempfile
import unittest
from pathlib import Path

from app.project.project_manager import ProjectManager


class TestSetlistDelete(unittest.TestCase):
    def test_delete_selected_song_keeps_source_file_and_chooses_neighbor(self):
        with tempfile.TemporaryDirectory() as folder:
            project_path = Path(folder) / "project.json"
            audio_path = Path(folder) / "reference.mp3"
            audio_path.write_bytes(b"source remains untouched")
            manager = ProjectManager(str(project_path))
            songs = [manager.import_song(title=name, audio_path=str(audio_path),
                                         chart_text="[Intro]\nC")
                     for name in ("A", "B", "C")]
            manager.open_song(songs[1])
            removed = manager.remove_songs_from_active_setlist([songs[1].id])
            self.assertEqual([song.title for song in removed], ["B"])
            self.assertEqual([song.title for song in manager.project.get_active_setlist().songs],
                             ["A", "C"])
            self.assertEqual(manager.project.get_active_setlist().active_song_id, songs[2].id)
            self.assertIsNone(manager.active_session)
            self.assertTrue(audio_path.exists())
            saved = json.loads(project_path.read_text(encoding="utf-8"))
            self.assertEqual([song["title"] for song in saved["setlists"][0]["songs"]],
                             ["A", "C"])

    def test_delete_other_songs_preserves_active_session_then_allows_empty_list(self):
        with tempfile.TemporaryDirectory() as folder:
            manager = ProjectManager(str(Path(folder) / "project.json"))
            songs = [manager.import_song(title=name, chart_text="C")
                     for name in ("A", "B", "C")]
            session = manager.open_song(songs[1])
            manager.remove_songs_from_active_setlist([songs[0].id, songs[2].id])
            self.assertIs(manager.active_session, session)
            self.assertEqual([song.title for song in manager.project.get_active_setlist().songs],
                             ["B"])
            manager.remove_songs_from_active_setlist([songs[1].id])
            self.assertEqual(manager.project.get_active_setlist().songs, [])
            self.assertIsNone(manager.project.get_active_setlist().active_song_id)


if __name__ == "__main__":
    unittest.main()
