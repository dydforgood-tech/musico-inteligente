"""Testes para Edição Direta de Cifra e Letra no Músico Play-Along (v0.4)."""

import unittest
import tempfile
import os
import tkinter as tk

from app.song.song import Song
from app.song.song_session import SongSession
from app.project.project import Project
from app.project.project_manager import ProjectManager
from app.input.chart_parser import ChartParser
from app.ui.main_window import MainWindow


class TestChartEditing(unittest.TestCase):

    def setUp(self):
        self.sample_chart_1 = """[Intro]
C        G

[Verso 1]
Am       F
Eu estava pensando em você
"""
        self.sample_chart_2 = """[Intro]
D        A

[Verso 1]
Bm       G
Eu estava caminhando na rua
"""
        self.song = Song(
            title="Canção de Teste",
            artist="Banda Virtual",
            key="C Major",
            bpm=120.0,
            meter="4/4",
            chart_text=self.sample_chart_1
        )

    def test_song_session_update_chart_text(self):
        """Testa atualização de cifra diretamente no SongSession em execução."""
        session = SongSession(self.song)
        self.assertEqual(session.song.chart_text, self.sample_chart_1)
        self.assertEqual(session.current_chord, "C")

        # Atualiza para nova cifra com acordes em Ré Maior
        session.update_chart_text(self.sample_chart_2)

        self.assertEqual(session.song.chart_text, self.sample_chart_2)
        self.assertIsNotNone(session.song.chart_data)
        self.assertIn("D", [c.symbol.original_symbol for s in session.chart.sections for c in s.chords])
        self.assertEqual(session.current_chord, "D")

    def test_project_get_song_by_id(self):
        """Testa localização de música por ID em qualquer setlist do projeto."""
        proj = Project(name="Meu Projeto")
        active_set = proj.get_active_setlist()
        active_set.add_song(self.song)

        found = proj.get_song_by_id(self.song.id)
        self.assertIsNotNone(found)
        self.assertEqual(found.id, self.song.id)
        self.assertEqual(found.title, "Canção de Teste")

        # Busca ID inexistente
        self.assertIsNone(proj.get_song_by_id("id_inexistente_999"))

    def test_project_manager_update_song_chart(self):
        """Testa atualização de cifra através do ProjectManager e propagação à sessão ativa."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            proj_path = os.path.join(tmp_dir, "test_proj.json")
            pm = ProjectManager(project_file_path=proj_path)

            imported = pm.import_song(
                title="Música 1",
                artist="Artista 1",
                chart_text=self.sample_chart_1
            )
            session = pm.open_song(imported)
            self.assertEqual(session.current_chord, "C")

            # Atualiza o texto da cifra
            updated = pm.update_song_chart(imported.id, self.sample_chart_2)
            self.assertIsNotNone(updated)
            self.assertEqual(updated.chart_text, self.sample_chart_2)

            # Verifica propagação na sessão ativa
            self.assertEqual(pm.active_session.chart.title, "Música 1")
            self.assertEqual(pm.active_session.current_chord, "D")

            # Verifica persistência atômica no arquivo JSON
            self.assertTrue(os.path.exists(proj_path))
            pm2 = ProjectManager(project_file_path=proj_path)
            loaded_song = pm2.project.get_song_by_id(imported.id)
            self.assertIsNotNone(loaded_song)
            self.assertEqual(loaded_song.chart_text, self.sample_chart_2)

    def test_main_window_chart_view_is_editable(self):
        """Testa se o widget text_chart_view da MainWindow permanece com state='normal'."""
        tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(tmp_dir.cleanup)
        root = tk.Tk()
        root.withdraw()
        try:
            app = MainWindow(root, project_file_path=os.path.join(tmp_dir.name, "project.json"))
            # Verifica que o estado inicial é normal (editável)
            self.assertEqual(app.text_chart_view.cget("state"), "normal")

            # Simula digitação e verificação de flag de alterações não salvas
            app.text_chart_view.insert("end", "\n[Refrão]\nF    C\nNovo trecho")
            app._on_chart_key_release()
            self.assertTrue(app._has_unsaved_chart_edits)

            # Salva diretamente
            app._on_save_chart_direct()
            self.assertFalse(app._has_unsaved_chart_edits)

            # Verifica que a música ativa recebeu o texto atualizado
            active_song = app.project_manager.project.get_active_setlist().get_active_song()
            self.assertIn("Novo trecho", active_song.chart_text)
            self.assertEqual(app.text_chart_view.cget("state"), "normal")
        finally:
            root.destroy()


if __name__ == "__main__":
    unittest.main()
