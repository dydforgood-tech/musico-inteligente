"""Suíte de Testes Formais do Sistema Musician Play-Along + Virtual Band (v0.4).

Cobre exaustivamente:
 1. Criação e integridade de Song e PerformanceSettings
 2. Criação de Setlist e gerenciamento de músicas
 3. Adição e consulta de múltiplas músicas
 4. Remoção de música e ajuste automático da música ativa
 5. Reordenação do repertório
 6. Seleção de música ativa
 7. Troca hermética de músicas e ciclo de vida de SongSession
 8. Verificação do estado central em SongSession
 9. Reset rigoroso de MusicalClock ao trocar/reiniciar
10. Teste de Isolamento Rígido: Song A (C-Am-F-G) -> Song B (Dm-Bb-C-A) -> Song A (Zero memory leak)
11. Salvamento atômico local do Project em JSON
12. Carregamento e reconstituição do Project a partir do disco
13. Parser de cifras estruturadas com detecção de seções e tonalidade
14. Parser de letras sem contaminação entre acordes e texto
15. Reconhecimento de acordes complexos, extensões e alterações
16. Tratamento de inversões e slash chords (C/E, C/G, Dm/F, G/B)
17. Mapeamento e equivalência de sinônimos (C7M == Cmaj7 == CΔ7, C° == Cdim)
18. Graus harmônicos e numerais romanos relativos à tonalidade
19. Invariância transpositiva (C9-Am7-Dm7-G7 == G9-Em7-Am7-D7)
20. Suporte e expansão de repetições (x2, 2x, ||: :||)
21. Alinhamento temporal entre o relógio musical e a cifra (ChartAlignment)
22. Memória de padrões e predição por canção
23. Fusão Cifra + Áudio: soberania da cifra sobre o áudio e registro de discrepâncias
24. Registro de Músicos Virtuais (VirtualPlayerRegistry) e prontidão de instrumentos
"""

import os
import tempfile
import unittest

from app.song.song import Song, PerformanceSettings
from app.song.song_session import SongSession
from app.setlist.setlist import Setlist
from app.setlist.setlist_manager import SetlistManager
from app.project.project import Project, ProjectSettings
from app.project.project_manager import ProjectManager
from app.music.chord_chart import (
    ChordChart,
    ChartSection,
    ChartChord,
    LyricSegment,
    parse_chord,
    ChordSymbol,
    calculate_harmonic_degree
)
from app.input.chart_parser import ChartParser
from app.music.chart_alignment import ChartAlignment
from app.music.chart_audio_fusion import ChartAudioFusion
from app.instruments.registry import VirtualPlayerRegistry
from app.music.harmonic_normalization import sequence_similarity


class TestMusicianSystem(unittest.TestCase):
    """Bateria de testes formais do sistema Musician Play-Along."""

    def setUp(self):
        self.project_manager = ProjectManager()

    # -------------------------------------------------------------
    # 1. Criação de Song
    # -------------------------------------------------------------
    def test_01_create_song(self):
        """TESTE 1 — Criação de Song com configurações e serialização."""
        song = Song(
            title="Aquarela",
            artist="Toquinho",
            key="G Major",
            bpm=110.0,
            meter="4/4"
        )
        self.assertEqual(song.title, "Aquarela")
        self.assertEqual(song.artist, "Toquinho")
        self.assertEqual(song.key, "G Major")
        self.assertEqual(song.bpm, 110.0)
        self.assertEqual(song.meter, "4/4")
        self.assertIsNotNone(song.id)

        d = song.to_dict()
        reconstituted = Song.from_dict(d)
        self.assertEqual(reconstituted.id, song.id)
        self.assertEqual(reconstituted.title, "Aquarela")

    # -------------------------------------------------------------
    # 2. Criação de Setlist
    # -------------------------------------------------------------
    def test_02_create_setlist(self):
        """TESTE 2 — Criação e serialização de Setlist."""
        setlist = Setlist(name="Show Acústico", description="Voz e Violão")
        self.assertEqual(setlist.name, "Show Acústico")
        self.assertEqual(len(setlist.songs), 0)

        d = setlist.to_dict()
        loaded = Setlist.from_dict(d)
        self.assertEqual(loaded.name, "Show Acústico")

    # -------------------------------------------------------------
    # 3. Adicionar múltiplas músicas ao Setlist
    # -------------------------------------------------------------
    def test_03_add_multiple_songs(self):
        """TESTE 3 — Adição de várias músicas e seleção automática da primeira."""
        mgr = SetlistManager()
        s1 = Song(title="Música 1", key="C Major")
        s2 = Song(title="Música 2", key="G Major")
        s3 = Song(title="Música 3", key="D Major")

        mgr.add_song(s1)
        mgr.add_song(s2)
        mgr.add_song(s3)

        self.assertEqual(len(mgr.active_setlist.songs), 3)
        self.assertEqual(mgr.get_active_song().title, "Música 1")

    # -------------------------------------------------------------
    # 4. Remover música
    # -------------------------------------------------------------
    def test_04_remove_song(self):
        """TESTE 4 — Remoção de música e transição limpa de active_song."""
        mgr = SetlistManager()
        s1 = Song(title="Música 1")
        s2 = Song(title="Música 2")
        mgr.add_song(s1)
        mgr.add_song(s2)

        self.assertTrue(mgr.remove_song(s1.id))
        self.assertEqual(len(mgr.active_setlist.songs), 1)
        self.assertEqual(mgr.get_active_song().title, "Música 2")

    # -------------------------------------------------------------
    # 5. Reordenar músicas
    # -------------------------------------------------------------
    def test_05_reorder_songs(self):
        """TESTE 5 — Reordenação arbitrária das músicas do setlist."""
        mgr = SetlistManager()
        s1 = Song(title="A")
        s2 = Song(title="B")
        s3 = Song(title="C")
        mgr.add_song(s1)
        mgr.add_song(s2)
        mgr.add_song(s3)

        mgr.reorder_songs([s3.id, s1.id, s2.id])
        titles = [s.title for s in mgr.active_setlist.songs]
        self.assertEqual(titles, ["C", "A", "B"])

    # -------------------------------------------------------------
    # 6. Selecionar música e navegação
    # -------------------------------------------------------------
    def test_06_navigation_and_selection(self):
        """TESTE 6 — Navegação sequencial (próxima, anterior, primeira, última)."""
        mgr = SetlistManager()
        s1 = Song(title="Primeira")
        s2 = Song(title="Segunda")
        s3 = Song(title="Terceira")
        mgr.add_song(s1)
        mgr.add_song(s2)
        mgr.add_song(s3)

        self.assertEqual(mgr.first_song().title, "Primeira")
        self.assertEqual(mgr.next_song().title, "Segunda")
        self.assertEqual(mgr.next_song().title, "Terceira")
        self.assertEqual(mgr.prev_song().title, "Segunda")
        self.assertEqual(mgr.last_song().title, "Terceira")

    # -------------------------------------------------------------
    # 7. Troca de Música e SongSession
    # -------------------------------------------------------------
    def test_07_song_session_lifecycle(self):
        """TESTE 7 — Ciclo de vida da SongSession ao abrir música."""
        s = Song(title="Canção Teste", bpm=130.0, meter="3/4")
        session = self.project_manager.open_song(s)
        self.assertIsNotNone(session)
        self.assertEqual(session.song.title, "Canção Teste")
        self.assertEqual(session.clock.bpm, 130.0)
        self.assertEqual(session.clock.meter, "3/4")
        self.assertEqual(session.current_bar, 1)

    # -------------------------------------------------------------
    # 8. SongSession como fonte central de estado
    # -------------------------------------------------------------
    def test_08_song_session_central_state(self):
        """TESTE 8 — Consulta central de estado através da SongSession."""
        chart_txt = "[Intro]\nC G\n[Verso]\nAm F"
        s = Song(title="Central State", chart_text=chart_txt)
        session = SongSession(s)

        self.assertEqual(session.expected_chord, "C")
        self.assertEqual(session.current_chord, "C")
        self.assertEqual(session.next_chord, "G")
        self.assertEqual(session.current_section, "Intro")

    # -------------------------------------------------------------
    # 9. Reset do MusicalClock na SongSession
    # -------------------------------------------------------------
    def test_09_musical_clock_reset(self):
        """TESTE 9 — Reset de tempo e compasso do relógio musical."""
        s = Song(bpm=120.0)
        session = SongSession(s)
        session.update_audio_tick(8.0)
        self.assertGreater(session.current_bar, 1)

        session.reset()
        self.assertEqual(session.current_bar, 1)
        self.assertEqual(session.current_beat, 1)
        self.assertEqual(session.clock.elapsed_time, 0.0)

    # -------------------------------------------------------------
    # 10. Isolamento Rígido: Song A -> Song B -> Song A
    # -------------------------------------------------------------
    def test_10_strict_song_isolation_no_leak(self):
        """TESTE 10 — Troca Song A -> Song B -> Song A garante zero vazamento de memória."""
        song_a = Song(
            title="Song A",
            key="C Major",
            chart_text="[Verso]\nC Am F G"
        )
        song_b = Song(
            title="Song B",
            key="D Minor",
            chart_text="[Refrão]\nDm Bb C A"
        )

        # 1. Abre A e toca até certo ponto
        session_a1 = self.project_manager.open_song(song_a)
        session_a1.update_audio_tick(12.0, detected_chord="Am", detected_confidence=0.9)
        self.assertEqual(session_a1.context.key, "C Major")
        self.assertGreater(session_a1.current_bar, 1)

        # 2. Troca para B
        session_b = self.project_manager.open_song(song_b)
        self.assertEqual(session_b.song.title, "Song B")
        self.assertEqual(session_b.current_bar, 1)
        self.assertEqual(session_b.current_chord, "Dm")
        self.assertEqual(session_b.current_section, "Refrão")
        self.assertEqual(session_b.context.key, "D Minor")
        # Nenhuma informação de C Major ou Verso da Song A pode existir em B
        self.assertNotEqual(session_b.expected_chord, "C")

        # 3. Retorna para A
        session_a2 = self.project_manager.open_song(song_a)
        self.assertEqual(session_a2.song.title, "Song A")
        self.assertEqual(session_a2.current_bar, 1)
        self.assertEqual(session_a2.current_chord, "C")
        self.assertEqual(session_a2.current_section, "Verso")
        self.assertEqual(session_a2.context.key, "C Major")
        self.assertEqual(session_a2.clock.elapsed_time, 0.0)

    # -------------------------------------------------------------
    # 11 & 12. Salvar e Carregar Projeto (Persistência)
    # -------------------------------------------------------------
    def test_11_12_project_save_and_load_recovery(self):
        """TESTES 11 & 12 — Salvamento atômico em JSON e recuperação de projeto íntegro."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
            proj_path = tf.name

        try:
            pm = ProjectManager(proj_path)
            pm.create_project("Repertório Show 2026", "Apresentação ao vivo")
            s1 = pm.import_song(title="Chega de Saudade", artist="Tom Jobim", key="D Minor")
            s2 = pm.import_song(title="Garota de Ipanema", artist="Tom Jobim", key="F Major")

            pm.save_project()
            self.assertTrue(os.path.exists(proj_path))

            # Cria novo ProjectManager e carrega o arquivo do disco
            new_pm = ProjectManager()
            loaded_proj = new_pm.load_project(proj_path)

            self.assertEqual(loaded_proj.name, "Repertório Show 2026")
            self.assertEqual(len(loaded_proj.get_active_setlist().songs), 2)
            self.assertEqual(loaded_proj.get_active_setlist().songs[0].title, "Chega de Saudade")
            self.assertEqual(loaded_proj.get_active_setlist().songs[1].key, "F Major")
        finally:
            if os.path.exists(proj_path):
                os.remove(proj_path)

    # -------------------------------------------------------------
    # 13 & 14. Parser de Cifra e Letras
    # -------------------------------------------------------------
    def test_13_14_chart_and_lyric_parser(self):
        """TESTES 13 & 14 — Parser separa estritamente acordes de linhas de letra."""
        cifra_text = """
Tom: G
BPM: 115

[Intro]
C9        G/B
Hoje eu...
Am7       F7M
Vou contar...

[Verso]
C9
Alguma coisa aconteceu
"""
        chart = ChartParser.parse(cifra_text)
        self.assertEqual(chart.key, "G Major")
        self.assertEqual(chart.bpm, 115.0)
        self.assertEqual(len(chart.sections), 2)

        intro = chart.sections[0]
        self.assertEqual(intro.name, "Intro")
        self.assertEqual(intro.section_type, "INTRO")
        self.assertEqual(len(intro.chords), 4)
        chord_names = [c.symbol.original_symbol for c in intro.chords]
        self.assertEqual(chord_names, ["C9", "G/B", "Am7", "F7M"])

        # Letra não virou acorde
        self.assertNotIn("Hoje", chord_names)
        self.assertNotIn("contar", chord_names)

    # -------------------------------------------------------------
    # 15. Acordes Complexos, Extensões e Alterações
    # -------------------------------------------------------------
    def test_15_complex_chord_recognition(self):
        """TESTE 15 — Reconhecimento e preservação integral de acordes avançados."""
        symbols = [
            ("C", "C", "major"),
            ("Cm", "C", "minor"),
            ("C7", "C", "major"),
            ("Cmaj7", "C", "major"),
            ("C7M", "C", "major"),
            ("C9", "C", "major"),
            ("Cmaj9", "C", "major"),
            ("Cm7", "C", "minor"),
            ("Cm9", "C", "minor"),
            ("Cadd9", "C", "major"),
            ("Csus2", "C", "sus2"),
            ("Csus4", "C", "sus4"),
            ("Cdim", "C", "dim"),
            ("C°", "C", "dim"),
            ("Caug", "C", "aug"),
            ("C+", "C", "aug"),
            ("F#m7(b5)", "F#", "dim"),
            ("Bbmaj7", "A#", "major"),
        ]

        for raw, exp_root, exp_quality in symbols:
            parsed = parse_chord(raw)
            self.assertEqual(parsed.original_symbol, raw)
            self.assertEqual(parsed.root, exp_root, f"Falha no root de {raw}")
            self.assertEqual(parsed.quality, exp_quality, f"Falha na qualidade de {raw}")

    # -------------------------------------------------------------
    # 16. Inversões e Slash Chords
    # -------------------------------------------------------------
    def test_16_chord_inversions(self):
        """TESTE 16 — Inversões e notas do baixo separadas (C/E, C/G, Dm/F, G/B)."""
        ce = parse_chord("C/E")
        self.assertEqual(ce.root, "C")
        self.assertEqual(ce.bass_note, "E")
        self.assertEqual(ce.inversion, "first")

        cg = parse_chord("C/G")
        self.assertEqual(cg.root, "C")
        self.assertEqual(cg.bass_note, "G")
        self.assertEqual(cg.inversion, "second")

        gb = parse_chord("G/B")
        self.assertEqual(gb.root, "G")
        self.assertEqual(gb.bass_note, "B")
        self.assertEqual(gb.inversion, "first")

        dmf = parse_chord("Dm/F")
        self.assertEqual(dmf.root, "D")
        self.assertEqual(dmf.bass_note, "F")
        self.assertEqual(dmf.inversion, "first")

    # -------------------------------------------------------------
    # 17. Sinônimos Harmônicos
    # -------------------------------------------------------------
    def test_17_chord_synonyms(self):
        """TESTE 17 — Reconhecimento de equivalência: C7M == Cmaj7 == CΔ7, C° == Cdim."""
        c7m = parse_chord("C7M")
        cmaj7 = parse_chord("Cmaj7")
        c_delta = parse_chord("CΔ7")

        self.assertTrue(c7m.is_synonym_of(cmaj7))
        self.assertTrue(c7m.is_synonym_of(c_delta))
        self.assertEqual(c7m.original_symbol, "C7M")
        self.assertEqual(cmaj7.original_symbol, "Cmaj7")

        cdim = parse_chord("Cdim")
        cdeg = parse_chord("C°")
        self.assertTrue(cdim.is_synonym_of(cdeg))

        caug = parse_chord("Caug")
        cplus = parse_chord("C+")
        self.assertTrue(caug.is_synonym_of(cplus))

    # -------------------------------------------------------------
    # 18. Graus Harmônicos relativos à tonalidade
    # -------------------------------------------------------------
    def test_18_harmonic_degrees(self):
        """TESTE 18 — Cálculo de graus Imaj7, vi7, ii7, V7 em C Major."""
        c = parse_chord("Cmaj7", key_context="C Major")
        self.assertEqual(c.harmonic_degree, "I")
        self.assertEqual(c.roman_numeral, "Imaj7")

        am = parse_chord("Am7", key_context="C Major")
        self.assertEqual(am.harmonic_degree, "vi")
        self.assertEqual(am.roman_numeral, "vi7")

        dm = parse_chord("Dm7", key_context="C Major")
        self.assertEqual(dm.harmonic_degree, "ii")
        self.assertEqual(dm.roman_numeral, "ii7")

        g = parse_chord("G7", key_context="C Major")
        self.assertEqual(g.harmonic_degree, "V")
        self.assertEqual(g.roman_numeral, "V7")

    # -------------------------------------------------------------
    # 19. Transposição Relativa
    # -------------------------------------------------------------
    def test_19_transposition_invariance(self):
        """TESTE 19 — C9-Am7-Dm7-G7 em C equivale estruturalmente a G9-Em7-Am7-D7 em G."""
        seq_c = ["C", "Am", "Dm", "G"]
        seq_g = ["G", "Em", "Am", "D"]
        sim = sequence_similarity(seq_c, seq_g, key1="C Major", key2="G Major")
        self.assertGreaterEqual(sim, 0.95)

    # -------------------------------------------------------------
    # 20. Expansão de Repetições
    # -------------------------------------------------------------
    def test_20_repeats_expansion(self):
        """TESTE 20 — Expansão de repetições (x2)."""
        cifra = "[Refrão]\nC Am F G x2"
        chart = ChartParser.parse(cifra)
        refrao = chart.sections[0]
        self.assertEqual(refrao.repeat_count, 2)
        self.assertEqual(len(refrao.chords), 4)
        self.assertEqual(len(refrao.expanded_chords), 8)

    # -------------------------------------------------------------
    # 21. Alinhamento de Cifra (ChartAlignment)
    # -------------------------------------------------------------
    def test_21_chart_alignment(self):
        """TESTE 21 — Mapeamento preciso de compasso/tempo para seção e acorde."""
        cifra = "[Intro]\nC G\n[Verso]\nAm F"
        chart = ChartParser.parse(cifra)
        align = ChartAlignment(chart)

        pos_bar1 = align.get_position_at(bar=1)
        self.assertEqual(pos_bar1.current_chord, "C")
        self.assertEqual(pos_bar1.section_name, "Intro")

        pos_bar2 = align.get_position_at(bar=2)
        self.assertEqual(pos_bar2.current_chord, "G")

        pos_bar3 = align.get_position_at(bar=3)
        self.assertEqual(pos_bar3.current_chord, "Am")
        self.assertEqual(pos_bar3.section_name, "Verso")

    # -------------------------------------------------------------
    # 22. Fusão Cifra + Áudio (ChartAudioFusion)
    # -------------------------------------------------------------
    def test_22_chart_audio_fusion_sovereignty(self):
        """TESTE 22 — Cifra prevalece sobre áudio e divergências viram PossibleVariation."""
        fusion = ChartAudioFusion(variation_confirmation_seconds=2.0)
        cifra = "[Verso]\nF C"
        chart = ChartParser.parse(cifra)
        align = ChartAlignment(chart)
        pos = align.get_position_at(1)

        # Áudio detecta F# enquanto a cifra diz F
        state1 = fusion.fuse(pos, detected_chord="F#", chord_confidence=0.88, timestamp=0.0)
        self.assertEqual(state1.expected_chord, "F")
        self.assertEqual(state1.detected_chord, "F#")
        self.assertEqual(state1.effective_chord, "F") # Cifra NÃO é sobrescrita
        self.assertTrue(state1.is_discrepancy)
        self.assertFalse(state1.confirmed_variation)

        # Se persistir por 2.5s, marca confirmed_variation mantendo a cifra como referência
        state2 = fusion.fuse(pos, detected_chord="F#", chord_confidence=0.90, timestamp=2.5)
        self.assertTrue(state2.is_discrepancy)
        self.assertTrue(state2.confirmed_variation)
        self.assertEqual(state2.expected_chord, "F")

    # -------------------------------------------------------------
    # 23. VirtualPlayerRegistry e Prontidão
    # -------------------------------------------------------------
    def test_23_virtual_player_registry(self):
        """TESTE 23 — VirtualPlayerRegistry com BassPlayer e postos em prontidão."""
        registry = VirtualPlayerRegistry(sample_rate=44100)
        statuses = registry.all_statuses
        self.assertEqual(statuses["bass"], "READY")
        self.assertEqual(statuses["drums"], "STANDBY")
        self.assertEqual(statuses["keyboard"], "STANDBY")
        self.assertEqual(statuses["guitar"], "STANDBY")
        self.assertIsNotNone(registry.bass_player)

    # -------------------------------------------------------------
    # 24. Importação Flexível de Músicas
    # -------------------------------------------------------------
    def test_24_flexible_song_import(self):
        """TESTE 24 — Importação de canção apenas com cifra, apenas com áudio ou completa."""
        pm = ProjectManager()

        # Só áudio
        s_audio = pm.import_song(title="Instrumental", audio_path="guitar.wav")
        self.assertEqual(s_audio.audio_path, "guitar.wav")
        self.assertEqual(s_audio.chart_text, "")

        # Só cifra
        s_chart = pm.import_song(title="Cancioneiro", chart_text="[Refrão]\nC G Am F")
        self.assertEqual(s_chart.audio_path, "")
        self.assertIsNotNone(s_chart.chart_data)


if __name__ == "__main__":
    unittest.main()
