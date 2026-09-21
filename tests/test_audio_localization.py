"""Testes da LOCALIZAÇÃO POR ÁUDIO (v0.7).

O programa deve procurar, pelo áudio, ONDE o músico está na cifra e encaixar ali —
em vez de apenas percorrer a cifra em ordem pelo relógio.
"""

import unittest

from app.input.chart_parser import ChartParser
from app.music.chart_alignment import ChartAlignment
from app.music.musical_clock import MusicalClock
from app.music.position_estimator import PositionEstimator, TrackingState


class TestAudioLocalization(unittest.TestCase):
    def setUp(self):
        # Intro (C G) | Verso (Am F C G) | Refrão (F C G Am)
        self.chart = ChartParser.parse("[Intro]\nC  G\n[Verso]\nAm  F  C  G\n[Refrao]\nF  C  G  Am\n")
        self.alignment = ChartAlignment(self.chart)

    def _feed(self, est, clock, chords, t0=0.5, step=0.5):
        pos = None
        for i, ch in enumerate(chords):
            t = t0 + i * step
            clock.update(t)
            pos = est.update(timestamp=t, detected_chord=ch, detected_confidence=0.9)
        return pos

    def test_startup_lock_does_not_jump_to_refrao(self):
        """No início, uma progressão parecida não possui autoridade para pular a Intro."""
        clock = MusicalClock(bpm=120.0, meter="4/4")
        est = PositionEstimator(self.alignment, clock)
        pos = self._feed(est, clock, ["F", "C", "G", "Am"])
        self.assertNotEqual(pos.section_name, "Refrao")
        self.assertEqual(est.position_stability_metrics["global_relocations"], 0)

    def test_single_chord_does_not_jump(self):
        """Um único acorde divergente NÃO faz o sistema saltar pela música."""
        clock = MusicalClock(bpm=120.0, meter="4/4")
        est = PositionEstimator(self.alignment, clock)
        clock.update(0.5)
        pos = est.update(timestamp=0.5, detected_chord="Am", detected_confidence=0.9)
        # Com um só acorde não há evidência de progressão para localizar longe
        self.assertEqual(pos.current_bar, 1)

    def test_localize_api_returns_match(self):
        clock = MusicalClock(bpm=120.0, meter="4/4")
        est = PositionEstimator(self.alignment, clock)
        # alimenta a progressão do refrão no buffer
        self._feed(est, clock, ["F", "C", "G", "Am"])
        loc = est.localize(near_bar=1)
        self.assertIsNotNone(loc)
        bar, score, length = loc
        self.assertGreaterEqual(score, 0.75)
        self.assertGreaterEqual(length, 2)

    def test_tonic_sequence_does_not_bypass_startup_lock(self):
        """Tônicas compatíveis não podem pular a primeira seção durante startup."""
        clock = MusicalClock(bpm=120.0, meter="4/4")
        est = PositionEstimator(self.alignment, clock)
        pos = self._feed(est, clock, ["Fm", "C7", "G", "Am7"])
        self.assertNotEqual(pos.section_name, "Refrao")
        self.assertEqual(est.position_stability_metrics["global_relocations"], 0)

    def test_root_only_confirms_position(self):
        """Um acorde com tônica certa e qualidade errada confirma (não gera divergência)."""
        clock = MusicalClock(bpm=120.0, meter="4/4")
        est = PositionEstimator(self.alignment, clock)
        clock.update(0.5)
        # Compasso 1 espera C; detecta Cm7 (mesma tônica) -> deve confirmar
        est.update(timestamp=0.5, detected_chord="Cm7", detected_confidence=0.9)
        self.assertEqual(est.tracking_state, TrackingState.TRACKING)

    def test_follow_audio_can_be_disabled(self):
        """Com follow_audio desligado, a posição segue só o relógio (comportamento antigo)."""
        clock = MusicalClock(bpm=120.0, meter="4/4")
        est = PositionEstimator(self.alignment, clock)
        est._follow_audio = False
        pos = self._feed(est, clock, ["F", "C", "G", "Am"])
        # Sem localização por áudio, permanece próximo do início (relógio mal avançou)
        self.assertLessEqual(pos.current_bar, 3)


if __name__ == "__main__":
    unittest.main()
