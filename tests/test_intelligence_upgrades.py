"""Testes das melhorias de inteligência (v0.5):
1. Vocabulário estendido do detector de acordes (7, 7M, m7, sus4, sus2).
2. Detecção automática de transposição/capotraste.
3. Reancoramento de posição por progressão de acordes.
"""

import unittest
import numpy as np

from app.analysis.chord_detector import ChordDetector
from app.analysis.chroma_extractor import ChromaExtractor
from app.utils.audio_generator import generate_chord
from app.music.transposition_tracker import TranspositionTracker
from app.music.chord_chart import ChordChart, parse_chord
from app.music.chart_alignment import ChartAlignment
from app.music.musical_clock import MusicalClock
from app.music.position_estimator import PositionEstimator, TrackingState


class TestExtendedChordVocabulary(unittest.TestCase):
    """O ouvido, quando habilitado, confirma tétrades e acordes suspensos."""

    @classmethod
    def setUpClass(cls):
        cls.sr = 44100
        cls.chroma = ChromaExtractor()

    def _symbol(self, freqs, detect_extensions=True):
        detector = ChordDetector(detect_extensions=detect_extensions)
        audio = generate_chord(freqs, duration=0.5, sr=self.sr)
        chr_vec = self.chroma.extract(audio, self.sr)
        return detector.detect(chr_vec, audio, self.sr, timestamp=1.0).symbol

    def test_dominant_seventh_detected(self):
        # C E G Bb -> C7 (a 7ª menor é evidência limpa de dominante)
        sym = self._symbol([261.63, 329.63, 392.00, 466.16])
        self.assertTrue(sym.startswith("C7"), f"Esperado C7, obtido {sym}")

    def test_plain_major_not_upgraded_to_seventh(self):
        # C E G puro NÃO deve virar C7 (sem 7ª menor real)
        sym = self._symbol([261.63, 329.63, 392.00])
        self.assertEqual(sym, "C")

    def test_sus4_detected(self):
        # C F G (sem 3ª) -> Csus4
        sym = self._symbol([261.63, 349.23, 392.00])
        self.assertTrue(sym.startswith("Csus4"), f"Esperado Csus4, obtido {sym}")

    def test_plain_triad_when_extensions_disabled(self):
        # Mesmo C7 vira apenas 'C' quando o refinamento está desligado (comportamento padrão)
        sym = self._symbol([261.63, 329.63, 392.00, 466.16], detect_extensions=False)
        self.assertEqual(sym, "C")


class TestHarmonicChroma(unittest.TestCase):
    """Cromagrama por soma harmônica suprime vazamento e destrava tétrades (maj7/m7)."""

    @classmethod
    def setUpClass(cls):
        cls.sr = 44100

    def _pc(self, note):
        from app.music.theory import PITCH_CLASSES
        return PITCH_CLASSES.index(note)

    def test_phantom_overtone_is_suppressed(self):
        # Dó maior puro (C-E-G): o Si-fantasma (3º harmônico do Mi) deve quase sumir.
        fft = ChromaExtractor(method="fft")
        harm = ChromaExtractor(method="harmonic")
        audio = generate_chord([261.63, 329.63, 392.00], duration=0.5, sr=self.sr)
        b_fft = float(fft.extract(audio, self.sr)[self._pc("B")])
        b_harm = float(harm.extract(audio, self.sr)[self._pc("B")])
        self.assertLess(b_harm, 0.25, f"Si-fantasma não suprimido: {b_harm}")
        self.assertLess(b_harm, b_fft, "Cromagrama harmônico deveria reduzir o vazamento")

    def _symbol_harmonic(self, freqs):
        harm = ChromaExtractor(method="harmonic")
        det = ChordDetector(detect_extensions=True)
        audio = generate_chord(freqs, duration=0.5, sr=self.sr)
        return det.detect(harm.extract(audio, self.sr), audio, self.sr, timestamp=1.0)

    def test_major_seventh_unlocked_with_harmonic_chroma(self):
        # C E G B -> C7M (impossível no cromagrama FFT, viável no harmônico)
        ch = self._symbol_harmonic([261.63, 329.63, 392.00, 493.88])
        self.assertEqual(ch.root, "C")
        self.assertTrue(ch.symbol.startswith("C7M"), f"Esperado C7M, obtido {ch.symbol}")

    def test_minor_seventh_unlocked_with_harmonic_chroma(self):
        # A C E G -> Am7
        ch = self._symbol_harmonic([220.0, 261.63, 329.63, 392.00])
        self.assertEqual(ch.root, "A")
        self.assertTrue(ch.symbol.startswith("Am7"), f"Esperado Am7, obtido {ch.symbol}")

    def test_basic_triads_still_correct(self):
        for freqs, root in ([261.63, 329.63, 392.00], "C"), ([196.0, 246.94, 293.66], "G"):
            ch = self._symbol_harmonic(freqs)
            self.assertEqual(ch.root, root)

    def test_audio_analyzer_high_resolution_mode(self):
        # AudioAnalyzer no modo de alta resolução reconhece a tétrade de ponta a ponta
        from app.analysis.audio_analyzer import AudioAnalyzer
        analyzer = AudioAnalyzer(sample_rate=self.sr)
        analyzer.set_harmonic_resolution(True)
        audio = generate_chord([261.63, 329.63, 392.00, 493.88], duration=0.5, sr=self.sr)  # Cmaj7
        ctx = analyzer.analyze_chunk(audio, self.sr, timestamp=0.5)
        self.assertTrue(ctx.chord.startswith("C7M"), f"Esperado C7M no modo HR, obtido {ctx.chord}")


class TestChartPrior(unittest.TestCase):
    """A cifra atua como prior musical, desempatando acordes ambíguos."""

    def setUp(self):
        self.detector = ChordDetector()
        self.sr = 44100
        self.dummy = np.zeros(1024, dtype=np.float32)

    def _ambiguous_chroma(self):
        # C-E-G-A: ambíguo entre C major (C-E-G) e A minor (A-C-E)
        from app.music.theory import PITCH_CLASSES
        c = np.zeros(12, dtype=np.float32)
        for note, val in {"C": 1.0, "E": 0.9, "G": 0.6, "A": 0.6}.items():
            c[PITCH_CLASSES.index(note)] = val
        return c

    def test_prior_breaks_ambiguity_toward_chart(self):
        # Sem prior: escolhe algum dos dois; com prior "C" deve escolher C; com "Am" deve escolher Am.
        det_c = ChordDetector()
        chord_c = det_c.detect(self._ambiguous_chroma(), self.dummy, self.sr,
                               timestamp=1.0, expected_chord="C")
        self.assertEqual(chord_c.root, "C")

        det_a = ChordDetector()
        chord_a = det_a.detect(self._ambiguous_chroma(), self.dummy, self.sr,
                               timestamp=1.0, expected_chord="Am")
        self.assertEqual(chord_a.root, "A")
        self.assertEqual(chord_a.quality, "minor")

    def test_prior_does_not_override_strong_audio(self):
        # Áudio claramente G maior; prior 'C' NÃO deve sequestrar a detecção.
        from app.music.theory import PITCH_CLASSES
        c = np.zeros(12, dtype=np.float32)
        for note, val in {"G": 1.0, "B": 0.9, "D": 0.9}.items():
            c[PITCH_CLASSES.index(note)] = val
        chord = self.detector.detect(c, self.dummy, self.sr, timestamp=1.0, expected_chord="C")
        self.assertEqual(chord.root, "G")

    def test_confidence_is_raw_not_inflated_by_prior(self):
        chroma = self._ambiguous_chroma()
        no_prior = ChordDetector().detect(chroma, self.dummy, self.sr, timestamp=1.0)
        with_prior = ChordDetector().detect(chroma, self.dummy, self.sr,
                                            timestamp=1.0, expected_chord=no_prior.symbol)
        # A confiança do vencedor coincidente não é inflada pelo bônus do prior
        self.assertAlmostEqual(no_prior.confidence, with_prior.confidence, places=5)


class TestAnalyzerPriorWiring(unittest.TestCase):
    """A expectativa empurrada é usada pelo AudioAnalyzer na detecção."""

    def test_pushed_expectation_influences_detection(self):
        from app.analysis.audio_analyzer import AudioAnalyzer
        from app.analysis.chroma_extractor import ChromaExtractor
        from app.utils.audio_generator import generate_chord

        sr = 44100
        # Áudio ambíguo entre C major e A minor (C-E-G-A)
        audio = generate_chord([261.63, 329.63, 392.00, 440.00], duration=0.5, sr=sr)

        a1 = AudioAnalyzer(sample_rate=sr)
        a1.set_harmonic_expectation("Am", "A Minor")
        ctx1 = a1.analyze_chunk(audio, sr, timestamp=0.5)

        a2 = AudioAnalyzer(sample_rate=sr)
        a2.set_harmonic_expectation("C", "C Major")
        ctx2 = a2.analyze_chunk(audio, sr, timestamp=0.5)

        # A expectativa da cifra deve inclinar a leitura ambígua para lados diferentes
        self.assertTrue(ctx1.chord.startswith("A"), f"Esperava tônica A com prior Am, obtido {ctx1.chord}")
        self.assertTrue(ctx2.chord.startswith("C"), f"Esperava tônica C com prior C, obtido {ctx2.chord}")


class TestTranspositionTracker(unittest.TestCase):
    """Detecção de deslocamento global consistente (capotraste / tom diferente)."""

    def test_consistent_offset_is_detected(self):
        tracker = TranspositionTracker()
        # Cifra: C, G, Am ; Áudio: D, A, Bm  (+2 semitons)
        self.assertEqual(tracker.observe("C", "D"), 0)   # ainda sem observações suficientes
        self.assertEqual(tracker.observe("G", "A"), 0)
        offset = tracker.observe("Am", "Bm")             # 3ª observação consistente
        self.assertEqual(offset, 2)
        self.assertTrue(tracker.is_transposed)
        self.assertTrue(tracker.matches_under_transposition("F", "G"))

    def test_random_errors_do_not_trigger_transposition(self):
        tracker = TranspositionTracker()
        tracker.observe("C", "D")   # +2
        tracker.observe("G", "C")   # +5
        tracker.observe("Am", "F")  # +8
        self.assertEqual(tracker.semitones, 0)
        self.assertFalse(tracker.is_transposed)


class TestProgressionReanchor(unittest.TestCase):
    """O estimador se reancora quando a SEQUÊNCIA recente casa com a cifra."""

    def _build_estimator(self):
        chart_text = (
            "[Intro]\n"
            "C  G  Am  F\n"
            "C  G  Am  F\n"
        )
        chart = ChordChart.from_text(chart_text) if hasattr(ChordChart, "from_text") else None
        return chart

    def test_progression_match_finds_sequence(self):
        from app.input.chart_parser import ChartParser
        chart = ChartParser.parse("[Intro]\nC  G  Am  F\nC  G  Am  F\n")
        alignment = ChartAlignment(chart)
        clock = MusicalClock(bpm=120.0, meter="4/4")
        estimator = PositionEstimator(alignment, clock)

        match = estimator.find_progression_match(["G", "Am", "F"], start_bar_hint=1, window_bars=8)
        self.assertIsNotNone(match)
        self.assertGreaterEqual(match.confidence, 0.75)


if __name__ == "__main__":
    unittest.main()
