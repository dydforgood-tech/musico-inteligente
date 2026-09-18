"""Módulo de análise DSP e Music Information Retrieval (MIR)."""

from app.analysis.pitch_detector import PitchDetector, PitchResult
from app.analysis.chord_detector import Chord, ChordHistory
from app.analysis.harmonic_analyzer import HarmonicAnalyzer
from app.analysis.key_detector import KeyDetector, KeyResult, KeyHistory, KeyChangeDetector
from app.analysis.tempo_detector import TempoDetector, BeatTracker, RhythmDetector, TempoResult, BeatEvent

__all__ = [
    "PitchDetector",
    "PitchResult",
    "Chord",
    "ChordHistory",
    "HarmonicAnalyzer",
    "KeyDetector",
    "KeyResult",
    "KeyHistory",
    "KeyChangeDetector",
    "TempoDetector",
    "BeatTracker",
    "RhythmDetector",
    "TempoResult",
    "BeatEvent"
]
