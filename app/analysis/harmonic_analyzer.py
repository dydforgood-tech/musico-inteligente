"""Camada de Análise Harmônica (Separação entre Pitch Monofônico e Harmonia Polifônica)."""

from abc import ABC, abstractmethod
from typing import Optional
import numpy as np

from app.analysis.pitch_detector import PitchResult
from app.analysis.chord_detector import Chord, ChordHistory, ChordDetector


class HarmonicAnalyzer(ABC):
    """Interface abstrata para análise harmônica polifônica."""

    @abstractmethod
    def reset(self) -> None:
        pass

    @abstractmethod
    def analyze_harmony(
        self,
        audio_chunk: np.ndarray,
        sample_rate: int,
        dominant_pitch: Optional[PitchResult] = None,
        chroma_vector: Optional[np.ndarray] = None,
        timestamp: float = 0.0,
        expected_chord: Optional[str] = None,
        key: Optional[str] = None
    ) -> Chord:
        pass

    @property
    @abstractmethod
    def chord_history(self) -> ChordHistory:
        pass


class DefaultHarmonicAnalyzer(HarmonicAnalyzer):
    """Analisador harmônico padrão.
    
    Combina o cromagrama das 12 notas, o espectro de frequências graves (baixo)
    e o pitch dominante para reconhecer o acorde e suas inversões.
    """

    def __init__(self, detect_extensions: bool = False):
        self._chord_detector = ChordDetector(detect_extensions=detect_extensions)
        self._chord_history = ChordHistory(max_entries=10)

    def set_detect_extensions(self, enabled: bool) -> None:
        """Liga/desliga o reconhecimento de tétrades/suspensos em tempo de execução."""
        self._chord_detector = ChordDetector(detect_extensions=enabled)

    def reset(self) -> None:
        self._chord_detector.reset()
        self._chord_history.clear()

    @property
    def chord_history(self) -> ChordHistory:
        return self._chord_history

    def analyze_harmony(
        self,
        audio_chunk: np.ndarray,
        sample_rate: int,
        dominant_pitch: Optional[PitchResult] = None,
        chroma_vector: Optional[np.ndarray] = None,
        timestamp: float = 0.0,
        expected_chord: Optional[str] = None,
        key: Optional[str] = None
    ) -> Chord:
        if chroma_vector is None or len(chroma_vector) < 12:
            return Chord(timestamp=timestamp)

        chord = self._chord_detector.detect(
            chroma=chroma_vector,
            audio_chunk=audio_chunk,
            sample_rate=sample_rate,
            timestamp=timestamp,
            expected_chord=expected_chord,
            key=key
        )

        if chord.symbol != "--":
            self._chord_history.update(chord)

        return chord
