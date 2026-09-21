"""Ritmo harmônico observado em beats, independente do BPM instantâneo."""

from dataclasses import dataclass, field
from typing import List, Optional
import re

from app.input.chart_semantic_classifier import ChartSemanticClassifier


COMMON_BEAT_DURATIONS = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0)


def quantize_beats(raw_beats: float) -> float:
    """Quantiza erros humanos pequenos sem limitar a música aos valores comuns."""
    raw = max(0.0, float(raw_beats))
    if raw <= 0.0:
        return 0.0
    nearest = min(COMMON_BEAT_DURATIONS, key=lambda value: abs(value - raw))
    if abs(raw - nearest) <= max(0.18, nearest * 0.16):
        return nearest
    return round(raw * 4.0) / 4.0


def section_key(section_name: str) -> str:
    """Agrupa `Verse 1` e `Verse 2`, mantendo Intro e Chorus distintos."""
    clean = re.sub(r"\d+", "", (section_name or "UNKNOWN").upper())
    return " ".join(clean.split()) or "UNKNOWN"


@dataclass(frozen=True)
class HarmonicRhythmEvent:
    symbol: str
    start_beat: float
    raw_duration_beats: float
    quantized_duration_beats: float
    confidence: float
    section_key: str


@dataclass
class HarmonicRhythmPattern:
    """Perfil temporal aprendido para uma seção durante a sessão atual."""
    id: str
    section_key: str
    chord_sequence: List[str]
    duration_sequence: List[float]
    observations: int = 1
    confidence: float = 0.0
    variance: List[float] = field(default_factory=list)

    @property
    def total_beats(self) -> float:
        return sum(self.duration_sequence)


@dataclass(frozen=True)
class HarmonicRhythmState:
    current_chord: str = "--"
    current_elapsed_beats: float = 0.0
    expected_chord_duration_beats: float = 0.0
    beats_until_change: float = 0.0
    next_chord: str = "--"
    duration_confidence: float = 0.0
    pattern_confidence: float = 0.0
    pattern_id: str = "--"
    observation_count: int = 0


class HarmonicRhythmTracker:
    """Fecha apenas acordes estabilizados e aprende ocorrências completas de seção."""

    def __init__(self, memory):
        self._memory = memory
        self.reset()

    def reset(self) -> None:
        self._section_key = "UNKNOWN"
        self._active_chord = "--"
        self._active_start_beat = 0.0
        self._active_confidence = 0.0
        self._section_events: List[HarmonicRhythmEvent] = []
        self._timeline: List[HarmonicRhythmEvent] = []
        self._last_state = HarmonicRhythmState()

    @property
    def timeline(self) -> List[HarmonicRhythmEvent]:
        return list(self._timeline)

    @property
    def recent_events(self) -> List[HarmonicRhythmEvent]:
        return list(self._section_events[-4:])

    @property
    def state(self) -> HarmonicRhythmState:
        return self._last_state

    def _finish_active(self, end_beat: float) -> None:
        if self._active_chord == "--":
            return
        raw = max(0.0, end_beat - self._active_start_beat)
        if raw < 0.30:
            return
        event = HarmonicRhythmEvent(
            symbol=self._active_chord, start_beat=self._active_start_beat,
            raw_duration_beats=raw, quantized_duration_beats=quantize_beats(raw),
            confidence=self._active_confidence, section_key=self._section_key)
        self._section_events.append(event)
        self._timeline.append(event)

    def _commit_section(self) -> None:
        if len(self._section_events) < 2:
            return
        confidence = sum(event.confidence for event in self._section_events) / len(self._section_events)
        self._memory.register_harmonic_rhythm(
            section_key=self._section_key,
            chord_sequence=[event.symbol for event in self._section_events],
            duration_sequence=[event.quantized_duration_beats for event in self._section_events],
            confidence=confidence)

    def observe(self, absolute_beat: float, section_name: str, detected_chord: str,
                chord_confidence: float) -> HarmonicRhythmState:
        """Consome apenas acorde já estabilizado; UNKNOWN mantém o acorde vigente."""
        beat = max(0.0, float(absolute_beat))
        key = section_key(section_name)
        if key != self._section_key:
            self._finish_active(beat)
            self._commit_section()
            self._section_key = key
            self._active_chord = "--"
            self._section_events = []

        valid = (bool(detected_chord) and detected_chord not in ("--", "UNKNOWN", "N") and
                 chord_confidence >= 0.35 and
                 ChartSemanticClassifier.is_chord_shaped(detected_chord))
        if valid:
            if self._active_chord == "--":
                self._active_chord = detected_chord
                self._active_start_beat = beat
                self._active_confidence = chord_confidence
            elif detected_chord == self._active_chord:
                self._active_confidence = self._active_confidence * 0.85 + chord_confidence * 0.15
            else:
                # A entrada já passou pelo ChordStabilizer; esta é uma mudança confirmada,
                # não uma leitura crua de um frame isolado.
                self._finish_active(beat)
                self._active_chord = detected_chord
                self._active_start_beat = beat
                self._active_confidence = chord_confidence

        elapsed = max(0.0, beat - self._active_start_beat) if self._active_chord != "--" else 0.0
        pattern = self._memory.find_harmonic_rhythm(self._section_key, self._active_chord,
                                                     len(self._section_events))
        if pattern is None:
            self._last_state = HarmonicRhythmState(current_chord=self._active_chord,
                                                    current_elapsed_beats=elapsed)
            return self._last_state
        index = min(len(pattern.chord_sequence) - 1, len(self._section_events))
        expected = pattern.duration_sequence[index]
        next_chord = (pattern.chord_sequence[index + 1]
                      if index + 1 < len(pattern.chord_sequence) else "--")
        variance = pattern.variance[index] if index < len(pattern.variance) else 0.0
        duration_conf = max(0.0, min(1.0, pattern.confidence * (1.0 - min(.5, variance / max(1.0, expected)))))
        self._last_state = HarmonicRhythmState(
            current_chord=self._active_chord, current_elapsed_beats=elapsed,
            expected_chord_duration_beats=expected,
            beats_until_change=max(0.0, expected - elapsed), next_chord=next_chord,
            duration_confidence=duration_conf, pattern_confidence=pattern.confidence,
            pattern_id=pattern.id, observation_count=pattern.observations)
        return self._last_state
