"""Motor de Performance Rítmica do Baixista Virtual (BassPerformanceEngine).

Responsável por converter os passos do padrão em eventos temporais concretos (BassNoteEvent):
- Sincronização estrita de milissegundos com o MusicalClock (sem time.sleep()).
- Cálculo do timestamp exato de início de cada nota (start_time).
- Articulação musical (duração da nota vs intervalo entre tempos).
- Dinâmica de intensidade (velocity por acentuação rítmica).
- Estrutura para humanização (desativada por padrão para testes determinísticos).
"""

from typing import List, Tuple
from app.instruments.bass_model import BassNoteEvent, BassDecision, BassPatternType
from app.music.constants import BASS_NOTE_ARTICULATION_RATIO


class BassPerformanceEngine:
    """Calcula timing, duração e dinâmica de cada nota gerada pelo padrão."""

    def __init__(
        self,
        articulation_ratio: float = BASS_NOTE_ARTICULATION_RATIO,
        humanization_enabled: bool = False
    ):
        self._articulation_ratio = articulation_ratio
        self.humanization_enabled = humanization_enabled

    def create_events_for_bar(
        self,
        decision: BassDecision,
        pattern_steps: List[Tuple[int, str, int, str]],
        bar: int,
        bar_start_time: float,
        beat_duration: float,
        total_beats: int = 4
    ) -> List[BassNoteEvent]:
        """Gera a lista de BassNoteEvents para o compasso sincronizado."""
        events: List[BassNoteEvent] = []

        is_sustained = (decision.pattern_type == BassPatternType.SUSTAINED)

        for step in pattern_steps:
            beat_idx, note_name, midi_note, reason = step

            # Timestamp de início baseado na grade do MusicalClock
            start_time = bar_start_time + (beat_idx - 1) * beat_duration

            # Duração
            if is_sustained:
                duration = beat_duration * total_beats * 0.95
            else:
                duration = beat_duration * self._articulation_ratio

            # Dinâmica e acentuação por tempo musical
            if beat_idx == 1:
                velocity = 100  # Acento no tempo forte
            elif beat_idx == 3:
                velocity = 92   # Tempo meio-forte
            else:
                velocity = 85   # Contratempos / tempos fracos

            ev = BassNoteEvent(
                note=note_name,
                midi_note=midi_note,
                start_time=round(start_time, 4),
                duration=round(duration, 4),
                velocity=velocity,
                beat=beat_idx,
                bar=bar,
                confidence=decision.confidence,
                reason=reason
            )
            events.append(ev)

        return events
