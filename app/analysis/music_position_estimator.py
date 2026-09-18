"""Estimador de Posição Estrutural na Música (MusicPositionEstimator v0.3).

Responde à pergunta fundamental da banda: 'Onde estamos na música?'
Calcula em tempo real compasso, tempo, seção atual, progresso na seção atual [0.0 a 1.0],
padrão em execução e índice da ocorrência, sem saltos e sincronizado com o MusicalClock.
"""

from typing import Optional
from app.music.music_structure import MusicPosition, MusicSection, MusicalPattern
from app.music.musical_clock import MusicalClock
from app.music.musical_context import MusicalContext
from app.music.chart_alignment import ChartPosition


class MusicPositionEstimator:
    """Calcula continuamente a posição estrutural do áudio na composição."""

    def __init__(self):
        self._current_position = MusicPosition()

    @property
    def current_position(self) -> MusicPosition:
        return self._current_position

    def reset(self) -> None:
        """Reinicia os cálculos de posição para o início neutro."""
        self._current_position = MusicPosition()

    def estimate_position(
        self,
        timestamp: float,
        clock: MusicalClock,
        current_section: Optional[MusicSection] = None,
        current_pattern: Optional[MusicalPattern] = None,
        occurrence_index: int = 1,
        musical_context: Optional[MusicalContext] = None,
        chart_position: Optional[ChartPosition] = None,
    ) -> MusicPosition:
        """Calcula a posição estrutural atual e o progresso relativo na seção ativa."""
        # Este componente descreve estrutura; não localiza nem aplica offsets.
        bar = max(1, musical_context.bar if musical_context is not None else clock.bar)
        beat = max(1, musical_context.beat if musical_context is not None else clock.beat)
        phase = musical_context.beat_position if musical_context is not None else clock.beat_position
        bar_pos = (beat - 1 + phase) / clock.beats_per_bar

        sec_name = current_section.section_type if current_section else "UNKNOWN"
        pat_id = current_pattern.id if current_pattern else (current_section.pattern_id if current_section else "--")
        
        # Cálculo do progresso na seção ativa [0.0 a 1.0]
        if current_section and current_section.duration_bars > 0:
            bars_elapsed = (bar - current_section.start_bar) + bar_pos
            progress = max(0.0, min(1.0, bars_elapsed / current_section.duration_bars))
            conf = current_section.confidence
        elif current_pattern and current_pattern.duration_bars > 0:
            bars_elapsed = ((bar - 1) % current_pattern.duration_bars) + bar_pos
            progress = max(0.0, min(1.0, bars_elapsed / current_pattern.duration_bars))
            conf = current_pattern.confidence
        else:
            progress = 0.0
            conf = 0.40

        if chart_position is not None:
            sec_name = chart_position.section_name
            progress = chart_position.section_progress
            conf = chart_position.confidence

        self._current_position = MusicPosition(
            current_time=round(timestamp, 3),
            current_bar=bar,
            current_beat=beat,
            current_section=sec_name,
            section_progress=round(float(progress), 3),
            pattern_id=pat_id,
            pattern_occurrence_index=occurrence_index,
            confidence=round(conf, 2)
        )

        return self._current_position
