"""Estimador de Posição Estrutural na Música (MusicPositionEstimator v0.3).

Responde à pergunta fundamental da banda: 'Onde estamos na música?'
Calcula em tempo real compasso, tempo, seção atual, progresso na seção atual [0.0 a 1.0],
padrão em execução e índice da ocorrência, sem saltos e sincronizado com o MusicalClock.
"""

from typing import Optional
from app.music.music_structure import MusicPosition, MusicSection, MusicalPattern
from app.music.musical_clock import MusicalClock


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
        occurrence_index: int = 1
    ) -> MusicPosition:
        """Calcula a posição estrutural atual e o progresso relativo na seção ativa."""
        bar = max(1, clock.bar)
        beat = max(1, clock.beat)
        bar_pos = clock.bar_position # Fração [0.0 a 1.0) dentro do compasso atual

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
