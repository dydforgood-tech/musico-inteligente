"""Relógio Musical (MusicalClock) sincronizado com a posição real do áudio.

Converte tempo contínuo (segundos) e andamento (BPM) em posições musicais discretas:
compasso (bar), tempo (beat), fase do tempo (beat_position) e fórmula de compasso (meter).
"""

from typing import Tuple
from app.music.constants import (
    DEFAULT_METER,
    DEFAULT_BEATS_PER_BAR,
    DEFAULT_BEAT_UNIT,
    DEFAULT_BPM,
    BEAT_PULSE_TOLERANCE_SEC,
)


class MusicalClock:
    """Relógio rítmico que calcula a métrica musical a partir da posição temporal real do áudio.
    
    Projetado para suportar qualquer fórmula de compasso (4/4, 3/4, 2/4, 6/8, etc.),
    e manter sincronismo instantâneo com seek, play, pause e stop.
    """

    def __init__(self, bpm: float = DEFAULT_BPM, meter: str = DEFAULT_METER):
        self._bpm = max(0.0, float(bpm))
        self._meter = meter
        self._beats_per_bar, self._beat_unit = self._parse_meter(meter)

        # Estado rítmico instantâneo
        self._elapsed_time: float = 0.0
        self._bar: int = 1
        self._beat: int = 1
        self._beat_position: float = 0.0       # [0.0 a 1.0) dentro do tempo atual
        self._bar_position: float = 0.0        # [0.0 a 1.0) dentro do compasso atual
        self._is_beat: bool = False

    @staticmethod
    def _parse_meter(meter_str: str) -> Tuple[int, int]:
        """Extrai numerador (tempos por compasso) e denominador (unidade de tempo) da fórmula."""
        try:
            parts = meter_str.strip().split('/')
            if len(parts) == 2:
                num = int(parts[0])
                den = int(parts[1])
                if num > 0 and den > 0:
                    return num, den
        except Exception:
            pass
        return DEFAULT_BEATS_PER_BAR, DEFAULT_BEAT_UNIT

    @property
    def bpm(self) -> float:
        return self._bpm

    @bpm.setter
    def bpm(self, value: float) -> None:
        self._bpm = max(0.0, float(value))

    @property
    def meter(self) -> str:
        return self._meter

    @property
    def beats_per_bar(self) -> int:
        return self._beats_per_bar

    @property
    def beat_unit(self) -> int:
        return self._beat_unit

    @property
    def bar(self) -> int:
        """Número do compasso atual (1-indexado: 1, 2, 3...)."""
        return self._bar

    @property
    def beat(self) -> int:
        """Número do tempo no compasso atual (1-indexado: 1 a beats_per_bar)."""
        return self._beat

    @property
    def beat_position(self) -> float:
        """Posição fracionária dentro do tempo atual [0.0 a 1.0)."""
        return self._beat_position

    @property
    def bar_position(self) -> float:
        """Posição fracionária dentro do compasso atual [0.0 a 1.0)."""
        return self._bar_position

    @property
    def elapsed_time(self) -> float:
        """Tempo decorrido no áudio em segundos."""
        return self._elapsed_time

    @property
    def is_beat(self) -> bool:
        """Indica se o instante atual coincide com o clique/pulso da batida."""
        return self._is_beat

    @property
    def beat_duration(self) -> float:
        """Duração de um tempo rítmico em segundos."""
        if self._bpm <= 0:
            return 0.5  # Padrão seguro para 120 BPM
        # Para semínima (unidade 4), 60 / BPM
        # Se for colcheia (unidade 8), tempo = (60 / BPM) * (4 / 8)
        return (60.0 / self._bpm) * (4.0 / self._beat_unit)

    @property
    def bar_duration(self) -> float:
        """Duração total de um compasso completo em segundos."""
        return self.beat_duration * self._beats_per_bar

    def set_meter(self, meter_str: str) -> None:
        """Configura a fórmula de compasso (ex: '4/4', '3/4', '6/8')."""
        self._beats_per_bar, self._beat_unit = self._parse_meter(meter_str)
        self._meter = f"{self._beats_per_bar}/{self._beat_unit}"
        self.update(self._elapsed_time)

    def reset(self) -> None:
        """Reinicia o relógio musical para o início (t=0.0)."""
        self._elapsed_time = 0.0
        self._bar = 1
        self._beat = 1
        self._beat_position = 0.0
        self._bar_position = 0.0
        self._is_beat = False

    def seek(self, timestamp: float) -> None:
        """Move o relógio diretamente para o timestamp especificado."""
        self.update(timestamp)

    def tick(self, dt: float) -> None:
        """Avança o relógio musical em dt segundos."""
        self.update(self._elapsed_time + dt)

    def update(self, timestamp: float, bpm: float = None, external_is_beat: bool = None) -> None:
        """Atualiza a posição musical com base no timestamp real do áudio.
        
        Garante sincronia instantânea em caso de seek, pause, reprodução contínua ou stop.
        """
        self._elapsed_time = max(0.0, float(timestamp))
        if bpm is not None and bpm > 0:
            self._bpm = float(bpm)

        if self._bpm <= 0.0:
            self._bar = 1
            self._beat = 1
            self._beat_position = 0.0
            self._bar_position = 0.0
            self._is_beat = False
            return

        b_dur = self.beat_duration
        measure_dur = self.bar_duration

        if b_dur <= 0 or measure_dur <= 0:
            return

        # Total de batidas decorridas desde o início (com precisão de ponto flutuante)
        # Adicionar epsilon para evitar imprecisão numérica em múltiplos exatos (ex: 0.500000000001)
        eps = 1e-6
        total_beats = (self._elapsed_time + eps) / b_dur

        # Compasso (1-indexado)
        self._bar = int(self._elapsed_time // measure_dur) + 1

        # Tempo no compasso (1-indexado de 1 a beats_per_bar)
        self._beat = int(total_beats % self._beats_per_bar) + 1

        # Posição fracionária dentro do tempo [0.0 a 1.0)
        self._beat_position = float(total_beats % 1.0)

        # Posição fracionária dentro do compasso [0.0 a 1.0)
        self._bar_position = float((self._elapsed_time % measure_dur) / measure_dur)

        # Verificação do pulso de batida
        if external_is_beat is not None:
            self._is_beat = external_is_beat
        else:
            time_into_beat = (self._elapsed_time % b_dur)
            # Pulso no início da batida (janela de tolerância)
            self._is_beat = (time_into_beat <= BEAT_PULSE_TOLERANCE_SEC or 
                             time_into_beat >= (b_dur - BEAT_PULSE_TOLERANCE_SEC))
