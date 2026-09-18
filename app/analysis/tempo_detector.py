"""Detecção de Andamento (BPM), Rastreamento de Batidas (Beat Tracking) e Métricas Rítmicas."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
import librosa


@dataclass
class BeatEvent:
    """Evento temporal de pulso rítmico."""
    timestamp: float = 0.0          # Tempo exato do beat em segundos
    beat_number: int = 1            # Contagem dentro do compasso (1, 2, 3, 4)
    confidence: float = 0.0         # Força/clareza do onset
    bpm_at_beat: float = 120.0      # Andamento instantâneo no instante do beat


@dataclass
class TempoResult:
    """Resultado da análise rítmica de um bloco de áudio."""
    bpm: float = 0.0                # Andamento estimado em batidas por minuto
    is_beat: bool = False           # Verdadeiro se um pulso de batida ocorreu neste frame
    beat_number: int = 1            # Número do tempo (1 a 4, etc.)
    beat_position: float = 0.0      # Posição fracionária no compasso [0.0 a 1.0]
    time_signature: str = "4/4"     # Fórmula de compasso estimada
    confidence: float = 0.0         # Confiança da periodicidade rítmica


class TempoDetector(ABC):
    """Interface abstrata para detecção de BPM e grade rítmica."""

    @abstractmethod
    def analyze_audio(self, audio_data: np.ndarray, sample_rate: int) -> float:
        pass


class BeatTracker(ABC):
    """Interface abstrata para rastreamento de batidas em tempo real."""

    @abstractmethod
    def get_tempo_at_time(self, timestamp: float, tolerance_sec: float = 0.075) -> TempoResult:
        pass


class RhythmDetector(ABC):
    """Interface abstrata para análise de padrões rítmicos."""

    @abstractmethod
    def analyze_subdivision(self, audio_data: np.ndarray, sample_rate: int) -> str:
        pass


class OnsetTempoDetector(TempoDetector):
    """Detector de BPM e Beat Tracker baseado em envelope de transientes (onset strength).
    
    Combina pré-análise da grade de batidas e sincronização em tempo real frame a frame.
    """

    def __init__(self):
        self._bpm: float = 0.0
        self._beat_times: np.ndarray = np.array([], dtype=np.float32)
        self._last_beat_idx: int = -1
        self._beats_per_measure: int = 4

    @property
    def bpm(self) -> float:
        return self._bpm

    @property
    def beat_times(self) -> np.ndarray:
        return self._beat_times

    def reset(self) -> None:
        self._bpm = 0.0
        self._beat_times = np.array([], dtype=np.float32)
        self._last_beat_idx = -1

    def analyze_audio(self, audio_data: np.ndarray, sample_rate: int) -> float:
        """Analisa o sinal completo para extrair o BPM global e a grade precisa de tempos."""
        if len(audio_data) < sample_rate * 2 or sample_rate <= 0:
            return 0.0

        try:
            # Decimação para ~22050 Hz acelera o cálculo de onset STFT em mais de 3x mantendo precisão rítmica
            if sample_rate > 22050:
                hop = 2
                decimated_audio = audio_data[::hop]
                eff_sr = sample_rate // hop
            else:
                decimated_audio = audio_data
                eff_sr = sample_rate

            # Extrair envelope de onset e calcular BPM e frames de batidas
            tempo, beat_frames = librosa.beat.beat_track(y=decimated_audio, sr=eff_sr)
            self._bpm = float(np.mean(tempo))
            self._beat_times = librosa.frames_to_time(beat_frames, sr=eff_sr).astype(np.float32)
            self._last_beat_idx = -1
            print(f"[TempoDetector] BPM detectado: {self._bpm:.1f} | Total de batidas: {len(self._beat_times)}")
            return self._bpm
        except Exception as e:
            print(f"[TempoDetector] Erro ao detectar tempo: {e}")
            self._bpm = 0.0
            return 0.0

    def get_tempo_at_time(self, timestamp: float, tolerance_sec: float = 0.075) -> TempoResult:
        """Determina se o instante atual de reprodução coincide com um pulso de batida."""
        if self._bpm <= 0.0:
            return TempoResult(bpm=0.0, is_beat=False, beat_number=1, beat_position=0.0)

        # Se tivermos a grade pré-calculada
        if len(self._beat_times) > 0:
            # Encontrar a batida mais próxima
            diffs = np.abs(self._beat_times - timestamp)
            closest_idx = int(np.argmin(diffs))
            min_diff = diffs[closest_idx]

            is_beat = (min_diff <= tolerance_sec)
            beat_number = (closest_idx % self._beats_per_measure) + 1

            # Calcular fase fracionária no compasso (0.0 a 1.0)
            measure_duration = (60.0 / self._bpm) * self._beats_per_measure
            beat_pos = (timestamp % measure_duration) / measure_duration

            return TempoResult(
                bpm=round(self._bpm, 1),
                is_beat=is_beat,
                beat_number=beat_number,
                beat_position=float(beat_pos),
                time_signature="4/4",
                confidence=0.88
            )

        # Caso contrário (estimativa analítica por tempo)
        beat_interval = 60.0 / self._bpm
        measure_duration = beat_interval * self._beats_per_measure
        beat_phase = (timestamp % beat_interval) / beat_interval
        is_beat = (beat_phase < 0.15 or beat_phase > 0.85)
        current_beat = int((timestamp % measure_duration) / beat_interval) + 1
        beat_pos = (timestamp % measure_duration) / measure_duration

        return TempoResult(
            bpm=round(self._bpm, 1),
            is_beat=is_beat,
            beat_number=current_beat,
            beat_position=float(beat_pos),
            time_signature="4/4",
            confidence=0.75
        )
