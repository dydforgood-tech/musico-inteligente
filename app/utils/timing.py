"""Medição rigorosa de latência de processamento e latência musical perceptível."""

import time
from dataclasses import dataclass


@dataclass
class LatencyMetrics:
    """Métricas detalhadas de latência do sistema em milissegundos."""
    processing_ms: float = 0.0          # Tempo de execução do cálculo da CPU
    analysis_window_ms: float = 0.0     # Duração temporal do bloco de áudio analisado
    stabilization_ms: float = 0.0       # Atraso de suavização temporal / filtro
    estimated_musical_latency_ms: float = 0.0  # Latência musical estimada de ponta a ponta


class LatencyTracker:
    """Rastreador de latência com medição em nanossegundos via perf_counter."""

    def __init__(self, sample_rate: int = 44100, chunk_size: int = 2048, stabilization_ms: float = 15.0):
        self._sample_rate = sample_rate
        self._chunk_size = chunk_size
        self._stabilization_ms = stabilization_ms
        self._analysis_window_ms = (chunk_size / float(sample_rate)) * 1000.0 if sample_rate > 0 else 0.0

        self._start_time: float = 0.0
        self._last_metrics = LatencyMetrics(
            analysis_window_ms=self._analysis_window_ms,
            stabilization_ms=self._stabilization_ms
        )

    def update_config(self, sample_rate: int, chunk_size: int) -> None:
        """Atualiza a taxa de amostragem ou tamanho do bloco de análise."""
        self._sample_rate = sample_rate
        self._chunk_size = chunk_size
        self._analysis_window_ms = (chunk_size / float(sample_rate)) * 1000.0 if sample_rate > 0 else 0.0

    def start_measurement(self) -> None:
        """Inicia a contagem do tempo de processamento do bloco atual."""
        self._start_time = time.perf_counter()

    def end_measurement(self) -> LatencyMetrics:
        """Finaliza a contagem e retorna as métricas consolidadas."""
        elapsed = (time.perf_counter() - self._start_time) * 1000.0  # ms
        estimated_total = (self._analysis_window_ms * 0.5) + elapsed + self._stabilization_ms

        self._last_metrics = LatencyMetrics(
            processing_ms=elapsed,
            analysis_window_ms=self._analysis_window_ms,
            stabilization_ms=self._stabilization_ms,
            estimated_musical_latency_ms=estimated_total
        )
        return self._last_metrics

    @property
    def last_metrics(self) -> LatencyMetrics:
        return self._last_metrics
