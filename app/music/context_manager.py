"""Gerenciador do Contexto Musical (MusicalContextManager) do Virtual Band AI.

Atua como ponte orquestradora entre os detectores brutos de DSP/MIR e o estado musical compartilhado:
  1. Aplica estabilização temporal em acordes e tonalidade.
  2. Alimenta o histórico cronológico de acordes e tonalidades.
  3. Sincroniza o MusicalClock com o tempo real do áudio.
  4. Consolida métricas quadripartidas de latência (processamento, janela, estabilização e estimada).
  5. Atualiza o MusicalContext thread-safe pronto para consumo pela UI e futuros músicos virtuais.
"""

from typing import Optional, List
import numpy as np

from app.music.musical_context import MusicalContext
from app.music.musical_clock import MusicalClock
from app.analysis.chord_history import ChordHistory, ChordStabilizer, ChordEvent
from app.analysis.key_history import KeyHistory, KeyStabilizer, KeyEvent
from app.analysis.pitch_detector import PitchResult
from app.analysis.chord_detector import Chord
from app.analysis.key_detector import KeyResult
from app.analysis.tempo_detector import TempoResult
from app.utils.timing import LatencyMetrics


class MusicalContextManager:
    """Orquestrador do estado musical de alto nível e memória temporal."""

    def __init__(self, sample_rate: int = 44100, bpm: float = 120.0, meter: str = "4/4"):
        self._sample_rate = sample_rate

        # Contexto musical ativo
        self._context = MusicalContext(sample_rate=sample_rate, bpm=bpm, meter=meter)

        # Relógio rítmico musical
        self._clock = MusicalClock(bpm=bpm, meter=meter)

        # Estabilizador e Histórico de Acordes
        self._chord_history = ChordHistory()
        self._chord_stabilizer = ChordStabilizer(history=self._chord_history)

        # Estabilizador e Histórico de Tonalidade
        self._key_history = KeyHistory()
        self._key_stabilizer = KeyStabilizer(history=self._key_history)

    @property
    def context(self) -> MusicalContext:
        return self._context

    @property
    def clock(self) -> MusicalClock:
        return self._clock

    @property
    def chord_history(self) -> ChordHistory:
        return self._chord_history

    @property
    def chord_stabilizer(self) -> ChordStabilizer:
        return self._chord_stabilizer

    @property
    def key_history(self) -> KeyHistory:
        return self._key_history

    @property
    def key_stabilizer(self) -> KeyStabilizer:
        return self._key_stabilizer

    def reset(self) -> None:
        """Reinicia todos os acumuladores de memória e o relógio."""
        self._context.reset()
        self._clock.reset()
        self._chord_stabilizer.reset()
        self._key_stabilizer.reset()

    def set_meter(self, meter_str: str) -> None:
        """Configura nova fórmula de compasso."""
        self._clock.set_meter(meter_str)
        self._context.meter = self._clock.meter
        self._context.time_signature = self._clock.meter

    def update(
        self,
        timestamp: float,
        sample_rate: int,
        pitch_res: Optional[PitchResult] = None,
        chord_res: Optional[Chord] = None,
        key_res: Optional[KeyResult] = None,
        tempo_res: Optional[TempoResult] = None,
        chroma_vector: Optional[np.ndarray] = None,
        lat_metrics: Optional[LatencyMetrics] = None
    ) -> MusicalContext:
        """Consolida as informações de todos os analisadores no MusicalContext."""
        ctx = self._context
        ctx.timestamp = timestamp
        ctx.sample_rate = sample_rate

        # 1. Atualizar Nota e Pitch Dominante
        if pitch_res is not None and pitch_res.is_voiced and pitch_res.confidence > 0.25:
            ctx.note = pitch_res.full_note
            ctx.frequency = pitch_res.frequency_hz
            ctx.note_confidence = pitch_res.confidence
            ctx.cents_deviation = pitch_res.cents_deviation
        else:
            ctx.note = "--"
            ctx.frequency = 0.0
            ctx.note_confidence = 0.0
            ctx.cents_deviation = 0.0

        if chroma_vector is not None:
            ctx.chroma_vector = chroma_vector.tolist() if isinstance(chroma_vector, np.ndarray) else list(chroma_vector)

        # 2. Estabilização e Histórico de Acordes
        if chord_res is not None:
            stable_chord = self._chord_stabilizer.process(
                raw_symbol=chord_res.symbol,
                confidence=chord_res.confidence,
                timestamp=timestamp,
                root=chord_res.root,
                quality=chord_res.quality,
                bass_note=chord_res.bass_note,
                inversion=chord_res.inversion,
                detected_notes=chord_res.detected_notes
            )
            ctx.chord = stable_chord
            ctx.previous_chord = self._chord_stabilizer.previous_chord
            ctx.chord_start_time = self._chord_stabilizer.chord_start_time
            ctx.chord_duration = self._chord_stabilizer.chord_duration
            ctx.chord_confidence = self._chord_stabilizer.chord_confidence
            ctx.chord_root = self._chord_stabilizer.chord_root
            ctx.chord_quality = self._chord_stabilizer.chord_quality
            ctx.bass_note = self._chord_stabilizer.bass_note
            ctx.inversion = self._chord_stabilizer.inversion
            ctx.detected_notes = self._chord_stabilizer.detected_notes
        else:
            ctx.chord = "--"
            ctx.previous_chord = "--"
            ctx.chord_duration = 0.0
            ctx.chord_confidence = 0.0

        # 3. Estabilização e Histórico de Tonalidade
        if key_res is not None:
            recent_chords = self._chord_history.get_recent(8)
            all_key_scores = getattr(key_res, "all_key_scores", None)
            stable_key = self._key_stabilizer.process(
                raw_key=key_res.key_name,
                confidence=key_res.confidence,
                timestamp=timestamp,
                root=key_res.root,
                scale_type=key_res.scale_type,
                recent_chords=recent_chords,
                all_key_scores=all_key_scores
            )
            ctx.key = stable_key
            ctx.previous_key = self._key_stabilizer.previous_key
            ctx.key_start_time = self._key_stabilizer.key_start_time
            ctx.key_duration = self._key_stabilizer.key_duration
            ctx.key_confidence = self._key_stabilizer.key_confidence
            ctx.key_candidate = self._key_stabilizer.candidate_key
            ctx.candidate_confidence = self._key_stabilizer.candidate_confidence
            ctx.candidate_duration = self._key_stabilizer.candidate_duration
            ctx.local_tonal_center = self._key_stabilizer.local_tonal_center
        else:
            ctx.key = "--"
            ctx.previous_key = "--"
            ctx.key_duration = 0.0
            ctx.key_confidence = 0.0
            ctx.key_candidate = "--"
            ctx.candidate_confidence = 0.0
            ctx.candidate_duration = 0.0
            ctx.local_tonal_center = "--"


        # 4. Relógio Musical (MusicalClock) e Posição Rítmica
        bpm_val = tempo_res.bpm if (tempo_res and tempo_res.bpm > 0) else ctx.bpm
        is_beat_val = tempo_res.is_beat if tempo_res else None

        self._clock.update(timestamp=timestamp, bpm=bpm_val, external_is_beat=is_beat_val)

        ctx.bpm = self._clock.bpm
        ctx.meter = self._clock.meter
        # Sem cifra, a posição musical coincide com a referência temporal.
        ctx.clock_bar = self._clock.bar
        ctx.clock_beat = self._clock.beat
        ctx.bar_offset = 0
        ctx.bar = ctx.clock_bar
        ctx.beat = ctx.clock_beat
        ctx.beat_position = self._clock.beat_position
        ctx.is_beat = self._clock.is_beat

        # 5. Métricas de Latência Quadripartida
        stab_delay = self._chord_stabilizer.stabilization_delay_ms
        ctx.stabilization_delay = stab_delay

        if lat_metrics is not None:
            ctx.processing_latency = lat_metrics.processing_ms
            ctx.analysis_window = lat_metrics.analysis_window_ms
            # Latência musical estimada: metade da janela de análise + tempo de CPU + atraso de estabilização
            ctx.estimated_musical_latency = (
                (lat_metrics.analysis_window_ms * 0.5) +
                lat_metrics.processing_ms +
                stab_delay
            )
        else:
            ctx.processing_latency = 0.0
            ctx.analysis_window = (4096 / float(sample_rate)) * 1000.0 if sample_rate > 0 else 92.8
            ctx.estimated_musical_latency = (ctx.analysis_window * 0.5) + stab_delay

        # 6. Cálculo da Confiança Global Ponderada
        weights = []
        scores = []
        if ctx.chord != "--":
            weights.append(0.40)
            scores.append(ctx.chord_confidence)
        if ctx.key != "--":
            weights.append(0.35)
            scores.append(ctx.key_confidence)
        if ctx.note != "--":
            weights.append(0.25)
            scores.append(ctx.note_confidence)

        if scores:
            ctx.confidence = float(np.average(scores, weights=weights))
        else:
            ctx.confidence = 0.0

        # 7. Sincronizar todos os aliases para garantir total retrocompatibilidade
        ctx.sync_aliases()

        return ctx
