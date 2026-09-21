"""Sessão de Execução Musical da Música Ativa (SongSession v0.4).

Gerencia exclusivamente o estado de execução temporário e volátil da música selecionada:
MusicalContext, MusicalClock, ChartAlignment, ChartAudioFusion e VirtualPlayers.
Garante isolamento hermético entre trocas de músicas para evitar memory leaks.
"""

from typing import Optional, Dict, Any, List
import copy
from dataclasses import fields
from dataclasses import replace
from enum import Enum

from app.song.song import Song
from app.music.musical_context import MusicalContext
from app.music.musical_clock import MusicalClock
from app.music.chord_chart import ChordChart, ChartSection, ChartChord, parse_chord
from app.input.chart_parser import ChartParser
from app.music.chart_alignment import ChartAlignment, ChartPosition
from app.music.chart_audio_fusion import ChartAudioFusion, FusedMusicalState
from app.analysis.music_structure_analyzer import MusicStructureAnalyzer
from app.music.pattern_memory import PatternMemory
from app.music.prediction_engine import PredictionEngine, MusicPrediction
from app.music.position_estimator import PositionEstimator, TrackingState, EstimatedPosition
from app.analysis.tempo_detector import TempoResult
from app.music.follow_confidence import calculate_follow_confidence
from app.music.harmonic_rhythm import HarmonicRhythmTracker


class PerformanceState(str, Enum):
    """Estado da banda; distinto do rastreamento de posição e do relógio."""
    PLAYING = "PLAYING"
    UNCERTAIN = "UNCERTAIN"
    HOLDING = "HOLDING"
    WAITING = "WAITING"
    RECOVERING = "RECOVERING"
    ENDED = "ENDED"


class SongSession:
    """Representa a sessão ativa de execução e reprodução de uma Song.
    
    Toda a interface gráfica e os instrumentos virtuais devem consultar esta classe
    para obter a verdade sobre o momento musical presente.
    """
    _SHORT_PAUSE_SECONDS = 2.5
    _WAITING_SECONDS = 4.0
    _END_SILENCE_SECONDS = 8.0
    _RECOVERY_MIN_SECONDS = 0.30

    def __init__(self, song: Song, mode: str = "PLAYBACK"):
        self._song: Song = song
        self._mode: str = mode          # "PLAYBACK", "FOLLOW", "MANUAL"
        self._is_playing: bool = False
        self._is_paused: bool = False

        # 1. Relógio Musical isolado
        self._clock: MusicalClock = MusicalClock(
            bpm=song.performance_settings.bpm_override or song.bpm,
            meter=song.performance_settings.meter_override or song.meter
        )

        # 2. Contexto Musical isolado
        self._context: MusicalContext = MusicalContext(
            bpm=self._clock.bpm,
            meter=self._clock.meter,
            key=song.performance_settings.key_override or song.key
        )

        # 3. Cifra estruturada (parseada da canção)
        if song.chart_data:
            self._chart: ChordChart = ChordChart.from_dict(song.chart_data)
        elif song.chart_text:
            self._chart = ChartParser.parse(
                text=song.chart_text,
                default_key=song.key,
                default_bpm=song.bpm,
                default_meter=song.meter,
                title=song.title,
                artist=song.artist
            )
        else:
            self._chart = ChordChart(
                title=song.title,
                artist=song.artist,
                key=song.key,
                bpm=song.bpm,
                meter=song.meter
            )

        # 4. Alinhamento e Fusão
        self._alignment: ChartAlignment = ChartAlignment(self._chart)
        self._fusion: ChartAudioFusion = ChartAudioFusion()

        # 5. Motores de Estrutura e Predição isolados
        self._structure_analyzer: MusicStructureAnalyzer = MusicStructureAnalyzer()
        self._prediction_engine: PredictionEngine = self._structure_analyzer.prediction_engine
        self._harmonic_rhythm = HarmonicRhythmTracker(self._structure_analyzer.pattern_memory)

        # 6. Estimador Contínuo de Posição Musical (v0.4)
        self._position_estimator: PositionEstimator = PositionEstimator(
            alignment=self._alignment,
            clock=self._clock,
            structure_analyzer=self._structure_analyzer,
            prediction_engine=self._prediction_engine
        )
        # Informa o capotraste da cifra (CifraClub) para localizar pelo som real
        try:
            self._position_estimator.set_capo(self._chart.capo_semitones)
        except Exception:
            pass

        # Se a música já possui padrões e estruturas salvos, restaura-os
        if song.pattern_memory_data:
            pass # Pode ser deserializado no futuro

        # 7. Estado Instantâneo Consolidado
        self._position_generation = 0
        self._chart_alignment_confidence = max(0.0, min(1.0, float(
            song.metadata.get("chart_overall_confidence", 1.0))))
        self._follow_stability = 0.70
        self._last_follow_timestamp: Optional[float] = None
        self._follow_observations_available = False
        self._performance_state = PerformanceState.WAITING
        self._last_activity_time: Optional[float] = None
        self._recovery_started_at: Optional[float] = None
        self._recovery_evidence = 0
        self._current_chart_pos: ChartPosition = self._position_estimator.refresh_position()
        self._last_confident_chart_pos = self._current_chart_pos
        self._current_fused_state: FusedMusicalState = FusedMusicalState(
            expected_chord=self._current_chart_pos.current_chord,
            effective_chord=self._current_chart_pos.current_chord,
            next_expected_chord=self._current_chart_pos.next_chord,
            current_section=self._current_chart_pos.section_name,
            next_section=self._current_chart_pos.next_section_name
        )
        self._publish_position(update_structure=False)
        self._player_states: Dict[str, Any] = {
            "bass": "READY",
            "drums": "READY",
            "keyboard": "READY",
            "guitar": "READY",
        }

    # ============================================================
    # Propriedades de Consulta Central
    # ============================================================
    @property
    def song(self) -> Song:
        return self._song

    @property
    def mode(self) -> str:
        return self._mode

    @mode.setter
    def mode(self, new_mode: str) -> None:
        self._mode = new_mode

    @property
    def is_playing(self) -> bool:
        return self._is_playing

    @property
    def is_paused(self) -> bool:
        return self._is_paused

    @property
    def performance_state(self) -> str:
        return self._performance_state.value

    @property
    def clock(self) -> MusicalClock:
        return self._clock

    @property
    def context(self) -> MusicalContext:
        return self._context

    @property
    def chart(self) -> ChordChart:
        return self._chart

    @property
    def alignment(self) -> ChartAlignment:
        return self._alignment

    @property
    def fusion(self) -> ChartAudioFusion:
        return self._fusion

    @property
    def structure_analyzer(self) -> MusicStructureAnalyzer:
        return self._structure_analyzer

    @property
    def current_bar(self) -> int:
        """Compasso musical oficial, após localização/reancoragem na cifra."""
        return self._current_chart_pos.current_bar

    @property
    def current_beat(self) -> int:
        return int(self._current_chart_pos.current_beat)

    @property
    def clock_bar(self) -> int:
        """Compasso bruto calculado a partir do tempo do áudio."""
        return self._clock.bar

    @property
    def clock_beat(self) -> int:
        return self._clock.beat

    @property
    def current_chord(self) -> str:
        return self._current_fused_state.effective_chord

    @property
    def expected_chord(self) -> str:
        return self._current_fused_state.expected_chord

    @property
    def detected_chord(self) -> str:
        return self._current_fused_state.detected_chord

    @property
    def next_chord(self) -> str:
        return self._current_fused_state.next_expected_chord

    @property
    def current_section(self) -> str:
        return self._current_fused_state.current_section

    @property
    def next_section(self) -> str:
        return self._current_fused_state.next_section

    @property
    def current_lyric(self) -> str:
        return self._current_fused_state.lyric

    @property
    def chart_position(self) -> ChartPosition:
        return self._current_chart_pos

    @property
    def position_estimator(self) -> PositionEstimator:
        return self._position_estimator

    @property
    def tracking_state(self) -> str:
        return self._current_chart_pos.tracking_state

    @property
    def position_confidence(self) -> float:
        return self._current_chart_pos.confidence

    @property
    def current_line(self) -> int:
        return self._current_chart_pos.line_index

    @property
    def prediction(self) -> Optional[MusicPrediction]:
        return self._prediction_engine.latest_prediction

    @property
    def player_states(self) -> Dict[str, Any]:
        return dict(self._player_states)

    # ============================================================
    # Ciclo de Vida e Limpeza
    # ============================================================
    def start(self) -> None:
        self._is_playing = True
        self._is_paused = False
        if self._performance_state == PerformanceState.ENDED:
            self._set_performance_state(PerformanceState.WAITING)

    def pause(self) -> None:
        self._is_paused = True

    def resume(self) -> None:
        self._is_paused = False
        self._is_playing = True

    def stop(self) -> None:
        self._is_playing = False
        self._is_paused = False
        self.reset()

    def reset(self) -> None:
        """Reinicia os relógios, fusão e contextos para o início da música (t=0.0)."""
        self._position_generation += 1
        self._performance_state = PerformanceState.WAITING
        self._last_activity_time = None
        self._recovery_started_at = None
        self._recovery_evidence = 0
        self._follow_stability = 0.70
        self._last_follow_timestamp = None
        self._follow_observations_available = False
        self._clock.bpm = self._song.performance_settings.bpm_override or self._song.bpm
        self._clock.reset()
        self._context.reset()
        self._context.bpm = self._clock.bpm
        self._context.meter = self._clock.meter
        self._context.key = self._song.key
        self._fusion.reset()
        self._structure_analyzer.reset()
        self._harmonic_rhythm.reset()
        self._position_estimator.reset()
        self._current_chart_pos = self._position_estimator.refresh_position()
        self._last_confident_chart_pos = self._current_chart_pos
        self._current_fused_state = FusedMusicalState(
            expected_chord=self._current_chart_pos.current_chord,
            effective_chord=self._current_chart_pos.current_chord,
            next_expected_chord=self._current_chart_pos.next_chord,
            current_section=self._current_chart_pos.section_name,
            next_section=self._current_chart_pos.next_section_name
        )
        self._publish_position(update_structure=False)

    def close(self) -> None:
        """Libera integralmente todos os recursos e estados da sessão anterior."""
        self.stop()
        self.reset()

    # ============================================================
    # Atualização Contínua de Tempo e Áudio
    # ============================================================
    def update_audio_tick(
        self,
        timestamp: float,
        detected_chord: str = "--",
        detected_confidence: float = 0.0,
        detected_key: str = "--",
        detected_bpm: float = 0.0,
        source_context: Optional[MusicalContext] = None,
        tempo_result: Optional[TempoResult] = None,
    ) -> FusedMusicalState:
        """Tempo bruto → estimador → snapshot musical → todos os consumidores."""
        if source_context is not None:
            self._follow_observations_available = True
        if timestamp < self._clock.elapsed_time - 0.01:
            # Seek da fonte invalida o horário de qualquer evento preparado.
            self._position_generation += 1
        manual_bpm = self._song.performance_settings.bpm_override
        initial_observation = self._clock.elapsed_time == 0.0 and self._clock.tempo_confidence == 0.0
        bpm_input = (manual_bpm or
                     (detected_bpm if detected_bpm > 0 and
                      (tempo_result is None or initial_observation) else None))
        self._clock.update(
            timestamp, bpm=bpm_input,
            beat_timestamp=tempo_result.beat_timestamp if tempo_result else None,
            observation_confidence=tempo_result.confidence if tempo_result else 0.0)
        self._harmonic_rhythm.observe(
            self._clock.total_beats, self._current_chart_pos.section_name,
            detected_chord, detected_confidence)
        activity = self._has_musical_activity(source_context, detected_chord,
                                               detected_confidence, tempo_result)
        should_localize = self._update_performance_state(
            timestamp, activity, activity_observable=source_context is not None)
        if should_localize:
            old_offset = self._position_estimator.bar_offset
            self._current_chart_pos = self._position_estimator.update(
                timestamp=timestamp, detected_chord=detected_chord,
                detected_confidence=detected_confidence, detected_key=detected_key,
                detected_note=source_context.note if source_context is not None else "--",
                note_confidence=source_context.note_confidence if source_context is not None else 0.0,
                harmonic_rhythm_events=self._harmonic_rhythm.timeline)
            if self._position_estimator.bar_offset != old_offset:
                self._position_generation += 1
            if activity and self._current_chart_pos.confidence >= 0.50:
                self._last_confident_chart_pos = self._current_chart_pos
        else:
            # Mantém a última posição confirmada: pausa não é avanço nem fim da música.
            self._current_chart_pos = replace(self._last_confident_chart_pos,
                                               absolute_time=timestamp)
        return self._publish_position(detected_chord, detected_confidence, detected_key, source_context)

    def _has_musical_activity(self, source_context: Optional[MusicalContext],
                               chord: str, chord_confidence: float,
                               tempo_result: Optional[TempoResult]) -> bool:
        if chord not in ("", "--", "UNKNOWN", "N") and chord_confidence >= 0.35:
            return True
        if tempo_result and tempo_result.beat_timestamp is not None and tempo_result.confidence >= 0.60:
            return True
        if source_context is None:
            return False
        return (source_context.audio_activity >= 0.010 or
                source_context.note_confidence >= 0.35 or
                source_context.chord_confidence >= 0.35)

    def _set_performance_state(self, state: PerformanceState) -> None:
        if state != self._performance_state:
            self._performance_state = state
            self._position_generation += 1

    def _near_chart_end(self) -> bool:
        return (self._last_confident_chart_pos.current_bar >=
                max(1, self._alignment.total_bars - 1))

    def _update_performance_state(self, timestamp: float, activity: bool,
                                  activity_observable: bool) -> bool:
        """Atualiza apenas a decisão de tocar; posição e relógio mantêm responsabilidades próprias."""
        if self._performance_state == PerformanceState.ENDED:
            return False
        # Chamadores de teste/importação podem fornecer somente tempo e cifra. Isso não é
        # evidência de silêncio do músico; preserva o modo temporal sem sensor de atividade.
        if not activity_observable:
            self._last_activity_time = timestamp
            if self._performance_state == PerformanceState.WAITING:
                self._set_performance_state(PerformanceState.PLAYING)
            return True
        if activity:
            if self._performance_state in (PerformanceState.WAITING,
                                           PerformanceState.HOLDING,
                                           PerformanceState.UNCERTAIN):
                self._set_performance_state(PerformanceState.RECOVERING)
                self._recovery_started_at = timestamp
                self._recovery_evidence = 0
            self._last_activity_time = timestamp
            if self._performance_state == PerformanceState.RECOVERING:
                self._recovery_evidence += 1
                elapsed = timestamp - (self._recovery_started_at or timestamp)
                at_downbeat = self._clock.beat == 1 and self._clock.beat_position <= 0.12
                if ((self._recovery_evidence >= 2 or elapsed >= self._RECOVERY_MIN_SECONDS)
                        and self._position_estimator.tracking_state != TrackingState.LOST
                        and at_downbeat):
                    self._set_performance_state(PerformanceState.PLAYING)
            elif self._performance_state == PerformanceState.PLAYING:
                pass
            return True

        if self._last_activity_time is None:
            return False
        silence = max(0.0, timestamp - self._last_activity_time)
        if self._performance_state == PerformanceState.PLAYING and silence >= 0.35:
            self._set_performance_state(PerformanceState.UNCERTAIN)
        if self._performance_state in (PerformanceState.PLAYING, PerformanceState.UNCERTAIN,
                                       PerformanceState.RECOVERING) and silence >= self._SHORT_PAUSE_SECONDS:
            self._set_performance_state(PerformanceState.HOLDING)
        if self._performance_state == PerformanceState.HOLDING and silence >= self._WAITING_SECONDS:
            self._set_performance_state(PerformanceState.WAITING)
        if (self._performance_state == PerformanceState.WAITING and
                silence >= self._END_SILENCE_SECONDS and self._near_chart_end()):
            self._set_performance_state(PerformanceState.ENDED)
        return self._performance_state not in (PerformanceState.HOLDING,
                                                PerformanceState.WAITING,
                                                PerformanceState.ENDED)

    def _publish_position(self, detected_chord: str = "--", detected_confidence: float = 0.0,
                          detected_key: str = "--", source_context: Optional[MusicalContext] = None,
                          update_structure: bool = True) -> FusedMusicalState:
        """Publica o mesmo snapshot na fusão, contexto, estrutura e predição."""
        position = self._current_chart_pos
        timestamp = position.absolute_time
        previous = self._context.chord
        previous_start = self._context.chord_start_time
        previous_time = self._context.timestamp
        if source_context is not None:
            for item in fields(MusicalContext):
                setattr(self._context, item.name, copy.deepcopy(getattr(source_context, item.name)))
        state = self._fusion.fuse(position, detected_chord, detected_confidence, timestamp)
        self._current_fused_state = state
        ctx = self._context
        ctx.timestamp = timestamp
        ctx.bpm = self._clock.bpm
        ctx.meter = self._clock.meter
        ctx.bar = position.current_bar
        ctx.beat = int(position.current_beat)
        ctx.beat_position = position.current_beat - ctx.beat
        ctx.is_beat = self._clock.is_beat
        ctx.initial_bpm = self._clock.initial_bpm
        ctx.target_bpm = self._clock.target_bpm
        ctx.tempo_confidence = self._clock.tempo_confidence
        ctx.phase_confidence = self._clock.phase_confidence
        ctx.phase_error_ms = self._clock.phase_error_ms
        ctx.tempo_tracking_state = self._clock.tracking_state
        ctx.clock_bar = self.clock_bar
        ctx.clock_beat = self.clock_beat
        ctx.bar_offset = self._position_estimator.bar_offset
        ctx.line_index = position.line_index
        ctx.tracking_state = position.tracking_state
        ctx.position_confidence = position.confidence
        ctx.chord = state.effective_chord
        ctx.next_expected_chord = position.next_chord
        ctx.next_change_bar = (position.current_bar + position.bars_until_chord_change
                               if position.next_chord != "--" and position.bars_until_chord_change > 0 else 0)
        ctx.next_change_beat = 1
        ctx.next_expected_section = position.next_section_name
        ctx.position_generation = self._position_generation
        ctx.chart_available = bool(self._chart.sections and position.current_chord != "--")
        ctx.chart_alignment_confidence = self._chart_alignment_confidence if ctx.chart_available else 0.0
        ctx.performance_state = self.performance_state
        ctx.confirmed_variation_chord = (detected_chord if state.confirmed_variation
                                         and detected_confidence >= 0.85 else "--")
        ctx.chord_confidence = state.confidence
        rhythm = self._harmonic_rhythm.state
        ctx.current_chord_elapsed_beats = rhythm.current_elapsed_beats
        ctx.expected_chord_duration_beats = rhythm.expected_chord_duration_beats
        ctx.beats_until_chord_change = rhythm.beats_until_change
        ctx.duration_confidence = rhythm.duration_confidence
        ctx.pattern_confidence = rhythm.pattern_confidence
        ctx.harmonic_rhythm_pattern = rhythm.pattern_id
        ctx.harmonic_rhythm_observations = rhythm.observation_count
        ctx.rhythmic_next_chord = rhythm.next_chord
        if previous != ctx.chord or timestamp < previous_time:
            ctx.previous_chord = previous
            ctx.chord_start_time = timestamp
        else:
            ctx.chord_start_time = previous_start
        ctx.chord_duration = max(0.0, timestamp - ctx.chord_start_time)
        if ctx.chord not in ("", "--", "UNKNOWN", "N"):
            symbol = parse_chord(ctx.chord, self._chart.key)
            ctx.chord_root = symbol.root
            ctx.chord_quality = symbol.quality
            ctx.bass_note = symbol.bass_note or symbol.root
            ctx.inversion = "slash" if symbol.bass_note and symbol.bass_note != symbol.root else "root"
        ctx.current_section = state.current_section
        ctx.section_progress = position.section_progress
        ctx.structure_confidence = position.confidence
        ctx.predicted_next_section = state.next_section
        if detected_key and detected_key != "--":
            ctx.key = detected_key
        if update_structure:
            # Históricos DSP permanecem no AudioAnalyzer; estrutura aprende do contexto efetivo.
            self._structure_analyzer.update_online(
                ctx, None, None, self._clock,
                chart_position=position if self._chart.sections else None)
        self._update_follow_confidence(ctx, state, timestamp)
        ctx.sync_aliases()
        return state

    def _update_follow_confidence(self, ctx: MusicalContext, state: FusedMusicalState,
                                  timestamp: float) -> None:
        """Publica uma decisão global sem substituir as métricas originais."""
        elapsed = max(0.0, timestamp - self._last_follow_timestamp) if self._last_follow_timestamp is not None else 0.0
        self._last_follow_timestamp = timestamp
        stable = (self._performance_state == PerformanceState.PLAYING and
                  ctx.tracking_state != TrackingState.LOST.value and
                  not state.is_discrepancy)
        target = 0.95 if stable else 0.30
        alpha = min(0.45, 0.12 + elapsed * 0.35)
        self._follow_stability += (target - self._follow_stability) * alpha
        ctx.recent_stability = max(0.0, min(1.0, self._follow_stability))
        # Transporte/cifra sem uma fonte observável é ensaio determinístico, não
        # evidência de acompanhamento ruim. Assim que chega áudio real, as
        # confianças medidas voltam a comandar a classificação.
        tempo = ctx.tempo_confidence
        phase = ctx.phase_confidence
        if not self._follow_observations_available:
            tempo = max(tempo, 0.70)
            phase = max(phase, 0.70)
        follow = calculate_follow_confidence(
            tempo=tempo, phase=phase,
            position=ctx.position_confidence, harmonic=ctx.chord_confidence,
            chart_alignment=ctx.chart_alignment_confidence,
            stability=ctx.recent_stability)
        ctx.confidence = follow.score
        ctx.follow_confidence_level = follow.level
        ctx.refresh_total_latency()

    def get_follow_diagnostics(self) -> Dict[str, Any]:
        ctx = self._context
        return {
            "follow_confidence": ctx.follow_confidence,
            "follow_level": ctx.follow_confidence_level,
            "tempo_confidence": ctx.tempo_confidence,
            "phase_confidence": ctx.phase_confidence,
            "position_confidence": ctx.position_confidence,
            "chord_confidence": ctx.chord_confidence,
            "capture_latency": ctx.capture_latency,
            "analysis_latency": ctx.analysis_latency,
            "scheduling_latency": ctx.scheduling_latency,
            "output_latency": ctx.output_latency,
            "total_latency": ctx.total_estimated_latency,
        }

    def format_follow_diagnostics(self) -> str:
        d = self.get_follow_diagnostics()
        return (f"FOLLOW CONFIDENCE: {d['follow_level']} ({d['follow_confidence']:.2f}) | "
                f"TEMPO CONFIDENCE: {d['tempo_confidence']:.2f} | "
                f"PHASE CONFIDENCE: {d['phase_confidence']:.2f} | "
                f"POSITION CONFIDENCE: {d['position_confidence']:.2f} | "
                f"CHORD CONFIDENCE: {d['chord_confidence']:.2f} | "
                f"CAPTURE LATENCY: {d['capture_latency']:.1f} ms | "
                f"ANALYSIS LATENCY: {d['analysis_latency']:.1f} ms | "
                f"SCHEDULER LATENCY: {d['scheduling_latency']:.1f} ms | "
                f"OUTPUT LATENCY: {d['output_latency']:.1f} ms | "
                f"TOTAL ESTIMATED LATENCY: {d['total_latency']:.1f} ms")

    def get_position_diagnostics(self) -> Dict[str, Any]:
        """Diagnóstico derivado; não mantém outra posição independente."""
        return {
            "clock_bar": self.clock_bar, "clock_beat": self.clock_beat,
            "current_bar": self.current_bar, "current_beat": self.current_beat,
            "line_index": self.chart_position.line_index,
            "section": self.chart_position.section_name, "tracking_state": self.tracking_state,
            "bar_offset": self._position_estimator.bar_offset,
            "expected_chord": self.expected_chord, "detected_chord": self.detected_chord,
            "performance_state": self.performance_state,
        }

    def format_position_diagnostics(self) -> str:
        data = self.get_position_diagnostics()
        return (
            f"CLOCK: Bar {data['clock_bar']} Beat {data['clock_beat']} | "
            f"MUSICAL POSITION: Bar {data['current_bar']} Beat {data['current_beat']} | "
            f"CHART LINE: {data['line_index']} | SECTION: {data['section']} | "
            f"TRACKING: {data['tracking_state']} | BAR OFFSET: {data['bar_offset']:+d} | "
            f"EXPECTED: {data['expected_chord']} | DETECTED: {data['detected_chord']}"
            f" | PERFORMANCE: {data['performance_state']}"
        )

    def get_tempo_diagnostics(self) -> Dict[str, Any]:
        return {
            "initial_bpm": self._clock.initial_bpm,
            "current_bpm": self._clock.current_bpm,
            "target_bpm": self._clock.target_bpm,
            "tempo_confidence": self._clock.tempo_confidence,
            "beat_phase": self._clock.beat_phase,
            "phase_error_ms": self._clock.phase_error_ms,
            "tracking_state": self._clock.tracking_state,
        }

    def format_tempo_diagnostics(self) -> str:
        d = self.get_tempo_diagnostics()
        return (f"INITIAL BPM: {d['initial_bpm']:.1f} | CURRENT BPM: {d['current_bpm']:.1f} | "
                f"TARGET BPM: {d['target_bpm']:.1f} | TEMPO CONF: {d['tempo_confidence']:.2f} | "
                f"BEAT PHASE: {d['beat_phase']:.2f} | PHASE ERROR: {d['phase_error_ms']:+.0f} ms | "
                f"STATE: {d['tracking_state']}")

    def format_harmonic_rhythm_diagnostics(self) -> str:
        """Snapshot para depurar cifra, detector estabilizado e duração em beats."""
        ctx = self._context
        return (f"SECTION: {ctx.current_section} | CURRENT: {ctx.chord} | "
                f"ELAPSED: {ctx.current_chord_elapsed_beats:.2f} beats | "
                f"EXPECTED DURATION: {ctx.expected_chord_duration_beats:.2f} beats | "
                f"BEATS UNTIL CHANGE: {ctx.beats_until_chord_change:.2f} | "
                f"NEXT: {ctx.next_expected_chord} | DETECTED: {ctx.smoothed_detected_chord} | "
                f"CHORD CONF: {ctx.chord_confidence:.2f} | DURATION CONF: {ctx.duration_confidence:.2f} | "
                f"POSITION CONF: {ctx.position_confidence:.2f} | PATTERN: {ctx.harmonic_rhythm_pattern} | "
                f"OBSERVATIONS: {ctx.harmonic_rhythm_observations}")

    def get_harmonic_rhythm_timeline(self) -> List[Dict[str, Any]]:
        """Log consultável de eventos fechados; não é uma segunda posição musical."""
        return [{
            "start_beat": round(event.start_beat, 2), "section": event.section_key,
            "detected": event.symbol, "raw_duration_beats": round(event.raw_duration_beats, 2),
            "duration_beats": event.quantized_duration_beats,
            "confidence": round(event.confidence, 2),
        } for event in self._harmonic_rhythm.timeline]

    # ============================================================
    # Navegação Manual (Ensaio / Rehearsal Mode)
    # ============================================================
    def transpose_to(self, target_key: str) -> ChordChart:
        """Transpõe a cifra ativa para ``target_key`` e atualiza tudo que dela depende.

        Reconstrói o alinhamento e o estimador, atualiza o contexto e persiste na canção
        (chart_data, chart_text e key), preservando a posição atual de reprodução.
        """
        self._chart.transpose_to_key(target_key)
        self._alignment = ChartAlignment(self._chart)
        self._position_estimator.set_alignment(self._alignment)
        self._context.key = target_key

        # Persiste na canção para que a mudança sobreviva à sessão
        self._song.chart_data = self._chart.to_dict()
        if self._song.chart_text:
            self._song.chart_text = self._chart.raw_text
        self._song.key = target_key

        # Reavalia a posição atual sob a nova cifra
        self._position_generation += 1
        self._current_chart_pos = self._position_estimator.refresh_position()
        self._publish_position()
        return self._chart

    def seek_to_bar(self, bar: int) -> ChartPosition:
        """Salta a reprodução/estudo diretamente para um compasso específico."""
        self._position_generation += 1
        self._current_chart_pos = self._position_estimator.seek_to_bar(bar)
        self._publish_position()
        return self._current_chart_pos

    def next_bar(self) -> ChartPosition:
        return self.seek_to_bar(self.current_bar + 1)

    def prev_bar(self) -> ChartPosition:
        return self.seek_to_bar(max(1, self.current_bar - 1))

    def next_section_jump(self) -> ChartPosition:
        """Avança para o primeiro compasso da próxima seção da cifra."""
        curr_sec_name = self._current_chart_pos.section_name
        for bar in range(self.current_bar + 1, self._alignment.total_bars + 1):
            pos = self._alignment.get_position_at(bar)
            if pos.section_name != curr_sec_name:
                return self.seek_to_bar(bar)
        return self._current_chart_pos

    def prev_section(self) -> ChartPosition:
        """Retrocede para o início da seção atual ou da seção anterior."""
        curr_sec_name = self._current_chart_pos.section_name
        # Primeiro tenta achar o início da seção atual
        first_bar_of_curr = self.current_bar
        for bar in range(self.current_bar - 1, 0, -1):
            pos = self._alignment.get_position_at(bar)
            if pos.section_name == curr_sec_name:
                first_bar_of_curr = bar
            else:
                break
        
        if self.current_bar > first_bar_of_curr:
            return self.seek_to_bar(first_bar_of_curr)

        # Se já estava no início da atual, vai para o início da anterior
        for bar in range(first_bar_of_curr - 1, 0, -1):
            pos = self._alignment.get_position_at(bar)
            if pos.section_name != curr_sec_name:
                # Acha o primeiro compasso daquela seção anterior
                prev_sec_name = pos.section_name
                start_prev = bar
                for b in range(bar - 1, 0, -1):
                    if self._alignment.get_position_at(b).section_name == prev_sec_name:
                        start_prev = b
                    else:
                        break
                return self.seek_to_bar(start_prev)
        return self.seek_to_bar(1)

    def next_chord_jump(self) -> ChartPosition:
        """Salta para o próximo acorde na progressão da cifra."""
        curr_chord = self._current_chart_pos.current_chord
        for bar in range(self.current_bar + 1, self._alignment.total_bars + 1):
            pos = self._alignment.get_position_at(bar)
            if pos.current_chord != curr_chord:
                return self.seek_to_bar(bar)
        return self._current_chart_pos

    def prev_chord_jump(self) -> ChartPosition:
        """Retrocede para o acorde imediatamente anterior."""
        curr_chord = self._current_chart_pos.current_chord
        for bar in range(self.current_bar - 1, 0, -1):
            pos = self._alignment.get_position_at(bar)
            if pos.current_chord != curr_chord:
                return self.seek_to_bar(bar)
        return self.seek_to_bar(1)

    def update_chart_text(self, new_chart_text: str) -> None:
        """Atualiza a cifra em tempo de execução, re-parseando e sincronizando o alinhamento."""
        self._song.chart_text = new_chart_text
        parsed = ChartParser.parse(
            text=new_chart_text,
            default_key=self._song.key,
            default_bpm=self._song.bpm,
            default_meter=self._song.meter,
            title=self._song.title,
            artist=self._song.artist
        )
        self._song.chart_data = parsed.to_dict()
        self._song.lyrics_text = "\n".join(
            l.get("text", "") if isinstance(l, dict) else getattr(l, "text", str(l))
            for s in self._song.chart_data.get("sections", []) for l in s.get("lyrics", [])
        )
        self._song.update_timestamp()
        self._chart = parsed
        self._song.key = parsed.key
        self._song.bpm = parsed.bpm
        self._song.meter = parsed.meter
        self._song.time_signature = parsed.meter
        self._clock.bpm = self._song.performance_settings.bpm_override or parsed.bpm
        self._clock.set_meter(self._song.performance_settings.meter_override or parsed.meter)
        self._context.bpm = self._clock.bpm
        self._context.meter = self._clock.meter
        self._context.key = parsed.key
        self._alignment = ChartAlignment(self._chart)
        self._position_estimator.set_alignment(self._alignment)
        self._position_estimator.set_capo(self._chart.capo_semitones)
        self._position_generation += 1
        self._current_chart_pos = self._position_estimator.refresh_position()
        self._publish_position()
