"""Sessão de Execução Musical da Música Ativa (SongSession v0.4).

Gerencia exclusivamente o estado de execução temporário e volátil da música selecionada:
MusicalContext, MusicalClock, ChartAlignment, ChartAudioFusion e VirtualPlayers.
Garante isolamento hermético entre trocas de músicas para evitar memory leaks.
"""

from typing import Optional, Dict, Any, List
import copy
from dataclasses import fields

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


class SongSession:
    """Representa a sessão ativa de execução e reprodução de uma Song.
    
    Toda a interface gráfica e os instrumentos virtuais devem consultar esta classe
    para obter a verdade sobre o momento musical presente.
    """

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
        self._current_chart_pos: ChartPosition = self._position_estimator.refresh_position()
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
        self._clock.bpm = self._song.performance_settings.bpm_override or self._song.bpm
        self._clock.reset()
        self._context.reset()
        self._context.bpm = self._clock.bpm
        self._context.meter = self._clock.meter
        self._context.key = self._song.key
        self._fusion.reset()
        self._structure_analyzer.reset()
        self._position_estimator.reset()
        self._current_chart_pos = self._position_estimator.refresh_position()
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
        manual_bpm = self._song.performance_settings.bpm_override
        initial_observation = self._clock.elapsed_time == 0.0 and self._clock.tempo_confidence == 0.0
        bpm_input = (manual_bpm or
                     (detected_bpm if detected_bpm > 0 and
                      (tempo_result is None or initial_observation) else None))
        self._clock.update(
            timestamp, bpm=bpm_input,
            beat_timestamp=tempo_result.beat_timestamp if tempo_result else None,
            observation_confidence=tempo_result.confidence if tempo_result else 0.0)
        self._current_chart_pos = self._position_estimator.update(
            timestamp=timestamp, detected_chord=detected_chord,
            detected_confidence=detected_confidence, detected_key=detected_key,
            detected_note=source_context.note if source_context is not None else "--",
            note_confidence=source_context.note_confidence if source_context is not None else 0.0)
        return self._publish_position(detected_chord, detected_confidence, detected_key, source_context)

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
        ctx.chord_confidence = state.confidence
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
        ctx.sync_aliases()
        return state

    def get_position_diagnostics(self) -> Dict[str, Any]:
        """Diagnóstico derivado; não mantém outra posição independente."""
        return {
            "clock_bar": self.clock_bar, "clock_beat": self.clock_beat,
            "current_bar": self.current_bar, "current_beat": self.current_beat,
            "line_index": self.chart_position.line_index,
            "section": self.chart_position.section_name, "tracking_state": self.tracking_state,
            "bar_offset": self._position_estimator.bar_offset,
            "expected_chord": self.expected_chord, "detected_chord": self.detected_chord,
        }

    def format_position_diagnostics(self) -> str:
        data = self.get_position_diagnostics()
        return (
            f"CLOCK: Bar {data['clock_bar']} Beat {data['clock_beat']} | "
            f"MUSICAL POSITION: Bar {data['current_bar']} Beat {data['current_beat']} | "
            f"CHART LINE: {data['line_index']} | SECTION: {data['section']} | "
            f"TRACKING: {data['tracking_state']} | BAR OFFSET: {data['bar_offset']:+d} | "
            f"EXPECTED: {data['expected_chord']} | DETECTED: {data['detected_chord']}"
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
        self._current_chart_pos = self._position_estimator.refresh_position()
        self._publish_position()
        return self._chart

    def seek_to_bar(self, bar: int) -> ChartPosition:
        """Salta a reprodução/estudo diretamente para um compasso específico."""
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
        self._current_chart_pos = self._position_estimator.refresh_position()
        self._publish_position()
