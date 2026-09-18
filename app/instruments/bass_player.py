"""Baixista Virtual Autônomo (BassPlayer v0.2).

Orquestrador central do baixista virtual:
1. Consome o MusicalContext (acorde estabilizado, tom, BPM, beat, compasso).
2. Coordena BassDecisionEngine (decisão harmônica de notas).
3. Coordena BassPatternGenerator (estruturação do padrão rítmico).
4. Coordena BassPerformanceEngine (sincronização de timing e articulação).
5. Dispara notas em tempo real no BassSynthesizer para mixagem com o áudio do instrumento real.
6. Mantém histórico de decisões para inspeção em tempo real e modo debug na interface gráfica.
"""

from typing import Optional, List, Dict, Tuple
import collections

from app.instruments.base import VirtualInstrument
from app.instruments.bass_model import BassNoteEvent, BassDecision, BassPatternType
from app.instruments.bass_decision import BassDecisionEngine
from app.instruments.bass_pattern import BassPatternGenerator
from app.instruments.bass_performance import BassPerformanceEngine
from app.instruments.bass_synthesizer import BassSynthesizer
from app.music.musical_context import MusicalContext
from app.music.constants import BASS_DEFAULT_VOLUME


class BassPlayer(VirtualInstrument):
    """Baixista Virtual Inteligente que acompanha a harmonia e o andamento em tempo real."""

    def __init__(
        self,
        sample_rate: int = 44100,
        volume: float = BASS_DEFAULT_VOLUME,
        pattern: BassPatternType = BassPatternType.AUTO
    ):
        self._enabled: bool = True
        self._pattern_override: BassPatternType = pattern

        # Módulos especializados com responsabilidades separadas
        self._decision_engine = BassDecisionEngine()
        self._pattern_generator = BassPatternGenerator()
        self._performance_engine = BassPerformanceEngine()
        self._synthesizer = BassSynthesizer(sample_rate=sample_rate, volume=volume)

        # Estado instantâneo de execução
        self._current_decision: Optional[BassDecision] = None
        self._current_event: Optional[BassNoteEvent] = None
        self._last_triggered_beat: Optional[Tuple[int, int]] = None
        self._last_chord: str = "--"

        # Histórico recente para modo debug e UI (ring-buffer)
        self._recent_events: collections.deque = collections.deque(maxlen=30)
        self._last_context: Optional[MusicalContext] = None

    @property
    def instrument_name(self) -> str:
        return "Virtual Bassist"

    @property
    def synthesizer(self) -> BassSynthesizer:
        return self._synthesizer

    @property
    def decision_engine(self) -> BassDecisionEngine:
        return self._decision_engine

    @property
    def pattern_generator(self) -> BassPatternGenerator:
        return self._pattern_generator

    @property
    def performance_engine(self) -> BassPerformanceEngine:
        return self._performance_engine

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, val: bool) -> None:
        self._enabled = bool(val)
        self._synthesizer.enabled = self._enabled
        if not self._enabled:
            self._synthesizer.clear()

    @property
    def pattern(self) -> BassPatternType:
        return self._pattern_override

    @pattern.setter
    def pattern(self, p: BassPatternType) -> None:
        self._pattern_override = p

    @property
    def volume(self) -> float:
        return self._synthesizer.volume

    @volume.setter
    def volume(self, v: float) -> None:
        self._synthesizer.volume = v

    @property
    def current_decision(self) -> Optional[BassDecision]:
        return self._current_decision

    @property
    def current_event(self) -> Optional[BassNoteEvent]:
        return self._current_event

    def get_recent_events(self) -> List[BassNoteEvent]:
        """Retorna lista dos eventos de baixo recentes."""
        return list(self._recent_events)

    def get_recent_decisions(self) -> List[Dict]:
        """Retorna representação serializada para exibição na UI e log de debug."""
        return [ev.to_dict() for ev in self._recent_events]

    def get_prediction(self, current_bar: int = 1, current_beat: int = 1) -> Tuple[str, str]:
        """Calcula a previsão da próxima nota a ser executada pelo baixista."""
        if not self._enabled or self._current_decision is None:
            return ("--", "Aguardando harmonia")

        dec = self._current_decision
        if dec.pattern_type == BassPatternType.SUSTAINED:
            return (f"{dec.root_note_name} (Sustentada)", f"Sustenta até Compasso {current_bar + 1}")

        # Padrões com múltiplos passos por compasso
        steps = self._pattern_generator.generate_pattern(dec, beats=4, chord_duration_beats=4.0)
        for step in steps:
            b, note_name, _, reason = step
            if b > current_beat:
                return (f"{note_name} ({reason})", f"Tempo {b} do Compasso {current_bar}")

        # Se passou de todos os tempos deste compasso, prevê o tempo 1 do próximo compasso
        if steps:
            first_step = steps[0]
            pred_text = f"{first_step[1]} ({first_step[3]})"
            target_text = f"Tempo {first_step[0]} do Compasso {current_bar + 1}"
            
            # Enriquece a predição se houver mudança de seção prevista com alta certeza
            if (self._last_context and 
                self._last_context.predicted_next_section not in ("UNKNOWN", "--") and 
                self._last_context.bars_until_change <= 1 and 
                self._last_context.prediction_confidence >= 0.70):
                next_sec = self._last_context.predicted_next_section
                target_text += f" [-> {next_sec}]"

            return (pred_text, target_text)

        return ("--", "Aguardando próximo compasso")

    def reset(self) -> None:
        """Reinicia estado do baixista, histórico e sintetizador."""
        self._decision_engine.reset()
        self._synthesizer.clear()
        self._current_decision = None
        self._current_event = None
        self._last_triggered_beat = None
        self._last_chord = "--"
        self._last_context = None
        self._recent_events.clear()

    def generate_bar_events(self, context: MusicalContext, bar: int, bar_start_time: float) -> List[BassNoteEvent]:
        """Gera antecipadamente todos os eventos do compasso para o acorde do contexto."""
        if context.chord == "--":
            return []

        bpm = context.bpm if context.bpm > 0 else 120.0
        beat_dur = 60.0 / bpm
        beats = 4

        # 1. Decisão harmônica
        decision = self._decision_engine.decide(context, pattern_override=self._pattern_override)
        self._current_decision = decision

        # 2. Geração do padrão
        pattern_steps = self._pattern_generator.generate_pattern(
            decision,
            beats=beats,
            chord_duration_beats=4.0
        )

        # 3. Performance sincronizada com o MusicalClock
        events = self._performance_engine.create_events_for_bar(
            decision=decision,
            pattern_steps=pattern_steps,
            bar=bar,
            bar_start_time=bar_start_time,
            beat_duration=beat_dur,
            total_beats=beats
        )
        return events

    def on_musical_context(self, context: MusicalContext) -> Optional[BassNoteEvent]:
        """Processa o contexto musical em tempo real e dispara notas de baixo no instante do tempo musical."""
        if not self._enabled or context.chord == "--":
            return None

        self._last_context = context

        # Rastreia compasso e tempo atuais
        bar = max(1, context.bar)
        beat = max(1, context.beat)
        beat_key = (bar, beat)

        # Evita disparos duplicados dentro do mesmo tempo musical
        if self._last_triggered_beat == beat_key:
            return None

        self._last_triggered_beat = beat_key

        bpm = context.bpm if context.bpm > 0 else 120.0
        beat_dur = 60.0 / bpm

        # 1. Decisão harmônica
        decision = self._decision_engine.decide(context, pattern_override=self._pattern_override)
        self._current_decision = decision

        # 2. Padrão rítmico
        chord_beats = max(1.0, (context.chord_duration / beat_dur)) if context.chord_duration > 0 else 4.0
        pattern_steps = self._pattern_generator.generate_pattern(
            decision,
            beats=4,
            chord_duration_beats=chord_beats
        )

        # 3. Localiza a nota correspondente a este tempo musical específico
        matched_step = None
        for step in pattern_steps:
            if step[0] == beat:
                matched_step = step
                break

        if matched_step is None:
            return None

        _, note_name, midi_note, reason = matched_step

        # Enriquece razão com antecipação estrutural preditiva
        if (beat == 4 and 
            context.prediction_confidence >= 0.75 and 
            context.bars_until_change <= 1 and 
            context.predicted_next_section not in ("UNKNOWN", "--")):
            reason = f"{reason} [Antecipa {context.predicted_next_section}]"

        # Duração e articulação
        is_sustained = (decision.pattern_type == BassPatternType.SUSTAINED)
        duration = (beat_dur * 4.0 * 0.95) if is_sustained else (beat_dur * 0.85)

        # Dinâmica
        velocity = 100 if beat == 1 else 92 if beat == 3 else 85

        ev = BassNoteEvent(
            note=note_name,
            midi_note=midi_note,
            start_time=round(context.timestamp, 4),
            duration=round(duration, 4),
            velocity=velocity,
            beat=beat,
            bar=bar,
            confidence=decision.confidence,
            reason=reason
        )

        # Dispara som no sintetizador local
        self._synthesizer.trigger_note(midi_note, velocity, duration)

        # Registra no histórico recente
        self._recent_events.append(ev)
        self._current_event = ev

        return ev
