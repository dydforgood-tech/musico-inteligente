"""Histórico Temporal e Estabilizador de Acordes (v0.1-C e Contexto Temporal).

Responsável por:
1. Filtrar oscilações e ruídos transitórios do detector de acordes (Debounce / Histerese).
2. Registrar eventos discretos de acordes confirmados (ChordEvent) com start_time, end_time, duration e confidence.
3. Não criar eventos repetidos ou centenas de entradas para um mesmo acorde mantido.
4. Fornecer informações de acorde anterior (previous_chord) e duração do acorde atual (chord_duration).
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict
import collections
import numpy as np

from app.music.constants import (
    MIN_CHORD_CONFIDENCE,
    MIN_CHORD_STABILITY_TIME,
    CHORD_CHANGE_CONFIRMATION_TIME,
    MAX_CHORD_HISTORY_ENTRIES,
)


@dataclass
class ChordEvent:
    """Evento consolidado que representa um acorde que foi tocado com duração definida."""
    chord: str                          # Cifra completa (ex: "C", "G/B", "Am")
    start_time: float                   # Início exato do acorde em segundos
    end_time: float                     # Fim do acorde em segundos
    duration: float                     # Duração total em segundos (end_time - start_time)
    confidence: float                   # Grau médio de confiança (0.0 a 1.0)
    root: str = "--"                    # Fundamental (ex: "C")
    quality: str = "--"                 # Qualidade (ex: "major", "minor")
    bass_note: str = "--"               # Nota no baixo (ex: "B" para "G/B")
    inversion: str = "root"             # Tipo de inversão ("root", "first", "second")

    @property
    def symbol(self) -> str:
        """Alias para o símbolo da cifra."""
        return self.chord

    def to_dict(self) -> Dict:
        return {
            "time_str": self.format_timestamp(self.start_time),
            "chord": self.chord,
            "duration_str": f"{self.duration:.2f} s",
            "confidence_str": f"{int(self.confidence * 100)}%",
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "confidence": self.confidence,
        }

    @staticmethod
    def format_timestamp(seconds: float) -> str:
        if seconds < 0 or np.isnan(seconds):
            return "00:00.00"
        m = int(seconds // 60)
        s = seconds % 60
        return f"{m:02d}:{s:05.2f}"


class ChordHistory:
    """Repositório ordenado de eventos discretos de acordes confirmados.
    
    Por padrão armazena o histórico completo da faixa sem descarte (max_entries=None).
    """

    def __init__(self, max_entries: Optional[int] = MAX_CHORD_HISTORY_ENTRIES):
        self._max_entries = max_entries
        self._events: collections.deque = collections.deque(maxlen=max_entries)
        self._version: int = 0

    @property
    def version(self) -> int:
        """Número de versão incrementado a cada novo evento confirmado."""
        return self._version

    def add_event(self, event: ChordEvent) -> None:
        """Adiciona um evento de acorde confirmado ao histórico."""
        if event.chord == "--" or event.duration < 0.05:
            return
        self._events.append(event)
        self._version += 1

    def record(self, timestamp: float, chord: str, confidence: float = 0.8) -> None:
        """Adiciona evento simples por timestamp e símbolo de acorde."""
        if chord == "--":
            return
        self._events.append(ChordEvent(
            chord=chord,
            start_time=timestamp,
            end_time=timestamp + 0.5,
            duration=0.5,
            confidence=confidence
        ))
        self._version += 1


    def update(self, chord) -> None:
        """Método de conveniência para compatibilidade com a interface anterior."""
        if not hasattr(chord, "symbol") or chord.symbol == "--":
            return
        # Se o último evento for o mesmo acorde, incrementa duração
        if self._events and self._events[-1].chord == chord.symbol:
            dt = max(0.0, getattr(chord, "timestamp", 0.0) - self._events[-1].start_time)
            self._events[-1].duration = dt
            self._events[-1].end_time = getattr(chord, "timestamp", 0.0)
        else:
            ts = getattr(chord, "timestamp", 0.0)
            self._events.append(ChordEvent(
                chord=chord.symbol,
                start_time=ts,
                end_time=ts,
                duration=0.05,
                confidence=getattr(chord, "confidence", 0.8),
                root=getattr(chord, "root", "--"),
                quality=getattr(chord, "quality", "--"),
                bass_note=getattr(chord, "bass_note", "--"),
                inversion=getattr(chord, "inversion", "root")
            ))

    def get_events(self) -> List[ChordEvent]:
        """Retorna todos os eventos registrados."""
        return list(self._events)

    def get_recent(self, count: int = 8) -> List[ChordEvent]:
        """Retorna os N eventos mais recentes."""
        return list(self._events)[-count:]

    def get_summary_text(self) -> str:
        """Formata resumo da progressão harmônica recente para exibição na UI."""
        if not self._events:
            return "Aguardando progressão de acordes..."
        items = [f"{e.chord} ({e.duration:.1f}s)" for e in list(self._events)[-5:]]
        return " → ".join(items)

    def clear(self) -> None:
        """Limpa todo o histórico."""
        self._events.clear()

    def __len__(self) -> int:
        return len(self._events)


class ChordStabilizer:
    """Motor de estabilização temporal para detecção de acordes.
    
    Equilibra estabilidade (rejeição de falsos positivos transitórios)
    e velocidade de resposta musical.
    """

    def __init__(
        self,
        min_confidence: float = MIN_CHORD_CONFIDENCE,
        min_stability_time: float = MIN_CHORD_STABILITY_TIME,
        confirmation_time: float = CHORD_CHANGE_CONFIRMATION_TIME,
        history: Optional[ChordHistory] = None
    ):
        self._min_confidence = min_confidence
        self._min_stability_time = min_stability_time
        self._confirmation_time = confirmation_time
        self._history = history if history is not None else ChordHistory()

        # Estado atual confirmado
        self._current_chord: str = "--"
        self._chord_start_time: float = 0.0
        self._chord_duration: float = 0.0
        self._chord_root: str = "--"
        self._chord_quality: str = "--"
        self._bass_note: str = "--"
        self._inversion: str = "root"
        self._detected_notes: List[str] = []
        self._chord_confidence: float = 0.0

        # Acorde anterior confirmado
        self._previous_chord: str = "--"

        # Acumulador de confiança do acorde vigente
        self._confidences: List[float] = []

        # Estado do acorde candidato (para debounce / confirmação de transição)
        self._candidate_chord: Optional[str] = None
        self._candidate_start_time: float = 0.0
        self._candidate_raw_event: Optional[Dict] = None
        self._candidate_confidences: List[float] = []
        self._candidate_frame_count: int = 0

        # Rastreamento de saltos no tempo (Seek)
        self._last_timestamp: float = -1.0

    @property
    def history(self) -> ChordHistory:
        return self._history

    @property
    def current_chord(self) -> str:
        return self._current_chord

    @property
    def previous_chord(self) -> str:
        return self._previous_chord

    @property
    def chord_start_time(self) -> float:
        return self._chord_start_time

    @property
    def chord_duration(self) -> float:
        return self._chord_duration

    @property
    def chord_confidence(self) -> float:
        return self._chord_confidence

    @property
    def chord_root(self) -> str:
        return self._chord_root

    @property
    def chord_quality(self) -> str:
        return self._chord_quality

    @property
    def bass_note(self) -> str:
        return self._bass_note

    @property
    def inversion(self) -> str:
        return self._inversion

    @property
    def detected_notes(self) -> List[str]:
        return self._detected_notes

    @property
    def stabilization_delay_ms(self) -> float:
        """Tempo de atraso de estabilização configurado em milissegundos."""
        return self._confirmation_time * 1000.0

    def reset(self) -> None:
        """Reinicia o estabilizador e limpa o histórico."""
        self._finalize_current_event_if_active()
        self._current_chord = "--"
        self._chord_start_time = 0.0
        self._chord_duration = 0.0
        self._chord_root = "--"
        self._chord_quality = "--"
        self._bass_note = "--"
        self._inversion = "root"
        self._detected_notes = []
        self._chord_confidence = 0.0
        self._previous_chord = "--"
        self._confidences.clear()
        self._candidate_chord = None
        self._candidate_start_time = 0.0
        self._candidate_raw_event = None
        self._candidate_confidences.clear()
        self._candidate_frame_count = 0
        self._last_timestamp = -1.0
        self._history.clear()

    def _finalize_current_event_if_active(self, end_time: Optional[float] = None) -> None:
        """Encerra o acorde atual e o adiciona ao histórico se tiver duração mínima."""
        if self._current_chord != "--" and self._confidences:
            actual_end = end_time if end_time is not None else (self._chord_start_time + self._chord_duration)
            dur = max(0.01, actual_end - self._chord_start_time)
            if dur >= self._min_stability_time:
                avg_conf = float(np.mean(self._confidences))
                event = ChordEvent(
                    chord=self._current_chord,
                    start_time=self._chord_start_time,
                    end_time=actual_end,
                    duration=dur,
                    confidence=avg_conf,
                    root=self._chord_root,
                    quality=self._chord_quality,
                    bass_note=self._bass_note,
                    inversion=self._inversion
                )
                self._history.add_event(event)

    def process(
        self,
        raw_symbol: str,
        confidence: float,
        timestamp: float,
        root: str = "--",
        quality: str = "--",
        bass_note: str = "--",
        inversion: str = "root",
        detected_notes: Optional[List[str]] = None
    ) -> str:
        """Processa a detecção instantânea do frame aplicando estabilização temporal."""
        timestamp = max(0.0, float(timestamp))
        notes = detected_notes or []

        # Detecção de Seek brusco (salto para trás ou para frente > 1.5s)
        if self._last_timestamp >= 0.0 and abs(timestamp - self._last_timestamp) > 1.5:
            self._finalize_current_event_if_active(self._last_timestamp)
            self._current_chord = "--"
            self._confidences.clear()
            self._candidate_chord = None

        self._last_timestamp = timestamp

        # Se a confiança for insuficiente ou sem acorde detectado
        if confidence < self._min_confidence or raw_symbol == "--":
            if self._current_chord != "--":
                # Acumular duração no acorde vigente (rejeição a falhas de 1 frame de silêncio)
                self._chord_duration = max(0.0, timestamp - self._chord_start_time)
            return self._current_chord

        # Caso 1: Primeiro acorde inicial detectado
        if self._current_chord == "--":
            self._current_chord = raw_symbol
            self._chord_start_time = timestamp
            self._chord_duration = 0.0
            self._chord_root = root
            self._chord_quality = quality
            self._bass_note = bass_note
            self._inversion = inversion
            self._detected_notes = list(notes)
            self._chord_confidence = confidence
            self._confidences = [confidence]
            self._candidate_chord = None
            return self._current_chord

        # Caso 2: O acorde detectado continua sendo o mesmo confirmado
        if raw_symbol == self._current_chord:
            self._chord_duration = max(0.0, timestamp - self._chord_start_time)
            self._confidences.append(confidence)
            # Atualização suave da confiança média
            self._chord_confidence = (self._chord_confidence * 0.85) + (confidence * 0.15)
            self._chord_root = root
            self._chord_quality = quality
            self._bass_note = bass_note
            self._inversion = inversion
            if notes:
                self._detected_notes = list(notes)
            # Descartar qualquer candidato anterior, pois o acorde original se reafirmou
            self._candidate_chord = None
            self._candidate_frame_count = 0
            return self._current_chord

        # Caso 3: Acorde detectado difere do atual (início ou continuação de candidato)
        if self._candidate_chord != raw_symbol:
            # Novo candidato a mudança
            self._candidate_chord = raw_symbol
            self._candidate_start_time = timestamp
            self._candidate_confidences = [confidence]
            self._candidate_frame_count = 1
            self._candidate_raw_event = {
                "root": root,
                "quality": quality,
                "bass_note": bass_note,
                "inversion": inversion,
                "notes": list(notes)
            }
            # O acorde atual ainda permanece vigente enquanto o candidato não for confirmado
            self._chord_duration = max(0.0, timestamp - self._chord_start_time)
            return self._current_chord

        # O candidato é o mesmo do frame anterior: acumula persistência
        self._candidate_confidences.append(confidence)
        self._candidate_frame_count += 1
        candidate_duration = timestamp - self._candidate_start_time

        # Critério de confirmação:
        # Atingiu o tempo de confirmação temporal (ex: 200 ms) OU persistiu por >= 3 frames consecutivos
        is_confirmed = (
            candidate_duration >= self._confirmation_time or
            (self._candidate_frame_count >= 3 and candidate_duration >= (self._confirmation_time * 0.5))
        )

        if is_confirmed:
            # Transição confirmada!
            # 1. Finalizar o acorde anterior no momento em que o novo acorde realmente começou
            self._finalize_current_event_if_active(self._candidate_start_time)

            # 2. Registrar a transição para previous_chord
            self._previous_chord = self._current_chord

            # 3. Promover o candidato a acorde oficial
            self._current_chord = self._candidate_chord
            self._chord_start_time = self._candidate_start_time
            self._chord_duration = max(0.0, timestamp - self._candidate_start_time)
            self._confidences = list(self._candidate_confidences)
            self._chord_confidence = float(np.mean(self._candidate_confidences))

            if self._candidate_raw_event:
                self._chord_root = self._candidate_raw_event["root"]
                self._chord_quality = self._candidate_raw_event["quality"]
                self._bass_note = self._candidate_raw_event["bass_note"]
                self._inversion = self._candidate_raw_event["inversion"]
                self._detected_notes = self._candidate_raw_event["notes"]

            # Limpar candidato
            self._candidate_chord = None
            self._candidate_frame_count = 0
            self._candidate_raw_event = None

            return self._current_chord

        # Ainda em processo de avaliação do candidato: manter acorde atual ativo
        self._chord_duration = max(0.0, timestamp - self._chord_start_time)
        return self._current_chord
