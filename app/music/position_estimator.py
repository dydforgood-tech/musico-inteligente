"""Estimador Contínuo de Posição Musical na Cifra (PositionEstimator v0.4).

Combina múltiplas fontes de percepção e expectativas:
1. Relógio Musical (MusicalClock - tempo, BPM, compasso, beat)
2. Análise de Áudio (Pitch, Chroma, Acorde Detectado, Confiança)
3. Cifra Estruturada (ChordChart, ChartAlignment, Linhas Físicas)
4. Memória de Padrões e Estrutura (MusicStructureAnalyzer)
5. Motor de Predição (PredictionEngine)
6. Histórico Recente de Posições

Princípio Fundamental:
A ausência ou divergência de detecção de acordes NUNCA congela o avanço da posição.
O relógio musical e a estrutura garantem continuidade temporal monotônica com
recuperação contextual por janela local e correspondência de progressão.
"""

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Tuple, Any

from app.music.chart_alignment import ChartAlignment, ChartPosition
from app.music.chord_chart import ChordChart, parse_chord
from app.music.theory import note_to_pc
from app.music.musical_clock import MusicalClock
from app.analysis.music_structure_analyzer import MusicStructureAnalyzer
from app.music.prediction_engine import PredictionEngine
from app.music.transposition_tracker import TranspositionTracker


class TrackingState(str, Enum):
    """Estados operacionais do rastreamento da posição musical."""
    TRACKING = "TRACKING"       # Posição plenamente confirmada e confiável
    UNCERTAIN = "UNCERTAIN"     # Dúvida momentânea (falha breve de áudio ou variação)
    LOST = "LOST"               # Ausência prolongada de confirmação (avanço 100% temporal)
    RECOVERING = "RECOVERING"   # Candidato harmônico encontrado na vizinhança local


@dataclass
class ProgressionMatch:
    """Resultado da correspondência de uma sequência de acordes na cifra."""
    matched_chords: List[str]
    expected_chords: List[str]
    start_bar: int
    end_bar: int
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "matched_chords": list(self.matched_chords),
            "expected_chords": list(self.expected_chords),
            "start_bar": self.start_bar,
            "end_bar": self.end_bar,
            "confidence": round(self.confidence, 2),
        }


@dataclass
class EstimatedPosition:
    """Posição estimada consolidada pelo PositionEstimator."""
    bar: int
    beat: float
    line_index: int
    current_chord: str                  # Acorde efetivo utilizado
    expected_chord: str                 # Acorde nominal previsto pela cifra
    detected_chord: str                 # Acorde percebido no áudio
    predicted_chord: str                # Previsão harmônica estrutural
    section_name: str
    section_progress: float
    confidence: float                   # Confiança na posição [0.0 a 1.0]
    chord_confidence: float             # Confiança no acorde detectado [0.0 a 1.0]
    tracking_state: TrackingState
    action_note: str = ""
    timestamp: float = 0.0


class PositionEstimator:
    """Estimador contínuo de posição musical na cifra com recuperação inteligente."""

    def __init__(
        self,
        alignment: ChartAlignment,
        clock: MusicalClock,
        structure_analyzer: Optional[MusicStructureAnalyzer] = None,
        prediction_engine: Optional[PredictionEngine] = None,
        local_window_bars: int = 4
    ):
        self._alignment = alignment
        self._clock = clock
        self._structure_analyzer = structure_analyzer
        self._prediction_engine = prediction_engine
        self._local_window = local_window_bars

        # Estado instantâneo do estimador
        self._tracking_state: TrackingState = TrackingState.TRACKING
        self._position_confidence: float = 0.95
        self._last_confirmed_bar: int = 1
        self._last_confirmed_time: float = 0.0
        self._last_detected_chord: str = "--"
        self._consecutive_unknown_seconds: float = 0.0
        self._consecutive_mismatch_seconds: float = 0.0

        # Histórico recente de acordes detectados e esperados (últimos 8)
        self._recent_detected: deque = deque(maxlen=8)
        self._recent_expected: deque = deque(maxlen=8)

        # Detector de transposição/capotraste global entre a cifra e a execução
        self._transposition = TranspositionTracker()
        # Capotraste CONHECIDO (semitons): o áudio soa este tanto acima dos shapes escritos
        self._capo_semitones: int = 0

        # LOCALIZAÇÃO POR ÁUDIO: o áudio decide ONDE na cifra o músico está.
        # O relógio dá o avanço suave; este deslocamento (offset) mapeia o compasso do
        # relógio para o compasso da CIFRA. A progressão detectada reancora o offset.
        self._follow_audio: bool = True
        self._bar_offset: int = 0  # chart_bar = clock_bar - offset

        # Última posição consolidada
        self._current_estimated_position: Optional[EstimatedPosition] = None

    def reset(self) -> None:
        """Reinicia o estado interno do estimador para o início da música."""
        self._tracking_state = TrackingState.TRACKING
        self._position_confidence = 0.95
        self._last_confirmed_bar = 1
        self._last_confirmed_time = 0.0
        self._last_detected_chord = "--"
        self._consecutive_unknown_seconds = 0.0
        self._consecutive_mismatch_seconds = 0.0
        self._recent_detected.clear()
        self._recent_expected.clear()
        self._transposition.reset()
        self._bar_offset = 0
        self._current_estimated_position = None

    @property
    def tracking_state(self) -> TrackingState:
        return self._tracking_state

    @property
    def position_confidence(self) -> float:
        return self._position_confidence

    @property
    def current_estimated_position(self) -> Optional[EstimatedPosition]:
        return self._current_estimated_position

    @property
    def transposition_semitones(self) -> int:
        """Deslocamento global detectado entre a cifra e a execução (capotraste/tom)."""
        return self._transposition.semitones

    def set_alignment(self, alignment: ChartAlignment) -> None:
        """Atualiza o alinhamento estruturado da cifra."""
        self._alignment = alignment

    def set_capo(self, semitones: int) -> None:
        """Informa o capotraste (semitons) para comparar o áudio soante com os shapes escritos."""
        self._capo_semitones = max(0, int(semitones))

    def _expected_as_sounding(self, expected_chord: str) -> str:
        """Converte o acorde escrito (shape) para o som real, aplicando o capotraste."""
        if self._capo_semitones == 0 or not expected_chord or expected_chord == "--":
            return expected_chord
        from app.music.transposition import transpose_chord_symbol
        return transpose_chord_symbol(expected_chord, self._capo_semitones)

    def update(
        self,
        timestamp: float,
        detected_chord: str = "--",
        detected_confidence: float = 0.0,
        detected_key: str = "--"
    ) -> ChartPosition:
        """Atualiza a estimativa de posição: o ÁUDIO localiza onde estamos na cifra."""
        # 1. Tempo contínuo (o relógio dá o avanço suave; o áudio decide o LUGAR)
        clock_bar = max(1, self._clock.bar)
        current_beat = self._clock.beat + self._clock.beat_position
        dt = max(0.0, timestamp - self._last_confirmed_time) if self._last_confirmed_time > 0 else 0.0
        self._last_confirmed_time = timestamp

        confident_audio = bool(detected_chord) and detected_chord not in ("--", "UNKNOWN", "N") and detected_confidence >= 0.25

        # Registra o acorde detectado (buffer para localização por progressão)
        if confident_audio and (not self._recent_detected or self._recent_detected[-1] != detected_chord):
            self._recent_detected.append(detected_chord)

        # 1.5. LOCALIZAÇÃO POR ÁUDIO — encaixa a posição onde a progressão recente casa
        # na cifra, reancorando o deslocamento relógio→cifra. É isto que faz o programa
        # SEGUIR o músico em vez de apenas percorrer a cifra em ordem pelo tempo.
        localize_note = ""
        if self._follow_audio and confident_audio:
            tentative_bar = max(1, clock_bar - self._bar_offset)
            loc = self._global_localize(near_bar=tentative_bar)
            if loc is not None:
                loc_bar, loc_score, loc_len = loc
                distance = abs(loc_bar - tentative_bar)
                # Quanto maior o salto, mais forte a evidência exigida (evita saltos falsos).
                # 0.78 no salto longo permite RE-ACHAR a música por uma progressão de
                # tônicas (cada acerto só de raiz vale 0.8), mesmo com qualidades erradas.
                strong_enough = (
                    (distance <= 2 and loc_len >= 2 and loc_score >= 0.75)
                    or (distance > 2 and loc_len >= 3 and loc_score >= 0.78)
                )
                if strong_enough and distance >= 1:
                    self._bar_offset = clock_bar - loc_bar
                    localize_note = (
                        f"Localizado pelo áudio no Comp. {loc_bar} "
                        f"({loc_len} acordes, {int(loc_score * 100)}%)"
                    )

        current_bar = max(1, clock_bar - self._bar_offset)

        # Consulta posição nominal na cifra pelo compasso localizado
        nominal_pos = self._alignment.get_position_at(
            bar=current_bar,
            beat=current_beat,
            absolute_time=timestamp
        )
        expected_chord = nominal_pos.current_chord

        # Registra no buffer de histórico
        if expected_chord and expected_chord != "--":
            if not self._recent_expected or self._recent_expected[-1] != expected_chord:
                self._recent_expected.append(expected_chord)

        # Acorde previsto estruturalmente (Prediction fallback)
        predicted_chord = nominal_pos.next_chord
        pred = getattr(self._prediction_engine, "latest_prediction", None)
        if pred and getattr(pred, "predicted_next_chord", None) and pred.predicted_next_chord != "--":
            predicted_chord = pred.predicted_next_chord

        # 2. Avaliação Harmônica do Áudio
        is_unknown_audio = not detected_chord or detected_chord in ("--", "UNKNOWN", "N") or detected_confidence < 0.25

        if is_unknown_audio:
            # ----------------------------------------------------
            # CASO A: ÁUDIO DESCONHECIDO / INDETERMINADO
            # REGRA FUNDAMENTAL: NÃO PARAR. NÃO CONGELAR.
            # O Relógio Musical e a Cifra continuam avançando!
            # ----------------------------------------------------
            self._consecutive_unknown_seconds += dt
            self._consecutive_mismatch_seconds = 0.0

            # Degradação suave da confiança temporal
            if self._consecutive_unknown_seconds > 8.0:
                self._tracking_state = TrackingState.LOST
                self._position_confidence = max(0.40, self._position_confidence - (dt * 0.03))
                action_note = "Avanço temporal (Detecção ausente há > 8s - LOST)"
            elif self._consecutive_unknown_seconds > 2.5:
                self._tracking_state = TrackingState.UNCERTAIN
                self._position_confidence = max(0.65, self._position_confidence - (dt * 0.04))
                action_note = "Avanço temporal (Detecção incerta - UNCERTAIN)"
            else:
                self._tracking_state = TrackingState.TRACKING
                self._position_confidence = max(0.85, 0.95 - (self._consecutive_unknown_seconds * 0.04))
                action_note = "Avanço temporal contínuo"

            effective_chord = expected_chord if expected_chord != "--" else predicted_chord

        else:
            # Áudio com acorde detectado convincente
            self._consecutive_unknown_seconds = 0.0
            self._last_detected_chord = detected_chord

            # Comparação harmônica com o esperado
            is_match = False
            is_transposed_match = False
            root_only_match = False
            if expected_chord and expected_chord != "--":
                det_sym = parse_chord(detected_chord)

                def _same(a_sym):
                    return a_sym.is_synonym_of(det_sym) or (a_sym.root == det_sym.root and a_sym.quality == det_sym.quality)

                # O áudio pode ser reportado no domínio SOANTE (com capotraste) ou como o
                # SHAPE escrito. Aceitamos qualquer um dos dois para robustez.
                exp_sounding = self._expected_as_sounding(expected_chord)
                is_match = _same(parse_chord(exp_sounding))
                if not is_match and exp_sounding != expected_chord:
                    is_match = _same(parse_chord(expected_chord))

                # Aprende um deslocamento global RESIDUAL (afinação diferente etc.)
                self._transposition.observe(exp_sounding, detected_chord)
                if not is_match and self._transposition.matches_under_transposition(exp_sounding, detected_chord):
                    # A execução está transposta de forma coerente: confirma a posição
                    # (a cifra continua soberana; apenas deixamos de reportar falsa divergência).
                    is_match = True
                    is_transposed_match = True

                # Fallback pela TÔNICA: se a raiz bate (mas a qualidade detectada divergiu),
                # ainda consideramos a posição confirmada — o detector erra a qualidade, não
                # o baixo/tônica. Evita que a recuperação brigue com a localização por áudio.
                if not is_match and self._roots_equal(parse_chord(exp_sounding), det_sym):
                    is_match = True
                    root_only_match = True

            if is_match:
                # ----------------------------------------------------
                # CASO B: CONCORDÂNCIA PLENA (ÁUDIO CONFIRMA A CIFRA)
                # ----------------------------------------------------
                self._tracking_state = TrackingState.TRACKING
                # Uma confirmação harmônica forte deve restaurar a confiança de imediato
                # (ponderada pela confiança do áudio), em vez de subir apenas +0.08 por frame.
                # Assim o sistema recupera rapidamente após uma perda prolongada (LOST).
                recovery_target = 0.60 + 0.35 * max(0.0, min(1.0, detected_confidence))
                self._position_confidence = min(
                    0.98,
                    max(self._position_confidence + 0.08, recovery_target)
                )
                self._last_confirmed_bar = current_bar
                self._consecutive_mismatch_seconds = 0.0
                effective_chord = expected_chord
                if is_transposed_match:
                    action_note = f"Confirmada com transposição de {self._transposition.semitones:+d} semitons (capotraste/tom)"
                elif root_only_match:
                    action_note = f"Confirmada pela tônica ({detected_chord} ~ {expected_chord})"
                else:
                    action_note = "Posição harmônica confirmada"

            else:
                # ----------------------------------------------------
                # CASO C: DIVERGÊNCIA (ÁUDIO DETECTOU ACORDE DIFERENTE)
                # ----------------------------------------------------
                self._consecutive_mismatch_seconds += dt
                effective_chord = expected_chord  # A cifra permanece soberana

                # Se a divergência for muito recente (< 2s), pode ser variação de performance ou antecipação
                if self._consecutive_mismatch_seconds < 2.0:
                    action_note = f"Variação momentânea ({detected_chord} vs {expected_chord})"
                    self._position_confidence = max(0.70, self._position_confidence - (dt * 0.05))
                else:
                    self._tracking_state = TrackingState.UNCERTAIN
                    self._position_confidence = max(0.50, self._position_confidence - (dt * 0.08))

                    # ------------------------------------------------
                    # RECUPERAÇÃO INTELIGENTE POR JANELA LOCAL
                    # ------------------------------------------------
                    local_match = self._probe_local_search_window(
                        center_bar=current_bar,
                        detected_chord=detected_chord,
                        window_bars=self._local_window
                    )

                    if local_match:
                        matched_bar, matched_chord = local_match
                        # Só ajusta se estiver monotonicamente próximo (evita saltos bruscos para trás)
                        bar_diff = matched_bar - current_bar
                        if -1 <= bar_diff <= 2:
                            self._tracking_state = TrackingState.RECOVERING
                            action_note = f"Recuperado na janela local (Comp. {matched_bar} - {matched_chord})"
                            self._position_confidence = min(0.88, self._position_confidence + 0.10)
                            # Se for uma pequena diferença de 1 compasso, atualiza o compasso
                            current_bar = matched_bar
                            nominal_pos = self._alignment.get_position_at(current_bar, current_beat, timestamp)
                            effective_chord = matched_chord
                        else:
                            action_note = f"Candidato local descartado (Comp. {matched_bar} fora da tolerância)"
                    else:
                        # RECUPERAÇÃO POR PROGRESSÃO: um único acorde diverge, mas a
                        # SEQUÊNCIA recente de acordes ainda casa com um trecho da cifra.
                        # É mais robusto que o casamento de acorde isolado.
                        prog = self.find_progression_match(
                            list(self._recent_detected),
                            start_bar_hint=current_bar,
                            window_bars=self._local_window * 2
                        )
                        if prog and prog.confidence >= 0.75 and -1 <= (prog.end_bar - current_bar) <= 3:
                            self._tracking_state = TrackingState.RECOVERING
                            current_bar = prog.end_bar
                            nominal_pos = self._alignment.get_position_at(current_bar, current_beat, timestamp)
                            effective_chord = nominal_pos.current_chord
                            self._position_confidence = min(0.90, self._position_confidence + 0.15)
                            action_note = (
                                f"Reancorado por progressão "
                                f"({' → '.join(prog.matched_chords)}) no Comp. {current_bar}"
                            )
                        else:
                            action_note = "Avançando temporalmente (Divergência harmônica sustentada)"

        # Se a localização por áudio reancorou a posição, esse é o evento mais relevante
        if localize_note:
            action_note = localize_note
            if self._tracking_state == TrackingState.LOST:
                self._tracking_state = TrackingState.RECOVERING

        # 3. Consolidação do Estado e Retorno
        self._current_estimated_position = EstimatedPosition(
            bar=current_bar,
            beat=current_beat,
            line_index=nominal_pos.line_index,
            current_chord=effective_chord,
            expected_chord=expected_chord,
            detected_chord=detected_chord if not is_unknown_audio else "--",
            predicted_chord=predicted_chord,
            section_name=nominal_pos.section_name,
            section_progress=nominal_pos.section_progress,
            confidence=self._position_confidence,
            chord_confidence=detected_confidence,
            tracking_state=self._tracking_state,
            action_note=action_note,
            timestamp=timestamp
        )

        return ChartPosition(
            current_bar=current_bar,
            current_beat=current_beat,
            section_id=nominal_pos.section_id,
            section_name=nominal_pos.section_name,
            section_type=nominal_pos.section_type,
            section_index=nominal_pos.section_index,
            current_chord=effective_chord,
            next_chord=nominal_pos.next_chord,
            next_section_name=nominal_pos.next_section_name,
            chord_index_in_section=nominal_pos.chord_index_in_section,
            chord_index=nominal_pos.chord_index,
            element_index=nominal_pos.element_index,
            line_index=nominal_pos.line_index,
            section_progress=nominal_pos.section_progress,
            bars_until_chord_change=nominal_pos.bars_until_chord_change,
            bars_until_section_change=nominal_pos.bars_until_section_change,
            current_lyric=nominal_pos.current_lyric,
            is_last_chord=nominal_pos.is_last_chord,
            absolute_time=timestamp,
            confidence=self._position_confidence,
            tracking_state=self._tracking_state.value
        )

    @staticmethod
    def _chords_equal(a_sym, b_sym) -> bool:
        """Igualdade harmônica estrita (sinônimos ou mesma tônica + qualidade)."""
        return a_sym.is_synonym_of(b_sym) or (a_sym.root == b_sym.root and a_sym.quality == b_sym.quality)

    @staticmethod
    def _roots_equal(a_sym, b_sym) -> bool:
        """Igualdade apenas pela TÔNICA (raiz), tolerante a erro de qualidade na detecção.

        O detector de áudio às vezes erra a qualidade (maior/menor/7ª), mas acerta a
        tônica. Casar pela raiz torna a localização muito mais robusta na prática.
        """
        pa, pb = note_to_pc(a_sym.root), note_to_pc(b_sym.root)
        return pa >= 0 and pa == pb

    def _match_score(self, a_sym, b_sym) -> float:
        """Pontuação de casamento: 1.0 exato, 0.8 só pela tônica, 0.0 sem relação."""
        if self._chords_equal(a_sym, b_sym):
            return 1.0
        if self._roots_equal(a_sym, b_sym):
            return 0.8
        return 0.0

    def _chart_chord_sequence(self):
        """Sequência (bar, símbolo, parse) dos acordes da cifra no domínio SOANTE (capo).

        Remove repetições consecutivas para casar progressões independentemente da duração.
        """
        seq = []
        total = self._alignment.total_bars
        for b in range(1, total + 1):
            pos = self._alignment.get_position_at(b)
            ch = pos.current_chord
            if ch and ch != "--":
                if not seq or seq[-1][1] != ch:
                    seq.append((b, ch, parse_chord(self._expected_as_sounding(ch))))
        return seq

    def _global_localize(self, near_bar: int = 1):
        """Procura NO DOCUMENTO INTEIRO onde a progressão detectada recentemente encaixa.

        Retorna (compasso_atual, score, qtd_acordes) do melhor encaixe, preferindo o mais
        próximo de ``near_bar`` em caso de empate (evita saltar para um refrão idêntico
        distante). É o coração do "seguir o áudio": o programa acha ONDE o músico está.
        """
        recent = [c for c in self._recent_detected if c and c not in ("--", "UNKNOWN", "N")]
        if len(recent) < 2:
            return None
        recent = recent[-4:]  # os últimos acordes distintos bastam para localizar
        recent_syms = [parse_chord(c) for c in recent]
        L = len(recent_syms)

        seq = self._chart_chord_sequence()
        if len(seq) < L:
            return None

        best = None  # ((score, -distância), end_bar, score, L)
        for i in range(0, len(seq) - L + 1):
            window = seq[i:i + L]
            # Pontuação tolerante à tônica (1.0 exato, 0.8 só pela raiz)
            score_sum = sum(self._match_score(window[j][2], recent_syms[j]) for j in range(L))
            ratio = score_sum / L
            if ratio >= 0.75:
                end_bar = window[-1][0]
                dist = abs(end_bar - near_bar)
                key = (ratio, -dist)
                if best is None or key > best[0]:
                    best = (key, end_bar, ratio, L)
        if best is None:
            return None
        return best[1], best[2], best[3]

    def localize(self, near_bar: int = 1):
        """API pública: melhor encaixe da progressão detectada na cifra (ou None)."""
        return self._global_localize(near_bar=near_bar)

    def _probe_local_search_window(
        self,
        center_bar: int,
        detected_chord: str,
        window_bars: int = 4
    ) -> Optional[Tuple[int, str]]:
        """Pesquisa um acorde detectado exclusivamente dentro de uma janela local de compassos.
        
        NUNCA pesquisa no documento inteiro para evitar saltos falsos para outras seções.
        """
        det_sym = parse_chord(detected_chord)
        start_b = max(1, center_bar - window_bars)
        end_b = min(self._alignment.total_bars, center_bar + window_bars)

        # 1. Tenta correspondência exata ou sinônimo na vizinhança imediata
        for b in range(start_b, end_b + 1):
            pos = self._alignment.get_position_at(b)
            if pos.current_chord and pos.current_chord != "--":
                cand_sym = parse_chord(pos.current_chord)
                if cand_sym.is_synonym_of(det_sym) or (cand_sym.root == det_sym.root and cand_sym.quality == det_sym.quality):
                    return b, pos.current_chord

        return None

    def find_progression_match(
        self,
        recent_chords: List[str],
        start_bar_hint: int = 1,
        window_bars: int = 8
    ) -> Optional[ProgressionMatch]:
        """Verifica se uma sequência de múltiplos acordes detectados coincide com a progressão da cifra."""
        if len(recent_chords) < 2:
            return None

        clean_recent = [parse_chord(c) for c in recent_chords if c and c not in ("--", "UNKNOWN")]
        if len(clean_recent) < 2:
            return None

        start_b = max(1, start_bar_hint - window_bars)
        end_b = min(self._alignment.total_bars, start_bar_hint + window_bars)

        # Extrai sequência esperada na vizinhança
        expected_seq = []
        for b in range(start_b, end_b + 1):
            pos = self._alignment.get_position_at(b)
            if pos.current_chord and pos.current_chord != "--":
                if not expected_seq or expected_seq[-1][1] != pos.current_chord:
                    expected_seq.append((b, pos.current_chord, parse_chord(pos.current_chord)))

        # Compara subsequências
        req_len = min(len(clean_recent), len(expected_seq))
        if req_len < 2:
            return None

        recent_tails = clean_recent[-req_len:]
        for i in range(len(expected_seq) - req_len + 1):
            window = expected_seq[i:i + req_len]
            matches = sum(
                1 for j in range(req_len)
                if window[j][2].is_synonym_of(recent_tails[j]) or (window[j][2].root == recent_tails[j].root and window[j][2].quality == recent_tails[j].quality)
            )
            match_ratio = matches / req_len
            if match_ratio >= 0.75:
                return ProgressionMatch(
                    matched_chords=[w[1] for w in window],
                    expected_chords=[c.original_symbol for c in recent_tails],
                    start_bar=window[0][0],
                    end_bar=window[-1][0],
                    confidence=match_ratio
                )

        return None
