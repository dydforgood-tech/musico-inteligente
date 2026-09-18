"""Detector de Tonalidade (Key) baseado no algoritmo Krumhansl-Schmuckler adaptativo."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
import collections
import numpy as np

from app.music.theory import PITCH_CLASSES
from app.music.constants import KEY_LEAKY_DECAY, KEY_LONG_TERM_DECAY


@dataclass
class KeyResult:
    """Resultado da estimativa de tonalidade."""
    key_name: str = "--"         # Ex: "G Major", "A Minor" (estimativa instantânea / curto prazo)
    root: str = "--"             # Ex: "G"
    scale_type: str = "--"       # "Major" ou "Minor"
    confidence: float = 0.0      # Grau de correlação estatística (0.0 a 1.0)
    correlation_score: float = 0.0
    # Estimativa de longo prazo (memória acumulada da música inteira / seção)
    long_term_key: str = "--"
    long_term_root: str = "--"
    long_term_scale: str = "--"
    long_term_confidence: float = 0.0
    # Pontuações detalhadas de todas as 24 tonalidades: { "G Major": (corr_curto, corr_longo) }
    all_key_scores: Dict[str, Tuple[float, float]] = field(default_factory=dict)


@dataclass
class KeyHistoryEntry:
    timestamp: float
    key: str
    confidence: float
    is_confirmed_modulation: bool = False


class KeyHistory:
    """Histórico temporal de tonalidades estimadas para rastreamento de modulação harmônica."""

    def __init__(self, max_entries: Optional[int] = None):
        self._entries: collections.deque = collections.deque(maxlen=max_entries)

    def record(self, timestamp: float, key_name: str, confidence: float, is_confirmed: bool = False) -> None:
        if key_name == "--":
            return
        if self._entries and self._entries[-1].key == key_name:
            return
        self._entries.append(KeyHistoryEntry(timestamp, key_name, confidence, is_confirmed))

    def get_entries(self) -> List[KeyHistoryEntry]:
        return list(self._entries)

    def clear(self) -> None:
        self._entries.clear()


class KeyDetector(ABC):
    """Interface abstrata para estimadores de tonalidade."""

    @abstractmethod
    def estimate_key(self, current_chroma: np.ndarray, timestamp: float = 0.0) -> KeyResult:
        pass

    @property
    @abstractmethod
    def history(self) -> KeyHistory:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass


class KrumhanslSchmucklerKeyDetector(KeyDetector):
    """Implementação estatística de Krumhansl-Schmuckler com memória deslizante dupla (Dual-Timeframe).
    
    Mantém dois acumuladores:
    1. Curto prazo (leaky decay ~0.96): sensibilidade imediata a modulações locais e transientes.
    2. Longo prazo (leaky decay ~0.995): agregação temporal de ~20s, acumulando todas as notas da
       escala diatônica e rejeitando oscilações causadas por acordes individuais.
    """

    # Perfis de Krumhansl-Kessler para escalas Maior e Menor
    KS_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88], dtype=np.float32)
    KS_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17], dtype=np.float32)

    def __init__(
        self,
        decay: float = KEY_LEAKY_DECAY,
        long_term_decay: float = KEY_LONG_TERM_DECAY
    ):
        self._short_decay = decay
        self._long_decay = long_term_decay

        self._short_chroma = np.zeros(12, dtype=np.float32)
        self._long_chroma = np.zeros(12, dtype=np.float32)

        self._history = KeyHistory(max_entries=None)
        self._last_result = KeyResult()
        self._update_counter = 0

        # Pré-computa a matriz normalizada de templates para todas as 24 tonalidades (12 maiores + 12 menores)
        # para cálculo vetorial ultrarrápido em NumPy
        self._key_names: List[str] = []
        self._key_roots: List[str] = []
        self._key_types: List[str] = []
        templates = []

        for i, root in enumerate(PITCH_CLASSES):
            p_maj = np.roll(self.KS_MAJOR, i)
            p_maj_norm = (p_maj - np.mean(p_maj)) / (np.std(p_maj) + 1e-9)
            templates.append(p_maj_norm)
            self._key_names.append(f"{root} Major")
            self._key_roots.append(root)
            self._key_types.append("Major")

        for i, root in enumerate(PITCH_CLASSES):
            p_min = np.roll(self.KS_MINOR, i)
            p_min_norm = (p_min - np.mean(p_min)) / (np.std(p_min) + 1e-9)
            templates.append(p_min_norm)
            self._key_names.append(f"{root} Minor")
            self._key_roots.append(root)
            self._key_types.append("Minor")

        self._templates_matrix = np.array(templates, dtype=np.float32)  # Shape (24, 12)

    @property
    def history(self) -> KeyHistory:
        return self._history

    def reset(self) -> None:
        self._short_chroma.fill(0.0)
        self._long_chroma.fill(0.0)
        self._history.clear()
        self._last_result = KeyResult()
        self._update_counter = 0

    def estimate_key(self, current_chroma: np.ndarray, timestamp: float = 0.0) -> KeyResult:
        if len(current_chroma) < 12 or np.max(current_chroma) < 0.05:
            return self._last_result

        # Atualiza acumuladores de curto e longo prazo
        self._short_chroma = (self._short_decay * self._short_chroma) + ((1.0 - self._short_decay) * current_chroma)
        self._long_chroma = (self._long_decay * self._long_chroma) + ((1.0 - self._long_decay) * current_chroma)
        self._update_counter += 1

        # Requer acumulação mínima de frames para estabilidade inicial
        if self._update_counter < 5:
            return self._last_result

        # Normaliza cromagrama de curto prazo
        c_s = self._short_chroma
        c_s_std = float(np.std(c_s))
        if c_s_std > 1e-4:
            c_s_norm = (c_s - np.mean(c_s)) / c_s_std
            corrs_short = np.dot(self._templates_matrix, c_s_norm) / 12.0
        else:
            corrs_short = np.zeros(24, dtype=np.float32)

        # Normaliza cromagrama de longo prazo
        c_l = self._long_chroma
        c_l_std = float(np.std(c_l))
        if c_l_std > 1e-4:
            c_l_norm = (c_l - np.mean(c_l)) / c_l_std
            corrs_long = np.dot(self._templates_matrix, c_l_norm) / 12.0
        else:
            corrs_long = np.zeros(24, dtype=np.float32)

        # Melhor estimativa de curto prazo (instantânea)
        best_short_idx = int(np.argmax(corrs_short))
        best_short_corr = float(corrs_short[best_short_idx])
        best_short_conf = float(np.clip((best_short_corr + 1.0) / 2.0, 0.0, 1.0))

        # Melhor estimativa de longo prazo (memória contextual de notas)
        best_long_idx = int(np.argmax(corrs_long))
        best_long_corr = float(corrs_long[best_long_idx])
        best_long_conf = float(np.clip((best_long_corr + 1.0) / 2.0, 0.0, 1.0))

        # Dicionário completo das 24 tonalidades com (corr_curto, corr_longo)
        all_scores: Dict[str, Tuple[float, float]] = {}
        for idx, name in enumerate(self._key_names):
            all_scores[name] = (float(corrs_short[idx]), float(corrs_long[idx]))

        result = KeyResult(
            key_name=self._key_names[best_short_idx],
            root=self._key_roots[best_short_idx],
            scale_type=self._key_types[best_short_idx],
            confidence=best_short_conf,
            correlation_score=best_short_corr,
            long_term_key=self._key_names[best_long_idx],
            long_term_root=self._key_roots[best_long_idx],
            long_term_scale=self._key_types[best_long_idx],
            long_term_confidence=best_long_conf,
            all_key_scores=all_scores
        )
        self._last_result = result
        self._history.record(timestamp, result.key_name, result.confidence)

        return result



class KeyChangeDetector(ABC):
    """Interface conceitual para validação de modulações harmônicas reais.
    
    Diferencia acordes de empréstimo modal ou notas de passagem de uma mudança de tom definitiva.
    """

    @abstractmethod
    def evaluate_transition(self, current_key: str, candidate_key: str, stability_duration: float) -> bool:
        pass

