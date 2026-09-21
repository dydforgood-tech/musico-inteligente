"""Histórico atômico das alterações estruturais do editor de cifra."""

from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class ChartEditState:
    """Snapshot mínimo que reconstrói documento, seleção e timeline derivada."""

    chart_text: str
    selected_block_index: int = -1


class EditHistory:
    """Undo/redo linear; a sessão musical é reconstruída ao aplicar o snapshot."""

    def __init__(self, limit: int = 100):
        self._limit = max(1, int(limit))
        self._undo: List[ChartEditState] = []
        self._redo: List[ChartEditState] = []
        self._current: Optional[ChartEditState] = None

    @property
    def can_undo(self) -> bool:
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)

    @property
    def current(self) -> Optional[ChartEditState]:
        return self._current

    def reset(self, state: ChartEditState) -> None:
        self._current = state
        self._undo.clear()
        self._redo.clear()

    def record(self, before: ChartEditState, after: ChartEditState) -> bool:
        """Registra uma ação visual inteira como uma única transação."""
        if before == after:
            self._current = after
            return False
        self._current = before
        self._undo.append(before)
        if len(self._undo) > self._limit:
            del self._undo[0]
        self._current = after
        self._redo.clear()
        return True

    def undo(self) -> Optional[ChartEditState]:
        if not self._undo or self._current is None:
            return None
        self._redo.append(self._current)
        self._current = self._undo.pop()
        return self._current

    def redo(self) -> Optional[ChartEditState]:
        if not self._redo or self._current is None:
            return None
        self._undo.append(self._current)
        self._current = self._redo.pop()
        return self._current
