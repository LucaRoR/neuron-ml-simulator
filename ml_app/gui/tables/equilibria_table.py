"""This table exposes the Qt structure of the equilibria for the math inspector. This is used by gui/math_inspector_window.py"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Any

import numpy as np
from PyQt6.QtCore import QAbstractTableModel, QModelIndex, Qt


@dataclass(frozen=True)
class EqRow:
    u: float
    w: float
    stability: str
    trace: float
    det: float
    lam1: complex
    lam2: complex
    near_hopf: bool
    near_sn: bool

def _fmt_complex(z: complex) -> str:
    return f"{z.real:.4g}{'+' if z.imag>=0 else '-'}i{abs(z.imag):.4g}"

class EquilibriaTableModel(QAbstractTableModel):
    COLS = ["#", "u* (mV)", "w*", "type", "tr(J)", "det(J)", "λ1", "λ2", "flags"]

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._rows: List[EqRow] = []

    def set_rows(self, rows: List[EqRow]) -> None:
        self.beginResetModel()
        self._rows = list(rows)
        self.endResetModel()

    def row_at(self, i: int) -> Optional[EqRow]:
        if 0 <= i < len(self._rows):
            return self._rows[i]
        return None

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._rows)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self.COLS)

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal and 0 <= section < len(self.COLS):
            return self.COLS[section]
        return None

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if not index.isValid():
            return None
        r, c = index.row(), index.column()
        if r < 0 or r >= len(self._rows):
            return None
        row = self._rows[r]

        if role == Qt.ItemDataRole.DisplayRole:
            if c == 0:
                return str(r + 1)
            if c == 1:
                return f"{row.u:.6g}"
            if c == 2:
                return f"{row.w:.6g}"
            if c == 3:
                return row.stability
            if c == 4:
                return f"{row.trace:.6g}"
            if c == 5:
                return f"{row.det:.6g}"
            if c == 6:
                return _fmt_complex(row.lam1)
            if c == 7:
                return _fmt_complex(row.lam2)
            if c == 8:
                flags = []
                if row.near_hopf:
                    flags.append("Hopf")
                if row.near_sn:
                    flags.append("SN")
                return " | ".join(flags)

        if role == Qt.ItemDataRole.TextAlignmentRole:
            if c in (0, 1, 2, 4, 5, 6, 7):
                return int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            return int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        return None