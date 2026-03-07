from __future__ import annotations

from typing import Optional, List
from pathlib import Path

import numpy as np

from PyQt6.QtCore import Qt, QTimer, QUrl
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtGui import QCloseEvent, QFont
from PyQt6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QTabWidget,
    QPlainTextEdit,
    QTableView,
    QToolBar,
    QCheckBox,
    QPushButton,
    QSplitter,
    QMessageBox,
)

from .controls_panel import GuiState
from .tables.equilibria_table import EquilibriaTableModel, EqRow
from ..model.ml_equations import jacobian
from ..model.parameters import MLParameters


class MathInspectorWindow(QMainWindow):

    def __init__(self, main_window, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._main = main_window

        self.setWindowTitle("Mathematical Inspector")

        self._last_state: Optional[GuiState] = None
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(250)
        self._timer.timeout.connect(self._update_now)

        self._eps_tr = 1e-2
        self._eps_det = 1e-2

        # Path to local MathJax (offline)
        # Expected: ml_app/gui/assets/mathjax/es5/tex-chtml.js
        self._mathjax_base = Path(__file__).resolve().parent / "assets" / "mathjax" / "es5"
        self._mathjax_script = self._mathjax_base / "tex-chtml.js"

        self._build_ui()

    # -----------------
    # Public API
    # -----------------
    def set_state(self, state: GuiState) -> None:
        self._last_state = state
        if not self.isVisible():
            return
        if self._auto_chk.isChecked():
            self._timer.start()

    # -----------------
    # UI
    # -----------------
    def _build_ui(self) -> None:
        tb = QToolBar("Math Inspector", self)
        self.addToolBar(tb)

        self._auto_chk = QCheckBox("Auto-update")
        self._auto_chk.setChecked(True)
        tb.addWidget(self._auto_chk)

        self._refresh_btn = QPushButton("Refresh")
        self._refresh_btn.clicked.connect(self._on_refresh_clicked)
        tb.addWidget(self._refresh_btn)

        central = QWidget(self)
        root = QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        self._tabs = QTabWidget()
        root.addWidget(self._tabs)

        # Summary tab
        sum_tab = QWidget()
        sum_layout = QVBoxLayout(sum_tab)
        sum_layout.setContentsMargins(0, 0, 0, 0)
        sum_layout.setSpacing(6)

        self._summary_html = QWebEngineView()
        self._summary_html.setContextMenuPolicy(Qt.ContextMenuPolicy.NoContextMenu)

        split = QSplitter(Qt.Orientation.Vertical)
        split.addWidget(self._summary_html)
        split.setStretchFactor(0, 2)
        split.setStretchFactor(1, 1)
        split.setSizes([360, 180])

        sum_layout.addWidget(split)
        self._tabs.addTab(sum_tab, "Summary")

        # Equilibria tab
        eq_tab = QWidget()
        eq_layout = QVBoxLayout(eq_tab)
        eq_layout.setContentsMargins(0, 0, 0, 0)
        eq_layout.setSpacing(6)

        self._eq_model = EquilibriaTableModel(self)
        self._eq_view = QTableView()
        self._eq_view.setModel(self._eq_model)

        self._eq_view.setAlternatingRowColors(True)
        self._eq_view.setSortingEnabled(True)
        self._eq_view.setShowGrid(False)
        self._eq_view.horizontalHeader().setDefaultAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        self._eq_view.horizontalHeader().setHighlightSections(False)
        self._eq_view.setWordWrap(False)
        self._eq_view.setCornerButtonEnabled(False)
        self._eq_view.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        self._eq_view.setSelectionMode(QTableView.SelectionMode.SingleSelection)
        self._eq_view.horizontalHeader().setStretchLastSection(True)
        self._eq_view.verticalHeader().setVisible(False)
        self._eq_view.selectionModel().selectionChanged.connect(self._on_eq_selection_changed)

        eq_layout.addWidget(self._eq_view)
        self._tabs.addTab(eq_tab, "Equilibria")

        # Linearization tab
        lin_tab = QWidget()
        lin_layout = QVBoxLayout(lin_tab)
        lin_layout.setContentsMargins(0, 0, 0, 0)
        lin_layout.setSpacing(6)

        self._lin_math = QWebEngineView()
        self._lin_math.setContextMenuPolicy(Qt.ContextMenuPolicy.NoContextMenu)
        self._lin_math.setMinimumHeight(160)

        self._lin_text = QPlainTextEdit()
        self._lin_text.setReadOnly(True)
        self._lin_text.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)

        lin_layout.addWidget(self._lin_math)
        lin_layout.addWidget(self._lin_text, stretch=1)
        self._tabs.addTab(lin_tab, "Linearization")

        self.setCentralWidget(central)
        self.resize(760, 540)

        # Optional: preload placeholder
        self._lin_math.setHtml(
            "<html><body style='color:#e6e6e6; font-family: sans-serif; padding:12px;'>"
            "Select an equilibrium.</body></html>"
        )

        self.setStyleSheet("""
        QMainWindow { background: #0f1115; }
        QWidget { color: #e6e6e6; font-size: 12px; }
        QToolBar { background: #151923; border: 0px; padding: 6px; spacing: 8px; }
        QToolBar QCheckBox, QToolBar QPushButton { margin-right: 8px; }

        QPushButton {
            background: #232a3a;
            border: 1px solid #2f3850;
            border-radius: 8px;
            padding: 6px 10px;
        }
        QPushButton:hover { background: #2a3350; }
        QPushButton:pressed { background: #1c2233; }

        QTabWidget::pane {
            border: 1px solid #222a3b;
            border-radius: 10px;
            top: -1px;
            background: #111520;
        }
        QTabBar::tab {
            background: #151923;
            border: 1px solid #222a3b;
            border-bottom: none;
            padding: 7px 12px;
            border-top-left-radius: 10px;
            border-top-right-radius: 10px;
            margin-right: 4px;
        }
        QTabBar::tab:selected { background: #111520; }

        QPlainTextEdit, QTextEdit, QTextBrowser {
            background: #0b0d12;
            border: 1px solid #222a3b;
            border-radius: 10px;
            padding: 10px;
            selection-background-color: #3a4a7a;
        }

        QTableView {
            background: #0b0d12;
            border: 1px solid #222a3b;
            border-radius: 10px;
            gridline-color: #1b2233;
            selection-background-color: #2f3b63;
            selection-color: #ffffff;
            alternate-background-color: #0e1017;
        }
        QHeaderView::section {
            background: #151923;
            border: none;
            padding: 6px 8px;
            font-weight: 600;
            color: #e6e6e6;
        }
        """)

    # -----------------
    # Actions
    # -----------------
    def _on_refresh_clicked(self) -> None:
        if self._last_state is None:
            return
        self._timer.stop()
        self._update_now()

    def _update_now(self) -> None:
        st = self._last_state
        if st is None:
            return

        try:
            rows, summary_text = self._compute_rows_and_summary(st)
        except Exception as e:
            QMessageBox.critical(self, "Math inspector error", f"{type(e).__name__}: {e}")
            return

        self._eq_model.set_rows(rows)
        self._set_summary_html(st=st, rows=rows, summary_text=summary_text)

        sel = self._eq_view.selectionModel()
        if sel is None or not sel.hasSelection():
            self._lin_text.setPlainText("Select an equilibrium to see Jacobian / eigenvalues.")
            self._lin_math.setHtml("")

    def _compute_rows_and_summary(self, st: GuiState) -> tuple[List[EqRow], str]:
        par: MLParameters = st.par
        I_ext = float(st.I_ext)

        eqs = self._main.analysis.equilibria(
            I_ext=I_ext,
            par=par,
            u_min=-90.0,
            u_max=60.0,
            n_scan=5001,
            mr_tol=1e-4,
            classify=True,
        )

        rows: List[EqRow] = []
        hopf_cands = 0
        sn_cands = 0

        for e in eqs:
            eig = getattr(e, "eigvals", None)

            if eig is not None:
                eig = np.asarray(eig)
                if eig.size >= 2 and np.isfinite(eig).all():
                    tr = float(np.real(eig[0] + eig[1]))
                    det = float(np.real(eig[0] * eig[1]))
                else:
                    J = jacobian(e.u, e.w, I_ext, par)
                    eig = np.linalg.eigvals(J)
                    tr = float(np.trace(J))
                    det = float(np.linalg.det(J))
            else:
                J = jacobian(e.u, e.w, I_ext, par)
                eig = np.linalg.eigvals(J)
                tr = float(np.trace(J))
                det = float(np.linalg.det(J))

            near_hopf = (det > self._eps_det) and (abs(tr) < self._eps_tr)
            near_sn = abs(det) < self._eps_det
            hopf_cands += int(near_hopf)
            sn_cands += int(near_sn)

            stab = str(e.stability) if e.stability is not None else "unknown"

            lam1 = complex(eig[0]) if len(eig) > 0 else 0j
            lam2 = complex(eig[1]) if len(eig) > 1 else 0j

            rows.append(
                EqRow(
                    u=float(e.u),
                    w=float(e.w),
                    stability=stab.replace("_", " "),
                    trace=tr,
                    det=det,
                    lam1=lam1,
                    lam2=lam2,
                    near_hopf=near_hopf,
                    near_sn=near_sn,
                )
            )

        rows.sort(key=lambda r: r.u)

        lines: List[str] = []
        lines.append("=== Morris–Lecar variant ===")
        lines.append("du/dt = f(u,w; I_ext, parameters)")
        lines.append("dw/dt = g(u,w; parameters)")
        lines.append("")
        lines.append(f"I_ext = {I_ext:.6g}")
        lines.append(f"Equilibria found (scan u∈[-90,60], n=5001): {len(rows)}")
        lines.append(f"Hopf candidates: {hopf_cands}")
        lines.append(f"Saddle-node candidates: {sn_cands}")
        if len(rows) >= 3:
            lines.append(">=3 equilibria detected -> bistability may be possible.")
        lines.append("")
        lines.append("Tip: switch to manual refresh near bifurcations if updates feel heavy.")

        return rows, "\n".join(lines)

    def _on_eq_selection_changed(self, *_args) -> None:
        sel = self._eq_view.selectionModel()
        if sel is None or not sel.hasSelection():
            self._lin_text.setPlainText("Select an equilibrium to see Jacobian / eigenvalues.")
            self._lin_math.setHtml("")
            return
        rows = sel.selectedRows()
        if not rows:
            self._lin_text.setPlainText("Select an equilibrium to see Jacobian / eigenvalues.")
            self._lin_math.setHtml("")
            return

        idx = rows[0]
        if not idx.isValid():
            self._lin_text.setPlainText("Select an equilibrium to see Jacobian / eigenvalues.")
            self._lin_math.setHtml("")
            return

        row = self._eq_model.row_at(idx.row())
        if row is None or self._last_state is None:
            self._lin_text.setPlainText("Select an equilibrium to see Jacobian / eigenvalues.")
            self._lin_math.setHtml("")
            return

        st = self._last_state
        par = st.par
        I_ext = float(st.I_ext)

        J = jacobian(row.u, row.w, I_ext, par)
        eig = np.linalg.eigvals(J)

        self._lin_text.setPlainText(_format_linearization(row, J, eig))

        lam1 = complex(eig[0]); lam2 = complex(eig[1])
        latex = (
            r"\begin{aligned}"
            rf"&J(u^*,w^*)=\begin{{pmatrix}}"
            rf"{J[0,0]:.4g} & {J[0,1]:.4g}\\"
            rf"{J[1,0]:.4g} & {J[1,1]:.4g}"
            rf"\end{{pmatrix}}\\"
            rf"&\mathrm{{tr}}(J)={row.trace:.4g},\quad \det(J)={row.det:.4g}\\"
            rf"&\lambda_1={lam1.real:.4g}{'+' if lam1.imag>=0 else '-'}{abs(lam1.imag):.4g}i,\quad "
            rf"\lambda_2={lam2.real:.4g}{'+' if lam2.imag>=0 else '-'}{abs(lam2.imag):.4g}i"
            r"\end{aligned}"
        )
        self._set_latex_html(latex)

    def _set_latex_html(self, latex: str) -> None:
        # Offline MathJax loader
        if not self._mathjax_script.exists():
            msg = (
                "<html><body style='color:#e6e6e6; font-family: sans-serif; padding:12px;'>"
                "MathJax not found.<br>"
                "Expected:<br><code>"
                f"{self._mathjax_script}"
                "</code></body></html>"
            )
            self._lin_math.setHtml(msg)
            return

        base_url = QUrl.fromLocalFile(str(self._mathjax_base) + "/")
        script_url = QUrl.fromLocalFile(str(self._mathjax_script)).toString()

        html = f"""
        <!doctype html>
            <html>
            <head>
            <meta charset="utf-8">
            <style>
                body {{
                margin: 0;
                padding: 12px 14px;
                background: #0b0d12;
                color: #e6e6e6;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
                font-size: 14px;
                -webkit-font-smoothing: antialiased;
                text-rendering: geometricPrecision;
                }}
                .math {{
                display: block;
                overflow-x: auto;
                overflow-y: hidden;
                white-space: nowrap;
                }}
                mjx-container {{
                color: #e6e6e6 !important;
                font-size: 115%;
                font-weight: 500;
                }}
                mjx-container * {{
                color: #e6e6e6 !important;
                }}
            </style>

            <script>
                window.MathJax = {{
                tex: {{
                    inlineMath: [['\\\\(', '\\\\)']],
                    displayMath: [['\\\\[', '\\\\]']],
                    packages: {{'[+]': ['ams']}}
                }},
                options: {{
                    renderActions: {{
                    addMenu: []
                    }}
                }}
                }};
            </script>

            <script defer src="{script_url}"></script>
            </head>
            <body>
            <div class="math">\\[{latex}\\]</div>
            </body>
            </html>
        """
        self._lin_math.setHtml(html, baseUrl=base_url)

    def _set_summary_html(self, *, st: GuiState, rows: List[EqRow], summary_text: str) -> None:
        # If MathJax assets are missing, fall back to raw text only.
        if not self._mathjax_script.exists():
            self._summary_html.setHtml(
                "<html><body style='color:#e6e6e6; font-family:sans-serif; padding:12px;'>"
                "MathJax not found for Summary.<br>"
                "Showing raw text only.</body></html>"
            )
            return

        par: MLParameters = st.par
        I_ext = float(st.I_ext)

        # Small LaTeX “header”: model equations + (optional) parameter highlights
        latex = r"""
        \[
        \begin{aligned}
        C\,\frac{du}{dt} &= I_{\mathrm{ext}} - I_{\mathrm{Na}}(u) - I_{\mathrm{K}}(u,w) - I_{\mathrm{L}}(u) \\
        \frac{dw}{dt} &= \frac{w_\infty(u)-w}{\tau_w(u)}
        \end{aligned}
        \]
        """

        # Render a small parameter table (HTML)
        # (You can style this heavily later; keep it simple & readable now.)
        par_rows = []
        for name in par.__dataclass_fields__.keys():
            val = getattr(par, name)
            par_rows.append(f"<tr><td class='k'>{name}</td><td class='v'>{val}</td></tr>")
        par_table = "\n".join(par_rows)

        # Some “cards”
        hopf = sum(1 for r in rows if r.near_hopf)
        sn = sum(1 for r in rows if r.near_sn)

        base_url = QUrl.fromLocalFile(str(self._mathjax_base) + "/")
        script_url = QUrl.fromLocalFile(str(self._mathjax_script)).toString()

        html = f"""
        <!doctype html>
        <html>
        <head>
        <meta charset="utf-8" />
        <style>
            :root {{
            --bg: #0b0d12;
            --panel: #0f1420;
            --panel2: #111a2b;
            --border: #222a3b;
            --text: #e6e6e6;
            --muted: #a9b2c7;
            --accent: #7aa2ff;
            }}
            body {{
            margin: 0;
            padding: 14px;
            background: var(--bg);
            color: var(--text);
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
            font-size: 13.5px;
            -webkit-font-smoothing: antialiased;
            text-rendering: geometricPrecision;
            }}
            .grid {{
            display: grid;
            grid-template-columns: 1.2fr 0.8fr;
            gap: 12px;
            }}
            .card {{
            background: linear-gradient(180deg, var(--panel), var(--panel2));
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 12px 12px;
            }}
            .title {{
            font-weight: 700;
            font-size: 14.5px;
            margin: 0 0 8px 0;
            letter-spacing: 0.2px;
            }}
            .meta {{
            color: var(--muted);
            font-size: 12.5px;
            line-height: 1.35;
            white-space: pre-wrap;
            }}
            table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 12.5px;
            }}
            td {{
            padding: 6px 6px;
            border-bottom: 1px solid rgba(255,255,255,0.06);
            vertical-align: top;
            }}
            td.k {{
            color: var(--muted);
            width: 55%;
            }}
            td.v {{
            text-align: right;
            font-variant-numeric: tabular-nums;
            }}
            .badges {{
            display:flex; gap:8px; flex-wrap:wrap;
            margin-top: 8px;
            }}
            .badge {{
            padding: 4px 8px;
            border-radius: 999px;
            border: 1px solid rgba(255,255,255,0.10);
            background: rgba(122,162,255,0.10);
            color: var(--text);
            font-size: 12px;
            }}
            .math {{
            overflow-x: auto;
            overflow-y: hidden;
            }}
            mjx-container, mjx-container * {{
            color: var(--text) !important;
            }}
        </style>

        <script>
            window.MathJax = {{
            tex: {{
                inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
                displayMath: [['$$','$$'], ['\\\\[','\\\\]']]
            }},
            options: {{
                skipHtmlTags: ['script','noscript','style','textarea','pre','code']
            }}
            }};
        </script>
        <script defer src="{script_url}"></script>
        </head>
        <body>
        <div class="grid">
            <div class="card">
            <div class="title">Overview</div>
            <div class="meta">
    I_ext = {I_ext:.6g}
    Equilibria found: {len(rows)}
    Hopf candidates: {hopf}
    Saddle-node candidates: {sn}
            </div>
            <div class="badges">
                <div class="badge">Auto-update: {"ON" if self._auto_chk.isChecked() else "OFF"}</div>
                <div class="badge">Scan window u∈[-90,60]</div>
            </div>
            <div class="math">{latex}</div>
            </div>

            <div class="card">
            <div class="title">Parameters</div>
            <table>{par_table}</table>
            </div>

            <div class="card" style="grid-column: 1 / -1;">
            <div class="title">Raw summary</div>
            <div class="meta">{summary_text}</div>
            </div>
        </div>

        <script>
            // Ask MathJax to typeset after load
            window.addEventListener("load", () => {{
            if (window.MathJax && window.MathJax.typesetPromise) {{
                window.MathJax.typesetPromise();
            }}
            }});
        </script>
        </body>
        </html>
        """

        self._summary_html.setHtml(html, base_url)

    def closeEvent(self, event: QCloseEvent) -> None:
        event.ignore()
        self.hide()
        if hasattr(self._main, "controls"):
            self._main.controls.set_math_inspector_checked(False)


def _format_linearization(row: EqRow, J: np.ndarray, eig: np.ndarray) -> str:
    lam1 = complex(eig[0]) if len(eig) > 0 else 0j
    lam2 = complex(eig[1]) if len(eig) > 1 else 0j

    lines: List[str] = []
    lines.append(f"Equilibrium: u*={row.u:.6g} mV, w*={row.w:.6g}")
    lines.append(f"type: {row.stability}")
    lines.append("")
    lines.append("Jacobian J(u*,w*):")
    lines.append(f"[ {J[0,0]: .6g}   {J[0,1]: .6g} ]")
    lines.append(f"[ {J[1,0]: .6g}   {J[1,1]: .6g} ]")
    lines.append("")
    lines.append(f"tr(J)  = {row.trace:.6g}")
    lines.append(f"det(J) = {row.det:.6g}")
    lines.append("")
    lines.append(f"λ1 = {lam1.real:.6g} {'+' if lam1.imag>=0 else '-'} {abs(lam1.imag):.6g} i")
    lines.append(f"λ2 = {lam2.real:.6g} {'+' if lam2.imag>=0 else '-'} {abs(lam2.imag):.6g} i")

    flags = []
    if row.near_hopf:
        flags.append("near Hopf (|tr| small, det>0)")
    if row.near_sn:
        flags.append("near saddle-node (|det| small)")
    if flags:
        lines.append("")
        lines.append("Flags: " + " | ".join(flags))

    return "\n".join(lines)