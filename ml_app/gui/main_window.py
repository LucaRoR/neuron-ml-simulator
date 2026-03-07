from __future__ import annotations

from typing import Optional

import numpy as np

from PyQt6.QtGui import QAction
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QMainWindow,
    QWidget,
    QSplitter,
    QStatusBar,
    QMessageBox,
    QApplication
)


from ..model.analysis_engine import AnalysisEngine
from .math_inspector_window import MathInspectorWindow
from .controls_panel import ControlsPanel, GuiState
from .phaseplane_canvas import PhasePlaneCanvas
from .timeseries_canvas import TimeSeriesCanvas

from ..model.simulation import simulate

class MainWindow(QMainWindow):
    """Main application window. Listens to ControlsPanel, manages plotting updates and displays simulation results."""
    def __init__(self, parent:Optional[QWidget] = None) -> None:
        super().__init__(parent)

        self.setWindowTitle("Neuron ML Simulator")

        self.analysis = AnalysisEngine(max_cache=12)
        self.math_window: Optional[MathInspectorWindow] = None

        #Widgets
        self.controls = ControlsPanel(self)
        self.phase_canvas = PhasePlaneCanvas(self, analysis=self.analysis)
        self.ts_canvas = TimeSeriesCanvas(self)

        #Layout
        self._build_layout()

        #Status bar
        self._status = QStatusBar(self)
        self.setStatusBar(self._status)

        self._build_menu()

        self.controls.stateChanged.connect(self._on_state_changed)
        self.controls.runRequested.connect(self._on_run_requested)

        self.controls.mathInspectorToggled.connect(self._on_math_inspector_toggled)

        self._on_state_changed(self.controls.current_state())

    #UI

    def _build_layout(self) -> None:
        #Phase plane and time series
        right_splitter = QSplitter(Qt.Orientation.Vertical, self)
        right_splitter.addWidget(self.phase_canvas)
        right_splitter.addWidget(self.ts_canvas)
        right_splitter.setStretchFactor(0, 1)
        right_splitter.setStretchFactor(1, 1)

        #Controls and plots
        main_splitter = QSplitter(Qt.Orientation.Horizontal, self)
        main_splitter.addWidget(self.controls)
        main_splitter.addWidget(right_splitter)
        main_splitter.setStretchFactor(0, 0)
        main_splitter.setStretchFactor(1, 1)

        self.setCentralWidget(main_splitter)
        self.resize(1200, 800)
    
    #Math inspector

    def _build_menu(self) -> None:
        view_menu = self.menuBar().addMenu("&View")

        act_math = QAction("Mathematical Inspector", self)
        act_math.setShortcut("Ctrl+M")
        act_math.triggered.connect(self._toggle_math_inspector)

        view_menu.addAction(act_math)

    def _on_math_inspector_toggled(self, checked: bool) -> None:
        if checked:
            #show
            if self.math_window is None:
                self.math_window = MathInspectorWindow(self, parent=self)
            self.math_window.show()
            self.math_window.raise_()
            self.math_window.activateWindow()
            self.math_window.set_state(self.controls.current_state())
        else:
            #hide
            if self.math_window is not None:
                self.math_window.hide()

    def _toggle_math_inspector(self) -> None:
        if self.math_window is None:
            self.math_window = MathInspectorWindow(self, parent=self)
        if self.math_window.isVisible():
            self.math_window.hide()
            self.controls.set_math_inspector_checked(False)
        else:
            self.math_window.show()
            self.math_window.raise_()
            self.math_window.activateWindow()
            self.math_window.set_state(self.controls.current_state())
            self.controls.set_math_inspector_checked(True)
    
    #Signal handlers
    def _on_state_changed(self, state: GuiState) -> None:
        self.phase_canvas.set_state_and_view(state.par, state.I_ext, state.phase_view)

        self.ts_canvas.set_view(state.ts_view)
        self._status.showMessage(
            f"I_ext={state.I_ext:.6g} | y0=({state.y0_u:.3g}, {state.y0_w:.3g}) "
            f"| t in [{state.sim.t0:g},{state.sim.t1:g}] dt={state.sim.dt:g} "
            f"| method={state.sim.method}",
            0,
        )

        if self.math_window is not None:
            self.math_window.set_state(state)
    
    def _on_run_requested(self, state: GuiState) -> None:
        self._on_state_changed(state)
        y0 = np.array([state.y0_u, state.y0_w], dtype=float) #initial conditions vector

        #While the simulation runs, the cursor becomes busy to prevent freeze-related confusion
        app = QApplication.instance()
        if app is not None:
            app.setOverrideCursor(Qt.CursorShape.BusyCursor)

        try:
            res = simulate(y0=y0, I_ext=state.I_ext, par=state.par, config=state.sim)
        except Exception as e:
            self.ts_canvas.clear()
            self._status.showMessage("Simulation failed.", 0)
            QMessageBox.critical(
                self,
                "Simulation error",
                f"The simulation raised an exception:\n\n{type(e).__name__}: {e}",
            )
            return
        finally:
            if app is not None:
                app.restoreOverrideCursor()

        self.ts_canvas.set_result(res)
        self.phase_canvas.set_trajectory_result(res, state.I_ext, state.par)

        #We inform the user about the status of the simulation.
        msg = "OK" if res.success else "FAILED"
        spikes = res.events.get("spikes", None) if hasattr(res, "events") else None
        n_spikes = int(len(spikes)) if spikes is not None else 0

        self._status.showMessage(
            f"Run: {msg} | nfev={res.nfev} | spikes={n_spikes} | {res.message}",
            0,
        )