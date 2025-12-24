from __future__ import annotations

from typing import Optional

import numpy as np

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QMainWindow,
    QWidget,
    QSplitter,
    QStatusBar,
    QMessageBox,
    QApplication
)

from .controls_panel import ControlsPanel, GuiState
from .phaseplane_canvas import PhasePlaneCanvas
from .timeseries_canvas import TimeSeriesCanvas

from ..model.simulation import simulate

class MainWindow(QMainWindow):
    """Main application window. Listens to ControlsPanel, manages plotting updates and displays simulation results."""
    def __init__(self, parent:Optional[QWidget] = None) -> None:
        super().__init__(parent)

        self.setWindowTitle("Neuron ML Simulator")

        #Widgets
        self.controls = ControlsPanel(self)
        self.phase_canvas = PhasePlaneCanvas(self)
        self.ts_canvas = TimeSeriesCanvas(self)

        #Layout
        self._build_layout()

        #Status bar
        self._status = QStatusBar(self)
        self.setStatusBar(self._status)

        self.controls.stateChanged.connect(self._on_state_changed)
        self.controls.runRequested.connect(self._on_run_requested)

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
    
    #Signal handlers
    def _on_state_changed(self, state: GuiState) -> None:
        self.phase_canvas.set_view(state.phase_view)
        self.phase_canvas.set_state(state.par, state.I_ext)

        self.ts_canvas.set_view(state.ts_view)
        self._status.showMessage(
            f"I_ext={state.I_ext:.6g} | y0=({state.y0_u:.3g}, {state.y0_w:.3g}) "
            f"| t in [{state.sim.t0:g},{state.sim.t1:g}] dt={state.sim.dt:g} "
            f"| method={state.sim.method}",
            0,
        )
    
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

        #We inform the user about the status of the simulation.
        msg = "OK" if res.success else "FAILED"
        spikes = res.events.get("spike", None) if hasattr(res, "events") else None
        n_spikes = int(len(spikes)) if spikes is not None else 0

        self._status.showMessage(
            f"Run: {msg} | nfev={res.nfev} | spikes={n_spikes} | {res.message}",
            0,
        )