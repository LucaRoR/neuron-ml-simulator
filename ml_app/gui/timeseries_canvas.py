from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict

import numpy as np

from PyQt6.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from ..model.simulation import SimulationResult

@dataclass(frozen=True)
class TimeSeriesView:
    t_min: Optional[float] = None #ms
    t_max: Optional[float] = None #ms

    show_u: bool = True
    show_w: bool = True
    show_spikes: bool = True

    show_grid: bool = True
    lw_u: float = 1.8 #linewdith
    lw_w: float = 1.8

class TimeSeriesCanvas(QWidget):
    def __init__(self, 
                 parent: Optional[QWidget] = None,
                 *,
                 view: TimeSeriesView = TimeSeriesView()) -> None:
        super().__init__(parent)

        self._view = view
        self._res: Optional[SimulationResult] = None

        self._fig = Figure(constrained_layout=True)

        self._ax_u = self._fig.add_subplot(211)
        self._ax_w = self._fig.add_subplot(212, sharex=self._ax_u)

        self._canvas = FigureCanvas(self._fig)

        layout = QVBoxLayout()
        layout.addWidget(self._canvas)
        self.setLayout(layout)

        self._setup_axes()

    #-----------
    # Public API
    #-----------

    def set_result(self, res: SimulationResult) -> None:
        """We give the program the simulation result and make it replot"""
        self._res = res
        self.redraw()
    
    def clear(self) -> None:
        """We delete everything in the canvas"""
        self._res = None
        self.redraw()
    
    def set_view(self, view:TimeSeriesView) -> None:
        """We replot when settings (zoom/parameters) are changed"""
        self._view = view
        self.redraw()

    def redraw(self) -> None:
        """Replot the canvas"""
        self._ax_u.clear()
        self._ax_w.clear()
        self._setup_axes()

        if self._res is None:
            self._ax_u.set_title("Run a simulation to plot.")
            self._canvas.draw_idle()
            return
        
        res = self._res
        view = self._view

        #Basic controls on the function to avoid weird input/output
        t = np.asarray(res.t, dtype=float)
        y = np.asarray(res.y, dtype=float)
        if (y.ndim != 2) or y.shape[0] != 2:
            raise ValueError(f"SimulationResult.y must have shape (2, N). Got {y.shape}")
        
        u = y[0,:]
        w = y[1,:]

        #Plot signals
        if view.show_u:
            self._ax_u.plot(t, u, linewidth=view.lw_u, label="u(t)")

        if view.show_w:
            self._ax_w.plot(t, w, linewidth=view.lw_w, label="w(t)")
        
        #Plot spikes
        if view.show_spikes:
            self._plot_spike_markers(res.events)
        
        #Manual zoom
        self._apply_time_window(t)

        #Set title
        status = "success" if res.success else "failed"
        self._ax_u.set_title(f"Time series ({status}). Number of evaluations: {res.nfev}")

        #Show legends
        if view.show_u:
            self._ax_u.legend(loc="best")
        
        if view.show_w:
            self._ax_w.legend(loc="best")
        
        self._canvas.draw_idle()

    #-----------------
    # Internal helpers
    #-----------------

    def _setup_axes(self):
        v = self._view

        self._ax_u.set_ylabel("u (mV)")
        self._ax_w.set_ylabel("w")
        self._ax_w.set_xlabel("t (ms)")

        if v.show_grid:
            self._ax_u.grid(True, alpha=0.25)
            self._ax_w.grid(True, alpha=0.25)
        else:
            self._ax_u.grid(False)
            self._ax_w.grid(False)

    def _plot_spike_markers(self, events: Dict[str, np.ndarray]) -> None:
        if not events:
            return
        
        te = events.get("spikes", None)
        if te is None:
            return
        
        te = np.asarray(te, dtype=float).reshape(-1)
        if te.size == 0:
            return
        
        for t_spike in te:
            self._ax_u.axvline(float(t_spike), linewidth=1.0, alpha=0.35)
            self._ax_w.axvline(float(t_spike), linewidth=1.0, alpha=0.35)
    
    def _apply_time_window(self, t: np.ndarray) -> None:
        v = self._view
        if t.size == 0:
            return
        
        #autoscale
        if v.t_min is None and v.t_max is None:
            self._ax_w.set_xlim(float(t[0]), float(t[-1]))
            return

        t0 = float(t[0])
        t1 = float(t[-1])

        x_min = t0 if v.t_min is None else float(v.t_min)
        x_max = t1 if v.t_max is None else float(v.t_max)

        #if the interval is invalid, resort to standard values
        if x_max <= x_min:
            x_min, x_max = t0, t1
        
        self._ax_w.set_xlim(x_min, x_max)