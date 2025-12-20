from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np

from PyQt6.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from ..model.parameters import MLParameters
from ..model.nullclines import u_nullcline, w_nullcline
from ..model.equilibria import find_equilibria, Equilibrium
from ..model.ml_equations import f, g

@dataclass(frozen=True)
class PhasePlaneView: #view configuration
    u_min: float = -85.0 #mV
    u_max: float = 60.0 #mV
    w_min: float = 0.0
    w_max: float = 1.0

    n_u: int = 2000 #resolution for nullclines
    vf_nu: int = 25 #vector field grid in u
    vf_nw: int = 20 #vector field grid in w

    show_vector_field: bool = True
    show_equilibria: bool = True
    show_nullclines: bool = True

class PhasePlaneCanvas(QWidget): #render the (u,w)-plane. It contains nullclines, equilibria, vector field 
    def __init__(self, 
                 parent: Optional[QWidget] = None,
                 *,
                 view: PhasePlaneView = PhasePlaneView()
    ) -> None:
        super().__init__(parent)

        self._view = view
        self._par: Optional[MLParameters] = None
        self._I_ext: Optional[float] = None

        self._fig = Figure(constrained_layout=True) #create a Matplotlib figure
        self._ax = self._fig.add_subplot(111) #create axes
        self._canvas = FigureCanvas(self._fig) #Matplotlib figure becomes a Qt widget

        layout = QVBoxLayout()
        layout.addWidget(self._canvas)
        self.setLayout(layout)

        self._setup_axes()

    #-----------
    # Public API
    #-----------

    def set_state(self, par: MLParameters, I_ext: float) -> None:
        """Whenever par or I_ext change, call this function and redraw. Access point for the rest of the API"""
        self._par = par
        self._I_ext = float(I_ext)
        self.redraw()
    
    def set_view(self, view: PhasePlaneView) -> None:
        """Whenever the plotting window/zoom changes, call this and redraw."""
        self._view = view
        self.redraw()
    
    def redraw(self) -> None:
        self._ax.clear()
        self._setup_axes()
        
        if self._par is None or self._I_ext is None:
            #If there is nothing to draw, just display the title and an empty canvas.
            self._ax.set_title("Phase plane. Set parameters to plot.")
            self._canvas.draw_idle()
            return
        
        par = self._par
        I_ext = self._I_ext
        view = self._view

        if view.show_vector_field:
            self._plot_vector_field(I_ext, par)
        
        if view.show_nullclines:
            self._plot_nullclines(I_ext, par)

        if view.show_equilibria:
            self._plot_equilibria(I_ext, par)

        self._ax.set_title(f"Phase Plane. I_ext = {I_ext:.3f}")
        self._canvas.draw_idle()
    
    #-----------------
    # Internal helpers
    #-----------------

    def _setup_axes(self) -> None:
        v = self._view
        self._ax.set_xlim(v.u_min, v.u_max)
        self._ax.set_ylim(v.w_min, v.w_max)
        self._ax.set_xlabel("u (mV)")
        self._ax.set_ylabel("w")
        self._ax.grid(True, alpha=0.25) #alpha regulates the transparency of the axes

    def _plot_nullclines(self, I_ext: float, par: MLParameters) -> None:
        v = self._view
        u = np.linspace(v.u_min, v.u_max, v.n_u)

        w_w = w_nullcline(u, par)
        w_u = u_nullcline(u, I_ext, par)

        self._ax.plot(u, w_w, linewidth=2.0, label="w nullcline")
        self._ax.plot(u, w_u, linewidth=2.0, label="u nullcline")

        self._ax.legend(loc="best")

    def _plot_equilibria(self, I_ext: float, par: MLParameters) -> None:
        v = self._view
        eqs = find_equilibria(I_ext, par, u_min=v.u_min, u_max=v.u_max, n_scan=5001, mr_tol=1e-4, classify=True)
        for eq in eqs:
            marker, size = self._marker_for_equilibrium(eq)
            color = self._color_for_equilibrium(eq)
            self._ax.scatter(eq.u, eq.w, s=size, marker=marker, color=color, zorder=5)

            if eq.stability is not None:
                self._ax.annotate(
                    eq.stability.replace("_", " "),
                    (eq.u, eq.w),
                    textcoords="offset points",
                    xytext=(6, 6),
                    fontsize=8,
                )
    
    def _plot_vector_field(self, I_ext: float, par: MLParameters) -> None:
        v = self._view

        #create a grid of points
        uu = np.linspace(v.u_min, v.u_max, v.vf_nu)
        ww = np.linspace(v.w_min, v.w_max, v.vf_nw)
        U, W = np.meshgrid(uu, ww)

        DU = np.empty_like(U, dtype=float)
        DW = np.empty_like(W, dtype=float)

        #compute the vectors on the grid
        for i in range(U.shape[0]):
            for j in range(U.shape[1]):
                u = float(U[i, j])
                w = float(W[i, j])
                DU[i,j] = f(u, w, I_ext, par)
                DW[i,j] = g(u, w, par)
        
        
        #we normalize the vectors to avoid having a few arrows clogging up the whole space
        norm = np.sqrt(DU ** 2 + DW ** 2)
        norm[norm == 0.0] = 1.0
        DU_n = DU / norm
        DW_n = DW / norm

        self._ax.quiver(U, W, DU_n, DW_n, angles="xy", scale_units="xy", scale=1/6, width=0.003, alpha=0.4, zorder=1)
    
    @staticmethod
    def _color_for_equilibrium(eq: Equilibrium) -> str:
        s = (eq.stability or "").lower()
        if ("stable_focus" in s) or ("stable_node" in s):
            return "tab:green"
        if "saddle" in s:
            return "tab:orange"
        if ("unstable_focus" in s) or ("unstable_node" in s):
            return "tab:red"
        else:
            return "tab:gray"
    
    @staticmethod
    def _marker_for_equilibrium(eq: Equilibrium) -> Tuple[str, float]:
        s = (eq.stability or "").lower()
        if ("stable_focus" in s) or ("stable_node" in s):
            return "o", 60.0
        if "saddle" in s:
            return "X", 80.0
        if ("unstable_focus" in s) or ("unstable_node" in s):
            return "^", 60.0
        else:
            return "s", 50.0

