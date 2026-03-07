from __future__ import annotations

from dataclasses import dataclass, astuple
from typing import Tuple, Optional

import numpy as np

import time

from PyQt6.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from scipy.integrate import solve_ivp

from ..model.analysis_engine import AnalysisEngine
from ..model.parameters import MLParameters
from ..model.equilibria import Equilibrium
from ..model.ml_equations import f, g, w_inf
from ..model.separatrix import compute_separatrix, ViewWindow
from ..model.limit_cycle import compute_limit_cycle


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
    show_trajectory: bool = True
    show_bifurcations: bool = False
    show_separatrix: bool = False
    show_limit_cycle: bool = False

class PhasePlaneCanvas(QWidget): #render the (u,w)-plane. It contains nullclines, equilibria, vector field 
    def __init__(self, 
                 parent: Optional[QWidget] = None,
                 *,
                 view: PhasePlaneView = PhasePlaneView(),
                 analysis: AnalysisEngine,
    ) -> None:
        super().__init__(parent)

        self._view = view
        self._analysis = analysis
        self._par: Optional[MLParameters] = None
        self._I_ext: Optional[float] = None

        self._fig = Figure(constrained_layout=True) #create a Matplotlib figure
        self._ax = self._fig.add_subplot(111) #create axes
        self._canvas = FigureCanvas(self._fig) #Matplotlib figure becomes a Qt widget

        self._vf_cache_key = None #cache for vector field
        self._vf_cache_res = None

        self._quiver = None

        self._null_cache_key = None #cache for nullclines
        self._null_cache_res = None

        self._eq_cache_key = None #cache for equilibria
        self._eq_cache_res = None

        self._eq_sc_stable = self._ax.scatter([], [], s=60.0, marker="o", zorder=5, color="tab:green")
        self._eq_sc_saddle = self._ax.scatter([], [], s=80.0, marker="X", zorder=5, color="tab:orange")
        self._eq_sc_unstable = self._ax.scatter([], [], s=60.0, marker="^", zorder=5, color="tab:red")
        self._eq_texts = []

        self._bif_cache_key = None
        self._bif_cache_res = None  # (sns, hopf)

        self._sn_scatter = self._ax.scatter([], [], marker="s", s=45, zorder=6)
        self._hopf_scatter = self._ax.scatter([], [], marker="x", s=55, zorder=6)
        self._bif_texts = []

        self._sep_cache_key = None #cache for separatrix
        self._sep_cache_res = None

        self._sep_lines = []

        self._traj_cache_key = None #cache for trajectory
        self._traj_cache_res = None

        self._traj_line, = self._ax.plot([], [], linewidth=2.2, zorder=50)
        self._traj_line.set_visible(False)

        self._lc_cache_key = None
        self._lc_cache_res = None
        self._lc_last_good = None
        self._lc_line, = self._ax.plot([], [], linewidth=2.2, zorder=60)
        self._lc_line.set_visible(False)

        self._last_timings = {}

        self.INTERACTIVE_EQ_SCAN = 1501
        self.FINAL_EQ_SCAN = 5001

        self._nullcline_w_line, = self._ax.plot([], [], linewidth=2.0, label="w nullcline")
        self._nullcline_u_line, = self._ax.plot([], [], linewidth=2.0, label="u nullcline")

        layout = QVBoxLayout()
        layout.addWidget(self._canvas)
        self.setLayout(layout)

        self._setup_axes()

    #-----------
    # Public API
    #-----------

    def set_state(self, par: MLParameters, I_ext: float) -> None:
        I_ext = float(I_ext)
        if self._par == par and self._I_ext == I_ext:
            return
        self._lc_last_good = None
        self._lc_cache_key = None
        self._lc_cache_res = None
        self._par = par
        self._I_ext = I_ext
        self.redraw()

    def set_view(self, view: PhasePlaneView) -> None:
        if self._view == view:
            return
        self._view = view
        self.redraw()

    def set_state_and_view(self, par: MLParameters, I_ext: float, view: PhasePlaneView) -> None:
        I_ext = float(I_ext)
        changed = (self._par != par) or (self._I_ext != I_ext) or (self._view != view)
        if (self._par != par) or (self._I_ext != I_ext):
            self._lc_last_good = None
            self._lc_cache_key = None
            self._lc_cache_res = None
        self._par = par
        self._I_ext = I_ext
        self._view = view
        if changed:
            self.redraw()

    def set_trajectory_result(self, res, I_ext: float, par: MLParameters) -> None:
        self._traj_cache_res = res
        self._traj_cache_key = (float(I_ext), self._par_key(par))
        self.redraw()
    
    def redraw(self) -> None:
        if self._par is None or self._I_ext is None:
            #If there is nothing to draw, just display the title and an empty canvas.
            self._ax.set_title("Phase plane. Set parameters to plot.")
            self._canvas.draw_idle()
            return
        
        par = self._par
        I_ext = self._I_ext
        view = self._view
        self._ax.set_xlim(view.u_min, view.u_max)
        self._ax.set_ylim(*self._safe_w_bounds())
    

        t_total0 = time.perf_counter()
        tim = {}

        if view.show_vector_field:
            t0 = time.perf_counter()
            self._plot_vector_field(I_ext, par)
            tim["vf"] = time.perf_counter() - t0
        
        else:
            if self._quiver is not None:
                self._quiver.remove()
                self._quiver = None

        if view.show_nullclines:
            t0 = time.perf_counter()
            self._plot_nullclines(I_ext, par)
            tim["null"] = time.perf_counter() - t0
            self._nullcline_w_line.set_visible(True)
            self._nullcline_u_line.set_visible(True)
            handles, labels = self._ax.get_legend_handles_labels()
            if labels:
                self._ax.legend(loc="best")
        else:
            self._nullcline_w_line.set_visible(False)
            self._nullcline_u_line.set_visible(False)

        if view.show_equilibria:
            t0 = time.perf_counter()
            self._plot_equilibria(I_ext, par)
            tim["eq"] = time.perf_counter() - t0
        else:
            self._eq_sc_stable.set_offsets(np.empty((0, 2)))
            self._eq_sc_saddle.set_offsets(np.empty((0, 2)))
            self._eq_sc_unstable.set_offsets(np.empty((0, 2)))
            for t in self._eq_texts:
                t.remove()
            self._eq_texts.clear()

        if view.show_bifurcations:
            t0 = time.perf_counter()
            self._plot_bifurcations(par)
            tim["bif"] = time.perf_counter() - t0
        else:
            self._sn_scatter.set_offsets(np.empty((0, 2)))
            self._hopf_scatter.set_offsets(np.empty((0, 2)))
            for t in self._bif_texts:
                t.remove()
            self._bif_texts.clear()
        
        if view.show_trajectory and (self._traj_cache_res is not None):
            t0 = time.perf_counter()
            key = (float(I_ext), self._par_key(par))
            if key == self._traj_cache_key:
                y = self._traj_cache_res.y
                self._traj_line.set_data(y[0], y[1])
                self._traj_line.set_visible(True)
            else:
                self._traj_line.set_visible(False)
            tim["traj"] = time.perf_counter() - t0
        else:
            self._traj_line.set_visible(False)
        
        if view.show_separatrix:
            w0, w1 = self._safe_w_bounds()
            t0 = time.perf_counter()


            key = (
                float(I_ext),
                self._par_key(par),
                float(view.u_min), float(view.u_max), float(w0), float(w1),
            )

            if key != self._sep_cache_key:
                win = ViewWindow(view.u_min, view.u_max, w0, w1)
                self._sep_cache_res = compute_separatrix(I_ext, par, window=win)
                self._sep_cache_key = key

            res = self._sep_cache_res
            if res is None:
                # hide any old lines
                for ln in self._sep_lines:
                    ln.set_visible(False)
            else:
                branches = res.branches

                # ensure we have enough line artists
                while len(self._sep_lines) < len(branches):
                    ln, = self._ax.plot([], [], "--", linewidth=1.5, zorder=40)
                    self._sep_lines.append(ln)

                # update lines for existing branches
                for ln, br in zip(self._sep_lines, branches):
                    ln.set_data(br.u, br.w)
                    ln.set_visible(True)

                # hide extra lines if branches decreased
                for ln in self._sep_lines[len(branches):]:
                    ln.set_visible(False)
            
            tim["sep"] = time.perf_counter() - t0

        else:
            #clear cache when not displaying separatrix
            self._sep_cache_key = None
            self._sep_cache_res = None
            for ln in self._sep_lines:
                ln.set_visible(False)

        if view.show_limit_cycle:
            w0, w1 = self._safe_w_bounds()
            key = (
                float(I_ext),
                self._par_key(par),
                float(view.u_min), float(view.u_max), float(w0), float(w1),
            )
            if key != self._lc_cache_key:
                self._lc_cache_res = compute_limit_cycle(
                    I_ext, par,
                    u_min=view.u_min, u_max=view.u_max,
                    T_total=1500.0, T_transient=800.0,
                    max_step=0.5,
                    seed_cycle=self._lc_last_good,
                )
                self._lc_cache_key = key

            res = self._lc_cache_res
            if res is not None:
                self._lc_last_good = res
            if res is None:
                self._lc_line.set_visible(False)
            else:
                self._lc_line.set_data(res.u, res.w)
                self._lc_line.set_visible(True)
        else:
            self._lc_cache_key = None
            self._lc_cache_res = None
            self._lc_line.set_visible(False)


        tim["total"] = time.perf_counter() - t_total0
        self._last_timings = tim
        vf = tim.get("vf", 0.0); nc = tim.get("null", 0.0); eq = tim.get("eq", 0.0); sp = tim.get("sep", 0.0); tot = tim.get("total", 0.0)
        self._ax.set_title(
            f"Phase Plane. I_ext={I_ext:.3f} | total={tot*1e3:.0f}ms vf={vf*1e3:.0f} nc={nc*1e3:.0f} eq={eq*1e3:.0f} sep={sp*1e3:.0f}"
            )
        self._canvas.draw_idle()
    
    #-----------------
    # Internal helpers
    #-----------------

    def _setup_axes(self) -> None:
        v = self._view
        self._ax.set_xlim(v.u_min, v.u_max)
        y0, y1 = float(v.w_min), float(v.w_max)
        y0, y1 = self._safe_w_bounds()
        self._ax.set_ylim(y0, y1)
        self._ax.set_xlabel("u (mV)")
        self._ax.set_ylabel("w")
        self._ax.grid(True, alpha=0.25) #alpha regulates the transparency of the axes

    def _plot_nullclines(self, I_ext: float, par: MLParameters) -> None:
        v = self._view
        u, w_w, w_u = self._analysis.nullclines(
            I_ext=I_ext,
            par=par,
            u_min=v.u_min,
            u_max=v.u_max,
            n_u=v.n_u,
        )
        self._nullcline_w_line.set_data(u, w_w)
        self._nullcline_u_line.set_data(u, w_u)

    def _plot_equilibria(self, I_ext: float, par: MLParameters) -> None:
        v = self._view
        eqs = self._analysis.equilibria(
                I_ext=I_ext,
                par=par,
                u_min=v.u_min,
                u_max=v.u_max,
                n_scan=self.INTERACTIVE_EQ_SCAN,
                mr_tol=1e-4,
                classify=True,
            )
        for t in self._eq_texts:
            t.remove()
        self._eq_texts.clear()

        stable = []
        saddle = []
        unstable = []

        for eq in eqs:
            s = (eq.stability or "").lower()
            pt = (eq.u, eq.w)

            if ("stable_focus" in s) or ("stable_node" in s):
                stable.append(pt)
            elif "saddle" in s:
                saddle.append(pt)
            elif ("unstable_focus" in s) or ("unstable_node" in s):
                unstable.append(pt)

            if eq.stability is not None:
                self._eq_texts.append(
                    self._ax.annotate(
                        eq.stability.replace("_", " "),
                        (eq.u, eq.w),
                        textcoords="offset points",
                        xytext=(6, 6),
                        fontsize=8,
                    )
                )

        self._eq_sc_stable.set_offsets(np.array(stable) if stable else np.empty((0, 2)))
        self._eq_sc_saddle.set_offsets(np.array(saddle) if saddle else np.empty((0, 2)))
        self._eq_sc_unstable.set_offsets(np.array(unstable) if unstable else np.empty((0, 2)))
    
    def _plot_vector_field(self, I_ext: float, par: MLParameters) -> None:
        v = self._view
        w0, w1 = self._safe_w_bounds()

        key = (
            float(I_ext),
            self._par_key(par),
            float(v.u_min), float(v.u_max), float(w0), float(w1),
            int(v.vf_nu), int(v.vf_nw),
        )

        if key != self._vf_cache_key:
            uu = np.linspace(v.u_min, v.u_max, v.vf_nu)
            ww = np.linspace(w0, w1, v.vf_nw)
            U, W = np.meshgrid(uu, ww)

            DU = f(U, W, I_ext, par)
            DW = g(U, W, par)

            # Normalize in plot geometry, then set a fixed displayed length.
            su = float(v.u_max - v.u_min)
            sw = float(w1 - w0)

            eps = 1e-9
            su = max(su, eps)
            sw = max(sw, eps)

            DU_s = DU / su
            DW_s = DW / sw

            norm_s = np.hypot(DU_s, DW_s)

            DU_dir = np.zeros_like(DU)
            DW_dir = np.zeros_like(DW)

            mask = norm_s > 1e-9
            DU_dir[mask] = DU_s[mask] / norm_s[mask]
            DW_dir[mask] = DW_s[mask] / norm_s[mask]

            # Choose arrow length as a fraction of axis span
            L = 0.01

            DU_n = DU_dir * su * L
            DW_n = DW_dir * sw * L

            self._vf_cache_res = (U, W, DU_n, DW_n)
            self._vf_cache_key = key
            if self._quiver is not None:
                self._quiver.remove()
                self._quiver = None

        U, W, DU_n, DW_n = self._vf_cache_res
        if self._quiver is None:
            self._quiver = self._ax.quiver(
                U, W, DU_n, DW_n,
                angles="xy", scale_units="xy",
                scale=1/6, width=0.003, alpha=0.4, zorder=1
            )
        else:
            self._quiver.set_UVC(DU_n, DW_n)


    def _plot_bifurcations(self, par: MLParameters) -> None:
        v = self._view
        sns, hopf = self._analysis.bifurcations(
            par=par,
            u_min=self._view.u_min,
            u_max=self._view.u_max,
            n_scan=self.FINAL_EQ_SCAN,
        )

        for t in self._bif_texts:
            t.remove()
        self._bif_texts.clear()

        # Saddle-node
        sn_pts = []
        for sn in sns:
            w = float(w_inf(sn.u, par))
            sn_pts.append((sn.u, w))
            self._bif_texts.append(
                self._ax.annotate(
                    f"SN\nI={sn.I_ext:.3g}",
                    (sn.u, w),
                    textcoords="offset points",
                    xytext=(6, -10),
                    fontsize=8,
                )
            )

        # Hopf
        hopf_pts = []
        for hp in hopf:
            w = float(w_inf(hp.u, par))
            hopf_pts.append((hp.u, w))
            self._bif_texts.append(
                self._ax.annotate(
                    f"Hopf\nI={hp.I_ext:.3g}",
                    (hp.u, w),
                    textcoords="offset points",
                    xytext=(6, 6),
                    fontsize=8,
                )
            )

        self._sn_scatter.set_offsets(np.array(sn_pts) if sn_pts else np.empty((0, 2)))
        self._hopf_scatter.set_offsets(np.array(hopf_pts) if hopf_pts else np.empty((0, 2)))
        
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
        
    @staticmethod
    def _par_key(par: MLParameters):
        return astuple(par)
        
    def _safe_w_bounds(self) -> tuple[float, float]:
        v = self._view
        y0, y1 = float(v.w_min), float(v.w_max)

        if y1 > y0:
            return y0, y1

        # collapse or swapped -> set default values
        return 0, 1

