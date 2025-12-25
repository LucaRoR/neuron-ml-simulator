from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.integrate import solve_ivp

from .parameters import MLParameters
from .equilibria import find_equilibria
from .ml_equations import ml_rhs, jacobian

@dataclass(frozen=True)
class ViewWindow:
    u_min: float
    u_max: float
    w_min: float
    w_max: float

    def contains(self, u: float, w: float, margin: float = 0.0) -> bool:
        du = margin * (self.u_max - self.u_min)
        dw = margin * (self.w_max - self.w_min)
        return (
            (self.u_min - du) <= u <= (self.u_max + du)
            and (self.w_min - dw) <= w <= (self.w_max + dw)
        )

@dataclass(frozen=True)
class SeparatrixBranch:
    u: np.ndarray
    w: np.ndarray

@dataclass(frozen=True)
class SeparatrixResult:
    saddle_u: float
    saddle_w: float
    branches: list[SeparatrixBranch]

def compute_separatrix(
    I_ext: float,
    par: MLParameters,
    window: Optional[ViewWindow] = None,
    *,
    # Equilibrium scan range (for locating the saddle)
    u_scan_min: Optional[float] = None,
    u_scan_max: Optional[float] = None,
    n_scan: int = 5001,
    # Backward integration controls
    T: float = 500.0,
    max_step: float = 0.5,
    rtol: float = 1e-6,
    atol: float = 1e-9,
    # Perturbation sizes (if None, chosen from window span or sane defaults)
    eps_u: Optional[float] = None,
    eps_w: Optional[float] = None,
    # Window exit margin (fraction of span)
    exit_margin: float = 0.02,
) -> Optional[SeparatrixResult]:
    if u_scan_min is None:
        u_scan_min = window.u_min if window is not None else -100.0
    if u_scan_max is None:
        u_scan_max = window.u_max if window is not None else 50.0
    
    eqs = find_equilibria(
        I_ext,
        par,
        u_min=float(u_scan_min),
        u_max=float(u_scan_max),
        n_scan=int(n_scan),
        classify=True,
    )

    #Looks for saddles. If none are present, returns None
    saddles = []
    for e in eqs:
        stab = getattr(e, "stability", None)
        if stab is None:
            continue
        if isinstance(stab, str) and "saddle" in stab.lower():
            saddles.append(e)

    if not saddles:
        return None

    saddle = saddles[0]
    if window is not None:
        for e in saddles:
            u0 = float(getattr(e, "u"))
            w0 = float(getattr(e, "w"))
            if window.contains(u0, w0, margin=0.0):
                saddle = e
                break
    
    u_s = float(getattr(saddle, "u"))
    w_s = float(getattr(saddle, "w"))

    #we compute eigenvalues and eigenvectors at the saddle
    J = jacobian(u_s, w_s, I_ext, par)
    vals, vecs = np.linalg.eig(np.asarray(J, dtype=float))

    re = np.real(vals)
    stable_idx = np.where(re < 0.0)[0]
    if stable_idx.size == 0:
        idx = int(np.argmin(re))
    else:
        idx = int(stable_idx[np.argmin(re[stable_idx])])

    v = np.real(vecs[:, idx]).astype(float)

    vnorm = float(np.linalg.norm(v))
    if not np.isfinite(vnorm) or vnorm == 0.0:
        return None
    v = v / vnorm
    
    #Perturbation scales
    if window is not None:
        span_u = max(1e-9, window.u_max - window.u_min)
        span_w = max(1e-9, window.w_max - window.w_min)
        du0 = 1e-3 * span_u if eps_u is None else float(eps_u)
        dw0 = 1e-3 * span_w if eps_w is None else float(eps_w)
    else:
        du0 = 1e-2 if eps_u is None else float(eps_u)
        dw0 = 1e-4 if eps_w is None else float(eps_w)
    
    #Initial conditions along the eigendirection
    y_plus = np.array([u_s + du0 * v[0], w_s + dw0 * v[1]], dtype=float)
    y_minus = np.array([u_s - du0 * v[0], w_s - dw0 * v[1]], dtype=float)

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        return np.asarray(ml_rhs(t, y, I_ext, par), dtype=float)

    events = None
    if window is not None:

        def leave_window(t: float, y: np.ndarray) -> float:
            u, w = float(y[0]), float(y[1])
            inside = window.contains(u, w, margin=float(exit_margin))
            # Must NOT be zero at start when inside.
            # Positive inside, negative outside -> zero crossing at the boundary.
            return 1.0 if inside else -1.0

        leave_window.terminal = True
        leave_window.direction = -1  # trigger when going from + to -
        events = leave_window

    branches: list[SeparatrixBranch] = []
    
    #Integrate backwards
    for y0 in (y_plus, y_minus):
        sol = solve_ivp(
            rhs,
            t_span=(0.0, -float(T)),  # backward time
            y0=y0,
            max_step=float(max_step),
            rtol=float(rtol),
            atol=float(atol),
            events=events,
        )

        if sol.y is None or sol.y.shape[1] < 2:
            continue

        u_path = np.asarray(sol.y[0, :], dtype=float)
        w_path = np.asarray(sol.y[1, :], dtype=float)

        # Filter non-finite points (rare but possible if integration blows up)
        mask = np.isfinite(u_path) & np.isfinite(w_path)
        u_path = u_path[mask]
        w_path = w_path[mask]

        if u_path.size >= 2:
            branches.append(SeparatrixBranch(u=u_path, w=w_path))

    if not branches:
        return None

    return SeparatrixResult(saddle_u=u_s, saddle_w=w_s, branches=branches)