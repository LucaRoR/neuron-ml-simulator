from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.integrate import solve_ivp

from .parameters import MLParameters
from .ml_equations import ml_rhs, w_inf
from .equilibria import find_equilibria

@dataclass(frozen=True)
class LimitCycleResult:
    period_ms: float
    u: np.ndarray
    w: np.ndarray
    section_u: float
    crossings_t: np.ndarray

def compute_limit_cycle(
    I_ext: float,
    par: MLParameters,
    *,
    u_min: float = -100.0,
    u_max: float = 60.0,
    # integration horizon (ms)
    T_total: float = 1500.0,
    T_transient: float = 800.0,
    # sampling along one period
    n_samples: int = 1200,
    # solver controls
    max_step: float = 0.5,
    rtol: float = 1e-6,
    atol: float = 1e-9,
    # periodicity check
    min_crossings: int = 6,
    cv_tol: float = 0.02,   # coefficient of variation threshold for period stability
    state_tol = 1e-2,
    amp_min_u: float = 1.0,
    amp_rel_tol: float = 0.05,
    amp_decay_tol: float = 0.10,
    n_amp_periods: int = 4,
    # section choice
    u_section: Optional[float] = None,
    # initial condition
    y0: Optional[np.ndarray] = None,
    seed_cycle: Optional[LimitCycleResult] = None,
) -> Optional[LimitCycleResult]:
    I_ext = float(I_ext)

    if u_section is None:
        # prefer an unstable equilibrium's u if present
        eqs = find_equilibria(I_ext, par, u_min=u_min, u_max=u_max, n_scan=3001, classify=True)
        u_sec = None
        for e in eqs:
            s = (e.stability or "").lower()
            if "unstable" in s:
                u_sec = float(e.u)
                break
        if u_sec is None:
            u_sec = 0.5 * (u_min + u_max)
        u_section = float(u_sec)
    else:
        u_section = float(u_section)
    
    if y0 is None and seed_cycle is not None and seed_cycle.u.size > 10:

        u_prev = np.asarray(seed_cycle.u, dtype=float)
        w_prev = np.asarray(seed_cycle.w, dtype=float)

        # Choose the point closest to the section u=u_section
        k = int(np.argmin(np.abs(u_prev - u_section)))

        # Small nudge to avoid landing exactly on the section
        eps_u = 1e-3
        y0 = np.array([u_prev[k] + eps_u, w_prev[k]], dtype=float)

    elif y0 is None:
        # Cold start fallback
        y0 = np.array([u_section + 1e-2, float(w_inf(u_section, par)) + 1e-4], dtype=float)
    else:
        y0 = np.asarray(y0, dtype=float).reshape(2)
    
    def poincare_event(t: float, y: np.ndarray) -> float:
        return float(y[0]) - u_section
    poincare_event.direction = 1.0
    poincare_event.terminal = False

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        return np.asarray(ml_rhs(t, y, I_ext, par), dtype=float)
    
    sol = solve_ivp(
        rhs,
        t_span=(0.0, float(T_total)),
        y0=y0,
        events=poincare_event,
        dense_output=True,
        max_step=float(max_step),
        rtol=float(rtol),
        atol=float(atol),
    )
    if (not sol.success) or (sol.t_events is None) or (len(sol.t_events) == 0):
        return None

    te = np.asarray(sol.t_events[0], dtype=float)
    te = te[te >= float(T_transient)]
    if te.size < min_crossings:
        return None
    
    # Check period stability
    periods = np.diff(te)
    if periods.size < (min_crossings - 1):
        return None

    tail = periods[-min(10, periods.size):]  # last intervals
    meanT = float(np.mean(tail))
    if not np.isfinite(meanT) or meanT <= 0:
        return None

    cv = float(np.std(tail) / meanT) if meanT > 0 else np.inf
    if cv > cv_tol:
        # likely not converged to a limit cycle
        return None
    
    t0 = float(te[-2])
    t1 = float(te[-1])
    T = t1 - t0
    if T <= 0:
        return None

    ts = np.linspace(t0, t1, int(n_samples), dtype=float)
    Y = sol.sol(ts)

    u = np.asarray(Y[0, :], dtype=float)
    w = np.asarray(Y[1, :], dtype=float)

    # filter finite points
    m = np.isfinite(u) & np.isfinite(w)
    u = u[m]; w = w[m]
    if u.size < 10:
        return None
    
    y0 = np.asarray(sol.sol(t0), dtype=float).reshape(2)
    y1 = np.asarray(sol.sol(t1), dtype=float).reshape(2)

    du = (y1[0] - y0[0]) / 10.0
    dw = (y1[1] - y0[1]) / 0.05
    closure = np.hypot(du, dw)

    if closure > state_tol:
        return None

    k = min(n_amp_periods, periods.size)

    if k >= 2:
        amps = []

        for j in range(k):
            a0 = te[-(k + 1 - j)]
            a1 = te[-(k - j)]

            if a1 <= a0:
                continue

            tseg = np.linspace(a0, a1, 250)
            Yseg = sol.sol(tseg)

            useg = np.asarray(Yseg[0, :])
            amps.append(np.max(useg) - np.min(useg))

        amps = np.asarray(amps)

        if amps.size >= 2:

            if amps[-1] < amp_min_u:
                return None

            meanA = np.mean(amps)

            if meanA <= 0 or np.std(amps) / meanA > amp_rel_tol:
                return None

            if amps[-1] < (1 - amp_decay_tol) * np.max(amps):
                return None

    return LimitCycleResult(
        period_ms=float(T),
        u=u,
        w=w,
        section_u=float(u_section),
        crossings_t=np.asarray(te, dtype=float),
    )