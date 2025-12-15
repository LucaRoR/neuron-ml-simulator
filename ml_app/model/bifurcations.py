from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, List, Tuple, Optional

import numpy as np
from scipy.optimize import brentq

from .ml_equations import w_inf, jacobian, meff_inf
from .parameters import MLParameters

def _solve_I_ext (u: float, par: MLParameters) -> float: #we will use this to find the value of I_ext from an equilibrium (u*, w_inf(u*))
    return (
        par.g_Na * meff_inf(u, par) * (u - par.E_Na)
        + par.g_K  * w_inf(u, par) * (u - par.E_K)
        + par.g_L  * (u - par.E_L)
    )

@dataclass (frozen=True)
class ManifoldPoint:
    u: float
    w: float
    I_ext : float
    J: np.ndarray
    tr: float
    det: float
    eig: np.ndarray

def manifold_point(u: float, par: MLParameters) -> ManifoldPoint: #we evaluate every value we need from u
    w = float(w_inf(u, par))
    I_ext = float(_solve_I_ext(u, par))
    J = jacobian(u, w, I_ext, par)
    tr = float(np.trace(J))
    det = float(np.linalg.det(J))
    eig = np.linalg.eigvals(J)
    return ManifoldPoint(u, w, I_ext, J, tr, det, eig)

def _brackets_over_u(
        us: np.ndarray,
        phi: Callable[[float], float],
        *,
        finite_only: bool = True,
) -> List[Tuple[float, float]]:
    """Given a function phi(u), find intervals [u0, u1] in which phi(u) changes sign or hits values very close to 0"""

    vals = np.array([phi(float(u)) for u in us], dtype=float)

    brackets: List[Tuple[float, float]] = []
    eps = 1e-8

    for i in range(len(us) - 1):
        u0, u1 = float(us[i]), float(us[i + 1])
        f0, f1 = float(vals[i]), float(vals[i + 1])

        if finite_only and (not np.isfinite(f0) or not np.isfinite(f1)):
            continue

        #if either endpoint is very close to 0, add the interval to the list
        if abs(f0) < eps or abs(f1) < eps:
            brackets.append((u0, u1))
            continue

        if f0 * f1 < 0.0:
            brackets.append((u0, u1))
    
    return brackets

@dataclass(frozen=True)
class SaddleNode:
    u: float
    I_ext: float
    tr: float

def find_saddle_nodes(
        par: MLParameters,
        *,
        u_min: float = -100.0,
        u_max: float = 80.0,
        n_scan: int = 5001,
        tr_tol: float = 1e-6,
        mr_tol: float = 1e-4
) -> List[SaddleNode]:
    """Determine the presence of saddle nodes. We find values of I_ext for which det J changes sign."""
    if u_max <= u_min:
        raise ValueError("u_max must be > u_min")
    if n_scan < 10:
        raise ValueError("n_scan too small. Use at least ~1000 for reliability.")
    
    us = np.linspace(u_min, u_max, n_scan)

    def det_on_manifold(u: float) -> float: #exctract the determinant from manifold_point for simplicity
        mp = manifold_point(u, par)
        return mp.det

    brackets = _brackets_over_u(us, det_on_manifold)
    sns: List[SaddleNode] = []
    for (a,b) in brackets:
        try:
            u_sn = brentq(det_on_manifold, a, b, maxiter=200)
        #brentq can break if the function is weird in the interval. Shouldn't happen for our model.
        except ValueError:
            continue

        mp = manifold_point(u_sn, par)

        #sn requires one eigenvalue to be 0, the other non-zero. We enforce this.
        if abs(mp.tr) < tr_tol:
            continue

        sns.append(SaddleNode(u=mp.u, I_ext=mp.I_ext, tr = mp.tr))

    #Merge possible duplicates
    sns = sorted(sns, key=lambda sn: sn.I_ext)
    merged: List[SaddleNode] = []

    for sn in sns:
        if not merged or abs(sn.I_ext - merged[-1].I_ext) > mr_tol:
            merged.append(sn)
    
    return merged

@dataclass(frozen=True)
class HopfPoint:
    u: float
    I_ext: float
    det: float

def find_hopf_points(
        par: MLParameters,
        *,
        u_min: float = -100.0,
        u_max: float = 80.0,
        n_scan: int = 5001,
        det_tol: float = 1e-6,
        mr_tol: float = 1e-4
) -> List[HopfPoint]:
    """Determine the presence of Hopf bifurcations. We find values of I_ext for which det J > 0 and tr J = 0."""
    if u_max <= u_min:
        raise ValueError("u_max must be > u_min")
    if n_scan < 10:
        raise ValueError("n_scan too small. Use at least ~1000 for reliability.")
    
    us = np.linspace(u_min, u_max, n_scan)

    def tr_on_manifold(u: float) -> float:
        return manifold_point(u, par).tr
    
    brackets = _brackets_over_u(us, tr_on_manifold)
    hopf: List[HopfPoint] = []

    for (a,b) in brackets:
        #We only want intervals for which det J > 0. Assuming that the intervals are sufficiently small, we only check the midpoint of such intervals.
        mid = 0.5 * (a + b)
        if manifold_point(mid, par).det <= det_tol:
            continue

        try:
            u_h = brentq(tr_on_manifold, a, b, maxiter=200)
        #brentq can break if the function is weird in the interval. Shouldn't happen for our model.
        except ValueError:
            continue
    
        mp = manifold_point(u_h, par)
        if mp.det <= det_tol:
            continue

        hopf.append(HopfPoint(u=mp.u, I_ext=mp.I_ext, det=mp.det))

    hopf = sorted(hopf, key=lambda hp: hp.I_ext)
    merged: List[HopfPoint] = []
    for hp in hopf:
        if not merged or abs(hp.I_ext - merged[-1].I_ext) > mr_tol:
            merged.append(hp)
    
    return merged