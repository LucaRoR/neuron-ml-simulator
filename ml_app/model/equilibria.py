from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from scipy.optimize import brentq

from .ml_equations import f, w_inf, jacobian
from .parameters import MLParameters

@dataclass(frozen=True)
class Equilibrium:
    u: float
    w: float
    eigvals: Optional[np.ndarray] = None #array to allow complex values
    stability: Optional[str] = None #"stable focus/node", "unstable focus/node", "saddle", "other_(Hopf_or_center)"


def _F(u: float, I_ext: float, par: MLParameters): #we replace the w variable from f with the steady-state function
    return f(u, w_inf(u, par), I_ext, par)

#Mathematically, when F(u(I_ext))=0, we have an equilibrium.

def find_equilibria(
        I_ext: float,
        par: MLParameters,
        *,
        u_min: float = -90.0,
        u_max: float = 60.0,
        n_scan: int = 5001,
        mr_tol: float = 1e-4,
        classify: bool = True,
) -> List[Equilibrium]:
    """
    Find equilibria (u*,w*) for a fixed I_ext.
    We scan u in [u_min, u_max] and look for sign changes of F(u).
    We use brentq to implement Brent's method to find a root u* of F in the interval.
    We set w*=w_inf(u*).
    We optionally classify the equilibrium with the eigenvalues of the Jacobian.
    """

    if u_max <= u_min:
        raise ValueError("u_max must be > u_min")
    if n_scan < 10:
        raise ValueError("n_scan too small. Use at least ~1000 for reliability.")
    
    us = np.linspace(u_min, u_max, n_scan)
    Fs = np.array([_F(float(u), I_ext, par) for u in us], dtype=float)

    roots: List[float] = []

    #Define brackets to check
    for i in range(len(us)-1):
        u0,u1 = float(us[i]), float(us[i+1])
        F0, F1 = float(Fs[i]),float(Fs[i+1])

        #If the value is virtually 0, keep it
        if np.isfinite(F0) and abs(F0) < 1e-10:
            roots.append(u0)
            continue

        #We want to avoid the program breaking for np.nan
        if not (np.isfinite(F0) and np.isfinite(F1)):
            continue
        
        #Sign change means there is a root in the interval. We use Brent's method to determine the value.
        if F0 * F1 < 0.0:
            try:
                r = brentq(lambda x: _F(x, I_ext, par), u0, u1, maxiter=200)
                roots.append(float(r))
            #brentq can break if the function is weird in the interval. Shouldn't happen for our model.
            except ValueError:
                pass
    
    #Merge possible duplicates
    roots = sorted(roots)
    merged: List[float] = []
    for r in roots:
        #If the difference between the identified roots is < rtol, we count them as the same root.
        if not merged or abs(r - merged[-1]) > mr_tol:
            merged.append(r)
    
    eqs: List[Equilibrium] = []
    for u_star in merged:
        w_star = float(w_inf(u_star, par))

        if classify:
            #Compute the eigenvalues of the Jacobian
            J = jacobian(u_star, w_star, I_ext, par)
            eig = np.linalg.eigvals(J)
        
            #Classification
            re = np.real(eig)
            im = np.imag(eig)

            if np.all(re < 0):
                stab = "stable_focus" if np.any(np.abs(im) > 1e-10) else "stable_node"
            elif np.all(re > 0):
                stab = "unstable_focus" if np.any(np.abs(im) > 1e-10) else "unstable_node"
            elif re[0] * re[1] < 0:
                stab = "saddle"
            else:
                stab = "other_(Hopf_or_center)"

            eqs.append(Equilibrium(u=u_star, w=w_star, eigvals=eig, stability=stab))
        else:
            eqs.append(Equilibrium(u=u_star, w=w_star))

    return sorted(eqs, key=lambda e: e.u)