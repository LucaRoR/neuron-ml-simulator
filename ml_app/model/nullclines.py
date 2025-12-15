from __future__ import annotations
import numpy as np
from .ml_equations import w_inf, meff_inf
from .parameters import MLParameters

def w_nullcline(u: np.ndarray, par: MLParameters) -> np.ndarray: #the w-nullcline is defined by dw/dt=0. It is equal to the steady-state function w_inf
    return w_inf(u, par)

def u_nullcline(u: np.ndarray, I_ext: float, par: MLParameters) -> np.ndarray: #the u-nullcline is defined by du/dt=0.
    u = np.asarray(u, dtype=float)

    numerator = (
        I_ext - par.g_Na * meff_inf(u, par) * (u - par.E_Na) - par.g_L * (u - par.E_L)
    )

    denominator = par.g_K * (u - par.E_K)


    #Mask the vertical asymptote u = E_K
    eps = 1e-6
    w = np.full_like(u, np.nan, dtype=float)

    mask = np.abs(denominator) > eps

    w[mask] = numerator[mask] / denominator[mask] #compute the u-nullcline

    #Mask meaningless values of w
    w[(w<0.0) | (w>1.0)] = np.nan
    return w