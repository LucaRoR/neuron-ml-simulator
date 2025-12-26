import numpy as np
from .parameters import MLParameters

#We avoid division by zero
def _safe_slope(n: float, eps: float = 1e-3):
    if abs(n) < eps:
        return eps
    return n

#Steady-state functions and time constants
def meff_inf(u: float, par: MLParameters) -> float:
    V2_meff = _safe_slope(par.V2_meff)
    return 0.5 * (1 + np.tanh((u - par.V1_meff) / V2_meff))

def w_inf(u: float, par: MLParameters) -> float:
    V4_w = _safe_slope(par.V4_w)
    return 0.5 * (1 + np.tanh((u - par.V3_w) / V4_w))

def tau_w(u: float, par: MLParameters) -> float:
    V4_w = _safe_slope(par.V4_w)
    return 1 / (2 * par.Phi_w * np.cosh((u - par.V3_w) / V4_w))


#Ionic currents
def I_Na(u: float, par:MLParameters) -> float:
    return par.g_Na * meff_inf(u, par) * (u - par.E_Na)

def I_K(u: float, w: float, par:MLParameters) -> float:
    return par.g_K * w * (u - par.E_K)

def I_L(u: float, par:MLParameters) -> float:
    return par.g_L * (u - par.E_L)

#ODEs left-hand-side
def f(u: float, w:float, I_ext:float, par: MLParameters) -> float:
    return (I_ext - I_Na(u, par) - I_K(u, w, par) - I_L(u, par)) / par.C

def g(u:float, w:float, par: MLParameters) -> float:
    return (w_inf(u, par) - w) / tau_w(u, par)

#ML model right-hand-side
def ml_rhs(t:float, y:np.ndarray, I_ext:float, par:MLParameters) -> np.ndarray:
    u,w = y
    du_dt = f(u, w, I_ext, par)
    dw_dt = g(u, w, par)
    return np.array([du_dt, dw_dt], dtype=float)

#Jacobian matrix
def jacobian(u:float, w:float, I_ext:float, par:MLParameters) -> np.ndarray:
    eps = 1e-7
    #Partial derivatives for f
    f0 = f(u, w, I_ext, par)
    df_du = (f(u + eps, w, I_ext, par) - f0) / eps
    df_dw = (f(u, w + eps, I_ext, par) - f0) / eps

    #Partial derivatives for g
    g0 = g(u, w, par)
    dg_du = (g(u + eps, w, par)- g0) / eps
    dg_dw = (g(u, w + eps, par) - g0) / eps

    return np.array([[df_du, df_dw],
                    [dg_du,dg_dw]], dtype = float)

                    