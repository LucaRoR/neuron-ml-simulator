import numpy as np

def compute_best_fit(V1_m, V2_m, V1_h, V2_h):
    # Voltage range in mV
    V = np.linspace(-80, 40, 4001)

    # Parameters:
    Vhalf_m = V1_m
    k_m = V2_m / 2.0

    Vhalf_h = V1_h
    k_h = V2_h / 2.0

    def m_inf(V):
        return 1.0 / (1.0 + np.exp((Vhalf_m - V) / k_m))

    def h_inf(V):
        return 1.0 / (1.0 + np.exp((V - Vhalf_h) / k_h))

    p_inf = m_inf(V)**3 * h_inf(V)

    mask = (V >= -80) & (V <= -20)
    V_fit = V[mask]
    p_fit = p_inf[mask]

    p_min = p_fit.min()
    p_max = p_fit.max()
    p_norm = (p_fit - p_min) / (p_max - p_min + 1e-12)

    U1_vals = np.linspace(-60, -30, 121)
    U2_vals = np.linspace(4, 16, 121)

    best_err = np.inf
    best_U1 = None
    best_U2 = None

    VV = V_fit[None, :]
    U2_arr = U2_vals[:, None]

    for U1 in U1_vals:
        U1_arr = U1
        preds = 0.5 * (1.0 + np.tanh((VV - U1_arr) / U2_arr))
        errs = np.sum((preds - p_norm[None, :])**2, axis=1)
        idx = np.argmin(errs)
        if errs[idx] < best_err:
            best_err = errs[idx]
            best_U1 = U1
            best_U2 = U2_vals[idx]

    return [best_U1, best_U2, best_err]
