from dataclasses import dataclass
from .least_square_fit_Na import compute_best_fit
@dataclass

class MLParameters:
    #m effective/h,m toggle
    meff_toggle: bool = True

    #Membrane capacitance
    C: float = 1 #muF/cm^2

    #Maximal conductances
    g_Na: float = 120.0 #mS/cm^2
    g_K: float = 40.0 #mS/cm^2
    g_L: float = 0.2 #mS/cm^2

    #Reversal potentials
    E_Na: float = 55.0 #mV
    E_K: float = -80.0 #mV
    E_L: float = -65.0 #mV

    #Steady-state Boltzmann parameters
    #m_inf(u) = 0.5 * (1 + tanh((u-V1) / V2))
    V1_m: float = -25.1 #mV
    V2_m: float = 23.0 #mV

    #h_inf(u) = 0.5 * (1 - tanh((u-V1) / V2))
    V1_h: float = -58.3 #mV
    V2_h: float = 13.4 #mV

    #w_inf(u) = 0.5 * (1 + tanh((u-V3) / V4))
    V3_w: float = 4.54 #mV
    V4_w: float = 39.7 #mV

    #m_eff(u) =  0.5 * (1 + tanh((u-V1) / V2))  
    V1_meff: float = -44.5 #mV
    V2_meff: float = 10.7 #mV
    
    #Time-constant of w parameters
    #tau_w(u) = 1 / (2 * phi * cosh((u-V3) / V4))
    Phi_w: float = 0.1 #mHz