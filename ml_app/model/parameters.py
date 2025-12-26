from dataclasses import dataclass

@dataclass
class MLParameters:
    #Membrane capacitance
    C: float = 20.0 #muF/cm^2

    #Maximal conductances
    g_Na: float = 4.0 #mS/cm^2
    g_K: float = 8.0 #mS/cm^2
    g_L: float = 2.0 #mS/cm^2

    #Reversal potentials
    E_Na: float = 120.0 #mV
    E_K: float = -84.0 #mV
    E_L: float = -60.0 #mV

    #Steady-state Boltzmann parameters
    #m_eff(u) =  0.5 * (1 + tanh((u-V1) / V2))  
    V1_meff: float = -1.2 #mV
    V2_meff: float = 18.0 #mV

    #w_inf(u) = 0.5 * (1 + tanh((u-V3) / V4))
    V3_w: float = 12.0 #mV
    V4_w: float = 17.4 #mV
    
    #Time-constant of w parameters
    #tau_w(u) = 1 / (2 * phi * cosh((u-V3) / V4))
    Phi_w: float = 0.067 #mHz