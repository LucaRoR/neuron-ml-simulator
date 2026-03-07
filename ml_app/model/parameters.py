from __future__ import annotations
from dataclasses import dataclass, replace
from typing import Dict, Optional

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

@dataclass(frozen=True)
class MLPreset:
    key: str
    label: str
    description: str
    par: MLParameters
    I_ext: Optional[float] = None
    y0_u: Optional[float] = None
    y0_w: Optional[float] = None
    behavior: Optional[str] = None
    note: Optional[str] = None


ML_PRESETS: Dict[str, MLPreset] = {

    "snic": MLPreset(
        key="snic",
        label="SNIC",
        description="Physiological Morris–Lecar Type I / SNIC-style regime.",
        behavior="SNIC",
        par=MLParameters(
            C=20.0,
            g_Na=4.0,
            g_K=8.0,
            g_L=2.0,
            E_Na=120.0,
            E_K=-84.0,
            E_L=-60.0,
            V1_meff=-1.2,
            V2_meff=18.0,
            V3_w=12.0,
            V4_w=17.4,
            Phi_w=0.067,
        ),
        I_ext=0.0,
        y0_u=-60.0,
        y0_w=0.0,
        note="Standard physiological Type I / SNIC-style Morris–Lecar set.",
    ),

    "sn_bistable": MLPreset(
        key="sn_bistable",
        label="SN (bistable)",
        description="Morris–Lecar bistable SN regime.",
        behavior="SN",
        par=MLParameters(
            C=20.0,
            g_Na=4.0,
            g_K=8.0,
            g_L=2.0,
            E_Na=120.0,
            E_K=-80.0,
            E_L=-60.0,
            V1_meff=-1.2,
            V2_meff=18.0,
            V3_w=12.0,
            V4_w=5.7,
            Phi_w=0.067,
        ),
        I_ext=0.0,
        y0_u=-60.0,
        y0_w=0.0,
        note="Bistable integrator",
    ),

    "subcritical_hopf": MLPreset(
        key="subcritical_hopf",
        label="Subcritical Hopf",
        description="Canonical Morris–Lecar Hopf regime.",
        behavior="subcritical Hopf",
        par=MLParameters(
            C=20.0,
            g_Na=4.4,
            g_K=8.0,
            g_L=2.0,
            E_Na=120.0,
            E_K=-84.0,
            E_L=-60.0,
            V1_meff=-1.2,
            V2_meff=18.0,
            V3_w=2.0,
            V4_w=30.0,
            Phi_w=0.04,
        ),
        I_ext=0.0,
        y0_u=-60.0,
        y0_w=0.0,
        note="Ermentrout–Terman Hopf set.",
    ),

    "supercritical_hopf": MLPreset(
        key="supercritical_hopf",
        label="Supercritical Hopf",
        description="Dimensionless Morris–Lecar Set II (Campbell–Kobelevskiy).",
        behavior="supercritical Hopf",
        par=MLParameters(
            C=1.0,
            g_Na=0.5,
            g_K=2.0,
            g_L=0.5,
            E_Na=120.0,
            E_K=-84.0,
            E_L=-60.0,
            V1_meff=-1.2,
            V2_meff=18.0,
            V3_w=12.0,
            V4_w=17.4,    
            Phi_w=1.0 / 3.0,
        ),
        I_ext=0.0,
        y0_u=0.0,
        y0_w=0.0,
        note="Campbell–Kobelevskiy Set II.",
    ),
}


def get_ml_preset(key: str) -> Optional[MLPreset]:
    return ML_PRESETS.get(key)


def clone_ml_preset(key: str) -> Optional[MLPreset]:
    preset = get_ml_preset(key)
    if preset is None:
        return None
    return MLPreset(
        key=preset.key,
        label=preset.label,
        description=preset.description,
        par=replace(preset.par),
        I_ext=preset.I_ext,
        y0_u=preset.y0_u,
        y0_w=preset.y0_w,
        behavior=preset.behavior,
        note=preset.note,
    )