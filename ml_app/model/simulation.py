from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Dict, Any

import numpy as np
from scipy.integrate import solve_ivp

from .ml_equations import ml_rhs
from .parameters import MLParameters

@dataclass
class SimulationConfig:
    t0: float = 0.0
    t1: float = 200.0
    dt: float = 0.05

    method: str = "RK45" #Runge-Kutta-Fehlberg order 4(5)
    rtol: float = 1e-6 #relative error tollerance
    atol: float = 1e-9 #absolute error tollerance
    max_step: float = np.inf #we may use this if the solver works funny with fast dynamics

    spike_threshold: Optional[float] = None #we may use this if we want to stop the integration when a certain value is exceeded
    stop_on_spike: bool = False

@dataclass
class SimulationResult:
    t: np.ndarray #times returned by solver
    y: np.ndarray #solutions array. y[0,:] is u(t); y[1,:] is w(t)
    success: bool #did the solver finish successfully?
    message: str #explanation if it fails
    nfev: int #number of RHS evaluations (see ml_equations.ml_rhs)
    events: Dict[str, np.ndarray] #event times (spikes, etc.)
    raw: Any #full solve_ivp object

def _make_spike_event(threshold: float) -> Callable[[float, np.ndarray], float]: #event function
    def event(t: float, y: np.ndarray) -> float:
        u = float(y[0])
        return u - threshold
    
    event.direction = 1.0 #the function must go from negative to positive
    event.terminal = False #this is overridden by stop_on_spike
    return event

def simulate(y0: np.ndarray, I_ext: float, par: MLParameters, config: Optional[SimulationConfig]=None) -> SimulationResult:
    if config is None:
        config = SimulationConfig()
    y0 = np.asarray(y0, dtype=float).reshape(-1) #y0 becomes a 1D float array
    if y0.shape[0] != 2:
        raise ValueError(f"y0 must have shape (2,), got {y0.shape}")
    
    #create uniform output grid for GUI
    t_eval = np.arange(config.t0, config.t1 + config.dt, config.dt, dtype = float)

    #solve_ivp expecets a f(t,y) function, therefore we capture I_ext 
    def fun(t: float, y:np.ndarray) -> np.ndarray:
        return ml_rhs(t, y, I_ext, par)

    events_list = []
    events_names = []

    if config.spike_threshold is not None:
        spike_event = _make_spike_event(config.spike_threshold)
        spike_event.terminal = bool(config.stop_on_spike)
        events_list.append(spike_event)
        events_names.append("spikes")

    sol = solve_ivp(
        fun = fun,
        t_span=(config.t0, config.t1),
        y0=y0,
        method=config.method,
        t_eval = t_eval,
        rtol = config.rtol,
        atol = config.atol,
        max_step = config.max_step,
        events = events_list if events_list else None,
        vectorized=False,
        dense_output=False
    )

    events: Dict[str, np.ndarray] = {}

    if events_list:
        for name, te in zip(events_names, sol.t_events):
            events[name] = np.asarray(te, dtype=float)

    return SimulationResult(
        t=np.asarray(sol.t, dtype = float),
        y=np.asarray(sol.y, dtype = float),
        success=bool(sol.success),
        message=str(sol.message),
        nfev=int(sol.nfev),
        events=events,
        raw=sol
    )

def simulate_many(y0_list: np.ndarray, I_ext: float, par: MLParameters, config: Optional[SimulationConfig]=None) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    if config is None: #to avoid problems with the GUI mutating config
        config = SimulationConfig()
    y0_list = np.asarray(y0_list, dtype=float) #initial conditions
    if y0_list.ndim != 2 or y0_list.shape[1] != 2: #initial conditions must be a 2D vector
        raise ValueError("y0_list must have shape (K,2)")
    
    Y_list = []
    spikes_list = []
    t_ref = None

    for k, y0 in enumerate(y0_list):
        res=simulate(y0=y0, I_ext=I_ext, par=par, config=config)
        if k == 0:
            t_ref = res.t
        Y_list.append(res.y)
        spikes_list.append(res.events.get("spikes", np.array([], dtype=float))) #only runs if res.events["spikes"] exists

    Y = np.stack(Y_list, axis=0) #stack into 3D array
    events = {"spikes": np.array(spikes_list, dtype=object)}
    return t_ref, Y, events


def spike_stats(res) -> dict:
    te = res.events.get("spikes", np.array([], dtype=float))
    te = np.asarray(te, dtype=float)

    out = {"n_spikes": int(te.size)}
    if te.size >= 1:
        out["latency_ms"] = float(te[0])
    else:
        out["latency_ms"] = None

    if te.size >= 2:
        isi = np.diff(te)
        out["mean_isi_ms"] = float(np.mean(isi))
        out["freq_hz"] = float(1000.0 / np.mean(isi))  # ms -> Hz
    else:
        out["mean_isi_ms"] = None
        out["freq_hz"] = None

    return out