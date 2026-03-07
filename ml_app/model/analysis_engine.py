from __future__ import annotations

from dataclasses import dataclass
from collections import OrderedDict
from typing import Optional, Tuple

import numpy as np

from .parameters import MLParameters
from .nullclines import u_nullcline, w_nullcline
from .equilibria import find_equilibria, Equilibrium
from .bifurcations import find_saddle_nodes, find_hopf_points

@dataclass(frozen=True)
class NullclineQuery:
    I_ext: float
    par_key: Tuple
    u_min: float
    u_max: float
    n_u: int


@dataclass(frozen=True)
class EquilibriaQuery:
    I_ext: float
    par_key: Tuple
    u_min: float
    u_max: float
    n_scan: int
    mr_tol: float
    classify: bool


@dataclass(frozen=True)
class BifurcationQuery:
    par_key: Tuple
    u_min: float
    u_max: float
    n_scan: int


#Memory safety ceiling
class _LRU:
    def __init__(self, maxsize: int = 8) -> None:
        self._maxsize = int(maxsize)
        self._d: "OrderedDict[object, object]" = OrderedDict()

    def get(self, key: object) -> object | None:
        if key in self._d:
            self._d.move_to_end(key)
            return self._d[key]
        return None

    def put(self, key: object, val: object) -> None:
        self._d[key] = val
        self._d.move_to_end(key)
        while len(self._d) > self._maxsize:
            self._d.popitem(last=False)

    def clear(self) -> None:
        self._d.clear()

        
class AnalysisEngine:
    def __init__(
        self,
        *,
        max_cache: int = 8,
    ) -> None:
        self._null_cache = _LRU(max_cache)
        self._eq_cache = _LRU(max_cache)
        self._bif_cache = _LRU(max_cache)

    @staticmethod
    def par_key(par: MLParameters) -> Tuple:
        return tuple(getattr(par, f) for f in par.__dataclass_fields__.keys())

    def nullclines(
        self,
        *,
        I_ext: float,
        par: MLParameters,
        u_min: float,
        u_max: float,
        n_u: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        q = NullclineQuery(
            I_ext=float(I_ext),
            par_key=self.par_key(par),
            u_min=float(u_min),
            u_max=float(u_max),
            n_u=int(n_u),
        )
        cached = self._null_cache.get(q)
        if cached is not None:
            u, w_w, w_u = cached
            return (u.copy(), w_w.copy(), w_u.copy())

        u = np.linspace(q.u_min, q.u_max, q.n_u)
        w_w = w_nullcline(u, par)
        w_u = u_nullcline(u, I_ext, par)

        res = (u, w_w, w_u)
        self._null_cache.put(q, res)
        return res
    
    def equilibria(
        self,
        *,
        I_ext: float,
        par: MLParameters,
        u_min: float,
        u_max: float,
        n_scan: int,
        mr_tol: float = 1e-4,
        classify: bool = True,
    ) -> list[Equilibrium]:
        q = EquilibriaQuery(
            I_ext=float(I_ext),
            par_key=self.par_key(par),
            u_min=float(u_min),
            u_max=float(u_max),
            n_scan=int(n_scan),
            mr_tol=float(mr_tol),
            classify=bool(classify),
        )
        cached = self._eq_cache.get(q)
        if cached is not None:
            return list(cached)

        eqs = find_equilibria(
            I_ext=I_ext,
            par=par,
            u_min=u_min,
            u_max=u_max,
            n_scan=n_scan,
            mr_tol=mr_tol,
            classify=classify,
        )
        self._eq_cache.put(q, eqs)
        return eqs
    
    def bifurcations(
        self,
        *,
        par: MLParameters,
        u_min: float,
        u_max: float,
        n_scan: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        q = BifurcationQuery(
            par_key=self.par_key(par),
            u_min=float(u_min),
            u_max=float(u_max),
            n_scan=int(n_scan),
        )
        cached = self._bif_cache.get(q)
        if cached is not None:
            sns, hopf = cached
            return (sns.copy(), hopf.copy())
        
        sns = find_saddle_nodes(par=par, u_min=u_min, u_max=u_max, n_scan=n_scan)
        hopf = find_hopf_points(par=par, u_min=u_min, u_max=u_max, n_scan=n_scan)

        res = (sns, hopf)
        self._bif_cache.put(q, res)
        return res
    
    def clear(self) -> None:
        self._null_cache.clear()
        self._eq_cache.clear()
        self._bif_cache.clear()