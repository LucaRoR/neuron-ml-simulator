# ml_app/model/test_bifurcations.py
from __future__ import annotations

import numpy as np

# Adjust these imports to your project structure:
from ml_app.model.parameters import MLParameters
from ml_app.model.bifurcations import (
    manifold_point,
    find_saddle_nodes,
    find_hopf_points,
)

def _fmt_complex(z: complex) -> str:
    return f"{z.real:+.6e}{z.imag:+.6e}j"

def main() -> None:
    par = MLParameters()

    # --- Scan settings (tweak if needed) ---
    u_min, u_max = -100.0, 80.0
    n_scan = 5001

    print("\n=== Testing saddle-node detection ===")
    sns = find_saddle_nodes(
        par,
        u_min=u_min,
        u_max=u_max,
        n_scan=n_scan,
        tr_tol=1e-6,
        mr_tol=1e-4,
    )
    if not sns:
        print("No saddle-node candidates found in the scan window.")
    else:
        for k, sn in enumerate(sns, start=1):
            mp = manifold_point(sn.u, par)
            print(f"\nSN #{k}")
            print(f"  u*      = {mp.u:.6f} mV")
            print(f"  w*      = {mp.w:.6f}")
            print(f"  I_ext   = {mp.I_ext:.6f}")
            print(f"  det(J)  = {mp.det:.6e}   (should be ~0)")
            print(f"  tr(J)   = {mp.tr:.6e}   (should be != 0)")
            print(f"  eig(J)  = [{_fmt_complex(mp.eig[0])}, {_fmt_complex(mp.eig[1])}]")

    print("\n=== Testing Hopf detection ===")
    hopfs = find_hopf_points(
        par,
        u_min=u_min,
        u_max=u_max,
        n_scan=n_scan,
        det_tol=1e-6,
        mr_tol=1e-4,
    )
    if not hopfs:
        print("No Hopf candidates found in the scan window.")
    else:
        for k, hp in enumerate(hopfs, start=1):
            mp = manifold_point(hp.u, par)
            print(f"\nHopf #{k}")
            print(f"  u*      = {mp.u:.6f} mV")
            print(f"  w*      = {mp.w:.6f}")
            print(f"  I_ext   = {mp.I_ext:.6f}")
            print(f"  tr(J)   = {mp.tr:.6e}   (should be ~0)")
            print(f"  det(J)  = {mp.det:.6e}   (should be > 0)")
            print(f"  eig(J)  = [{_fmt_complex(mp.eig[0])}, {_fmt_complex(mp.eig[1])}]")

            # Check stability side (useful sanity check)
            # For det>0, sign(tr) distinguishes stable/unstable focus/node.
            side = "stable (tr<0)" if mp.tr < 0 else "unstable (tr>0)"
            print(f"  local type side: {side}")

    # --- Optional: quick plots of det and tr over u ---
    try:
        import matplotlib.pyplot as plt

        us = np.linspace(u_min, u_max, n_scan)
        dets = np.array([manifold_point(float(u), par).det for u in us])
        trs  = np.array([manifold_point(float(u), par).tr  for u in us])

        plt.figure()
        plt.plot(us, dets)
        plt.axhline(0.0)
        plt.title("det(J) along equilibrium manifold")
        plt.xlabel("u (mV)")
        plt.ylabel("det(J)")

        plt.figure()
        plt.plot(us, trs)
        plt.axhline(0.0)
        plt.title("tr(J) along equilibrium manifold")
        plt.xlabel("u (mV)")
        plt.ylabel("tr(J)")

        plt.show()

    except ImportError:
        print("\n(matplotlib not installed; skipping plots)")

if __name__ == "__main__":
    main()
