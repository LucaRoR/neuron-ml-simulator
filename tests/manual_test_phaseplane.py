from __future__ import annotations

import sys

from PyQt6.QtWidgets import QApplication, QMainWindow

from ml_app.model.parameters import MLParameters
from ml_app.gui.phaseplane_canvas import PhasePlaneCanvas, PhasePlaneView


def main() -> int:
    app = QApplication(sys.argv)

    # --- Create your model state ---
    par = MLParameters()
    I_ext = -1000.0

    # --- Create the canvas ---
    view = PhasePlaneView(
        u_min=-90.0,
        u_max=60.0,
        w_min=0.0,
        w_max=1.0,
        show_vector_field=True,
        show_nullclines=True,
        show_equilibria=True,
    )
    canvas = PhasePlaneCanvas(view=view)
    canvas.set_state(par, I_ext)

    # --- Put it into a window ---
    win = QMainWindow()
    win.setWindowTitle("PhasePlaneCanvas smoke test")
    win.setCentralWidget(canvas)
    win.resize(1000, 700)
    win.show()

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
