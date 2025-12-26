# neuron-ml-simulator

**Status: Beta Version (v0.9.0)**

Interactive Python simulator for a modified Morris-Lecar model (Na/K-based), with phase-plane and time-series visualisation.

## Installation

Clone the repository:

```bash
git clone https://github.com/LucaRoR/neuron-ml-simulator.git
cd neuron-ml-simulator
```

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

## Run

From the repository root:

```bash
python3 main.py
```

## Features

- PyQt6 GUI:

  - Phase plane (vector field, nullclines, equilibria, bifurcations, separatrix)
  - Time series (voltage, recovery variable)
  - Parameter controls

- Model:
  - ML equations
  - Numerical simulation (SciPy `solve_ivp`)
  - Nullclines, equilibria, and basic bifurcation utilities

## Project structure

```text
neuron-ml-simulator/
├── main.py
├── README.md
├── requirements.txt
├── LICENSE
├── Paper/
│ └── A Dynamical Systems View of a Neuron.pdf
└── ml_app/
├── __init__.py
├── config.py
├── model/
│ ├── __init__.py
│ ├── parameters.py
│ ├── ml_equations.py
│ ├── simulation.py
│ ├── nullclines.py
│ ├── equilibria.py
│ ├── bifurcations.py
│ └── separatrix.py
├── gui/
│ ├── __init__.py
│ ├── main_window.py
│ ├── controls_panel.py
│ ├── phaseplane_canvas.py
│ └── timeseries_canvas.py
├── resources/
│ ├── __init__.py
│ ├── styles.qss
│ └── icons/
│ │ └── app_icon.svg
```

## Acknowledgements

This project is inspired by classical Hodgkin–Huxley and Morris–Lecar neuron models, but all code and the current implementation were independently written.

## License

[MIT](https://choosealicense.com/licenses/mit/)
