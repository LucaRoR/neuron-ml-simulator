"""This script allows the user to modify the parameters before and during the simulation. It implements a single object, GuiState, for the state of the GUI.
NB: This script imports heavily from the model directory. For documentation on what the variables mean, refer to the backend files."""

from __future__ import annotations

import numpy as np

from dataclasses import dataclass, replace, is_dataclass, fields
from typing import Optional, Dict, Any, Tuple

from PyQt6.QtCore import pyqtSignal, QTimer, Qt
from PyQt6.QtWidgets import(
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QDoubleSpinBox,
    QSpinBox,
    QCheckBox,
    QComboBox,
    QScrollArea
)

from ..model.parameters import MLParameters
from ..model.simulation import SimulationConfig
from .phaseplane_canvas import PhasePlaneView
from .timeseries_canvas import TimeSeriesView


@dataclass(frozen=True)
class GuiState:
    """We create a single object for the state of the GUI"""
    par: MLParameters
    I_ext: float
    y0_u: float
    y0_w: float
    sim: SimulationConfig
    phase_view: PhasePlaneView
    ts_view: TimeSeriesView

class ControlsPanel(QWidget):
    """ All GUI inputs are regulated in this class.
    MainWindow listens to stateChanged(state) and runRequested(state)
    """
    stateChanged = pyqtSignal(GuiState)
    runRequested = pyqtSignal(GuiState)

    def __init__(
            self,
            parent: Optional[QWidget] = None,
            *,
            i_par: Optional[MLParameters] = None, #this and the ones below, initial conditions 
            i_I_ext: float = 0.0,
            i_y0: Tuple[float, float] = (-60.0, 0.0),
            i_sim: Optional[SimulationConfig] = None,
            i_phase_view: Optional[PhasePlaneView] = None,
            i_ts_view: Optional[TimeSeriesView] = None,
            debounce_ms: int = 150, #ms
    ) -> None:
        super().__init__(parent)

        self._par = i_par if i_par is not None else MLParameters()
        self._I_ext = float(i_I_ext)
        self._y0_u = float(i_y0[0])
        self._y0_w = float(i_y0[1])
        self._sim = i_sim if i_sim is not None else SimulationConfig()
        self._phase_view = i_phase_view if i_phase_view is not None else PhasePlaneView()
        self._ts_view = i_ts_view if i_ts_view is not None else TimeSeriesView()

        #We want to avoid that the program spams redraws while the user drags sliders or spinboxes
        self._emit_timer = QTimer(self)
        self._emit_timer.setSingleShot(True)
        self._emit_timer.setInterval(int(debounce_ms))
        self._emit_timer.timeout.connect(self._emit_state_now)

        self._par_widgets: dict[str, QWidget] = {}
        self._par_labels: dict[str, QWidget] = {}

        self._build_ui()
        self._sync_widgets_from_state()
        self._emit_state_now()
    
    #-----------
    # Public API
    #-----------

    def current_state(self) -> GuiState:
        """Assigns the chosen parameters to GuiState"""
        return GuiState(
            par = self._par,
            I_ext=self._I_ext,
            y0_u=self._y0_u,
            y0_w=self._y0_w,
            sim=self._sim,
            phase_view=self._phase_view,
            ts_view=self._ts_view,
        )
    
    #----
    # UI
    #----

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(10)

        #Quick settings
        q_box = QGroupBox("Quick controls")
        q_layout = QFormLayout(q_box)

        self._I_ext_spin = QDoubleSpinBox()
        self._I_ext_spin.setRange(-5000.0, 5000.0)
        self._I_ext_spin.setDecimals(6)
        self._I_ext_spin.setSingleStep(1.0)
        self._I_ext_spin.valueChanged.connect(self._on_I_ext_changed)
        q_layout.addRow("I_ext", self._I_ext_spin)

        y0_row = QHBoxLayout()
        self._y0_u_spin = QDoubleSpinBox()
        self._y0_u_spin.setRange(-200.0, 200.0)
        self._y0_u_spin.setDecimals(4)
        self._y0_u_spin.setSingleStep(1.0)
        self._y0_u_spin.valueChanged.connect(self._on_y0_changed)

        self._y0_w_spin = QDoubleSpinBox()
        self._y0_w_spin.setRange(-2.0, 2.0)
        self._y0_w_spin.setDecimals(6)
        self._y0_w_spin.setSingleStep(0.01)
        self._y0_w_spin.valueChanged.connect(self._on_y0_changed)

        y0_row.addWidget(QLabel("u0"))
        y0_row.addWidget(self._y0_u_spin)
        y0_row.addSpacing(6)
        y0_row.addWidget(QLabel("w0"))
        y0_row.addWidget(self._y0_w_spin)
        q_layout.addRow("Initial conditions", y0_row)

        #Simulation button
        run_row = QHBoxLayout()
        self._run_btn = QPushButton("Compute")
        self._run_btn.clicked.connect(self._on_run_clicked)
        run_row.addWidget(self._run_btn)
        q_layout.addRow("", run_row)

        root.addWidget(q_box)

        #Simulation box (scrollable)
        sim_box = QGroupBox("Simulation")
        sim_outer = QVBoxLayout(sim_box)

        sim_scroll = QScrollArea()
        sim_scroll.setWidgetResizable(True)
        sim_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        sim_scroll.setMaximumHeight(240)

        sim_inner = QWidget()
        sim_layout = QFormLayout(sim_inner)
        sim_layout.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        sim_layout.setFormAlignment(Qt.AlignmentFlag.AlignTop)

        self._t1_spin = QDoubleSpinBox()
        self._t1_spin.setRange(0.1, 50000.0)
        self._t1_spin.setDecimals(3)
        self._t1_spin.setSingleStep(10.0)
        self._t1_spin.valueChanged.connect(self._on_sim_changed)
        sim_layout.addRow("t1 (ms)", self._t1_spin)

        self._dt_spin = QDoubleSpinBox()
        self._dt_spin.setRange(1e-4, 10.0)
        self._dt_spin.setDecimals(6)
        self._dt_spin.setSingleStep(0.01)
        self._dt_spin.valueChanged.connect(self._on_sim_changed)
        sim_layout.addRow("dt (ms)", self._dt_spin)

        self._method_com = QComboBox()
        self._method_com.addItems(["RK45", "RK23", "DOP853", "Radau", "BDF", "LSODA"]) #The user can choose the method: we want to avoid issues with stiff problems.
        self._method_com.currentTextChanged.connect(self._on_sim_changed)
        sim_layout.addRow("method", self._method_com)

        self._rtol_spin = QDoubleSpinBox()
        self._rtol_spin.setRange(1e-12, 1e-1)
        self._rtol_spin.setDecimals(12)
        self._rtol_spin.setSingleStep(1e-6)
        self._rtol_spin.valueChanged.connect(self._on_sim_changed)
        sim_layout.addRow("relative error tolerance", self._rtol_spin)

        self._atol_spin = QDoubleSpinBox()
        self._atol_spin.setRange(1e-15, 1e-1)
        self._atol_spin.setDecimals(15)
        self._atol_spin.setSingleStep(1e-9)
        self._atol_spin.valueChanged.connect(self._on_sim_changed)
        sim_layout.addRow("absolute error tolerance", self._atol_spin)

        self._max_step_spin = QDoubleSpinBox()
        self._max_step_spin.setRange(0.0, 1e9)
        self._max_step_spin.setDecimals(6)
        self._max_step_spin.setSingleStep(1.0)
        self._max_step_spin.valueChanged.connect(self._on_sim_changed)
        sim_layout.addRow("max step (ms)", self._max_step_spin)

        self._spike_thr_spin = QDoubleSpinBox()
        self._spike_thr_spin.setRange(-200.0, 200.0)
        self._spike_thr_spin.setDecimals(4)
        self._spike_thr_spin.setSingleStep(1.0)
        self._spike_thr_spin.valueChanged.connect(self._on_sim_changed)
        sim_layout.addRow("spike threshold (mV)", self._spike_thr_spin)

        self._stop_on_spike_chk = QCheckBox("Stop integration on spike")
        self._stop_on_spike_chk.stateChanged.connect(self._on_sim_changed)
        sim_layout.addRow("", self._stop_on_spike_chk)
        
        sim_scroll.setWidget(sim_inner)
        sim_outer.addWidget(sim_scroll)

        root.addWidget(sim_box)

        #Phase plane view controls
        phase_box = QGroupBox("Phase plane view")
        phase_layout = QFormLayout(phase_box)

        self._pp_u_min = QDoubleSpinBox()
        self._pp_u_max = QDoubleSpinBox()
        self._pp_w_min = QDoubleSpinBox()
        self._pp_w_max = QDoubleSpinBox()

        for sb in (self._pp_u_min, self._pp_u_max):
            sb.setRange(-500.0, 500.0)
            sb.setDecimals(3)
            sb.setSingleStep(5.0)
            sb.valueChanged.connect(self._on_phase_view_changed)
        
        for sb in (self._pp_w_min, self._pp_w_max):
            sb.setRange(-5.0, 5.0)
            sb.setDecimals(6)
            sb.setSingleStep(0.05)
            sb.valueChanged.connect(self._on_phase_view_changed)
        
        u_row = QHBoxLayout()
        u_row.addWidget(QLabel("min"))
        u_row.addWidget(self._pp_u_min)
        u_row.addSpacing(6)
        u_row.addWidget(QLabel("max"))
        u_row.addWidget(self._pp_u_max)
        phase_layout.addRow("u-range (mV)", u_row)

        w_row = QHBoxLayout()
        w_row.addWidget(QLabel("min"))
        w_row.addWidget(self._pp_w_min)
        w_row.addSpacing(6)
        w_row.addWidget(QLabel("max"))
        w_row.addWidget(self._pp_w_max)
        phase_layout.addRow("w-range", w_row)

        self._show_vec_chk = QCheckBox("Show vector field")
        self._show_eq_chk = QCheckBox("Show equilibria")
        self._show_null_chk = QCheckBox("Show nullclines")
        self._show_bif_chk = QCheckBox("Show bifurcations")
        self._show_sep_chk = QCheckBox("Show separatrix")
        self._show_vec_chk.stateChanged.connect(self._on_phase_view_changed)
        self._show_eq_chk.stateChanged.connect(self._on_phase_view_changed)
        self._show_null_chk.stateChanged.connect(self._on_phase_view_changed)
        self._show_bif_chk.stateChanged.connect(self._on_phase_view_changed)
        self._show_sep_chk.stateChanged.connect(self._on_phase_view_changed)

        phase_layout.addRow("", self._show_vec_chk)
        phase_layout.addRow("", self._show_eq_chk)
        phase_layout.addRow("", self._show_null_chk)
        phase_layout.addRow("", self._show_bif_chk)
        phase_layout.addRow("", self._show_sep_chk)

        root.addWidget(phase_box)

        #Time series view controls
        ts_box = QGroupBox("Time series view")
        ts_layout = QFormLayout(ts_box)

        self._ts_t_min = QDoubleSpinBox()
        self._ts_t_max = QDoubleSpinBox()
        for sb in (self._ts_t_min, self._ts_t_max):
            sb.setRange(0.0, 1e9)
            sb.setDecimals(3)
            sb.setSingleStep(10.0)
            sb.valueChanged.connect(self._on_ts_view_changed)

        t_row = QHBoxLayout()
        t_row.addWidget(QLabel("min"))
        t_row.addWidget(self._ts_t_min)
        t_row.addSpacing(6)
        t_row.addWidget(QLabel("max"))
        t_row.addWidget(self._ts_t_max)
        ts_layout.addRow("t-window (ms)", t_row)

        self._show_u_chk = QCheckBox("Show u(t)")
        self._show_w_chk = QCheckBox("Show w(t)")
        self._show_u_chk.stateChanged.connect(self._on_ts_view_changed)
        self._show_w_chk.stateChanged.connect(self._on_ts_view_changed)
        ts_layout.addRow("", self._show_u_chk)
        ts_layout.addRow("", self._show_w_chk)

        root.addWidget(ts_box)

        #Parameters
        par_box = QGroupBox("Model parameters")
        par_outer = QVBoxLayout(par_box)

        par_scroll = QScrollArea()
        par_scroll.setWidgetResizable(True)
        par_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        par_inner = QWidget()
        self._par_form = QFormLayout(par_inner)
        self._par_form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        self._par_form.setFormAlignment(Qt.AlignmentFlag.AlignTop)

        self._build_parameter_form(self._par_form)

        par_scroll.setWidget(par_inner)
        par_outer.addWidget(par_scroll)

        root.addWidget(par_box, stretch=1)

        root.addStretch(1)
    
    def _build_parameter_form(self, form: QFormLayout) -> None:
        if not is_dataclass(self._par):
            form.addRow(QLabel("It is not a dataclass."), QLabel(""))
            return

        for f in fields(self._par):
            name = f.name
            val = getattr(self._par, name)

            w = self._make_widget_for_value(name, val)
            self._par_widgets[name] = w

            lab = QLabel(name)
            self._par_labels[name] = lab
            form.addRow(lab, w)
            
    #--------------
    #Widget creator
    #--------------

    def _make_widget_for_value(self, name: str, val: Any) -> QWidget:
        if isinstance(val, bool):
            chk = QCheckBox()
            chk.stateChanged.connect(lambda _=0, n=name: self._on_par_widget_changed(n))
            return chk
        
        if isinstance(val, int) and not isinstance(val, bool):
            sb = QSpinBox()
            mn, mx, step = self._range_for_int(name, val)
            sb.setRange(mn, mx)
            sb.setSingleStep(step)
            sb.valueChanged.connect(lambda _=0, n=name: self._on_par_widget_changed(n))
            return sb
        
        if isinstance(val, float):
            sb = QDoubleSpinBox()
            mn, mx, step, dec = self._range_for_float(name, val)
            sb.setRange(mn, mx)
            sb.setSingleStep(step)
            sb.setDecimals(dec)
            sb.valueChanged.connect(lambda _=0, n=name: self._on_par_widget_changed(n))
            return sb

        #We don't expect weird input, but the program doesn't crash and we display it as a string.
        lab = QLabel(str(val))
        lab.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        return lab
    
    def _range_for_int(self, name:str, val: int) -> Tuple[int, int, int]:
        return(0, max(10, 10 * abs(val) + 10), 1)
    
    def _range_for_float(self, name: str, val:float) -> Tuple[float, float, float, int]:
        #Voltages and reversal potentials (mV)
        if name.startswith("V2_") or name.startswith("V4_"):
            return (1e-6, 200.0, 0.1, 6)
        
        if name.startswith("E_") or name.startswith("V"):
            return(-200.0, 200.0, 1.0, 4)
        
        #Conductances (mS/cm^2)
        if name.startswith("g_"):
            return(0.0, 500.0, 0.5, 6)
        
        #Capacitance (muF/cm^2)
        if name == "C":
            return(0.01, 100.0, 0.1, 6)
        
        #Phi
        if "Phi" in name or "phi" in name:
            return (0.0, 10.0, 0.01, 8)
        
        #Generic spans
        span = max(1.0, 5.0 * abs(val))
        mn = val - span
        mx = val + span
        step = max(1e-6, span / 200)
        dec = 8 if (abs(val) < 1e-2) else 6
        return (mn, mx, step, dec)
    
    def _set_par_row_visible(self, name: str, visible: bool) -> None:
        lab = self._par_labels.get(name)
        w = self._par_widgets.get(name)

        if lab is not None:
            lab.setVisible(visible)
        if w is not None:
            w.setVisible(visible)
    
    
    #-----
    #Synch
    #-----
    def _sync_widgets_from_state(self) -> None:
        self._I_ext_spin.blockSignals(True)
        self._I_ext_spin.setValue(self._I_ext)
        self._I_ext_spin.blockSignals(False)

        self._y0_u_spin.blockSignals(True)
        self._y0_u_spin.setValue(self._y0_u)
        self._y0_u_spin.blockSignals(False)

        self._y0_w_spin.blockSignals(True)
        self._y0_w_spin.setValue(self._y0_w)
        self._y0_w_spin.blockSignals(False)

        self._t1_spin.setValue(float(self._sim.t1))
        self._dt_spin.setValue(float(self._sim.dt))
        self._method_com.setCurrentText(str(self._sim.method))
        self._rtol_spin.setValue(float(self._sim.rtol))
        self._atol_spin.setValue(float(self._sim.atol))
        if np.isinf(self._sim.max_step):
            self._max_step_spin.setValue(0.0)
        else:
            self._max_step_spin.setValue(float(self._sim.max_step))

        #We handle the outputs of spike_threshold
        if self._sim.spike_threshold is None:
            self._spike_thr_spin.setValue(0.0)
            self._spike_thr_spin.setEnabled(False)
        else:
            self._spike_thr_spin.setEnabled(True)
            self._spike_thr_spin.setValue(float(self._sim.spike_threshold))


        self._stop_on_spike_chk.setChecked(bool(self._sim.stop_on_spike))

        #Phase plane
        for sb in (self._pp_u_min, self._pp_u_max, self._pp_w_min, self._pp_w_max,
                   self._show_vec_chk, self._show_eq_chk, self._show_null_chk, self._show_bif_chk, self._show_sep_chk):
            sb.blockSignals(True)
        try:
            self._pp_u_min.setValue(float(self._phase_view.u_min))
            self._pp_u_max.setValue(float(self._phase_view.u_max))
            self._pp_w_min.setValue(float(self._phase_view.w_min))
            self._pp_w_max.setValue(float(self._phase_view.w_max))
            self._show_vec_chk.setChecked(bool(self._phase_view.show_vector_field))
            self._show_eq_chk.setChecked(bool(self._phase_view.show_equilibria))
            self._show_null_chk.setChecked(bool(self._phase_view.show_nullclines))
            self._show_bif_chk.setChecked(bool(self._phase_view.show_bifurcations))
            self._show_sep_chk.setChecked(bool(self._phase_view.show_separatrix))

        finally:
            for sb in (self._pp_u_min, self._pp_u_max, self._pp_w_min, self._pp_w_max,
                       self._show_vec_chk, self._show_eq_chk, self._show_null_chk, self._show_bif_chk, self._show_sep_chk):
                sb.blockSignals(False)

        #Timeseries
        ts_ws = (self._ts_t_min, self._ts_t_max, self._show_u_chk, self._show_w_chk)
        for w in ts_ws:
            w.blockSignals(True)
        try:
            self._ts_t_min.setValue(float(self._ts_view.t_min if self._ts_view.t_min is not None else 0.0))
            self._ts_t_max.setValue(float(self._ts_view.t_max if self._ts_view.t_max is not None else 1.0))
            self._show_u_chk.setChecked(bool(self._ts_view.show_u))
            self._show_w_chk.setChecked(bool(self._ts_view.show_w))
        finally:
            for w in ts_ws:
                w.blockSignals(False)

        #Parameters
        for name, w in self._par_widgets.items():
            val = getattr(self._par, name)
            self._set_widget_value(w, val)

    def _set_widget_value(self, w:QWidget, val:Any) -> None:
        w.blockSignals(True)
        try:
            if isinstance(w, QCheckBox):
                w.setChecked(bool(val))
            elif isinstance(w, QSpinBox):
                w.setValue(int(val))
            elif isinstance(w, QDoubleSpinBox):
                w.setValue(float(val))
            elif isinstance(w, QLabel):
                w.setText(str(val))
        finally:
            w.blockSignals(False)

    def _read_widget_value(self, w: QWidget) -> Any:
        if isinstance(w, QCheckBox):
            return bool(w.isChecked())
        if isinstance(w, QSpinBox):
            return int(w.value())
        if isinstance(w, QDoubleSpinBox):
            return float(w.value())
        if isinstance(w, QLabel):
            return w.text()
        return None
    
    #--------------
    #Event handlers
    #--------------

    def _schedule_emit(self) -> None:
        self._emit_timer.start()
    
    def _emit_state_now(self) -> None:
        self.stateChanged.emit(self.current_state())
    
    def _on_run_clicked(self) -> None:
        self._emit_timer.stop()
        self._emit_state_now()
        self.runRequested.emit(self.current_state())
    
    def _on_I_ext_changed(self, v: float) -> None:
        self._I_ext = float(v)
        self._schedule_emit()

    def _on_y0_changed(self, _v: float) -> None:
        self._y0_u = float(self._y0_u_spin.value())
        self._y0_w = float(self._y0_w_spin.value())
        self._schedule_emit()
    
    def _on_sim_changed(self, *_:Any) -> None:
        #if spike threshold is disabled, treat it as None
        spike_thr = float(self._spike_thr_spin.value()) if self._spike_thr_spin.isEnabled() else None
        
        max_step_val = float(self._max_step_spin.value())
        if max_step_val <= 0.0:
            max_step_val = float("inf") #0 or below mean infinity

        self._sim = replace(
            self._sim,
            t1=float(self._t1_spin.value()),
            dt=float(self._dt_spin.value()),
            method=str(self._method_com.currentText()),
            rtol=float(self._rtol_spin.value()),
            atol=float(self._atol_spin.value()),
            max_step=max_step_val,
            spike_threshold=spike_thr,
            stop_on_spike=bool(self._stop_on_spike_chk.isChecked()),
        )
        self._schedule_emit()

    def _on_phase_view_changed(self, *_:Any) -> None:       
        self._phase_view = replace(
            self._phase_view,
            u_min=float(self._pp_u_min.value()),
            u_max=float(self._pp_u_max.value()),
            w_min=float(self._pp_w_min.value()),
            w_max=float(self._pp_w_max.value()),
            show_vector_field=bool(self._show_vec_chk.isChecked()),
            show_equilibria=bool(self._show_eq_chk.isChecked()),
            show_nullclines=bool(self._show_null_chk.isChecked()),
            show_bifurcations=bool(self._show_bif_chk.isChecked()),
            show_separatrix=bool(self._show_sep_chk.isChecked())
        )
        self._schedule_emit()
    
    def _on_ts_view_changed(self, *_: Any) -> None:
        #if tmin, tmax == 0 use auto
        tmin = float(self._ts_t_min.value())
        tmax = float(self._ts_t_max.value())
        t_min = None if tmin == 0.0 else tmin
        t_max = None if tmax == 0.0 else tmax

        self._ts_view = replace(
            self._ts_view,
            t_min=t_min,
            t_max=t_max,
            show_u=bool(self._show_u_chk.isChecked()),
            show_w=bool(self._show_w_chk.isChecked()),
        )
        self._schedule_emit()

    def _on_par_widget_changed(self, name: str) -> None:
        """We update MLParameters immutably"""
        w = self._par_widgets[name]
        new_val = self._read_widget_value(w)
        self._par = replace(self._par, **{name: new_val})

        self._schedule_emit()