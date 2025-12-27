"""
engine_plant_v2.py
Minimal Engine-Control plant wrapper around AeroBenchVVPython (v2).

This isolates the "Engine Control" benchmark idea:
  - State focus: Vt (true airspeed), pow (engine power lag state)
  - Input: throttle (scalar)
Everything else is kept at steady straight-and-level flight and not exposed.

Requires AeroBenchVVPython v2 on PYTHONPATH:
    from aerobench.lowlevel.subf16_model import subf16_model
    from aerobench.run_f16_sim import make_f16_model   (fallback if available)

If those imports fail, run:  pip install numpy scipy
and clone AeroBenchVVPython (v2).
"""

from dataclasses import dataclass
from typing import Tuple, Optional

import math
import numpy as np

# --- Try to import AeroBench v2 low-level model ---
try:
    from aerobench.lowlevel.subf16_model import subf16_model
except Exception as e:
    subf16_model = None
    _import_error = e
else:
    _import_error = None

# Optional factory for model parameters (some v2 trees expose it)
try:
    # Some repos define make_f16_model() here:
    from aerobench.run_f16_sim import make_f16_model  # type: ignore
except Exception:
    make_f16_model = None


@dataclass
class EnginePlantConfig:
    # Trim / fixed values for straight & level
    alt_ft: float = 550.0         # altitude in feet (v2 uses feet)
    gamma_rad: float = 0.0        # flight-path angle
    theta_rad: float = 0.0        # pitch angle
    phi_rad: float = 0.0          # roll
    beta_rad: float = 0.0         # sideslip
    psi_rad: float = 0.0          # heading
    p_rad_s: float = 0.0          # body roll rate
    q_rad_s: float = 0.0          # body pitch rate
    r_rad_s: float = 0.0          # body yaw rate
    pn_ft: float = 0.0            # north position
    pe_ft: float = 0.0            # east position

    # Integrator settings
    dt_internal: float = 0.02     # substep for RK4 (sec)

    # Control scaling: throttle fraction (0..1) -> u[0] value passed to subf16_model
    # Some implementations use throttle in [0,1]; others in "percent"
    # Set to 1.0 if your model expects 0..1; set to 100.0 if it expects 0..100.
    throttle_scale: float = 1.0

    # Saturation limits
    throttle_min: float = 0.0
    throttle_max: float = 1.0
    #added by me to fix argument issue
    model_name: str = "stevens"   # or "morelli"
    from dataclasses import dataclass

@dataclass
class EnginePlantConfig:
    alt_ft: float = 550.0
    gamma_rad: float = 0.0
    theta_rad: float = 0.0
    phi_rad: float = 0.0
    beta_rad: float = 0.0
    psi_rad: float = 0.0
    p_rad_s: float = 0.0
    q_rad_s: float = 0.0
    r_rad_s: float = 0.0
    pn_ft: float = 0.0
    pe_ft: float = 0.0

    dt_internal: float = 0.02

    throttle_scale: float = 1.0   # keep as fraction [0,1] for THIS repo
    throttle_min: float = 0.0
    throttle_max: float = 1.0

    model_name: str = "stevens"

    # NEW: freeze all states except VT and pow to mimic 2-state engine case
    engine_only: bool = True

    # Engine + aero model selection
    # If your repo exposes a "model" object, pass it in EnginePlantV2(..., f16_model=that)
    # otherwise the code will call subf16_model(x, u, None) which works in many forks.
    # You can also pass f16_model later via set_model().
    pass


class EnginePlantV2:
    """
    A very small wrapper that:
      - stores a full 13+ state vector expected by subf16_model
      - only exposes throttle (u[0]); sets all other controls (elevator/aileron/rudder) to 0
      - integrates forward with RK4 using subf16_model

    State convention (typical for AeroBench/Stevens):
      x[0]  = VT      [ft/s]
      x[1]  = alpha   [rad]
      x[2]  = beta    [rad]
      x[3]  = phi     [rad]
      x[4]  = theta   [rad]
      x[5]  = psi     [rad]
      x[6]  = p       [rad/s]
      x[7]  = q       [rad/s]
      x[8]  = r       [rad/s]
      x[9]  = pn      [ft]
      x[10] = pe      [ft]
      x[11] = h       [ft]   (altitude)
      x[12] = pow     [unitless engine lag]

    Control vector u (length 4):
      u[0] = throttle  (fraction 0..1 or percent 0..100 depending on repo)
      u[1] = elevator  [deg]
      u[2] = aileron   [deg]
      u[3] = rudder    [deg]
    """

    def __init__(self,
                 cfg: EnginePlantConfig = EnginePlantConfig(),
                 f16_model: Optional[object] = None):
        if subf16_model is None:
            raise ImportError(
                "Could not import aerobench.lowlevel.subf16_model. "
                f"Original error: {_import_error}\n"
                "Make sure AeroBenchVVPython (v2) is installed or on PYTHONPATH."
            )

        self.cfg = cfg
        self.f16_model = f16_model  # optional model parameters object used by subf16_model
        self.state = np.zeros(13, dtype=float)
        self.reset()

    # ----------------------------- Utilities -----------------------------

    def set_model(self, f16_model_obj):
        """Attach a model/parameters object if your subf16_model requires it."""
        self.f16_model = f16_model_obj

    def _u_vec(self, throttle: float) -> np.ndarray:
        # throttle ∈ [0,1]
        t = float(np.clip(throttle, self.cfg.throttle_min, self.cfg.throttle_max))
        elevator_deg = 0.0
        aileron_deg  = 0.0
        rudder_deg   = 0.0
        # CORRECT ORDER FOR THIS REPO:
        return np.array([t, elevator_deg, aileron_deg, rudder_deg], dtype=float)

    def _rk4(self, x: np.ndarray, u: np.ndarray, h: float) -> np.ndarray:
        def f(x_local):
            xd, *_ = subf16_model(x_local, u, self.cfg.model_name)
            return np.asarray(xd, dtype=float)

        k1 = f(x)
        k2 = f(x + 0.5 * h * k1)
        k3 = f(x + 0.5 * h * k2)
        k4 = f(x + h * k3)

        if self.cfg.engine_only:
            # integrate only VT (0) and pow (12); hold other states fixed
            x_next = x.copy()
            x_next[0]  = x[0]  + (h/6.0) * (k1[0]  + 2*k2[0]  + 2*k3[0]  + k4[0])
            x_next[12] = x[12] + (h/6.0) * (k1[12] + 2*k2[12] + 2*k3[12] + k4[12])
            return x_next
        else:
            return x + (h/6.0) * (k1 + 2*k2 + 2*k3 + k4)

    # ----------------------------- Public API -----------------------------

    def reset(self,
              Vt0: float = 500.0,
              alt0_ft: Optional[float] = None,
              pow0: float = 10.0) -> Tuple[float, float]:
        """
        Reset the plant to (approximately) straight & level with given speed/alt/pow.
        Returns (Vt, pow).
        """
        alt_ft = self.cfg.alt_ft if alt0_ft is None else float(alt0_ft)

        x = np.zeros(13, dtype=float)
        x[0]  = float(Vt0)                    # VT [ft/s]
        x[1]  = 0.0                           # alpha
        x[2]  = self.cfg.beta_rad
        x[3]  = self.cfg.phi_rad
        x[4]  = self.cfg.theta_rad
        x[5]  = self.cfg.psi_rad
        x[6]  = self.cfg.p_rad_s
        x[7]  = self.cfg.q_rad_s
        x[8]  = self.cfg.r_rad_s
        x[9]  = self.cfg.pn_ft
        x[10] = self.cfg.pe_ft
        x[11] = float(alt_ft)                 # altitude
        x[12] = float(pow0)                   # engine lag state

        self.state = x
        return self.state[0], self.state[12]

    def step_engine(self, throttle: float, dt: float) -> Tuple[float, float]:
        """
        Advance the plant by dt seconds with a (possibly fractional) throttle command.

        Returns (Vt, pow) after the step for convenience.
        """
        # sub-cycle integration for stability
        n_sub = max(1, int(round(dt / self.cfg.dt_internal)))
        h = dt / n_sub
        u = self._u_vec(throttle)

        for _ in range(n_sub):
            self.state = self._rk4(self.state, u, h)

        # Return the two states of interest
        return float(self.state[0]), float(self.state[12])

    # Convenience accessors (for your Gym env)
    @property
    def Vt(self) -> float:
        return float(self.state[0])

    @property
    def pow(self) -> float:
        return float(self.state[12])
