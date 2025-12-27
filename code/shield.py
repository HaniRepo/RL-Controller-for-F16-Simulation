# shield.py
import numpy as np

class SimpleThrottleShield:
    """
    If |Vt-sp|/sp > 10%, push throttle toward setpoint; always slew-limit.
    """
    def __init__(self, slew=0.05, u_min=0.0, u_max=1.0):
        self.slew, self.u_min, self.u_max = float(slew), float(u_min), float(u_max)
        self.u_prev = 0.5

    def reset(self, u0=0.5):
        self.u_prev = float(np.clip(u0, self.u_min, self.u_max))

    def filter(self, vt, sp, u_rl):
        err_rel = (vt - sp) / max(1e-6, sp)
        u_rl = float(np.clip(u_rl, self.u_min, self.u_max))
        u_tgt = 0.75 if abs(err_rel) > 0.10 and err_rel < 0 else \
                0.25 if abs(err_rel) > 0.10 and err_rel > 0 else u_rl
        u = np.clip(u_tgt, self.u_prev - self.slew, self.u_prev + self.slew)
        u = float(np.clip(u, self.u_min, self.u_max))
        self.u_prev = u
        return u
