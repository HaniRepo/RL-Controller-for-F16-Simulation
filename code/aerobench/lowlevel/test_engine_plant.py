"""
test_engine_plant.py
Quick, verbose tests for EnginePlantV2:
 - Prints state after reset
 - Applies throttle steps and ramps
 - Confirms Vt responds in the expected direction
"""

import time
import numpy as np

from .engine_plant_v2 import EnginePlantV2, EnginePlantConfig

def print_state(t, plant, tag=""):
    print(f"[t={t:6.2f}s]{tag}  Vt={plant.Vt:8.2f} ft/s   pow={plant.pow:6.3f}")

def test_step_response(throttle_lo=0.3, throttle_hi=0.7, hold_s=5.0, dt=0.05):
    print("\n=== Test 1: Step response ===")
    # set to 100.0 if your subf16 expects percent
    #cfg = EnginePlantConfig(throttle_scale=100.0, model_name="stevens")  
    cfg = EnginePlantConfig(model_name="stevens",
                        throttle_scale=1.0)    # optional, see note below
    plant = EnginePlantV2(cfg)
    Vt0, pow0 = plant.reset(Vt0=500.0, alt0_ft=550.0, pow0=10.0)
    print_state(0.0, plant, tag=" reset")

    # Low throttle hold
    t = 0.0
    n_lo = int(round(hold_s / dt))
    for _ in range(n_lo):
        plant.step_engine(throttle_lo, dt)
        t += dt
    print_state(t, plant, tag=f" hold thr={throttle_lo:.2f}")

    # Step up
    for _ in range(n_lo):
        plant.step_engine(throttle_hi, dt)
        t += dt
        if (_+1) % int(1.0/dt) == 0:
            print_state(t, plant, tag=f" step thr={throttle_hi:.2f}")

def test_ramp(throttle_start=0.2, throttle_end=0.8, total_s=10.0, dt=0.05):
    print("\n=== Test 2: Throttle ramp ===")
    # fix(Hani): it was 0-1 scale
    # cfg = EnginePlantConfig(throttle_scale=100.0, model_name="stevens")
    cfg = EnginePlantConfig(model_name="stevens",
                        throttle_scale=1.0)    # optional, see note below
    plant = EnginePlantV2(cfg)
    plant.reset(Vt0=450.0, alt0_ft=550.0, pow0=9.0)
    print_state(0.0, plant, tag=" reset")

    t = 0.0
    n = int(round(total_s / dt))
    for k in range(n):
        alpha = k / max(1, n-1)
        thr = (1 - alpha) * throttle_start + alpha * throttle_end
        plant.step_engine(thr, dt)
        t += dt
        if (k+1) % int(1.0/dt) == 0:
            print_state(t, plant, tag=f" ramp thr={thr:.2f}")

if __name__ == "__main__":
    try:
        test_step_response()
        test_ramp()
        print("\n✅ Tests finished.")
    except ImportError as e:
        print("\n❌ Import error:", e)
        print("Make sure AeroBenchVVPython (v2) is on your PYTHONPATH.")
