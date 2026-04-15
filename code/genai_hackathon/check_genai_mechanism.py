import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from f16_engine_env import F16EngineEnv
from conformal_shield import OneStepVTLinear, split_conformal_q, collect_calibration, worst_case_band_violation
from stress_wrappers import SetpointJumpWrapper


MODEL = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "shield_pack", "ppo_f16_engine_baseline.zip")
)
SEED = 42


def _base_env(e):
    while hasattr(e, "env"):
        e = e.env
    return e


def make_env():
    e = Monitor(F16EngineEnv(sp=500.0, dt=0.1, ep_len_s=60.0, seed=SEED))
    e = SetpointJumpWrapper(e, t_jump_s=20.0, sp_new = 530)
    return e


def build_predictor(env, model, calib_eps=3, delta=0.30):
    base = _base_env(env)
    Vt, pw, thr = collect_calibration(base, policy=model, episodes=calib_eps, random_throttle=False, seed=123)

    pred = OneStepVTLinear()
    pred.fit(Vt, pw, thr)

    pred_next = [pred.predict_next(v0, p0, u0) for v0, p0, u0 in zip(Vt[:-1], pw[:-1], thr[:-1])]
    q = split_conformal_q(np.asarray(pred_next) - Vt[1:], delta=delta)

    return pred, q


def roll_pred(pred, vt0, pw0, u_const, K=3):
    vt, pw = float(vt0), float(pw0)
    seq = []
    for _ in range(K):
        vt = pred.predict_next(vt, pw, u_const)
        pw += 0.3 * (u_const - pw)
        seq.append(vt)
    return np.asarray(seq)


def main():
    env = make_env()
    model = PPO.load(
        MODEL,
        env=env,
        custom_objects={
            "learning_rate": 3e-4,
            "clip_range": 0.2,
        },
    )

    pred, q = build_predictor(env, model)

    obs, info = env.reset(seed=SEED)
    base = _base_env(env)

    print("\n=== GenAI mechanism sanity check ===")
    print("We inspect candidate actions around the setpoint jump.\n")

    done = False
    step_idx = 0

    while not done:
        a_rl, _ = model.predict(obs, deterministic=True)
        u_rl = float(np.asarray(a_rl).reshape(-1)[0])

        vt = float(base.sim.Vt)
        pw = float(base.sim.pow)
        sp = float(base.sp)

        # inspect around / after jump
        if step_idx in [180, 195, 205, 220, 240]:
            if vt < sp:
                candidates = [u_rl, min(1.0, u_rl + 0.03), min(1.0, u_rl + 0.06), min(1.0, u_rl + 0.10)]
            else:
                candidates = [u_rl, max(0.0, u_rl - 0.03), max(0.0, u_rl - 0.06), max(0.0, u_rl - 0.10)]

            print(f"\nStep {step_idx}, t={step_idx*0.1:.1f}s, Vt={vt:.2f}, sp={sp:.2f}, PPO={u_rl:.3f}")

            best_u = None
            best_rho = -1e9

            for u in candidates:
                seq = roll_pred(pred, vt, pw, u, K=3)
                rho = worst_case_band_violation(seq, q, sp, tol=0.02)
                print(f"  candidate u={u:.3f} -> predicted robustness {rho:.4f}")

                if rho > best_rho:
                    best_rho = rho
                    best_u = u

            print(f"  selected action: {best_u:.3f} (best predicted rho={best_rho:.4f})")

        obs, r, done, trunc, info = env.step(np.array([u_rl], dtype=np.float32))
        if trunc:
            break
        step_idx += 1


if __name__ == "__main__":
    main()