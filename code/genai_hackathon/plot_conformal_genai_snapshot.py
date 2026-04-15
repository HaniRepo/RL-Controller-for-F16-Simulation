import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from f16_engine_env import F16EngineEnv
from stress_wrappers import SetpointJumpWrapper
from conformal_shield import (
    OneStepVTLinear,
    split_conformal_q,
    collect_calibration,
    worst_case_band_violation,
)

MODEL = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "shield_pack", "ppo_f16_engine_baseline.zip")
)
OUTDIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "figures")
)

SEED = 42


def _base_env(e):
    while hasattr(e, "env"):
        e = e.env
    return e


def make_env():
    e = Monitor(F16EngineEnv(sp=500.0, dt=0.1, ep_len_s=60.0, seed=SEED))
    e = SetpointJumpWrapper(e, t_jump_s=20.0, sp_new=530.0)
    return e


def build_predictor(env, model, calib_eps=3, delta=0.30):
    base = _base_env(env)

    Vt, pw, thr = collect_calibration(
        base, policy=model, episodes=calib_eps, random_throttle=False, seed=123
    )

    pred = OneStepVTLinear()
    pred.fit(Vt, pw, thr)

    pred_next = [
        pred.predict_next(v0, p0, u0)
        for v0, p0, u0 in zip(Vt[:-1], pw[:-1], thr[:-1])
    ]
    q = split_conformal_q(np.asarray(pred_next) - Vt[1:], delta=delta)

    return pred, q


def roll_pred(pred, vt0, pw0, u_const, K=5):
    vt, pw = float(vt0), float(pw0)
    seq = []
    for _ in range(K):
        vt = pred.predict_next(vt, pw, u_const)
        pw += 0.3 * (u_const - pw)
        seq.append(vt)
    return np.asarray(seq, dtype=float)


def build_candidates(u_rl, vt, sp):
    # directional candidates
    if vt < sp:
        offsets = [0.0, 0.03, 0.06, 0.10]
    else:
        offsets = [-0.10, -0.06, -0.03, 0.0]

    cands = [float(np.clip(u_rl + off, 0.0, 1.0)) for off in offsets]

    uniq = []
    seen = set()
    for u in cands:
        key = round(u, 6)
        if key not in seen:
            seen.add(key)
            uniq.append(u)
    return uniq


def pick_snapshot_step():
    # choose a time slightly before / around jump
    return 195  # ~19.5s


def main():
    os.makedirs(OUTDIR, exist_ok=True)

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

    target_step = pick_snapshot_step()
    step_idx = 0
    done = False

    while not done:
        a_rl, _ = model.predict(obs, deterministic=True)
        u_rl = float(np.asarray(a_rl).reshape(-1)[0])

        vt = float(base.sim.Vt)
        pw = float(base.sim.pow)
        sp = float(base.sp)

        if step_idx == target_step:
            candidates = build_candidates(u_rl, vt, sp)

            best_u = u_rl
            best_rho = -1e9

            cand_info = []

            for u in candidates:
                seq = roll_pred(pred, vt, pw, u, K=5)
                rho = worst_case_band_violation(seq, q, sp, tol=0.04)
                cand_info.append((u, seq, rho))

                if rho > best_rho:
                    best_rho = rho
                    best_u = u

            seq_ppo = roll_pred(pred, vt, pw, u_rl, K=5)
            seq_gen = roll_pred(pred, vt, pw, best_u, K=5)

            horizon = np.arange(1, len(seq_ppo) + 1) * base.dt

            plt.figure(figsize=(7, 4.5))

            # STL tolerance band
            tol = 0.04
            lower = sp * (1 - tol)
            upper = sp * (1 + tol)
            plt.fill_between(horizon, lower, upper, alpha=0.2, label="STL tolerance band")

            # PPO predicted band
            plt.plot(horizon, seq_ppo, linewidth=2, label=f"PPO action={u_rl:.3f}")
            plt.fill_between(horizon, seq_ppo - q, seq_ppo + q, alpha=0.2)

            # GenAI predicted band
            plt.plot(horizon, seq_gen, linewidth=2, linestyle="--", label=f"GenAI selected={best_u:.3f}")
            plt.fill_between(horizon, seq_gen - q, seq_gen + q, alpha=0.2)

            # setpoint
            plt.axhline(sp, linestyle=":", linewidth=2, label=f"Setpoint={sp:.1f}")

            plt.xlabel("Prediction horizon (s)")
            plt.ylabel("Predicted airspeed $V_t$")
            plt.title(f"Runtime safety snapshot at t={step_idx*base.dt:.1f}s")
            plt.legend()
            plt.tight_layout()

            out_path = os.path.join(OUTDIR, "conformal_genai_snapshot.png")
            plt.savefig(out_path, dpi=220)
            plt.close()

            # also save candidate robustness values
            txt_path = os.path.join(OUTDIR, "candidate_robustness_snapshot.txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(f"Snapshot at step {step_idx}, t={step_idx*base.dt:.1f}s\n")
                f.write(f"Vt={vt:.3f}, sp={sp:.3f}, PPO action={u_rl:.3f}\n\n")
                for u, seq, rho in cand_info:
                    f.write(f"candidate action={u:.3f}, predicted robustness={rho:.6f}\n")
                f.write(f"\nSelected action={best_u:.3f}, best robustness={best_rho:.6f}\n")

            print("Saved figure:", out_path)
            print("Saved details:", txt_path)
            return

        obs, r, done, trunc, info = env.step(np.array([u_rl], dtype=np.float32))
        if trunc:
            break
        step_idx += 1


if __name__ == "__main__":
    main()