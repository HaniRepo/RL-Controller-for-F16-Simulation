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
from conformal_shield import OneStepVTLinear, split_conformal_q, collect_calibration, worst_case_band_violation


MODEL = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "shield_pack", "ppo_f16_engine_baseline.zip")
)
OUTDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "figures"))
SEED = 42


def _base_env(e):
    while hasattr(e, "env"):
        e = e.env
    return e


def make_env(sp0=500.0, sp_new=530.0, t_jump_s=20.0):
    e = Monitor(F16EngineEnv(sp=sp0, dt=0.1, ep_len_s=60.0, seed=SEED))
    e = SetpointJumpWrapper(e, t_jump_s=t_jump_s, sp_new=sp_new)
    return e


def build_predictor(env, model, calib_eps=3, delta=0.30):
    base = _base_env(env)

    Vt, pw, thr = collect_calibration(
        base,
        policy=model,
        episodes=calib_eps,
        random_throttle=False,
        seed=123,
    )

    pred = OneStepVTLinear()
    pred.fit(Vt, pw, thr)

    pred_next = [
        pred.predict_next(v0, p0, u0)
        for v0, p0, u0 in zip(Vt[:-1], pw[:-1], thr[:-1])
    ]
    q = split_conformal_q(np.asarray(pred_next) - Vt[1:], delta=delta)
    return pred, q


def roll_pred(pred, vt0, pw0, u_const, K=3):
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


def choose_genai_action(pred, q, vt, pw, sp, u_rl, K=3, tol=0.04):
    candidates = build_candidates(u_rl, vt, sp)

    ppo_seq = roll_pred(pred, vt, pw, u_rl, K=K)
    best_u = float(u_rl)
    best_rho = worst_case_band_violation(ppo_seq, q, sp, tol=tol)

    for u in candidates:
        seq = roll_pred(pred, vt, pw, u, K=K)
        rho = worst_case_band_violation(seq, q, sp, tol=tol)
        if rho > best_rho:
            best_rho = rho
            best_u = float(u)

    changed = abs(best_u - float(u_rl)) > 1e-9
    return best_u, best_rho, changed


def main():
    os.makedirs(OUTDIR, exist_ok=True)

    env = make_env(sp0=500.0, sp_new=530.0, t_jump_s=20.0)
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

    dt = base.dt
    tol = 0.04

    times = []
    vt_trace = []
    sp_trace = []
    ppo_trace = []
    genai_trace = []
    changed_times = []
    changed_vt = []

    done = False
    step_idx = 0

    while not done:
        a_rl, _ = model.predict(obs, deterministic=True)
        u_rl = float(np.asarray(a_rl).reshape(-1)[0])

        vt_now = float(base.sim.Vt)
        pw_now = float(base.sim.pow)
        sp_now = float(base.sp)

        u_gen, _, changed = choose_genai_action(
            pred, q, vt_now, pw_now, sp_now, u_rl, K=3, tol=tol
        )

        obs, r, done, trunc, info = env.step(np.array([u_gen], dtype=np.float32))

        t = float(info.get("t", step_idx * dt))
        vt_after = float(info["Vt"])
        sp_after = float(base.sp)

        times.append(t)
        vt_trace.append(vt_after)
        sp_trace.append(sp_after)
        ppo_trace.append(u_rl)
        genai_trace.append(u_gen)

        if changed:
            changed_times.append(t)
            changed_vt.append(vt_after)

        if trunc:
            break
        step_idx += 1

    times = np.asarray(times)
    vt_trace = np.asarray(vt_trace)
    sp_trace = np.asarray(sp_trace)

    lower = sp_trace * (1.0 - tol)
    upper = sp_trace * (1.0 + tol)

    plt.figure(figsize=(8.5, 4.8))
    plt.plot(times, vt_trace, linewidth=2, label="Actual airspeed $V_t$")
    plt.plot(times, sp_trace, linestyle="--", linewidth=2, label="Setpoint")
    plt.fill_between(times, lower, upper, alpha=0.20, label="STL tolerance band")

    if len(changed_times) > 0:
        plt.scatter(
            changed_times,
            changed_vt,
            s=36,
            marker="o",
            zorder=5,
            label="GenAI action changed"
        )

    plt.xlabel("Time (s)")
    plt.ylabel("Airspeed")
    plt.title("Full rollout with GenAI decision changes")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()

    out_png = os.path.join(OUTDIR, "full_rollout_genai.png")
    plt.savefig(out_png, dpi=220)
    plt.close()

    out_txt = os.path.join(OUTDIR, "full_rollout_genai_summary.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(f"Total steps: {len(times)}\n")
        f.write(f"Number of changed actions: {len(changed_times)}\n")
        if len(times) > 0:
            f.write(f"Change rate: {len(changed_times)/len(times):.4f}\n")
        f.write(f"Tolerance used: {tol}\n")

    print("Saved figure:", out_png)
    print("Saved summary:", out_txt)


if __name__ == "__main__":
    main()