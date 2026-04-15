import os
import sys
import csv
import json

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from f16_engine_env import F16EngineEnv
from stress_wrappers import SetpointJumpWrapper
from stl_monitor import settling_spec_last_window
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
    os.path.join(os.path.dirname(__file__), "selector_ablation_out")
)

SEED = 42
N_EPISODES = 5


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


def build_candidates(u_rl, vt, sp, mode="directional", num_candidates=5, spread=0.06):
    u_rl = float(u_rl)

    if num_candidates < 1:
        return [u_rl]

    if mode == "symmetric":
        if num_candidates == 1:
            offsets = [0.0]
        else:
            offsets = np.linspace(-spread, spread, num_candidates)
    elif mode == "directional":
        if vt < sp:
            offsets = np.linspace(0.0, spread, num_candidates)
        else:
            offsets = np.linspace(-spread, 0.0, num_candidates)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    cands = [float(np.clip(u_rl + off, 0.0, 1.0)) for off in offsets]

    # unique while preserving order
    uniq = []
    seen = set()
    for u in cands:
        key = round(u, 6)
        if key not in seen:
            seen.add(key)
            uniq.append(u)
    return uniq


def choose_action(pred, q, vt, pw, sp, u_rl, K=3, tol=0.04,
                  mode="directional", num_candidates=5, spread=0.06):
    candidates = build_candidates(
        u_rl=u_rl,
        vt=vt,
        sp=sp,
        mode=mode,
        num_candidates=num_candidates,
        spread=spread,
    )

    ppo_seq = roll_pred(pred, vt, pw, u_rl, K=K)
    ppo_rho = worst_case_band_violation(ppo_seq, q, sp, tol=tol)

    best_u = float(u_rl)
    best_rho = float(ppo_rho)

    for u in candidates:
        seq = roll_pred(pred, vt, pw, u, K=K)
        rho = worst_case_band_violation(seq, q, sp, tol=tol)
        if rho > best_rho:
            best_rho = float(rho)
            best_u = float(u)

    return {
        "ppo_action": float(u_rl),
        "selected_action": float(best_u),
        "ppo_pred_rho": float(ppo_rho),
        "selected_pred_rho": float(best_rho),
        "predicted_gain": float(best_rho - ppo_rho),
        "changed": float(abs(best_u - float(u_rl)) > 1e-6),
        "n_candidates": len(candidates),
    }


def evaluate_episode(model, pred, q, seed, K, tol, window_s, mode, num_candidates, spread,
                     sp0=500.0, sp_new=530.0, t_jump_s=20.0):
    env = make_env(sp0=sp0, sp_new=sp_new, t_jump_s=t_jump_s)
    base = _base_env(env)

    obs, info = env.reset(seed=seed)

    vt_trace = []
    changed_steps = 0
    total_steps = 0
    gain_sum = 0.0

    detailed_rows = []

    done = False
    while not done:
        a_rl, _ = model.predict(obs, deterministic=True)
        u_rl = float(np.asarray(a_rl).reshape(-1)[0])

        vt = float(base.sim.Vt)
        pw = float(base.sim.pow)
        sp = float(base.sp)

        decision = choose_action(
            pred=pred,
            q=q,
            vt=vt,
            pw=pw,
            sp=sp,
            u_rl=u_rl,
            K=K,
            tol=tol,
            mode=mode,
            num_candidates=num_candidates,
            spread=spread,
        )

        u = decision["selected_action"]

        obs, r, done, trunc, info = env.step(np.array([u], dtype=np.float32))
        vt_trace.append(info["Vt"])

        changed_steps += int(decision["changed"])
        total_steps += 1
        gain_sum += decision["predicted_gain"]

        # keep a few rows around the jump for qualitative inspection
        t = info.get("t", 0.0)
        if 18.0 <= t <= 24.0:
            detailed_rows.append({
                "t": float(t),
                "Vt": float(vt),
                "sp": float(sp),
                "ppo_action": decision["ppo_action"],
                "selected_action": decision["selected_action"],
                "ppo_pred_rho": decision["ppo_pred_rho"],
                "selected_pred_rho": decision["selected_pred_rho"],
                "predicted_gain": decision["predicted_gain"],
            })

        if trunc:
            break

    sat, rho = settling_spec_last_window(
        np.asarray(vt_trace),
        sp=base.sp,
        dt=base.dt,
        window_s=window_s,
        tol=tol,
    )

    return {
        "sat": float(sat),
        "rho": float(rho),
        "change_rate": float(changed_steps / max(1, total_steps)),
        "avg_predicted_gain": float(gain_sum / max(1, total_steps)),
        "detail_rows": detailed_rows,
    }


def run_ablation():
    os.makedirs(OUTDIR, exist_ok=True)

    env0 = make_env()
    model = PPO.load(
        MODEL,
        env=env0,
        custom_objects={
            "learning_rate": 3e-4,
            "clip_range": 0.2,
        },
    )

    pred, q = build_predictor(env0, model, calib_eps=3, delta=0.30)

    configs = [
        {"name": "dir_c3_k1", "mode": "directional", "num_candidates": 3, "K": 1, "spread": 0.06},
        {"name": "dir_c5_k3", "mode": "directional", "num_candidates": 5, "K": 3, "spread": 0.06},
        {"name": "dir_c7_k3", "mode": "directional", "num_candidates": 7, "K": 3, "spread": 0.08},
        {"name": "sym_c5_k3", "mode": "symmetric",  "num_candidates": 5, "K": 3, "spread": 0.06},
    ]

    summary_rows = []
    mechanism_rows = []

    print("\n=== Selector Ablation ===")
    print(f"{'Config':12s} | {'Sat%':>6s} | {'Mean rho':>8s} | {'Change%':>8s} | {'Avg gain':>9s}")
    print("-" * 60)

    for cfg in configs:
        sats = []
        rhos = []
        change_rates = []
        gains = []

        first_detail = None

        for i in range(N_EPISODES):
            result = evaluate_episode(
                model=model,
                pred=pred,
                q=q,
                seed=SEED + i,
                K=cfg["K"],
                tol=0.04,
                window_s=8.0,
                mode=cfg["mode"],
                num_candidates=cfg["num_candidates"],
                spread=cfg["spread"],
                sp0=500.0,
                sp_new=530.0,
                t_jump_s=20.0,
            )

            sats.append(result["sat"])
            rhos.append(result["rho"])
            change_rates.append(result["change_rate"])
            gains.append(result["avg_predicted_gain"])

            if first_detail is None and result["detail_rows"]:
                first_detail = result["detail_rows"]

        row = {
            "config": cfg["name"],
            "mode": cfg["mode"],
            "num_candidates": cfg["num_candidates"],
            "K": cfg["K"],
            "spread": cfg["spread"],
            "sat": float(np.mean(sats)),
            "rho": float(np.mean(rhos)),
            "change_rate": float(np.mean(change_rates)),
            "avg_predicted_gain": float(np.mean(gains)),
        }
        summary_rows.append(row)

        if first_detail is not None:
            for r in first_detail[:8]:
                mechanism_rows.append({
                    "config": cfg["name"],
                    **r
                })

        print(
            f"{cfg['name']:12s} | {100*row['sat']:6.1f} | {row['rho']:8.4f} | "
            f"{100*row['change_rate']:8.1f} | {row['avg_predicted_gain']:9.5f}"
        )

    with open(os.path.join(OUTDIR, "selector_ablation_summary.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "config", "mode", "num_candidates", "K", "spread",
                "sat", "rho", "change_rate", "avg_predicted_gain"
            ],
        )
        w.writeheader()
        w.writerows(summary_rows)

    with open(os.path.join(OUTDIR, "selector_mechanism_examples.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "config", "t", "Vt", "sp", "ppo_action", "selected_action",
                "ppo_pred_rho", "selected_pred_rho", "predicted_gain"
            ],
        )
        w.writeheader()
        w.writerows(mechanism_rows)

    with open(os.path.join(OUTDIR, "selector_ablation_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2, ensure_ascii=False)

    print("\nSaved to:", OUTDIR)
    print(" - selector_ablation_summary.csv")
    print(" - selector_mechanism_examples.csv")


if __name__ == "__main__":
    run_ablation()