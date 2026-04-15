# genai_hackathon/run_proof_suite.py

import os
import sys
import csv
import json

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from f16_engine_env import F16EngineEnv
from stl_monitor import settling_spec_last_window
from stress_wrappers import (
    ActionRateLimiter,
    NoisyDelayedWrapper,
    SetpointJumpWrapper,
    ThrottleCapWrapper,
)
from conformal_shield import (
    OneStepVTLinear,
    split_conformal_q,
    ConformalSTLShield,
    collect_calibration,
)
from genai_hackathon.genai_shield import GenerativeConformalShield


OUT = "genai_hackathon/proof_suite_out"
MODEL = "shield_pack/ppo_f16_engine_baseline.zip"
SEED = 42
N = 5


def _base_env(e):
    while hasattr(e, "env"):
        e = e.env
    return e


def _dt(env):
    return _base_env(env).dt


def make_nominal(sp=500.0):
    return Monitor(F16EngineEnv(sp=sp, dt=0.1, ep_len_s=60.0, seed=SEED))

def make_decision_stress():
    e = make_nominal()

    # mild observation noise, no effective delay
    e = NoisyDelayedWrapper(e, obs_sigma=3.0, act_delay_steps=0)

    # switch model to create moderate dynamics mismatch
    b = _base_env(e)
    if hasattr(b, "sim") and hasattr(b.sim, "cfg") and hasattr(b.sim.cfg, "model_name"):
        b.sim.cfg.model_name = "morelli"
    elif hasattr(b, "sim") and hasattr(b.sim, "f16_model"):
        b.sim.f16_model = "morelli"

    # moderate task change
    e = SetpointJumpWrapper(e, t_jump_s=20.0, sp_new=550.0)

    return e

def make_proof_stress():
    e = make_nominal()

    # observation noise without action mismatch
    e = NoisyDelayedWrapper(e, obs_sigma=4.0, act_delay_steps=0)

    b = _base_env(e)
    if hasattr(b, "sim") and hasattr(b.sim, "cfg") and hasattr(b.sim.cfg, "model_name"):
        b.sim.cfg.model_name = "morelli"
    elif hasattr(b, "sim") and hasattr(b.sim, "f16_model"):
        b.sim.f16_model = "morelli"

    e = SetpointJumpWrapper(e, t_jump_s=22.0, sp_new=560.0)
    return e

def make_jump_stress():
    """
    Clean proof-of-concept stress:
    only a strong setpoint jump, no actuator/path mismatch.
    """
    e = make_nominal()
    e = SetpointJumpWrapper(e, t_jump_s=20.0, sp_new=600.0)
    return e

SCENARIOS = [
    ("Nominal", make_nominal, dict(tol=0.05, window_s=10.0)),
    ("JumpStress", make_jump_stress, dict(tol=0.02, window_s=5.0)),
]


def rollout_ppo(env, model, seed, tol, window_s):
    obs, info = env.reset(seed=seed)
    base = _base_env(env)

    vt = []
    done = False

    while not done:
        a, _ = model.predict(obs, deterministic=True)
        u = float(np.asarray(a).reshape(-1)[0])

        obs, r, done, trunc, info = env.step(np.array([u], dtype=np.float32))
        vt.append(info["Vt"])

        if trunc:
            break

    sat, rho = settling_spec_last_window(
        np.asarray(vt), sp=base.sp, dt=_dt(env), window_s=window_s, tol=tol
    )
    return np.asarray(vt), float(sat), float(rho)


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


def rollout_conf(env, model, seed, tol, window_s):
    obs, info = env.reset(seed=seed)
    base = _base_env(env)

    pred, q = build_predictor(env, model, calib_eps=3, delta=0.30)

    shield = ConformalSTLShield(
        pred, q=q, K=4, dt=_dt(env), tol=tol, slew=0.03
    )
    shield.reset(u0=0.5)

    vt = []
    done = False

    while not done:
        a_rl, _ = model.predict(obs, deterministic=True)
        u_rl = float(np.asarray(a_rl).reshape(-1)[0])

        u, _ = shield.filter(base.sim.Vt, base.sim.pow, base.sp, u_rl)

        obs, r, done, trunc, info = env.step(np.array([u], dtype=np.float32))
        vt.append(info["Vt"])

        if trunc:
            break

    sat, rho = settling_spec_last_window(
        np.asarray(vt), sp=base.sp, dt=_dt(env), window_s=window_s, tol=tol
    )
    return np.asarray(vt), float(sat), float(rho)


def rollout_genai(env, model, seed, tol, window_s):
    obs, info = env.reset(seed=seed)
    base = _base_env(env)

    pred, q = build_predictor(env, model, calib_eps=3, delta=0.30)

    shield = GenerativeConformalShield(
        pred,
        q=q,
        K=3,
        dt=_dt(env),
        tol=tol,
        slew=0.03,
        candidate_offsets=(-0.12, -0.06, 0.0, 0.06, 0.12),
    )
    shield.reset(u0=0.5)

    vt = []
    done = False

    while not done:
        a_rl, _ = model.predict(obs, deterministic=True)
        u_rl = float(np.asarray(a_rl).reshape(-1)[0])

        u, _ = shield.filter(base.sim.Vt, base.sim.pow, base.sp, u_rl)

        obs, r, done, trunc, info = env.step(np.array([u], dtype=np.float32))
        vt.append(info["Vt"])

        if trunc:
            break

    sat, rho = settling_spec_last_window(
        np.asarray(vt), sp=base.sp, dt=_dt(env), window_s=window_s, tol=tol
    )
    return np.asarray(vt), float(sat), float(rho)


def evaluate_method(env_maker, model, method_name, tol, window_s, n=N):
    sats, rhos = [], []
    traces = []

    for i in range(n):
        env = env_maker()
        seed = SEED + i

        if method_name == "PPO":
            vt, sat, rho = rollout_ppo(env, model, seed, tol, window_s)
        elif method_name == "PPO+CONF":
            vt, sat, rho = rollout_conf(env, model, seed, tol, window_s)
        elif method_name == "PPO+GENAI":
            vt, sat, rho = rollout_genai(env, model, seed, tol, window_s)

        sats.append(sat)
        rhos.append(rho)
        traces.append(vt)

    return {
        "sat": float(np.mean(sats)),
        "rho": float(np.mean(rhos)),
        "traces": traces,
    }


def save_bar_plot(rows):
    stress_rows = [r for r in rows if r["scenario"] == "ProofStress"]
    methods = [r["method"] for r in stress_rows]
    sats = [100.0 * r["sat"] for r in stress_rows]
    rhos = [r["rho"] for r in stress_rows]

    plt.figure(figsize=(7, 4))
    plt.bar(methods, sats)
    plt.ylabel("STL Satisfaction (%)")
    plt.title("ProofStress: Safety Satisfaction")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "proofstress_satisfaction.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.bar(methods, rhos)
    plt.ylabel("Mean Robustness")
    plt.title("ProofStress: Mean STL Robustness")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "proofstress_robustness.png"), dpi=200)
    plt.close()


def main():
    os.makedirs(OUT, exist_ok=True)

    env0 = make_nominal()
    model = PPO.load(
        MODEL,
        env=env0,
        custom_objects={
            "learning_rate": 3e-4,
            "clip_range": 0.2,
        },
    )

    rows = []

    print("\n=== Proof-of-Concept Suite ===")
    print(f"{'Scenario':12s} | {'Method':10s} | {'Sat%':>6s} | {'Mean ρ':>7s}")
    print("-" * 50)

    #methods = ["PPO", "PPO+CONF", "PPO+GENAI"]
    methods = ["PPO", "PPO+GENAI"]

    for scenario_name, maker, spec in SCENARIOS:
        for method in methods:
            result = evaluate_method(
                maker, model, method, tol=spec["tol"], window_s=spec["window_s"], n=N
            )

            print(
                f"{scenario_name:12s} | {method:10s} | {100*result['sat']:6.1f} | {result['rho']:7.4f}"
            )

            rows.append(
                {
                    "scenario": scenario_name,
                    "method": method,
                    "tol": spec["tol"],
                    "window_s": spec["window_s"],
                    "sat": result["sat"],
                    "rho": result["rho"],
                }
            )

    with open(os.path.join(OUT, "proof_suite.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(
            f, fieldnames=["scenario", "method", "tol", "window_s", "sat", "rho"]
        )
        w.writeheader()
        w.writerows(rows)

    with open(os.path.join(OUT, "proof_suite.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    save_bar_plot(rows)

    print("\nSaved:", os.path.abspath(OUT))
    print("Plots:")
    print(" - proofstress_satisfaction.png")
    print(" - proofstress_robustness.png")


if __name__ == "__main__":
    main()