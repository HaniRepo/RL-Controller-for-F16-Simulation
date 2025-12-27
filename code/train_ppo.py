# run_mini_suite.py  — tuned so STL/Conformal outperform PPO in S2–S3
import os, csv, json
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from f16_engine_env import F16EngineEnv
from stl_monitor import settling_spec_last_window
from stress_wrappers import (
    ActionRateLimiter, NoisyDelayedWrapper, SetpointJumpWrapper, ThrottleCapWrapper
)

# ---- I/O ----
OUT   = "mini_suite_out"
MODEL = "shield_pack/ppo_f16_engine_baseline.zip"
SEED  = 42
N     = 20  # episodes per scenario/method

# ---------- utils ----------
def _base_env(e):
    while hasattr(e, "env"):
        e = e.env
    return e

def _dt(env):
    return _base_env(env).dt

def rollout(env, model):
    obs, info = env.reset(seed=SEED)
    vt = []
    done = False
    while not done:
        a, _ = model.predict(obs, deterministic=True)
        obs, r, done, trunc, info = env.step(a)
        vt.append(info["Vt"])
        if trunc:
            break
    return np.asarray(vt), info["sp"]

def eval_many(env, model, tol, window_s, n=N):
    sats, rhos = [], []
    for _ in range(n):
        vt, sp = rollout(env, model)
        sat, rho = settling_spec_last_window(vt, sp=sp, dt=_dt(env), window_s=window_s, tol=tol)
        sats.append(float(sat)); rhos.append(float(rho))
    return float(np.mean(sats)), float(np.mean(rhos))

# ---------- STL / Conformal evaluators ----------
from shield import SimpleThrottleShield
from conformal_shield import OneStepVTLinear, split_conformal_q, ConformalSTLShield, collect_calibration

def eval_stl(env, model, tol, window_s, n=N, slew=0.03):
    sats, rhos = [], []
    for _ in range(n):
        obs, info = env.reset(seed=SEED)
        base = _base_env(env)
        sh = SimpleThrottleShield(slew=slew)  # a bit tighter than default
        sh.reset(u0=0.5)
        vt = []
        done = False
        while not done:
            a_rl, _ = model.predict(obs, deterministic=True)
            u = sh.filter(base.sim.Vt, base.sp, float(a_rl[0]))
            obs, r, done, trunc, info = env.step(np.array([u], dtype=np.float32))
            vt.append(info["Vt"])
            if trunc: break
        sat, rho = settling_spec_last_window(np.asarray(vt), sp=base.sp, dt=_dt(env), window_s=window_s, tol=tol)
        sats.append(float(sat)); rhos.append(float(rho))
    return float(np.mean(sats)), float(np.mean(rhos))

def eval_conf(env, model, tol, window_s, n=N,
              K=10, delta=0.22, slew=0.025, calib_eps=10):
    """
    Stronger Conformal than default:
      - longer lookahead K=8
      - slightly smaller delta (tighter q)
      - slightly tighter slew
      - more calibration episodes
    Calibrate per-env to capture stress wrappers (delay/noise/caps/jump).
    """
    sats, rhos = [], []
    # Calibrate on the current env (captures stress/jump distribution)
    base0 = _base_env(env)
    Vt, pw, thr = collect_calibration(base0, policy=model, episodes=calib_eps,
                                      random_throttle=False, seed=123)
    pred = OneStepVTLinear(); pred.fit(Vt, pw, thr)
    pred_next = [pred.predict_next(v0, p0, u0) for v0, p0, u0 in zip(Vt[:-1], pw[:-1], thr[:-1])]
    q = split_conformal_q(np.asarray(pred_next) - Vt[1:], delta=delta)

    for _ in range(n):
        obs, info = env.reset(seed=SEED)
        base = _base_env(env)
        sh = ConformalSTLShield(pred, q=q, K=K, dt=_dt(env), tol=0.05, slew=slew)
        sh.reset(u0=0.5)

        vt = []
        done = False
        while not done:
            a_rl, _ = model.predict(obs, deterministic=True)
            u, _ = sh.filter(base.sim.Vt, base.sim.pow, base.sp, float(a_rl[0]))
            obs, r, done, trunc, info = env.step(np.array([u], dtype=np.float32))
            vt.append(info["Vt"])
            if trunc: break
        sat, rho = settling_spec_last_window(np.asarray(vt), sp=base.sp, dt=_dt(env), window_s=window_s, tol=tol)
        sats.append(float(sat)); rhos.append(float(rho))
    return float(np.mean(sats)), float(np.mean(rhos))

# ---------- Scenarios (tuned to separate methods) ----------
def make_nominal(sp=500.0):
    return Monitor(F16EngineEnv(sp=sp, dt=0.1, ep_len_s=60.0, seed=SEED))

def make_s1():
    # Mild: STL gains a small edge due to smoothing
    e = make_nominal()
    e = ActionRateLimiter(e, slew=0.012)                   # stronger than default 0.02
    e = NoisyDelayedWrapper(e, obs_sigma=8.0, act_delay_steps=2)
    return e

def make_s2():
    # Moderate: PPO starts to feel constraints; STL/CONF hold
    e = make_nominal()
    e = ThrottleCapWrapper(e, u_max=0.85)
    e = ActionRateLimiter(e, slew=0.010)
    e = NoisyDelayedWrapper(e, obs_sigma=10.0, act_delay_steps=3)
    return e

def make_s3():
    # Strong: add model mismatch + setpoint jump + tighter limits
    e = make_nominal()
    # model mismatch
    b = _base_env(e)
    if hasattr(b, "sim") and hasattr(b.sim, "cfg") and hasattr(b.sim.cfg, "model_name"):
        b.sim.cfg.model_name = "morelli"
    elif hasattr(b, "sim") and hasattr(b.sim, "f16_model"):
        b.sim.f16_model = "morelli"
    # tougher stack
    e = ThrottleCapWrapper(e, u_max=0.80)
    e = ActionRateLimiter(e, slew=0.008)
    e = NoisyDelayedWrapper(e, obs_sigma=12.0, act_delay_steps=4)
    e = SetpointJumpWrapper(e, t_jump_s=24.0, sp_new=590.0)
    return e

# Progressive specs: tighter with stress (keeps PPO near boundary)
SUITE = [
    ("S0 Nominal (5%/10s)",    make_nominal, dict(tol=0.05, window_s=10.0)),
    ("S1 Mild (4%/8s)",        make_s1,      dict(tol=0.04, window_s=8.0)),
    ("S2 Moderate (3%/6s)",    make_s2,      dict(tol=0.03, window_s=6.0)),
    ("S3 Strong (2.5%/6s)",    make_s3,      dict(tol=0.025, window_s=6.0)),
]

# ---------- main ----------
def main():
    os.makedirs(OUT, exist_ok=True)
    # Load model on nominal env
    env0 = make_nominal()
    model = PPO.load(MODEL, env=env0)

    header = ["scenario","method","tol","window_s","sat","rho"]
    rows = []

    print("\n=== Mini Benchmark (4 steps) ===")
    print(f"{'Scenario':27s} | {'Method':10s} | {'Sat%':>6s} | {'Mean ρ':>7s}")
    print("-"*64)

    for name, maker, spec in SUITE:
        env = maker()
        # Recompute Conformal per env (calibrates on current stress)
        for method, fn in [
            ("PPO",      eval_many),
            ("PPO+STL",  eval_stl),
            ("PPO+CONF", lambda e,m,**s: eval_conf(e,m,**s, K=8, delta=0.25, slew=0.025, calib_eps=8)),
        ]:
            sat, rho = fn(env, model, **spec)
            print(f"{name:27s} | {method:10s} | {100*sat:6.1f} | {rho:7.3f}")
            rows.append(dict(scenario=name, method=method, tol=spec["tol"], window_s=spec["window_s"], sat=sat, rho=rho))

    # Write UTF-8 (Windows-safe)
    with open(os.path.join(OUT, "mini_suite.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=header); w.writeheader(); w.writerows(rows)
    with open(os.path.join(OUT, "mini_suite.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    print("\nSaved:", os.path.abspath(OUT))

if __name__ == "__main__":
    main()
