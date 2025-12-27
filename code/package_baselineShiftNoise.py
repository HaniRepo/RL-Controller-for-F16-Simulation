# package_baselineShiftNoise.py
import os, json, shutil, sys, subprocess, pathlib
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from f16_engine_env import F16EngineEnv
from stl_monitor import settling_spec_last_window
from utils_plot import plot_vt

# wrappers (make sure stress_wrappers.py contains these classes)
from stress_wrappers import (
    NoisyDelayedWrapper,
    ActionRateLimiter,
    SetpointJumpWrapper,
    ThrottleCapWrapper,     # you added this earlier
)

PACK_DIR = "shield_pack"
MODEL_SRC = "ppo_f16_engine.zip"

SEED = 42
N_EVAL = 20

# ---------------- helpers ----------------
def _get_dt(env):
    if hasattr(env, "dt"):
        return env.dt
    if hasattr(env, "env") and hasattr(env.env, "dt"):
        return env.env.dt
    try:
        return env.unwrapped.dt
    except Exception:
        pass
    raise AttributeError("Could not find dt on env (wrapped by Monitor?)")

def _base_env(env):
    """Return the underlying (unwrapped) env that has .sim, .sp, .dt, etc."""
    e = env
    while hasattr(e, "env"):
        e = e.env
    return e

def rollout(env, model, deterministic=True):
    obs, info = env.reset(seed=SEED)
    vt_hist, t_hist, sp_hist = [], [], [info.get("sp", _base_env(env).sp)]
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, r, done, trunc, info = env.step(action)
        vt_hist.append(info["Vt"]); t_hist.append(info["t"]); sp_hist.append(info.get("sp", sp_hist[-1]))
        if trunc:
            break
    return np.array(t_hist), np.array(vt_hist), sp_hist[-1] if len(sp_hist)>0 else info["sp"]

# ---- vanilla evaluations (5% band, last 10s) ----
def eval_many(env, model, n_episodes=N_EVAL):
    sats, rhos = [], []
    dt_val = _get_dt(env)
    for _ in range(n_episodes):
        t, vt, sp = rollout(env, model, deterministic=True)
        sat, rho = settling_spec_last_window(vt, sp=sp, dt=dt_val, window_s=10.0, tol=0.05)
        sats.append(float(sat)); rhos.append(float(rho))
    return float(np.mean(sats)), float(np.mean(rhos))

def eval_many_shielded(env, model, n_episodes=N_EVAL):
    # Simple rule-based STL shield
    from shield import SimpleThrottleShield
    sats, rhos = [], []
    dt_val = _get_dt(env)
    for _ in range(n_episodes):
        obs, info = env.reset()
        base = _base_env(env)
        sh = SimpleThrottleShield(slew=0.05)
        sh.reset(u0=0.5)

        vt_hist = []
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            u_rl = float(action[0])
            vt = base.sim.Vt
            sp = base.sp
            u = sh.filter(vt, sp, u_rl)
            obs, r, done, trunc, info = env.step(np.array([u], dtype=np.float32))
            vt_hist.append(info["Vt"])
            if trunc:
                break

        sat, rho = settling_spec_last_window(np.array(vt_hist), sp=base.sp, dt=dt_val, window_s=10.0, tol=0.05)
        sats.append(float(sat)); rhos.append(float(rho))
    return float(np.mean(sats)), float(np.mean(rhos))

def eval_many_conformal(env, model, n_episodes=N_EVAL, n_calib_eps=8, delta=0.30, K=6):
    """
    Conformal STL shield:
      - calibrate on policy rollouts (better q),
      - shorter lookahead (K=6),
      - slightly looser delta=0.30 (less over-conservative).
    """
    from conformal_shield import OneStepVTLinear, split_conformal_q, ConformalSTLShield, collect_calibration

    base = _base_env(env)

    # Calibrate on the policy distribution (not random throttle)
    Vt, pw, thr = collect_calibration(base, policy=model, episodes=n_calib_eps,
                                      random_throttle=False, seed=123)

    pred = OneStepVTLinear()
    pred.fit(Vt, pw, thr)

    # residuals for conformal q
    pred_next = [pred.predict_next(vt0, pw0, u0) for vt0, pw0, u0 in zip(Vt[:-1], pw[:-1], thr[:-1])]
    q = split_conformal_q(np.asarray(pred_next) - Vt[1:], delta=delta)

    sats, rhos = [], []
    dt_val = _get_dt(env)

    for _ in range(n_episodes):
        obs, info = env.reset()
        base = _base_env(env)

        # slightly tighter slew; STL tol already 5%
        from conformal_shield import ConformalSTLShield
        sh = ConformalSTLShield(pred, q=q, K=K, dt=dt_val, tol=0.05, slew=0.03)
        sh.reset(u0=0.5)

        vt_hist = []
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            u_rl = float(action[0])
            u, _rho_worst = sh.filter(base.sim.Vt, base.sim.pow, base.sp, u_rl)
            obs, r, done, trunc, info = env.step(np.array([u], dtype=np.float32))
            vt_hist.append(info["Vt"])
            if trunc:
                break

        sat, rho = settling_spec_last_window(np.array(vt_hist), sp=base.sp, dt=dt_val, window_s=10.0, tol=0.05)
        sats.append(float(sat)); rhos.append(float(rho))
    return float(np.mean(sats)), float(np.mean(rhos))

# ---- parametric evaluations (e.g., 2% band, last 5s) ----
def eval_many_with_params(env, model, n_episodes=20, tol=0.05, window_s=10.0):
    sats, rhos = [], []
    dt_val = _get_dt(env)
    for _ in range(n_episodes):
        t, vt, sp = rollout(env, model, deterministic=True)
        sat, rho = settling_spec_last_window(vt, sp=sp, dt=dt_val, window_s=window_s, tol=tol)
        sats.append(float(sat)); rhos.append(float(rho))
    return float(np.mean(sats)), float(np.mean(rhos))

def eval_many_shielded_with_params(env, model, n_episodes=20, tol=0.05, window_s=10.0):
    from shield import SimpleThrottleShield
    sats, rhos = [], []
    dt_val = _get_dt(env)
    for _ in range(n_episodes):
        obs, info = env.reset()
        base = _base_env(env)
        sh = SimpleThrottleShield(slew=0.05)
        sh.reset(u0=0.5)
        vt_hist, done = [], False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            u_rl = float(action[0])
            u = sh.filter(base.sim.Vt, base.sp, u_rl)
            obs, r, done, trunc, info = env.step(np.array([u], dtype=np.float32))
            vt_hist.append(info["Vt"])
            if trunc: break
        sat, rho = settling_spec_last_window(np.array(vt_hist), sp=base.sp, dt=dt_val, window_s=window_s, tol=tol)
        sats.append(float(sat)); rhos.append(float(rho))
    return float(np.mean(sats)), float(np.mean(rhos))

def eval_many_conformal_with_params(env, model, n_episodes=20, tol=0.05, window_s=10.0,
                                    n_calib_eps=8, delta=0.30, K=6):
    from conformal_shield import OneStepVTLinear, split_conformal_q, ConformalSTLShield, collect_calibration
    base = _base_env(env)
    Vt, pw, thr = collect_calibration(base, policy=model, episodes=n_calib_eps,
                                      random_throttle=False, seed=123)
    pred = OneStepVTLinear(); pred.fit(Vt, pw, thr)
    pred_next = [pred.predict_next(vt0, pw0, u0) for vt0, pw0, u0 in zip(Vt[:-1], pw[:-1], thr[:-1])]
    q = split_conformal_q(np.asarray(pred_next) - Vt[1:], delta=delta)
    sats, rhos = [], []
    dt_val = _get_dt(env)
    for _ in range(n_episodes):
        obs, info = env.reset()
        base = _base_env(env)
        sh = ConformalSTLShield(pred, q=q, K=K, dt=dt_val, tol=0.05, slew=0.03)
        sh.reset(u0=0.5)
        vt_hist, done = [], False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            u_rl = float(action[0])
            u, _ = sh.filter(base.sim.Vt, base.sim.pow, base.sp, u_rl)
            obs, r, done, trunc, info = env.step(np.array([u], dtype=np.float32))
            vt_hist.append(info["Vt"])
            if trunc: break
        sat, rho = settling_spec_last_window(np.array(vt_hist), sp=base.sp, dt=dt_val, window_s=window_s, tol=tol)
        sats.append(float(sat)); rhos.append(float(rho))
    return float(np.mean(sats)), float(np.mean(rhos))

# ---------------- main ----------------
def main():
    os.makedirs(PACK_DIR, exist_ok=True)

    # Config snapshot
    CONFIG = {
        "task": "F16 Engine Control (2-var)",
        "setpoint": 500.0,
        "dt": 0.1,
        "horizon_s": 60.0,
        "seed": SEED,
        "algo": "PPO",
        "ppo": {"n_steps": 2048, "batch_size": 64, "learning_rate": 3e-4,
                "gamma": 0.995, "gae_lambda": 0.95, "clip_range": 0.2},
        "obs": "[Vt/sp, pow_norm]",
        "action": "throttle in [0,1]",
        "reward_notes": "tracking error + smoothness + progress bonus + in-band bonus",
        "stl_spec": "G_[50,60] ( |(Vt-sp)/sp| <= 0.05 )  (checked via last-10s window)"
    }
    with open(os.path.join(PACK_DIR, "config.json"), "w") as f:
        json.dump(CONFIG, f, indent=2)

    # --- NOMINAL env & model ---
    env = F16EngineEnv(sp=CONFIG["setpoint"], dt=CONFIG["dt"], ep_len_s=CONFIG["horizon_s"], seed=SEED)
    env = Monitor(env)

    if not os.path.exists(MODEL_SRC):
        raise FileNotFoundError(f"Model file not found: {MODEL_SRC}")
    model_dst = os.path.join(PACK_DIR, "ppo_f16_engine_baseline.zip")
    shutil.copyfile(MODEL_SRC, model_dst)

    model = PPO.load(model_dst, env=env)

    # One canonical rollout (plot + CSV)
    t, vt, sp = rollout(env, model, deterministic=True)
    dt_val = _get_dt(env)
    sat, rho = settling_spec_last_window(vt, sp=sp, dt=dt_val, window_s=10.0, tol=0.05)

    fig = plot_vt(t, vt, sp, title="Airspeed Tracking (PPO baseline)")
    fig.savefig(os.path.join(PACK_DIR, "vt_tracking.png"), dpi=160)

    import csv
    with open(os.path.join(PACK_DIR, "rollout.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_sec", "Vt_ftps"])
        for ti, vi in zip(t, vt):
            w.writerow([float(ti), float(vi)])

    # --- Nominal batch evaluations (5%, last 10s) ---
    sat_rate, rho_mean = eval_many(env, model, n_episodes=N_EVAL)
    sat_rate_sh, rho_mean_sh = eval_many_shielded(env, model, n_episodes=N_EVAL)
    sat_rate_conf, rho_mean_conf = eval_many_conformal(env, model,
                                                       n_episodes=N_EVAL, n_calib_eps=8,
                                                       delta=0.30, K=6)

    # ---------- Distribution shift: setpoint 450 / 550 ----------
    def eval_at_setpoint(env_in, model_in, sp_val, n_episodes=20):
        base = _base_env(env_in)
        sp_old = base.sp
        base.sp = sp_val
        dt_local = _get_dt(env_in)
        sats, rhos = [], []
        for _ in range(n_episodes):
            t1, vt1, _ = rollout(env_in, model_in, deterministic=True)
            s1, r1 = settling_spec_last_window(vt1, sp=sp_val, dt=dt_local, window_s=10.0, tol=0.05)
            sats.append(float(s1)); rhos.append(float(r1))
        base.sp = sp_old
        return float(np.mean(sats)), float(np.mean(rhos))

    def eval_conformal_at_sp(env_in, model_in, sp_val, n_episodes=20, delta=0.30, K=6):
        from conformal_shield import OneStepVTLinear, split_conformal_q, ConformalSTLShield, collect_calibration
        base = _base_env(env_in)
        sp_old = base.sp
        base.sp = sp_val

        # calibrate on policy at this setpoint (policy-driven calibration)
        Vt_c, pw_c, thr_c = collect_calibration(base, policy=model_in, episodes=6,
                                                random_throttle=False, seed=321)
        pred_c = OneStepVTLinear(); pred_c.fit(Vt_c, pw_c, thr_c)
        pred_next_c = [pred_c.predict_next(vt0, pw0, u0) for vt0, pw0, u0 in zip(Vt_c[:-1], pw_c[:-1], thr_c[:-1])]
        q_c = split_conformal_q(np.asarray(pred_next_c) - Vt_c[1:], delta=delta)

        dt_local = _get_dt(env_in)
        sats, rhos = [], []
        for _ in range(n_episodes):
            obs, info = env_in.reset()
            base2 = _base_env(env_in)
            sh = ConformalSTLShield(pred_c, q=q_c, K=K, dt=dt_local, tol=0.05, slew=0.03)
            sh.reset(u0=0.5)

            vt_hist, done = [], False
            while not done:
                action, _ = model_in.predict(obs, deterministic=True)
                u_rl = float(action[0])
                u_applied, _ = sh.filter(base2.sim.Vt, base2.sim.pow, base2.sp, u_rl)
                obs, r, done, trunc, info = env_in.step(np.array([u_applied], dtype=np.float32))
                vt_hist.append(info["Vt"])
                if trunc: break

            s1, r1 = settling_spec_last_window(np.array(vt_hist), sp=sp_val, dt=dt_local, window_s=10.0, tol=0.05)
            sats.append(float(s1)); rhos.append(float(r1))

        base.sp = sp_old
        return float(np.mean(sats)), float(np.mean(rhos))

    sat_450, rho_450 = eval_at_setpoint(env, model, sp_val=450.0, n_episodes=20)
    sat_550, rho_550 = eval_at_setpoint(env, model, sp_val=550.0, n_episodes=20)
    satc_450, rhoc_450 = eval_conformal_at_sp(env, model, sp_val=450.0)
    satc_550, rhoc_550 = eval_conformal_at_sp(env, model, sp_val=550.0)

    METRICS = {
        "single_rollout": {"stl_satisfied": bool(sat), "rho": float(rho)},
        "batch_eval": {
            "episodes": N_EVAL,
            "stl_satisfaction_rate": float(sat_rate),
            "rho_mean": float(rho_mean)
        },
        "shielded_eval": {
            "episodes": N_EVAL,
            "stl_satisfaction_rate": float(sat_rate_sh),
            "rho_mean": float(rho_mean_sh)
        },
        "conformal_shielded_eval": {
            "episodes": N_EVAL,
            "stl_satisfaction_rate": float(sat_rate_conf),
            "rho_mean": float(rho_mean_conf)
        },
        "shift_eval": {
            "setpoint_450": {
                "ppo": {"stl_satisfaction_rate": sat_450, "rho_mean": rho_450},
                "conformal": {"stl_satisfaction_rate": satc_450, "rho_mean": rhoc_450},
            },
            "setpoint_550": {
                "ppo": {"stl_satisfaction_rate": sat_550, "rho_mean": rho_550},
                "conformal": {"stl_satisfaction_rate": satc_550, "rho_mean": rhoc_550},
            }
        }
    }

    print("\n--- Distribution shift (setpoint) ---")
    print(f"sp=450: PPO {sat_450*100:.1f}% (ρ={rho_450:.3f})  |  CONF {satc_450*100:.1f}% (ρ={rhoc_450:.3f})")
    print(f"sp=550: PPO {sat_550*100:.1f}% (ρ={rho_550:.3f})  |  CONF {satc_550*100:.1f}% (ρ={rhoc_550:.3f})")

    # ---------- STRESS TEST env: Morelli + cap + slew + noise/delay + jump ----------
    eval_env = F16EngineEnv(sp=CONFIG["setpoint"], dt=CONFIG["dt"], ep_len_s=CONFIG["horizon_s"], seed=SEED)
    eval_env = Monitor(eval_env)

    # flip to Morelli (model mismatch)
    base_e = eval_env
    while hasattr(base_e, "env"): base_e = base_e.env
    if hasattr(base_e, "sim") and hasattr(base_e.sim, "cfg") and hasattr(base_e.sim.cfg, "model_name"):
        base_e.sim.cfg.model_name = "morelli"
    elif hasattr(base_e, "sim") and hasattr(base_e.sim, "f16_model"):
        base_e.sim.f16_model = "morelli"

    # STRONGER stress (cap -> rate -> noise+delay -> jump)
    eval_env = ThrottleCapWrapper(eval_env, u_max=0.80)
    eval_env = ActionRateLimiter(eval_env, slew=0.008)
    eval_env = NoisyDelayedWrapper(eval_env, obs_sigma=12.0, act_delay_steps=4)
    eval_env = SetpointJumpWrapper(eval_env, t_jump_s=20.0, sp_new=590.0)

    # --- STRESS TEST with harder spec (2% band, 5s window) ---
    sat_rate_s,    rho_mean_s    = eval_many_with_params(eval_env, model, n_episodes=N_EVAL, tol=0.02, window_s=5.0)
    sat_rate_sh_s, rho_mean_sh_s = eval_many_shielded_with_params(eval_env, model, n_episodes=N_EVAL, tol=0.02, window_s=5.0)
    sat_rate_conf_s, rho_mean_conf_s = eval_many_conformal_with_params(eval_env, model, n_episodes=N_EVAL,
                                                                       tol=0.02, window_s=5.0,
                                                                       n_calib_eps=8, delta=0.30, K=6)

    METRICS["stress_eval"] = {
        "baseline":   {"episodes": N_EVAL, "stl_satisfaction_rate": float(sat_rate_s),    "rho_mean": float(rho_mean_s)},
        "shielded":   {"episodes": N_EVAL, "stl_satisfaction_rate": float(sat_rate_sh_s), "rho_mean": float(rho_mean_sh_s)},
        "conformal":  {"episodes": N_EVAL, "stl_satisfaction_rate": float(sat_rate_conf_s),"rho_mean": float(rho_mean_conf_s)},
        "wrappers":   {"obs_sigma": 12.0, "act_delay_steps": 4, "rate_slew": 0.008, "throttle_cap": 0.80, "sp_jump_to": 590.0, "t_jump_s": 20.0},
        "spec":       {"tol": 0.02, "window_s": 5.0}
    }

    # ---------- write metrics & versions ----------
    with open(os.path.join(PACK_DIR, "metrics.json"), "w") as f:
        json.dump(METRICS, f, indent=2)

    versions = {}
    try:
        import gymnasium, stable_baselines3, numpy, matplotlib
        versions = {
            "python": sys.version.replace("\n", " "),
            "gymnasium": gymnasium.__version__,
            "stable_baselines3": stable_baselines3.__version__,
            "numpy": numpy.__version__,
            "matplotlib": matplotlib.__version__
        }
    except Exception as e:
        versions["warn"] = f"could not collect some versions: {e}"

    try:
        res = subprocess.run([sys.executable, "-m", "pip", "freeze"],
                             capture_output=True, text=True, check=False)
        pathlib.Path(os.path.join(PACK_DIR, "requirements.lock.txt")).write_text(res.stdout)
    except Exception as e:
        versions["pip_freeze_error"] = str(e)

    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"],
                             capture_output=True, text=True, check=True)
        versions["git_commit"] = res.stdout.strip()
    except Exception:
        versions["git_commit"] = None

    with open(os.path.join(PACK_DIR, "versions.json"), "w") as f:
        json.dump(versions, f, indent=2)

    # ---------- console summary ----------
    print("\n--- STRESS TEST (2% band, 5s window; Morelli+cap+slew+noise/delay+jump) ---")
    print(f"{'Method':30s} {'Satisfaction rate':>20s} {'Mean ρ':>12s}")
    print("-"*65)
    print(f"{'PPO (baseline)':30s} {sat_rate_s*100:>19.1f}% {rho_mean_s:>12.3f}")
    print(f"{'PPO + STL shield':30s} {sat_rate_sh_s*100:>19.1f}% {rho_mean_sh_s:>12.3f}")
    print(f"{'PPO + Conformal':30s} {sat_rate_conf_s*100:>19.1f}% {rho_mean_conf_s:>12.3f}")

    print("\n=== STL Satisfaction Comparison (NOMINAL, 5%/10s) ===")
    print(f"{'Method':30s} {'Satisfaction rate':>20s} {'Mean ρ':>12s}")
    print("-"*65)
    print(f"{'PPO (baseline)':30s} "
          f"{sat_rate*100:>19.1f}% {rho_mean:>12.3f}")
    print(f"{'PPO + STL shield':30s} "
          f"{sat_rate_sh*100:>19.1f}% {rho_mean_sh:>12.3f}")
    print(f"{'PPO + Conformal STL shield':30s} "
          f"{sat_rate_conf*100:>19.1f}% {rho_mean_conf:>12.3f}")
    print("\n(single_rollout) STL satisfied:",
          bool(sat), "   ρ =", float(rho))

    print("\n✅ Baseline pack created at:", os.path.abspath(PACK_DIR))
    print("   - Model:", model_dst)
    print("   - Plot: vt_tracking.png")
    print("   - CSV: rollout.csv")
    print("   - Metrics: metrics.json")
    print("   - Config: config.json")
    print("   - Versions: versions.json, requirements.lock.txt")

if __name__ == "__main__":
    main()
