# package_baseline.py
import os, json, shutil, sys, subprocess, pathlib
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from f16_engine_env import F16EngineEnv
from stl_monitor import settling_spec_last_window
from utils_plot import plot_vt

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
    # unwrap common wrappers (Monitor, VecEnv wrappers, etc.)
    while hasattr(e, "env"):
        e = e.env
    return e

def rollout(env, model, deterministic=True):
    obs, info = env.reset(seed=SEED)
    vt_hist, t_hist = [], []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, r, done, trunc, info = env.step(action)
        vt_hist.append(info["Vt"]); t_hist.append(info["t"])
        if trunc:
            break
    return np.array(t_hist), np.array(vt_hist), info["sp"]

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

def eval_many_conformal(env, model, n_episodes=N_EVAL, n_calib_eps=8, delta=0.20, K=10):
    """
    Conformal STL shield:
      - calibrate on policy rollouts (better q),
      - shorter lookahead (K=10),
      - slightly looser delta=0.20 (tighter q -> less over-conservative).
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

    # Build env, load model
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

    # Batch evaluations
    sat_rate, rho_mean = eval_many(env, model, n_episodes=N_EVAL)
    sat_rate_sh, rho_mean_sh = eval_many_shielded(env, model, n_episodes=N_EVAL)
    sat_rate_conf, rho_mean_conf = eval_many_conformal(env, model,
                                                       n_episodes=N_EVAL, n_calib_eps=8,
                                                       delta=0.30, K=6)

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
        }
    }

        # ---------- Distribution shift: setpoint 450 / 550 ----------
    def eval_at_setpoint(env, model, sp, n_episodes=20):
        base = _base_env(env)
        sp_old = base.sp
        base.sp = sp
        dt_val = _get_dt(env)
        sats, rhos = [], []
        for _ in range(n_episodes):
            t, vt, _ = rollout(env, model, deterministic=True)
            sat, rho = settling_spec_last_window(vt, sp=sp, dt=dt_val, window_s=10.0, tol=0.05)
            sats.append(float(sat)); rhos.append(float(rho))
        base.sp = sp_old
        return float(np.mean(sats)), float(np.mean(rhos))

    def eval_conformal_at_sp(env, model, sp, n_episodes=20, delta=0.30, K=6):
        from conformal_shield import OneStepVTLinear, split_conformal_q, ConformalSTLShield, collect_calibration
        base = _base_env(env)
        sp_old = base.sp
        base.sp = sp

        # calibrate on policy at this setpoint (policy-driven calibration)
        Vt, pw, thr = collect_calibration(base, policy=model, episodes=6,
                                          random_throttle=False, seed=321)
        pred = OneStepVTLinear(); pred.fit(Vt, pw, thr)
        pred_next = [pred.predict_next(vt0, pw0, u0) for vt0, pw0, u0 in zip(Vt[:-1], pw[:-1], thr[:-1])]
        q = split_conformal_q(np.asarray(pred_next) - Vt[1:], delta=delta)

        dt_val = _get_dt(env)
        sats, rhos = [], []
        for _ in range(n_episodes):
            obs, info = env.reset()
            base2 = _base_env(env)
            sh = ConformalSTLShield(pred, q=q, K=K, dt=dt_val, tol=0.05, slew=0.03)
            sh.reset(u0=0.5)

            vt_hist, done = [], False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                u_rl = float(action[0])
                u, _ = sh.filter(base2.sim.Vt, base2.sim.pow, base2.sp, u_rl)
                obs, r, done, trunc, info = env.step(np.array([u], dtype=np.float32))
                vt_hist.append(info["Vt"])
                if trunc: break

            sat, rho = settling_spec_last_window(np.array(vt_hist), sp=sp, dt=dt_val, window_s=10.0, tol=0.05)
            sats.append(float(sat)); rhos.append(float(rho))

        base.sp = sp_old
        return float(np.mean(sats)), float(np.mean(rhos))

    # run the shift evals
    sat_450, rho_450 = eval_at_setpoint(env, model, sp=450.0, n_episodes=20)
    sat_550, rho_550 = eval_at_setpoint(env, model, sp=550.0, n_episodes=20)
    satc_450, rhoc_450 = eval_conformal_at_sp(env, model, sp=450.0)
    satc_550, rhoc_550 = eval_conformal_at_sp(env, model, sp=550.0)

    METRICS["shift_eval"] = {
        "setpoint_450": {
            "ppo": {"stl_satisfaction_rate": sat_450, "rho_mean": rho_450},
            "conformal": {"stl_satisfaction_rate": satc_450, "rho_mean": rhoc_450},
        },
        "setpoint_550": {
            "ppo": {"stl_satisfaction_rate": sat_550, "rho_mean": rho_550},
            "conformal": {"stl_satisfaction_rate": satc_550, "rho_mean": rhoc_550},
        }
    }

    # quick printout
    print("\n--- Distribution shift (setpoint) ---")
    print(f"sp=450: PPO {sat_450*100:.1f}% (ρ={rho_450:.3f})  |  CONF {satc_450*100:.1f}% (ρ={rhoc_450:.3f})")
    print(f"sp=550: PPO {sat_550*100:.1f}% (ρ={rho_550:.3f})  |  CONF {satc_550*100:.1f}% (ρ={rhoc_550:.3f})")



    with open(os.path.join(PACK_DIR, "metrics.json"), "w") as f:
        json.dump(METRICS, f, indent=2)

    # Versions & reproducibility info
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

    # Console comparison table
    print("\n=== STL Satisfaction Comparison ===")
    print(f"{'Method':30s} {'Satisfaction rate':>20s} {'Mean ρ':>12s}")
    print("-"*65)
    print(f"{'PPO (baseline)':30s} "
          f"{METRICS['batch_eval']['stl_satisfaction_rate']*100:>19.1f}% "
          f"{METRICS['batch_eval']['rho_mean']:>12.3f}")
    print(f"{'PPO + STL shield':30s} "
          f"{METRICS['shielded_eval']['stl_satisfaction_rate']*100:>19.1f}% "
          f"{METRICS['shielded_eval']['rho_mean']:>12.3f}")
    print(f"{'PPO + Conformal STL shield':30s} "
          f"{METRICS['conformal_shielded_eval']['stl_satisfaction_rate']*100:>19.1f}% "
          f"{METRICS['conformal_shielded_eval']['rho_mean']:>12.3f}")
    print("\n(single_rollout) STL satisfied:",
          METRICS['single_rollout']['stl_satisfied'],
          "   ρ =", METRICS['single_rollout']['rho'])

    print("\n✅ Baseline pack created at:", os.path.abspath(PACK_DIR))
    print("   - Model:", model_dst)
    print("   - Plot: vt_tracking.png")
    print("   - CSV: rollout.csv")
    print("   - Metrics: metrics.json")
    print("   - Config: config.json")
    print("   - Versions: versions.json, requirements.lock.txt")

if __name__ == "__main__":
    main()
