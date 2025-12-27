import os, json, csv
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from f16_engine_env import F16EngineEnv
from stl_monitor import settling_spec_last_window
from utils_plot import plot_vt
from shield import SimpleThrottleShield
from conformal_shield import OneStepVTLinear, split_conformal_q, ConformalSTLShield, collect_calibration

OUTDIR = "figs"
MODEL_PATH = "shield_pack/ppo_f16_engine_baseline.zip"
SEED = 42

def _base_env(env):
    e = env
    while hasattr(e, "env"):
        e = e.env
    return e

def _get_dt(env):
    if hasattr(env, "dt"): return env.dt
    if hasattr(env, "env") and hasattr(env.env, "dt"): return env.env.dt
    return env.unwrapped.dt

def rollout_baseline(env, model, deterministic=True):
    obs, info = env.reset(seed=SEED)
    t_hist, vt_hist = [], []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, r, done, trunc, info = env.step(action)
        t_hist.append(info["t"]); vt_hist.append(info["Vt"])
        if trunc: break
    return np.array(t_hist), np.array(vt_hist), info["sp"]

def rollout_shield(env, model):
    base = _base_env(env)
    sh = SimpleThrottleShield(slew=0.05); sh.reset(u0=0.5)
    obs, info = env.reset(seed=SEED)
    t_hist, vt_hist = [], []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        u = sh.filter(base.sim.Vt, base.sp, float(action[0]))
        obs, r, done, trunc, info = env.step(np.array([u], dtype=np.float32))
        t_hist.append(info["t"]); vt_hist.append(info["Vt"])
        if trunc: break
    return np.array(t_hist), np.array(vt_hist), base.sp

def build_conformal(env, model, delta=0.30, K=6, slew=0.03, calib_eps=6):
    base = _base_env(env)
    Vt, pw, thr = collect_calibration(base, policy=model, episodes=calib_eps,
                                      random_throttle=False, seed=123)
    pred = OneStepVTLinear(); pred.fit(Vt, pw, thr)
    pred_next = [pred.predict_next(vt0, pw0, u0) for vt0, pw0, u0 in zip(Vt[:-1], pw[:-1], thr[:-1])]
    q = split_conformal_q(np.asarray(pred_next) - Vt[1:], delta=delta)
    sh = ConformalSTLShield(pred, q=q, K=K, dt=_get_dt(env), tol=0.05, slew=slew)
    sh.reset(u0=0.5)
    return sh

def rollout_conformal(env, model, sh):
    base = _base_env(env)
    obs, info = env.reset(seed=SEED)
    t_hist, vt_hist = [], []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        u, _ = sh.filter(base.sim.Vt, base.sim.pow, base.sp, float(action[0]))
        obs, r, done, trunc, info = env.step(np.array([u], dtype=np.float32))
        t_hist.append(info["t"]); vt_hist.append(info["Vt"])
        if trunc: break
    return np.array(t_hist), np.array(vt_hist), base.sp

def save_tracking_plot(t, vt, sp, title, path_png):
    fig = plot_vt(t, vt, sp, title=title)
    fig.savefig(path_png, dpi=160)
    plt.close(fig)

def eval_stl(vt, sp, dt, window_s=10.0, tol=0.05):
    sat, rho = settling_spec_last_window(vt, sp=sp, dt=dt, window_s=window_s, tol=tol)
    return bool(sat), float(rho)

def set_setpoint(env, sp):
    base = _base_env(env)
    base.sp = float(sp)

def main():
    os.makedirs(OUTDIR, exist_ok=True)

    # env + model
    env = Monitor(F16EngineEnv(sp=500.0, dt=0.1, ep_len_s=60.0, seed=SEED))
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model not found: " + MODEL_PATH)
    model = PPO.load(MODEL_PATH, env=env)

    # scenarios
    scenarios = [
        ("nominal_500", 500.0),
        ("shift_450",   450.0),
        ("shift_550",   550.0),
    ]

    # figure 1/2/3: time-series comparisons per scenario
    summary = {}
    for tag, sp in scenarios:
        set_setpoint(env, sp)

        # Baseline
        t_b, vt_b, sp_b = rollout_baseline(env, model)
        sat_b, rho_b = eval_stl(vt_b, sp_b, _get_dt(env))
        save_tracking_plot(t_b, vt_b, sp_b, f"Baseline PPO (sp={sp})", os.path.join(OUTDIR, f"vt_{tag}_baseline.png"))

        # STL shield
        t_s, vt_s, sp_s = rollout_shield(env, model)
        sat_s, rho_s = eval_stl(vt_s, sp_s, _get_dt(env))
        save_tracking_plot(t_s, vt_s, sp_s, f"PPO + STL shield (sp={sp})", os.path.join(OUTDIR, f"vt_{tag}_shield.png"))

        # Conformal shield
        sh = build_conformal(env, model, delta=0.30, K=6, slew=0.03, calib_eps=6)
        t_c, vt_c, sp_c = rollout_conformal(env, model, sh)
        sat_c, rho_c = eval_stl(vt_c, sp_c, _get_dt(env))
        save_tracking_plot(t_c, vt_c, sp_c, f"PPO + Conformal STL (sp={sp})", os.path.join(OUTDIR, f"vt_{tag}_conformal.png"))

        summary[tag] = {
            "baseline":  {"sat": sat_b, "rho": rho_b},
            "shield":    {"sat": sat_s, "rho": rho_s},
            "conformal": {"sat": sat_c, "rho": rho_c},
        }

    # bar chart (nominal)
    labels = ["Baseline", "STL shield", "Conformal"]
    sat_nom = [summary["nominal_500"][k]["sat"] for k in ["baseline","shield","conformal"]]
    rho_nom = [summary["nominal_500"][k]["rho"] for k in ["baseline","shield","conformal"]]
    plt.figure(figsize=(5,3))
    plt.bar(labels, [100*float(x) for x in sat_nom])
    plt.ylabel("STL satisfaction [%]"); plt.ylim(0,110)
    plt.title("Nominal (sp=500)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "bar_nominal_stl.png"), dpi=150)
    plt.close()

    # small text report
    report = {
        "scenarios": summary
    }
    with open(os.path.join(OUTDIR, "summary.json"), "w") as f:
        json.dump(report, f, indent=2)

    print("\nSaved figures to:", os.path.abspath(OUTDIR))
    for tag in scenarios:
        name = tag[0]
        print(" -", f"vt_{name}_baseline.png, vt_{name}_shield.png, vt_{name}_conformal.png")
    print(" - bar_nominal_stl.png")
    print(" - summary.json")

if __name__ == "__main__":
    main()
