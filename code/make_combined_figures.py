import os, json, csv
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from f16_engine_env import F16EngineEnv
from stl_monitor import settling_spec_last_window
from shield import SimpleThrottleShield
from conformal_shield import OneStepVTLinear, split_conformal_q, ConformalSTLShield, collect_calibration

OUTDIR = "figs_combined"
MODEL_PATH = "shield_pack/ppo_f16_engine_baseline.zip"
SEED = 42

# ---------- helpers ----------
def _base_env(env):
    e = env
    while hasattr(e, "env"): e = e.env
    return e

def _get_dt(env):
    if hasattr(env,"dt"): return env.dt
    if hasattr(env,"env") and hasattr(env.env,"dt"): return env.env.dt
    return env.unwrapped.dt

def set_setpoint(env, sp):
    _base_env(env).sp = float(sp)

def rollout(env, model, action_filter=None):
    """
    action_filter(u_rl, base_env) -> u_filtered (or None for baseline)
    Returns (t, y, sp)
    """
    obs, info = env.reset(seed=SEED)
    t_hist, vt_hist = [], []
    base = _base_env(env)
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        u = float(action[0])
        if action_filter is not None:
            u = action_filter(u, base)
        obs, r, done, trunc, info = env.step(np.array([u], dtype=np.float32))
        t_hist.append(info["t"])
        vt_hist.append(info["Vt"])
        if trunc: break
    return np.array(t_hist), np.array(vt_hist), base.sp

def build_conformal(env, model, delta=0.30, K=6, slew=0.03, calib_eps=6):
    base = _base_env(env)
    Vt, pw, thr = collect_calibration(base, policy=model,
                                      episodes=calib_eps, random_throttle=False, seed=123)
    pred = OneStepVTLinear(); pred.fit(Vt, pw, thr)
    pred_next = [pred.predict_next(vt0, pw0, u0) for vt0, pw0, u0 in zip(Vt[:-1], pw[:-1], thr[:-1])]
    q = split_conformal_q(np.asarray(pred_next) - Vt[1:], delta=delta)
    sh = ConformalSTLShield(pred, q=q, K=K, dt=_get_dt(env), tol=0.05, slew=slew)
    sh.reset(u0=0.5)
    return sh

# ---------- control metrics ----------
def compute_step_metrics(t, y, sp, settle_band=0.05, steady_window=5.0, y0=None):
    """
    y: response (Vt), sp: setpoint, t: seconds increasing
    settle_band: fractional band around sp (e.g., 0.05 = 5%)
    steady_window: seconds at end used for steady-state error
    y0: initial value (if None, use y[0])
    Returns dict with: overshoot_pct, t_settle, ess, ess_abs, stl_sat, stl_rho
    """
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=float)
    dt = t[1]-t[0] if len(t) > 1 else 0.1

    if y0 is None: y0 = y[0]
    step_mag = abs(sp - y0) if abs(sp - y0) > 1e-6 else max(1.0, abs(sp)*0.01)

    # overshoot/undershoot (works for up or down steps)
    sign = np.sign(sp - y0)  # +1 up-step, -1 down-step
    # error sign so that positive means "beyond" the setpoint
    beyond = sign * (y - sp)
    peak_beyond = np.max(beyond)
    overshoot_pct = max(0.0, 100.0 * peak_beyond / step_mag)

    # settling time: first time inside band and stays inside thereafter
    band = settle_band * abs(sp)
    inside = np.abs(y - sp) <= band
    t_settle = np.inf
    if np.any(inside):
        idx = np.where(inside)[0]
        for i in idx:
            # if from this i to the end we always stay inside
            if np.all(inside[i:]):
                t_settle = t[i]
                break

    # steady-state error: average of last steady_window seconds
    N_tail = max(1, int(round(steady_window / max(dt,1e-6))))
    ess = float(np.mean(y[-N_tail:]) - sp)
    ess_abs = abs(ess)

    # STL (same spec used elsewhere): last 10 s within 5% band
    # We can reuse settling_spec_last_window logic: on the last 10s (or available)
    # Here we just compute ρ on the whole trace's last 10 s using the band.
    from stl_monitor import settling_spec_last_window
    stl_sat, stl_rho = settling_spec_last_window(y, sp=sp, dt=dt, window_s=10.0, tol=settle_band)

    return {
        "overshoot_pct": float(overshoot_pct),
        "t_settle_s": float(t_settle if np.isfinite(t_settle) else -1.0),
        "ess": float(ess),
        "ess_abs": float(ess_abs),
        "stl_sat": bool(stl_sat),
        "stl_rho": float(stl_rho)
    }

def add_metrics_box(ax, metrics_by_method, loc="upper right"):
    """Add a small text box with metrics for each method."""
    lines = ["Method | OS% | t_settle [s] | |e_ss|"]
    for name, m in metrics_by_method.items():
        lines.append(f"{name}: {m['overshoot_pct']:.1f} | {m['t_settle_s']:.2f} | {m['ess_abs']:.3f}")
    text = "\n".join(lines)
    ax.text(0.98, 0.02, text, transform=ax.transAxes,
            va="bottom", ha="right", fontsize=9,
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="0.5"))

# ---------- main ----------
def main():
    os.makedirs(OUTDIR, exist_ok=True)

    # env + model
    env = Monitor(F16EngineEnv(sp=500.0, dt=0.1, ep_len_s=60.0, seed=SEED))
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model not found: " + MODEL_PATH)
    model = PPO.load(MODEL_PATH, env=env)

    scenarios = [("nominal_500", 500.0), ("shift_450", 450.0), ("shift_550", 550.0)]

    all_metrics = {}

    for tag, sp in scenarios:
        set_setpoint(env, sp)

        # Baseline
        t_b, vt_b, sp_b = rollout(env, model)
        m_b = compute_step_metrics(t_b, vt_b, sp_b)

        # STL shield
        sh_simple = SimpleThrottleShield(slew=0.05); sh_simple.reset(u0=0.5)
        t_s, vt_s, sp_s = rollout(env, model, action_filter=lambda u_rl, base: sh_simple.filter(base.sim.Vt, base.sp, u_rl))
        m_s = compute_step_metrics(t_s, vt_s, sp_s)

        # Conformal shield
        sh_conf = build_conformal(env, model, delta=0.30, K=6, slew=0.03, calib_eps=6)
        t_c, vt_c, sp_c = rollout(env, model, action_filter=lambda u_rl, base: sh_conf.filter(base.sim.Vt, base.sim.pow, base.sp, u_rl)[0])
        m_c = compute_step_metrics(t_c, vt_c, sp_c)

        # Combined plot with legend + metrics box
        fig, ax = plt.subplots(figsize=(7.5, 4))
        ax.plot(t_b, vt_b, label="PPO (baseline)")
        ax.plot(t_s, vt_s, label="PPO + STL shield")
        ax.plot(t_c, vt_c, label="PPO + Conformal shield")
        ax.axhline(sp, color="k", ls="--", lw=1, label="Setpoint")
        ax.set_xlabel("Time [s]"); ax.set_ylabel("Airspeed Vt [ft/s]")
        ax.set_title(f"Airspeed tracking comparison (sp={int(sp)})")
        ax.legend()
        add_metrics_box(ax, {
            "PPO": m_b,
            "STL": m_s,
            "CONF": m_c
        })
        fig.tight_layout()
        fig.savefig(os.path.join(OUTDIR, f"compare_{tag}_with_metrics.png"), dpi=170)
        plt.close(fig)

        # Store metrics
        all_metrics[tag] = {"baseline": m_b, "shield": m_s, "conformal": m_c}

    # Write JSON + CSV and a pretty printout
    json_path = os.path.join(OUTDIR, "metrics.json")
    with open(json_path, "w") as f:
        json.dump(all_metrics, f, indent=2)

    csv_path = os.path.join(OUTDIR, "metrics.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scenario", "method", "overshoot_pct", "t_settle_s", "ess", "ess_abs", "stl_sat", "stl_rho"])
        for scen, mm in all_metrics.items():
            for name, m in mm.items():
                w.writerow([scen, name, m["overshoot_pct"], m["t_settle_s"], m["ess"], m["ess_abs"], m["stl_sat"], m["stl_rho"]])

    # Console table
    print("\n=== Control Metrics (per scenario) ===")
    for scen, mm in all_metrics.items():
        print(f"\n[{scen}]")
        print(f"{'Method':12s} {'OS[%]':>8s} {'t_settle[s]':>12s} {'|e_ss|':>10s} {'STL ρ':>10s}")
        for name_key, disp in [("baseline","PPO"), ("shield","STL"), ("conformal","CONF")]:
            m = mm[name_key]
            print(f"{disp:12s} {m['overshoot_pct']:8.2f} {m['t_settle_s']:12.2f} {m['ess_abs']:10.3f} {m['stl_rho']:10.3f}")

    print("\nSaved to:", os.path.abspath(OUTDIR))
    print("  - compare_nominal_500_with_metrics.png")
    print("  - compare_shift_450_with_metrics.png")
    print("  - compare_shift_550_with_metrics.png")
    print("  - metrics.json, metrics.csv")

if __name__ == "__main__":
    main()
