import os, json, math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from f16_engine_env import F16EngineEnv
from stl_monitor import settling_spec_last_window
from stress_wrappers import NoisyDelayedWrapper, ActionRateLimiter, SetpointJumpWrapper, ThrottleCapWrapper

# ---- CONFIG ----------------------------------------------------------
OUT_DIR = "fig_bundle"
MODEL_FILE = "ppo_f16_engine.zip"

# stress stack used in the paper
STRESS = dict(
    cap=0.80,              # throttle cap
    slew=0.008,            # action slew limit
    obs_sigma=12.0,        # observation noise (ft/s scale on normalized Vt/sp channel)
    delay_steps=4,         # action delay
    t_jump=20.0,           # setpoint jump time (s)
    sp_new=590.0,          # new setpoint at t_jump
)

# nominal setup of env
NOMINAL = dict(sp=500.0, dt=0.1, horizon=60.0, seed=7)

# STL specs to report
STL_NOMINAL = dict(tol=0.05, window_s=10.0)  # 5% band, last 10 s
STL_STRESS  = dict(tol=0.02, window_s=5.0)   # 2% band, last 5 s

# Conformal params
CONF = dict(delta=0.30, K=6, tol=0.05, slew=0.03)  # tol here is inner shield check
# ---------------------------------------------------------------------


# ---------------- utilities ----------------
def _base_env(e):
    while hasattr(e, "env"):
        e = e.env
    return e

def make_stress_env(sp=NOMINAL["sp"], dt=NOMINAL["dt"], horizon=NOMINAL["horizon"], seed=NOMINAL["seed"]):
    """Build the stress-test environment used for figures."""
    env = F16EngineEnv(sp=sp, dt=dt, ep_len_s=horizon, seed=seed)
    env = Monitor(env)

    # flip aerodynamic model to "morelli"
    base = _base_env(env)
    if hasattr(base, "sim") and hasattr(base.sim, "cfg") and hasattr(base.sim.cfg, "model_name"):
        base.sim.cfg.model_name = "morelli"
    elif hasattr(base, "sim") and hasattr(base.sim, "f16_model"):
        base.sim.f16_model = "morelli"

    # stack stressors
    env = ThrottleCapWrapper(env, u_max=STRESS["cap"])
    env = ActionRateLimiter(env, slew=STRESS["slew"])
    env = NoisyDelayedWrapper(env, obs_sigma=STRESS["obs_sigma"], act_delay_steps=STRESS["delay_steps"])
    env = SetpointJumpWrapper(env, t_jump_s=STRESS["t_jump"], sp_new=STRESS["sp_new"])
    return env

def rollout(env, policy, mode="ppo"):
    """
    mode:
      "ppo"         -> raw PPO policy actions
      "stl"         -> PPO + SimpleThrottleShield
      "conformal"   -> PPO + Conformal STL shield
    returns: t, vt, sp, u
    """
    obs, info = env.reset()
    t_hist, vt_hist, sp_hist, u_hist = [], [], [info["sp"]], []

    sh = None
    if mode == "stl":
        from shield import SimpleThrottleShield
        sh = SimpleThrottleShield(slew=0.05)
        sh.reset(u0=0.5)
    elif mode == "conformal":
        from conformal_shield import OneStepVTLinear, split_conformal_q, ConformalSTLShield, collect_calibration
        base = _base_env(env)
        # Calibrate on policy rollouts under current stress env
        Vt, pw, thr = collect_calibration(base, policy=policy, episodes=8, random_throttle=False, seed=123)
        pred = OneStepVTLinear(); pred.fit(Vt, pw, thr)
        pred_next = [pred.predict_next(vt0, pw0, u0) for vt0,pw0,u0 in zip(Vt[:-1], pw[:-1], thr[:-1])]
        q = split_conformal_q(np.asarray(pred_next) - Vt[1:], delta=CONF["delta"])
        sh = ConformalSTLShield(pred, q=q, K=CONF["K"], dt=NOMINAL["dt"], tol=CONF["tol"], slew=CONF["slew"])
        sh.reset(u0=0.5)

    done = False
    while not done:
        action, _ = policy.predict(obs, deterministic=True)
        u_applied = float(action[0])

        if mode == "stl":
            base = _base_env(env)
            u_applied = sh.filter(base.sim.Vt, base.sp, u_applied)

        if mode == "conformal":
            base = _base_env(env)
            u_applied, _ = sh.filter(base.sim.Vt, base.sim.pow, base.sp, u_applied)

        obs, r, done, trunc, info = env.step(np.array([u_applied], dtype=np.float32))
        t_hist.append(info["t"]); vt_hist.append(info["Vt"]); sp_hist.append(info["sp"]); u_hist.append(u_applied)
        if trunc: break

    return np.array(t_hist), np.array(vt_hist), np.array(sp_hist[:-1]), np.array(u_hist)

def classical_metrics(t, y, sp, tol=0.05, window_s=10.0):
    """
    Control-style metrics:
      overshoot % (relative to sp),
      peak time (s),
      settling time (enter band ±tol*sp and stay for the last 'window_s'),
      steady-state error (y_end - sp).
    """
    y = np.asarray(y); sp = float(sp if np.isscalar(sp) else sp[-1])
    # Overshoot and peak time
    idx_peak = int(np.argmax(y))
    y_peak = float(y[idx_peak]); t_peak = float(t[idx_peak])
    overshoot = max(0.0, (y_peak - sp) / max(sp, 1e-6)) * 100.0

    # Settling: last window within ± tol
    band = tol * abs(sp)
    settled = np.all(np.abs(y[t >= (t[-1] - window_s)] - sp) <= band)
    if settled:
        # walk backward to find first time from which the signal stays in band
        st = t[-1]
        for i in range(len(y)-1, -1, -1):
            if np.abs(y[i] - sp) > band:
                break
            st = t[i]
        settling_time = float(st)
    else:
        settling_time = math.nan

    ss_err = float(y[-1] - sp)
    return dict(overshoot_pct=overshoot, peak_time_s=t_peak, settling_time_s=settling_time, steady_state_error=ss_err)

def stl_check(vt, sp, dt, tol, window_s):
    sat, rho = settling_spec_last_window(np.asarray(vt), sp=float(sp[-1]) if not np.isscalar(sp) else float(sp),
                                         dt=dt, window_s=window_s, tol=tol)
    return bool(sat), float(rho)

def ensure_outdir():
    os.makedirs(OUT_DIR, exist_ok=True)

# ---------------- plotting helpers ----------------
def add_conformance_band(ax, sp, tol):
    sp = float(sp if np.isscalar(sp) else sp[-1])
    lo, hi = sp*(1.0 - tol), sp*(1.0 + tol)
    ax.axhspan(lo, hi, alpha=0.10, color="tab:green", label=f"±{int(tol*100)}% band")

def save_traces_with_zoom(t_all, vt_all, sp, labels, colors, linestyles,
                          title, filename, tol_band=0.05,
                          zoom_xlim=(40, 60), zoom_ylim=None):
    plt.figure(figsize=(10.5, 3.6))
    ax = plt.gca()
    add_conformance_band(ax, sp[-1], tol_band)

    # plot main traces
    for t, vt, lab, c, ls in zip(t_all, vt_all, labels, colors, linestyles):
        ax.plot(t, vt, lw=2.2, label=lab, color=c, linestyle=ls)

    ax.plot(t_all[0], sp, "k:", lw=2.0, label="Setpoint")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Airspeed Vt [ft/s]")
    ax.set_title(title)
    ax.legend(loc="best", frameon=True)

    # inset zoom
    axins = inset_axes(ax, width="38%", height="45%", loc="upper left")
    for t, vt, c, ls in zip(t_all, vt_all, colors, linestyles):
        axins.plot(t, vt, lw=2.0, color=c, linestyle=ls)
    axins.plot(t_all[0], sp, "k:", lw=1.8)
    axins.set_xlim(*zoom_xlim)
    if zoom_ylim is not None:
        axins.set_ylim(*zoom_ylim)
    axins.grid(alpha=0.2)

    plt.tight_layout()
    fn = os.path.join(OUT_DIR, filename)
    plt.savefig(fn, dpi=300)
    plt.close()
    print("Saved", fn)

# --- NEW: interval-based metrics (useful for setpoint jumps) ---
def compute_step_metrics_on_interval(t, y, sp, t0, settle_band=0.05, steady_window=5.0):
    """
    Compute classic step metrics only using samples with t >= t0.
    - overshoot_pct: relative to |sp - y(t0)| (handles up/down steps)
    - t_settle_s: first time >= t0 when |y-sp| <= settle_band*|sp| AND stays inside to the end; -1 if never.
    - ess, ess_abs: mean error over the last 'steady_window' seconds of the interval
    - stl_sat, stl_rho: same last-10s STL on the interval subset
    """
    t = np.asarray(t); y = np.asarray(y, dtype=float)
    idx = t >= t0
    t_loc, y_loc = t[idx], y[idx]
    if len(t_loc) < 2:
        return {"overshoot_pct": 0.0, "t_settle_s": -1.0, "ess": 0.0, "ess_abs": 0.0, "stl_sat": False, "stl_rho": -1.0}

    dt = t_loc[1] - t_loc[0]
    y0 = y_loc[0]
    step_mag = max(1.0, abs(sp - y0))
    sign = np.sign(sp - y0)
    beyond = sign * (y_loc - sp)
    overshoot_pct = max(0.0, 100.0 * np.max(beyond) / step_mag)

    band = settle_band * abs(sp)
    inside = np.abs(y_loc - sp) <= band
    t_settle = -1.0
    if np.any(inside):
        idx_inside = np.where(inside)[0]
        for k in idx_inside:
            if np.all(inside[k:]):
                t_settle = float(t_loc[k])  # absolute time stamp
                break

    # steady-state error over last steady_window seconds (interval tail)
    N_tail = max(1, int(round(steady_window / max(dt, 1e-6))))
    ess = float(np.mean(y_loc[-N_tail:]) - sp)
    ess_abs = abs(ess)

    # STL on last 10s of the interval
    from stl_monitor import settling_spec_last_window
    stl_sat, stl_rho = settling_spec_last_window(y_loc, sp=sp, dt=dt, window_s=10.0, tol=settle_band)

    return {
        "overshoot_pct": float(overshoot_pct),
        "t_settle_s": t_settle,  # absolute time (not offset)
        "ess": float(ess),
        "ess_abs": float(ess_abs),
        "stl_sat": bool(stl_sat),
        "stl_rho": float(stl_rho),
    }

# --- NEW: annotate a step on the figure ---
def annotate_step_metrics(ax, t, y, sp, metrics, t0, band_frac=0.05):
    band = band_frac * abs(sp)
    # setpoint line + band (local)
    ax.axhline(sp, color="k", ls=":", lw=1.5)
    ax.fill_between([t0, t[-1]], sp-band, sp+band, color="green", alpha=0.08, zorder=0)

    # mark step time
    ax.axvline(t0, color="k", ls="--", lw=1.0)

    # mark settling time if finite
    if metrics["t_settle_s"] > 0:
        ax.axvline(metrics["t_settle_s"], color="tab:gray", ls="--", lw=1.0)
        ax.text(metrics["t_settle_s"], sp + 1.5*band, "settled", ha="left", va="bottom", color="tab:gray")

    # tiny box with numbers
    ax.text(0.02, 0.02,
            f"OS={metrics['overshoot_pct']:.1f}% | t_settle={metrics['t_settle_s'] if metrics['t_settle_s']>0 else '–'} | |e_ss|={metrics['ess_abs']:.3f}",
            transform=ax.transAxes, ha="left", va="bottom",
            fontsize=9, bbox=dict(facecolor="white", edgecolor="0.5", alpha=0.8))

def save_diff_and_actions(t, vt_baseline, vt_stl, vt_conf, u_base, u_stl, u_conf, filename):
    # time-align for plotting differences
    L = min(map(len, [vt_baseline, vt_stl, vt_conf, u_base, u_stl, u_conf, t]))
    t = t[:L]; vt_b = vt_baseline[:L]; vt_s = vt_stl[:L]; vt_c = vt_conf[:L]
    u_b = u_base[:L]; u_s = u_stl[:L]; u_c = u_conf[:L]

    plt.figure(figsize=(10.5,4.0))
    ax = plt.gca()
    ax.plot(t, vt_s - vt_b, "r-.", lw=2.0, label="ΔVt (STL − PPO)")
    ax.plot(t, vt_c - vt_b, "orange", ls="--", lw=2.0, label="ΔVt (CONF − PPO)")
    ax.axhline(0, color="k", lw=1)
    ax.set_xlabel("Time [s]"); ax.set_ylabel("ΔVt [ft/s]")
    ax.set_title("Effect of shields (top: ΔVt)\n(bottom: applied throttle)")
    ax.legend(loc="upper right", frameon=True)

    # small band for actions
    from matplotlib.transforms import blended_transform_factory
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    y0, y1 = -0.40, -0.04
    ax.fill_between(t, y0, y1, color="0.95", transform=trans, zorder=-1)
    def norm_u(u):
        u = np.clip(u, 0, 1)
        return y0 + (y1 - y0) * u
    ax.plot(t, norm_u(u_b), "b-", lw=1.8, label="u PPO")
    ax.plot(t, norm_u(u_s), "r-.", lw=1.8, label="u STL")
    ax.plot(t, norm_u(u_c), "orange", ls="--", lw=1.8, label="u CONF")

    plt.tight_layout()
    fn = os.path.join(OUT_DIR, filename)
    plt.savefig(fn, dpi=300)
    plt.close()
    print("Saved", fn)

def save_metrics_table_image(metrics, filename, title="Summary metrics (stress case)"):
    # simple rendered text table for quick paper drop-in if needed
    lines = [title, ""]
    for k, v in metrics.items():
        lines.append(f"{k}: {v}")
    text = "\n".join(lines)

    plt.figure(figsize=(7,3))
    plt.axis("off")
    plt.text(0.02, 0.98, text, va="top", ha="left", family="monospace")
    fn = os.path.join(OUT_DIR, filename)
    plt.savefig(fn, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved", fn)


# ---------------- main pipeline ----------------
def main():
    ensure_outdir()

    # build stress env and load policy
    env = make_stress_env()
    model = PPO.load(MODEL_FILE, env=env)

    # three runs
    t0, vt0, sp0, u0 = rollout(env, model, mode="ppo")
    env = make_stress_env()
    t1, vt1, sp1, u1 = rollout(env, model, mode="stl")
    env = make_stress_env()
    t2, vt2, sp2, u2 = rollout(env, model, mode="conformal")

    # quick delta check
    Lc = min(len(vt0), len(vt2))
    print(f"Max |Vt_PPO - Vt_CONF| over common horizon: {np.max(np.abs(vt0[:Lc]-vt2[:Lc])):.3f} ft/s")

    # STL checks
    sat_nom_ppo,  rho_nom_ppo  = stl_check(vt0, sp0, NOMINAL["dt"], **STL_NOMINAL)
    sat_nom_stl,  rho_nom_stl  = stl_check(vt1, sp1, NOMINAL["dt"], **STL_NOMINAL)
    sat_nom_conf, rho_nom_conf = stl_check(vt2, sp2, NOMINAL["dt"], **STL_NOMINAL)

    sat_str_ppo,  rho_str_ppo  = stl_check(vt0, sp0, NOMINAL["dt"], **STL_STRESS)
    sat_str_stl,  rho_str_stl  = stl_check(vt1, sp1, NOMINAL["dt"], **STL_STRESS)
    sat_str_conf, rho_str_conf = stl_check(vt2, sp2, NOMINAL["dt"], **STL_STRESS)

    # classical control metrics on stress case (w.r.t. final setpoint)
    met_ppo  = classical_metrics(t0, vt0, sp0[-1], tol=STL_STRESS["tol"], window_s=STL_STRESS["window_s"])
    met_stl  = classical_metrics(t1, vt1, sp1[-1], tol=STL_STRESS["tol"], window_s=STL_STRESS["window_s"])
    met_conf = classical_metrics(t2, vt2, sp2[-1], tol=STL_STRESS["tol"], window_s=STL_STRESS["window_s"])

    # save CSVs
    def save_csv(name, t, vt, sp, u):
        import csv
        fn = os.path.join(OUT_DIR, f"{name}.csv")
        with open(fn, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t_sec", "Vt_ftps", "sp_ftps", "throttle"])
            for ti, yi, si, ui in zip(t, vt, sp, u):
                w.writerow([float(ti), float(yi), float(si), float(ui)])
        print("Saved", fn)
    save_csv("rollout_ppo", t0, vt0, sp0, u0)
    save_csv("rollout_stl", t1, vt1, sp1, u1)
    save_csv("rollout_conformal", t2, vt2, sp2, u2)

    # --- Figures ---
    # 1) all traces with inset + band (stress band for visibility = 2%)
    save_traces_with_zoom(
        t_all=[t0, t1, t2],
        vt_all=[vt0, vt1, vt2],
        sp=sp0,
        labels=["PPO", "PPO + STL shield", "PPO + Conformal"],
        colors=["tab:blue", "tab:red", "tab:orange"],  # <— added
        linestyles=["-", "-.", "--"],                  # <— added
        title="Airspeed under stress (Morelli + cap + slew + noise/delay + setpoint jump)",
        filename="01_traces_with_zoom.png",
        tol_band=STL_STRESS["tol"],
        zoom_xlim=(35, 60),
        zoom_ylim=None
    )

    # 2) difference + actions
    save_diff_and_actions(t0, vt0, vt1, vt2, u0, u1, u2, filename="02_diff_and_actions.png")

    # 2b) LOCAL step metrics after the setpoint jump (clear settling/overshoot view)
    t_jump = STRESS["t_jump"]
    sp_new = STRESS["sp_new"]

    m_b_loc = compute_step_metrics_on_interval(
        t0, vt0, sp_new, t0=t_jump,
        settle_band=STL_STRESS["tol"], steady_window=STL_STRESS["window_s"]
    )
    m_s_loc = compute_step_metrics_on_interval(
        t1, vt1, sp_new, t0=t_jump,
        settle_band=STL_STRESS["tol"], steady_window=STL_STRESS["window_s"]
    )
    m_c_loc = compute_step_metrics_on_interval(
        t2, vt2, sp_new, t0=t_jump,
        settle_band=STL_STRESS["tol"], steady_window=STL_STRESS["window_s"]
    )

    fig, ax = plt.subplots(figsize=(11, 3.6))
    # full traces so readers see the pre-jump context
    ax.plot(t0, vt0, label="PPO", color="tab:blue", lw=2.0)
    ax.plot(t1, vt1, label="PPO + STL", color="tab:orange", ls="--", lw=2.0)
    ax.plot(t2, vt2, label="PPO + Conformal", color="tab:red", ls="-.", lw=2.0)
    ax.set_xlabel("Time [s]"); ax.set_ylabel("Airspeed Vt [ft/s]")
    ax.set_title("Local step metrics after setpoint jump")

    # annotate for conformal (shows band, jump time, settling time if any)
    annotate_step_metrics(ax, t2, vt2, sp_new, m_c_loc, t0=t_jump, band_frac=STL_STRESS["tol"])

    # small summary box with all three methods
    ax.text(
        0.02, 0.02,
        "after jump (local):\n"
        f"PPO:   OS={m_b_loc['overshoot_pct']:.1f}%  "
        f"t_settle={m_b_loc['t_settle_s'] if m_b_loc['t_settle_s']>0 else '–'}  "
        f"|e_ss|={m_b_loc['ess_abs']:.3f}\n"
        f"STL:   OS={m_s_loc['overshoot_pct']:.1f}%  "
        f"t_settle={m_s_loc['t_settle_s'] if m_s_loc['t_settle_s']>0 else '–'}  "
        f"|e_ss|={m_s_loc['ess_abs']:.3f}\n"
        f"CONF:  OS={m_c_loc['overshoot_pct']:.1f}%  "
        f"t_settle={m_c_loc['t_settle_s'] if m_c_loc['t_settle_s']>0 else '–'}  "
        f"|e_ss|={m_c_loc['ess_abs']:.3f}",
        transform=ax.transAxes, ha="left", va="bottom", fontsize=9,
        bbox=dict(facecolor="white", edgecolor="0.5", alpha=0.85)
    )

    ax.legend(loc="upper left")
    plt.tight_layout()
    fn = os.path.join(OUT_DIR, "02b_stress_local_step_metrics.png")
    plt.savefig(fn, dpi=300); plt.close(); print("Saved", fn)

    # 3) each method with shaded *nominal* STL band (5%) for a clean visual
    for tag, (t, vt, sp) in {"ppo":(t0,vt0,sp0), "stl":(t1,vt1,sp1), "conf":(t2,vt2,sp2)}.items():
        plt.figure(figsize=(8.5,3.2))
        add_conformance_band(plt.gca(), sp[-1], STL_NOMINAL["tol"])
        plt.plot(t, vt, lw=2.2, label=tag.upper())
        plt.plot(t, sp, "k:", lw=2.0, label="Setpoint")
        plt.xlabel("Time [s]"); plt.ylabel("Airspeed Vt [ft/s]")
        plt.title(f"{tag.upper()} vs. nominal 5% band (for reference)")
        plt.legend(); plt.tight_layout()
        fn = os.path.join(OUT_DIR, f"03_{tag}_vs_nominal_band.png")
        plt.savefig(fn, dpi=300); plt.close(); print("Saved", fn)

    # 4) Control metrics summary as JSON and a lightweight image
    metrics = {
        "stl_nominal": {
            "ppo": {"sat": sat_nom_ppo, "rho": rho_nom_ppo},
            "stl": {"sat": sat_nom_stl, "rho": rho_nom_stl},
            "conformal": {"sat": sat_nom_conf, "rho": rho_nom_conf},
            "spec": STL_NOMINAL
        },
        "stl_stress": {
            "ppo": {"sat": sat_str_ppo, "rho": rho_str_ppo},
            "stl": {"sat": sat_str_stl, "rho": rho_str_stl},
            "conformal": {"sat": sat_str_conf, "rho": rho_str_conf},
            "spec": STL_STRESS
        },
        "classical_metrics_stress": {
            "ppo": met_ppo,
            "stl": met_stl,
            "conformal": met_conf
        },
        "stress_stack": STRESS
    }
    with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    save_metrics_table_image(metrics["classical_metrics_stress"], "04_metrics_text.png",
                             title="Classical Metrics (stress case, 2% band, 5s window)")

    print("\n✅ Figure bundle written to:", os.path.abspath(OUT_DIR))
    print("   - 01_traces_with_zoom.png")
    print("   - 02_diff_and_actions.png")
    print("   - 03_ppo_vs_nominal_band.png / 03_stl_vs_nominal_band.png / 03_conf_vs_nominal_band.png")
    print("   - 04_metrics_text.png")
    print("   - metrics.json + CSVs for all three rollouts")

if __name__ == "__main__":
    main()
