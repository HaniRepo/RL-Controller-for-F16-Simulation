# reproduce_baseline.py
import os, json, numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from f16_engine_env import F16EngineEnv
from stl_monitor import settling_spec_last_window
from utils_plot import plot_vt

PACK_DIR = "baseline_pack"
MODEL_PATH = os.path.join(PACK_DIR, "ppo_f16_engine_baseline.zip")

def main():
    with open(os.path.join(PACK_DIR, "config.json")) as f:
        cfg = json.load(f)

    env = F16EngineEnv(sp=cfg["setpoint"], dt=cfg["dt"], ep_len_s=cfg["horizon_s"], seed=cfg.get("seed", 42))
    env = Monitor(env)
    model = PPO.load(MODEL_PATH, env=env)

    # canonical rollout
    obs, info = env.reset()
    vt_hist, t_hist = [], []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, r, done, trunc, info = env.step(action)
        vt_hist.append(info["Vt"]); t_hist.append(info["t"])
        if trunc: break

    t = np.array(t_hist); vt = np.array(vt_hist); sp = cfg["setpoint"]
    sat, rho = settling_spec_last_window(vt, sp=sp, dt=cfg["dt"], window_s=10.0, tol=0.05)
    print(f"Reproduced run → STL(last 10s): {sat} (rho={rho:.4f})")

    fig = plot_vt(t, vt, sp, title="PPO Airspeed Tracking (Reproduced)")
    fig.savefig(os.path.join(PACK_DIR, "vt_tracking_reproduced.png"), dpi=160)
    print("Saved:", os.path.join(PACK_DIR, "vt_tracking_reproduced.png"))

if __name__ == "__main__":
    main()
