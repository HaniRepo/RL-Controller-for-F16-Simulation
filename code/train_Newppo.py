import os, argparse, numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from f16_engine_env import F16EngineEnv
from stl_monitor import settling_spec_last_window
from utils_plot import plot_vt
from conformal_shield import OneStepVTLinear, split_conformal_q, ConformalSTLShield, collect_calibration

def eval_conformal(env, model, n_calib_eps=8, delta=0.10, K=20):
    # collect calibration with random throttle (covers more regimes)
    Vt, pw, thr = collect_calibration(env, policy=None, episodes=n_calib_eps, random_throttle=True, seed=123)

    # fit one-step predictor and residuals
    pred = OneStepVTLinear()
    pred.fit(Vt, pw, thr)

    # compute residuals on calibration set
    pred_next = []
    for vt0, pw0, u0 in zip(Vt[:-1], pw[:-1], thr[:-1]):
        pred_next.append(pred.predict_next(vt0, pw0, u0))
    residuals = np.asarray(pred_next) - Vt[1:]
    q = split_conformal_q(residuals, delta=delta)

    # make shield
    sh = ConformalSTLShield(pred, q=q, K=K, dt=env.dt, tol=0.05, slew=0.05)

    # one rollout using conformal shield
    obs, info = env.reset()
    sh.reset(u0=0.5)
    vt_hist, t_hist = [], []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        u_rl = float(action[0])
        u, _rho_worst = sh.filter(env.sim.Vt, env.sim.pow, env.sp, u_rl)
        obs, r, done, trunc, info = env.step(np.array([u], dtype=np.float32))
        vt_hist.append(info["Vt"]); t_hist.append(info["t"])
        if trunc: break
    return np.array(t_hist), np.array(vt_hist), env.sp

def rollout(env, model):
    obs, info = env.reset()
    vt_hist, t_hist = [], []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, r, done, trunc, info = env.step(action)
        vt_hist.append(info["Vt"]); t_hist.append(info["t"])
        if trunc: break
    return np.array(t_hist), np.array(vt_hist), info["sp"]

def main(args):
    env = F16EngineEnv(sp=args.setpoint, dt=args.dt, ep_len_s=args.horizon)
    env = Monitor(env)
    model = PPO("MlpPolicy", env, verbose=1, n_steps=2048, batch_size=64, learning_rate=3e-4)

    model.learn(total_timesteps=args.steps)
    model.save(args.model_path)

    t, vt, sp = rollout(env, model)
    sat, rho = settling_spec_last_window(vt, sp=sp, dt=args.dt, window_s=10.0, tol=0.05)
    print(f"STL(last 10s within 5% band): {sat} (rho={rho:.4f})")

    fig = plot_vt(t, vt, sp, title="PPO Airspeed Tracking")
    fig.savefig("vt_tracking.png", dpi=160)

    print("Evaluating PPO + Conformal STL shield...")
    t_c, vt_c, sp = eval_conformal(env, model, n_calib_eps=8, delta=0.10, K=20)
    sat_c, rho_c = settling_spec_last_window(vt_c, sp=sp, dt=args.dt, window_s=10.0, tol=0.05)
    print(f"PPO + Conformal: STL={sat_c}  rho={rho_c:.3f}")
    fig = plot_vt(t_c, vt_c, sp, title="Airspeed Tracking (PPO + Conformal STL)")
    fig.savefig("vt_tracking_conformal.png", dpi=160)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--setpoint", type=float, default=500.0)
    p.add_argument("--dt", type=float, default=0.1)
    p.add_argument("--horizon", type=float, default=60.0)
    p.add_argument("--steps", type=int, default=300_000)
    p.add_argument("--model-path", type=str, default="./ppo_f16_engine.zip")
    args = p.parse_args()
    main(args)
