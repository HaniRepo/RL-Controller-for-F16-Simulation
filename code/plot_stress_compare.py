import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from f16_engine_env import F16EngineEnv
from stress_wrappers import NoisyDelayedWrapper, ActionRateLimiter, SetpointJumpWrapper, ThrottleCapWrapper

def _base_env(e):
    while hasattr(e, 'env'):
        e = e.env
    return e

def make_stress_env(seed=7):
    env = F16EngineEnv(sp=500.0, dt=0.1, ep_len_s=60.0, seed=seed)
    env = Monitor(env)
    # model mismatch (Morelli)
    base = _base_env(env)
    if hasattr(base, "sim") and hasattr(base.sim, "cfg") and hasattr(base.sim.cfg, "model_name"):
        base.sim.cfg.model_name = "morelli"
    elif hasattr(base, "sim") and hasattr(base.sim, "f16_model"):
        base.sim.f16_model = "morelli"
    # stress stack
    env = ThrottleCapWrapper(env, u_max=0.80)
    env = ActionRateLimiter(env, slew=0.008)
    env = NoisyDelayedWrapper(env, obs_sigma=12.0, act_delay_steps=4)
    env = SetpointJumpWrapper(env, t_jump_s=20.0, sp_new=590.0)
    return env

def rollout(env, policy, use_conformal=False):
    obs, info = env.reset()
    t_hist, vt_hist, sp_hist = [], [], [info["sp"]]
    u_hist = []  # applied throttle each step

    if use_conformal:
        from conformal_shield import OneStepVTLinear, split_conformal_q, ConformalSTLShield, collect_calibration
        base = _base_env(env)
        Vt, pw, thr = collect_calibration(base, policy=policy, episodes=8, random_throttle=False, seed=123)
        pred = OneStepVTLinear(); pred.fit(Vt, pw, thr)
        pred_next = [pred.predict_next(vt0,pw0,u0) for vt0,pw0,u0 in zip(Vt[:-1], pw[:-1], thr[:-1])]
        q = split_conformal_q(np.asarray(pred_next) - Vt[1:], delta=0.30)
        sh = ConformalSTLShield(pred, q=q, K=6, dt=0.1, tol=0.05, slew=0.03)
        sh.reset(u0=0.5)

    done = False
    while not done:
        action, _ = policy.predict(obs, deterministic=True)
        u_applied = float(action[0])

        if use_conformal:
            base = _base_env(env)
            u_applied, _ = sh.filter(base.sim.Vt, base.sim.pow, base.sp, u_applied)

        obs, r, done, trunc, info = env.step(np.array([u_applied], dtype=np.float32))

        t_hist.append(info["t"])
        vt_hist.append(info["Vt"])
        sp_hist.append(info["sp"])
        u_hist.append(u_applied)

        if trunc:
            break

    return np.array(t_hist), np.array(vt_hist), np.array(sp_hist[:-1]), np.array(u_hist)

def main():
    # rollout PPO
    env_ppo = make_stress_env(seed=7)
    model = PPO.load("ppo_f16_engine.zip", env=env_ppo)
    t0, vt0, sp0, u0 = rollout(env_ppo, model, use_conformal=False)

    # rollout PPO + Conformal (fresh env with same stress)
    env_conf = make_stress_env(seed=7)
    t1, vt1, sp1, u1 = rollout(env_conf, model, use_conformal=True)

    # sanity check difference
    common = min(len(vt0), len(vt1))
    max_abs = float(np.max(np.abs(vt0[:common] - vt1[:common])))
    print(f"Max |Vt_PPO - Vt_CONF| over common horizon: {max_abs:.3f} ft/s")

    # --- Figure 1: Vt traces ---
    plt.figure(figsize=(9,3.2))
    plt.plot(t0, vt0, label="PPO", linewidth=2.0)
    plt.plot(t1, vt1, label="PPO + Conformal", linewidth=2.0, linestyle="--")
    plt.plot(t0, sp0, label="Setpoint", linestyle=":", linewidth=2.0)
    plt.xlabel("Time [s]"); plt.ylabel("Airspeed Vt [ft/s]")
    plt.title("Stress case: Morelli + cap + slew + noise/delay + setpoint jump")
    plt.legend(frameon=True); plt.tight_layout()
    out1 = "shield_pack/stress_compare.png"
    plt.savefig(out1, dpi=160)
    print("Saved", out1)

    # --- Figure 2: ΔVt and actions ---
    dt0 = t0[:common]; dvt = vt1[:common] - vt0[:common]
    plt.figure(figsize=(9,4.0))
    ax1 = plt.gca()
    ax1.plot(dt0, dvt, label="ΔVt = Vt(CONF) - Vt(PPO)")
    ax1.axhline(0, linestyle=":", linewidth=1.5)
    ax1.set_xlabel("Time [s]"); ax1.set_ylabel("ΔVt [ft/s]")
    ax1.set_title("Conformal effect (top: ΔVt)\n(bottom: applied throttle)")
    # actions on a second axis area
    from matplotlib.transforms import blended_transform_factory
    trans = blended_transform_factory(ax1.transData, ax1.transAxes)
    # draw a small inset-like band for actions
    y0, y1 = -0.38, -0.02
    ax1.fill_between(dt0, y0, y1, color="0.95", transform=trans, zorder=-1)
    # resample u to same length and plot (normalize to [y0,y1] for display only)
    uu0 = u0[:common]; uu1 = u1[:common]
    def norm_u(u):
        u = np.clip(u, 0, 1)
        return y0 + (y1 - y0) * (u - 0) / (1 - 0)
    ax1.plot(dt0, norm_u(uu0), label="u PPO", linewidth=1.8)
    ax1.plot(dt0, norm_u(uu1), label="u Conformal", linewidth=1.8, linestyle="--")
    ax1.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    out2 = "shield_pack/stress_compare_diff.png"
    plt.savefig(out2, dpi=160)
    print("Saved", out2)

if __name__ == "__main__":
    main()
