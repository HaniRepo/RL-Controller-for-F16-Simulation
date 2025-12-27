# f16_engine_env.py
import gymnasium as gym
import numpy as np

from aerobench.lowlevel.engine_plant_v2 import EnginePlantV2, EnginePlantConfig

class F16EngineEnv(gym.Env):
    """
    RL env for the F-16 Engine-Control case (2 variables: Vt, pow; action: throttle in [0,1]).
    Uses EnginePlantV2 (with engine_only=True) as the plant.
    Observation: [Vt/sp, pow]
    Action: [throttle] in [0,1]
    Episode length: 60s by default
    """
    metadata = {"render_modes": []}

    def __init__(self, sp=500.0, dt=0.1, ep_len_s=60.0, seed=None):
        super().__init__()
        self.dt = float(dt)
        self.T = float(ep_len_s)
        self.sp = float(sp)
        self.rng = np.random.default_rng(seed)

        # --- Plant (engine-only integration, throttle in [0,1]) ---
        cfg = EnginePlantConfig(
            #model_name="stevens",   # to check odds
            model_name="morelli",
            throttle_scale=1.0,   # repo expects throttle as fraction in [0,1]
            engine_only=True      # integrate only VT and pow to mimic 2-state case
        )
        self.sim = EnginePlantV2(cfg)

        # --- Define Gym spaces (MUST be set in __init__) ---
        # Observation: [Vt/sp, pow]; give loose but finite bounds
        obs_high = np.array([5.0, 2.0], dtype=np.float32)  # was [5.0, 100.0]
        self.observation_space = gym.spaces.Box(-obs_high, obs_high, dtype=np.float32)

        # Action: throttle in [0,1]
        self.action_space = gym.spaces.Box(
            low=np.array([0.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32
        )

        # internal
        self._t = 0.0
        self._a_prev = np.array([0.5], dtype=np.float32)

    # ---------- helpers ----------
    def _obs(self):
        Vt = self.sim.Vt
        pow_ = self.sim.pow / 100.0  # <-- normalize if pow is ~0..100
        return np.array([Vt / max(1e-6, self.sp), pow_], dtype=np.float32)

    def _reward(self, obs, action):
        Vt_actual = self.sim.Vt
        err_rel = abs((Vt_actual - self.sp) / max(1e-6, self.sp))
        u = float(action[0])
        du = float(u - self._a_prev[0])

        # progress bonus: reward reductions in error
        prog = getattr(self, "_prev_err", None)
        r_prog = (prog - err_rel) if prog is not None else 0.0
        self._prev_err = err_rel

        # base terms
        w_e, w_s, w_sat, w_prog = 1.0, 0.005, 0.005, 0.1
        r = - w_e * err_rel - w_s * (du * du) - (w_sat if (u <= 1e-6 or u >= 1.0 - 1e-6) else 0.0)
        r += w_prog * r_prog  # positive if error decreased

        # small bonus for being inside the 5% band
        if err_rel <= 0.05:
            r += 0.05
        return r

    # ---------- Gym API ----------
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._t = 0.0
        self._a_prev = np.array([0.5], dtype=np.float32)
        self._prev_err = None


        # randomize initial speed around setpoint
        Vt0 = float(self.rng.uniform(0.9 * self.sp, 1.1 * self.sp))
        self.sim.reset(Vt0=Vt0, alt0_ft=550.0, pow0=10.0)

        return self._obs(), {"sp": self.sp, "t": self._t}

    def step(self, action):
        # clip to action space and advance plant
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        u = float(action[0])

        self.sim.step_engine(throttle=u, dt=self.dt)

        self._t += self.dt
        obs = self._obs()
        reward = self._reward(obs, action)
        terminated = bool(self._t >= self.T)
        truncated = False
        info = {"t": self._t, "sp": self.sp, "throttle": u, "Vt": self.sim.Vt}

        self._a_prev = action.copy()
        return obs, reward, terminated, truncated, info

    def render(self):
        pass
