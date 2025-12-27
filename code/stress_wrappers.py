# stress_wrappers.py
import numpy as np
import gymnasium as gym
from collections import deque

def _base_env(e):
    while hasattr(e, "env"): e = e.env
    return e

class NoisyDelayedWrapper(gym.Wrapper):
    """Obs noise on Vt/sp and k-step action delay."""
    def __init__(self, env, obs_sigma=2.0, act_delay_steps=2, u_clip=(0.0,1.0)):
        super().__init__(env)
        self.obs_sigma = float(obs_sigma)
        self.act_delay_steps = int(act_delay_steps)
        self.u_min, self.u_max = map(float, u_clip)
        self._fifo = deque([0.5]*max(1,self.act_delay_steps), maxlen=max(1,self.act_delay_steps))

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._fifo = deque([0.5]*max(1,self.act_delay_steps), maxlen=max(1,self.act_delay_steps))
        return self._noisy_obs(obs), info

    def step(self, action):
        self._fifo.append(float(action[0]))
        u = np.array([np.clip(self._fifo[0], self.u_min, self.u_max)], dtype=np.float32)
        obs, r, done, trunc, info = self.env.step(u)
        return self._noisy_obs(obs), r, done, trunc, info

    def _noisy_obs(self, obs):
        o = np.array(obs, dtype=np.float32).copy()
        base = _base_env(self.env)
        sp = getattr(base, "sp", 500.0)
        if o.size > 0 and sp > 1e-6:
            o[0] += np.random.normal(0.0, self.obs_sigma / sp)  # noise on normalized Vt/sp
        return o

class ActionRateLimiter(gym.Wrapper):
    """Throttle slew rate limit before passing to the env."""
    def __init__(self, env, slew=0.02, u0=0.5):
        super().__init__(env)
        self.slew = float(slew)
        self.u_prev = float(u0)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.u_prev = 0.5
        return obs, info

    def step(self, action):
        u_cmd = float(action[0])
        du = np.clip(u_cmd - self.u_prev, -self.slew, self.slew)
        u = np.array([np.clip(self.u_prev + du, 0.0, 1.0)], dtype=np.float32)
        self.u_prev = float(u)
        return self.env.step(u)

class SetpointJumpWrapper(gym.Wrapper):
    """At t = t_jump_s, change sp to sp_new mid-episode (task switch)."""
    def __init__(self, env, t_jump_s=30.0, sp_new=560.0):
        super().__init__(env)
        self.t_jump_s = float(t_jump_s)
        self.sp_new = float(sp_new)
        self._switched = False

    def reset(self, **kwargs):
        self._switched = False
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, r, done, trunc, info = self.env.step(action)
        base = _base_env(self)
        t = info.get("t", None)
        if (t is not None) and (not self._switched) and (t >= self.t_jump_s):
            base.sp = self.sp_new
            self._switched = True
        return obs, r, done, trunc, info
class ThrottleCapWrapper(gym.Wrapper):
    """Caps throttle to <= u_max before the env sees it (simulated partial thrust loss)."""
    def __init__(self, env, u_max=0.80):
        super().__init__(env)
        self.u_max = float(u_max)
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
    def step(self, action):
        a = np.array(action, dtype=np.float32).copy()
        a[0] = np.clip(a[0], 0.0, self.u_max)
        return self.env.step(a)