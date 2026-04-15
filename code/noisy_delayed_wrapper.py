import numpy as np
import gymnasium as gym
from collections import deque

class NoisyDelayedWrapper(gym.Wrapper):
    """Obs noise on Vt/sp and optional k-step action delay."""
    def __init__(self, env, obs_sigma=2.0, act_delay_steps=0, u_clip=(0.0,1.0)):
        super().__init__(env)
        self.obs_sigma = float(obs_sigma)
        self.act_delay_steps = int(act_delay_steps)
        self.u_min, self.u_max = map(float, u_clip)
        self._fifo = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if self.act_delay_steps > 0:
            self._fifo = deque([0.5]*self.act_delay_steps, maxlen=self.act_delay_steps)
        else:
            self._fifo = None
        return self._noisy_obs(obs), info

    def step(self, action):
        u_in = float(np.asarray(action).reshape(-1)[0])

        if self.act_delay_steps > 0:
            self._fifo.append(u_in)
            u_apply = float(np.clip(self._fifo[0], self.u_min, self.u_max))
        else:
            u_apply = float(np.clip(u_in, self.u_min, self.u_max))

        u = np.array([u_apply], dtype=np.float32)
        obs, r, done, trunc, info = self.env.step(u)
        return self._noisy_obs(obs), r, done, trunc, info

    def _noisy_obs(self, obs):
        o = np.array(obs, dtype=np.float32).copy()
        base = _base_env(self.env)
        sp = getattr(base, "sp", 500.0)
        if o.size > 0 and sp > 1e-6:
            o[0] += np.random.normal(0.0, self.obs_sigma / sp)
        return o
