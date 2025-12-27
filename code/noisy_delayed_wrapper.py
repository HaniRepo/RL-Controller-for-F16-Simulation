import numpy as np
import gymnasium as gym
from collections import deque

class NoisyDelayedWrapper(gym.Wrapper):
    def __init__(self, env, obs_sigma=2.0, act_delay_steps=2, u_clip=(0.0,1.0)):
        super().__init__(env)
        self.obs_sigma = obs_sigma      # ft/s noise on Vt channel
        self.act_delay_steps = act_delay_steps
        self.u_min, self.u_max = u_clip
        self._act_fifo = deque([0.5]*act_delay_steps, maxlen=act_delay_steps)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._act_fifo = deque([0.5]*self.act_delay_steps, maxlen=self.act_delay_steps)
        return self._noisy_obs(obs), info

    def step(self, action):
        # delay: apply oldest action
        self._act_fifo.append(float(action[0]))
        u = np.array([np.clip(self._act_fifo[0], self.u_min, self.u_max)], dtype=np.float32)
        obs, r, done, trunc, info = self.env.step(u)
        return self._noisy_obs(obs), r, done, trunc, info

    def _noisy_obs(self, obs):
        o = np.array(obs, dtype=np.float32).copy()
        # assume first obs component relates to Vt/sp; add mild noise
        o[0] += np.random.normal(0.0, self.obs_sigma/ max(1e-6, self.env.sp))
        return o
