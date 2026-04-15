# genai_hackathon/genai_shield.py

import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from conformal_shield import worst_case_band_violation


class GenerativeConformalShield:
    """
    Lightweight GenAI shield:
    generate several candidate actions around PPO output,
    roll them forward with the predictor,
    select the candidate with best worst-case STL robustness.
    """

    def __init__(
        self,
        predictor,
        q,
        K=4,
        dt=0.1,
        tol=0.05,
        slew=0.03,
        candidate_offsets=(-0.03, -0.015, 0.0, 0.015, 0.03)
        #candidate_offsets=(-0.10, -0.05, 0.0, 0.05, 0.10), TOO harsh
    ):
        self.p = predictor
        self.q = float(q)
        self.K = int(K)
        self.dt = float(dt)
        self.tol = float(tol)
        self.slew = float(slew)
        self.candidate_offsets = list(candidate_offsets)
        self.u_prev = 0.5

    def reset(self, u0=0.5):
        self.u_prev = float(u0)

    def _roll_pred(self, vt0, pw0, u_const):
        vt, pw = float(vt0), float(pw0)
        seq = []

        for _ in range(self.K):
            vt = self.p.predict_next(vt, pw, u_const)
            pw += 0.3 * (u_const - pw)  # simple engine lag approximation
            seq.append(vt)

        return np.asarray(seq, dtype=float)

    def _build_candidates(self, u_rl, vt, sp):
        """
        Directional candidate generation:
        if below setpoint, explore stronger throttle actions;
        if above setpoint, explore lower throttle actions.
        """
        if vt < sp:
            offsets = [0.0, 0.03, 0.06, 0.10, 0.14]
        else:
            offsets = [-0.14, -0.10, -0.06, -0.03, 0.0]

        cands = []
        for off in offsets:
            u = np.clip(u_rl + off, 0.0, 1.0)
            u = np.clip(u, self.u_prev - self.slew, self.u_prev + self.slew)
            u = np.clip(u, 0.0, 1.0)
            cands.append(float(u))

        uniq = []
        seen = set()
        for u in cands:
            key = round(u, 6)
            if key not in seen:
                seen.add(key)
                uniq.append(u)
        return uniq

    def filter(self, vt, pw, sp, u_rl):
        candidates = self._build_candidates(float(u_rl), float(vt), float(sp))

        best_u = float(u_rl)
        best_rho = -1e9

        for u in candidates:
            seq = self._roll_pred(vt, pw, u)
            rho = worst_case_band_violation(seq, self.q, sp, tol=self.tol)

            if rho > best_rho:
                best_rho = float(rho)
                best_u = float(u)

        self.u_prev = best_u
        return best_u, best_rho