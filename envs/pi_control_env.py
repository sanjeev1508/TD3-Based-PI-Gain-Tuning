"""
pi_control_env.py
=================
Custom Gymnasium environment for PI gain tuning with TD3.

Observation space: [error_normalized, disturbance_A]  shape=(2,)
Action space:      [alpha, beta]  (Kp_corr, Ki_corr)  shape=(2,)

Constraint satisfied:
  Kp_corr = f(error, disturbance_A)   ← ONLY these two inputs
  Ki_corr = f(error, disturbance_A)   ← ONLY these two inputs
"""

import os
import sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from simulation.plant_simulator import (
    PlantSimulator,
    KP_FIXED, KI_FIXED, SETPOINT,
    DEFAULT_B, DEFAULT_C, DEFAULT_D,
)
from data.generate_profiles import PROFILES


class PIControlEnv(gym.Env):
    """
    Gymnasium environment for TD3-based PI gain tuning.

    At each step:
      obs    = [e(t)/setpoint, A(t)]
      action = [alpha, beta]   (Kp_corr, Ki_corr ∈ [0.1, 5.0])
      Kp_final = Kp_fixed * alpha
      Ki_final = Ki_fixed * beta
    """

    metadata = {"render_modes": []}

    # Episode length (steps at dt=0.01 → 10 seconds simulated)
    EPISODE_STEPS = 1000

    # Action limits
    ALPHA_LOW, ALPHA_HIGH = 0.1, 5.0
    BETA_LOW,  BETA_HIGH  = 0.1, 5.0

    # Error normalisation bound
    ERR_CLIP = 0.3     # clip normalised error to ±0.3

    def __init__(
        self,
        B: float = DEFAULT_B,
        C: float = DEFAULT_C,
        D: float = DEFAULT_D,
        Kp_fixed: float = KP_FIXED,
        Ki_fixed: float = KI_FIXED,
        setpoint: float = SETPOINT,
        dt: float = 0.01,
        randomise_plant: bool = True,
    ):
        super().__init__()
        self.B_nom      = B
        self.C_nom      = C
        self.D_nom      = D
        self.Kp_fixed   = Kp_fixed
        self.Ki_fixed   = Ki_fixed
        self.setpoint   = setpoint
        self.dt         = dt
        self.randomise_plant = randomise_plant

        # Observation: [error_norm, disturbance_A]
        obs_low  = np.array([-self.ERR_CLIP, 0.0], dtype=np.float32)
        obs_high = np.array([ self.ERR_CLIP, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)

        # Action: [alpha, beta]
        act_low  = np.array([self.ALPHA_LOW,  self.BETA_LOW],  dtype=np.float32)
        act_high = np.array([self.ALPHA_HIGH, self.BETA_HIGH], dtype=np.float32)
        self.action_space = spaces.Box(act_low, act_high, dtype=np.float32)

        # Internal state
        self._sim: PlantSimulator = None
        self._step_count   = 0
        self._profile_name = None
        self._dist_fn      = None
        self._prev_alpha   = 1.0
        self._prev_beta    = 1.0
        self._settle_count = 0     # consecutive steps within 1% error

    # ── Gymnasium API ─────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Randomise plant parameters ±10%
        rng = self.np_random
        B = self.B_nom * (1.0 + rng.uniform(-0.1, 0.1)) if self.randomise_plant else self.B_nom
        C = self.C_nom * (1.0 + rng.uniform(-0.1, 0.1)) if self.randomise_plant else self.C_nom
        D = self.D_nom * (1.0 + rng.uniform(-0.1, 0.1)) if self.randomise_plant else self.D_nom

        self._sim = PlantSimulator(
            B=B, C=C, D=D,
            Kp=self.Kp_fixed,
            Ki=self.Ki_fixed,
            setpoint=self.setpoint,
            dt=self.dt,
        )
        self._step_count = 0
        self._prev_alpha = 1.0
        self._prev_beta  = 1.0
        self._settle_count = 0

        # Random disturbance profile for this episode
        names = list(PROFILES.keys())
        idx   = rng.integers(0, len(names))
        self._profile_name = names[idx]
        self._dist_fn      = PROFILES[self._profile_name]

        obs = self._get_obs()
        return obs, {}

    def step(self, action: np.ndarray):
        alpha = float(np.clip(action[0], self.ALPHA_LOW, self.ALPHA_HIGH))
        beta  = float(np.clip(action[1], self.BETA_LOW,  self.BETA_HIGH))

        Kp_final = self.Kp_fixed * alpha
        Ki_final = self.Ki_fixed * beta

        t_now = self._step_count * self.dt
        A     = float(np.clip(self._dist_fn(t_now), 0.0, 1.0))

        result = self._sim.step(Kp=Kp_final, Ki=Ki_final, disturbance=A)

        self._step_count += 1

        e     = result["error"]
        y     = result["output"]
        e_norm = np.clip(e / self.setpoint, -self.ERR_CLIP, self.ERR_CLIP)

        reward = self._compute_reward(e, y, A, alpha, beta)

        self._prev_alpha = alpha
        self._prev_beta  = beta

        done      = False
        truncated = self._step_count >= self.EPISODE_STEPS

        obs  = np.array([e_norm, A], dtype=np.float32)
        info = {
            "alpha": alpha, "beta": beta, "A": A,
            "error": e, "output": y,
        }
        return obs, reward, done, truncated, info

    # ── Reward ────────────────────────────────────────────────────────────────

    def _compute_reward(self, e, y, A, alpha, beta) -> float:
        sp = self.setpoint

        # Primary error penalty
        error_penalty = -abs(e) / sp

        # Overshoot penalty (increasing disturbance → more load → undershoot)
        # Decreasing disturbance may cause overshoot
        if y > sp:
            overshoot_pen = -2.0 * max(0.0, (y - sp) / sp - 0.15)
        else:
            overshoot_pen = 0.0

        if y < sp:
            undershoot_pen = -2.0 * max(0.0, (sp - y) / sp - 0.15)
        else:
            undershoot_pen = 0.0

        # Settle bonus: 30 consecutive steps within 1%
        if abs(e) / sp < 0.01:
            self._settle_count += 1
        else:
            self._settle_count = 0
        settle_bonus = 1.0 if self._settle_count >= 30 else 0.0

        # Smoothness penalty (avoid jerky gain changes)
        smooth_pen = (
            -0.1 * abs(alpha - self._prev_alpha)
            - 0.1 * abs(beta  - self._prev_beta)
        )

        return error_penalty + overshoot_pen + undershoot_pen + settle_bonus + smooth_pen

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        e = self._sim.error
        e_norm = float(np.clip(e / self.setpoint, -self.ERR_CLIP, self.ERR_CLIP))
        A = float(np.clip(self._dist_fn(0.0), 0.0, 1.0)) if self._dist_fn else 0.0
        return np.array([e_norm, A], dtype=np.float32)

    def render(self):
        pass

    def close(self):
        pass
