"""
train_td3.py
============
TD3 training script using stable-baselines3.

Algorithm: TD3 (Twin Delayed Deep Deterministic Policy Gradient)
  - Twin critics prevent Q-value overestimation
  - Delayed policy updates (every 2 critic steps) → stable training
  - Target policy noise → prevents exploiting sharp Q-peaks
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import (
    EvalCallback, CallbackList, BaseCallback
)
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from envs.pi_control_env import PIControlEnv

MODELS_DIR = os.path.join(_ROOT, "models")
BEST_MODEL  = os.path.join(MODELS_DIR, "best_model")


# ─── ITAE logging callback ────────────────────────────────────────────────────
class ITAECallback(BaseCallback):
    """Logs mean ITAE from the last batch of rollouts every log_freq steps."""

    def __init__(self, log_freq: int = 10_000, verbose: int = 1):
        super().__init__(verbose)
        self.log_freq = log_freq
        self._ep_itae: list[float] = []

    def _on_step(self) -> bool:
        # Accumulate ITAE from info dicts
        for info in self.locals.get("infos", []):
            if "terminal_observation" in info:
                buf = self.training_env.envs[0].episode_itae
                if buf:
                    self._ep_itae.append(buf)

        if self.n_calls % self.log_freq == 0 and self._ep_itae:
            mean_itae = np.mean(self._ep_itae[-20:])
            if self.verbose:
                print(f"  [ITAE @ {self.n_calls:>7d} steps]  mean = {mean_itae:.4f}")
            self._ep_itae.clear()
        return True


def make_env(randomise_plant: bool = True):
    """Factory for creating a monitored env."""
    def _init():
        env = PIControlEnv(randomise_plant=randomise_plant)
        env = Monitor(env)
        return env
    return _init


def train_td3(
    total_timesteps: int = 300_000,
    force_retrain: bool = False,
    verbose: int = 1,
) -> TD3:
    """
    Train the TD3 agent.

    Parameters
    ----------
    total_timesteps : training budget
    force_retrain   : if True, retrain even if saved model exists
    verbose         : SB3 verbosity level

    Returns
    -------
    Trained TD3 model
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    zip_path = BEST_MODEL + ".zip"

    if os.path.exists(zip_path) and not force_retrain:
        print(f"[train_td3] Found existing model at {zip_path}. Loading...")
        model = TD3.load(BEST_MODEL)
        return model

    print("[train_td3] Creating environment...")
    train_env = DummyVecEnv([make_env(randomise_plant=True)])
    eval_env  = DummyVecEnv([make_env(randomise_plant=False)])

    # Exploration noise
    n_actions    = train_env.action_space.shape[-1]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=0.1 * np.ones(n_actions),
    )

    # TD3 hyperparameters
    model = TD3(
        policy         = "MlpPolicy",
        env            = train_env,
        learning_rate  = 3e-4,
        buffer_size    = 300_000,
        learning_starts= 10_000,
        batch_size     = 256,
        tau            = 0.005,
        gamma          = 0.99,
        train_freq     = 1,
        policy_delay   = 2,
        target_policy_noise = 0.2,
        target_noise_clip   = 0.5,
        action_noise   = action_noise,
        policy_kwargs  = {"net_arch": [400, 300]},
        verbose        = verbose,
        tensorboard_log= os.path.join(_ROOT, "logs", "td3_tensorboard"),
    )

    # Callbacks
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path = MODELS_DIR,
        log_path             = os.path.join(MODELS_DIR, "eval_logs"),
        eval_freq            = 5_000,
        n_eval_episodes      = 5,
        deterministic        = True,
        verbose              = verbose,
    )

    print(f"[train_td3] Starting training for {total_timesteps:,} timesteps...")
    model.learn(
        total_timesteps  = total_timesteps,
        callback         = CallbackList([eval_cb]),
        progress_bar     = True,
        reset_num_timesteps = True,
    )

    # Save final model too
    final_path = os.path.join(MODELS_DIR, "final_model")
    model.save(final_path)
    print(f"[train_td3] Final model saved → {final_path}.zip")
    print(f"[train_td3] Best model saved  → {BEST_MODEL}.zip")
    return model


def load_model() -> TD3:
    """Load the best saved model."""
    zip_path = BEST_MODEL + ".zip"
    if not os.path.exists(zip_path):
        raise FileNotFoundError(
            f"No saved model found at {zip_path}. Run training first."
        )
    print(f"[load_model] Loading from {zip_path}")
    return TD3.load(BEST_MODEL)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--force",     action="store_true")
    args = parser.parse_args()

    train_td3(total_timesteps=args.timesteps, force_retrain=args.force)
