"""
run_evaluation.py
=================
Closed-loop evaluation of the TD3 agent vs. baseline PI controller.

For each profile the loop at each timestep:
  1. Computes error e(t) = 900 - y(t)
  2. Builds obs = [e(t)/900, A(t)]
  3. Calls alpha, beta = model.predict(obs, deterministic=True)
  4. Applies Kp_final = Kp_fixed * alpha,  Ki_final = Ki_fixed * beta
  5. Advances PI + plant ODE by dt=0.01s
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import pandas as pd

from simulation.plant_simulator import (
    PlantSimulator, simulate_profile,
    KP_FIXED, KI_FIXED, SETPOINT,
    DEFAULT_B, DEFAULT_C, DEFAULT_D,
)
from data.generate_profiles import PROFILES


# ─── Closed-loop run with TD3 model ──────────────────────────────────────────

def run_closed_loop(
    model,
    disturbance_fn,
    duration: float = 1000.0,
    dt: float = 0.01,
    B: float = DEFAULT_B,
    C: float = DEFAULT_C,
    D: float = DEFAULT_D,
    Kp_fixed: float = KP_FIXED,
    Ki_fixed: float = KI_FIXED,
    setpoint: float = SETPOINT,
) -> dict:
    """
    Run closed loop with TD3 agent adjusting PI gains.

    Returns
    -------
    dict of arrays: time, setpoint, actual_value, error, disturbance_A,
                    alpha, beta, Kp_applied, Ki_applied, control_output
    """
    sim = PlantSimulator(B=B, C=C, D=D, setpoint=setpoint, dt=dt)
    n_steps = int(duration / dt)

    arrays = {k: np.zeros(n_steps) for k in [
        "time", "setpoint", "actual_value", "error",
        "disturbance_A", "alpha", "beta",
        "Kp_applied", "Ki_applied", "control_output",
    ]}

    ERR_CLIP = 0.3

    for i in range(n_steps):
        t_now = i * dt
        A     = float(np.clip(disturbance_fn(t_now), 0.0, 1.0))
        e     = sim.error

        e_norm = float(np.clip(e / setpoint, -ERR_CLIP, ERR_CLIP))
        obs    = np.array([e_norm, A], dtype=np.float32)

        action, _ = model.predict(obs, deterministic=True)
        alpha = float(np.clip(action[0], 0.1, 5.0))
        beta  = float(np.clip(action[1], 0.1, 5.0))

        Kp_final = Kp_fixed * alpha
        Ki_final = Ki_fixed * beta

        result = sim.step(Kp=Kp_final, Ki=Ki_final, disturbance=A)

        arrays["time"][i]           = t_now
        arrays["setpoint"][i]       = setpoint
        arrays["actual_value"][i]   = result["output"]
        arrays["error"][i]          = result["error"]
        arrays["disturbance_A"][i]  = A
        arrays["alpha"][i]          = alpha
        arrays["beta"][i]           = beta
        arrays["Kp_applied"][i]     = Kp_final
        arrays["Ki_applied"][i]     = Ki_final
        arrays["control_output"][i] = result["control_output"]

    return arrays


def run_baseline(
    disturbance_fn,
    duration: float = 1000.0,
    dt: float = 0.01,
    **kwargs,
) -> dict:
    """Run baseline PI (no ML, fixed gains α=1, β=1)."""
    return simulate_profile(
        disturbance_fn=disturbance_fn,
        duration=duration,
        dt=dt,
        **kwargs,
    )


def evaluate_all_profiles(
    model,
    output_dir: str = None,
    duration: float = 1000.0,
    verbose: bool = True,
) -> dict:
    """
    Run TD3 + baseline evaluation on all 6 profiles.

    Returns
    -------
    dict[profile_name] = {"td3": ..., "baseline": ...}
    """
    if output_dir is None:
        output_dir = os.path.join(_ROOT, "evaluation")
    os.makedirs(output_dir, exist_ok=True)

    results = {}
    for name, dist_fn in PROFILES.items():
        if verbose:
            print(f"  Evaluating profile {name}...")

        td3_data   = run_closed_loop(model, dist_fn, duration=duration)
        base_data  = run_baseline(dist_fn, duration=duration)

        # Save CSVs
        pd.DataFrame(td3_data).to_csv(
            os.path.join(output_dir, f"td3_profile_{name}.csv"), index=False
        )
        pd.DataFrame(base_data).to_csv(
            os.path.join(output_dir, f"baseline_profile_{name}.csv"), index=False
        )

        results[name] = {"td3": td3_data, "baseline": base_data}

    return results


if __name__ == "__main__":
    from training.train_td3 import load_model

    print("Loading model...")
    model = load_model()
    print("Running evaluation on all profiles...")
    results = evaluate_all_profiles(model, verbose=True)
    print("Done.")
