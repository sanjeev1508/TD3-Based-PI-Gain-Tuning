"""
generate_profiles.py
====================
Generates all 6 disturbance profiles (A–F) and saves CSV files.
Each CSV has columns:
  time, setpoint, actual_value, error, disturbance_A,
  Kp_applied, Ki_applied, control_output
"""

import os
import sys
import numpy as np
import pandas as pd

# ── path setup ────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from simulation.plant_simulator import (
    simulate_profile,
    KP_FIXED, KI_FIXED, SETPOINT,
    DEFAULT_B, DEFAULT_C, DEFAULT_D,
)

DATA_DIR  = os.path.join(_HERE)
DURATION  = 1000.0   # seconds
DT        = 0.01     # 10 ms


# ─── Disturbance profile functions ────────────────────────────────────────────

def profile_A(t: float) -> float:
    """5% step-sheds: triangular ramp, 0→1 then 1→0 over 1000s."""
    ramp_up_end   = 500.0   # 0→1 in 500s (steps every 50s)
    steps_up   = int(t / 50) if t < ramp_up_end else None
    steps_down = int((t - ramp_up_end) / 50) if t >= ramp_up_end else None

    if t < ramp_up_end:
        return min(1.0, np.floor(t / 50) * 0.05)
    else:
        elapsed = t - ramp_up_end
        return max(0.0, 1.0 - np.floor(elapsed / 50) * 0.05)


def profile_B(t: float) -> float:
    """10% step-sheds: 0→1 every 100s then 1→0."""
    ramp_up_end = 500.0
    if t < ramp_up_end:
        return min(1.0, np.floor(t / 100) * 0.10)
    else:
        elapsed = t - ramp_up_end
        return max(0.0, 1.0 - np.floor(elapsed / 100) * 0.10)


def profile_C(t: float) -> float:
    """15% step-sheds: 0→1 every ~67s then 1→0."""
    step_dur = 1000.0 / 14.0        # ≈ 71.4 s per step to cover 0→1 in 7 steps then back
    ramp_up_end = 500.0
    if t < ramp_up_end:
        return min(1.0, np.floor(t / (ramp_up_end / 7)) * 0.15)
    else:
        elapsed = t - ramp_up_end
        return max(0.0, 1.0 - np.floor(elapsed / (500.0 / 7)) * 0.15)


def profile_D(t: float) -> float:
    """20% step-sheds: 0→1 every 100s then 1→0."""
    ramp_up_end = 500.0
    if t < ramp_up_end:
        return min(1.0, np.floor(t / 100) * 0.20)
    else:
        elapsed = t - ramp_up_end
        return max(0.0, 1.0 - np.floor(elapsed / 100) * 0.20)


def profile_E(t: float) -> float:
    """30% step-sheds: 0→1 in ~4 steps each ~100s then 1→0."""
    ramp_up_end = 400.0
    if t < ramp_up_end:
        return min(1.0, np.floor(t / 100) * 0.30)
    else:
        elapsed = t - ramp_up_end
        return max(0.0, 1.0 - np.floor(elapsed / (600.0 / 4)) * 0.30)


def profile_F(t: float) -> float:
    """Large step: A=0.85 for [0,400], A=0.25 for [400,1000]."""
    return 0.85 if t < 400.0 else 0.25


PROFILES = {
    "A": profile_A,
    "B": profile_B,
    "C": profile_C,
    "D": profile_D,
    "E": profile_E,
    "F": profile_F,
}


def generate_all_profiles(output_dir: str = DATA_DIR, verbose: bool = True):
    """Generate all profiles and save as CSV files."""
    os.makedirs(output_dir, exist_ok=True)

    results = {}
    for name, dist_fn in PROFILES.items():
        if verbose:
            print(f"  Generating profile {name}...", end="", flush=True)

        data = simulate_profile(
            disturbance_fn=dist_fn,
            duration=DURATION,
            dt=DT,
            B=DEFAULT_B, C=DEFAULT_C, D=DEFAULT_D,
            Kp=KP_FIXED, Ki=KI_FIXED,
            setpoint=SETPOINT,
        )

        df = pd.DataFrame(data)
        csv_path = os.path.join(output_dir, f"data_profile_{name}.csv")
        df.to_csv(csv_path, index=False)
        results[name] = data

        if verbose:
            print(f" saved → {csv_path}")

    return results


if __name__ == "__main__":
    print("Generating disturbance profiles...")
    generate_all_profiles(verbose=True)
    print("Done.")
