"""
compute_metrics.py
==================
Time-domain and frequency-domain (Bode) metrics for PI gain tuning evaluation.

Time domain:
  O%   = overshoot percentage
  U%   = undershoot percentage
  T    = settling time (|e| < 1% of setpoint continuously)
  SS   = steady-state time (|e| < 0.25% continuously)
  ITAE = ∫ t·|e(t)| dt

Frequency domain (open-loop L(s)):
  L(s) = C(s) × G(s) = [B(Kp·s + Ki)] / [s³ + Cs² + Ds]
  Bandwidth  = frequency where |L| drops to -3 dB
  Gain Margin   = negative of |L| (dB) at phase = -180°
  Phase Margin  = 180° + phase(L) at |L| = 0 dB
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import pandas as pd
from scipy import signal

from simulation.plant_simulator import (
    KP_FIXED, KI_FIXED, SETPOINT,
    DEFAULT_B, DEFAULT_C, DEFAULT_D,
)


# ─── Time domain ──────────────────────────────────────────────────────────────

def compute_time_metrics(
    time:      np.ndarray,
    output:    np.ndarray,
    setpoint:  float = SETPOINT,
    tol_settle: float = 0.01,
    tol_ss:     float = 0.0025,
) -> dict:
    """
    Compute time-domain performance metrics.

    Parameters
    ----------
    tol_settle : |e|/SP threshold for settling time  (default 1%)
    tol_ss     : |e|/SP threshold for steady-state   (default 0.25%)
    """
    error    = setpoint - output
    rel_err  = np.abs(error) / setpoint

    # Peak overshoot and undershoot
    overshoot_pct  = float(max(0.0, (output.max() - setpoint) / setpoint * 100))
    undershoot_pct = float(max(0.0, (setpoint - output.min()) / setpoint * 100))

    # Settling time: last index where rel_err > tol_settle
    exceed  = np.where(rel_err > tol_settle)[0]
    settling_time = float(time[exceed[-1]]) if len(exceed) else 0.0

    # Steady-state time
    exceed_ss = np.where(rel_err > tol_ss)[0]
    ss_time   = float(time[exceed_ss[-1]]) if len(exceed_ss) else 0.0

    # ITAE
    itae = float(np.trapz(time * np.abs(error), time))

    return {
        "overshoot_pct":  round(overshoot_pct, 4),
        "undershoot_pct": round(undershoot_pct, 4),
        "settling_time":  round(settling_time, 4),
        "ss_time":        round(ss_time, 4),
        "itae":           round(itae, 4),
    }


# ─── Frequency domain ─────────────────────────────────────────────────────────

def compute_freq_metrics(
    Kp: float,
    Ki: float,
    B:  float = DEFAULT_B,
    C:  float = DEFAULT_C,
    D:  float = DEFAULT_D,
    worN: int = 2000,
) -> dict:
    """
    Compute Bode-based stability margins for the open-loop L(s).

    L(s) = B*(Kp*s + Ki) / (s^3 + C*s^2 + D*s)

    Returns
    -------
    dict with: bandwidth_rad_s, gain_margin_dB, phase_margin_deg
    """
    num_L = [B * Kp, B * Ki]
    den_L = [1.0, C, D, 0.0]

    w = np.logspace(-2, 3, worN)
    _, H = signal.freqs(num_L, den_L, worN=w)

    mag_dB    = 20.0 * np.log10(np.abs(H) + 1e-12)
    phase_deg = np.degrees(np.unwrap(np.angle(H)))

    # ─ Bandwidth: first frequency where |L| ≤ 0 dB (unity gain crossover)
    # Use -3 dB drop from DC for bandwidth
    dc_mag = mag_dB[0]
    bw_idx = np.where(mag_dB <= dc_mag - 3.0)[0]
    bandwidth = float(w[bw_idx[0]]) if len(bw_idx) else float(w[-1])

    # ─ Phase margin: 180° + phase at gain crossover (|L|=0 dB)
    gc_idx = np.where(np.diff(np.sign(mag_dB)))[0]
    if len(gc_idx):
        pm = 180.0 + float(phase_deg[gc_idx[0]])
    else:
        pm = float("inf")

    # ─ Gain margin: -magnitude at phase crossover (-180°)
    pc_idx = np.where(np.diff(np.sign(phase_deg + 180.0)))[0]
    if len(pc_idx):
        gm = -float(mag_dB[pc_idx[0]])
    else:
        gm = float("inf")

    return {
        "bandwidth_rad_s":   round(bandwidth, 4),
        "gain_margin_dB":    round(gm, 4),
        "phase_margin_deg":  round(pm, 4),
    }


def get_bode_curves(
    Kp: float, Ki: float,
    B: float = DEFAULT_B,
    C: float = DEFAULT_C,
    D: float = DEFAULT_D,
    worN: int = 2000,
) -> tuple:
    """
    Return (w, mag_dB, phase_deg) for the open-loop L(s).
    Used by the plotting module.
    """
    num_L = [B * Kp, B * Ki]
    den_L = [1.0, C, D, 0.0]
    w = np.logspace(-2, 3, worN)
    _, H = signal.freqs(num_L, den_L, worN=w)
    mag_dB    = 20.0 * np.log10(np.abs(H) + 1e-12)
    phase_deg = np.degrees(np.unwrap(np.angle(H)))
    return w, mag_dB, phase_deg


# ─── Full comparison table ────────────────────────────────────────────────────

def build_metrics_table(eval_results: dict, setpoint: float = SETPOINT) -> pd.DataFrame:
    """
    Build a comprehensive metrics comparison table.

    Parameters
    ----------
    eval_results : dict[profile_name] = {"td3": arrays, "baseline": arrays}
    """
    rows = []
    for profile_name, data in eval_results.items():
        for method in ("baseline", "td3"):
            arr = data[method]
            m   = compute_time_metrics(arr["time"], arr["actual_value"], setpoint)

            # Mean gains for frequency domain
            Kp_mean = float(np.mean(arr["Kp_applied"]))
            Ki_mean = float(np.mean(arr["Ki_applied"]))
            fm = compute_freq_metrics(Kp_mean, Ki_mean)

            rows.append({
                "Profile":         profile_name,
                "Method":          method.upper(),
                "O%":              m["overshoot_pct"],
                "U%":              m["undershoot_pct"],
                "Settling_s":      m["settling_time"],
                "SS_Time_s":       m["ss_time"],
                "ITAE":            m["itae"],
                "Bandwidth_rad_s": fm["bandwidth_rad_s"],
                "GainMargin_dB":   fm["gain_margin_dB"],
                "PhaseMargin_deg": fm["phase_margin_deg"],
            })

    df = pd.DataFrame(rows)
    return df


def print_summary_table(df: pd.DataFrame):
    """Print a formatted summary to console."""
    print("\n" + "=" * 100)
    print("PERFORMANCE METRICS SUMMARY")
    print("=" * 100)

    for profile in df["Profile"].unique():
        sub = df[df["Profile"] == profile]
        base = sub[sub["Method"] == "BASELINE"].iloc[0]
        td3  = sub[sub["Method"] == "TD3"].iloc[0]

        def pct_improvement(b, t, lower_is_better=True):
            if b == 0:
                return "N/A"
            delta = (b - t) / abs(b) * 100 if lower_is_better else (t - b) / abs(b) * 100
            sign  = "↓" if delta > 0 else "↑"
            return f"{sign} {abs(delta):.0f}%"

        print(f"\n  Profile {profile}")
        print(f"  {'Metric':<22} {'Baseline':>12} {'TD3':>12} {'Improvement':>14}")
        print(f"  {'-'*62}")
        for col, label in [
            ("O%",         "Overshoot %"),
            ("U%",         "Undershoot %"),
            ("Settling_s", "Settling Time (s)"),
            ("SS_Time_s",  "Steady-State (s)"),
            ("ITAE",       "ITAE"),
        ]:
            b_val = base[col]
            t_val = td3[col]
            impr  = pct_improvement(b_val, t_val)
            print(f"  {label:<22} {b_val:>12.4f} {t_val:>12.4f} {impr:>14}")

    print("\n" + "=" * 100)

    # Frequency domain table
    print("\nFREQUENCY DOMAIN (Bode Analysis)")
    print("-" * 60)
    print(f"  {'Profile':<10} {'Method':<10} {'BW (rad/s)':>12} {'GM (dB)':>10} {'PM (°)':>10}")
    print(f"  {'-'*54}")
    for _, row in df.iterrows():
        print(f"  {row['Profile']:<10} {row['Method']:<10} "
              f"{row['Bandwidth_rad_s']:>12.4f} {row['GainMargin_dB']:>10.4f} "
              f"{row['PhaseMargin_deg']:>10.4f}")
    print()


if __name__ == "__main__":
    # Quick test
    m = compute_time_metrics(
        np.linspace(0, 100, 10000),
        SETPOINT * (1 - np.exp(-np.linspace(0, 10, 10000))),
    )
    print("Time metrics test:", m)

    fm = compute_freq_metrics(KP_FIXED, KI_FIXED)
    print("Freq metrics test:", fm)
