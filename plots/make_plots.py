"""
make_plots.py
=============
Generates all 4 required publication-quality plots.

Design: dark background (#111418), yellow accent (#F5C400),
        green for TD3 (#2DBD6E), red for baseline (#E8534A).
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

from evaluation.compute_metrics import (
    get_bode_curves, compute_freq_metrics, compute_time_metrics
)
from simulation.plant_simulator import KP_FIXED, KI_FIXED, SETPOINT
from simulation.plant_simulator import DEFAULT_B, DEFAULT_C, DEFAULT_D

# ─── Style constants ──────────────────────────────────────────────────────────
BG      = "#111418"
FG      = "#E8E8E8"
YELLOW  = "#F5C400"
GREEN   = "#2DBD6E"
RED     = "#E8534A"
BLUE    = "#4A9EE8"
ORANGE  = "#F5A623"
ACCENT  = "#9B59B6"

PLOTS_DIR = os.path.join(_ROOT, "plots")


def _style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(BG)
    ax.tick_params(colors=FG, labelsize=9)
    ax.xaxis.label.set_color(FG)
    ax.yaxis.label.set_color(FG)
    ax.title.set_color(YELLOW)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333840")
    ax.grid(True, color="#252930", linewidth=0.5, linestyle="--")
    if title:   ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    if xlabel:  ax.set_xlabel(xlabel, fontsize=9)
    if ylabel:  ax.set_ylabel(ylabel, fontsize=9)


def _fig(nrows=1, ncols=1, w=14, h=6):
    fig, axes = plt.subplots(nrows, ncols, figsize=(w, h))
    fig.patch.set_facecolor(BG)
    return fig, axes


# ════════════════════════════════════════════════════════════════════════════
# PLOT 1 — Error Comparison (Incremental | Decremental)
# ════════════════════════════════════════════════════════════════════════════

def plot_error_comparison(
    incr_data: dict,   # {"td3": arrays, "baseline": arrays}
    decr_data: dict,
    save_path: str,
    setpoint: float = SETPOINT,
):
    fig, axes = _fig(1, 2, w=16, h=7)

    panels = [
        (axes[0], incr_data, "Incremental Disturbance (A: 0 → 1)"),
        (axes[1], decr_data, "Decremental Disturbance (A: 1 → 0)"),
    ]

    for ax, data, title in panels:
        td3  = data["td3"]
        base = data["baseline"]

        t     = td3["time"]
        e_td3 = td3["error"] / setpoint * 100       # as % of setpoint
        e_bas = base["error"] / setpoint * 100
        A     = td3["disturbance_A"]

        ax2 = ax.twinx()
        ax2.set_facecolor(BG)
        ax2.tick_params(colors="#888", labelsize=8)
        ax2.plot(t, A, color=ORANGE, alpha=0.35, lw=1.2, label="Disturbance A")
        ax2.set_ylabel("Disturbance A", color="#888", fontsize=8)
        ax2.set_ylim(-0.05, 1.5)

        # Requirement envelope
        ax.axhspan(-15, 15, alpha=0.06, color=YELLOW, label="±15% Envelope")
        ax.axhline( 15, color=YELLOW, lw=0.8, ls="--", alpha=0.6)
        ax.axhline(-15, color=YELLOW, lw=0.8, ls="--", alpha=0.6)
        ax.axhline(  0, color=FG,     lw=0.6, alpha=0.3)

        ax.plot(t, e_bas, color=RED,   lw=1.2, alpha=0.85, label="Baseline PI error %")
        ax.plot(t, e_td3, color=GREEN, lw=1.5, alpha=0.92, label="TD3 error %")

        # Annotate settling
        tol = 1.0  # 1% of setpoint
        exc = np.where(np.abs(e_td3) > tol)[0]
        if len(exc):
            t_set = t[exc[-1]]
            ax.axvline(t_set, color=GREEN, lw=0.8, ls=":", alpha=0.7)
            ax.annotate(
                f"Settle\n{t_set:.1f}s",
                xy=(t_set, 0), xytext=(t_set + t[-1]*0.03, 12),
                arrowprops=dict(arrowstyle="->", color=GREEN, lw=0.8),
                color=GREEN, fontsize=7,
            )

        _style_ax(ax, title=title, xlabel="Time (s)", ylabel="Error e(t) [% of SP]")
        ax.set_xlim(t[0], t[-1])

        handles = [
            Line2D([0], [0], color=RED,    lw=1.5, label="Baseline PI"),
            Line2D([0], [0], color=GREEN,  lw=1.5, label="TD3 Adaptive"),
            Line2D([0], [0], color=YELLOW, lw=1.5, ls="--", label="±15% Envelope"),
            Line2D([0], [0], color=ORANGE, lw=1.5, alpha=0.6, label="Disturbance A"),
        ]
        ax.legend(handles=handles, fontsize=8, facecolor="#1A1F26",
                  edgecolor="#333", labelcolor=FG, loc="upper right")

    fig.suptitle("Error e(t): Baseline PI vs TD3 Adaptive PI", fontsize=14,
                 fontweight="bold", color=YELLOW, y=1.01)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ════════════════════════════════════════════════════════════════════════════
# PLOT 2 — Control Output Comparison
# ════════════════════════════════════════════════════════════════════════════

def plot_control_output(incr_data, decr_data, save_path):
    fig, axes = _fig(1, 2, w=16, h=7)

    panels = [
        (axes[0], incr_data, "Incremental Disturbance"),
        (axes[1], decr_data, "Decremental Disturbance"),
    ]

    for ax, data, title in panels:
        td3  = data["td3"]
        base = data["baseline"]
        t    = td3["time"]
        A    = td3["disturbance_A"]

        ax2 = ax.twinx()
        ax2.set_facecolor(BG)
        ax2.fill_between(t, A, alpha=0.12, color=ORANGE)
        ax2.plot(t, A, color=ORANGE, alpha=0.5, lw=1.0, label="Disturbance A")
        ax2.set_ylabel("Disturbance A", color="#888", fontsize=8)
        ax2.tick_params(colors="#888", labelsize=8)
        ax2.set_ylim(-0.05, 1.5)

        ax.plot(t, base["control_output"], color=RED,   lw=1.2, alpha=0.8,
                label="Baseline u(t)")
        ax.plot(t, td3["control_output"],  color=GREEN, lw=1.5, alpha=0.9,
                label="TD3 u(t)")

        handles = [
            Line2D([0], [0], color=RED,    lw=1.5, label="Baseline PI"),
            Line2D([0], [0], color=GREEN,  lw=1.5, label="TD3 Adaptive"),
            Line2D([0], [0], color=ORANGE, lw=1.5, alpha=0.6, label="Disturbance A"),
        ]
        _style_ax(ax, title=title, xlabel="Time (s)", ylabel="Control Output u(t)")
        ax.set_xlim(t[0], t[-1])
        ax.legend(handles=handles, fontsize=8, facecolor="#1A1F26",
                  edgecolor="#333", labelcolor=FG)

    fig.suptitle("Control Output u(t): Actuator Effort", fontsize=14,
                 fontweight="bold", color=YELLOW, y=1.01)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ════════════════════════════════════════════════════════════════════════════
# PLOT 3 — Gain Adjustment Over Time
# ════════════════════════════════════════════════════════════════════════════

def plot_gain_adjustment(td3_data: dict, save_path: str, title_suffix: str = ""):
    fig, ax = _fig(1, 1, w=14, h=6)
    t     = td3_data["time"]
    alpha = td3_data["alpha"]
    beta  = td3_data["beta"]
    A     = td3_data["disturbance_A"]

    ax2 = ax.twinx()
    ax2.set_facecolor(BG)
    ax2.fill_between(t, A, alpha=0.10, color=ORANGE)
    ax2.plot(t, A, color=ORANGE, alpha=0.55, lw=1.0, label="Disturbance A")
    ax2.set_ylabel("Disturbance A", color="#888", fontsize=8)
    ax2.tick_params(colors="#888", labelsize=8)
    ax2.set_ylim(-0.05, 1.6)

    ax.plot(t, alpha, color=YELLOW, lw=1.5, label="α(t)  Kp_corr")
    ax.plot(t, beta,  color=GREEN,  lw=1.5, label="β(t)  Ki_corr")
    ax.axhline(1.0, color=FG, lw=0.6, ls=":", alpha=0.3)

    # Annotations
    ax.annotate("Kp_final = Kp_fixed × α", xy=(t[len(t)//10], alpha[len(t)//10]),
                xytext=(t[len(t)//6], 4.2),
                arrowprops=dict(arrowstyle="->", color=YELLOW, lw=0.8),
                color=YELLOW, fontsize=8)
    ax.annotate("Ki_final = Ki_fixed × β", xy=(t[len(t)//10], beta[len(t)//10]),
                xytext=(t[len(t)//4], 3.5),
                arrowprops=dict(arrowstyle="->", color=GREEN, lw=0.8),
                color=GREEN, fontsize=8)

    handles = [
        Line2D([0],[0], color=YELLOW, lw=1.5, label="α(t)  Kp_corr"),
        Line2D([0],[0], color=GREEN,  lw=1.5, label="β(t)  Ki_corr"),
        Line2D([0],[0], color=ORANGE, lw=1.5, alpha=0.6, label="Disturbance A"),
    ]
    _style_ax(ax, title=f"TD3 Gain Correction Factors α(t) and β(t){title_suffix}",
              xlabel="Time (s)", ylabel="Correction Factor")
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(0, 5.5)
    ax.legend(handles=handles, fontsize=9, facecolor="#1A1F26",
              edgecolor="#333", labelcolor=FG)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ════════════════════════════════════════════════════════════════════════════
# PLOT 4 — Bode Plot
# ════════════════════════════════════════════════════════════════════════════

def plot_bode(
    td3_data:  dict,
    base_data: dict,
    save_path: str,
    B: float = DEFAULT_B,
    C: float = DEFAULT_C,
    D: float = DEFAULT_D,
):
    Kp_td3   = float(np.mean(td3_data["Kp_applied"]))
    Ki_td3   = float(np.mean(td3_data["Ki_applied"]))
    Kp_base  = KP_FIXED
    Ki_base  = KI_FIXED

    w_td3,   mag_td3,   ph_td3   = get_bode_curves(Kp_td3,  Ki_td3,  B, C, D)
    w_base,  mag_base,  ph_base  = get_bode_curves(Kp_base, Ki_base, B, C, D)

    fig, (ax_mag, ax_ph) = _fig(2, 1, w=12, h=9)

    # ── Magnitude ─────────────────────────────────────────────────────────
    ax_mag.semilogx(w_base, mag_base, color=RED,   lw=1.8, label="Baseline PI")
    ax_mag.semilogx(w_td3,  mag_td3,  color=GREEN, lw=1.8, label="TD3 Adaptive")
    ax_mag.axhline(0,    color=FG,     lw=0.7, ls="--", alpha=0.5, label="0 dB")
    ax_mag.axhline(-3,   color=BLUE,   lw=0.7, ls=":",  alpha=0.7, label="-3 dB (BW)")

    # Mark gain crossover for TD3
    gc = np.where(np.diff(np.sign(mag_td3)))[0]
    if len(gc):
        ax_mag.axvline(w_td3[gc[0]], color=GREEN, lw=0.8, ls=":", alpha=0.6)
        ax_mag.annotate(f"GC\n{w_td3[gc[0]]:.2f} rad/s",
                        xy=(w_td3[gc[0]], 0),
                        xytext=(w_td3[gc[0]]*1.4, 10),
                        color=GREEN, fontsize=7,
                        arrowprops=dict(arrowstyle="->", color=GREEN, lw=0.7))

    _style_ax(ax_mag, title="Bode Plot — Magnitude", ylabel="Magnitude (dB)")
    ax_mag.set_ylim(-60, 60)
    ax_mag.legend(fontsize=8, facecolor="#1A1F26", edgecolor="#333", labelcolor=FG)

    # ── Phase ──────────────────────────────────────────────────────────────
    ax_ph.semilogx(w_base, ph_base, color=RED,    lw=1.8, label="Baseline PI")
    ax_ph.semilogx(w_td3,  ph_td3,  color=GREEN,  lw=1.8, label="TD3 Adaptive")
    ax_ph.axhline(-180, color=YELLOW, lw=0.7, ls="--", alpha=0.7, label="-180° (GM)")

    # Mark phase crossover for TD3
    pc = np.where(np.diff(np.sign(ph_td3 + 180.0)))[0]
    if len(pc):
        ax_ph.axvline(w_td3[pc[0]], color=YELLOW, lw=0.8, ls=":", alpha=0.6)
        ax_ph.annotate(f"PC\n{w_td3[pc[0]]:.2f} rad/s",
                       xy=(w_td3[pc[0]], -180),
                       xytext=(w_td3[pc[0]]*1.4, -160),
                       color=YELLOW, fontsize=7,
                       arrowprops=dict(arrowstyle="->", color=YELLOW, lw=0.7))

    _style_ax(ax_ph, title="Bode Plot — Phase",
              xlabel="Frequency (rad/s)", ylabel="Phase (°)")
    ax_ph.set_ylim(-270, 0)
    ax_ph.legend(fontsize=8, facecolor="#1A1F26", edgecolor="#333", labelcolor=FG)

    fig.suptitle("Bode Plot: Open-Loop Transfer Function L(s)", fontsize=14,
                 fontweight="bold", color=YELLOW)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ─── Runner ───────────────────────────────────────────────────────────────────

def generate_all_plots(eval_results: dict, plots_dir: str = PLOTS_DIR):
    """
    Generate all 4 plots.

    eval_results: dict[profile_name] = {"td3": arrays, "baseline": arrays}
    Uses Profile A/B as incremental, E/D as decremental, F for Bode/gains.
    """
    os.makedirs(plots_dir, exist_ok=True)

    # Pick representative profiles
    incr_name = "A"   # incremental
    decr_name = "E"   # decremental
    bode_name = "F"   # large step

    incr_data = eval_results.get(incr_name, eval_results[list(eval_results.keys())[0]])
    decr_data = eval_results.get(decr_name, eval_results[list(eval_results.keys())[-1]])
    bode_data = eval_results.get(bode_name, eval_results[list(eval_results.keys())[0]])

    print("Generating Plot 1: Error comparison...")
    plot_error_comparison(incr_data, decr_data,
                          os.path.join(plots_dir, "plot1_error.png"))

    print("Generating Plot 2: Control output...")
    plot_control_output(incr_data, decr_data,
                        os.path.join(plots_dir, "plot2_output.png"))

    print("Generating Plot 3: Gain adjustment...")
    plot_gain_adjustment(bode_data["td3"],
                         os.path.join(plots_dir, "plot3_gains.png"),
                         title_suffix=" — Profile F (Large Step)")

    print("Generating Plot 4: Bode plot...")
    plot_bode(bode_data["td3"], bode_data["baseline"],
              os.path.join(plots_dir, "plot4_bode.png"))
