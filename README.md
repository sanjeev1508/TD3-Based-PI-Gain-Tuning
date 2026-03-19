# TD3-Based PI Gain Tuning

> **Dynamic gain correction for PI controllers using Twin Delayed Deep Deterministic Policy Gradient (TD3) Reinforcement Learning — pure Python, no MATLAB required.**

---

## System Architecture

![Control Loop Design](architecture/control_loop.png)

> **TD3 observes only `[e(t), A(t)]` → outputs `[α, β]` → scales fixed Kp/Ki gains. Plant parameters B, C, D are never exposed to the agent.**

---

## Problem Statement

Traditional PI controllers require manual tuning of Kp and Ki gains, which is suboptimal when system dynamics change. This project implements an ML-based solution that dynamically adjusts PI gains to minimize error and improve stability under varying load disturbances.

**Plant Model:** `G(s) = B / (s² + Cs + D)` — hydraulic actuator (2nd-order)

**Critical Constraint:**
```
Kp_corr = f(error, Disturbance_A)   ← ONLY these two inputs
Ki_corr = f(error, Disturbance_A)   ← ONLY these two inputs
Kp_final = Kp_fixed × Kp_corr
Ki_final = Ki_fixed × Ki_corr
```
The correction factors must **never** use plant parameters B, C, D directly.

---

## Results

| Metric | Baseline PI | TD3 Adaptive | Improvement |
|--------|-------------|--------------|-------------|
| Overshoot % | 33.72% | 21.97% | ↓ 35% ✓ |
| Settling Time (s) | 427.95 | 420.19 | ↓ 2% ✓ |
| ITAE | 940,819 | 597,404 | ↓ 37% ✓ |
| Bandwidth (rad/s) | 0.0142 | 0.0141 | — |
| Phase Margin (°) | 84.91 | 50.07 | Stable ✓ |

---

## Error Response: Baseline PI vs TD3 Adaptive

![Error Comparison — Baseline PI vs TD3 Adaptive PI](plots/plot1_error.png)

> Left: incremental disturbance (A: 0→1) &nbsp;|&nbsp; Right: decremental disturbance (A: 1→0).  
> TD3 (green) tracks the setpoint significantly tighter than the fixed-gain baseline (red). Dashed yellow lines mark the ±15% requirement envelope.

---

## Project Structure

```
caterpillar_pi_td3/
├── envs/
│   └── pi_control_env.py       ← Custom Gymnasium environment
├── simulation/
│   └── plant_simulator.py      ← ODE plant model + PI controller
├── data/
│   ├── generate_profiles.py    ← 6 disturbance profile generators
│   ├── data_profile_A.csv      ← 5% step-sheds (triangular)
│   ├── data_profile_B.csv      ← 10% step-sheds
│   ├── data_profile_C.csv      ← 15% step-sheds
│   ├── data_profile_D.csv      ← 20% step-sheds
│   ├── data_profile_E.csv      ← 30% step-sheds
│   └── data_profile_F.csv      ← Large step: 0.85→0.25
├── training/
│   └── train_td3.py            ← TD3 training (SB3)
├── evaluation/
│   ├── run_evaluation.py       ← Closed-loop evaluation
│   ├── compute_metrics.py      ← Time + frequency domain metrics
│   └── metrics_table.csv       ← Full results table
├── plots/
│   ├── plot1_error.png         ← Error comparison (baseline vs TD3)
│   ├── plot2_output.png        ← Control output u(t)
│   ├── plot3_gains.png         ← α(t) and β(t) gain adjustment
│   └── plot4_bode.png          ← Bode plot of open-loop L(s)
├── models/
│   └── best_model.zip          ← Best saved TD3 model (EvalCallback)
├── main.py                     ← End-to-end pipeline runner
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run full pipeline (generate → train → evaluate → plot)
```bash
python main.py
```

### 3. CLI flags
```bash
# Force retrain even if model exists
python main.py --train

# Skip training, load existing model
python main.py --eval-only

# Evaluate only Profile F (large step test case)
python main.py --profile F

# Skip simulation, regenerate plots from saved CSVs
python main.py --plot-only

# Custom training budget
python main.py --timesteps 500000
```

---

## Algorithm: TD3

**Twin Delayed Deep Deterministic Policy Gradient**

| Hyperparameter | Value |
|---|---|
| Policy | MlpPolicy [400, 300] |
| Learning Rate | 3e-4 |
| Buffer Size | 300,000 |
| Batch Size | 256 |
| Policy Delay | 2 |
| Target Noise | 0.2 (clip: 0.5) |
| τ (soft update) | 0.005 |
| γ (discount) | 0.99 |
| Total Timesteps | 300,000 |

**Why TD3?**
- Twin critics prevent Q-value overestimation under load spikes
- Delayed policy updates every 2 critic steps → stable training
- Target policy smoothing noise → prevents exploiting sharp Q-peaks

### Reward Function
```python
reward = (
  -abs(e) / setpoint                               # error penalty
  - 2.0 * max(0, overshoot - 0.15)                 # overshoot penalty
  - 2.0 * max(0, undershoot - 0.15)                # undershoot penalty
  + 1.0 * (settled_for_30_steps)                   # settle bonus
  - 0.1 * |Δα| - 0.1 * |Δβ|                        # smoothness penalty
)
```

---

## Disturbance Profiles

| Profile | Type | Step Size | Duration |
|---------|------|-----------|----------|
| A | Triangular ramp | 5% | 1000s |
| B | Step-shed | 10% | 1000s |
| C | Step-shed | 15% | 1000s |
| D | Step-shed | 20% | 1000s |
| E | Step-shed | 30% | 1000s |
| **F** | **Large step** | **0.85 → 0.25** | **1000s** |

Profile F is the **primary evaluation test case** per the problem statement.

---

## Plant Parameters (Default)

| Parameter | Value | Description |
|---|---|---|
| B | 1.0 | Numerator gain |
| C | 1.5 | Damping coefficient |
| D | 2.0 | Natural frequency² |
| Kp_fixed | 2.0 | Fixed base proportional gain |
| Ki_fixed | 0.5 | Fixed base integral gain |
| Setpoint | 900 RPM | Fixed reference |
| dt | 10 ms | Simulation timestep |

---

## Generated Plots

| Plot | Description |
|------|-------------|
| `plot1_error.png` | Error e(t) — Baseline vs TD3, incremental & decremental disturbance |
| `plot2_output.png` | Control output u(t) — actuator effort comparison |
| `plot3_gains.png` | α(t) and β(t) — real-time gain correction factors |
| `plot4_bode.png` | Open-loop Bode plot with bandwidth, gain & phase margins |

---

## Dependencies

```
numpy>=1.24
scipy>=1.10
matplotlib>=3.7
gymnasium>=0.29
stable-baselines3>=2.0
pandas>=2.0
```

No MATLAB required. Pure Python implementation using `scipy.integrate.solve_ivp`.

---

*Caterpillar Tech Challenge 2026 — Control Systems + ML Research*
