"""
plant_simulator.py
==================
ODE-based plant model G(s) = B / (s^2 + Cs + D) and PI controller.
Pure Python / SciPy — no MATLAB required.

State-space:
  x1' = x2
  x2' = B*u - D*x1 - C*x2
  y   = x1

PI Controller:
  e(t) = setpoint - y(t)
  u(t) = Kp * e(t) + Ki * integral(e)
"""

import numpy as np
from scipy.integrate import solve_ivp


# ─── Default plant & controller parameters ───────────────────────────────────
DEFAULT_B  = 1.0
DEFAULT_C  = 1.5
DEFAULT_D  = 2.0
KP_FIXED   = 2.0
KI_FIXED   = 0.5
SETPOINT   = 900.0


class PlantSimulator:
    """
    Simulates the second-order plant with a PI controller.

    Parameters
    ----------
    B, C, D : plant transfer function coefficients
    Kp, Ki  : proportional and integral gains
    setpoint: reference setpoint
    dt      : simulation timestep (seconds)
    """

    def __init__(
        self,
        B: float = DEFAULT_B,
        C: float = DEFAULT_C,
        D: float = DEFAULT_D,
        Kp: float = KP_FIXED,
        Ki: float = KI_FIXED,
        setpoint: float = SETPOINT,
        dt: float = 0.01,
    ):
        self.B = B
        self.C = C
        self.D = D
        self.Kp = Kp
        self.Ki = Ki
        self.setpoint = setpoint
        self.dt = dt
        self.reset()

    # ── State management ─────────────────────────────────────────────────────
    def reset(self):
        """Reset state to zero initial conditions."""
        self.x1 = 0.0   # output y = x1
        self.x2 = 0.0   # derivative of y
        self.integral_error = 0.0
        self.t = 0.0

    @property
    def output(self) -> float:
        return self.x1

    @property
    def error(self) -> float:
        return self.setpoint - self.x1

    # ── ODE right-hand side ──────────────────────────────────────────────────
    def _ode_rhs(self, t, state, u):
        """Plant ODE: ẋ₁=x₂,  ẋ₂=B·u - D·x₁ - C·x₂"""
        x1, x2 = state
        dx1 = x2
        dx2 = self.B * u - self.D * x1 - self.C * x2
        return [dx1, dx2]

    # ── Single step ──────────────────────────────────────────────────────────
    def step(self, Kp: float = None, Ki: float = None, disturbance: float = 0.0):
        """
        Advance simulation by one timestep dt.

        The disturbance A modifies the effective plant gain experienced by the
        controller (multiplicative load on the plant input channel):
            u_eff = u * (1 - disturbance)

        Parameters
        ----------
        Kp, Ki    : gains to use this step (defaults to self.Kp / self.Ki)
        disturbance: A ∈ [0,1], load disturbance

        Returns
        -------
        dict with keys: time, output, error, control_output
        """
        Kp = Kp if Kp is not None else self.Kp
        Ki = Ki if Ki is not None else self.Ki

        e = self.error
        self.integral_error += e * self.dt
        u = Kp * e + Ki * self.integral_error

        # Disturbance reduces effective actuator authority
        u_eff = u * (1.0 - 0.5 * disturbance)

        sol = solve_ivp(
            self._ode_rhs,
            [self.t, self.t + self.dt],
            [self.x1, self.x2],
            args=(u_eff,),
            method="RK45",
            max_step=self.dt,
        )
        self.x1, self.x2 = sol.y[:, -1]
        self.t += self.dt

        return {
            "time": self.t,
            "output": self.x1,
            "error": self.setpoint - self.x1,
            "control_output": u,
        }


# ─── Batch simulation (for dataset generation) ───────────────────────────────
def simulate_profile(
    disturbance_fn,
    duration: float = 1000.0,
    dt: float = 0.01,
    B: float = DEFAULT_B,
    C: float = DEFAULT_C,
    D: float = DEFAULT_D,
    Kp: float = KP_FIXED,
    Ki: float = KI_FIXED,
    setpoint: float = SETPOINT,
    alpha_fn=None,
    beta_fn=None,
) -> dict:
    """
    Run a full simulation with a disturbance profile.

    Parameters
    ----------
    disturbance_fn : callable(t) → A ∈ [0,1]
    alpha_fn       : callable(error, A) → alpha (Kp correction); None = 1.0
    beta_fn        : callable(error, A) → beta  (Ki correction); None = 1.0

    Returns
    -------
    dict of numpy arrays: time, setpoint, actual_value, error,
                          disturbance_A, Kp_applied, Ki_applied, control_output
    """
    sim = PlantSimulator(B=B, C=C, D=D, setpoint=setpoint, dt=dt)
    n_steps = int(duration / dt)

    arrays = {k: np.zeros(n_steps) for k in [
        "time", "setpoint", "actual_value", "error",
        "disturbance_A", "Kp_applied", "Ki_applied", "control_output"
    ]}

    for i in range(n_steps):
        t_now = i * dt
        A = float(np.clip(disturbance_fn(t_now), 0.0, 1.0))
        e = sim.error

        # Compute correction factors (ONLY use error and disturbance_A)
        alpha = alpha_fn(e, A) if alpha_fn else 1.0
        beta  = beta_fn(e, A)  if beta_fn  else 1.0

        Kp_final = Kp * alpha
        Ki_final = Ki * beta

        result = sim.step(Kp=Kp_final, Ki=Ki_final, disturbance=A)

        arrays["time"][i]           = t_now
        arrays["setpoint"][i]       = setpoint
        arrays["actual_value"][i]   = result["output"]
        arrays["error"][i]          = result["error"]
        arrays["disturbance_A"][i]  = A
        arrays["Kp_applied"][i]     = Kp_final
        arrays["Ki_applied"][i]     = Ki_final
        arrays["control_output"][i] = result["control_output"]

    return arrays


# ─── Performance metrics ─────────────────────────────────────────────────────
def compute_performance(time: np.ndarray, output: np.ndarray,
                        setpoint: float, tol_settle=0.01,
                        tol_ss=0.0025) -> dict:
    """
    Compute O%, U%, settling time T, steady-state time SS, and ITAE.
    """
    error = setpoint - output
    rel_error = np.abs(error) / setpoint

    # Overshoot and undershoot
    overshoot_pct  = max(0.0, (np.max(output) - setpoint) / setpoint * 100)
    undershoot_pct = max(0.0, (setpoint - np.min(output)) / setpoint * 100)

    # Settling time: last time error exceeds tol_settle
    exceed_settle = np.where(rel_error > tol_settle)[0]
    settling_time = time[exceed_settle[-1]] if len(exceed_settle) > 0 else 0.0

    # Steady-state time
    exceed_ss = np.where(rel_error > tol_ss)[0]
    ss_time = time[exceed_ss[-1]] if len(exceed_ss) > 0 else 0.0

    # ITAE
    itae = float(np.trapz(time * np.abs(error), time))

    return {
        "overshoot_pct":  overshoot_pct,
        "undershoot_pct": undershoot_pct,
        "settling_time":  settling_time,
        "ss_time":        ss_time,
        "itae":           itae,
    }


if __name__ == "__main__":
    # Quick sanity check
    import matplotlib.pyplot as plt

    def const_dist(t):
        return 0.5

    result = simulate_profile(const_dist, duration=50.0)
    plt.figure()
    plt.plot(result["time"], result["actual_value"], label="Output")
    plt.axhline(SETPOINT, color="r", linestyle="--", label="Setpoint")
    plt.xlabel("Time (s)")
    plt.ylabel("RPM")
    plt.title("Sanity check: constant disturbance A=0.5")
    plt.legend()
    plt.tight_layout()
    plt.show()
    m = compute_performance(result["time"], result["actual_value"], SETPOINT)
    print(m)
