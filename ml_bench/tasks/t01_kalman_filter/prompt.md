# Task 1 — Kalman Filter for a 2D Constant-Velocity Tracker

You are estimating the hidden state of a 2D moving target. The plant is
**known**: a constant-velocity tracker with commanded acceleration inputs and
position-only measurements.

```
Discrete-time state-space (dt = 0.1):

    state    x[k] = [px, py, vx, vy]    (n = 4)
    input    u[k] = [ax, ay]            (m = 2; commanded accelerations)
    measure  y[k] = [px, py] + v[k]     (p = 2; positions only)

    x[k+1] = A x[k] + B u[k] + w[k],   w[k] ~ N(0, Q)
    y[k]   = C x[k] + v[k],             v[k] ~ N(0, R)
```

Velocities are **not** observed; you must infer them from the position stream
plus the dynamics model.

## Your task

Write `<workspace>/solution.py` defining the function below. The workspace
path is exposed via the `MLBENCH_WORKSPACE` environment variable; fall back
to `/workspace` if it is unset. The judge will import your `solution.py`
and call `estimate` directly — you do not need to load any files yourself
unless you want to inspect the example trajectories.

```python
def estimate(
    A: np.ndarray,         # (n, n)
    B: np.ndarray,         # (n, m)
    C: np.ndarray,         # (p, n)
    Q: np.ndarray,         # (n, n) process noise covariance
    R: np.ndarray,         # (p, p) measurement noise covariance
    U: np.ndarray,         # (N, T, m) commanded inputs across N trajectories
    Y: np.ndarray,         # (N, T, p) noisy position measurements
    x0_mean: np.ndarray,   # (n,)     prior mean for x[0]
    x0_cov:  np.ndarray,   # (n, n)   prior covariance for x[0]
) -> np.ndarray:           # (N, T, n) filtered state estimates x_hat[k|k]
    ...
```

Return the **filtered** estimate `x_hat[k | y[0..k], u[0..k-1]]` for every
trajectory `i` and every timestep `k = 0..T-1`. The judge will score per-element
RMSE between `x_hat` and the (hidden) ground-truth state.

## What's in the workspace

`<workspace>/data/example.npz` contains four example trajectories so you can
inspect data shapes and typical magnitudes. To resolve the workspace path
portably:

```python
import os
from pathlib import Path
WORKSPACE = Path(os.environ.get("MLBENCH_WORKSPACE", "/workspace"))
EXAMPLE = WORKSPACE / "data" / "example.npz"
```

The keys are:

```
A, B, C, Q, R           plant + noise covariances
U, Y                    inputs + measurements   (4, 50, m / p)
x0_mean, x0_cov         prior on initial state
X_true                  ground truth state      (4, 50, 4)   ← only in example
```

The judge's hidden test sets have the **same plant matrices** but draw fresh
trajectories and use a different `R`. Your `estimate(...)` must adapt because
`R` is passed as an argument.

## Pass criteria (multiplicative gating)

The judge gates each item below; failing any one zeros the final score:

1. **Type contract** — return type `np.ndarray`, shape `(N, T, 4)`, dtype
   floating, all entries finite.
2. **Determinism** — calling `estimate(...)` twice on the same inputs returns
   identical output (`atol = 1e-6`).
3. **Variance probe** — on a mixed input batch, the output is not constant
   (per-component std summed across axes > `0.05`).
4. **Beats persistence baseline** — RMSE on the in-distribution test set is
   below `0.65 × baseline`, where `baseline` is `[Y_x, Y_y, 0, 0]` (i.e.,
   echoing measurements and assuming zero velocity).
5. **Absolute RMSE bound** — RMSE on the in-distribution test set is `≤ 0.55`.

The continuous score is `max(0, 1 − rmse_id / baseline_rmse)` if all gates
pass, else 0.

A separate **out-of-distribution probe** is evaluated on trajectories with
elevated measurement noise (3.3× larger σ_v) and the resulting RMSE is
reported alongside the ID RMSE. The OOD result does not gate the score; the
ID/OOD generalization gap is logged.

## Notes

- A correctly implemented Kalman filter is the optimal estimator for this
  linear-Gaussian setup. Hand-rolled smoothing of `Y` will pass the gates
  on positions but will fail on velocities, dragging the overall RMSE above
  threshold.
- Do not call out to a network. Total wall-clock for `import solution` plus
  one call to `estimate(...)` on `N = 50` must be under 60 seconds on CPU.
