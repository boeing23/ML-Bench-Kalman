# Task 2 — System Identification of a Linear State-Space System

You are an ML engineer working on a cyber-physical system. A two-input,
two-output linear time-invariant plant is governed by an unknown
state-space model:

```
x[k+1] = A x[k] + B u[k] + w[k]    (process noise)
y[k]   = C x[k]            + v[k]  (measurement noise)
```

with `dim(x) = 3`, `dim(u) = 2`, `dim(y) = 2`. The plant is **the same**
across every trajectory in this task; you have many input-output traces of
that single plant. Your job is to identify the dynamics from the training
traces and use the identified model to **forecast** outputs on held-out
trajectories given a short warmup of measurements.

## Data

`<workspace>/data/train.npz` contains the arrays below. The workspace path
is exposed via the `MLBENCH_WORKSPACE` environment variable; fall back to
`/workspace` if it is unset:

```python
import os
from pathlib import Path
WORKSPACE = Path(os.environ.get("MLBENCH_WORKSPACE", "/workspace"))
TRAIN = WORKSPACE / "data" / "train.npz"
```


| key | shape                | dtype   | meaning                              |
|-----|----------------------|---------|--------------------------------------|
| `U` | `(200, 80, 2)`       | float32 | input signals u[k] for k=0..79       |
| `Y` | `(200, 80, 2)`       | float32 | measured outputs y[k] for k=0..79    |

The training inputs are uniformly random in `[-1, 1]` (PRBS-like).

## Your task

Write your solution to `<workspace>/solution.py`. The file **must** define a
top-level callable:

```python
def fit_and_predict(
    U_train: np.ndarray,         # (N_tr, T, 2)  — training inputs
    Y_train: np.ndarray,         # (N_tr, T, 2)  — training outputs
    U_test: np.ndarray,          # (N_te, T, 2)  — full test inputs
    Y_test_warmup: np.ndarray,   # (N_te, T_warm, 2) — first T_warm test outputs
) -> np.ndarray:
    """
    Fit a model of the plant from the training I/O traces, then for each test
    trajectory forecast the outputs y[T_warm], y[T_warm+1], ..., y[T-1]
    given the full test input sequence and the first T_warm measured outputs.

    Returns Y_pred of shape (N_te, T - T_warm, 2), float dtype, finite values.
    The k-th entry along axis=1 is the predicted y[T_warm + k].
    """
```

Constants for this task: `T = 80`, `T_warm = 10`, `P = Q = 2`. The judge
will pass `U_test` and `Y_test_warmup` consistent with these shapes.

You may import `numpy`, `scipy`, `sklearn`, `pandas`, `torch`. The judge
calls `fit_and_predict` once with `N_te = 50`. Total wall-clock for
`import solution` plus that one call must stay under **120 seconds** on CPU.

## Pass criteria

The judge runs the following and gates each:

1. **Type contract** — `fit_and_predict(...)` returns `np.ndarray` of shape
   `(N_te, T - T_warm, 2)` with floating dtype and all finite values.
2. **Determinism** — called twice on the same inputs returns identical
   output (`atol = 1e-6`).
3. **Beats persistence** — your forecast RMSE on the in-distribution test
   set must be **strictly less than 0.5 ×** the RMSE of the persistence
   baseline (predict `y[T_warm-1]` for every future step). This rules out
   trivial constant / echo predictors.
4. **In-distribution RMSE ≤ 0.30** on a held-out test set drawn from the
   same input distribution as `train.npz`.

A separate **out-of-distribution** test set uses a *different* input
excitation (low-frequency sinusoidal sweeps instead of PRBS) on the same
plant. Its RMSE is logged but does not gate the score. The gap between
ID and OOD RMSE is reported as the headline generalization metric.

The continuous score is

```
score = max(0, 1 - id_rmse / persistence_rmse) × (all gates pass)
```

so a perfect predictor scores 1, and any predictor as bad as persistence
scores 0.

## Notes

- The plant is linear and time-invariant — classical system-identification
  techniques (subspace ID, ARX least squares, prediction-error methods)
  work very well here. A small neural net is overkill and risks overfitting
  the input distribution, which the OOD probe will surface.
- Process and measurement noise are both small (σ ≈ 0.05). The achievable
  RMSE floor is roughly the measurement-noise standard deviation for
  short-horizon forecasts and grows mildly with horizon.
- Do not attempt to read any file outside `WORKSPACE / "data"`.
- Do not call out to a network.
