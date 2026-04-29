# Task 5 — Fault Detection on a Two-Tank Sensor Stream

You are an ML engineer working on a cyber-physical monitoring system. A two-tank fluid
process is instrumented with six sensors sampled at 10 Hz:

```
channel 0  level_1   (m)
channel 1  level_2   (m)
channel 2  flow_in   (m^3/s)
channel 3  flow_out  (m^3/s)
channel 4  temp_1    (°C)
channel 5  temp_2    (°C)
```

Each example is a window of `T = 64` consecutive timesteps × 6 channels (shape
`(64, 6)`, dtype `float32`). Labels are binary: `0 = nominal`, `1 = fault`.

## Your task

Train a binary classifier that detects whether a window contains a sensor fault.
The training set is at `<workspace>/data/train.npz` with arrays `X` of shape
`(2000, 64, 6)` and `y` of shape `(2000,)`.

The workspace path is exposed via the `MLBENCH_WORKSPACE` environment variable;
fall back to `/workspace` if it is unset. Use this pattern at the top of your
solution to locate `train.npz` portably:

```python
import os
from pathlib import Path
WORKSPACE = Path(os.environ.get("MLBENCH_WORKSPACE", "/workspace"))
DATA = WORKSPACE / "data" / "train.npz"
```

Write your solution to `<workspace>/solution.py`. The file **must** define a
top-level callable:

```python
def predict(X: np.ndarray) -> np.ndarray:
    """
    X: float32 array of shape (N, 64, 6).
    Returns: float array of shape (N,) with predicted P(fault) in [0, 1].
    """
```

You may import `numpy`, `scipy`, `sklearn`, `pandas`, `torch`. You may train
inside `solution.py` at import time, save artifacts under the workspace path
(`WORKSPACE / "model.pkl"` etc.), or load them inside `predict`. Total
wall-clock for `import solution` plus one call to `predict(X)` on `N = 400`
must be under **120 seconds** on CPU.

## Pass criteria

The judge runs the following and gates each:

1. **Type contract** — `predict(X)` returns `np.ndarray`, shape `(N,)`, dtype
   floating, all values in `[0, 1]`.
2. **Determinism** — `predict(X)` called twice on the same input returns
   identical output (`atol=1e-6`).
3. **Variance probe** — on a mixed input batch, `std(predict(X)) > 0.05`
   (rules out constant predictors).
4. **In-distribution F1 ≥ 0.85** on a held-out test set drawn from the same
   distribution as `train.npz`.

A separate **out-of-distribution probe** (different fault types you have not
seen in training) is also evaluated and reported, but is not required for
pass. The gap between in-distribution and OOD F1 is logged.

## Notes

- Do not attempt to read any file outside `WORKSPACE / "data"`. The judge's
  test data is not present in your sandbox.
- Do not call out to a network.
- Inside the time budget, simpler classical pipelines (hand-crafted features +
  logistic regression / gradient boosting) tend to be competitive with small
  neural networks on this input size.
