"""Deterministic data generator for Task 1 (Kalman filter).

Plant: 2D constant-velocity tracker (canonical Kalman teaching example).
    state x = [px, py, vx, vy]   (n = 4)
    input u = [ax, ay]           (m = 2; commanded accelerations)
    obs   y = [px, py] + noise   (p = 2; position-only, velocities are hidden)

Discrete-time dynamics (dt = 0.1):
    x[k+1] = A x[k] + B u[k] + w[k],  w ~ N(0, Q)
    y[k]   = C x[k] + v[k],            v ~ N(0, R)

A "trajectory" is T = 50 timesteps. Each split shares the same plant
matrices (A, B, C, Q) but draws fresh trajectories. The OOD probe uses
3.3x larger measurement noise (R) — the model's filter must adapt
because R is a function input.

Splits:
    data/example.npz       — 4 trajectories, σ_v = 0.3 (visible to model)
    hidden/test_id.npz     — 50 trajectories, σ_v = 0.3
    hidden/test_ood.npz    — 50 trajectories, σ_v = 1.0
"""
from __future__ import annotations

import numpy as np
from pathlib import Path

DT = 0.1
T = 50
N_TRAIN_EXAMPLE = 4
N_TEST_ID = 50
N_TEST_OOD = 50

SIGMA_V_ID = 0.3
SIGMA_V_OOD = 1.0
SIGMA_W = 0.6                  # process noise std on velocity components

EXAMPLE_SEED = 5
TEST_ID_SEED = 17
TEST_OOD_SEED = 31


def _plant():
    A = np.array([
        [1.0, 0.0,  DT, 0.0],
        [0.0, 1.0, 0.0,  DT],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    B = np.array([
        [0.0,        0.0],
        [0.0,        0.0],
        [DT,         0.0],
        [0.0,         DT],
    ])
    C = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ])
    # Process noise only on the velocity components (random accelerations
    # not captured by u). Variance for vx, vy: SIGMA_W**2 * dt.
    Q = np.diag([1e-8, 1e-8, (SIGMA_W ** 2) * DT, (SIGMA_W ** 2) * DT])
    return A, B, C, Q


def _x0():
    x0_mean = np.zeros(4)
    x0_cov = np.diag([1.0, 1.0, 0.5, 0.5])
    return x0_mean, x0_cov


def _simulate_split(N: int, sigma_v: float, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    A, B, C, Q = _plant()
    x0_mean, x0_cov = _x0()
    R = (sigma_v ** 2) * np.eye(2)

    n = A.shape[0]
    m = B.shape[1]
    p = C.shape[0]

    X_true = np.zeros((N, T, n), dtype=np.float64)
    Y = np.zeros((N, T, p), dtype=np.float64)
    U = np.zeros((N, T, m), dtype=np.float64)

    sqrt_Q = np.linalg.cholesky(Q + 1e-12 * np.eye(n))
    sqrt_x0 = np.linalg.cholesky(x0_cov + 1e-12 * np.eye(n))
    sqrt_R = np.linalg.cholesky(R)

    for i in range(N):
        x = x0_mean + sqrt_x0 @ rng.standard_normal(n)
        # Smooth random acceleration commands
        u_seq = 0.4 * rng.standard_normal((T, m))
        for k in range(T):
            U[i, k] = u_seq[k]
            Y[i, k] = C @ x + sqrt_R @ rng.standard_normal(p)
            X_true[i, k] = x
            w = sqrt_Q @ rng.standard_normal(n)
            x = A @ x + B @ u_seq[k] + w

    return {
        "A": A,
        "B": B,
        "C": C,
        "Q": Q,
        "R": R,
        "U": U,
        "Y": Y,
        "x0_mean": x0_mean,
        "x0_cov": x0_cov,
        "X_true": X_true,
    }


def _save(blob: dict, path: Path, *, include_truth: bool) -> None:
    keys = ["A", "B", "C", "Q", "R", "U", "Y", "x0_mean", "x0_cov"]
    if include_truth:
        keys = keys + ["X_true"]
    np.savez_compressed(path, **{k: blob[k] for k in keys})


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    task_dir = repo_root / "ml_bench" / "tasks" / "t01_kalman_filter"
    data_dir = task_dir / "data"
    hidden_dir = task_dir / "hidden"
    data_dir.mkdir(parents=True, exist_ok=True)
    hidden_dir.mkdir(parents=True, exist_ok=True)

    # The example file shows the model the data shapes / typical magnitudes.
    # We include X_true so the model can sanity-check its own filter on the
    # examples — this is allowed because the test sets are different
    # trajectories drawn from the same (or harder) distribution.
    example = _simulate_split(N_TRAIN_EXAMPLE, SIGMA_V_ID, EXAMPLE_SEED)
    _save(example, data_dir / "example.npz", include_truth=True)

    test_id = _simulate_split(N_TEST_ID, SIGMA_V_ID, TEST_ID_SEED)
    _save(test_id, hidden_dir / "test_id.npz", include_truth=True)

    test_ood = _simulate_split(N_TEST_OOD, SIGMA_V_OOD, TEST_OOD_SEED)
    _save(test_ood, hidden_dir / "test_ood.npz", include_truth=True)

    print(f"example:   {N_TRAIN_EXAMPLE} trajs, sigma_v={SIGMA_V_ID}, T={T}")
    print(f"test_id:   {N_TEST_ID} trajs, sigma_v={SIGMA_V_ID}")
    print(f"test_ood:  {N_TEST_OOD} trajs, sigma_v={SIGMA_V_OOD}")
    print(f"shapes: U={test_id['U'].shape} Y={test_id['Y'].shape} X_true={test_id['X_true'].shape}")


if __name__ == "__main__":
    main()
