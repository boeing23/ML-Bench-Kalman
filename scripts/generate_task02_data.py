"""Deterministic data generator for Task 2 (system identification).

Models a single linear time-invariant state-space system:

    x[k+1] = A x[k] + B u[k] + w[k]   (process noise w ~ N(0, sigma_w^2 I))
    y[k]   = C x[k] + v[k]            (measurement noise v ~ N(0, sigma_v^2 I))

with p=2 inputs, q=2 outputs, n_state=3. The same A, B, C is used for ALL
trajectories — the task is to *identify* this one system from many I/O
traces, then forecast outputs on held-out trajectories given a short warmup
of measurements.

Training-time excitation (visible to model):
    PRBS-like — u[k] iid uniform in [-1, 1].

OOD excitation (held-out, judge-only):
    Sinusoidal sweep — u[k] = A * sin(2*pi*f*k + phi) with f drawn from
    a band low enough to push the system at frequencies the PRBS data
    doesn't cover well.

Outputs:
    ml_bench/tasks/t02_system_identification/data/train.npz       (visible)
        keys: U (N_tr, T, P), Y (N_tr, T, Q)
    ml_bench/tasks/t02_system_identification/hidden/test_id.npz   (judge only)
    ml_bench/tasks/t02_system_identification/hidden/test_ood.npz  (judge only)
        keys: U (N_te, T, P), Y (N_te, T, Q)
            (judge slices warmup vs. target.)
"""
from __future__ import annotations

import numpy as np
from pathlib import Path


T = 80
T_WARM = 10
P = 2
Q = 2
N_STATE = 3

N_TRAIN = 200
N_TEST_ID = 50
N_TEST_OOD = 50

SIGMA_W = 0.05
SIGMA_V = 0.05
SPECTRAL_RADIUS = 0.85

SYSTEM_SEED = 7
TRAIN_SEED = 11
TEST_ID_SEED = 23
TEST_OOD_SEED = 41


def _make_system(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((N_STATE, N_STATE))
    eigs = np.abs(np.linalg.eigvals(A))
    A = A * (SPECTRAL_RADIUS / eigs.max())
    B = rng.standard_normal((N_STATE, P)) * 0.7
    C = rng.standard_normal((Q, N_STATE)) * 0.6
    return A.astype(np.float64), B.astype(np.float64), C.astype(np.float64)


def _simulate(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    U: np.ndarray,
    sigma_w: float,
    sigma_v: float,
    rng: np.random.Generator,
) -> np.ndarray:
    N, T_sim, _ = U.shape
    n_state = A.shape[0]
    q = C.shape[0]
    Y = np.zeros((N, T_sim, q), dtype=np.float64)
    for i in range(N):
        x = rng.standard_normal(n_state) * 0.1
        for k in range(T_sim):
            v = rng.standard_normal(q) * sigma_v
            Y[i, k] = C @ x + v
            w = rng.standard_normal(n_state) * sigma_w
            x = A @ x + B @ U[i, k] + w
    return Y


def _inputs_prbs(N: int, T: int, p: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(-1.0, 1.0, size=(N, T, p)).astype(np.float64)


def _inputs_sinusoidal(N: int, T: int, p: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    U = np.zeros((N, T, p), dtype=np.float64)
    t = np.arange(T, dtype=np.float64)
    for i in range(N):
        for j in range(p):
            f = rng.uniform(0.02, 0.12)
            phi = rng.uniform(0.0, 2.0 * np.pi)
            amp = rng.uniform(0.7, 1.0)
            U[i, :, j] = amp * np.sin(2.0 * np.pi * f * t + phi)
    return U


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    task_dir = repo_root / "ml_bench" / "tasks" / "t02_system_identification"
    data_dir = task_dir / "data"
    hidden_dir = task_dir / "hidden"
    data_dir.mkdir(parents=True, exist_ok=True)
    hidden_dir.mkdir(parents=True, exist_ok=True)

    A, B, C = _make_system(SYSTEM_SEED)

    rng_tr = np.random.default_rng(TRAIN_SEED + 1000)
    U_tr = _inputs_prbs(N_TRAIN, T, P, TRAIN_SEED)
    Y_tr = _simulate(A, B, C, U_tr, SIGMA_W, SIGMA_V, rng_tr)

    rng_id = np.random.default_rng(TEST_ID_SEED + 1000)
    U_id = _inputs_prbs(N_TEST_ID, T, P, TEST_ID_SEED)
    Y_id = _simulate(A, B, C, U_id, SIGMA_W, SIGMA_V, rng_id)

    rng_ood = np.random.default_rng(TEST_OOD_SEED + 1000)
    U_ood = _inputs_sinusoidal(N_TEST_OOD, T, P, TEST_OOD_SEED)
    Y_ood = _simulate(A, B, C, U_ood, SIGMA_W, SIGMA_V, rng_ood)

    np.savez_compressed(
        data_dir / "train.npz",
        U=U_tr.astype(np.float32),
        Y=Y_tr.astype(np.float32),
    )
    np.savez_compressed(
        hidden_dir / "test_id.npz",
        U=U_id.astype(np.float32),
        Y=Y_id.astype(np.float32),
    )
    np.savez_compressed(
        hidden_dir / "test_ood.npz",
        U=U_ood.astype(np.float32),
        Y=Y_ood.astype(np.float32),
    )

    print(
        f"system: A eigvals "
        f"{np.sort(np.abs(np.linalg.eigvals(A)))[::-1].round(3).tolist()}"
    )
    print(f"train:    U={U_tr.shape} Y={Y_tr.shape} std(Y)={Y_tr.std():.3f}")
    print(f"test_id:  U={U_id.shape} Y={Y_id.shape} std(Y)={Y_id.std():.3f}")
    print(f"test_ood: U={U_ood.shape} Y={Y_ood.shape} std(Y)={Y_ood.std():.3f}")


if __name__ == "__main__":
    main()
