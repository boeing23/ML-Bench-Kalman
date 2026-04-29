"""Deterministic data generator for Task 5 (CPS fault detection).

Models a two-tank fluid system. 6 sensor channels:
    [level_1, level_2, flow_in, flow_out, temp_1, temp_2]
Each sample is a window of T=64 timesteps × 6 channels.

Training-time fault types (visible to model):
    A — stuck-at on a level sensor
    B — linear drift on a flow sensor
    C — sinusoidal oscillation on a temperature sensor

OOD fault types (held-out, judge-only):
    D — large transient spike-and-recover
    E — multiplicative gain attenuation over the window

Outputs:
    ml_bench/tasks/t05_fault_detection/data/train.npz       (visible to model)
    ml_bench/tasks/t05_fault_detection/hidden/test_id.npz   (judge only, A/B/C)
    ml_bench/tasks/t05_fault_detection/hidden/test_ood.npz  (judge only, D/E)
"""
from __future__ import annotations

import numpy as np
from pathlib import Path

T = 64           # timesteps per window
M = 6            # channels
DT = 0.1         # seconds per step
NORMAL_RNG_SEED = 7
TRAIN_RNG_SEED = 11
TEST_ID_RNG_SEED = 23
TEST_OOD_RNG_SEED = 41


def _normal_window(rng: np.random.Generator) -> np.ndarray:
    t = np.arange(T) * DT
    phase1 = rng.uniform(0, 2 * np.pi)
    phase2 = rng.uniform(0, 2 * np.pi)
    level_1 = 5.0 + 0.6 * np.sin(0.5 * t + phase1) + 0.05 * rng.standard_normal(T)
    level_2 = 4.5 + 0.5 * np.cos(0.5 * t + phase2) + 0.05 * rng.standard_normal(T)
    flow_in = 1.0 + 0.2 * np.sin(0.7 * t + phase1) + 0.03 * rng.standard_normal(T)
    flow_out = 0.95 + 0.15 * np.sin(0.7 * t + phase2 + 0.3) + 0.03 * rng.standard_normal(T)
    temp_1 = 60.0 + 0.4 * np.sin(0.1 * t) + 0.1 * rng.standard_normal(T)
    temp_2 = 58.0 + 0.4 * np.cos(0.1 * t) + 0.1 * rng.standard_normal(T)
    return np.stack([level_1, level_2, flow_in, flow_out, temp_1, temp_2], axis=1).astype(np.float32)


def _inject_fault(window: np.ndarray, fault_type: str, rng: np.random.Generator) -> np.ndarray:
    w = window.copy()
    t = np.arange(T) * DT
    if fault_type == "A":
        ch = rng.choice([0, 1])
        stuck_value = float(w[0, ch])
        w[:, ch] = stuck_value
    elif fault_type == "B":
        ch = rng.choice([2, 3])
        slope = rng.uniform(0.05, 0.15) * rng.choice([-1.0, 1.0])
        w[:, ch] = w[:, ch] + slope * t
    elif fault_type == "C":
        ch = rng.choice([4, 5])
        amp = rng.uniform(1.5, 3.0)
        freq = rng.uniform(2.0, 4.0)
        w[:, ch] = w[:, ch] + amp * np.sin(freq * t + rng.uniform(0, 2 * np.pi))
    elif fault_type == "D":
        ch = rng.choice(M)
        spike_idx = rng.integers(10, T - 15)
        spike_width = rng.integers(3, 7)
        spike_amp = rng.uniform(3.0, 6.0) * rng.choice([-1.0, 1.0])
        kernel = np.exp(-((np.arange(T) - spike_idx) ** 2) / (2 * spike_width ** 2))
        w[:, ch] = w[:, ch] + spike_amp * kernel
    elif fault_type == "E":
        ch = rng.choice(M)
        gain_end = rng.uniform(0.3, 0.6)
        ramp = np.linspace(1.0, gain_end, T)
        w[:, ch] = w[:, ch] * ramp
    else:
        raise ValueError(f"unknown fault type {fault_type}")
    return w


def _build_split(n_normal: int, n_fault: int, fault_types: list[str], seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X_list, y_list = [], []
    for _ in range(n_normal):
        X_list.append(_normal_window(rng))
        y_list.append(0)
    for i in range(n_fault):
        ftype = fault_types[i % len(fault_types)]
        base = _normal_window(rng)
        X_list.append(_inject_fault(base, ftype, rng))
        y_list.append(1)
    X = np.stack(X_list, axis=0).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)
    perm = np.random.default_rng(seed + 1).permutation(len(y))
    return X[perm], y[perm]


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    task_dir = repo_root / "ml_bench" / "tasks" / "t05_fault_detection"
    data_dir = task_dir / "data"
    hidden_dir = task_dir / "hidden"
    data_dir.mkdir(parents=True, exist_ok=True)
    hidden_dir.mkdir(parents=True, exist_ok=True)

    X_train, y_train = _build_split(1000, 1000, ["A", "B", "C"], TRAIN_RNG_SEED)
    X_test_id, y_test_id = _build_split(200, 200, ["A", "B", "C"], TEST_ID_RNG_SEED)
    X_test_ood, y_test_ood = _build_split(200, 200, ["D", "E"], TEST_OOD_RNG_SEED)

    np.savez_compressed(data_dir / "train.npz", X=X_train, y=y_train)
    np.savez_compressed(hidden_dir / "test_id.npz", X=X_test_id, y=y_test_id)
    np.savez_compressed(hidden_dir / "test_ood.npz", X=X_test_ood, y=y_test_ood)

    print(f"train:    X={X_train.shape} y={y_train.shape} pos_rate={y_train.mean():.3f}")
    print(f"test_id:  X={X_test_id.shape} y={y_test_id.shape} pos_rate={y_test_id.mean():.3f}")
    print(f"test_ood: X={X_test_ood.shape} y={y_test_ood.shape} pos_rate={y_test_ood.mean():.3f}")


if __name__ == "__main__":
    main()
