"""Behavioral + adversarial tests for Task 2 (system identification).

Run by the judge harness AFTER the model has produced /workspace/solution.py.
Reads hidden test sets from this directory (judge-side only).

Outputs a single JSON line on stdout summarizing every probe + the
final continuous score.

Score formula (multiplicative gating):
    score = max(0, 1 - id_rmse / persistence_rmse)
            * type_ok * determinism_ok * beats_persistence_ok * id_threshold_ok
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


HIDDEN_DIR = Path(__file__).resolve().parent
_WS = os.environ.get("MLBENCH_WORKSPACE", "/workspace")
SOLUTION_PATH = Path(_WS) / "solution.py"
TRAIN_PATH = Path(_WS) / "data" / "train.npz"

T_WARM = 10
DETERMINISM_ATOL = 1e-6
BEATS_PERSISTENCE_RATIO = 0.5
ID_RMSE_THRESHOLD = 0.30


def _load_solution(path: Path):
    spec = importlib.util.spec_from_file_location("solution", str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "fit_and_predict"):
        raise AttributeError("solution.py must define fit_and_predict(...)")
    return module.fit_and_predict


def _split(U: np.ndarray, Y: np.ndarray):
    Y_warm = Y[:, :T_WARM, :]
    Y_target = Y[:, T_WARM:, :]
    return U, Y_warm, Y_target


def _persistence_pred(Y_warm: np.ndarray, horizon: int) -> np.ndarray:
    last = Y_warm[:, -1:, :]
    return np.broadcast_to(last, (Y_warm.shape[0], horizon, Y_warm.shape[2])).copy()


def _rmse(Y_pred: np.ndarray, Y_target: np.ndarray) -> float:
    return float(np.sqrt(np.mean((Y_pred - Y_target) ** 2)))


def _check_type_contract(Y_pred, expected_shape) -> tuple[bool, str]:
    if not isinstance(Y_pred, np.ndarray):
        return False, f"return type {type(Y_pred).__name__} != np.ndarray"
    if Y_pred.shape != expected_shape:
        return False, f"shape {Y_pred.shape} != expected {expected_shape}"
    if not np.issubdtype(Y_pred.dtype, np.floating):
        return False, f"dtype {Y_pred.dtype} not floating"
    if not np.all(np.isfinite(Y_pred)):
        return False, "non-finite values in output"
    return True, "ok"


def main() -> int:
    result: dict = {
        "task": "t02_system_identification",
        "gates": {},
        "metrics": {},
        "errors": [],
    }
    t0 = time.time()

    try:
        fit_and_predict = _load_solution(SOLUTION_PATH)
    except Exception as exc:
        result["errors"].append(f"load_solution: {exc!r}")
        result["score"] = 0.0
        print(json.dumps(result))
        return 1

    try:
        train_blob = np.load(TRAIN_PATH)
        U_train, Y_train = train_blob["U"], train_blob["Y"]
    except Exception as exc:
        result["errors"].append(f"load_train: {exc!r}")
        result["score"] = 0.0
        print(json.dumps(result))
        return 1

    id_blob = np.load(HIDDEN_DIR / "test_id.npz")
    ood_blob = np.load(HIDDEN_DIR / "test_ood.npz")
    U_id, Y_warm_id, Y_tgt_id = _split(id_blob["U"], id_blob["Y"])
    U_ood, Y_warm_ood, Y_tgt_ood = _split(ood_blob["U"], ood_blob["Y"])

    expected_shape_id = Y_tgt_id.shape

    try:
        Y_pred_id_a = fit_and_predict(U_train, Y_train, U_id, Y_warm_id)
    except Exception as exc:
        result["errors"].append(f"first_call: {exc!r}")
        result["score"] = 0.0
        print(json.dumps(result))
        return 1

    type_ok, type_msg = _check_type_contract(Y_pred_id_a, expected_shape_id)
    result["gates"]["type_contract"] = {"ok": type_ok, "msg": type_msg}

    if type_ok:
        try:
            Y_pred_id_b = fit_and_predict(U_train, Y_train, U_id, Y_warm_id)
            det_ok = bool(np.allclose(Y_pred_id_a, Y_pred_id_b, atol=DETERMINISM_ATOL))
            det_msg = (
                "ok" if det_ok
                else f"non-deterministic: max|Δ|={np.abs(Y_pred_id_a - Y_pred_id_b).max():.3g}"
            )
        except Exception as exc:
            det_ok, det_msg = False, f"exception: {exc!r}"
    else:
        det_ok, det_msg = False, "skipped (type contract failed)"
    result["gates"]["determinism"] = {"ok": det_ok, "msg": det_msg}

    horizon = Y_tgt_id.shape[1]
    persistence_id = _persistence_pred(Y_warm_id, horizon)
    persistence_rmse_id = _rmse(persistence_id, Y_tgt_id)

    if type_ok:
        id_rmse = _rmse(Y_pred_id_a, Y_tgt_id)
    else:
        id_rmse = float("inf")

    beats_persistence_ok = id_rmse < BEATS_PERSISTENCE_RATIO * persistence_rmse_id
    result["gates"]["beats_persistence"] = {
        "ok": bool(beats_persistence_ok),
        "msg": (
            f"id_rmse={id_rmse:.4f} vs persistence_rmse={persistence_rmse_id:.4f} "
            f"(needed < {BEATS_PERSISTENCE_RATIO:.2f}×)"
        ),
    }

    id_threshold_ok = id_rmse <= ID_RMSE_THRESHOLD
    result["gates"]["id_rmse_threshold"] = {
        "ok": bool(id_threshold_ok),
        "msg": f"id_rmse={id_rmse:.4f} (threshold ≤ {ID_RMSE_THRESHOLD})",
    }

    ood_rmse = float("nan")
    if type_ok:
        try:
            Y_pred_ood = fit_and_predict(U_train, Y_train, U_ood, Y_warm_ood)
            type_ok_ood, _ = _check_type_contract(Y_pred_ood, Y_tgt_ood.shape)
            if type_ok_ood:
                ood_rmse = _rmse(Y_pred_ood, Y_tgt_ood)
        except Exception as exc:
            result["errors"].append(f"ood_call: {exc!r}")

    persistence_rmse_ood = _rmse(_persistence_pred(Y_warm_ood, horizon), Y_tgt_ood)

    result["metrics"]["in_distribution_rmse"] = id_rmse if np.isfinite(id_rmse) else None
    result["metrics"]["out_of_distribution_rmse"] = ood_rmse if np.isfinite(ood_rmse) else None
    result["metrics"]["persistence_rmse_id"] = persistence_rmse_id
    result["metrics"]["persistence_rmse_ood"] = persistence_rmse_ood
    if np.isfinite(id_rmse) and np.isfinite(ood_rmse):
        result["metrics"]["generalization_gap"] = ood_rmse - id_rmse
    result["metrics"]["id_rmse_threshold"] = ID_RMSE_THRESHOLD

    gates_pass = bool(type_ok and det_ok and beats_persistence_ok and id_threshold_ok)
    if gates_pass and np.isfinite(id_rmse):
        score = max(0.0, 1.0 - id_rmse / persistence_rmse_id)
    else:
        score = 0.0
    result["score"] = float(score)
    result["wall_clock_s"] = round(time.time() - t0, 3)
    print(json.dumps(result))
    return 0 if score > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
