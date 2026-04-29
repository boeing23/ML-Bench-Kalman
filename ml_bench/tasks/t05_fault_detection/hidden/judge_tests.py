"""Behavioral + adversarial tests for Task 5.

Run by the judge harness AFTER the model has produced /workspace/solution.py.
Reads hidden test sets from this directory (judge-side only).

Outputs a single JSON line on stdout summarizing every probe + the
final continuous score.

Score formula (multiplicative gating):
    score = id_f1 * type_ok * determinism_ok * variance_ok
where each gate is 0/1 and id_f1 is in [0, 1]. OOD F1 is reported but
does not gate.
"""
from __future__ import annotations

import json
import os
import sys
import time
import importlib.util
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score


HIDDEN_DIR = Path(__file__).resolve().parent
_WS = os.environ.get("MLBENCH_WORKSPACE", "/workspace")
SOLUTION_PATH = Path(_WS) / "solution.py"
TYPE_CONTRACT_REQUIRED = True
DETERMINISM_ATOL = 1e-6
VARIANCE_PROBE_MIN_STD = 0.05
ID_F1_THRESHOLD = 0.85


def _load_solution(path: Path):
    spec = importlib.util.spec_from_file_location("solution", str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "predict"):
        raise AttributeError("solution.py must define predict(X)")
    return module.predict


def _check_type_contract(predict, X_probe: np.ndarray) -> tuple[bool, str]:
    out = predict(X_probe)
    if not isinstance(out, np.ndarray):
        return False, f"predict did not return np.ndarray (got {type(out).__name__})"
    if out.shape != (X_probe.shape[0],):
        return False, f"output shape {out.shape} != expected ({X_probe.shape[0]},)"
    if not np.issubdtype(out.dtype, np.floating):
        return False, f"output dtype {out.dtype} is not floating"
    if not np.all(np.isfinite(out)):
        return False, "output contains non-finite values"
    if out.min() < -1e-6 or out.max() > 1.0 + 1e-6:
        return False, f"outputs out of [0,1]: min={out.min()} max={out.max()}"
    return True, "ok"


def _check_determinism(predict, X_probe: np.ndarray) -> tuple[bool, str]:
    a = predict(X_probe)
    b = predict(X_probe)
    if not np.allclose(a, b, atol=DETERMINISM_ATOL):
        return False, f"non-deterministic: max|Δ|={np.abs(a - b).max():.3g}"
    return True, "ok"


def _check_variance(predict, X_probe: np.ndarray) -> tuple[bool, str]:
    out = predict(X_probe)
    s = float(np.std(out))
    if s < VARIANCE_PROBE_MIN_STD:
        return False, f"output std {s:.4f} below {VARIANCE_PROBE_MIN_STD} (constant predictor?)"
    return True, "ok"


def _f1_at(predict, X: np.ndarray, y: np.ndarray) -> float:
    proba = predict(X)
    yhat = (np.asarray(proba) >= 0.5).astype(np.int64)
    return float(f1_score(y, yhat))


def main() -> int:
    result: dict = {
        "task": "t05_fault_detection",
        "gates": {},
        "metrics": {},
        "errors": [],
    }
    t0 = time.time()

    # --- 1. Load solution
    try:
        predict = _load_solution(SOLUTION_PATH)
    except Exception as exc:
        result["errors"].append(f"load_solution: {exc!r}")
        result["score"] = 0.0
        print(json.dumps(result))
        return 1

    # --- 2. Build probe inputs
    id_blob = np.load(HIDDEN_DIR / "test_id.npz")
    ood_blob = np.load(HIDDEN_DIR / "test_ood.npz")
    X_id, y_id = id_blob["X"], id_blob["y"]
    X_ood, y_ood = ood_blob["X"], ood_blob["y"]

    # mixed batch for adversarial probes (subset of ID test, deterministic order)
    X_probe = X_id[:32].astype(np.float32)

    # --- 3. Gating probes
    try:
        type_ok, type_msg = _check_type_contract(predict, X_probe)
    except Exception as exc:
        type_ok, type_msg = False, f"exception: {exc!r}"
    result["gates"]["type_contract"] = {"ok": type_ok, "msg": type_msg}

    try:
        det_ok, det_msg = _check_determinism(predict, X_probe)
    except Exception as exc:
        det_ok, det_msg = False, f"exception: {exc!r}"
    result["gates"]["determinism"] = {"ok": det_ok, "msg": det_msg}

    try:
        var_ok, var_msg = _check_variance(predict, X_probe)
    except Exception as exc:
        var_ok, var_msg = False, f"exception: {exc!r}"
    result["gates"]["variance_probe"] = {"ok": var_ok, "msg": var_msg}

    gates_pass = bool(type_ok and det_ok and var_ok)

    # --- 4. Behavioral metrics (only if gates pass for cleanliness)
    id_f1 = ood_f1 = 0.0
    if gates_pass:
        try:
            id_f1 = _f1_at(predict, X_id, y_id)
            ood_f1 = _f1_at(predict, X_ood, y_ood)
        except Exception as exc:
            result["errors"].append(f"f1_eval: {exc!r}")
    result["metrics"]["in_distribution_f1"] = id_f1
    result["metrics"]["out_of_distribution_f1"] = ood_f1
    result["metrics"]["generalization_gap"] = id_f1 - ood_f1
    result["metrics"]["id_f1_threshold"] = ID_F1_THRESHOLD
    result["metrics"]["passes_id_threshold"] = id_f1 >= ID_F1_THRESHOLD

    # --- 5. Continuous gated score
    score = id_f1 * float(gates_pass)
    result["score"] = score
    result["wall_clock_s"] = round(time.time() - t0, 3)
    print(json.dumps(result))
    return 0 if score > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
