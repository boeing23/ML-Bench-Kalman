"""Behavioral + adversarial tests for Task 1 (Kalman filter).

Run by the judge harness AFTER the model has produced /workspace/solution.py.

Score formula (multiplicative gating):
    behavioral = max(0, 1 - id_rmse / persistence_rmse)
    score      = behavioral
                 * type_ok * determinism_ok * variance_ok
                 * beats_persistence_ok * absolute_rmse_ok
"""
from __future__ import annotations

import json
import os
import sys
import time
import importlib.util
from pathlib import Path
import numpy as np


HIDDEN_DIR = Path(__file__).resolve().parent
_WS = os.environ.get("MLBENCH_WORKSPACE", "/workspace")
SOLUTION_PATH = Path(_WS) / "solution.py"

DETERMINISM_ATOL = 1e-6
VARIANCE_PROBE_MIN_STD = 0.05
BEATS_PERSISTENCE_RATIO = 0.65
ABSOLUTE_RMSE_BOUND = 0.55


def _load_solution(path: Path):
    spec = importlib.util.spec_from_file_location("solution", str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "estimate"):
        raise AttributeError("solution.py must define estimate(...)")
    return module.estimate


def _persistence_baseline(Y: np.ndarray, n: int) -> np.ndarray:
    """x_hat[k] = [Y[k, 0], Y[k, 1], 0, ..., 0]"""
    N, T, p = Y.shape
    out = np.zeros((N, T, n), dtype=np.float64)
    out[..., :p] = Y
    return out


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def _call(estimate, blob: dict) -> np.ndarray:
    return np.asarray(
        estimate(
            blob["A"], blob["B"], blob["C"], blob["Q"], blob["R"],
            blob["U"], blob["Y"], blob["x0_mean"], blob["x0_cov"],
        )
    )


def _check_type_contract(estimate, blob: dict) -> tuple[bool, str, np.ndarray | None]:
    try:
        out = _call(estimate, blob)
    except Exception as exc:
        return False, f"exception: {exc!r}", None
    if not isinstance(out, np.ndarray):
        return False, f"not np.ndarray (got {type(out).__name__})", None
    expected = (blob["Y"].shape[0], blob["Y"].shape[1], blob["A"].shape[0])
    if out.shape != expected:
        return False, f"shape {out.shape} != expected {expected}", None
    if not np.issubdtype(out.dtype, np.floating):
        return False, f"dtype {out.dtype} not floating", None
    if not np.all(np.isfinite(out)):
        return False, "output contains non-finite values", None
    return True, "ok", out


def _check_determinism(estimate, blob: dict) -> tuple[bool, str]:
    try:
        a = _call(estimate, blob)
        b = _call(estimate, blob)
    except Exception as exc:
        return False, f"exception: {exc!r}"
    if not np.allclose(a, b, atol=DETERMINISM_ATOL):
        return False, f"non-deterministic: max|Δ|={np.abs(a - b).max():.3g}"
    return True, "ok"


def _check_variance(out: np.ndarray) -> tuple[bool, str]:
    # Sum of per-component std across (N, T) — guards against constant outputs
    s = float(out.std(axis=(0, 1)).sum())
    if s < VARIANCE_PROBE_MIN_STD:
        return False, f"sum-of-component std {s:.4f} below {VARIANCE_PROBE_MIN_STD}"
    return True, "ok"


def main() -> int:
    result: dict = {
        "task": "t01_kalman_filter",
        "gates": {},
        "metrics": {},
        "errors": [],
    }
    t0 = time.time()

    # --- 1. Load solution
    try:
        estimate = _load_solution(SOLUTION_PATH)
    except Exception as exc:
        result["errors"].append(f"load_solution: {exc!r}")
        result["score"] = 0.0
        print(json.dumps(result))
        return 1

    # --- 2. Load splits
    id_blob = dict(np.load(HIDDEN_DIR / "test_id.npz"))
    ood_blob = dict(np.load(HIDDEN_DIR / "test_ood.npz"))
    n = int(id_blob["A"].shape[0])

    # --- 3. Type contract on a small probe slice
    probe_blob = {k: v for k, v in id_blob.items() if k != "X_true"}
    for arr_key in ("U", "Y"):
        probe_blob[arr_key] = id_blob[arr_key][:8]
    type_ok, type_msg, probe_out = _check_type_contract(estimate, probe_blob)
    result["gates"]["type_contract"] = {"ok": type_ok, "msg": type_msg}

    # --- 4. Determinism on the same probe
    det_ok, det_msg = _check_determinism(estimate, probe_blob)
    result["gates"]["determinism"] = {"ok": det_ok, "msg": det_msg}

    # --- 5. Variance probe (uses the type-contract output if available)
    if probe_out is not None:
        var_ok, var_msg = _check_variance(probe_out)
    else:
        var_ok, var_msg = False, "no output to check"
    result["gates"]["variance_probe"] = {"ok": var_ok, "msg": var_msg}

    structural_ok = type_ok and det_ok and var_ok

    # --- 6. Behavioral: ID RMSE vs persistence baseline + absolute bound
    id_rmse = ood_rmse = persistence_id = persistence_ood = float("nan")
    beats_ok = abs_ok = False
    if structural_ok:
        try:
            X_id = id_blob["X_true"]
            X_ood = ood_blob["X_true"]
            id_input = {k: v for k, v in id_blob.items() if k != "X_true"}
            ood_input = {k: v for k, v in ood_blob.items() if k != "X_true"}
            x_hat_id = _call(estimate, id_input)
            x_hat_ood = _call(estimate, ood_input)
            id_rmse = _rmse(x_hat_id, X_id)
            ood_rmse = _rmse(x_hat_ood, X_ood)
            persistence_id = _rmse(_persistence_baseline(id_blob["Y"], n), X_id)
            persistence_ood = _rmse(_persistence_baseline(ood_blob["Y"], n), X_ood)
            beats_ok = bool(id_rmse < BEATS_PERSISTENCE_RATIO * persistence_id)
            abs_ok = bool(id_rmse <= ABSOLUTE_RMSE_BOUND)
        except Exception as exc:
            result["errors"].append(f"behavioral: {exc!r}")

    result["gates"]["beats_persistence"] = {
        "ok": beats_ok,
        "msg": (f"id_rmse={id_rmse:.3f} threshold="
                f"{BEATS_PERSISTENCE_RATIO}*{persistence_id:.3f}="
                f"{BEATS_PERSISTENCE_RATIO * persistence_id:.3f}"),
    }
    result["gates"]["absolute_rmse"] = {
        "ok": abs_ok,
        "msg": f"id_rmse={id_rmse:.3f} bound={ABSOLUTE_RMSE_BOUND}",
    }

    result["metrics"]["id_rmse"] = id_rmse
    result["metrics"]["ood_rmse"] = ood_rmse
    result["metrics"]["persistence_rmse_id"] = persistence_id
    result["metrics"]["persistence_rmse_ood"] = persistence_ood
    result["metrics"]["generalization_gap"] = ood_rmse - id_rmse if not np.isnan(id_rmse) else float("nan")

    all_gates_ok = structural_ok and beats_ok and abs_ok
    if all_gates_ok and persistence_id > 0:
        behavioral = max(0.0, 1.0 - id_rmse / persistence_id)
    else:
        behavioral = 0.0
    score = behavioral if all_gates_ok else 0.0

    result["score"] = score
    result["wall_clock_s"] = round(time.time() - t0, 3)
    print(json.dumps(result))
    return 0 if score > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
