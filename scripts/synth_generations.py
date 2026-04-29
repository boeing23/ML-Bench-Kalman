"""Synthesize a generations.jsonl for end-to-end testing of the SFT formatter
without needing Qwen / Colab inference. Writes one passing reference per task
plus a couple of deliberately-broken cheaters.

The point: exercise the formatter's positive-pick + (optional) correction
paths on data we know the harness's exact answer to.

Usage:
    python scripts/synth_generations.py runs/round_0/generations.jsonl
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
TASKS_DIR = REPO_ROOT / "ml_bench" / "tasks"


# --- Cheater snippets (will fail the gates, judge may propose fixes) ---
CHEATER_T05_CONST = """\
import numpy as np
def predict(X: np.ndarray) -> np.ndarray:
    return np.full(X.shape[0], 0.9, dtype=np.float64)
"""

CHEATER_T01_PERSIST = """\
import numpy as np
def estimate(A, B, C, Q, R, U, Y, x0_mean, x0_cov):
    N, T, p = Y.shape
    n = A.shape[0]
    out = np.zeros((N, T, n), dtype=np.float64)
    out[..., :p] = Y
    return out
"""

CHEATER_T02_ZERO = """\
import numpy as np
def fit_and_predict(U_train, Y_train, U_test, Y_test_warmup):
    N_te, T_total, q = U_test.shape[0], U_test.shape[1], Y_test_warmup.shape[2]
    T_warm = Y_test_warmup.shape[1]
    return np.zeros((N_te, T_total - T_warm, q), dtype=np.float64)
"""


def _read_reference(task_id: str) -> str:
    return (TASKS_DIR / task_id / "reference_solution.py").read_text()


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: synth_generations.py <out_path>", file=sys.stderr)
        return 2
    out_path = Path(sys.argv[1])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    sample_idx = 0

    # One passing reference per task (will be a positive example).
    for tid in ["t01_kalman_filter", "t02_system_identification", "t05_fault_detection"]:
        records.append({
            "round": 0,
            "task_id": tid,
            "sample_idx": sample_idx,
            "solution_text": _read_reference(tid),
            "generated_by": "reference",
        })
        sample_idx += 1

    # Three deliberately-broken cheaters (will fail; judge may propose fixes).
    for tid, txt in [
        ("t05_fault_detection", CHEATER_T05_CONST),
        ("t01_kalman_filter", CHEATER_T01_PERSIST),
        ("t02_system_identification", CHEATER_T02_ZERO),
    ]:
        records.append({
            "round": 0,
            "task_id": tid,
            "sample_idx": sample_idx,
            "solution_text": txt,
            "generated_by": "synth_cheater",
        })
        sample_idx += 1

    with out_path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print(f"wrote {len(records)} synthetic generations to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
