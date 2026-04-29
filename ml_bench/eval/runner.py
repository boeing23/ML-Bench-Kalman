"""Episode runner: model output → score + judge feedback.

Usage:
    python -m ml_bench.eval.runner \
        --task t05_fault_detection \
        --solution path/to/solution.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

import yaml

from ml_bench.runtime.sandbox import run_episode

REPO_ROOT = Path(__file__).resolve().parents[2]
TASKS_DIR = REPO_ROOT / "ml_bench" / "tasks"


def _load_task(task_id: str) -> tuple[Path, dict, str]:
    task_dir = TASKS_DIR / task_id
    if not task_dir.exists():
        raise FileNotFoundError(f"task not found: {task_dir}")
    cfg = yaml.safe_load((task_dir / "task.yaml").read_text())
    prompt = (task_dir / "prompt.md").read_text()
    return task_dir, cfg, prompt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--solution", required=True, help="path to solution.py")
    ap.add_argument("--no-judge", action="store_true", help="skip Opus judge call")
    ap.add_argument("--out", default=None, help="write the result JSON here")
    args = ap.parse_args()

    task_dir, cfg, prompt = _load_task(args.task)
    solution_text = Path(args.solution).read_text()

    limits = cfg.get("limits", {}) or {}
    sandbox_result = run_episode(
        task_dir=task_dir,
        solution_text=solution_text,
        wall_clock_s=int(limits.get("wall_clock_s", 120)),
        memory_mb=int(limits.get("memory_mb", 4096)),
    )

    out: dict = {
        "task": args.task,
        "deterministic": {
            "score": sandbox_result.score,
            "judge_payload": sandbox_result.judge_payload,
            "timed_out": sandbox_result.timed_out,
            "exit_code": sandbox_result.exit_code,
        },
        "judge": None,
        "combined_score": sandbox_result.score,  # default if Opus skipped
    }

    if not args.no_judge:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("[runner] ANTHROPIC_API_KEY not set; skipping Opus judge",
                  file=sys.stderr)
        else:
            from ml_bench.judge.opus_judge import grade
            try:
                judge_score = grade(
                    task_id=args.task,
                    task_prompt=prompt,
                    solution_text=solution_text,
                    test_payload=sandbox_result.judge_payload,
                )
                out["judge"] = judge_score.model_dump()
                # Combined: deterministic score is the gate; Opus tunes within
                # passing solutions.  α=0.7 deterministic + β=0.3 judge_normalized.
                out["combined_score"] = (
                    0.7 * sandbox_result.score
                    + 0.3 * judge_score.normalized
                )
            except Exception as exc:
                out["judge_error"] = repr(exc)

    payload = json.dumps(out, indent=2)
    print(payload)
    if args.out:
        Path(args.out).write_text(payload)

    return 0 if sandbox_result.score > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
