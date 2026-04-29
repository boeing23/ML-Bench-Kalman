"""Pilot calibration: how well does Opus solve each ML-Bench task?

Sends each task's prompt to Opus N times, extracts the Python code from the
response, scores it through the sandbox, and reports per-task pass rate.

The Anthropic spec asks for ~30–60% Opus pass per task — outside that band
the bench produces a flat curve when training a small model.

Usage:
    python scripts/pilot_with_opus.py                              # all tasks, 5 samples
    python scripts/pilot_with_opus.py --task t01_kalman_filter --n 8
    python scripts/pilot_with_opus.py --temperature 0.7 --out runs/pilot_v2
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ml_bench.runtime.sandbox import run_episode
from ml_bench.training.chat_format import SYSTEM_PROMPT


PILOT_MODEL = "claude-opus-4-7"
TASKS_DIR = REPO_ROOT / "ml_bench" / "tasks"
DEFAULT_TASKS = [
    "t01_kalman_filter",
    "t02_system_identification",
    "t05_fault_detection",
]
TARGET_LO, TARGET_HI = 0.30, 0.60

_FENCE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def _extract_python(text: str) -> str:
    m = _FENCE_RE.search(text)
    return (m.group(1) if m else text).strip()


@dataclass
class SampleResult:
    task_id: str
    sample_idx: int
    response_text: str
    solution_text: str
    score: float
    metrics: dict
    timed_out: bool


def _opus_sample(client, prompt_text: str, *,
                 max_tokens: int, temperature: float) -> str:
    msg = client.messages.create(
        model=PILOT_MODEL,
        max_tokens=max_tokens,
        temperature=temperature,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt_text}],
    )
    return "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")


def run_pilot(task_ids: list[str], n_samples: int,
              max_tokens: int, temperature: float,
              out_dir: Path) -> dict:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    from anthropic import Anthropic
    client = Anthropic()

    results: dict[str, list[SampleResult]] = {}
    for tid in task_ids:
        print(f"\n=== {tid} ===", flush=True)
        task_dir = TASKS_DIR / tid
        cfg = yaml.safe_load((task_dir / "task.yaml").read_text()) or {}
        prompt_text = (task_dir / "prompt.md").read_text()
        limits = cfg.get("limits", {}) or {}
        per_task: list[SampleResult] = []
        for i in range(n_samples):
            print(f"  sample {i+1}/{n_samples} ...", flush=True)
            response = _opus_sample(
                client, prompt_text,
                max_tokens=max_tokens, temperature=temperature,
            )
            solution = _extract_python(response)
            res = run_episode(
                task_dir=task_dir,
                solution_text=solution,
                wall_clock_s=int(limits.get("wall_clock_s", 120)),
                memory_mb=int(limits.get("memory_mb", 4096)),
            )
            score = float(res.score)
            metrics = (res.judge_payload or {}).get("metrics", {})
            per_task.append(SampleResult(
                task_id=tid, sample_idx=i,
                response_text=response, solution_text=solution,
                score=score, metrics=metrics,
                timed_out=res.timed_out,
            ))
            print(f"    score={score:.3f}  metrics={metrics}", flush=True)
        results[tid] = per_task

    out_dir.mkdir(parents=True, exist_ok=True)
    for tid, lst in results.items():
        path = out_dir / f"{tid}_samples.jsonl"
        with path.open("w") as f:
            for s in lst:
                f.write(json.dumps({
                    "task_id": s.task_id,
                    "sample_idx": s.sample_idx,
                    "score": s.score,
                    "metrics": s.metrics,
                    "timed_out": s.timed_out,
                    "solution_text": s.solution_text,
                }) + "\n")

    summary = {}
    for tid, lst in results.items():
        scores = [s.score for s in lst]
        n = len(scores)
        passed = sum(1 for s in scores if s > 0)
        summary[tid] = {
            "n": n,
            "mean": sum(scores) / n if n else 0.0,
            "best": max(scores) if scores else 0.0,
            "pass_rate": passed / n if n else 0.0,
            "n_passed": passed,
        }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def _print_summary(summary: dict) -> None:
    print()
    print(f"{'task':<32} | {'n':>3} | {'mean':>6} | {'best':>6} | {'pass':>13}")
    print("-" * 72)
    for tid, s in summary.items():
        pr = s["pass_rate"]
        print(f"{tid:<32} | {s['n']:>3} | {s['mean']:>6.3f} | {s['best']:>6.3f} | "
              f"{pr*100:>4.0f}% ({s['n_passed']}/{s['n']})")
    print()
    print(f"Target Opus pass band: {TARGET_LO*100:.0f}-{TARGET_HI*100:.0f}%")
    for tid, s in summary.items():
        pr = s["pass_rate"]
        if TARGET_LO <= pr <= TARGET_HI:
            verdict = "in band"
        elif pr > TARGET_HI:
            verdict = f"too easy (pass={pr*100:.0f}%) — tighten gates"
        else:
            verdict = f"too hard (pass={pr*100:.0f}%) — relax gates or add scaffolding"
        print(f"  {tid:<32}  {verdict}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default=None,
                    help="run only one task (default: all 3 tier-1 tasks)")
    ap.add_argument("--n", type=int, default=5,
                    help="samples per task")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "runs" / "pilot")
    args = ap.parse_args()

    task_ids = [args.task] if args.task else DEFAULT_TASKS
    summary = run_pilot(
        task_ids=task_ids,
        n_samples=args.n,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        out_dir=args.out,
    )
    _print_summary(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
