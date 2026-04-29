"""SFT data formatter — turns generations.jsonl into a ChatML training pool.

Pipeline (per round):
  1. Read generations.jsonl: one record per generated solution.
  2. For each, run through the harness (sandbox + optional Opus judge).
  3. Filter:
       - Positive examples = top-k by deterministic score per task.
       - Corrections = failed generations whose `corrected_sketch` REWRITES
         the file and scores higher than the original when re-run.
  4. Format each survivor as ChatML messages.
  5. Optionally mix in prior rounds' SFT pools with geometric decay.
  6. Write sft_training.jsonl.

Generation record schema (input):
    {
      "round": int,
      "task_id": str,
      "sample_idx": int,
      "solution_text": str,
      "generated_by": str (optional metadata)
    }

SFT record schema (output):
    {
      "messages": [...],                # ChatML
      "task_id": str,
      "round": int,
      "source": "positive" | "correction",
      "deterministic_score": float,
      "judge_normalized": float | null,
      "origin_sample_idx": int
    }

CLI:
    python -m ml_bench.training.sft_formatter \
        --generations runs/round_0/generations.jsonl \
        --out runs/round_0/sft_training.jsonl \
        --top-k 4 \
        [--include-corrections] \
        [--mix-from runs/ --decay 0.5] \
        [--no-judge]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterator, Optional

import yaml

from ml_bench.runtime.sandbox import run_episode
from ml_bench.training.chat_format import (
    positive_example,
    correction_example,
    is_well_formed,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
TASKS_DIR = REPO_ROOT / "ml_bench" / "tasks"


@dataclass
class ScoredGeneration:
    round: int
    task_id: str
    sample_idx: int
    solution_text: str
    deterministic_score: float
    judge_feedback: str
    corrected_sketch: str
    judge_normalized: Optional[float]


def _load_task(task_id: str) -> tuple[Path, dict, str]:
    task_dir = TASKS_DIR / task_id
    cfg = yaml.safe_load((task_dir / "task.yaml").read_text())
    prompt = (task_dir / "prompt.md").read_text()
    return task_dir, cfg, prompt


def _read_jsonl(path: Path) -> Iterator[dict]:
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _score_one(
    task_id: str,
    task_dir: Path,
    task_prompt: str,
    cfg: dict,
    solution_text: str,
    *,
    use_judge: bool,
) -> tuple[float, str, str, Optional[float]]:
    """Run a single solution through the harness; return
    (det_score, judge_feedback, corrected_sketch, judge_normalized).
    """
    limits = cfg.get("limits", {}) or {}
    sandbox_result = run_episode(
        task_dir=task_dir,
        solution_text=solution_text,
        wall_clock_s=int(limits.get("wall_clock_s", 120)),
        memory_mb=int(limits.get("memory_mb", 4096)),
    )
    det_score = float(sandbox_result.score)
    test_payload = sandbox_result.judge_payload or {}

    judge_feedback = ""
    corrected_sketch = ""
    judge_normalized: Optional[float] = None

    if use_judge and os.environ.get("ANTHROPIC_API_KEY"):
        try:
            from ml_bench.judge.opus_judge import grade
            score = grade(
                task_id=task_id,
                task_prompt=task_prompt,
                solution_text=solution_text,
                test_payload=test_payload,
            )
            judge_feedback = score.feedback
            corrected_sketch = score.corrected_sketch
            judge_normalized = score.normalized
        except Exception as exc:
            print(f"[formatter] judge failed for {task_id}: {exc!r}", file=sys.stderr)

    return det_score, judge_feedback, corrected_sketch, judge_normalized


def score_generations(
    gens: list[dict],
    *,
    use_judge: bool,
) -> list[ScoredGeneration]:
    out: list[ScoredGeneration] = []
    task_cache: dict[str, tuple[Path, dict, str]] = {}

    for g in gens:
        tid = g["task_id"]
        if tid not in task_cache:
            task_cache[tid] = _load_task(tid)
        task_dir, cfg, prompt = task_cache[tid]

        det, fb, sketch, jn = _score_one(
            task_id=tid,
            task_dir=task_dir,
            task_prompt=prompt,
            cfg=cfg,
            solution_text=g["solution_text"],
            use_judge=use_judge,
        )
        out.append(ScoredGeneration(
            round=int(g["round"]),
            task_id=tid,
            sample_idx=int(g["sample_idx"]),
            solution_text=g["solution_text"],
            deterministic_score=det,
            judge_feedback=fb,
            corrected_sketch=sketch,
            judge_normalized=jn,
        ))
    return out


def _select_positives(scored: list[ScoredGeneration], top_k: int) -> list[ScoredGeneration]:
    """Per task: keep generations with det_score > 0, top-k by score."""
    by_task: dict[str, list[ScoredGeneration]] = {}
    for s in scored:
        if s.deterministic_score > 0:
            by_task.setdefault(s.task_id, []).append(s)
    positives: list[ScoredGeneration] = []
    for tid, lst in by_task.items():
        lst.sort(key=lambda s: s.deterministic_score, reverse=True)
        positives.extend(lst[:top_k])
    return positives


def _select_corrections(
    scored: list[ScoredGeneration],
    *,
    use_judge: bool,
) -> list[tuple[ScoredGeneration, ScoredGeneration]]:
    """For each failed generation with a non-empty corrected_sketch:
    re-run the sketch through the harness; keep iff score strictly improves.

    Returns list of (failed, corrected) pairs where corrected is a
    ScoredGeneration whose solution_text is the sketch and whose
    deterministic_score is the harness re-run score.
    """
    pairs: list[tuple[ScoredGeneration, ScoredGeneration]] = []
    task_cache: dict[str, tuple[Path, dict, str]] = {}

    for s in scored:
        if s.deterministic_score > 0:
            continue                                 # only failed → fixed pairs
        if not s.corrected_sketch.strip():
            continue                                 # judge didn't propose a fix
        if s.task_id not in task_cache:
            task_cache[s.task_id] = _load_task(s.task_id)
        task_dir, cfg, prompt = task_cache[s.task_id]
        det2, fb2, sk2, jn2 = _score_one(
            task_id=s.task_id,
            task_dir=task_dir,
            task_prompt=prompt,
            cfg=cfg,
            solution_text=s.corrected_sketch,
            use_judge=use_judge,
        )
        if det2 > s.deterministic_score:
            corrected = ScoredGeneration(
                round=s.round,
                task_id=s.task_id,
                sample_idx=s.sample_idx,
                solution_text=s.corrected_sketch,
                deterministic_score=det2,
                judge_feedback=fb2,
                corrected_sketch="",
                judge_normalized=jn2,
            )
            pairs.append((s, corrected))
    return pairs


def _format_records(
    positives: list[ScoredGeneration],
    corrections: list[tuple[ScoredGeneration, ScoredGeneration]],
) -> list[dict]:
    records: list[dict] = []
    task_prompt_cache: dict[str, str] = {}

    def _prompt(tid: str) -> str:
        if tid not in task_prompt_cache:
            task_prompt_cache[tid] = (TASKS_DIR / tid / "prompt.md").read_text()
        return task_prompt_cache[tid]

    for p in positives:
        msgs = positive_example(_prompt(p.task_id), p.solution_text)
        if not is_well_formed(msgs):
            continue
        records.append({
            "messages": msgs,
            "task_id": p.task_id,
            "round": p.round,
            "source": "positive",
            "deterministic_score": p.deterministic_score,
            "judge_normalized": p.judge_normalized,
            "origin_sample_idx": p.sample_idx,
        })

    for failed, corrected in corrections:
        msgs = correction_example(
            task_prompt=_prompt(failed.task_id),
            failed_solution=failed.solution_text,
            judge_feedback=failed.judge_feedback or "(no feedback recorded)",
            corrected_solution=corrected.solution_text,
        )
        if not is_well_formed(msgs):
            continue
        records.append({
            "messages": msgs,
            "task_id": failed.task_id,
            "round": failed.round,
            "source": "correction",
            "deterministic_score": corrected.deterministic_score,
            "judge_normalized": corrected.judge_normalized,
            "origin_sample_idx": failed.sample_idx,
        })

    return records


def _mix_prior_rounds(
    base_dir: Optional[Path],
    current_round: int,
    decay: float,
) -> list[dict]:
    """Pull SFT records from rounds < current_round; geometric-decay weighting
    determines how many records from each prior round survive into the mix.
    """
    if base_dir is None:
        return []
    out: list[dict] = []
    for prior in range(current_round - 1, -1, -1):
        prior_path = base_dir / f"round_{prior}" / "sft_training.jsonl"
        if not prior_path.exists():
            continue
        weight = decay ** (current_round - prior)
        prior_records = list(_read_jsonl(prior_path))
        n_keep = max(1, math.floor(weight * len(prior_records)))
        out.extend(prior_records[:n_keep])
    return out


def format_round(
    generations_path: Path,
    output_path: Path,
    *,
    top_k_per_task: int = 4,
    include_corrections: bool = False,
    use_judge: bool = True,
    mix_from: Optional[Path] = None,
    decay: float = 0.5,
) -> dict:
    gens = list(_read_jsonl(generations_path))
    if not gens:
        raise ValueError(f"no generations in {generations_path}")
    current_round = int(gens[0]["round"])

    print(f"[formatter] scoring {len(gens)} generations through harness "
          f"(judge={'on' if use_judge else 'off'})...")
    scored = score_generations(gens, use_judge=use_judge)

    positives = _select_positives(scored, top_k_per_task)
    corrections: list[tuple[ScoredGeneration, ScoredGeneration]] = []
    if include_corrections:
        print(f"[formatter] checking corrected_sketches against the harness...")
        corrections = _select_corrections(scored, use_judge=use_judge)

    records = _format_records(positives, corrections)
    prior = _mix_prior_rounds(mix_from, current_round, decay)

    all_records = records + prior
    _write_jsonl(output_path, all_records)

    summary = {
        "round": current_round,
        "n_generations_in": len(gens),
        "n_passing": sum(1 for s in scored if s.deterministic_score > 0),
        "n_positives_kept": len(positives),
        "n_corrections_kept": len(corrections),
        "n_prior_round_records": len(prior),
        "n_total_out": len(all_records),
        "output_path": str(output_path),
    }
    print(f"[formatter] {json.dumps(summary, indent=2)}")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--top-k", type=int, default=4,
                    help="positives kept per task")
    ap.add_argument("--include-corrections", action="store_true",
                    help="re-run judge.corrected_sketch through harness; "
                         "keep iff score strictly improves")
    ap.add_argument("--no-judge", action="store_true",
                    help="skip Opus calls (det score only)")
    ap.add_argument("--mix-from", type=Path, default=None,
                    help="root of runs/ to mix prior rounds from")
    ap.add_argument("--decay", type=float, default=0.5,
                    help="geometric decay for prior-round mixing")
    args = ap.parse_args()

    summary = format_round(
        generations_path=args.generations,
        output_path=args.out,
        top_k_per_task=args.top_k,
        include_corrections=args.include_corrections,
        use_judge=not args.no_judge,
        mix_from=args.mix_from,
        decay=args.decay,
    )
    return 0 if summary["n_total_out"] > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
