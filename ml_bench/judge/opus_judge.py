"""Opus judge: code quality + written feedback.

Deterministic tests already produced a behavioral score and probe results.
Opus only grades the things tests can't see: code quality, robustness,
explanation quality. The written feedback becomes the SFT signal.

Caching strategy:
- On-disk cache keyed by sha256(task_id + solution_text + test_payload).
- Anthropic prompt-cache breakpoint after the rubric block so the rubric
  + task description are served from cache (~70% input-token reduction).
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Optional

from .rubric import JUDGE_SYSTEM, JudgeScore


JUDGE_MODEL = "claude-opus-4-7"
CACHE_DIR_ENV = "MLBENCH_JUDGE_CACHE"
DEFAULT_CACHE_DIR = Path("judge_cache")


def _cache_key(task_id: str, solution_text: str, test_payload: dict) -> str:
    h = hashlib.sha256()
    h.update(task_id.encode())
    h.update(b"\x00")
    h.update(solution_text.encode())
    h.update(b"\x00")
    h.update(json.dumps(test_payload, sort_keys=True).encode())
    return h.hexdigest()


def _cache_dir() -> Path:
    p = Path(os.environ.get(CACHE_DIR_ENV, str(DEFAULT_CACHE_DIR)))
    p.mkdir(parents=True, exist_ok=True)
    return p


def _try_cache_read(key: str) -> Optional[JudgeScore]:
    f = _cache_dir() / f"{key}.json"
    if f.exists():
        try:
            return JudgeScore.model_validate_json(f.read_text())
        except Exception:
            return None
    return None


def _cache_write(key: str, score: JudgeScore) -> None:
    f = _cache_dir() / f"{key}.json"
    f.write_text(score.model_dump_json(indent=2))


def _build_user_message(task_prompt: str, solution_text: str, test_payload: dict) -> list[dict]:
    """Returns content blocks. The first two are cached; only the third varies
    per (task, solution) — but solution is what we're grading, so the cache
    win comes from re-judging the *same solution* across iterations."""
    return [
        {
            "type": "text",
            "text": "## Task prompt\n\n" + task_prompt,
            "cache_control": {"type": "ephemeral"},
        },
        {
            "type": "text",
            "text": (
                "## Deterministic test results\n\n"
                + json.dumps(test_payload, indent=2)
            ),
        },
        {
            "type": "text",
            "text": (
                "## Submitted solution\n\n```python\n"
                + solution_text
                + "\n```\n\n"
                "Return JSON only matching this schema: "
                + json.dumps(JudgeScore.model_json_schema())
            ),
        },
    ]


def grade(
    task_id: str,
    task_prompt: str,
    solution_text: str,
    test_payload: dict,
    *,
    use_cache: bool = True,
) -> JudgeScore:
    key = _cache_key(task_id, solution_text, test_payload)
    if use_cache:
        hit = _try_cache_read(key)
        if hit is not None:
            return hit

    try:
        from anthropic import Anthropic
    except ImportError as exc:
        raise RuntimeError("anthropic package not installed") from exc

    client = Anthropic()
    user_blocks = _build_user_message(task_prompt, solution_text, test_payload)

    msg = client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=4096,
        system=[
            {
                "type": "text",
                "text": JUDGE_SYSTEM,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": user_blocks}],
    )

    raw = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
    raw = raw.strip()
    if raw.startswith("```"):
        # strip ```json fences if present
        raw = raw.split("```", 2)[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip().rstrip("`").strip()

    try:
        score = JudgeScore.model_validate_json(raw)
    except Exception:
        # one retry on parse failure: ask the model to repair
        repair = client.messages.create(
            model=JUDGE_MODEL,
            max_tokens=4096,
            system="You produced invalid JSON. Return ONLY valid JSON matching the schema.",
            messages=[
                {"role": "user", "content": raw},
            ],
        )
        repaired = "".join(b.text for b in repair.content if getattr(b, "type", None) == "text").strip()
        score = JudgeScore.model_validate_json(repaired)

    if use_cache:
        _cache_write(key, score)
    return score
