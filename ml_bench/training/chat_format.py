"""ChatML message format for SFT training.

Each training example is a list of role/content messages compatible with
Qwen2.5-Instruct's chat template (HF tokenizer.apply_chat_template handles
the actual special tokens).

Schema:
    [
      {"role": "system", "content": <system prompt>},
      {"role": "user",   "content": <task prompt>},
      {"role": "assistant", "content": <solution code, fenced>},
    ]

Optional "correction" form for failed→fixed examples:
    [system, user, assistant_failed, user_feedback, assistant_corrected]
"""
from __future__ import annotations

from typing import Optional


SYSTEM_PROMPT = (
    "You are an expert ML engineer working on cyber-physical systems "
    "problems. You will be given a task with a strict contract — exact "
    "file paths, function signatures, return types, and pass thresholds. "
    "Write a complete solution that satisfies every requirement. Return "
    "only the contents of the requested file inside a single Python code "
    "block; no extra prose."
)


def _wrap_solution(solution_text: str) -> str:
    return f"```python\n{solution_text.strip()}\n```"


def positive_example(task_prompt: str, solution_text: str) -> list[dict]:
    """Build a 3-turn ChatML example for a passing solution."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task_prompt},
        {"role": "assistant", "content": _wrap_solution(solution_text)},
    ]


def correction_example(
    task_prompt: str,
    failed_solution: str,
    judge_feedback: str,
    corrected_solution: str,
) -> list[dict]:
    """Build a 5-turn ChatML example: original failed → feedback → corrected.

    Teaches the model the failed→fixed mapping conditioned on critique.
    """
    feedback_user_msg = (
        "Your previous attempt failed deterministic tests. Reviewer feedback:\n\n"
        f"{judge_feedback.strip()}\n\n"
        "Please return a corrected solution."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task_prompt},
        {"role": "assistant", "content": _wrap_solution(failed_solution)},
        {"role": "user", "content": feedback_user_msg},
        {"role": "assistant", "content": _wrap_solution(corrected_solution)},
    ]


def is_well_formed(messages: list[dict]) -> bool:
    """Sanity check before serializing."""
    if not messages or messages[0]["role"] != "system":
        return False
    if messages[-1]["role"] != "assistant":
        return False
    expected_alternation = ["system"] + ["user", "assistant"] * ((len(messages) - 1) // 2)
    if len(messages) - 1 != 2 * ((len(messages) - 1) // 2):
        return False
    if [m["role"] for m in messages] != expected_alternation:
        return False
    return all(isinstance(m["content"], str) and m["content"] for m in messages)
