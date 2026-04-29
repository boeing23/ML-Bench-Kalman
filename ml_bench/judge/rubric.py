"""Rubric schema for the Opus code-quality judge.

Deterministic tests handle correctness (behavioral score, type contract,
adversarial probes). Opus grades the parts deterministic tests can't:
clarity, robustness, and a written feedback paragraph that downstream
SFT pipelines use as a learning signal.
"""
from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field


class JudgeScore(BaseModel):
    """What Opus returns. Strict JSON, validated by pydantic."""

    code_quality: Literal[0, 1, 2, 3] = Field(
        ..., description="0=broken/unreadable, 3=production-quality"
    )
    robustness: Literal[0, 1, 2, 3] = Field(
        ..., description="Handles edge cases / bad inputs / numerical issues"
    )
    explanation: Literal[0, 1, 2, 3] = Field(
        ...,
        description="Quality of any inline comments / docstring / written reasoning. "
        "0 if absent.",
    )
    feedback: str = Field(
        ...,
        description="One paragraph of concrete, actionable critique. Reference "
        "specific lines or choices. Used as SFT signal downstream.",
    )
    corrected_sketch: str = Field(
        default="",
        description="If the solution failed deterministic tests, return a "
        "COMPLETE replacement for solution.py that would pass the gates "
        "(full Python source, not a diff or snippet). Empty string if the "
        "solution passed all gates.",
    )

    @property
    def normalized(self) -> float:
        """Mean of 0–3 scores normalized to [0, 1]."""
        return (self.code_quality + self.robustness + self.explanation) / 9.0


JUDGE_SYSTEM = """\
You are a senior ML engineer grading a small model's solution to a CPS×ML
engineering task. Deterministic tests have already scored correctness; your
job is to judge code quality, robustness, and the quality of any written
explanation, AND to write one paragraph of feedback the model can learn from.

Return STRICT JSON matching the schema given to you. No prose outside the JSON.

Score each axis on 0–3:
    0 = absent / broken
    1 = present but poor
    2 = competent, minor issues
    3 = clean, production-quality

The feedback paragraph must be concrete: name specific lines, name the
specific failure mode, name what to change. Generic praise or generic
criticism is worth nothing as a learning signal.

If deterministic tests reported failures (any gate failed or score = 0),
write `corrected_sketch` as a COMPLETE replacement for solution.py — the
full Python source that, dropped in place of the original, would pass all
gates. Not a diff, not a snippet. The downstream SFT pipeline will run
your replacement through the harness; if it scores higher than the
original, it joins the training pool. If you cannot produce a complete
working solution, leave `corrected_sketch` as the empty string.
"""
