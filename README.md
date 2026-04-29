# ML-Bench

An RL environment for training small open-source LLMs to solve **CPS × ML-engineering** problems
(state estimation, system identification, learning-based control, fault detection, hybrid systems).

Each task ships a **prompt** (unambiguous spec), a **judge** (deterministic tests + Opus
on code quality), **tools** (sandboxed file I/O + code exec), and **data** (train visible to model;
held-out + OOD splits visible only to the judge). The environment runs entirely in a single
Colab notebook; QLoRA fine-tunes Qwen2.5-7B across rounds using rejection-sampling SFT on
top-reward generations.

## Quickstart

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...

# Generate Task 5 data (deterministic, seeded)
python scripts/generate_task05_data.py

# Run a solution through the full pipeline (reference solution as smoke test)
python -m ml_bench.eval.runner \
  --task t05_fault_detection \
  --solution ml_bench/tasks/t05_fault_detection/reference_solution.py
```

## Task suite

Tier 1 (in scope): t01 Kalman, t02 SysID, t04 Learning-based control, t05 Fault detection,
t07 Hybrid automaton. Tier 2 (stretch): t03 MPC, t06 LTL verification of NN controllers.

## Layout

```
ml_bench/
  runtime/    sandbox executor
  judge/      Opus wrapper + rubric
  tasks/      one directory per task; each ships prompt/data/hidden/reference
  eval/       episode runner (model output → score + feedback)
scripts/      data generators, calibration helpers
notebook/     Colab entry point
```
