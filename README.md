# ML-Bench

An RL environment for training small open-source LLMs to solve **CPS × ML-engineering** problems
(state estimation, system identification, learning-based control, fault detection, hybrid systems).

Each task ships a **prompt** (unambiguous spec), a **judge** (deterministic tests + Opus
on code quality), **tools** (sandboxed file I/O + code exec), and **data** (train visible to model;
held-out + OOD splits visible only to the judge). The environment runs entirely in a single
Colab notebook; QLoRA fine-tunes Qwen2.5-7B across rounds using rejection-sampling SFT on
top-reward generations.

---

## Why this design — beyond RLVR/PRM/SFT

Frontier-model training leans on **deterministic oracles**: RLVR, PRM, SFT, RLHF. These
work where ground truth exists — math, code, instruction-following.

They break at the **research frontier**:

1. **No one knows the correct answer in advance.** The only honest oracle is *execution*:
   does the hypothesis generalize on held-out problems? — plus structural invariants from
   physics/math (energy conservation, dimensional consistency, posterior calibration,
   Lyapunov decrease). Theory follows execution, not the other way round.

2. **Reward policies skew toward the "exam-taker."** A model heavily optimized on
   board-exam-style tasks becomes proficient at the exam and fragile elsewhere. Strong
   exam scores do not predict correct outcomes on novel problems. Diversity collapses
   under terminal-reward optimization (Kirk et al on RLHF diversity loss; same pattern
   visible in DeepSeek-R1's saturated math reasoning vs degraded creative reasoning).

3. **Failed-but-novel attempts get discarded.** Pure top-k filtering by terminal reward
   throws away the agent who carried a structural insight that, nudged correctly, beats
   the current best. Preserve them via MAP-Elites bins, process rewards (AST-level
   structural probes), and correction pairs.

4. **Compare-notes across agents may unlock breakthroughs.** Once models are trained on
   verifiable tasks and extend to new research ideas, **shared notes** matter: agent A's
   partial finding plus agent B's different angle may click for either one. Closer to
   academic discourse than multi-agent RL. **Not a cooperative game** — no joint reward,
   no team formation, no opponent. Each agent optimizes its own reward; shared memory
   makes prior agents' findings available the way a literature review makes prior
   researchers' findings available.

5. **Architecture is paradigm-portable.** The substrate is domain-agnostic. CPS×ML is
   the test-bed; the same stack applies to math research, algorithm design, theory CS,
   biology / drug design, empirical ML, software engineering. Per domain only three
   things change: terminal verifier, structural-invariant probes, behavioral descriptors.

**Commitment**: Layer 2 terminal reward is generalization on held-out probes + structural
invariants, **never "match the reference solution."** A novel approach that beats the
reference on the held-out probe wins.

---

## Three-layer architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  Layer 1 — RLVR core                                             │
│  Hard agentic task + deterministic judge                         │
│  Gate-and-score (multiplicative) · hidden held-out + OOD probes  │
├──────────────────────────────────────────────────────────────────┤
│  Layer 2 — Trajectory env (post-verifiable)                      │
│  R = α·R_emp + β·R_inv + γ·R_nov + δ·R_cal                       │
│  R_emp  generalization on held-out (no oracle answer)            │
│  R_inv  structural invariants (PSD covariance, dimensional, ...) │
│  R_nov  MAP-Elites novelty bin (AST-extracted descriptors)       │
│  R_cal  reported uncertainty vs actual error (Brier-like)        │
├──────────────────────────────────────────────────────────────────┤
│  Layer 3 — MCP shared memory                                     │
│  SQLite-backed cross-episode notes                               │
│  Agent: recall/record   Judge: flag uncertain   Committee: review│
│  Cross-episode learning without weight updates                   │
└──────────────────────────────────────────────────────────────────┘
```

---

## Pipeline (per round, single Colab)

```
                  ┌─────────────────────────────────┐
                  │  prompt.md (per task)           │
                  └──────────────┬──────────────────┘
                                 ▼
                  ┌─────────────────────────────────┐
                  │  Qwen2.5-7B-Instruct (4-bit)    │
                  │  sample N solutions per task    │
                  └──────────────┬──────────────────┘
                                 ▼
            ┌────────────────────────────────────────────┐
            │  Sandbox judge (subprocess + rlimits)      │
            │  • type contract  • determinism atol 1e-6  │
            │  • variance probe • beats persistence      │
            │  • RMSE bound     • OOD / regime gates     │
            │  → deterministic score (gates multiplicative) │
            └──────────────┬─────────────────────────────┘
                           ▼
            ┌────────────────────────────────────────────┐
            │  Opus rubric judge (optional)              │
            │  code_quality / robustness / explanation   │
            │  + corrected_sketch for failed attempts    │
            │  → judge_normalized ∈ [0,1]                │
            └──────────────┬─────────────────────────────┘
                           ▼
              combined = 0.7·det + 0.3·judge_normalized
                           ▼
            ┌────────────────────────────────────────────┐
            │  SFT formatter                             │
            │  • top-k positives per task                │
            │  • verified corrections (re-run beats orig)│
            │  • mix prior rounds (geometric decay 0.5)  │
            │  → ChatML records                          │
            └──────────────┬─────────────────────────────┘
                           ▼
            ┌────────────────────────────────────────────┐
            │  QLoRA train (r=16, α=32, 2 epochs)        │
            └──────────────┬─────────────────────────────┘
                           ▼
                    next round (re-eval)
```

Deterministic score gates; judge tunes within passing solutions. Failed-but-novel
attempts retained via MAP-Elites bins (Layer 2) so wrong-now-right-later insights
survive across rounds.

---

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

---

## Task suite

Tier 1 (in scope): t01 Kalman, t02 SysID, t04 Learning-based control, t05 Fault detection,
t07 Hybrid automaton. Tier 2 (stretch): t03 MPC, t06 LTL verification of NN controllers.

| Task | Contract | Key gates |
|---|---|---|
| `t01_kalman_filter` | `estimate(A,B,C,Q,R,U,Y,x0_mean,x0_cov) → [N,T,n]` | type, determinism 1e-6, variance, persistence ratio 0.65, RMSE ≤ 0.55 |
| `t02_system_identification` | `identify(...) → A,B,C` | RMSE bound + OOD generalization gap |
| `t05_fault_detection` | `predict(X) → [N]` | F1 ID + OOD; class balance probe |

Cheater catalog (echo Y, constant zeros, random output, direct Y→X regression, hardcoded
matrices, reference verbatim) — all score 0 by construction.

---

## Layout

```
ml_bench/
  runtime/    sandbox executor (tmpdir + rlimits + last-JSON-line protocol)
  judge/      Opus wrapper + rubric (SHA256 disk cache, prompt-cache breakpoint)
  tasks/      one directory per task; each ships prompt / data / hidden / reference
  eval/       episode runner (model output → score + feedback)
  training/   SFT formatter + ChatML chat-format helpers
scripts/      data generators, Opus pilot calibration, synthetic generations
notebook/     Colab entry point (QLoRA iteration loop)
runs/         per-round generations + scores + SFT pool (gitignored)
results/      pilot writeups + logs
```

---

## Reward-hack analysis (summary)

| Hack | Mitigation |
|---|---|
| Direct Y→X regression | OOD-as-hard-gate |
| Hardcode plant matrices | Multi-regime probe |
| Submit reference verbatim | OOD + regime gates (different distribution) |
| Memory-poison false notes | Confidence field + committee review; notes never directly score |
| Cooperation-credit gaming | Citations weighted by downstream-reward improvement, depth-decayed λ, A/B-test gate |
| Behavioral-descriptor gaming | Descriptors from AST + runtime, not agent self-claim; bins must show RMSE delta |
| Judge-uncertainty exploitation | Uncertainty flags don't change current score; only inform future committee reviews |

Reward-denial test: cosmetic rewrites of the reference (renames, refactors, same algorithm)
produce identical score within 1e-6.
