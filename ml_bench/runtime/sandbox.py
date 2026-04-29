"""Sandbox runtime for ML-Bench tasks.

Each episode is run inside a temporary directory laid out as:

    <tmp>/
        workspace/        # the model's view (mounted)
            data/         # task data (visible to model)
            <prompt.md>   # → README.md
            solution.py   # written by the model
        hidden/           # judge-only (not accessible to the model)
            test_id.npz
            test_ood.npz
            judge_tests.py

The judge_tests.py is invoked in a subprocess with `cwd=<tmp>/hidden`
and PYTHONPATH containing only the standard environment (the model's
solution is loaded by the judge via the absolute path /workspace/solution.py
which we expose by symlink or by writing through).

For the local Colab / dev case we don't actually use chroot — we use
absolute path conventions and process-level resource limits. Network
isolation is best-effort (we can't fully revoke it from a subprocess
on macOS without OS-level help); the prompt forbids network use.
"""
from __future__ import annotations

import json
import os
import resource
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SandboxResult:
    score: float
    judge_payload: dict
    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool


def _set_limits(memory_mb: int) -> None:
    # POSIX rlimit; macOS honors RLIMIT_AS partially
    try:
        soft = memory_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (soft, soft))
    except (ValueError, OSError):
        pass


def run_episode(
    task_dir: Path,
    solution_text: str,
    *,
    wall_clock_s: int = 120,
    memory_mb: int = 4096,
) -> SandboxResult:
    """Run one episode: stage task → write solution → invoke judge.

    `task_dir` is the on-disk task package, e.g.
    `ml_bench/tasks/t05_fault_detection/`.
    """
    task_dir = Path(task_dir).resolve()
    data_src = task_dir / "data"
    hidden_src = task_dir / "hidden"
    prompt_src = task_dir / "prompt.md"
    if not hidden_src.exists():
        raise FileNotFoundError(f"missing hidden dir: {hidden_src}")
    if not data_src.exists():
        raise FileNotFoundError(f"missing data dir (run the data generator): {data_src}")

    with tempfile.TemporaryDirectory(prefix="mlbench_") as tmp:
        tmp_root = Path(tmp)
        workspace = tmp_root / "workspace"
        hidden = tmp_root / "hidden"
        workspace.mkdir()
        # We use a symlink so the judge can hard-code /workspace/solution.py
        # only when running on a system that allows it. For portability we
        # also expose the workspace path via an env var.
        shutil.copytree(data_src, workspace / "data")
        if prompt_src.exists():
            shutil.copyfile(prompt_src, workspace / "README.md")
        (workspace / "solution.py").write_text(solution_text)
        shutil.copytree(hidden_src, hidden)

        # Try to expose a /workspace symlink so judge can use absolute paths.
        # If we can't (no permission), the judge still works via env var.
        ws_link_ok = False
        try:
            os.symlink(workspace, "/workspace")
            ws_link_ok = True
        except (PermissionError, FileExistsError, OSError):
            ws_link_ok = False

        env = os.environ.copy()
        env["MLBENCH_WORKSPACE"] = str(workspace)
        env["PYTHONDONTWRITEBYTECODE"] = "1"

        cmd = [sys.executable, str(hidden / "judge_tests.py")]

        try:
            proc = subprocess.run(
                cmd,
                cwd=str(hidden),
                capture_output=True,
                text=True,
                timeout=wall_clock_s,
                env=env,
                preexec_fn=(lambda: _set_limits(memory_mb)) if os.name == "posix" else None,
            )
            timed_out = False
            stdout, stderr, exit_code = proc.stdout, proc.stderr, proc.returncode
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            stdout = exc.stdout.decode() if exc.stdout else ""
            stderr = (exc.stderr.decode() if exc.stderr else "") + f"\n[TIMEOUT after {wall_clock_s}s]"
            exit_code = -1
        finally:
            if ws_link_ok:
                try:
                    os.unlink("/workspace")
                except OSError:
                    pass

        # Last JSON line in stdout is the judge payload.
        judge_payload = {}
        last_line = ""
        for line in reversed(stdout.splitlines()):
            line = line.strip()
            if line.startswith("{"):
                last_line = line
                break
        if last_line:
            try:
                judge_payload = json.loads(last_line)
            except json.JSONDecodeError:
                judge_payload = {}

        score = float(judge_payload.get("score", 0.0)) if not timed_out else 0.0

        return SandboxResult(
            score=score,
            judge_payload=judge_payload,
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
            timed_out=timed_out,
        )
