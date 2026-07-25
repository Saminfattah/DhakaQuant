from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ALLOWED_COMMANDS = {
    "ingest",
    "validate",
    "build-features",
    "train",
    "predict",
    "signals",
    "daily-run",
    "run-all",
}
CONFIRMATION_REQUIRED = {"train", "run-all"}


def state_path(root: Path) -> Path:
    return root / "logs" / "ui_pipeline_state.json"


def process_log_path(root: Path) -> Path:
    return root / "logs" / "ui_pipeline_process.log"


def command_argv(command: str, python_executable: str | None = None) -> list[str]:
    if command not in ALLOWED_COMMANDS:
        raise ValueError(f"Pipeline command is not allowed: {command}")
    return [python_executable or sys.executable, "-m", "dse_quant.cli", command]


def _write_state(root: Path, payload: dict[str, Any]) -> None:
    path = state_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def read_state(root: Path) -> dict[str, Any]:
    path = state_path(root)
    if not path.exists():
        return {"status": "idle"}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"status": "unknown", "message": "Pipeline state file is unreadable."}


def _pid_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (OSError, TypeError, ValueError):
        return False


def current_state(root: Path) -> dict[str, Any]:
    state = read_state(root)
    pid = state.get("pid")
    if state.get("status") == "running" and pid and not _pid_is_alive(int(pid)):
        state = {
            **state,
            "status": "failed",
            "finished_at": datetime.now(UTC).isoformat(),
            "exit_code": -1,
            "message": "The background process ended without reporting completion.",
        }
        _write_state(root, state)
    return state


def is_running(root: Path) -> bool:
    return current_state(root).get("status") == "running"


def start_pipeline(root: Path, command: str, python_executable: str | None = None) -> int:
    if command not in ALLOWED_COMMANDS:
        raise ValueError(f"Pipeline command is not allowed: {command}")
    if is_running(root):
        raise RuntimeError("Another pipeline command is already running.")
    root = root.resolve()
    log_path = process_log_path(root)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    worker_argv = [
        python_executable or sys.executable,
        "-m",
        "ui.services.pipeline_worker",
        command,
        "--root",
        str(root),
    ]
    with log_path.open("a", encoding="utf-8") as output:
        process = subprocess.Popen(
            worker_argv,
            cwd=root,
            stdin=subprocess.DEVNULL,
            stdout=output,
            stderr=subprocess.STDOUT,
            shell=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    _write_state(
        root,
        {
            "status": "running",
            "command": command,
            "pid": process.pid,
            "started_at": datetime.now(UTC).isoformat(),
            "exit_code": None,
        },
    )
    return process.pid
