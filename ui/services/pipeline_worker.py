from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path

from dse_quant.cli import main as worker_main
from ui.services.pipeline_runner import ALLOWED_COMMANDS, state_path


def _write(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=sorted(ALLOWED_COMMANDS))
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    path = state_path(root)
    started_at = datetime.now(UTC).isoformat()
    _write(
        path,
        {
            "status": "running",
            "command": args.command,
            "pid": os.getpid(),
            "started_at": started_at,
            "exit_code": None,
        },
    )
    exit_code = worker_main([args.command])
    _write(
        path,
        {
            "status": "completed" if exit_code == 0 else "failed",
            "command": args.command,
            "pid": os.getpid(),
            "started_at": started_at,
            "finished_at": datetime.now(UTC).isoformat(),
            "exit_code": exit_code,
        },
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

