from __future__ import annotations

import logging
from pathlib import Path


def configure_logging(log_dir: Path, verbose: bool = False) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    level = logging.DEBUG if verbose else logging.INFO
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    root.addHandler(console)

    file_handler = logging.FileHandler(log_dir / "dse_quant.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

