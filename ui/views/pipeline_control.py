from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path

import streamlit as st

from ui.services.data_loader import tail_text
from ui.services.pipeline_runner import (
    ALLOWED_COMMANDS,
    CONFIRMATION_REQUIRED,
    command_argv,
    current_state,
    process_log_path,
    start_pipeline,
)

DESCRIPTIONS = {
    "ingest": "Update canonical daily prices only.",
    "validate": "Validate the current canonical dataset.",
    "build-features": "Rebuild deterministic feature data.",
    "train": "Train and replace the latest model after evaluation.",
    "predict": "Generate predictions from the current model.",
    "signals": "Regenerate labels and rankings from predictions.",
    "daily-run": "Update data, rebuild features, predict, and signal without retraining.",
    "run-all": "Run every stage and replace the current model.",
}


def _elapsed(state: dict) -> str:
    if not state.get("started_at"):
        return "—"
    start = datetime.fromisoformat(state["started_at"])
    end = datetime.fromisoformat(state["finished_at"]) if state.get("finished_at") else datetime.now(UTC)
    seconds = max(0, int((end - start).total_seconds()))
    minutes, seconds = divmod(seconds, 60)
    return f"{minutes}m {seconds}s"


def render(root: Path) -> None:
    st.markdown('<div class="page-kicker">Explicit local operations</div>', unsafe_allow_html=True)
    st.title("Pipeline Control")
    st.caption("No stage runs automatically. Commands are restricted to the worker’s fixed allowlist.")

    state = current_state(root)
    running = state.get("status") == "running"
    status_columns = st.columns(5)
    status_columns[0].metric("Status", str(state.get("status", "idle")).title())
    status_columns[1].metric("Command", state.get("command", "—"))
    status_columns[2].metric("Process", state.get("pid", "—"))
    status_columns[3].metric("Elapsed", _elapsed(state))
    status_columns[4].metric("Exit code", state.get("exit_code") if state.get("exit_code") is not None else "—")

    if running:
        st.info("A pipeline command is running. Use Refresh status to read its latest state.")
    elif state.get("status") == "failed":
        st.error("The latest pipeline command failed. Review the process log below.")
    elif state.get("status") == "completed":
        st.success("The latest pipeline command completed successfully.")

    command = st.selectbox("Pipeline command", sorted(ALLOWED_COMMANDS))
    st.caption(DESCRIPTIONS[command])
    argv = command_argv(command, sys.executable)
    st.code(" ".join(argv), language="powershell")

    confirmed = True
    if command in CONFIRMATION_REQUIRED:
        confirmed = st.checkbox(
            "I understand this command retrains and replaces the current approved model."
        )
    controls = st.columns(3)
    if controls[0].button(
        "Start command",
        type="primary",
        disabled=running or not confirmed,
        width="stretch",
    ):
        try:
            start_pipeline(root, command, sys.executable)
            st.success(f"Started {command}.")
            st.rerun()
        except (RuntimeError, ValueError, OSError) as exc:
            st.error(str(exc))
    if controls[1].button("Refresh status", width="stretch"):
        st.rerun()

    process_log = process_log_path(root)
    st.subheader("Background process log")
    st.code(tail_text(process_log, 80), language="text")
    if process_log.exists():
        st.download_button(
            "Download process log",
            process_log.read_bytes(),
            file_name="ui_pipeline_process.log",
            mime="text/plain",
        )
    with st.expander("Main worker log"):
        st.code(tail_text(root / "logs/dse_quant.log", 80), language="text")
