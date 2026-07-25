from __future__ import annotations

import json

import pytest

from ui.services import pipeline_runner


def test_allowlist_builds_safe_argv():
    argv = pipeline_runner.command_argv("daily-run", "python.exe")
    assert argv == ["python.exe", "-m", "dse_quant.cli", "daily-run"]


def test_arbitrary_command_is_rejected():
    with pytest.raises(ValueError, match="not allowed"):
        pipeline_runner.command_argv("daily-run; Remove-Item *", "python.exe")


def test_background_state_is_persisted(tmp_path, monkeypatch):
    class DummyProcess:
        pid = 12345

    monkeypatch.setattr(pipeline_runner.subprocess, "Popen", lambda *args, **kwargs: DummyProcess())
    pid = pipeline_runner.start_pipeline(tmp_path, "validate", "python.exe")
    state = pipeline_runner.read_state(tmp_path)
    assert pid == 12345
    assert state["status"] == "running"
    assert state["command"] == "validate"
    assert json.loads(pipeline_runner.state_path(tmp_path).read_text())["pid"] == 12345


def test_duplicate_background_command_is_blocked(tmp_path):
    path = pipeline_runner.state_path(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text('{"status": "running"}', encoding="utf-8")
    with pytest.raises(RuntimeError, match="already running"):
        pipeline_runner.start_pipeline(tmp_path, "validate", "python.exe")


def test_stale_background_state_is_marked_failed(tmp_path, monkeypatch):
    path = pipeline_runner.state_path(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text('{"status": "running", "pid": 99999}', encoding="utf-8")
    monkeypatch.setattr(pipeline_runner, "_pid_is_alive", lambda pid: False)
    state = pipeline_runner.current_state(tmp_path)
    assert state["status"] == "failed"
    assert state["exit_code"] == -1
