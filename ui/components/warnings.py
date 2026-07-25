from __future__ import annotations

from dataclasses import dataclass

import streamlit as st

from ui.components.metrics import percent


@dataclass(frozen=True)
class QualityGate:
    passed: bool
    actual: float | None
    required: float
    buy_watch_enabled: bool


def model_quality_gate(
    actual_precision: float | None,
    required_precision: float,
    require_quality: bool = True,
) -> QualityGate:
    passed = actual_precision is not None and actual_precision >= required_precision
    return QualityGate(
        passed=passed,
        actual=actual_precision,
        required=required_precision,
        buy_watch_enabled=passed or not require_quality,
    )


def render_quality_warning(gate: QualityGate) -> None:
    if gate.buy_watch_enabled:
        st.success(
            f"Model quality gate passed: validation precision {percent(gate.actual)} "
            f"(required {percent(gate.required)})."
        )
    else:
        st.warning(
            "Model validation precision is below the configured safety threshold. "
            f"BUY WATCH signals are disabled. Actual {percent(gate.actual)}; "
            f"required {percent(gate.required)}."
        )


def research_banner() -> None:
    st.markdown(
        '<div class="research-banner"><strong>Research Only</strong>'
        "<span>No trade execution · Not financial advice · Human review required</span></div>",
        unsafe_allow_html=True,
    )

