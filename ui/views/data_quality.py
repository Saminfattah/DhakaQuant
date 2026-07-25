from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from ui.components.metrics import date_text, number
from ui.services.data_loader import load_frame, load_json, load_signals


def _download(label: str, frame: pd.DataFrame, filename: str) -> None:
    st.download_button(
        label,
        frame.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
    )


def render(root: Path) -> None:
    st.markdown('<div class="page-kicker">Source and validation audit</div>', unsafe_allow_html=True)
    st.title("Data Quality")
    validation = load_json(root / "data/outputs/validation_report.json", default={})
    failures = load_json(root / "data/raw/failed_ranges.json", default=[])
    signals = load_signals(root)

    metrics = st.columns(6)
    metrics[0].metric("Rows", number(validation.get("rows")))
    metrics[1].metric("Tickers", number(validation.get("tickers")))
    metrics[2].metric("First date", date_text(validation.get("first_date")))
    metrics[3].metric("Latest date", date_text(validation.get("last_date")))
    metrics[4].metric("Validation", "Passed" if validation.get("valid") else "Review")
    metrics[5].metric("Stale tickers", int(signals["stale_price_flag"].fillna(False).sum()))

    issue_columns = st.columns(4)
    issue_columns[0].metric("Duplicate keys", number(validation.get("duplicate_keys")))
    issue_columns[1].metric("Missing required", number(validation.get("missing_required")))
    issue_columns[2].metric("Negative values", number(validation.get("negative_values")))
    issue_columns[3].metric("OHLC inconsistencies", number(validation.get("inconsistent_ohlc")))

    thin = validation.get("recent_thin_sessions", [])
    if thin:
        st.warning("Thin recent sessions: " + ", ".join(thin))
    else:
        st.success("Recent-session row-count checks passed.")

    boundary_path = root / "data/outputs/join_boundary_report.csv"
    if boundary_path.exists():
        boundary = load_frame(boundary_path, label="Join-boundary report")
        warnings = boundary[boundary["warning"].fillna(False)] if "warning" in boundary else boundary
        st.subheader("Kaggle → official DSE source transitions")
        st.caption(
            "Large source-boundary returns may indicate corporate-action adjustment differences, "
            "not genuine one-session returns."
        )
        st.dataframe(warnings, hide_index=True, width="stretch", height=420)
        _download("Download boundary report", boundary, "join_boundary_report.csv")

    st.subheader("Failed archive ranges")
    failure_frame = pd.DataFrame(failures)
    if failure_frame.empty:
        st.success("No unresolved DSE archive ranges were recorded in the latest ingestion.")
    else:
        st.dataframe(failure_frame, hide_index=True, width="stretch")
        _download("Download failed ranges", failure_frame, "failed_ranges.csv")

    with st.expander("Validation report JSON"):
        st.json(validation)
