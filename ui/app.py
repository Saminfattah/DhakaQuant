from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ui.components.metrics import date_text
from ui.components.warnings import research_banner
from ui.services.data_loader import load_signals
from ui.views import (
    data_quality,
    model_health,
    overview,
    pipeline_control,
    rankings,
    settings_page,
    signals,
    ticker_explorer,
)

APP_NAME = "DhakaQuant"

st.set_page_config(
    page_title=APP_NAME,
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

css_path = ROOT / "ui/styles/theme.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

st.sidebar.markdown(f"## {APP_NAME}")
st.sidebar.caption("Quantitative Equity Research")

try:
    sidebar_signals = load_signals(ROOT)
    freshness = sidebar_signals["data_freshness_date"].max() if "data_freshness_date" in sidebar_signals else None
    model_version = sidebar_signals["model_version"].iloc[0] if len(sidebar_signals) else None
except Exception:  # noqa: BLE001
    freshness = None
    model_version = None

st.sidebar.markdown(f"**Data:** {date_text(freshness)}")
st.sidebar.markdown(f"**Model:** {model_version or 'Not trained'}")
st.sidebar.divider()

PAGES = {
    "Overview": overview.render,
    "Signals": signals.render,
    "Rankings": rankings.render,
    "Ticker Explorer": ticker_explorer.render,
    "Model Health": model_health.render,
    "Data Quality": data_quality.render,
    "Pipeline Control": pipeline_control.render,
    "Settings": settings_page.render,
}

if "nav" not in st.session_state:
    st.session_state["nav"] = "Overview"
page = st.sidebar.radio("Navigate", list(PAGES), key="nav", label_visibility="collapsed")
st.sidebar.divider()
st.sidebar.caption("All data stays on this computer. Closing the browser does not delete it.")

research_banner()
try:
    PAGES[page](ROOT)
except FileNotFoundError as exc:
    st.info(f"{exc} Use Pipeline Control to generate the required artifact.")
except Exception as exc:  # noqa: BLE001
    st.error("This page could not load its local artifact.")
    with st.expander("Technical details"):
        st.exception(exc)
