from __future__ import annotations

from math import ceil
from pathlib import Path

import streamlit as st

from ui.services.data_loader import filter_signals, load_signals


def render(root: Path) -> None:
    st.markdown('<div class="page-kicker">Decision-support labels</div>', unsafe_allow_html=True)
    st.title("Signal Explorer")
    st.caption("Filter current model probabilities and inspect every risk reason before review.")
    frame = load_signals(root)

    row_one = st.columns([1.4, 1.2, 1, 1])
    search = row_one[0].text_input("Ticker search", placeholder="ACI")
    selected_signals = row_one[1].multiselect(
        "Signals", sorted(frame["signal"].dropna().unique().tolist())
    )
    minimum_up = row_one[2].slider("Minimum P(Up)", 0.0, 1.0, 0.0, 0.01)
    minimum_down = row_one[3].slider("Minimum P(Down)", 0.0, 1.0, 0.0, 0.01)

    row_two = st.columns(4)
    liquidity = row_two[0].selectbox("Liquidity", ["All", "Normal", "Flagged"])
    stale = row_two[1].selectbox("Stale price", ["All", "Normal", "Flagged"])
    floor = row_two[2].selectbox("Floor-price proxy", ["All", "Normal", "Flagged"])
    versions = row_two[3].multiselect(
        "Model version", sorted(frame["model_version"].dropna().unique().tolist())
    )

    filtered = filter_signals(
        frame,
        search=search,
        signals=selected_signals,
        minimum_up=minimum_up,
        minimum_down=minimum_down,
        liquidity=liquidity,
        stale=stale,
        floor=floor,
        model_versions=versions,
    )
    sort_options = {
        "Probability up": ("probability_up", False),
        "Probability down": ("probability_down", False),
        "Ticker": ("ticker", True),
        "Latest price": ("latest_price", False),
    }
    controls = st.columns([1, 1, 2])
    sort_label = controls[0].selectbox("Sort by", list(sort_options))
    page_size = controls[1].selectbox("Rows per page", [25, 50, 100], index=1)
    sort_column, ascending = sort_options[sort_label]
    filtered = filtered.sort_values(sort_column, ascending=ascending)
    pages = max(1, ceil(len(filtered) / page_size))
    page = controls[2].number_input("Page", min_value=1, max_value=pages, value=1)
    start = (int(page) - 1) * page_size
    shown = filtered.iloc[start : start + page_size]

    st.caption(f"{len(filtered):,} matching tickers · page {int(page)} of {pages}")
    st.dataframe(
        shown,
        hide_index=True,
        width="stretch",
        height=620,
        column_config={
            "probability_up": st.column_config.ProgressColumn("P(Up)", format="percent", min_value=0, max_value=1),
            "probability_down": st.column_config.ProgressColumn("P(Down)", format="percent", min_value=0, max_value=1),
            "latest_price": st.column_config.NumberColumn("Latest price", format="৳%.2f"),
            "liquidity_flag": st.column_config.CheckboxColumn("Liquidity risk"),
            "stale_price_flag": st.column_config.CheckboxColumn("Stale"),
            "floor_price_flag": st.column_config.CheckboxColumn("Floor proxy"),
        },
    )
    st.download_button(
        "Download filtered CSV",
        filtered.to_csv(index=False).encode("utf-8"),
        file_name="filtered_dse_signals.csv",
        mime="text/csv",
    )

    if len(filtered):
        open_ticker = st.selectbox("Inspect a matching ticker", filtered["ticker"].tolist())
        if st.button("Open Ticker Explorer"):
            st.session_state["selected_ticker"] = open_ticker
            st.session_state["nav"] = "Ticker Explorer"
            st.rerun()
