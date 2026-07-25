from __future__ import annotations

import pandas as pd

from dse_quant.processing.instrument_universe import (
    filter_model_universe,
    fixed_income_candidates,
    parse_instrument_profile,
)


def _profile_html(instrument_type: str, sector: str) -> str:
    return f"""
    <html><body>
      Type of Instrument {instrument_type}
      Face/par Value 100
      Market Lot 1
      Total No. of Outstanding Securities 1000
      Sector {sector}
      Closing Price Graph
    </body></html>
    """


def test_fixed_income_candidates_do_not_match_equity_suffixes():
    tickers = [
        "ACI",
        "IBP",
        "LINDEBD",
        "ABBLPBOND",
        "BANKASI1PB",
        "BEXGSUKUK",
        "TB10Y0135",
        "DEBBDLUGG",
    ]
    assert fixed_income_candidates(tickers) == {
        "ABBLPBOND",
        "BANKASI1PB",
        "BEXGSUKUK",
        "TB10Y0135",
        "DEBBDLUGG",
    }


def test_parse_official_dse_instrument_profiles():
    bond = parse_instrument_profile(
        _profile_html("Corporate Bond", "Corporate Bond"),
        ticker="ABBLPBOND",
        detail_url="https://www.dsebd.org/displayCompany.php?name=ABBLPBOND",
    )
    equity = parse_instrument_profile(
        _profile_html("Equity", "Pharmaceuticals & Chemicals"),
        ticker="IBP",
        detail_url="https://www.dsebd.org/displayCompany.php?name=IBP",
    )
    assert bond["is_fixed_income"] is True
    assert bond["sector"] == "Corporate Bond"
    assert equity["is_fixed_income"] is False
    assert equity["sector"] == "Pharmaceuticals & Chemicals"


def test_model_universe_excludes_only_officially_classified_fixed_income():
    prices = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-01", "2026-01-01", "2026-01-01"]),
            "ticker": ["ACI", "ABBLPBOND", "TB10Y0135"],
            "close": [100.0, 1000.0, 110.0],
        }
    )
    classification = pd.DataFrame(
        {
            "ticker": ["ABBLPBOND", "TB10Y0135"],
            "instrument_type": ["Corporate Bond", "Debt"],
            "sector": ["Corporate Bond", "G-SEC (T.Bond)"],
            "is_fixed_income": [True, True],
        }
    )
    filtered, audit, summary = filter_model_universe(prices, classification)
    assert filtered["ticker"].tolist() == ["ACI"]
    assert summary["fixed_income_tickers_excluded"] == 2
    excluded = audit.loc[audit["model_status"] == "excluded_fixed_income", "ticker"]
    assert set(excluded) == {"ABBLPBOND", "TB10Y0135"}
