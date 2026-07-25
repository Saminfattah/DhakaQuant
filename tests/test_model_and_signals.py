from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from dse_quant.signals.generator import _label


def test_lightgbm_seed_is_reproducible():
    features = pd.DataFrame({"a": np.arange(40), "b": np.arange(40) % 3})
    target = np.array([0, 1] * 20)
    first = LGBMClassifier(n_estimators=10, random_state=42, verbosity=-1).fit(features, target)
    second = LGBMClassifier(n_estimators=10, random_state=42, verbosity=-1).fit(features, target)
    np.testing.assert_allclose(first.predict_proba(features), second.predict_proba(features))


def test_buy_watch_boundary_and_quality_gate():
    config = {
        "buy_watch_threshold": 0.70,
        "sell_review_threshold": 0.60,
        "require_model_quality_for_buy": True,
    }
    row = pd.Series(
        {
            "probability_up": 0.70,
            "probability_down": 0.30,
            "liquidity_flag": False,
            "stale_price_flag": False,
            "floor_price_flag": False,
            "insufficient_history_flag": False,
            "reason_codes": "",
        }
    )
    assert _label(row, config, True)[0] == "BUY WATCH"
    assert _label(row, config, False)[0] == "HOLD"


def test_risk_flags_force_avoid():
    config = {
        "buy_watch_threshold": 0.70,
        "sell_review_threshold": 0.60,
        "require_model_quality_for_buy": True,
    }
    row = pd.Series(
        {
            "probability_up": 0.90,
            "probability_down": 0.10,
            "liquidity_flag": True,
            "stale_price_flag": False,
            "floor_price_flag": False,
            "insufficient_history_flag": False,
            "reason_codes": "LOW_LIQUIDITY",
        }
    )
    assert _label(row, config, True)[0] == "AVOID"

