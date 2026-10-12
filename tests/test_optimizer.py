"""Tests for optimizer.py and backtest.py on small, hand-checkable cases."""
from datetime import date

import numpy as np
import pandas as pd
import pytest

import data_fetch as dfm
from backtest import build_features, make_forecasts
from optimizer import BatteryConfig, solve, solve_rolling


def hourly(prices, start="2025-01-15"):
    idx = pd.date_range(pd.Timestamp(start).tz_localize(dfm.MARKET_TZ), periods=len(prices),
                        freq="h").tz_convert("UTC")
    return pd.DataFrame({"interval_start": idx, "lmp": np.asarray(prices, float)})


def test_buys_low_sells_high():
    cfg = BatteryConfig(power_mw=1, energy_mwh=1, rte=1.0, soc_min_frac=0, soc_max_frac=1,
                        soc_init_frac=0, degradation_cost=0)
    res = solve(hourly([20, 80, 50]), cfg)
    # with no losses and no wear cost, charging and discharging in the same hour is a
    # harmless tie for the LP, so check the net schedule
    assert res.dispatch["net_mw"].round(6).tolist() == [-1, 1, 0]
    assert res.total_revenue == pytest.approx(60)


def test_degradation_cost_blocks_tiny_spreads():
    cfg = BatteryConfig(power_mw=1, energy_mwh=1, rte=1.0, soc_min_frac=0, soc_max_frac=1,
                        soc_init_frac=0, degradation_cost=5)
    res = solve(hourly([20, 23]), cfg)                  # $3 spread < $5 wear cost
    assert res.total_discharge_mwh == pytest.approx(0)


def test_constraints_hold_on_mock_week(tmp_path):
    db = tmp_path / "m.db"
    dfm.generate_mock_lmp("2025-03-05", "2025-03-12", "CAPITL", db)   # includes the DST day
    lmp = dfm.load_lmp("2025-03-05", "2025-03-12", "CAPITL", db)
    cfg = BatteryConfig()
    res = solve_rolling(lmp, cfg)
    d = res.dispatch
    assert len(d) == len(lmp) == 8 * 24 - 1
    assert (d["charge_mw"] <= cfg.power_mw + 1e-6).all() and (d["discharge_mw"] <= cfg.power_mw + 1e-6).all()
    assert (d["charge_mw"] + d["discharge_mw"] <= cfg.power_mw + 1e-6).all()
    assert d["soc_mwh"].between(cfg.e_min - 1e-6, cfg.e_max + 1e-6).all()
    # SoC follows the energy balance hour by hour
    eta = cfg.eta_one_way
    soc_prev = np.r_[cfg.e_init, d["soc_mwh"].to_numpy()[:-1]]
    np.testing.assert_allclose(d["soc_mwh"], soc_prev + eta * d["charge_mw"] - d["discharge_mw"] / eta, atol=1e-5)


def test_forecast_schedule_never_beats_perfect_net(tmp_path):
    db = tmp_path / "m.db"
    dfm.generate_mock_lmp("2025-01-01", "2025-01-21", "CAPITL", db)
    lmp = dfm.load_lmp("2025-01-01", "2025-01-21", "CAPITL", db)
    lmp["flat"] = lmp["lmp"].mean() + np.random.default_rng(0).normal(0, 5, len(lmp))
    cfg = BatteryConfig()
    perfect = solve_rolling(lmp, cfg)
    on_noise = solve_rolling(lmp, cfg, plan_col="flat")
    assert on_noise.net_revenue <= perfect.net_revenue + 1e-6


def test_forecasts_use_only_past_days(tmp_path):
    db = tmp_path / "m.db"
    dfm.generate_mock_lmp("2024-10-01", "2025-01-31", "CAPITL", db)
    lmp = dfm.load_lmp("2024-10-01", "2025-01-31", "CAPITL", db)
    day = date(2025, 1, 20)
    base = make_forecasts(lmp, day, day)

    shocked = lmp.copy()
    local_day = shocked["interval_start"].dt.tz_convert(dfm.MARKET_TZ).dt.date
    shocked.loc[local_day >= day, "lmp"] += 1000            # change the day itself and later
    after = make_forecasts(shocked, day, day)

    for col in ["naive_d1", "naive_d7", "regression"]:
        np.testing.assert_allclose(base[col], after[col])  # forecasts must not move
    assert (after["lmp"] - base["lmp"]).min() == pytest.approx(1000)
