"""
backtest.py  —  BESS Dispatch Optimizer
=======================================
How much of the perfect-foresight revenue does a battery keep when it has to
plan on a price *forecast* instead of the actual day-ahead prices?

For each NYISO zone and each forecast:

  1. Perfect foresight  — plan the day on the actual DA LMPs (upper bound).
  2. Forecast dispatch  — plan the day on a forecast of tomorrow's DA LMPs,
                          made only from prices known before the DAM closes.
  3. Settle both schedules at the actual DA LMPs.

  capture ratio = revenue of (2) / revenue of (1)

Forecasts (all strictly day-ahead: market day D uses data up to day D-1)
-----------------------------------------------------------------------
  naive_d1    same hour yesterday
  naive_d7    same hour one week ago
  regression  linear model refit every day on the trailing 90 days:
              hour-of-day and weekend dummies, lags of 24h / 48h / 168h,
              and yesterday's average price (small ridge penalty)

No load, weather, gas or outage forecasts are used. A real desk would add
them, so these are deliberately simple baselines.

Usage
-----
    python backtest.py --db data/nyiso_dam.db --start 2025-01-01 --end 2025-12-31 \
        --nodes WEST CAPITL N.Y.C. LONGIL --out results
"""

from __future__ import annotations

import argparse
import logging
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from data_fetch import MARKET_TZ, load_lmp
from optimizer import BatteryConfig, solve_rolling

log = logging.getLogger(__name__)

FORECASTS = ["naive_d1", "naive_d7", "regression"]
TRAIN_DAYS = 90          # trailing window for the regression
HISTORY_DAYS = 100       # history loaded before the backtest start (lags + training)


# ---------------------------------------------------------------------------
# Forecasts
# ---------------------------------------------------------------------------

def _lag(series: pd.Series, hours: int) -> pd.Series:
    """Value of *series* exactly *hours* earlier (by timestamp, so gaps stay NaN)."""
    shifted = series.copy()
    shifted.index = shifted.index + pd.Timedelta(hours=hours)
    return shifted.reindex(series.index)


def build_features(lmp: pd.DataFrame) -> pd.DataFrame:
    """
    One row per hour with the information available before the DAM for that
    hour's market day closes. Index: interval_start (UTC).
    """
    s = lmp.set_index("interval_start")["lmp"].sort_index()
    local = s.index.tz_convert(MARKET_TZ)
    feats = pd.DataFrame(index=s.index)
    feats["lmp"] = s
    feats["market_day"] = local.date
    feats["hour"] = local.hour
    feats["weekend"] = (local.dayofweek >= 5).astype(float)
    feats["lag24"] = _lag(s, 24)
    feats["lag48"] = _lag(s, 48)
    feats["lag168"] = _lag(s, 168)
    daily_mean = s.groupby(feats["market_day"]).mean()
    prev_day = pd.Series(feats["market_day"]).map(
        lambda d: daily_mean.get(d - timedelta(days=1), np.nan)
    )
    feats["prev_day_mean"] = prev_day.values
    return feats


def _design(f: pd.DataFrame) -> np.ndarray:
    hours = np.zeros((len(f), 24))
    hours[np.arange(len(f)), f["hour"].to_numpy()] = 1.0
    cols = [f[c].to_numpy()[:, None] for c in ("weekend", "lag24", "lag48", "lag168", "prev_day_mean")]
    return np.hstack([hours] + cols)


def regression_forecast(feats: pd.DataFrame, days: list[date], ridge: float = 1.0) -> pd.Series:
    """Refit a small ridge regression each day on the trailing TRAIN_DAYS days."""
    out = pd.Series(np.nan, index=feats.index)
    usable = feats.dropna(subset=["lag24", "lag48", "lag168", "prev_day_mean"])
    for day in days:
        train = usable[(usable["market_day"] < day)
                       & (usable["market_day"] >= day - timedelta(days=TRAIN_DAYS))]
        test = usable[usable["market_day"] == day]
        if len(train) < 24 * 30 or test.empty:
            continue
        X, y = _design(train), train["lmp"].to_numpy()
        penalty = ridge * np.eye(X.shape[1])
        penalty[:24, :24] = 0.0                       # do not shrink the hour-of-day levels
        beta = np.linalg.solve(X.T @ X + penalty, X.T @ y)
        out.loc[test.index] = _design(test) @ beta
    return out


def make_forecasts(lmp: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    """Actual LMPs for [start, end] plus one column per forecast (NaN-free)."""
    feats = build_features(lmp)
    days = [d for d in pd.date_range(start, end).date]
    mask = feats["market_day"].between(start, end)
    out = feats.loc[mask, ["lmp", "market_day"]].copy()
    out["naive_d1"] = feats.loc[mask, "lag24"]
    out["naive_d7"] = feats.loc[mask, "lag168"]
    out["regression"] = regression_forecast(feats, days).loc[mask]
    # A forecast can be missing if the history has a gap: fall back to the
    # next simplest forecast, then to the day's own hour-of-day average so far.
    out["naive_d7"] = out["naive_d7"].fillna(out["naive_d1"])
    out["regression"] = out["regression"].fillna(out["naive_d1"])
    for col in FORECASTS:
        out[col] = out[col].fillna(out[col].ffill()).fillna(out["lmp"].mean())
    return out.reset_index()


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------

def run_zone(lmp: pd.DataFrame, node: str, start: date, end: date,
             cfg: BatteryConfig) -> tuple[pd.DataFrame, list, dict]:
    """Return (daily revenue table, summary rows) for one zone."""
    data = make_forecasts(lmp, start, end)
    n_days = data["market_day"].nunique()

    results = {"perfect": solve_rolling(data, cfg, plan_col="lmp")}
    for fc in FORECASTS:
        results[fc] = solve_rolling(data, cfg, plan_col=fc)

    perfect = results["perfect"]
    rows = []
    for name, res in results.items():
        err = data[name] - data["lmp"] if name != "perfect" else pd.Series(0.0, index=data.index)
        rows.append({
            "zone": node,
            "strategy": name,
            "days": n_days,
            "revenue_usd": res.total_revenue,
            "net_revenue_usd": res.net_revenue,
            "usd_per_kw_year": res.net_revenue / (cfg.power_mw * 1000) * 365 / n_days,
            "capture_ratio": res.net_revenue / perfect.net_revenue,
            "equiv_cycles_per_day": res.equiv_cycles / n_days,
            "mae": err.abs().mean(),
            "rmse": float(np.sqrt((err ** 2).mean())),
        })

    daily = pd.DataFrame({"market_day": data["market_day"]})
    for name, res in results.items():
        disp = res.dispatch
        net = disp["revenue_$"] - cfg.degradation_cost * disp["discharge_mw"]
        daily[name] = net.values
    daily = daily.groupby("market_day").sum()
    daily["zone"] = node
    return daily, rows, results


def run(db: Path, nodes: list[str], start: date, end: date, cfg: BatteryConfig,
        out_dir: Path) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary, dailies, perfect_runs, prices = [], [], {}, {}
    for node in nodes:
        lmp = load_lmp(start - timedelta(days=HISTORY_DAYS), end, node, db)
        if lmp.empty:
            log.warning("%s: no data in %s — skipped", node, db)
            continue
        log.info("%s: %d hours loaded, running backtest...", node, len(lmp))
        daily, rows, results = run_zone(lmp, node, start, end, cfg)
        summary += rows
        dailies.append(daily)
        perfect_runs[node] = results["perfect"]
        prices[node] = lmp
        best = max((r for r in rows if r["strategy"] != "perfect"), key=lambda r: r["capture_ratio"])
        log.info("%s: perfect $%.1f/kW-yr | best forecast %s captures %.0f%%",
                 node, rows[0]["usd_per_kw_year"], best["strategy"], 100 * best["capture_ratio"])

    table = pd.DataFrame(summary)
    table.to_csv(out_dir / "summary.csv", index=False)
    pd.concat(dailies).to_csv(out_dir / "daily_revenue.csv")
    (out_dir / "summary.md").write_text(_markdown_table(table, cfg, start, end))

    from charts import make_charts
    make_charts(table, pd.concat(dailies), perfect_runs, prices, start, end, out_dir)
    return table


def _markdown_table(table: pd.DataFrame, cfg: BatteryConfig, start: date, end: date) -> str:
    lines = [
        f"Backtest {start} to {end}, {cfg.power_mw:g} MW / {cfg.energy_mwh:g} MWh, "
        f"RTE {cfg.rte:.0%}, SoC {cfg.soc_min_frac:.0%}-{cfg.soc_max_frac:.0%}, "
        f"degradation ${cfg.degradation_cost:g}/MWh. Revenue is net of degradation cost.",
        "",
        "| Zone | Strategy | $/kW-year | Capture ratio | Cycles/day | MAE ($/MWh) | RMSE ($/MWh) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in table.itertuples():
        err = "" if r.strategy == "perfect" else f"{r.mae:.2f}"
        rm = "" if r.strategy == "perfect" else f"{r.rmse:.2f}"
        lines.append(f"| {r.zone} | {r.strategy} | {r.usd_per_kw_year:.1f} | "
                     f"{r.capture_ratio:.1%} | {r.equiv_cycles_per_day:.2f} | {err} | {rm} |")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s",
                        datefmt="%H:%M:%S")
    p = argparse.ArgumentParser(description="Forecast vs perfect-foresight BESS backtest",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--db", default="data/nyiso_dam.db")
    p.add_argument("--start", default="2025-01-01")
    p.add_argument("--end", default="2025-12-31")
    p.add_argument("--nodes", nargs="+", default=["WEST", "CAPITL", "N.Y.C.", "LONGIL"])
    p.add_argument("--out", default="results")
    p.add_argument("--mw", type=float, default=10.0)
    p.add_argument("--mwh", type=float, default=40.0)
    p.add_argument("--rte", type=float, default=0.85)
    p.add_argument("--degradation", type=float, default=3.0)
    a = p.parse_args()

    cfg = BatteryConfig(power_mw=a.mw, energy_mwh=a.mwh, rte=a.rte, degradation_cost=a.degradation)
    table = run(Path(a.db), a.nodes, date.fromisoformat(a.start), date.fromisoformat(a.end),
                cfg, Path(a.out))
    print()
    print(table.pivot(index="zone", columns="strategy", values="capture_ratio").round(3).to_string())


if __name__ == "__main__":
    _cli()
