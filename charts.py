"""
charts.py  —  BESS Dispatch Optimizer
=====================================
Static PNG charts for the backtest results (used in the README).

Colours follow one rule: the same colour always means the same strategy.
    perfect foresight  grey (reference)
    naive_d1           blue
    naive_d7           orange
    regression         aqua
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_fetch import MARKET_TZ
from optimizer import solve_rolling

INK, INK2, GRID, BG = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]      # validated categorical order
STRATEGY_COLORS = {"perfect": "#8a8984", "naive_d1": SERIES[0],
                   "naive_d7": SERIES[1], "regression": SERIES[2]}
LABELS = {"perfect": "perfect foresight", "naive_d1": "naive: yesterday",
          "naive_d7": "naive: last week", "regression": "regression"}
FORECASTS = ["naive_d1", "naive_d7", "regression"]

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": GRID,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "legend.frameon": False, "font.size": 10,
})


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def chart_capture(table: pd.DataFrame, path: Path) -> None:
    zones = list(dict.fromkeys(table["zone"]))
    fig, ax = plt.subplots(figsize=(10, 4))
    width = 0.26
    x = np.arange(len(zones))
    for i, fc in enumerate(FORECASTS):
        vals = [table[(table.zone == z) & (table.strategy == fc)]["capture_ratio"].iloc[0] for z in zones]
        bars = ax.bar(x + (i - 1) * width, vals, width * 0.92, color=STRATEGY_COLORS[fc], label=LABELS[fc])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.0%}", ha="center", fontsize=8, color=INK2)
    ax.set_xticks(x, zones)
    ax.set_ylim(0, 1.08)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_ylabel("share of perfect-foresight revenue")
    ax.set_title("Capture ratio: revenue when planning on a forecast vs. on actual prices")
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=3, loc="upper left")
    _save(fig, path)


def chart_revenue(table: pd.DataFrame, path: Path) -> None:
    zones = list(dict.fromkeys(table["zone"]))
    strategies = ["perfect"] + FORECASTS
    fig, ax = plt.subplots(figsize=(10, 4))
    width = 0.2
    x = np.arange(len(zones))
    for i, st in enumerate(strategies):
        vals = [table[(table.zone == z) & (table.strategy == st)]["usd_per_kw_year"].iloc[0] for z in zones]
        ax.bar(x + (i - 1.5) * width, vals, width * 0.92, color=STRATEGY_COLORS[st], label=LABELS[st])
    ax.set_xticks(x, zones)
    ax.set_ylabel("$ per kW-year (net of degradation)")
    ax.set_title("Day-ahead arbitrage revenue by zone")
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=4, loc="upper left")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
    _save(fig, path)


def chart_monthly(daily: pd.DataFrame, path: Path) -> None:
    zones = list(dict.fromkeys(daily["zone"]))
    n = len(zones)
    cols = 2 if n > 1 else 1
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(11, 3.2 * rows), squeeze=False)
    for ax, zone in zip(axes.flat, zones):
        d = daily[daily["zone"] == zone].copy()
        d.index = pd.to_datetime(d.index)
        monthly = d[["perfect"] + FORECASTS].resample("MS").sum() / 1000
        for st in ["perfect"] + FORECASTS:
            ax.plot(monthly.index, monthly[st], color=STRATEGY_COLORS[st], lw=2 if st == "perfect" else 1.6,
                    label=LABELS[st], marker="o", ms=3)
        ax.set_title(zone)
        ax.set_ylabel("thousand $ per month")
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b"))
    for ax in list(axes.flat)[n:]:
        ax.set_visible(False)
    axes.flat[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("Monthly net revenue, 10 MW / 40 MWh battery", x=0.01, ha="left", fontweight="bold")
    _save(fig, path)


def chart_price_shape(prices: dict, start: date, end: date, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    for i, (zone, lmp) in enumerate(prices.items()):
        s = lmp.set_index("interval_start")["lmp"]
        local = s.index.tz_convert(MARKET_TZ)
        keep = (local.date >= start) & (local.date <= end)
        prof = s[keep].groupby(local[keep].hour).mean()
        ax.plot(prof.index, prof.values, color=SERIES[i % len(SERIES)], lw=2, label=zone)
    ax.set_xticks([0, 6, 12, 18, 23])
    ax.legend(ncol=len(prices), loc="upper left")
    ax.set_xlabel("hour of day (Eastern time, hour beginning)")
    ax.set_ylabel("average DA LMP ($/MWh)")
    ax.set_title(f"Average day-ahead price by hour, {start:%b %Y} to {end:%b %Y}")
    ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1] * 1.08)
    _save(fig, path)


def chart_example_week(prices: dict, zone: str, start: date, end: date, path: Path) -> None:
    """A week of perfect-foresight dispatch, drawn with reporting.figure_dispatch."""
    from reporting import figure_dispatch

    lmp = prices[zone]
    local_day = lmp["interval_start"].dt.tz_convert(MARKET_TZ).dt.date
    in_range = lmp[(local_day >= start) & (local_day <= end)]
    if in_range.empty:
        return
    # the week with the highest average price, a good showcase of a spread
    weekly = in_range.set_index("interval_start")["lmp"].resample("W-MON").mean()
    week_start = (weekly.idxmax() - pd.Timedelta(days=7)).tz_convert(MARKET_TZ).date()
    wk = lmp[(local_day >= week_start) & (local_day < week_start + timedelta(days=7))]
    res = solve_rolling(wk.reset_index(drop=True))
    fig = figure_dispatch(res)
    fig.suptitle(f"{zone}, week of {week_start:%d %b %Y} — perfect-foresight dispatch "
                 f"(revenue ${res.total_revenue:,.0f}, {res.equiv_cycles:.1f} equiv. cycles)",
                 fontsize=10, y=0.98)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def make_charts(table, daily, perfect_runs, prices, start, end, out_dir: Path) -> None:
    out_dir = Path(out_dir)
    chart_capture(table, out_dir / "capture_ratio.png")
    chart_revenue(table, out_dir / "revenue_by_zone.png")
    chart_monthly(daily, out_dir / "monthly_revenue.png")
    chart_price_shape(prices, start, end, out_dir / "price_shape.png")
    zone = "N.Y.C." if "N.Y.C." in prices else next(iter(prices))
    chart_example_week(prices, zone, start, end, out_dir / "example_week.png")
