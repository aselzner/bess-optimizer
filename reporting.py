"""
reporting.py  —  BESS Dispatch Optimizer
=========================================
Generates a multi-panel matplotlib P&L dashboard from an OptimResult.

Outputs
-------
  Figure 1 — Dispatch overview   (price + actions, MW stack, SoC trace)
  Figure 2 — P&L summary         (daily revenue bars, cumulative, cycling)
  Figure 3 — Price vs dispatch   (scatter: did we charge cheap / discharge dear?)
  Figure 4 — Performance table   (per-day stats as a formatted table figure)

Usage
-----
    python reporting.py --start 2025-01-01 --end 2025-01-14 --node CAPITL --mock
    python reporting.py --start 2025-01-01 --end 2025-01-14 --save          # saves PNGs
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from data_fetch import DEFAULT_DB, generate_mock_lmp, load_lmp
from optimizer import BatteryConfig, OptimResult, solve_rolling

log = logging.getLogger(__name__)

# ── Palette ──────────────────────────────────────────────────────────────────
C_CHARGE    = "#378ADD"   # blue  — charging (buying)
C_DISCHARGE = "#D85A30"   # coral — discharging (selling)
C_IDLE      = "#B4B2A9"   # gray  — idle
C_SOC       = "#1D9E75"   # teal  — state of charge
C_LMP       = "#534AB7"   # purple — price line
C_POS       = "#3B6D11"   # green — positive revenue
C_NEG       = "#A32D2D"   # red   — negative revenue
C_CUM       = "#185FA5"   # blue  — cumulative line


# ---------------------------------------------------------------------------
# Figure 1 — Dispatch overview
# ---------------------------------------------------------------------------

def plot_dispatch(
    result: OptimResult,
    ax_price: plt.Axes,
    ax_power: plt.Axes,
    ax_soc: plt.Axes,
    title: str = "",
) -> None:
    """
    Fill three pre-created axes with:
      ax_price  — LMP line + charge/discharge shading
      ax_power  — stacked charge (negative) / discharge (positive) bars
      ax_soc    — state-of-charge fill with bounds
    """
    df  = result.dispatch
    cfg = result.config
    ts  = df["interval_start"]
    dt  = pd.Timedelta("50min")   # bar width

    # ── Price + action shading ────────────────────────────────────────────
    ax_price.plot(ts, df["lmp"], color=C_LMP, lw=1.0, zorder=3, label="DA LMP")
    ax_price.axhline(df["lmp"].mean(), color=C_LMP, lw=0.6, ls="--", alpha=0.5,
                     label=f"Mean ${df['lmp'].mean():.1f}/MWh")

    for _, row in df[df["discharge_mw"] > 0.05].iterrows():
        ax_price.axvspan(row["interval_start"],
                         row["interval_start"] + pd.Timedelta("1h"),
                         color=C_DISCHARGE, alpha=0.12, lw=0)
    for _, row in df[df["charge_mw"] > 0.05].iterrows():
        ax_price.axvspan(row["interval_start"],
                         row["interval_start"] + pd.Timedelta("1h"),
                         color=C_CHARGE, alpha=0.10, lw=0)

    ax_price.set_ylabel("LMP ($/MWh)", fontsize=9)
    ax_price.legend(fontsize=8, loc="upper right")
    ax_price.grid(axis="y", alpha=0.25, lw=0.5)
    if title:
        ax_price.set_title(title, fontsize=10, pad=6)

    # ── Power bars ───────────────────────────────────────────────────────
    ax_power.bar(ts,  df["discharge_mw"], width=dt, color=C_DISCHARGE,
                 alpha=0.85, label="Discharge", zorder=2)
    ax_power.bar(ts, -df["charge_mw"],    width=dt, color=C_CHARGE,
                 alpha=0.85, label="Charge (−ve)",   zorder=2)
    ax_power.axhline(0, color="gray", lw=0.5)
    ax_power.set_ylabel("Power (MW)", fontsize=9)
    ax_power.legend(fontsize=8, loc="upper right")
    ax_power.grid(axis="y", alpha=0.25, lw=0.5)

    # ── SoC ──────────────────────────────────────────────────────────────
    ax_soc.fill_between(ts, cfg.e_min, df["soc_mwh"],
                        color=C_SOC, alpha=0.35, lw=0)
    ax_soc.plot(ts, df["soc_mwh"], color=C_SOC, lw=1.1, label="SoC")
    ax_soc.axhline(cfg.e_min, color="gray", lw=0.7, ls="--",
                   label=f"Min {cfg.e_min:.0f} MWh")
    ax_soc.axhline(cfg.e_max, color="gray", lw=0.7, ls=":",
                   label=f"Max {cfg.e_max:.0f} MWh")
    ax_soc.set_ylabel("SoC (MWh)", fontsize=9)
    ax_soc.set_xlabel("Hour (UTC)", fontsize=9)
    ax_soc.set_ylim(0, cfg.energy_mwh * 1.05)
    ax_soc.legend(fontsize=8, loc="upper right")
    ax_soc.grid(axis="y", alpha=0.25, lw=0.5)

    for ax in (ax_price, ax_power, ax_soc):
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax.xaxis.set_major_locator(mdates.DayLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30,
                 ha="right", fontsize=8)
        ax.tick_params(axis="y", labelsize=8)


def figure_dispatch(result: OptimResult) -> plt.Figure:
    cfg = result.config
    title = (f"BESS Dispatch — {cfg.power_mw:.0f} MW / {cfg.energy_mwh:.0f} MWh  "
             f"RTE={cfg.rte:.0%}  |  Total revenue ${result.total_revenue:,.0f}  "
             f"|  {result.equiv_cycles:.1f} equiv. cycles")

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True,
                              gridspec_kw={"height_ratios": [3, 2, 2]})
    fig.suptitle(title, fontsize=10, y=0.98)
    plot_dispatch(result, *axes)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


# ---------------------------------------------------------------------------
# Figure 2 — P&L summary
# ---------------------------------------------------------------------------

def figure_pnl(result: OptimResult) -> plt.Figure:
    df = result.dispatch.copy()
    df["date"] = df["interval_start"].dt.date

    daily = df.groupby("date").agg(
        revenue    = ("revenue_$",    "sum"),
        discharge  = ("discharge_mw", "sum"),
        charge     = ("charge_mw",    "sum"),
        avg_lmp    = ("lmp",          "mean"),
        peak_lmp   = ("lmp",          "max"),
    ).reset_index()
    daily["cycles"]   = daily["discharge"] / result.config.energy_mwh
    daily["cum_rev"]  = daily["revenue"].cumsum()
    daily["spread"]   = daily["peak_lmp"] - daily["avg_lmp"]

    dates  = [str(d) for d in daily["date"]]
    x      = np.arange(len(daily))
    colors = [C_POS if r >= 0 else C_NEG for r in daily["revenue"]]

    fig = plt.figure(figsize=(14, 9))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.32)
    ax1 = fig.add_subplot(gs[0, :])   # daily revenue — full width
    ax2 = fig.add_subplot(gs[1, 0])   # cumulative revenue
    ax3 = fig.add_subplot(gs[1, 1])   # cycling

    fig.suptitle("P&L Summary", fontsize=11, y=0.99)

    # ── Daily revenue bars ────────────────────────────────────────────────
    bars = ax1.bar(x, daily["revenue"], color=colors, alpha=0.85, width=0.65)
    ax1.axhline(0, color="gray", lw=0.5)
    ax1.axhline(daily["revenue"].mean(), color=C_POS, lw=0.8, ls="--",
                label=f"Mean ${daily['revenue'].mean():,.0f}/day")
    for bar, rev in zip(bars, daily["revenue"]):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 rev + (max(daily["revenue"]) * 0.02),
                 f"${rev:,.0f}", ha="center", va="bottom", fontsize=7.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(dates, rotation=30, ha="right", fontsize=8)
    ax1.set_ylabel("Daily revenue ($)", fontsize=9)
    ax1.set_title("Daily revenue", fontsize=9)
    ax1.legend(fontsize=8)
    ax1.grid(axis="y", alpha=0.25, lw=0.5)
    ax1.tick_params(axis="y", labelsize=8)

    # ── Cumulative revenue ────────────────────────────────────────────────
    ax2.fill_between(x, 0, daily["cum_rev"], color=C_CUM, alpha=0.15)
    ax2.plot(x, daily["cum_rev"], color=C_CUM, lw=1.5, marker="o",
             markersize=4, label="Cumulative")
    ax2.set_xticks(x)
    ax2.set_xticklabels(dates, rotation=40, ha="right", fontsize=7)
    ax2.set_ylabel("Cumulative revenue ($)", fontsize=9)
    ax2.set_title("Cumulative revenue", fontsize=9)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: f"${v:,.0f}"))
    ax2.grid(alpha=0.25, lw=0.5)
    ax2.tick_params(axis="y", labelsize=8)

    # ── Daily cycling ─────────────────────────────────────────────────────
    ax3b = ax3.twinx()
    ax3.bar(x, daily["cycles"], color=C_SOC, alpha=0.7, width=0.55,
            label="Cycles/day")
    ax3b.plot(x, daily["revenue"] / daily["cycles"].replace(0, np.nan),
              color=C_DISCHARGE, lw=1.3, marker="s", markersize=4,
              label="$/cycle")
    ax3.set_xticks(x)
    ax3.set_xticklabels(dates, rotation=40, ha="right", fontsize=7)
    ax3.set_ylabel("Equiv. cycles / day", fontsize=9, color=C_SOC)
    ax3b.set_ylabel("Revenue / cycle ($)", fontsize=9, color=C_DISCHARGE)
    ax3.set_title("Cycling & revenue per cycle", fontsize=9)
    ax3.tick_params(axis="y", labelsize=8, colors=C_SOC)
    ax3b.tick_params(axis="y", labelsize=8, colors=C_DISCHARGE)

    h1, l1 = ax3.get_legend_handles_labels()
    h2, l2 = ax3b.get_legend_handles_labels()
    ax3.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper right")
    ax3.grid(axis="y", alpha=0.25, lw=0.5)

    return fig


# ---------------------------------------------------------------------------
# Figure 3 — Price vs dispatch scatter
# ---------------------------------------------------------------------------

def figure_price_vs_dispatch(result: OptimResult) -> plt.Figure:
    df  = result.dispatch.copy()
    cfg = result.config

    charging    = df[df["charge_mw"]    > 0.05]
    discharging = df[df["discharge_mw"] > 0.05]
    idle        = df[(df["charge_mw"] <= 0.05) & (df["discharge_mw"] <= 0.05)]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Price vs Dispatch — did the optimizer buy low and sell high?",
                 fontsize=10)

    # ── Left: scatter LMP vs action ──────────────────────────────────────
    ax = axes[0]
    ax.scatter(idle["lmp"],        idle["soc_mwh"],
               color=C_IDLE,      alpha=0.4, s=18, label="Idle",      zorder=2)
    ax.scatter(charging["lmp"],    charging["soc_mwh"],
               color=C_CHARGE,    alpha=0.7, s=28, label="Charging",  zorder=3)
    ax.scatter(discharging["lmp"], discharging["soc_mwh"],
               color=C_DISCHARGE, alpha=0.7, s=28, label="Discharging", zorder=3)

    ax.axvline(df["lmp"].mean(), color="gray", lw=0.7, ls="--",
               label=f"Mean LMP ${df['lmp'].mean():.1f}")
    ax.set_xlabel("LMP ($/MWh)", fontsize=9)
    ax.set_ylabel("SoC at end of hour (MWh)", fontsize=9)
    ax.set_title("LMP vs SoC by action", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, lw=0.5)
    ax.tick_params(labelsize=8)

    # ── Right: LMP histogram by action ──────────────────────────────────
    ax2 = axes[1]
    bins = np.linspace(df["lmp"].min(), df["lmp"].quantile(0.98), 30)

    ax2.hist(idle["lmp"],        bins=bins, color=C_IDLE,      alpha=0.5,
             label=f"Idle (n={len(idle)})",          density=True)
    ax2.hist(charging["lmp"],    bins=bins, color=C_CHARGE,    alpha=0.7,
             label=f"Charging (n={len(charging)})",  density=True)
    ax2.hist(discharging["lmp"], bins=bins, color=C_DISCHARGE, alpha=0.7,
             label=f"Discharging (n={len(discharging)})", density=True)

    ax2.axvline(charging["lmp"].mean()    if len(charging)    > 0 else 0,
                color=C_CHARGE,    lw=1.2, ls="--")
    ax2.axvline(discharging["lmp"].mean() if len(discharging) > 0 else 0,
                color=C_DISCHARGE, lw=1.2, ls="--")

    ax2.set_xlabel("LMP ($/MWh)", fontsize=9)
    ax2.set_ylabel("Density", fontsize=9)
    ax2.set_title("Price distribution by action", fontsize=9)
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.25, lw=0.5)
    ax2.tick_params(labelsize=8)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 4 — Performance stats table
# ---------------------------------------------------------------------------

def figure_stats_table(result: OptimResult) -> plt.Figure:
    df  = result.dispatch.copy()
    cfg = result.config
    df["date"] = df["interval_start"].dt.date

    daily = df.groupby("date").agg(
        revenue   = ("revenue_$",    "sum"),
        discharge = ("discharge_mw", "sum"),
        charge    = ("charge_mw",    "sum"),
        avg_lmp   = ("lmp",          "mean"),
        peak_lmp  = ("lmp",          "max"),
        min_lmp   = ("lmp",          "min"),
    ).reset_index()
    daily["cycles"]   = (daily["discharge"] / cfg.energy_mwh).round(2)
    daily["spread"]   = (daily["peak_lmp"] - daily["min_lmp"]).round(1)
    daily["$/cycle"]  = (daily["revenue"] / daily["cycles"].replace(0, np.nan)).round(0)

    # Totals row
    totals = {
        "date":      "TOTAL",
        "revenue":   daily["revenue"].sum(),
        "discharge": daily["discharge"].sum(),
        "charge":    daily["charge"].sum(),
        "avg_lmp":   daily["avg_lmp"].mean(),
        "peak_lmp":  daily["peak_lmp"].max(),
        "min_lmp":   daily["min_lmp"].min(),
        "cycles":    daily["cycles"].sum(),
        "spread":    daily["spread"].mean(),
        "$/cycle":   daily["revenue"].sum() / daily["cycles"].sum(),
    }
    display = pd.concat([daily, pd.DataFrame([totals])], ignore_index=True)

    col_labels = ["Date", "Revenue ($)", "Discharge\n(MWh)", "Charge\n(MWh)",
                  "Avg LMP\n($/MWh)", "Spread\n($/MWh)", "Cycles", "$/cycle"]
    rows = []
    for _, r in display.iterrows():
        rows.append([
            str(r["date"]),
            f"${r['revenue']:,.0f}",
            f"{r['discharge']:.1f}",
            f"{r['charge']:.1f}",
            f"${r['avg_lmp']:.1f}",
            f"${r['spread']:.1f}",
            f"{r['cycles']:.2f}",
            f"${r['$/cycle']:,.0f}" if not np.isnan(r["$/cycle"]) else "—",
        ])

    fig, ax = plt.subplots(figsize=(14, max(4, len(rows) * 0.45 + 1.5)))
    ax.axis("off")
    tbl = ax.table(
        cellText=rows,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1, 1.5)

    # Style header
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor("#2C2C2A")
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    # Style total row
    total_row = len(rows)
    for j in range(len(col_labels)):
        tbl[total_row, j].set_facecolor("#E6F1FB")
        tbl[total_row, j].set_text_props(fontweight="bold")

    # Alternating row shading
    for i in range(1, len(rows)):
        color = "#F8F8F6" if i % 2 == 0 else "white"
        for j in range(len(col_labels)):
            tbl[i, j].set_facecolor(color)

    ax.set_title("Daily performance summary", fontsize=10, pad=12)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main dashboard function
# ---------------------------------------------------------------------------

def build_dashboard(
    result: OptimResult,
    save_dir: Optional[Path] = None,
    show: bool = True,
) -> list[plt.Figure]:
    """
    Generate all four report figures.

    Parameters
    ----------
    result : OptimResult
    save_dir : Path, optional
        If provided, saves each figure as a PNG in this directory.
    show : bool
        If True, calls plt.show() after building all figures.

    Returns
    -------
    list of Figure objects
    """
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "font.family":    "sans-serif",
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "figure.facecolor":   "white",
        "axes.facecolor":     "white",
    })

    figs = [
        ("dispatch",         figure_dispatch(result)),
        ("pnl_summary",      figure_pnl(result)),
        ("price_vs_dispatch", figure_price_vs_dispatch(result)),
        ("stats_table",      figure_stats_table(result)),
    ]

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        for name, fig in figs:
            path = save_dir / f"bess_{name}.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            log.info("Saved %s", path)

    if show:
        plt.show()

    return [fig for _, fig in figs]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Generate BESS P&L dashboard",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--start",  required=True, metavar="YYYY-MM-DD")
    parser.add_argument("--end",    required=True, metavar="YYYY-MM-DD")
    parser.add_argument("--node",   default="CAPITL")
    parser.add_argument("--db",     default=str(DEFAULT_DB), metavar="PATH")
    parser.add_argument("--mock",   action="store_true")
    parser.add_argument("--mw",     type=float, default=10.0,  dest="power_mw")
    parser.add_argument("--mwh",    type=float, default=40.0,  dest="energy_mwh")
    parser.add_argument("--rte",    type=float, default=0.85)
    parser.add_argument("--save",   metavar="DIR", default=None,
                        help="Save PNGs to this directory")
    parser.add_argument("--no-show", action="store_true",
                        help="Don't open interactive windows (useful with --save)")
    args = parser.parse_args()

    if args.mock:
        generate_mock_lmp(args.start, args.end, node=args.node, db_path=args.db)

    lmp = load_lmp(args.start, args.end, node=args.node, db_path=args.db)
    if lmp.empty:
        print("No LMP data. Run with --mock or fetch real data first.")
        return

    cfg = BatteryConfig(
        power_mw=args.power_mw,
        energy_mwh=args.energy_mwh,
        rte=args.rte,
    )

    log.info("Optimising %d hours...", len(lmp))
    result = solve_rolling(lmp, cfg)
    log.info(result.summary())

    build_dashboard(
        result,
        save_dir=Path(args.save) if args.save else None,
        show=not args.no_show,
    )


if __name__ == "__main__":
    _cli()
