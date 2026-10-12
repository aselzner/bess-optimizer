"""
optimizer.py  —  BESS Dispatch Optimizer
=========================================
Solves a Linear Program (LP) that finds the revenue-maximising charge /
discharge schedule for a battery energy storage system (BESS) over a set
of hourly day-ahead LMP intervals.

Formulation
-----------
Decision variables (continuous, per hour t):
    c_t   charge power   [MW]  ≥ 0
    d_t   discharge power [MW] ≥ 0
    e_t   state of charge [MWh] ≥ 0

Objective (maximise):
    Σ_t  λ_t · (d_t − c_t) · Δt          where Δt = 1 h

Constraints (per hour t):
    SoC balance   e_t = e_{t-1} + √η · c_t − d_t / √η
    SoC bounds    E_min ≤ e_t ≤ E_max
    Power limits  c_t ≤ P_max,  d_t ≤ P_max
    No overlap    c_t + d_t ≤ P_max     (prevents simultaneous charge+discharge)
    Non-negative  c_t ≥ 0,  d_t ≥ 0,  e_t ≥ 0

The RTE is split symmetrically across both sides so total losses = η:
    η_one_way = √η   (e.g. √0.85 ≈ 0.922 per side)

Usage
-----
    python optimizer.py --start 2025-01-01 --end 2025-01-07 --node CAPITL --mock
"""

from __future__ import annotations

import argparse
import logging
import warnings
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd
import pulp

from data_fetch import DEFAULT_DB, generate_mock_lmp, load_lmp

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Battery parameters dataclass
# ---------------------------------------------------------------------------

@dataclass
class BatteryConfig:
    """Physical and operational parameters for the BESS."""

    #: Rated power capacity [MW] — max charge or discharge rate
    power_mw: float = 10.0

    #: Energy capacity [MWh]
    energy_mwh: float = 40.0

    #: Round-trip efficiency (0 < rte ≤ 1)
    rte: float = 0.85

    #: Minimum state of charge as fraction of energy_mwh
    soc_min_frac: float = 0.10

    #: Maximum state of charge as fraction of energy_mwh
    soc_max_frac: float = 0.90

    #: Initial SoC as fraction of energy_mwh (defaults to midpoint)
    soc_init_frac: float = 0.50

    #: If True, force terminal SoC = initial SoC (cyclic constraint)
    cyclic_soc: bool = False

    def __post_init__(self) -> None:
        if not (0 < self.rte <= 1):
            raise ValueError(f"rte must be in (0, 1], got {self.rte}")
        if not (0 <= self.soc_min_frac < self.soc_max_frac <= 1):
            raise ValueError(
                f"Need 0 ≤ soc_min_frac < soc_max_frac ≤ 1, "
                f"got {self.soc_min_frac} and {self.soc_max_frac}"
            )
        if self.power_mw <= 0:
            raise ValueError(f"power_mw must be > 0, got {self.power_mw}")
        if self.energy_mwh <= 0:
            raise ValueError(f"energy_mwh must be > 0, got {self.energy_mwh}")

    @property
    def e_min(self) -> float:
        """Absolute minimum SoC [MWh]."""
        return self.soc_min_frac * self.energy_mwh

    @property
    def e_max(self) -> float:
        """Absolute maximum SoC [MWh]."""
        return self.soc_max_frac * self.energy_mwh

    @property
    def e_init(self) -> float:
        """Initial SoC [MWh]."""
        return self.soc_init_frac * self.energy_mwh

    @property
    def eta_one_way(self) -> float:
        """One-way efficiency = √RTE, applied symmetrically to both sides."""
        return self.rte ** 0.5


# ---------------------------------------------------------------------------
# Optimisation result
# ---------------------------------------------------------------------------

@dataclass
class OptimResult:
    """Return value from solve()."""

    #: Hourly dispatch schedule — see column list in solve() docstring
    dispatch: pd.DataFrame

    #: PuLP solver status string (e.g. "Optimal")
    status: str

    #: Objective value = total revenue over the horizon [$]
    total_revenue: float

    #: Total energy charged [MWh]
    total_charge_mwh: float

    #: Total energy discharged [MWh]
    total_discharge_mwh: float

    #: Equivalent full cycles (discharge MWh / battery energy_mwh)
    equiv_cycles: float

    #: Battery config used
    config: BatteryConfig = field(repr=False)

    def summary(self) -> str:
        lines = [
            f"Status          : {self.status}",
            f"Total revenue   : ${self.total_revenue:,.2f}",
            f"Charge total    : {self.total_charge_mwh:.1f} MWh",
            f"Discharge total : {self.total_discharge_mwh:.1f} MWh",
            f"Equiv. cycles   : {self.equiv_cycles:.2f}",
            f"Hours optimised : {len(self.dispatch)}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core LP solver
# ---------------------------------------------------------------------------

def solve(
    lmp: pd.DataFrame,
    config: Optional[BatteryConfig] = None,
    solver_msg: bool = False,
) -> OptimResult:
    """
    Solve the BESS dispatch LP over the supplied LMP time series.

    Parameters
    ----------
    lmp : pd.DataFrame
        Must have columns: interval_start (tz-aware Timestamp), lmp (float).
        Rows must be contiguous hourly intervals (no gaps).
    config : BatteryConfig, optional
        Battery parameters.  Defaults to a 10 MW / 40 MWh / 85% RTE system.
    solver_msg : bool
        Pass True to print CBC solver output (verbose).

    Returns
    -------
    OptimResult
        Contains dispatch DataFrame and summary statistics.

    Dispatch DataFrame columns
    --------------------------
    interval_start   datetime64[us, UTC]
    lmp              float     $/MWh  day-ahead price
    charge_mw        float     MW     power drawn from grid (positive = charging)
    discharge_mw     float     MW     power delivered to grid
    net_mw           float     MW     net grid injection (discharge − charge)
    soc_mwh          float     MWh    state of charge at END of interval
    revenue_$        float     $      revenue this hour (net_mw × lmp × 1h)
    """
    if config is None:
        config = BatteryConfig()

    lmp = lmp.copy().reset_index(drop=True)
    _validate_lmp(lmp)

    T = len(lmp)
    prices = lmp["lmp"].tolist()
    eta = config.eta_one_way
    P = config.power_mw
    E_min = config.e_min
    E_max = config.e_max
    E_init = config.e_init

    # ── Build the LP ────────────────────────────────────────────────────────
    prob = pulp.LpProblem("BESS_Dispatch", pulp.LpMaximize)

    c = [pulp.LpVariable(f"c_{t}", lowBound=0, upBound=P) for t in range(T)]
    d = [pulp.LpVariable(f"d_{t}", lowBound=0, upBound=P) for t in range(T)]
    e = [pulp.LpVariable(f"e_{t}", lowBound=E_min, upBound=E_max) for t in range(T)]

    # Add to objective: penalise each MWh discharged by degradation cost
    degradation_cost_per_mwh = 3.0   # ~$3/MWh is typical for Li-ion at $200/kWh capex

    # Objective: maximise Σ λ_t · (d_t − c_t) − Σ γ · d_t
    prob += pulp.lpSum(prices[t] * (d[t] - c[t]) - degradation_cost_per_mwh * d[t] for t in range(T))

    # SoC balance: e_t = e_{t-1} + η_ow · c_t − d_t / η_ow
    for t in range(T):
        e_prev = E_init if t == 0 else e[t - 1]
        prob += e[t] == e_prev + eta * c[t] - d[t] / eta

    # No simultaneous charge + discharge
    for t in range(T):
        prob += c[t] + d[t] <= P

    # Cyclic SoC: terminal state = initial state
    if config.cyclic_soc:
        prob += e[T - 1] == E_init

    # ── Solve ────────────────────────────────────────────────────────────────
    solver = pulp.PULP_CBC_CMD(msg=int(solver_msg))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prob.solve(solver)

    status = pulp.LpStatus[prob.status]
    if status != "Optimal":
        log.warning("Solver returned status: %s", status)

    # ── Extract results ──────────────────────────────────────────────────────
    c_vals = [max(pulp.value(c[t]) or 0.0, 0.0) for t in range(T)]
    d_vals = [max(pulp.value(d[t]) or 0.0, 0.0) for t in range(T)]
    e_vals = [pulp.value(e[t]) or 0.0 for t in range(T)]

    net_mw = [d_vals[t] - c_vals[t] for t in range(T)]
    revenue = [prices[t] * net_mw[t] for t in range(T)]  # Δt = 1 h

    dispatch = pd.DataFrame(
        {
            "interval_start": lmp["interval_start"].values,
            "lmp": prices,
            "charge_mw": c_vals,
            "discharge_mw": d_vals,
            "net_mw": net_mw,
            "soc_mwh": e_vals,
            "revenue_$": revenue,
        }
    )

    total_rev = sum(revenue)
    total_chg = sum(c_vals)
    total_dis = sum(d_vals)
    cycles = total_dis / config.energy_mwh if config.energy_mwh > 0 else 0.0

    return OptimResult(
        dispatch=dispatch,
        status=status,
        total_revenue=total_rev,
        total_charge_mwh=total_chg,
        total_discharge_mwh=total_dis,
        equiv_cycles=cycles,
        config=config,
    )


# ---------------------------------------------------------------------------
# Daily rolling optimisation
# ---------------------------------------------------------------------------

def solve_rolling(
    lmp: pd.DataFrame,
    config: Optional[BatteryConfig] = None,
    solver_msg: bool = False,
) -> OptimResult:
    """
    Run day-ahead optimisation day-by-day, passing terminal SoC from one
    day as the initial SoC of the next.

    This more closely reflects real-world operations where the operator
    receives tomorrow's prices and submits bids for each 24-hour period.

    Parameters
    ----------
    lmp : pd.DataFrame
        Multi-day LMP time series (same schema as solve()).
    config : BatteryConfig, optional
    solver_msg : bool

    Returns
    -------
    OptimResult
        Concatenated dispatch across all days.
    """
    if config is None:
        config = BatteryConfig()

    lmp = lmp.copy().reset_index(drop=True)
    _validate_lmp(lmp)

    lmp["_date"] = lmp["interval_start"].dt.date
    days = sorted(lmp["_date"].unique())

    soc_carry = config.e_init
    frames: list[pd.DataFrame] = []

    for day in days:
        day_lmp = lmp[lmp["_date"] == day].drop(columns=["_date"]).reset_index(drop=True)

        # Override initial SoC with carry-over from previous day
        day_cfg = BatteryConfig(
            power_mw=config.power_mw,
            energy_mwh=config.energy_mwh,
            rte=config.rte,
            soc_min_frac=config.soc_min_frac,
            soc_max_frac=config.soc_max_frac,
            soc_init_frac=soc_carry / config.energy_mwh,
            cyclic_soc=False,
        )

        result = solve(day_lmp, day_cfg, solver_msg=solver_msg)
        log.info(
            "  %s  status=%-8s  revenue=$%7.2f  cycles=%.2f",
            day, result.status, result.total_revenue, result.equiv_cycles,
        )

        soc_carry = result.dispatch["soc_mwh"].iloc[-1]
        frames.append(result.dispatch)

    dispatch = pd.concat(frames, ignore_index=True)
    total_rev = dispatch["revenue_$"].sum()
    total_chg = dispatch["charge_mw"].sum()
    total_dis = dispatch["discharge_mw"].sum()
    cycles = total_dis / config.energy_mwh

    return OptimResult(
        dispatch=dispatch,
        status="Optimal",
        total_revenue=total_rev,
        total_charge_mwh=total_chg,
        total_discharge_mwh=total_dis,
        equiv_cycles=cycles,
        config=config,
    )


# ---------------------------------------------------------------------------
# Validation helper
# ---------------------------------------------------------------------------

def _validate_lmp(lmp: pd.DataFrame) -> None:
    required = {"interval_start", "lmp"}
    missing = required - set(lmp.columns)
    if missing:
        raise ValueError(f"LMP DataFrame missing columns: {missing}")
    if len(lmp) == 0:
        raise ValueError("LMP DataFrame is empty.")
    if lmp["lmp"].isna().any():
        raise ValueError("LMP column contains NaN values.")


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
        description="Run BESS dispatch LP optimisation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--start", required=True, metavar="YYYY-MM-DD")
    parser.add_argument("--end",   required=True, metavar="YYYY-MM-DD")
    parser.add_argument("--node",  default="CAPITL")
    parser.add_argument("--db",    default=str(DEFAULT_DB), metavar="PATH")
    parser.add_argument("--mock",  action="store_true",
                        help="Generate synthetic LMP data before optimising")
    parser.add_argument("--mw",    type=float, default=10.0, dest="power_mw")
    parser.add_argument("--mwh",   type=float, default=40.0, dest="energy_mwh")
    parser.add_argument("--rte",   type=float, default=0.85)
    parser.add_argument("--soc-min", type=float, default=0.10, dest="soc_min")
    parser.add_argument("--soc-max", type=float, default=0.90, dest="soc_max")
    parser.add_argument("--rolling", action="store_true",
                        help="Optimise day-by-day (rolling horizon)")
    parser.add_argument("--verbose", action="store_true",
                        help="Show CBC solver output")
    args = parser.parse_args()

    if args.mock:
        generate_mock_lmp(args.start, args.end, node=args.node, db_path=args.db)

    lmp = load_lmp(args.start, args.end, node=args.node, db_path=args.db)
    if lmp.empty:
        print("No LMP data found. Run with --mock or fetch real data first.")
        return

    cfg = BatteryConfig(
        power_mw=args.power_mw,
        energy_mwh=args.energy_mwh,
        rte=args.rte,
        soc_min_frac=args.soc_min,
        soc_max_frac=args.soc_max,
    )

    log.info("Battery: %.0f MW / %.0f MWh  RTE=%.0f%%  SoC=[%.0f%%–%.0f%%]",
             cfg.power_mw, cfg.energy_mwh, cfg.rte * 100,
             cfg.soc_min_frac * 100, cfg.soc_max_frac * 100)

    if args.rolling:
        log.info("Mode: rolling daily optimisation")
        result = solve_rolling(lmp, cfg, solver_msg=args.verbose)
    else:
        log.info("Mode: single-horizon optimisation (%d hours)", len(lmp))
        result = solve(lmp, cfg, solver_msg=args.verbose)

    print(f"\n{'='*45}")
    print(result.summary())
    print(f"{'='*45}")
    print("\nFirst 10 dispatch rows:")
    print(result.dispatch.head(10).to_string(index=False))


if __name__ == "__main__":
    _cli()
