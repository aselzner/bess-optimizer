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
    Σ_t  [ λ_t · (d_t − c_t) − γ · d_t ] · Δt     where Δt = 1 h
    γ = degradation cost per MWh discharged (BatteryConfig.degradation_cost)

Constraints (per hour t):
    SoC balance   e_t = e_{t-1} + √η · c_t − d_t / √η
    SoC bounds    E_min ≤ e_t ≤ E_max
    Power limits  c_t ≤ P_max,  d_t ≤ P_max
    No overlap    c_t + d_t ≤ P_max     (prevents simultaneous charge+discharge)
    Non-negative  c_t ≥ 0,  d_t ≥ 0,  e_t ≥ 0

The RTE is split symmetrically across both sides so total losses = η:
    η_one_way = √η   (e.g. √0.85 ≈ 0.922 per side)

Planning vs settlement prices
-----------------------------
By default the LP plans on the actual day-ahead LMPs (perfect foresight).
Pass ``plan_col="lmp_forecast"`` (any column name) to plan on a forecast
instead; revenue is always settled at the actual ``lmp`` column. The ratio
of the two revenues is the capture ratio (see backtest.py).

Usage
-----
    python optimizer.py --start 2025-01-01 --end 2025-01-07 --node CAPITL --mock
"""

from __future__ import annotations

import argparse
import logging
import warnings
from dataclasses import dataclass, field, replace
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd
import pulp

from data_fetch import DEFAULT_DB, MARKET_TZ, generate_mock_lmp, load_lmp

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

    #: Degradation cost [$ per MWh discharged], charged in the objective so the
    #: LP skips spreads too small to cover battery wear. ~$3/MWh is a common
    #: rough figure for Li-ion; treat it as an assumption.
    degradation_cost: float = 3.0

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
        if self.degradation_cost < 0:
            raise ValueError(f"degradation_cost must be >= 0, got {self.degradation_cost}")

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

    #: Energy revenue settled at actual LMPs over the horizon [$]
    #: (before degradation cost; see net_revenue)
    total_revenue: float

    #: Total energy charged [MWh]
    total_charge_mwh: float

    #: Total energy discharged [MWh]
    total_discharge_mwh: float

    #: Equivalent full cycles (discharge MWh / battery energy_mwh)
    equiv_cycles: float

    #: Battery config used
    config: BatteryConfig = field(repr=False)

    @property
    def degradation_cost_total(self) -> float:
        """Degradation cost over the horizon [$]."""
        return self.config.degradation_cost * self.total_discharge_mwh

    @property
    def net_revenue(self) -> float:
        """Energy revenue minus degradation cost [$]."""
        return self.total_revenue - self.degradation_cost_total

    def summary(self) -> str:
        lines = [
            f"Status          : {self.status}",
            f"Total revenue   : ${self.total_revenue:,.2f}",
            f"Net of degrad.  : ${self.net_revenue:,.2f}",
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
    plan_col: str = "lmp",
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
    plan_col : str
        Column whose prices the LP optimises against. "lmp" (default) is
        perfect foresight; a forecast column gives a realistic schedule.
        Revenue is always settled at the actual "lmp" column.

    Returns
    -------
    OptimResult
        Contains dispatch DataFrame and summary statistics.

    Dispatch DataFrame columns
    --------------------------
    interval_start   datetime64[ns, UTC]
    lmp              float     $/MWh  day-ahead price
    charge_mw        float     MW     power drawn from grid (positive = charging)
    discharge_mw     float     MW     power delivered to grid
    net_mw           float     MW     net grid injection (discharge − charge)
    soc_mwh          float     MWh    state of charge at END of interval
    revenue_$        float     $      revenue this hour (net_mw × lmp × 1h)
    plan_price       float     $/MWh  price the LP planned on (= lmp unless plan_col set)
    """
    if config is None:
        config = BatteryConfig()

    lmp = lmp.copy().reset_index(drop=True)
    _validate_lmp(lmp, plan_col)

    T = len(lmp)
    prices = lmp["lmp"].tolist()                 # settlement prices
    plan = lmp[plan_col].tolist()                # prices the LP sees
    eta = config.eta_one_way
    P = config.power_mw
    E_min = config.e_min
    E_max = config.e_max
    E_init = config.e_init

    # ── Build the LP ────────────────────────────────────────────────────────
    prob = pulp.LpProblem("BESS_Dispatch", pulp.LpMaximize)

    c = [_var(prob, f"c_{t}", 0, P) for t in range(T)]
    d = [_var(prob, f"d_{t}", 0, P) for t in range(T)]
    e = [_var(prob, f"e_{t}", E_min, E_max) for t in range(T)]

    # Objective: maximise Σ λ_t · (d_t − c_t) − Σ γ · d_t   (λ = planning prices)
    gamma = config.degradation_cost
    prob += pulp.lpSum(plan[t] * (d[t] - c[t]) - gamma * d[t] for t in range(T))

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
            "interval_start": lmp["interval_start"],   # keeps the UTC tz
            "lmp": prices,
            "charge_mw": c_vals,
            "discharge_mw": d_vals,
            "net_mw": net_mw,
            "soc_mwh": e_vals,
            "revenue_$": revenue,
            "plan_price": plan,
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
    plan_col: str = "lmp",
) -> OptimResult:
    """
    Run day-ahead optimisation day-by-day, passing terminal SoC from one
    day as the initial SoC of the next. Days are NYISO market days
    (America/New_York), so DST days have 23 or 25 hours.

    Each day is solved on its own, so the LP gives no value to energy left
    in the battery at midnight and tends to end each day near the minimum
    SoC. That is a simplification (see README).

    This more closely reflects real-world operations where the operator
    receives tomorrow's prices and submits bids for each 24-hour period.

    Parameters
    ----------
    lmp : pd.DataFrame
        Multi-day LMP time series (same schema as solve()).
    config : BatteryConfig, optional
    solver_msg : bool
    plan_col : str
        Price column the LP plans on (see solve()).

    Returns
    -------
    OptimResult
        Concatenated dispatch across all days.
    """
    if config is None:
        config = BatteryConfig()

    lmp = lmp.copy().reset_index(drop=True)
    _validate_lmp(lmp, plan_col)

    lmp["_date"] = lmp["interval_start"].dt.tz_convert(MARKET_TZ).dt.date
    days = sorted(lmp["_date"].unique())

    soc_carry = config.e_init
    frames: list[pd.DataFrame] = []

    for day in days:
        day_lmp = lmp[lmp["_date"] == day].drop(columns=["_date"]).reset_index(drop=True)

        # Override initial SoC with carry-over from previous day
        # Clamp guards against tiny solver round-off outside the bounds
        soc_frac = min(max(soc_carry / config.energy_mwh, config.soc_min_frac), config.soc_max_frac)
        day_cfg = replace(config, soc_init_frac=soc_frac, cyclic_soc=False)

        result = solve(day_lmp, day_cfg, solver_msg=solver_msg, plan_col=plan_col)
        log.debug(
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


def _var(prob: pulp.LpProblem, name: str, lo: float, hi: float):
    """Continuous variable; works with PuLP 2.x/3.x and the newer add_variable API."""
    if hasattr(prob, "add_variable"):
        return prob.add_variable(name, lo, hi)
    return pulp.LpVariable(name, lowBound=lo, upBound=hi)


# ---------------------------------------------------------------------------
# Validation helper
# ---------------------------------------------------------------------------

def _validate_lmp(lmp: pd.DataFrame, plan_col: str = "lmp") -> None:
    required = {"interval_start", "lmp", plan_col}
    missing = required - set(lmp.columns)
    if missing:
        raise ValueError(f"LMP DataFrame missing columns: {missing}")
    if len(lmp) == 0:
        raise ValueError("LMP DataFrame is empty.")
    if lmp["lmp"].isna().any():
        raise ValueError("LMP column contains NaN values.")
    if lmp[plan_col].isna().any():
        raise ValueError(f"Planning price column '{plan_col}' contains NaN values.")


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
    parser.add_argument("--degradation", type=float, default=3.0,
                        help="Degradation cost, $ per MWh discharged")
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
        degradation_cost=args.degradation,
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
