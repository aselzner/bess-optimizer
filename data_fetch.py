"""
data_fetch.py  —  BESS Dispatch Optimizer
==========================================
Fetches NYISO Day-Ahead Market (DAM) zonal Locational Based Marginal Prices
(LBMPs) from the public NYISO MIS portal and persists them to a local SQLite
database for use by the optimizer.

NYISO public data URL patterns (dataset P-2A, DAM zonal LBMP)
--------------------------------------------------------------
    Monthly archive : https://mis.nyiso.com/public/csv/damlbmp/{YYYYMM}01damlbmp_zone_csv.zip
                      (one CSV per day of the month inside the ZIP)
    Single day      : https://mis.nyiso.com/public/csv/damlbmp/{YYYYMMDD}damlbmp_zone.csv

Past months are downloaded as one ZIP each; days in the current month fall
back to the daily CSV files.

Each CSV has one row per zone per hour:
    Time Stamp | Name | PTID | LBMP ($/MWHr) |
    Marginal Cost Losses ($/MWHr) | Marginal Cost Congestion ($/MWHr)

"Time Stamp" is hour-beginning, in Eastern Prevailing Time. It is stored as
UTC. Date ranges in this module are NYISO market days (America/New_York).

This module exposes the public entry-points:
    fetch_lmp(start, end, node, db_path)        → pd.DataFrame
    fetch_many(start, end, nodes, db_path)      → None (one download per month for all nodes)
    load_lmp(start, end, node, db_path)         → pd.DataFrame  (DB-only, no HTTP)

Usage
-----
    python data_fetch.py --start 2025-01-01 --end 2025-12-31 --node CAPITL
    python data_fetch.py --start 2025-01-01 --end 2025-12-31 --nodes CAPITL N.Y.C. WEST
"""

from __future__ import annotations

import argparse
import io
import logging
import sqlite3
import zipfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

#: Base URL for NYISO day-ahead LBMP bulk downloads
NYISO_DAM_BASE = "https://mis.nyiso.com/public/csv/damlbmp"

#: NYISO market time zone (Eastern Prevailing Time)
MARKET_TZ = "America/New_York"

#: Default SQLite database path (relative to CWD)
DEFAULT_DB = Path("bess_lmp.db")

#: All available NYISO zone names (case-insensitive in queries)
NYISO_ZONES = [
    "CAPITL",
    "CENTRL",
    "DUNWOD",
    "GENESE",
    "H Q",
    "HUD VL",
    "LONGIL",
    "MHK VL",
    "MILLWD",
    "N.Y.C.",
    "NORTH",
    "NPX",
    "O H",
    "PJM",
    "WEST",
]

#: Schema version — bump when the table layout changes
_SCHEMA_VERSION = 1

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def _connect(db_path: Path) -> sqlite3.Connection:
    """Return a SQLite connection with WAL mode and foreign keys on."""
    conn = sqlite3.connect(db_path, detect_types=sqlite3.PARSE_DECLTYPES)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _init_db(conn: sqlite3.Connection) -> None:
    """Create tables and indexes if they do not already exist."""
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS schema_meta (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS lmp (
            interval_start  TEXT    NOT NULL,   -- ISO-8601 UTC, hour-beginning
            node            TEXT    NOT NULL,   -- NYISO zone name, upper-cased
            lmp             REAL    NOT NULL,   -- $/MWh
            mcl             REAL,               -- Marginal cost of losses  $/MWh
            mcc             REAL,               -- Marginal cost of congestion $/MWh
            fetched_at      TEXT    NOT NULL,   -- ISO-8601 UTC fetch timestamp
            PRIMARY KEY (interval_start, node)
        );

        CREATE INDEX IF NOT EXISTS idx_lmp_node_interval
            ON lmp (node, interval_start);
        """
    )
    conn.execute(
        "INSERT OR IGNORE INTO schema_meta VALUES ('version', ?)",
        (str(_SCHEMA_VERSION),),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Market-day helpers
# ---------------------------------------------------------------------------

def _as_date(d: date | str) -> date:
    return date.fromisoformat(d) if isinstance(d, str) else d


def _date_range(start: date, end: date):
    """Yield each date from start to end inclusive."""
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)


def _market_day_bounds_utc(start: date, end: date) -> tuple[pd.Timestamp, pd.Timestamp]:
    """UTC instants of local midnight on *start* and local midnight after *end*."""
    lo = pd.Timestamp(start).tz_localize(MARKET_TZ).tz_convert("UTC")
    hi = pd.Timestamp(end + timedelta(days=1)).tz_localize(MARKET_TZ).tz_convert("UTC")
    return lo, hi


def _hours_in_market_day(day: date) -> int:
    """23 on the spring-forward day, 25 on the fall-back day, else 24."""
    lo, hi = _market_day_bounds_utc(day, day)
    return int((hi - lo) / pd.Timedelta("1h"))


# ---------------------------------------------------------------------------
# HTTP / parse helpers
# ---------------------------------------------------------------------------

def _monthly_zip_url(month_start: date) -> str:
    return f"{NYISO_DAM_BASE}/{month_start:%Y%m}01damlbmp_zone_csv.zip"


def _daily_csv_url(day: date) -> str:
    return f"{NYISO_DAM_BASE}/{day:%Y%m%d}damlbmp_zone.csv"


def _find_col(columns: Iterable[str], *needles: str) -> Optional[str]:
    """First column whose lower-cased name contains all *needles*."""
    for col in columns:
        low = col.lower()
        if all(n in low for n in needles):
            return col
    return None


def _parse_csv(raw: pd.DataFrame, nodes: Optional[set[str]] = None) -> pd.DataFrame:
    """
    Normalise one NYISO DAM zonal LBMP CSV.

    Column names are matched loosely ("LBMP ($/MWHr)", "DAM Zonal LBMP",
    "Marginal Cost Losses ($/MWHr)", ...) because NYISO has changed the
    exact labels over time.

    Returns columns: interval_start (UTC), node, lmp, mcl, mcc
    """
    df = raw.copy()
    df.columns = [str(c).strip() for c in df.columns]

    ts_col = _find_col(df.columns, "time stamp") or _find_col(df.columns, "timestamp")
    name_col = _find_col(df.columns, "name")
    mcl_col = _find_col(df.columns, "losses")
    mcc_col = _find_col(df.columns, "congestion")
    lmp_col = next(
        (c for c in df.columns if "lbmp" in c.lower() and c not in (mcl_col, mcc_col)),
        None,
    )
    if not (ts_col and name_col and lmp_col):
        raise ValueError(f"Unrecognised NYISO CSV columns: {list(df.columns)}")

    out = pd.DataFrame(
        {
            "local_ts": pd.to_datetime(df[ts_col], format="%m/%d/%Y %H:%M", errors="coerce"),
            "node": df[name_col].astype(str).str.strip().str.upper(),
            "lmp": pd.to_numeric(df[lmp_col], errors="coerce"),
            "mcl": pd.to_numeric(df[mcl_col], errors="coerce") if mcl_col else float("nan"),
            "mcc": pd.to_numeric(df[mcc_col], errors="coerce") if mcc_col else float("nan"),
        }
    )
    if out["local_ts"].isna().all():                       # other timestamp layouts
        out["local_ts"] = pd.to_datetime(df[ts_col], errors="coerce")
    out = out.dropna(subset=["local_ts", "lmp"])

    if nodes is not None:
        out = out[out["node"].isin(nodes)]
    if out.empty:
        return pd.DataFrame(columns=["interval_start", "node", "lmp", "mcl", "mcc"])

    # Hour-beginning Eastern time → UTC. On the fall-back day the 01:00 hour
    # appears twice per zone; "infer" uses the row order to tell them apart,
    # so localise each zone separately, in file order.
    parts = []
    for _, grp in out.groupby("node", sort=False):
        grp = grp.copy()
        grp["interval_start"] = (
            grp["local_ts"]
            .dt.tz_localize(MARKET_TZ, ambiguous="infer", nonexistent="shift_forward")
            .dt.tz_convert("UTC")
        )
        parts.append(grp)
    out = pd.concat(parts, ignore_index=True)

    out = out.sort_values(["node", "interval_start"]).drop_duplicates(
        subset=["interval_start", "node"]
    )
    return out[["interval_start", "node", "lmp", "mcl", "mcc"]].reset_index(drop=True)


def _parse_zip(content: bytes, nodes: Optional[set[str]] = None) -> pd.DataFrame:
    """Parse a NYISO monthly ZIP archive (one CSV per day)."""
    frames: list[pd.DataFrame] = []
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        csv_names = sorted(n for n in zf.namelist() if n.lower().endswith(".csv"))
        if not csv_names:
            raise ValueError("ZIP archive contains no CSV files.")
        for csv_name in csv_names:
            with zf.open(csv_name) as fh:
                frames.append(_parse_csv(pd.read_csv(fh), nodes))
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame(columns=["interval_start", "node", "lmp", "mcl", "mcc"])
    return pd.concat(frames, ignore_index=True)


def _get(session: requests.Session, url: str, timeout: int) -> bytes:
    log.info("GET %s", url)
    resp = session.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.content


def _upsert(conn: sqlite3.Connection, df: pd.DataFrame) -> int:
    """Insert or replace rows in the lmp table. Returns row count written."""
    if df.empty:
        return 0

    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rows = [
        (
            row.interval_start.isoformat(),
            row.node,
            row.lmp,
            row.mcl if pd.notna(row.mcl) else None,
            row.mcc if pd.notna(row.mcc) else None,
            fetched_at,
        )
        for row in df.itertuples()
    ]
    conn.executemany(
        """
        INSERT OR REPLACE INTO lmp
            (interval_start, node, lmp, mcl, mcc, fetched_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()
    return len(rows)


def _missing_days(
    conn: sqlite3.Connection,
    start: date,
    end: date,
    node: str,
) -> list[date]:
    """
    Return NYISO market days in [start, end] that are incomplete or absent
    in the DB for *node* (23 or 25 hours expected on DST change days).
    """
    lo, hi = _market_day_bounds_utc(start, end)
    rows = conn.execute(
        """
        SELECT interval_start FROM lmp
        WHERE node = ? AND interval_start >= ? AND interval_start < ?
        """,
        (node.upper(), lo.isoformat(), hi.isoformat()),
    ).fetchall()
    have = pd.Series(
        pd.to_datetime([r["interval_start"] for r in rows], utc=True), dtype="datetime64[ns, UTC]"
    )
    counts = have.dt.tz_convert(MARKET_TZ).dt.date.value_counts()
    return [d for d in _date_range(start, end) if counts.get(d, 0) < _hours_in_market_day(d)]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_many(
    start: date | str,
    end: date | str,
    nodes: Iterable[str] = ("CAPITL",),
    db_path: Path | str = DEFAULT_DB,
    force_refresh: bool = False,
    timeout: int = 60,
) -> None:
    """
    Fetch NYISO day-ahead LBMPs for several zones in one pass.

    Past months are downloaded once as a monthly ZIP (it contains every zone),
    so fetching a year for three zones takes about 12 requests, not 1,095.
    Days in a month whose archive is not published yet are fetched from the
    daily CSV files. Only missing or incomplete days are downloaded unless
    *force_refresh* is True.
    """
    start, end = _as_date(start), _as_date(end)
    if start > end:
        raise ValueError(f"start ({start}) must be <= end ({end})")
    node_set = {n.upper() for n in nodes}

    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = _connect(db_path)
    _init_db(conn)

    if force_refresh:
        todo = set(_date_range(start, end))
    else:
        todo = set()
        for node in node_set:
            todo.update(_missing_days(conn, start, end, node))
    if not todo:
        log.info("All days cached for %s — no HTTP requests needed.", sorted(node_set))
        conn.close()
        return

    today_local = pd.Timestamp.now(tz=MARKET_TZ).date()
    months = sorted({date(d.year, d.month, 1) for d in todo})
    log.info("Fetching %d day(s) across %d month(s) for %s", len(todo), len(months), sorted(node_set))

    with requests.Session() as session:
        session.headers.update({"User-Agent": "BESS-Optimizer/1.0 (research)"})
        for month in months:
            month_days = sorted(d for d in todo if (d.year, d.month) == (month.year, month.month))
            next_month = date(month.year + month.month // 12, month.month % 12 + 1, 1)
            month_complete = next_month <= today_local

            if month_complete:
                try:
                    df = _parse_zip(_get(session, _monthly_zip_url(month), timeout), node_set)
                    log.info("  %s → %d rows stored", f"{month:%Y-%m}", _upsert(conn, df))
                    continue
                except Exception as exc:  # noqa: BLE001
                    log.warning("  %s monthly ZIP failed (%s) — trying daily files", f"{month:%Y-%m}", exc)

            for day in month_days:
                try:
                    raw = pd.read_csv(io.BytesIO(_get(session, _daily_csv_url(day), timeout)))
                    log.info("  %s → %d rows stored", day, _upsert(conn, _parse_csv(raw, node_set)))
                except requests.HTTPError as exc:
                    log.warning("  %s → HTTP %s (skipped)", day, exc.response.status_code)
                except Exception as exc:  # noqa: BLE001
                    log.warning("  %s → error: %s (skipped)", day, exc)

    for node in sorted(node_set):
        gaps = _missing_days(conn, start, end, node)
        if gaps:
            log.warning("%s: %d day(s) still incomplete, e.g. %s", node, len(gaps), gaps[:3])
    conn.close()


def fetch_lmp(
    start: date | str,
    end: date | str,
    node: str = "CAPITL",
    db_path: Path | str = DEFAULT_DB,
    force_refresh: bool = False,
    timeout: int = 60,
) -> pd.DataFrame:
    """
    Fetch NYISO day-ahead LBMPs for one zone and return them from the cache.

    Parameters
    ----------
    start, end : date or "YYYY-MM-DD" string
        Inclusive range of NYISO market days.
    node : str
        NYISO zone name (e.g. "CAPITL", "N.Y.C.", "LONGIL").
    db_path : Path or str
        SQLite database file path.  Created if it does not exist.
    force_refresh : bool
        If True, re-download all days regardless of cache state.
    timeout : int
        HTTP request timeout in seconds.

    Returns
    -------
    pd.DataFrame
        Columns: interval_start (UTC Timestamp), node, lmp, mcl, mcc
        Sorted by interval_start ascending.
    """
    fetch_many(start, end, [node], db_path, force_refresh=force_refresh, timeout=timeout)
    return load_lmp(start, end, node, db_path)


def load_lmp(
    start: date | str,
    end: date | str,
    node: str = "CAPITL",
    db_path: Path | str = DEFAULT_DB,
) -> pd.DataFrame:
    """
    Load cached LBMP data from SQLite without making any HTTP requests.

    *start* and *end* are inclusive NYISO market days (America/New_York),
    so a day always runs from local midnight to local midnight.

    Returns
    -------
    pd.DataFrame
        Columns: interval_start (UTC Timestamp), node, lmp, mcl, mcc
    """
    start, end = _as_date(start), _as_date(end)
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(
            f"Database not found: {db_path}. Run fetch_lmp() first."
        )

    lo, hi = _market_day_bounds_utc(start, end)
    conn = _connect(db_path)
    df = pd.read_sql_query(
        """
        SELECT interval_start, node, lmp, mcl, mcc
        FROM   lmp
        WHERE  node            = ?
          AND  interval_start >= ?
          AND  interval_start <  ?
        ORDER  BY interval_start ASC
        """,
        conn,
        params=(node.upper(), lo.isoformat(), hi.isoformat()),
    )
    conn.close()

    if df.empty:
        return df

    df["interval_start"] = pd.to_datetime(df["interval_start"], utc=True)
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Mock data generator (for tests / CI without NYISO access)
# ---------------------------------------------------------------------------

def generate_mock_lmp(
    start: date | str,
    end: date | str,
    node: str = "CAPITL",
    db_path: Path | str = DEFAULT_DB,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic hourly LMP data and persist it to SQLite.

    Produces a realistic price curve with:
      • Morning (7–9 h) and evening (17–20 h) price peaks
      • Random noise + occasional price spikes
      • Weekend discounts

    Useful for unit tests and offline development.

    Returns the same DataFrame shape as fetch_lmp().
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    if isinstance(start, str):
        start = date.fromisoformat(start)
    if isinstance(end, str):
        end = date.fromisoformat(end)

    # One row per hour of each NYISO market day (local midnight to midnight),
    # so DST days get 23 or 25 hours, like the real data.
    lo, hi = _market_day_bounds_utc(start, end)
    idx = pd.date_range(start=lo, end=hi, freq="h", inclusive="left")
    periods = len(idx)

    local = idx.tz_convert(MARKET_TZ)
    hours = local.hour
    weekday = local.dayofweek  # 0=Mon … 6=Sun

    # Base shape: two humps
    base = (
        30
        + 25 * np.exp(-0.5 * ((hours - 8) / 2) ** 2)   # morning peak
        + 20 * np.exp(-0.5 * ((hours - 18) / 2) ** 2)  # evening peak
    )

    # Weekend discount
    base = base * np.where(weekday >= 5, 0.75, 1.0)

    # Noise + occasional spike
    noise = rng.normal(0, 4, size=periods)
    spikes = rng.choice([0.0, 1.0], size=periods, p=[0.97, 0.03]) * rng.uniform(40, 120, size=periods)
    lmp = np.maximum(base + noise + spikes, -20.0)  # allow brief negatives

    mcl = lmp * rng.uniform(0.01, 0.03, size=periods)
    mcc = lmp * rng.uniform(-0.02, 0.02, size=periods)

    df = pd.DataFrame(
        {
            "interval_start": idx,
            "node": node.upper(),
            "lmp": lmp,
            "mcl": mcl,
            "mcc": mcc,
        }
    )

    db_path = Path(db_path)
    conn = _connect(db_path)
    _init_db(conn)
    _upsert(conn, df)
    conn.close()

    log.info(
        "Mock LMP generated: %d hours, node=%s, db=%s", periods, node.upper(), db_path
    )
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# CLI


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
        description="Fetch NYISO day-ahead zonal LBMP data into SQLite",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--start", required=True, metavar="YYYY-MM-DD", help="First market day")
    parser.add_argument("--end", required=True, metavar="YYYY-MM-DD", help="Last market day (inclusive)")
    parser.add_argument("--node", default="CAPITL", choices=NYISO_ZONES, help="NYISO zone name")
    parser.add_argument("--nodes", nargs="+", choices=NYISO_ZONES, metavar="ZONE",
                        help="Several zones at once (overrides --node)")
    parser.add_argument("--db", default=str(DEFAULT_DB), metavar="PATH", help="SQLite database path")
    parser.add_argument("--force", action="store_true", help="Re-download all days, ignoring cache")
    parser.add_argument("--mock", action="store_true",
                        help="Generate synthetic data instead of fetching from NYISO (for testing)")
    args = parser.parse_args()
    nodes = args.nodes or [args.node]

    for node in nodes:
        if args.mock:
            generate_mock_lmp(args.start, args.end, node, args.db)
    if not args.mock:
        fetch_many(args.start, args.end, nodes, args.db, force_refresh=args.force)

    for node in nodes:
        df = load_lmp(args.start, args.end, node, args.db)
        if df.empty:
            print(f"{node}: no data returned — check node name and date range.")
            continue
        print(f"\n{'='*55}")
        print(f"  Node : {node}")
        print(f"  Range: {args.start} → {args.end}")
        print(f"  Rows : {len(df):,}")
        print(f"{'='*55}")
        print(df["lmp"].describe().to_string())


if __name__ == "__main__":
    _cli()
