"""
data_fetch.py  —  BESS Dispatch Optimizer
==========================================
Fetches NYISO Day-Ahead Market (DAM) Locational Marginal Prices (LMPs)
from the public NYISO bulk-data portal and persists them to a local SQLite
database for use by the optimizer.

NYISO public data URL pattern
------------------------------
    https://mis.nyiso.com/public/csv/damlbmp/{YYYYMMDD}damlbmp_zone_csv.zip

Each ZIP contains one CSV per hour-ending interval:
    Time Stamp, Name, PTID, Marginal Cost Losses, Marginal Cost Congestion,
    DAM Zonal LBMP

This module exposes two public entry-points:
    fetch_lmp(start, end, node, db_path)   → pd.DataFrame
    load_lmp(start, end, node, db_path)    → pd.DataFrame  (DB-only, no HTTP)

Usage
-----
    python data_fetch.py --start 2025-01-01 --end 2025-01-07 --node CAPITL
"""

from __future__ import annotations

import argparse
import io
import logging
import sqlite3
import zipfile
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

#: Base URL for NYISO day-ahead LMP bulk downloads
NYISO_DAM_BASE = "https://mis.nyiso.com/public/csv/damlbmp"

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
# HTTP / parse helpers
# ---------------------------------------------------------------------------

def _date_range(start: date, end: date):
    """Yield each date from start to end inclusive."""
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)


def _build_url(day: date) -> str:
    return f"{NYISO_DAM_BASE}/{day:%Y%m%d}damlbmp_zone_csv.zip"


def _parse_zip(content: bytes, node: str) -> pd.DataFrame:
    """
    Parse a NYISO DAM LMP ZIP archive.

    The ZIP holds one or more CSV files named like:
        20250101damlbmp_zone_20250101T0000.csv

    Each CSV has columns:
        Time Stamp | Name | PTID | Marginal Cost Losses |
        Marginal Cost Congestion | DAM Zonal LBMP

    Returns a DataFrame with columns:
        interval_start (tz-aware UTC Timestamp), node, lmp, mcl, mcc
    """
    frames: list[pd.DataFrame] = []
    node_upper = node.upper()

    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
        if not csv_names:
            raise ValueError("ZIP archive contains no CSV files.")

        for csv_name in csv_names:
            with zf.open(csv_name) as fh:
                df = pd.read_csv(fh, parse_dates=["Time Stamp"])

            # Normalise column names
            df.columns = [c.strip() for c in df.columns]
            rename = {
                "Time Stamp": "interval_start",
                "Name": "node",
                "DAM Zonal LBMP": "lmp",
                "Marginal Cost Losses": "mcl",
                "Marginal Cost Congestion": "mcc",
            }
            df = df.rename(columns=rename)

            # Filter to requested node
            mask = df["node"].str.upper() == node_upper
            df = df[mask].copy()

            if df.empty:
                continue

            # NYISO timestamps are Eastern; convert to UTC
            df["interval_start"] = (
                df["interval_start"]
                .dt.tz_localize("America/New_York", ambiguous="infer", nonexistent="shift_forward")
                .dt.tz_convert("UTC")
            )

            frames.append(df[["interval_start", "node", "lmp", "mcl", "mcc"]])

    if not frames:
        return pd.DataFrame(columns=["interval_start", "node", "lmp", "mcl", "mcc"])

    result = pd.concat(frames, ignore_index=True)
    result["node"] = result["node"].str.upper()
    result = result.sort_values("interval_start").drop_duplicates(
        subset=["interval_start", "node"]
    )
    return result


# ---------------------------------------------------------------------------
# Core fetch logic
# ---------------------------------------------------------------------------

def _fetch_day(
    day: date,
    node: str,
    session: requests.Session,
    timeout: int = 30,
) -> pd.DataFrame:
    """Download and parse a single day's DAM LMP ZIP for *node*."""
    url = _build_url(day)
    log.info("GET %s", url)
    resp = session.get(url, timeout=timeout)
    resp.raise_for_status()
    return _parse_zip(resp.content, node)


def _upsert(conn: sqlite3.Connection, df: pd.DataFrame) -> int:
    """Insert or replace rows in the lmp table. Returns row count written."""
    if df.empty:
        return 0

    fetched_at = datetime.now(__import__("datetime").timezone.utc).isoformat(timespec="seconds")
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
    Return days in [start, end] that have fewer than 24 rows in the DB
    for *node* (i.e., incomplete or absent).
    """
    missing = []
    node_upper = node.upper()
    for day in _date_range(start, end):
        day_start = datetime(day.year, day.month, day.day).isoformat()
        next_day = day + timedelta(days=1)
        day_end = datetime(next_day.year, next_day.month, next_day.day).isoformat()
        row = conn.execute(
            """
            SELECT COUNT(*) AS cnt FROM lmp
            WHERE node = ?
              AND interval_start >= ?
              AND interval_start <  ?
            """,
            (node_upper, day_start, day_end),
        ).fetchone()
        if row["cnt"] < 24:
            missing.append(day)
    return missing


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_lmp(
    start: date | str,
    end: date | str,
    node: str = "CAPITL",
    db_path: Path | str = DEFAULT_DB,
    force_refresh: bool = False,
    timeout: int = 30,
) -> pd.DataFrame:
    """
    Fetch NYISO day-ahead LMPs for *node* between *start* and *end* inclusive.

    Data is cached in SQLite at *db_path*.  Only missing / incomplete days are
    downloaded from NYISO; subsequent calls are served from the local cache.

    Parameters
    ----------
    start, end : date or "YYYY-MM-DD" string
        Inclusive date range to fetch.
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
    if isinstance(start, str):
        start = date.fromisoformat(start)
    if isinstance(end, str):
        end = date.fromisoformat(end)
    if start > end:
        raise ValueError(f"start ({start}) must be <= end ({end})")

    db_path = Path(db_path)
    conn = _connect(db_path)
    _init_db(conn)

    if force_refresh:
        days_to_fetch = list(_date_range(start, end))
    else:
        days_to_fetch = _missing_days(conn, start, end, node)

    if days_to_fetch:
        log.info(
            "Fetching %d day(s) from NYISO for node=%s", len(days_to_fetch), node
        )
        with requests.Session() as session:
            session.headers.update({"User-Agent": "BESS-Optimizer/1.0 (research)"})
            for day in days_to_fetch:
                try:
                    df_day = _fetch_day(day, node, session, timeout=timeout)
                    written = _upsert(conn, df_day)
                    log.info("  %s → %d rows stored", day, written)
                except requests.HTTPError as exc:
                    log.warning("  %s → HTTP %s (skipped)", day, exc.response.status_code)
                except Exception as exc:  # noqa: BLE001
                    log.warning("  %s → error: %s (skipped)", day, exc)
    else:
        log.info("All days cached for node=%s — no HTTP requests needed.", node)

    conn.close()
    return load_lmp(start, end, node, db_path)


def load_lmp(
    start: date | str,
    end: date | str,
    node: str = "CAPITL",
    db_path: Path | str = DEFAULT_DB,
) -> pd.DataFrame:
    """
    Load cached LMP data from SQLite without making any HTTP requests.

    Parameters
    ----------
    start, end : date or "YYYY-MM-DD" string
    node : str
    db_path : Path or str

    Returns
    -------
    pd.DataFrame
        Columns: interval_start (UTC Timestamp), node, lmp, mcl, mcc
    """
    if isinstance(start, str):
        start = date.fromisoformat(start)
    if isinstance(end, str):
        end = date.fromisoformat(end)

    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(
            f"Database not found: {db_path}. Run fetch_lmp() first."
        )

    # end is inclusive → extend one day for the < comparison
    end_exclusive = end + timedelta(days=1)
    node_upper = node.upper()

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
        params=(
            node_upper,
            datetime(start.year, start.month, start.day).isoformat(),
            datetime(end_exclusive.year, end_exclusive.month, end_exclusive.day).isoformat(),
        ),
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

    periods = int((end - start).days + 1) * 24
    idx = pd.date_range(
        start=datetime(start.year, start.month, start.day, tzinfo=__import__("pytz").utc),
        periods=periods,
        freq="h",
    )

    hours = idx.hour
    weekday = idx.dayofweek  # 0=Mon … 6=Sun

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

def _cli() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Fetch NYISO day-ahead LMP data into SQLite",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--start", required=True, metavar="YYYY-MM-DD", help="First day to fetch"
    )
    parser.add_argument(
        "--end", required=True, metavar="YYYY-MM-DD", help="Last day to fetch (inclusive)"
    )
    parser.add_argument(
        "--node", default="CAPITL", choices=NYISO_ZONES, help="NYISO zone name"
    )
    parser.add_argument(
        "--db", default=str(DEFAULT_DB), metavar="PATH", help="SQLite database path"
    )
    parser.add_argument(
        "--force", action="store_true", help="Re-download all days, ignoring cache"
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Generate synthetic data instead of fetching from NYISO (for testing)",
    )
    args = parser.parse_args()

    if args.mock:
        df = generate_mock_lmp(args.start, args.end, args.node, args.db)
        print(f"\n[mock] Generated {len(df)} rows")
    else:
        df = fetch_lmp(args.start, args.end, args.node, args.db, force_refresh=args.force)

    if df.empty:
        print("No data returned — check node name and date range.")
        return

    # Summary statistics
    print(f"\n{'='*55}")
    print(f"  Node : {args.node}")
    print(f"  Range: {args.start} → {args.end}")
    print(f"  Rows : {len(df):,}")
    print(f"{'='*55}")
    print(df[["interval_start", "lmp"]].describe().to_string())
    print(f"\nFirst 5 rows:\n{df.head().to_string(index=False)}")


if __name__ == "__main__":
    _cli()
