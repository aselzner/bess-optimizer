"""Tests for data_fetch.py. No network access: NYISO responses are faked."""
import io
import re
import zipfile
from datetime import date

import pandas as pd
import pytest
import requests

import data_fetch as dfm

HEADER = ('"Time Stamp","Name","PTID","LBMP ($/MWHr)",'
          '"Marginal Cost Losses ($/MWHr)","Marginal Cost Congestion ($/MWHr)"\n')


def nyiso_csv(day: date, zones=("CAPITL", "N.Y.C."), price=lambda h, z: 30.0 + h) -> str:
    """Build a CSV in NYISO's DAM zonal LBMP layout for one market day."""
    local = pd.date_range(pd.Timestamp(day).tz_localize(dfm.MARKET_TZ),
                          pd.Timestamp(day + pd.Timedelta(days=1)).tz_localize(dfm.MARKET_TZ),
                          freq="h", inclusive="left")
    lines = [HEADER]
    for i, ts in enumerate(local):
        for z in zones:
            lines.append(f'"{ts:%m/%d/%Y %H:%M}","{z}",61757,{price(i, z):.2f},1.00,-0.50\n')
    return "".join(lines)


def zip_of(days) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for d in days:
            zf.writestr(f"{d:%Y%m%d}damlbmp_zone.csv", nyiso_csv(d))
    return buf.getvalue()


def test_urls():
    assert dfm._monthly_zip_url(date(2025, 3, 1)).endswith("/20250301damlbmp_zone_csv.zip")
    assert dfm._daily_csv_url(date(2025, 3, 7)).endswith("/20250307damlbmp_zone.csv")


def test_parse_csv_columns_and_utc():
    raw = pd.read_csv(io.StringIO(nyiso_csv(date(2025, 1, 15))))
    out = dfm._parse_csv(raw, {"CAPITL"})
    assert list(out.columns) == ["interval_start", "node", "lmp", "mcl", "mcc"]
    assert len(out) == 24
    assert str(out["interval_start"].dt.tz) == "UTC"
    assert out["interval_start"].iloc[0] == pd.Timestamp("2025-01-15 05:00", tz="UTC")  # EST = UTC-5
    assert out["lmp"].iloc[0] == 30.0 and out["mcl"].iloc[0] == 1.0 and out["mcc"].iloc[0] == -0.5


@pytest.mark.parametrize("day,hours", [(date(2024, 11, 3), 25), (date(2025, 3, 9), 23)])
def test_parse_csv_dst_days(day, hours):
    out = dfm._parse_csv(pd.read_csv(io.StringIO(nyiso_csv(day))), {"CAPITL", "N.Y.C."})
    for _, grp in out.groupby("node"):
        assert len(grp) == hours
        assert grp["interval_start"].is_unique
        assert (grp["interval_start"].diff().dropna() == pd.Timedelta("1h")).all()


def test_parse_csv_old_column_label():
    text = nyiso_csv(date(2025, 1, 15)).replace("LBMP ($/MWHr)", "DAM Zonal LBMP", 1)
    out = dfm._parse_csv(pd.read_csv(io.StringIO(text)), {"N.Y.C."})
    assert len(out) == 24 and out["node"].unique().tolist() == ["N.Y.C."]


class FakeResp:
    def __init__(self, content=b"", status=200):
        self.content, self.status_code = content, status

    def raise_for_status(self):
        if self.status_code != 200:
            err = requests.HTTPError(f"{self.status_code}")
            err.response = self
            raise err


def test_fetch_many_uses_monthly_zip_then_loads_market_days(tmp_path, monkeypatch):
    feb = [date(2025, 2, d) for d in range(1, 29)]
    calls = []

    def fake_get(self, url, timeout=None):
        calls.append(url)
        if url.endswith("20250201damlbmp_zone_csv.zip"):
            return FakeResp(zip_of(feb))
        return FakeResp(status=404)

    monkeypatch.setattr(requests.Session, "get", fake_get)
    db = tmp_path / "t.db"
    dfm.fetch_many("2025-02-03", "2025-02-05", ["CAPITL", "N.Y.C."], db)

    assert len(calls) == 1                                  # one ZIP for both zones
    out = dfm.load_lmp("2025-02-03", "2025-02-05", "N.Y.C.", db)
    assert len(out) == 72
    assert out["interval_start"].iloc[0] == pd.Timestamp("2025-02-03 05:00", tz="UTC")

    calls.clear()
    dfm.fetch_many("2025-02-03", "2025-02-05", ["CAPITL"], db)   # cached → no HTTP
    assert calls == []


def test_fetch_many_falls_back_to_daily_csv(tmp_path, monkeypatch):
    def fake_get(self, url, timeout=None):
        if url.endswith("damlbmp_zone_csv.zip"):
            return FakeResp(status=404)
        ymd = re.search(r"/(\d{8})damlbmp_zone\.csv$", url).group(1)
        day = date(int(ymd[:4]), int(ymd[4:6]), int(ymd[6:]))
        return FakeResp(nyiso_csv(day).encode())

    monkeypatch.setattr(requests.Session, "get", fake_get)
    db = tmp_path / "t.db"
    out = dfm.fetch_lmp("2024-11-02", "2024-11-03", "CAPITL", db)
    assert len(out) == 24 + 25                              # includes the fall-back day
    assert dfm._missing_days(dfm._connect(db), date(2024, 11, 2), date(2024, 11, 3), "CAPITL") == []


def test_mock_generator_matches_market_days(tmp_path):
    db = tmp_path / "m.db"
    dfm.generate_mock_lmp("2025-03-08", "2025-03-10", "CAPITL", db)
    out = dfm.load_lmp("2025-03-08", "2025-03-10", "CAPITL", db)
    assert len(out) == 24 + 23 + 24
