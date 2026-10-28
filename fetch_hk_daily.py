"""Fetch HK daily K-line data from Tencent's public quote endpoint.

Only for personal research. The endpoint is undocumented and may change
without notice; no rate guarantees are given. Cache the output to disk and
do not hammer it.

Field layout warning
--------------------
Each row arrives as::

    [date, open, CLOSE, high, low, volume, {...}, ...]

That is **open, close, high, low** -- NOT OHLC. Reading position 2 as "high"
is the single most common way to silently corrupt a dataset built from this
source. ``parse_rows`` validates OHLC ordering so that mistake fails loudly.

Paging
------
A single request tops out around 1200 rows (about five years). Longer history
is pulled backwards by pinning the ``end`` date to the day before the oldest
row already fetched.

Intraday caveat
---------------
During the session the newest row is still forming, and Hong Kong's closing
auction (CAS, 16:00-16:10) is not included until roughly 16:10 HKT. Pass
``skip_today=True`` if you need closed bars only.
"""

from __future__ import annotations

import datetime as dt
import json
import sys
import urllib.request
from pathlib import Path

import pandas as pd

ENDPOINT = "https://web.ifzq.gtimg.cn/appstock/app/hkfqkline/get"
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 " \
     "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
HKT = dt.timezone(dt.timedelta(hours=8))
PAGE = 1200


def _request(code: str, adjust: str, count: int, end: str | None,
             timeout: int) -> list:
    param = f"{code},day,,{end or ''},{count},{adjust}"
    url = f"{ENDPOINT}?param={param}"
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode("utf-8"))

    data = payload.get("data")
    if not isinstance(data, dict) or code not in data:
        return []
    block = data[code]
    return block.get(f"{adjust}day") or block.get("day") or []


def fetch(code: str = "hk00700", adjust: str = "qfq", pages: int = 6,
          timeout: int = 20) -> pd.DataFrame:
    """Return a Date-indexed OHLCV frame, oldest first."""
    collected: list[list] = []
    end: str | None = None

    for _ in range(pages):
        rows = _request(code, adjust, PAGE, end, timeout)
        if not rows:
            break
        collected.append(rows)
        oldest = rows[0][0]
        if len(rows) < PAGE:
            break
        end = (pd.Timestamp(oldest) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    if not collected:
        raise RuntimeError(f"no K-line rows returned for {code}")

    rows = [r for page in reversed(collected) for r in page]
    return parse_rows(rows)


def parse_rows(rows: list) -> pd.DataFrame:
    recs = []
    for r in rows:
        date, o, c, h, l, v = r[0], r[1], r[2], r[3], r[4], r[5]
        recs.append((date, float(o), float(h), float(l), float(c), float(v)))

    df = pd.DataFrame(recs, columns=["Date", "Open", "High", "Low", "Close", "Volume"])
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.drop_duplicates("Date").set_index("Date").sort_index()

    bad = df[(df["High"] < df["Low"])
             | (df["High"] < df["Open"]) | (df["High"] < df["Close"])
             | (df["Low"] > df["Open"]) | (df["Low"] > df["Close"])]
    if len(bad):
        raise ValueError(
            f"{len(bad)} rows violate OHLC ordering -- the endpoint's field "
            f"layout probably changed. Inspect the raw rows before parsing."
        )
    return df


def is_closed_bar(df: pd.DataFrame, now: dt.datetime | None = None) -> bool:
    """True when the newest row is not today's still-forming session."""
    now = now or dt.datetime.now(HKT)
    last = df.index[-1].date()
    if last != now.date():
        return True
    return now.hour >= 16 and now.minute >= 15


def load(path: str | Path) -> pd.DataFrame:
    """Read a cached CSV written by :func:`save`."""
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    keep = ["Open", "High", "Low", "Close", "Volume"]
    return df[keep].astype(float)


def save(df: pd.DataFrame, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, float_format="%.3f")


def update(path: str | Path, code: str = "hk00700", adjust: str = "qfq",
           skip_today: bool = False) -> pd.DataFrame:
    """Merge the endpoint's history into an existing cache and save it."""
    path = Path(path)
    fresh = fetch(code=code, adjust=adjust)
    if skip_today and not is_closed_bar(fresh):
        fresh = fresh.iloc[:-1]

    if path.exists():
        old = load(path)
        merged = pd.concat([old, fresh])
        merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    else:
        merged = fresh

    save(merged, path)
    return merged


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    skip = "--skip-today" in sys.argv
    code = args[0] if args else "hk00700"
    out = Path(args[1]) if len(args) > 1 else Path(f"data/{code}.csv")
    frame = update(out, code=code, skip_today=skip)
    if not is_closed_bar(frame):
        print("warning: newest bar is today's session and is not final "
              "(HK closing auction settles around 16:10 HKT)")
    print(f"{code}: {len(frame)} rows, {frame.index[0].date()} -> {frame.index[-1].date()}")
    print(frame.tail(3).to_string(float_format=lambda x: f"{x:.2f}"))
