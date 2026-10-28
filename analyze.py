"""Print a full Ichimoku + momentum read of the newest bar.

Run after refreshing data:

    python fetch_hk_daily.py hk00700 data/00700.csv --skip-today
    python analyze.py data/00700.csv

Sections: bar anatomy -> four-timeframe scorecard -> MACD/KDJ -> forward
cloud -> golden-cross timing -> N/4 price thresholds -> range position.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import ichimoku_core as core
from fetch_hk_daily import load

TIMEFRAMES = {
    "D": None,          # native daily bars
    "W": "W-FRI",
    "M": "ME",
    "Q": "QE",
}


# ---------------------------------------------------------------- indicators
def macd(close: pd.Series, fast=12, slow=26, signal=9):
    dif = close.ewm(span=fast, adjust=False).mean() - close.ewm(span=slow, adjust=False).mean()
    dea = dif.ewm(span=signal, adjust=False).mean()
    return dif, dea, 2 * (dif - dea)


def kdj(df: pd.DataFrame, period=9, k_smooth=3, d_smooth=3):
    low = df["Low"].rolling(period).min()
    high = df["High"].rolling(period).max()
    span = (high - low).replace(0, np.nan)
    rsv = ((df["Close"] - low) / span * 100).fillna(50)

    k = d = 50.0
    ks, ds = [], []
    for v in rsv.to_numpy():
        k = (k_smooth - 1) / k_smooth * k + 1 / k_smooth * v
        d = (d_smooth - 1) / d_smooth * d + 1 / d_smooth * k
        ks.append(k)
        ds.append(d)
    k_arr, d_arr = np.array(ks), np.array(ds)
    return pd.Series(k_arr, index=df.index), pd.Series(d_arr, index=df.index), \
        pd.Series(3 * k_arr - 2 * d_arr, index=df.index), rsv


# ------------------------------------------------------------------- sections
def bar_anatomy(df: pd.DataFrame, vol_window=20) -> None:
    o, h, l, c = (df[k].iloc[-1] for k in ("Open", "High", "Low", "Close"))
    prev = df["Close"].iloc[-2]
    vol = df["Volume"].iloc[-1]
    vol_ma = df["Volume"].iloc[-vol_window - 1:-1].mean()

    rng = h - l
    print("\n=== Bar anatomy ===")
    print(f"date        {df.index[-1].date()}")
    print(f"O/H/L/C     {o:.2f} / {h:.2f} / {l:.2f} / {c:.2f}")
    print(f"change      {c - prev:+.2f} ({(c / prev - 1) * 100:+.2f}%)")
    print(f"gap         {o - prev:+.2f} ({(o / prev - 1) * 100:+.2f}%)")
    print(f"range       {rng / prev * 100:.2f}%   close in range {((c - l) / rng * 100) if rng else 0:.0f}%")
    print(f"body        {abs(c - o):.2f} ({'up' if c > o else 'down'})  "
          f"upper {h - max(o, c):.2f}  lower {min(o, c) - l:.2f}")
    print(f"volume      {vol / 1e4:,.0f}wan = {vol / vol_ma:.2f}x {vol_window}d average")


def timeframe_table(df: pd.DataFrame) -> None:
    print("\n=== Four timeframes (score 0-4) ===")
    print(f"{'tf':<3}{'close':>9}{'tenkan':>9}{'kijun':>9}{'spanA':>9}{'spanB':>9}"
          f"{'cloud':>7}{'pos':>8}{'score':>7}")
    for label, rule in TIMEFRAMES.items():
        sub = df if rule is None else core.resample(df, rule)
        if len(sub) < 60:
            continue
        ich = core.compute(sub).dropna(subset=["spanA", "spanB"])
        row = ich.iloc[-1]
        print(f"{label:<3}{row['close']:>9.2f}{row['tenkan']:>9.2f}{row['kijun']:>9.2f}"
              f"{row['spanA']:>9.2f}{row['spanB']:>9.2f}{row['twist']:>7}{row['pos']:>8}"
              f"{core.score(row):>7}")
    print("  score = close>cloud + spanA>spanB + tenkan>kijun + close>close[-26]")


def momentum(df: pd.DataFrame) -> None:
    dif, dea, hist = macd(df["Close"])
    k, d, j, rsv = kdj(df)
    print("\n=== Momentum ===")
    print(f"MACD   DIF {dif.iloc[-1]:+.2f}  DEA {dea.iloc[-1]:+.2f}  "
          f"hist {hist.iloc[-1]:+.2f}  (hist pct {100 * (hist < hist.iloc[-1]).mean():.1f}%)")
    print(f"KDJ    K {k.iloc[-1]:.1f}  D {d.iloc[-1]:.1f}  J {j.iloc[-1]:.1f}  "
          f"RSV {rsv.iloc[-1]:.1f}")

    bar_first_low = hist.iloc[-3]
    print(f"       hist path {-4:+}: {hist.iloc[-4]:+.2f} -> {hist.iloc[-3]:+.2f} "
          f"-> {hist.iloc[-2]:+.2f} -> {hist.iloc[-1]:+.2f}")
    print(f"       bullish divergence needs: price < recent swing low AND hist > {bar_first_low:+.2f}")


def forward_cloud(df: pd.DataFrame, bars=10) -> None:
    fc = core.future_cloud(df, bars=bars)
    print(f"\n=== Forward cloud (next {bars} sessions) ===")
    for ts, row in fc.iterrows():
        top, bot = max(row["spanA"], row["spanB"]), min(row["spanA"], row["spanB"])
        print(f" {ts.date()}  bot {bot:7.2f}  top {top:7.2f}  thick {row['thick_pct']:.2f}%")


def _window_edges(high: np.ndarray, low: np.ndarray, period: int, ahead: int):
    """High/low extremes of the (period - ahead) historical bars still inside
    the window once ``ahead`` synthetic bars are appended."""
    keep = period - ahead
    if keep <= 0:
        return -np.inf, np.inf
    return high[-keep:].max(), low[-keep:].min()


def _line_at(price: float, top: float, bot: float) -> float:
    return (max(top, price) + min(bot, price)) / 2


def golden_cross_timing(df: pd.DataFrame, prices, max_bars=60) -> None:
    print("\n=== Golden cross timing (price held flat) ===")
    high, low = df["High"].to_numpy(), df["Low"].to_numpy()
    for p in prices:
        hit = None
        for h in range(1, max_bars + 1):
            t9, b9 = _window_edges(high, low, 9, h)
            t26, b26 = _window_edges(high, low, 26, h)
            if _line_at(p, t9, b9) > _line_at(p, t26, b26):
                hit = h
                break
        print(f" hold {p:7.2f} -> " + (f"{hit} sessions" if hit else f"no cross within {max_bars}"))


def thresholds(df: pd.DataFrame, horizons=(1, 2, 3, 5, 8, 10, 15, 20),
               lo=300.0, hi=800.0, step=0.05) -> None:
    """Smallest held-flat price that lifts the score to N/4 by T+h.

    Grid search on purpose: the mapping price -> tenkan/kijun is not monotone
    once the test price exceeds the window high, so a bisection can miss a
    solution that exists.
    """
    print("\n=== Score thresholds (price held flat, grid search) ===")
    high, low = df["High"].to_numpy(), df["Low"].to_numpy()
    close = df["Close"].to_numpy()
    ich = core.compute(df)
    raw_a, raw_b = ich["spanA_raw"].to_numpy(), ich["spanB_raw"].to_numpy()
    n, d = len(df), 26
    grid = np.arange(lo, hi + step, step)

    print(f"{'T+':<4}{'date':<12}{'1/4':>9}{'2/4':>9}{'3/4':>9}{'4/4':>9}"
          f"{'cloud top':>11}{'lagging':>9}")
    for h in horizons:
        ri = n - 1 + h - d
        if ri < 0 or ri >= n:
            continue
        cloud_top = max(raw_a[ri], raw_b[ri])
        cloud_bull = raw_a[ri] > raw_b[ri]
        lag_ref = close[ri]

        t9, b9 = _window_edges(high, low, 9, h)
        t26, b26 = _window_edges(high, low, 26, h)
        ten = (np.maximum(t9, grid) + np.minimum(b9, grid)) / 2
        kij = (np.maximum(t26, grid) + np.minimum(b26, grid)) / 2
        sc = ((grid > cloud_top).astype(int) + int(cloud_bull)
              + (ten > kij).astype(int) + (grid > lag_ref).astype(int))

        need = []
        for target in (1, 2, 3, 4):
            hit = grid[sc >= target]
            need.append(f"{hit[0]:9.2f}" if len(hit) else f"{'-':>9}")
        date = df.index[-1] + pd.tseries.offsets.BDay(h)
        print(f"T+{h:<3}{date.date()!s:<12}" + "".join(need)
              + f"{cloud_top:11.2f}{lag_ref:9.2f}")
    print("  dates use business days only; HK holidays are not subtracted")


def range_position(df: pd.DataFrame, lookback=250) -> None:
    high = df["High"].rolling(lookback).max().iloc[-1]
    low = df["Low"].rolling(lookback).min().iloc[-1]
    c = df["Close"].iloc[-1]
    print(f"\n=== Position in {lookback}-day range ===")
    print(f" range {low:.2f} .. {high:.2f}  at {100 * (c - low) / (high - low):.2f}%  "
          f"({(c / high - 1) * 100:.1f}% from high)")

    ich = core.compute(df).dropna(subset=["spanA", "spanB"])
    row = ich.iloc[-1]
    for name in ("tenkan", "kijun", "kumo_top", "kumo_bot"):
        print(f" {name:<9}{row[name]:8.2f}  ({(row[name] / c - 1) * 100:+.2f}%)")

    pos = ich["pos"]
    streak, cur = 0, pos.iloc[-1]
    for v in pos[::-1]:
        if v != cur:
            break
        streak += 1
    print(f" price has been '{cur}' the cloud for {streak} bars")


def main() -> None:
    ap = argparse.ArgumentParser(description="Ichimoku + momentum read of the newest bar")
    ap.add_argument("data", nargs="?", default="data/00700.csv")
    ap.add_argument("--bars", type=int, default=10, help="forward cloud horizon")
    ap.add_argument("--prices", type=float, nargs="*", default=None,
                    help="prices for golden-cross timing (default: last close +/- offsets)")
    args = ap.parse_args()

    path = Path(args.data)
    df = load(path)
    c = df["Close"].iloc[-1]
    prices = args.prices or [round(c), 400, 405, 410, 415, 420]

    print(f"dataset {path}  {len(df)} rows  {df.index[0].date()} -> {df.index[-1].date()}")
    bar_anatomy(df)
    timeframe_table(df)
    momentum(df)
    forward_cloud(df, args.bars)
    golden_cross_timing(df, prices)
    thresholds(df)
    range_position(df)


if __name__ == "__main__":
    main()
