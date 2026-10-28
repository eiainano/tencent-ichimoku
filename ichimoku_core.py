"""Ichimoku Kinko Hyo (一目均衡表) core calculations.

Terminology
-----------
* Tenkan-sen (转换线) = (highest high + lowest low) / 2 over the last 9 bars
* Kijun-sen  (基准线) = same over the last 26 bars
* Senkou A   (先行带A) = (tenkan + kijun) / 2, plotted 26 bars AHEAD
* Senkou B   (先行带B) = (highest high + lowest low) / 2 over 52 bars, plotted 26 bars AHEAD
* Chikou     (迟行线)  = close, plotted 26 bars BEHIND
* Kumo       (云)      = the band between Senkou A and Senkou B

Why "future cloud" is deterministic
-----------------------------------
The cloud drawn on bar ``t`` is built from *raw* (unshifted) values of bar
``t - 26``.  So the cloud for the next 26 bars is already fully determined
by data up to today.  No forecast is involved: it is pure arithmetic.

A very common bug is to take the *shifted* Senkou series and read its last
26 rows as "the future".  Those rows are the cloud of the last 26 **past**
bars.  ``future_cloud()`` below returns the real projection.

Chikou caveat
-------------
``Close.shift(-26)`` is NaN on the last 26 rows, so comparing it against the
current close always yields False.  The correct test for "price above the
lagging line" is ``close[-1] > close[-1 - 26]``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULTS = dict(tenkan=9, kijun=26, senkou_b=52, displacement=26)


def midpoint(high: pd.Series, low: pd.Series, period: int) -> pd.Series:
    """(rolling highest high + rolling lowest low) / 2."""
    return (high.rolling(period).max() + low.rolling(period).min()) / 2


def compute(df: pd.DataFrame, tenkan: int = 9, kijun: int = 26,
            senkou_b: int = 52, displacement: int = 26) -> pd.DataFrame:
    """Return raw (unshifted) Ichimoku lines plus the displayed cloud.

    ``df`` needs the columns Open, High, Low, Close.
    """
    high, low, close = df["High"], df["Low"], df["Close"]

    raw_tenkan = midpoint(high, low, tenkan)
    raw_kijun = midpoint(high, low, kijun)
    raw_span_a = (raw_tenkan + raw_kijun) / 2
    raw_span_b = midpoint(high, low, senkou_b)

    out = pd.DataFrame({
        "close": close,
        "tenkan": raw_tenkan,
        "kijun": raw_kijun,
        # *_raw values belong to their own bar; the *_plot values are what you
        # actually see drawn on the chart at that bar.
        "spanA_raw": raw_span_a,
        "spanB_raw": raw_span_b,
        "spanA": raw_span_a.shift(displacement),
        "spanB": raw_span_b.shift(displacement),
    })
    out["kumo_top"] = out[["spanA", "spanB"]].max(axis=1)
    out["kumo_bot"] = out[["spanA", "spanB"]].min(axis=1)
    out["pos"] = np.where(out["close"] > out["kumo_top"], "above",
                 np.where(out["close"] < out["kumo_bot"], "below", "in"))
    out["twist"] = np.where(out["spanA"] > out["spanB"], "bull", "bear")
    out["tenkan_kijun"] = np.where(out["tenkan"] > out["kijun"], "bull", "bear")

    # Lagging line, measured the only way that works on the latest bar.
    out["chikou_ref"] = out["close"].shift(displacement)
    out["chikou"] = np.where(out["close"] > out["chikou_ref"], "bull", "bear")
    out.loc[out["chikou_ref"].isna(), "chikou"] = np.nan
    return out


def future_cloud(df: pd.DataFrame, bars: int = 26, **kwargs) -> pd.DataFrame:
    """Cloud for the next ``bars`` sessions, anchored on the last bar.

    Returns a frame indexed by ``last_date + 1..bars`` business days with
    columns spanA / spanB / thick_pct.  Values come from the *raw* series,
    which is exactly what the indicator draws ahead of price.
    """
    params = {**DEFAULTS, **kwargs}
    ich = compute(df, **params)
    d = params["displacement"]

    raw_a = ich["spanA_raw"].to_numpy()
    raw_b = ich["spanB_raw"].to_numpy()
    n = len(ich)

    # Cloud on the future bar (n + j) comes from raw index (n + j - d).
    idx = [n - d + j - 1 for j in range(1, bars + 1)]
    idx = [i for i in idx if 0 <= i < n]
    span_a = raw_a[idx]
    span_b = raw_b[idx]

    dates = pd.bdate_range(ich.index[-1] + pd.Timedelta(days=1), periods=len(idx))
    out = pd.DataFrame({"spanA": span_a, "spanB": span_b}, index=dates)
    out["thick_pct"] = (np.maximum(out["spanA"], out["spanB"])
                        / np.minimum(out["spanA"], out["spanB"]) - 1) * 100
    return out


def score(row: pd.Series) -> int:
    """Four-element checklist, 0-4. Higher is more bullish."""
    return int(
        (row["close"] > row["kumo_top"])
        + (row["spanA"] > row["spanB"])
        + (row["tenkan"] > row["kijun"])
        + (row["close"] > row["chikou_ref"])
    )


def resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Aggregate OHLCV to a higher timeframe.

    Use a *labelled* rule (``W-FRI`` / ``ME`` / ``QE``) so each bar ends where
    you expect it to.  Do not use ``shift`` on the aggregate afterwards -- the
    last bar is by definition incomplete, and shifting it would drop the very
    information you are looking at.
    """
    agg = {"Open": "first", "High": "max", "Low": "min",
           "Close": "last", "Volume": "sum"}
    out = df.resample(rule).agg(agg).dropna()
    return out[out["Volume"] > 0]
