"""テクニカル指標。すべて pandas Series(日足終値など)を入力とする。"""

from __future__ import annotations

import numpy as np
import pandas as pd


def ema(series: pd.Series, window: int) -> pd.Series:
    """指数移動平均。"""
    return series.ewm(span=window, adjust=False).mean()


def sma(series: pd.Series, window: int) -> pd.Series:
    """単純移動平均。"""
    return series.rolling(window).mean()


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Wilder 方式の RSI(0-100)。"""
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    # 下落が一度も無い区間は RSI=100 相当
    return out.fillna(100.0).where(series.notna(), np.nan)


def atr(
    high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14
) -> pd.Series:
    """Average True Range(Wilder 平滑)。"""
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / window, adjust=False).mean()


def momentum(series: pd.Series, window: int) -> pd.Series:
    """期間リターン(モメンタム)。"""
    return series.pct_change(window)


def realized_vol(series: pd.Series, window: int, annualize: int = 252) -> pd.Series:
    """日次リターンの実現ボラティリティ(年率)。"""
    ret = series.pct_change()
    return ret.rolling(window).std() * np.sqrt(annualize)
