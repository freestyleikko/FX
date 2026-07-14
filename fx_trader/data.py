"""価格データの取得。

API 契約が無い前提のため、以下の2系統を用意する:

1. `generate_synthetic_ohlc` — 検証用の合成データ生成器。
   幾何ブラウン運動にトレンドレジーム(数か月単位で方向が切り替わるドリフト)を
   重ねた対円レートを生成する。
2. `load_csv_ohlc` — 自前で用意した CSV(date,open,high,low,close)の読み込み。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fx_trader.config import PAIRS

# 合成データの基準価格(2026年前半の実勢に近い水準)
_BASE_PRICES = {"USDJPY": 150.0, "EURJPY": 163.0, "GBPJPY": 190.0}
_ANNUAL_VOLS = {"USDJPY": 0.10, "EURJPY": 0.11, "GBPJPY": 0.13}


def generate_synthetic_ohlc(
    n_days: int = 1000,
    seed: int = 42,
    pairs: tuple[str, ...] = PAIRS,
    start: str = "2022-01-03",
) -> dict[str, pd.DataFrame]:
    """合成の日足 OHLC を生成する。

    トレンドレジーム: 平均 120 営業日ごとにドリフトの向き・強さが変わる。
    3ペアの日次ショックには相関(円要因の共通成分)を持たせる。
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, periods=n_days)

    # 円の共通ファクター + ペア固有ファクター(相関 ~0.6)
    common = rng.standard_normal(n_days)

    out: dict[str, pd.DataFrame] = {}
    for k, pair in enumerate(pairs):
        vol_d = _ANNUAL_VOLS[pair] / np.sqrt(252)
        specific = rng.standard_normal(n_days)
        shocks = (0.75 * common + 0.66 * specific) * vol_d

        # レジーム切替のあるドリフト
        drift = np.zeros(n_days)
        pos = 0
        while pos < n_days:
            length = int(rng.integers(60, 180))
            annual_drift = rng.uniform(-0.12, 0.15)  # 円安バイアスをわずかに持たせる
            drift[pos : pos + length] = annual_drift / 252
            pos += length

        log_ret = drift + shocks
        close = _BASE_PRICES[pair] * np.exp(np.cumsum(log_ret))
        open_ = np.empty(n_days)
        open_[0] = _BASE_PRICES[pair]
        open_[1:] = close[:-1] * np.exp(
            rng.standard_normal(n_days - 1) * vol_d * 0.2
        )
        intraday = np.abs(rng.standard_normal(n_days)) * vol_d * close
        high = np.maximum(open_, close) + intraday * 0.6
        low = np.minimum(open_, close) - intraday * 0.6

        out[pair] = pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close}, index=dates
        )
    return out


def build_anchored_ohlc(
    anchors: dict[str, list[tuple[str, float]]],
    annual_vols: dict[str, float],
    seed: int = 0,
) -> dict[str, pd.DataFrame]:
    """実勢レートのアンカー(日付, 価格)をブラウニアンブリッジで補間した
    近似日足 OHLC を生成する(シナリオ検証用)。

    アンカー日ではノイズがゼロに固定されるため、大局のトレンドはアンカー通り、
    日々の細かい値動きのみが乱数に依存する。
    """
    rng = np.random.default_rng(seed)
    out: dict[str, pd.DataFrame] = {}
    for pair, pair_anchors in anchors.items():
        anchor_dates = pd.DatetimeIndex([a[0] for a in pair_anchors])
        anchor_logs = np.log([a[1] for a in pair_anchors])
        index = pd.bdate_range(anchor_dates[0], anchor_dates[-1])
        n = len(index)

        # アンカーの対数価格を時間比で線形補間(基準パス)
        t = index.view("int64").astype(float)
        ta = anchor_dates.view("int64").astype(float)
        base = np.interp(t, ta, anchor_logs)

        # ブラウニアンブリッジノイズ(アンカー日でゼロに固定)
        vol_d = annual_vols[pair] / np.sqrt(252)
        noise = np.cumsum(rng.standard_normal(n) * vol_d)
        pin = np.interp(t, ta, np.interp(ta, t, noise))
        bridged = noise - pin

        close = np.exp(base + bridged)
        open_ = np.empty(n)
        open_[0] = close[0]
        open_[1:] = close[:-1] * np.exp(rng.standard_normal(n - 1) * vol_d * 0.2)
        intraday = np.abs(rng.standard_normal(n)) * vol_d * close * 0.6
        high = np.maximum(open_, close) + intraday
        low = np.minimum(open_, close) - intraday
        out[pair] = pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close}, index=index
        )
    return out


def load_csv_ohlc(
    csv_dir: str | Path, pairs: tuple[str, ...] = PAIRS
) -> dict[str, pd.DataFrame]:
    """`<csv_dir>/<PAIR>.csv` から日足 OHLC を読み込む。

    CSV フォーマット: date,open,high,low,close(ヘッダー行必須)
    """
    csv_dir = Path(csv_dir)
    out: dict[str, pd.DataFrame] = {}
    for pair in pairs:
        path = csv_dir / f"{pair}.csv"
        if not path.exists():
            raise FileNotFoundError(f"{path} がありません({pair} の日足CSVが必要)")
        df = pd.read_csv(path, parse_dates=["date"], index_col="date")
        missing = {"open", "high", "low", "close"} - set(df.columns)
        if missing:
            raise ValueError(f"{path}: 列が不足しています: {sorted(missing)}")
        out[pair] = df[["open", "high", "low", "close"]].sort_index()
    return out
