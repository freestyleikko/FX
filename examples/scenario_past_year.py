"""過去1年(2025年7月〜2026年7月)の実勢レートに基づくシナリオ検証。

この環境では日足の実データを直接取得できないため、公開情報から確認した
月次アンカー(実際のレート水準)の間をブラウニアンブリッジで補間した
近似日足データを複数シード生成し、戦略リターンの分布を推定する。

⚠️ 結果は「実データでの厳密なバックテスト」ではなく近似値。
   日々の細かい値動き(ストップ到達等)はシードに依存するため、
   複数シードの中央値とレンジで評価する。

アンカーの主な出所:
- USD/JPY: 2025/7 平均 146.8 → 2026/7/13 162.4(52週レンジ 145.5-162.9)
- GBP/JPY: 2025/7/13 198.75 → 2026/7/13 217.10(安値 195.3 = 2025/8/4)
- EUR/JPY: 2026年レンジ 181.3-187.6、2026/7 初旬 186 前後
- 政策金利: BOJ 0.5→0.75(2025/12)→1.0%(2026/6)、
  Fed 4.25-4.50→3.50-3.75%、ECB 2.0→2.25%、BOE 4.25→3.75%
  (バックテストでは期間平均の金利差を使用)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fx_trader.backtest import Backtester
from fx_trader.config import SystemConfig

# ----------------------------------------------------------------------
# 月次アンカー(実勢レートの近似水準)。
# 2025/2〜7 はウォームアップ用(この期間の売買は評価対象外)。
# ----------------------------------------------------------------------
ANCHORS: dict[str, list[tuple[str, float]]] = {
    "USDJPY": [
        ("2025-02-03", 154.0), ("2025-03-03", 150.0), ("2025-04-01", 149.0),
        ("2025-04-22", 140.5), ("2025-06-02", 143.5), ("2025-07-14", 147.3),
        ("2025-08-04", 146.5), ("2025-09-15", 147.3), ("2025-10-15", 151.5),
        ("2025-11-17", 155.0), ("2025-12-15", 155.5), ("2026-01-15", 155.0),
        ("2026-02-16", 156.5), ("2026-03-16", 157.5), ("2026-04-15", 158.5),
        ("2026-05-15", 158.5), ("2026-06-15", 159.5), ("2026-07-13", 162.4),
    ],
    "EURJPY": [
        ("2025-02-03", 159.0), ("2025-03-03", 158.5), ("2025-04-01", 161.0),
        ("2025-04-22", 160.0), ("2025-06-02", 163.5), ("2025-07-14", 172.2),
        ("2025-08-04", 169.5), ("2025-09-15", 173.0), ("2025-10-15", 176.0),
        ("2025-11-17", 179.5), ("2025-12-15", 181.0), ("2026-01-15", 180.5),
        ("2026-02-16", 182.0), ("2026-03-16", 183.0), ("2026-04-15", 184.0),
        ("2026-05-15", 184.5), ("2026-06-15", 184.0), ("2026-07-13", 186.0),
    ],
    "GBPJPY": [
        ("2025-02-03", 191.0), ("2025-03-03", 190.0), ("2025-04-01", 192.5),
        ("2025-04-22", 187.0), ("2025-06-02", 194.0), ("2025-07-14", 198.8),
        ("2025-08-04", 195.3), ("2025-09-15", 199.5), ("2025-10-15", 202.5),
        ("2025-11-17", 205.0), ("2025-12-15", 204.5), ("2026-01-15", 206.5),
        ("2026-02-16", 208.5), ("2026-03-16", 210.0), ("2026-04-15", 212.0),
        ("2026-05-15", 213.0), ("2026-06-15", 214.5), ("2026-07-13", 217.1),
    ],
}

# アンカー間の補間に載せる日次ボラティリティ(年率)
ANNUAL_VOLS = {"USDJPY": 0.08, "EURJPY": 0.085, "GBPJPY": 0.10}

# 評価期間(この日以降の損益を「過去1年のリターン」として集計)
EVAL_START = "2025-07-14"


def build_bridged_ohlc(seed: int) -> dict[str, pd.DataFrame]:
    """月次アンカーをブラウニアンブリッジで補間した日足OHLCを生成する。"""
    rng = np.random.default_rng(seed)
    out: dict[str, pd.DataFrame] = {}
    for pair, anchors in ANCHORS.items():
        anchor_dates = pd.DatetimeIndex([a[0] for a in anchors])
        anchor_logs = np.log([a[1] for a in anchors])
        index = pd.bdate_range(anchor_dates[0], anchor_dates[-1])
        n = len(index)

        # アンカーの対数価格を時間比で線形補間(基準パス)
        t = index.view("int64").astype(float)
        ta = anchor_dates.view("int64").astype(float)
        base = np.interp(t, ta, anchor_logs)

        # ブラウニアンブリッジノイズ(アンカー日でゼロに固定)
        vol_d = ANNUAL_VOLS[pair] / np.sqrt(252)
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


def make_config(profile: str = "standard") -> SystemConfig:
    config = SystemConfig.aggressive() if profile == "aggressive" else SystemConfig()
    # 評価期間(2025/7〜2026/7)の平均的な政策金利水準
    config.policy_rates = {"USD": 3.85, "EUR": 2.05, "GBP": 4.05, "JPY": 0.75}
    return config


def main() -> None:
    profile = "aggressive" if "--aggressive" in sys.argv else "standard"
    n_seeds = 30
    initial_equity = 1_000_000.0
    returns, max_dds, swaps, levs = [], [], [], []

    for seed in range(n_seeds):
        ohlc = build_bridged_ohlc(seed)
        bt = Backtester(make_config(profile), initial_equity=initial_equity)
        result = bt.run(ohlc)
        eq = result.equity_curve[result.equity_curve.index >= EVAL_START]
        ret = eq.iloc[-1] / eq.iloc[0] - 1.0
        peak = eq.cummax()
        max_dd = float(((eq - peak) / peak).min())
        returns.append(ret)
        max_dds.append(max_dd)
        swaps.append(result.swap_pnl)
        levs.append(float(result.daily_leverage.max()))

    returns_a = np.array(returns)
    print(f"=== 過去1年シナリオ検証(2025/7/14 → 2026/7/13, "
          f"30シード, profile={profile}) ===")
    print(f"リターン中央値   : {np.median(returns_a):+.2%}")
    print(f"リターン平均     : {returns_a.mean():+.2%}")
    print(f"リターン範囲     : {returns_a.min():+.2%} 〜 {returns_a.max():+.2%}")
    print(f"25-75パーセンタイル: {np.percentile(returns_a, 25):+.2%} 〜 "
          f"{np.percentile(returns_a, 75):+.2%}")
    print(f"勝ちシード率     : {(returns_a > 0).mean():.0%}")
    print(f"最大DD中央値     : {np.median(max_dds):.2%}")
    print(f"スワップ損益中央値: {np.median(swaps):+,.0f} 円 "
          f"(初期資金 {initial_equity:,.0f} 円)")
    print(f"最大レバレッジ   : {max(levs):.2f} 倍 (上限 5.00)")


if __name__ == "__main__":
    main()
