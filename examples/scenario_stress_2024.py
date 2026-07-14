"""ストレステスト: 2024年夏の円キャリー巻き戻し局面。

2024年7月〜9月、USD/JPY は約162円から140円割れまで急落
(日銀利上げ + 為替介入 + 米景気懸念による世界的なキャリー解消)。
この戦略は直前まで「円安トレンド + プラス金利差」で3ペアロングに
張り付いているため、まさに最悪のシナリオとなる。

特に急落の最終局面(2024/8/5)は「月曜日」に発生しており、
週末ノーポジションルール(金曜17:00〜月曜7:00 JST)の効果を
ON / OFF で比較検証する。レバレッジ上限はどちらも5倍。

アンカーは 2024 年の実勢レート水準の近似値。
実行: python examples/scenario_stress_2024.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fx_trader.backtest import Backtester
from fx_trader.config import SystemConfig
from fx_trader.data import build_anchored_ohlc

# 2024年の実勢レート水準(近似)。2〜6月はウォームアップ + ロング構築期間、
# 7月頭のピークから8月頭にかけて急落、9月中旬に二番底。
ANCHORS: dict[str, list[tuple[str, float]]] = {
    "USDJPY": [
        ("2024-02-01", 147.0), ("2024-03-01", 150.0), ("2024-04-26", 158.0),
        ("2024-05-03", 153.0), ("2024-06-03", 156.0), ("2024-06-28", 160.9),
        ("2024-07-03", 161.9), ("2024-07-12", 157.9), ("2024-07-25", 152.0),
        ("2024-08-05", 144.2), ("2024-08-15", 149.0), ("2024-09-16", 140.6),
        ("2024-09-27", 142.2), ("2024-10-21", 149.5), ("2024-11-15", 156.5),
        ("2024-12-30", 157.9),
    ],
    "EURJPY": [
        ("2024-02-01", 159.5), ("2024-03-01", 162.5), ("2024-04-26", 169.5),
        ("2024-05-03", 164.5), ("2024-06-03", 169.5), ("2024-06-28", 172.5),
        ("2024-07-10", 175.4), ("2024-07-12", 171.5), ("2024-07-25", 165.0),
        ("2024-08-05", 157.5), ("2024-08-15", 163.5), ("2024-09-16", 156.5),
        ("2024-09-27", 158.9), ("2024-10-21", 162.0), ("2024-11-15", 165.5),
        ("2024-12-30", 164.5),
    ],
    "GBPJPY": [
        ("2024-02-01", 186.5), ("2024-03-01", 189.5), ("2024-04-26", 197.5),
        ("2024-05-03", 192.0), ("2024-06-03", 199.0), ("2024-06-28", 203.5),
        ("2024-07-10", 208.6), ("2024-07-12", 204.5), ("2024-07-25", 196.0),
        ("2024-08-05", 184.5), ("2024-08-15", 191.5), ("2024-09-16", 185.0),
        ("2024-09-27", 190.5), ("2024-10-21", 194.5), ("2024-11-15", 198.0),
        ("2024-12-30", 197.5),
    ],
}

# 荒れ相場のためボラティリティは通常より高め
ANNUAL_VOLS = {"USDJPY": 0.12, "EURJPY": 0.12, "GBPJPY": 0.14}

# 2024年後半の平均的な政策金利水準
POLICY_RATES = {"USD": 5.00, "EUR": 3.50, "GBP": 5.00, "JPY": 0.15}

EVAL_START = "2024-07-01"


def run_profile(weekend_flat: bool, n_seeds: int = 30) -> None:
    initial_equity = 1_000_000.0
    returns, max_dds, ruined, loscuts = [], [], 0, 0

    for seed in range(n_seeds):
        ohlc = build_anchored_ohlc(ANCHORS, ANNUAL_VOLS, seed)
        config = SystemConfig.aggressive()
        config.policy_rates = dict(POLICY_RATES)
        config.weekend_flat = weekend_flat
        result = Backtester(config, initial_equity=initial_equity).run(ohlc)
        eq = result.equity_curve[result.equity_curve.index >= EVAL_START]
        if any(t.reason == "forced_loscut" for t in result.trades):
            loscuts += 1
        if (eq <= 0).any():
            ruined += 1
            returns.append(-1.0)
            max_dds.append(-1.0)
            continue
        returns.append(eq.iloc[-1] / eq.iloc[0] - 1.0)
        peak = eq.cummax()
        max_dds.append(float(((eq - peak) / peak).min()))

    returns_a = np.array(returns)
    label = "あり(金曜引け全決済)" if weekend_flat else "なし(持ち越し)"
    print(f"--- 週末フラット {label} / レバレッジ上限5倍 ---")
    print(f"  2024/7〜12 リターン中央値: {np.median(returns_a):+.2%} "
          f"(範囲 {returns_a.min():+.2%} 〜 {returns_a.max():+.2%})")
    print(f"  最大DD中央値            : {np.median(max_dds):.2%} "
          f"(最悪 {min(max_dds):.2%})")
    print(f"  強制ロスカット発生シード : {loscuts}/{n_seeds}")
    print(f"  破綻(残高ゼロ以下)率  : {ruined}/{n_seeds}")
    print()


def main() -> None:
    print("=== ストレステスト: 2024年 円キャリー巻き戻し(30シード) ===")
    print("(2024/7/1 時点でロング満載の状態から急落局面に突入)\n")
    run_profile(weekend_flat=True)
    run_profile(weekend_flat=False)


if __name__ == "__main__":
    main()
