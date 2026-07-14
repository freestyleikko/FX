"""バックテスト実行サンプル。

使い方:
    python examples/run_backtest.py                 # 合成データで実行
    python examples/run_backtest.py --csv-dir data  # 自前CSVで実行
    python examples/run_backtest.py --days 1500 --seed 7 --equity 3000000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fx_trader.backtest import Backtester
from fx_trader.config import SystemConfig
from fx_trader.data import generate_synthetic_ohlc, load_csv_ohlc


def main() -> None:
    parser = argparse.ArgumentParser(description="FX複合戦略バックテスト")
    parser.add_argument("--csv-dir", help="日足CSVのディレクトリ(<PAIR>.csv)")
    parser.add_argument("--days", type=int, default=1000, help="合成データの日数")
    parser.add_argument("--seed", type=int, default=42, help="合成データの乱数シード")
    parser.add_argument(
        "--equity", type=float, default=1_000_000, help="初期資金(円)"
    )
    args = parser.parse_args()

    config = SystemConfig()

    if args.csv_dir:
        ohlc = load_csv_ohlc(args.csv_dir)
        print(f"CSVデータを読み込みました: {args.csv_dir}")
    else:
        ohlc = generate_synthetic_ohlc(n_days=args.days, seed=args.seed)
        print(f"合成データで実行します(days={args.days}, seed={args.seed})")

    print("\n--- 金利差(対円・年率%) ---")
    for pair in config.pairs:
        print(f"  {pair}: {config.rate_diff(pair):+.2f}%")

    bt = Backtester(config, initial_equity=args.equity)
    result = bt.run(ohlc)

    print()
    print(result.report())

    print("\n--- 最終ポジションウェイト(自己資金比) ---")
    if len(result.weights_history) > 0:
        last = result.weights_history.iloc[-1]
        for pair, w in last.items():
            direction = "ロング" if w > 0 else "ショート" if w < 0 else "なし"
            print(f"  {pair}: {w:+.2f} ({direction})")


if __name__ == "__main__":
    main()
