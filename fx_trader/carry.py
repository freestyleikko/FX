"""金利差(キャリー)関連ロジック。

- carry_score: 金利差をシグナル成分(-1〜+1)に正規化する
- daily_swap: 保有ポジションに対する日次スワップ損益を計算する

対円ペアの場合、ベース通貨金利 > 円金利 ならロングでスワップ受取り、
ショートで支払いとなる。
"""

from __future__ import annotations

import math

from fx_trader.config import SystemConfig


def carry_score(rate_diff_pct: float, scale: float) -> float:
    """金利差(年率%pt)を -1〜+1 のスコアへ正規化する。

    tanh により scale 付近で緩やかに飽和する。金利差プラスはロング優位。
    """
    if scale <= 0:
        raise ValueError("scale must be positive")
    return math.tanh(rate_diff_pct / scale)


def daily_swap(
    units: float,
    price: float,
    rate_diff_pct: float,
    haircut: float = 0.0,
    days: int = 1,
) -> float:
    """日次スワップ損益(円建て)を返す。

    Args:
        units: ベース通貨数量(負ならショート)
        price: 現在価格(円)
        rate_diff_pct: ベース通貨金利 − 円金利(年率%)
        haircut: 業者取り分。受取りは (1-h) 倍、支払いは (1+h) 倍
        days: 付与日数(水曜3倍付与などを再現する場合に使用)
    """
    gross = units * price * (rate_diff_pct / 100.0) / 365.0 * days
    if gross >= 0:
        return gross * (1.0 - haircut)
    return gross * (1.0 + haircut)


def swap_for_positions(
    positions: dict[str, float],
    prices: dict[str, float],
    config: SystemConfig,
    days: int = 1,
) -> float:
    """全ポジションの日次スワップ合計(円建て)。

    Args:
        positions: pair -> units(負ならショート)
        prices: pair -> 現在価格
    """
    total = 0.0
    for pair, units in positions.items():
        if units == 0:
            continue
        total += daily_swap(
            units,
            prices[pair],
            config.rate_diff(pair),
            haircut=config.swap_haircut,
            days=days,
        )
    return total
