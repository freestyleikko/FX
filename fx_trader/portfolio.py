"""資金配分とレバレッジ制限。

各ペアのシグナルスコアをボラティリティ調整して目標ウェイト(自己資金比)に変換し、
総建玉が自己資金の max_gross_leverage(既定 5 倍)を超えないよう必ず縮小する。
"""

from __future__ import annotations

from fx_trader.config import SystemConfig
from fx_trader.models import Order


class PortfolioAllocator:
    """ボラティリティ調整 + レバレッジ上限つきの資金配分。"""

    def __init__(self, config: SystemConfig):
        self.config = config

    def target_weights(
        self, scores: dict[str, float], vols: dict[str, float]
    ) -> dict[str, float]:
        """ペアごとの目標ウェイト(自己資金に対する建玉比率、符号つき)を返す。

        weight = score × (target_pair_vol / 実現ボラ)
        を基本とし、以下の制限を順に適用する:
          1. 1ペアあたり ±max_pair_leverage
          2. 合計(絶対値)が max_gross_leverage を超える場合は比例縮小
        """
        p = self.config.portfolio
        weights: dict[str, float] = {}
        for pair, score in scores.items():
            vol = vols.get(pair)
            if vol is None or vol != vol or vol <= 0:  # NaN/欠損はサイズ0
                weights[pair] = 0.0
                continue
            w = score * (p.target_pair_vol / vol)
            # 1ペア上限
            w = max(-p.max_pair_leverage, min(p.max_pair_leverage, w))
            weights[pair] = w

        gross = sum(abs(w) for w in weights.values())
        if gross > p.max_gross_leverage:
            scale = p.max_gross_leverage / gross
            weights = {pair: w * scale for pair, w in weights.items()}
        return weights

    def build_orders(
        self,
        weights: dict[str, float],
        equity: float,
        prices: dict[str, float],
        current_units: dict[str, float],
    ) -> list[Order]:
        """目標ウェイトを数量に変換し、現在保有との差分が小さければスキップする。"""
        p = self.config.portfolio
        orders: list[Order] = []
        for pair, weight in weights.items():
            price = prices[pair]
            target_units = weight * equity / price
            held = current_units.get(pair, 0.0)
            # 目標との乖離が小さいときはリバランスしない(コスト削減)
            diff_notional = abs(target_units - held) * price
            if diff_notional < p.min_trade_fraction * equity:
                continue
            orders.append(
                Order(
                    pair=pair,
                    target_units=target_units,
                    reason=f"rebalance weight={weight:+.3f}",
                )
            )
        return orders
