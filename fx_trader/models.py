"""データ型定義。"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Side(Enum):
    LONG = 1
    FLAT = 0
    SHORT = -1


@dataclass
class Signal:
    """1ペア分のシグナル。score は -1(フルショート)〜 +1(フルロング)。"""

    pair: str
    score: float
    trend_score: float
    carry_score: float
    rsi: float

    @property
    def side(self) -> Side:
        if self.score > 0:
            return Side.LONG
        if self.score < 0:
            return Side.SHORT
        return Side.FLAT


@dataclass
class Position:
    """保有ポジション。units はベース通貨数量(負ならショート)。"""

    pair: str
    units: float
    entry_price: float
    stop_price: float | None = None
    take_profit_price: float | None = None

    @property
    def side(self) -> Side:
        if self.units > 0:
            return Side.LONG
        if self.units < 0:
            return Side.SHORT
        return Side.FLAT

    def notional(self, price: float) -> float:
        """円建ての建玉金額(絶対値)。"""
        return abs(self.units) * price

    def unrealized_pnl(self, price: float) -> float:
        """円建ての評価損益。"""
        return self.units * (price - self.entry_price)


@dataclass
class Order:
    """発注指示。target_units は目標数量(現在保有との差分を執行する)。"""

    pair: str
    target_units: float
    reason: str = ""


@dataclass
class AccountState:
    """口座状態のスナップショット。"""

    equity: float
    positions: dict[str, Position] = field(default_factory=dict)

    def gross_exposure(self, prices: dict[str, float]) -> float:
        """総建玉(円建て、絶対値合計)。"""
        return sum(
            p.notional(prices[pair]) for pair, p in self.positions.items() if p.units
        )

    def gross_leverage(self, prices: dict[str, float]) -> float:
        if self.equity <= 0:
            return float("inf")
        return self.gross_exposure(prices) / self.equity
