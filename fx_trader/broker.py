"""発注ゲートウェイの抽象インターフェース。

現時点では FX 業者の API 契約が無いため、実装はペーパートレード
(`PaperBroker`)のみ。実運用時は `BrokerGateway` を実装したクラス
(例: OANDA / GMO クリック証券などの API クライアント)に差し替える。
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from fx_trader.models import Order, Position


class BrokerGateway(ABC):
    """実ブローカー API を抽象化するインターフェース。"""

    @abstractmethod
    def fetch_prices(self) -> dict[str, float]:
        """全対象ペアの現在価格を取得する。"""

    @abstractmethod
    def fetch_positions(self) -> dict[str, Position]:
        """現在の保有ポジションを取得する。"""

    @abstractmethod
    def fetch_equity(self) -> float:
        """有効証拠金(円)を取得する。"""

    @abstractmethod
    def execute(self, order: Order) -> Position | None:
        """注文を執行し、約定後のポジションを返す(解消時は None)。"""


class PaperBroker(BrokerGateway):
    """ペーパートレード実装。価格は外部から `set_prices` で与える。"""

    def __init__(self, initial_equity: float = 1_000_000.0):
        self._equity = initial_equity
        self._positions: dict[str, Position] = {}
        self._prices: dict[str, float] = {}

    def set_prices(self, prices: dict[str, float]) -> None:
        # 評価損益を equity に反映してから価格を更新
        for pair, pos in self._positions.items():
            if pair in self._prices and pair in prices:
                self._equity += pos.units * (prices[pair] - self._prices[pair])
        self._prices.update(prices)

    def fetch_prices(self) -> dict[str, float]:
        return dict(self._prices)

    def fetch_positions(self) -> dict[str, Position]:
        return dict(self._positions)

    def fetch_equity(self) -> float:
        return self._equity

    def execute(self, order: Order) -> Position | None:
        price = self._prices[order.pair]
        if order.target_units == 0.0:
            self._positions.pop(order.pair, None)
            return None
        pos = Position(
            pair=order.pair, units=order.target_units, entry_price=price
        )
        self._positions[order.pair] = pos
        return pos
