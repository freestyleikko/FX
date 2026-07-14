"""FX自動売買システム(ロジック実装)。

USD/JPY・EUR/JPY・GBP/JPY の3通貨ペアを対象に、トレンドと金利差(キャリー)を
組み合わせたロング・ショート戦略を提供する。レバレッジは自己資金の5倍まで。
"""

from fx_trader.config import SystemConfig, PAIRS
from fx_trader.strategy import CompositeStrategy
from fx_trader.portfolio import PortfolioAllocator
from fx_trader.risk import RiskManager
from fx_trader.backtest import Backtester

__all__ = [
    "SystemConfig",
    "PAIRS",
    "CompositeStrategy",
    "PortfolioAllocator",
    "RiskManager",
    "Backtester",
]
