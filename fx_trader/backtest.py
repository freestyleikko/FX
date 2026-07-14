"""バックテストエンジン。

日足 OHLC を入力とし、以下を日次で処理する:

1. 保有ポジションへのスワップ付与(金利差 × 建玉 / 365)
2. 当日高安によるストップロス / テイクプロフィット判定(ギャップ考慮)
3. 終値での評価とリスクチェック(ドローダウン遮断・日次損失限度)
4. 終値でのシグナル計算とリバランス(取引コスト控除、レバレッジ≤5倍)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from fx_trader.carry import daily_swap
from fx_trader.config import SystemConfig
from fx_trader.indicators import atr, realized_vol
from fx_trader.models import Position
from fx_trader.portfolio import PortfolioAllocator
from fx_trader.risk import RiskManager
from fx_trader.strategy import CompositeStrategy


@dataclass
class TradeRecord:
    date: pd.Timestamp
    pair: str
    units_before: float
    units_after: float
    price: float
    reason: str


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    trades: list[TradeRecord]
    swap_pnl: float
    cost_paid: float
    daily_leverage: pd.Series
    weights_history: pd.DataFrame

    def metrics(self) -> dict[str, float]:
        eq = self.equity_curve
        ret = eq.pct_change().dropna()
        total_return = eq.iloc[-1] / eq.iloc[0] - 1.0
        years = len(eq) / 252.0
        cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1.0 / years) - 1.0 if years > 0 else 0.0
        vol = ret.std() * np.sqrt(252) if len(ret) > 1 else 0.0
        sharpe = (ret.mean() * 252) / vol if vol > 0 else 0.0
        peak = eq.cummax()
        max_dd = float(((eq - peak) / peak).min())
        return {
            "total_return": float(total_return),
            "cagr": float(cagr),
            "annual_vol": float(vol),
            "sharpe": float(sharpe),
            "max_drawdown": max_dd,
            "num_trades": float(len(self.trades)),
            "swap_pnl": float(self.swap_pnl),
            "cost_paid": float(self.cost_paid),
            "max_leverage": float(self.daily_leverage.max()),
        }

    def report(self) -> str:
        m = self.metrics()
        lines = [
            "=== バックテスト結果 ===",
            f"総リターン        : {m['total_return']:+.2%}",
            f"年率リターン(CAGR): {m['cagr']:+.2%}",
            f"年率ボラティリティ : {m['annual_vol']:.2%}",
            f"シャープレシオ     : {m['sharpe']:.2f}",
            f"最大ドローダウン   : {m['max_drawdown']:.2%}",
            f"取引回数          : {int(m['num_trades'])}",
            f"スワップ損益累計   : {m['swap_pnl']:+,.0f} 円",
            f"取引コスト累計     : {m['cost_paid']:,.0f} 円",
            f"最大レバレッジ     : {m['max_leverage']:.2f} 倍 (上限 5.00)",
        ]
        return "\n".join(lines)


@dataclass
class _State:
    equity: float
    positions: dict[str, Position] = field(default_factory=dict)


class Backtester:
    def __init__(self, config: SystemConfig, initial_equity: float = 1_000_000.0):
        config.validate()
        self.config = config
        self.initial_equity = initial_equity
        self.strategy = CompositeStrategy(config)
        self.allocator = PortfolioAllocator(config)
        self.risk = RiskManager(config)

    # ------------------------------------------------------------------
    def run(self, ohlc: dict[str, pd.DataFrame]) -> BacktestResult:
        """バックテストを実行する。

        Args:
            ohlc: pair -> DataFrame(index=DatetimeIndex,
                                    columns=[open, high, low, close])
        """
        cfg = self.config
        self.strategy.reset()
        self.risk.reset()

        # 共通の日付インデックス(全ペアが揃う日のみ)
        index = ohlc[cfg.pairs[0]].index
        for pair in cfg.pairs[1:]:
            index = index.intersection(ohlc[pair].index)
        index = index.sort_values()

        closes = {p: ohlc[p].loc[index, "close"] for p in cfg.pairs}
        scores = self.strategy.compute_scores(closes)
        atrs = {
            p: atr(
                ohlc[p].loc[index, "high"],
                ohlc[p].loc[index, "low"],
                closes[p],
                cfg.risk.atr_window,
            )
            for p in cfg.pairs
        }
        vols = {
            p: realized_vol(closes[p], cfg.portfolio.vol_window) for p in cfg.pairs
        }

        warmup = max(
            cfg.strategy.ema_slow,
            cfg.strategy.momentum_window,
            cfg.portfolio.vol_window,
        )

        state = _State(equity=self.initial_equity)
        equity_curve: dict[pd.Timestamp, float] = {}
        leverage_curve: dict[pd.Timestamp, float] = {}
        weights_history: dict[pd.Timestamp, dict[str, float]] = {}
        trades: list[TradeRecord] = []
        swap_total = 0.0
        cost_total = 0.0
        prev_closes: dict[str, float] = {}

        for i, date in enumerate(index):
            bars = {p: ohlc[p].loc[date] for p in cfg.pairs}
            self.risk.start_day(state.equity)

            # --- 1. スワップ付与(前日から持ち越したポジション) ---
            if prev_closes:
                for pair, pos in state.positions.items():
                    if pos.units == 0:
                        continue
                    s = daily_swap(
                        pos.units,
                        prev_closes[pair],
                        cfg.rate_diff(pair),
                        haircut=cfg.swap_haircut,
                    )
                    state.equity += s
                    swap_total += s

            # --- 2. ストップ / 利確判定(ギャップ時は寄付きで不利側約定) ---
            for pair in list(state.positions):
                pos = state.positions[pair]
                if pos.units == 0:
                    continue
                hit = self.risk.check_exit(
                    pos, high=bars[pair]["high"], low=bars[pair]["low"]
                )
                if hit is None:
                    continue
                trigger = (
                    pos.stop_price if hit == "stop_loss" else pos.take_profit_price
                )
                open_px = bars[pair]["open"]
                if pos.units > 0:
                    fill = min(trigger, open_px) if hit == "stop_loss" else trigger
                else:
                    fill = max(trigger, open_px) if hit == "stop_loss" else trigger
                ref = prev_closes.get(pair, pos.entry_price)
                state.equity += pos.units * (fill - ref)
                cost = abs(pos.units) * fill * cfg.transaction_cost
                state.equity -= cost
                cost_total += cost
                trades.append(
                    TradeRecord(date, pair, pos.units, 0.0, float(fill), hit)
                )
                del state.positions[pair]

            # --- 3. 終値評価 ---
            for pair, pos in state.positions.items():
                ref = prev_closes.get(pair, pos.entry_price)
                state.equity += pos.units * (bars[pair]["close"] - ref)
            self.risk.update_equity(state.equity)

            prices = {p: float(bars[p]["close"]) for p in cfg.pairs}

            # --- 4. リバランス ---
            if i >= warmup:
                if self.risk.circuit_breaker_active:
                    # サーキットブレーカー: 全決済
                    for pair in list(state.positions):
                        pos = state.positions[pair]
                        cost = abs(pos.units) * prices[pair] * cfg.transaction_cost
                        state.equity -= cost
                        cost_total += cost
                        trades.append(
                            TradeRecord(
                                date, pair, pos.units, 0.0, prices[pair],
                                "circuit_breaker",
                            )
                        )
                        del state.positions[pair]
                    self.strategy.reset()
                else:
                    day_scores = {}
                    day_vols = {}
                    for pair in cfg.pairs:
                        row = scores[pair].loc[date]
                        sig = self.strategy.signal_at(
                            pair, row["trend"], row["carry"], row["rsi"]
                        )
                        day_scores[pair] = sig.score
                        day_vols[pair] = float(vols[pair].loc[date])

                    weights = self.allocator.target_weights(day_scores, day_vols)
                    weights_history[date] = weights
                    current_units = {
                        p: pos.units for p, pos in state.positions.items()
                    }
                    orders = self.allocator.build_orders(
                        weights, state.equity, prices, current_units
                    )
                    for order in orders:
                        held = current_units.get(order.pair, 0.0)
                        target = order.target_units
                        # 新規建て禁止中はポジション縮小・解消のみ許可
                        if not self.risk.can_open_new:
                            if held == 0:
                                continue
                            if abs(target) > abs(held) or target * held < 0:
                                target = 0.0
                        if target == held:
                            continue
                        price = prices[order.pair]
                        cost = abs(target - held) * price * cfg.transaction_cost
                        state.equity -= cost
                        cost_total += cost
                        trades.append(
                            TradeRecord(date, order.pair, held, target, price,
                                        order.reason)
                        )
                        if target == 0.0:
                            state.positions.pop(order.pair, None)
                        else:
                            pos = Position(
                                pair=order.pair, units=target, entry_price=price
                            )
                            self.risk.attach_stops(
                                pos, float(atrs[order.pair].loc[date])
                            )
                            state.positions[order.pair] = pos

            # --- 5. レバレッジ上限の強制執行 ---
            # 建玉後の価格変動や残高減少でグロスが上限を超えた場合は比例縮小する
            if state.equity > 0:
                gross = sum(
                    pos.notional(prices[p]) for p, pos in state.positions.items()
                )
                cap = cfg.portfolio.max_gross_leverage * state.equity
                if gross > cap:
                    scale = cap / gross
                    for pair, pos in list(state.positions.items()):
                        new_units = pos.units * scale
                        cost = (
                            abs(pos.units - new_units)
                            * prices[pair]
                            * cfg.transaction_cost
                        )
                        state.equity -= cost
                        cost_total += cost
                        trades.append(
                            TradeRecord(
                                date, pair, pos.units, new_units, prices[pair],
                                "leverage_cap",
                            )
                        )
                        pos.units = new_units

            equity_curve[date] = state.equity
            leverage_curve[date] = (
                sum(
                    pos.notional(prices[p])
                    for p, pos in state.positions.items()
                )
                / state.equity
                if state.equity > 0
                else float("inf")
            )
            prev_closes = prices

        return BacktestResult(
            equity_curve=pd.Series(equity_curve, name="equity"),
            trades=trades,
            swap_pnl=swap_total,
            cost_paid=cost_total,
            daily_leverage=pd.Series(leverage_curve, name="gross_leverage"),
            weights_history=pd.DataFrame(weights_history).T,
        )
