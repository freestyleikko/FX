"""リスク管理。

- ATR ベースのストップロス / テイクプロフィット
- 口座ドローダウンによるサーキットブレーカー(全決済 + クールダウン)
- 日次損失限度による新規建て禁止
"""

from __future__ import annotations

from fx_trader.config import SystemConfig
from fx_trader.models import Position


class RiskManager:
    def __init__(self, config: SystemConfig):
        self.config = config
        self.peak_equity: float = 0.0
        self.cooldown_remaining: int = 0
        self.day_start_equity: float | None = None
        self._daily_blocked: bool = False

    # ------------------------------------------------------------------
    # ポジション単位
    # ------------------------------------------------------------------
    def attach_stops(self, position: Position, atr_value: float) -> Position:
        """建玉に ATR ベースのストップ / 利確価格を設定する。"""
        r = self.config.risk
        if position.units == 0 or atr_value != atr_value or atr_value <= 0:
            return position
        direction = 1.0 if position.units > 0 else -1.0
        position.stop_price = position.entry_price - direction * r.stop_loss_atr * atr_value
        position.take_profit_price = (
            position.entry_price + direction * r.take_profit_atr * atr_value
        )
        return position

    @staticmethod
    def check_exit(position: Position, high: float, low: float) -> str | None:
        """当日の高値・安値でストップ / 利確に到達したかを判定する。

        Returns:
            "stop_loss" / "take_profit" / None
        """
        if position.units == 0:
            return None
        if position.units > 0:
            if position.stop_price is not None and low <= position.stop_price:
                return "stop_loss"
            if (
                position.take_profit_price is not None
                and high >= position.take_profit_price
            ):
                return "take_profit"
        else:
            if position.stop_price is not None and high >= position.stop_price:
                return "stop_loss"
            if (
                position.take_profit_price is not None
                and low <= position.take_profit_price
            ):
                return "take_profit"
        return None

    # ------------------------------------------------------------------
    # 口座単位
    # ------------------------------------------------------------------
    def start_day(self, equity: float) -> None:
        """日次処理の開始。日初残高を記録し、クールダウンを消化する。"""
        self.day_start_equity = equity
        self._daily_blocked = False
        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1

    def update_equity(self, equity: float) -> None:
        """残高更新。ドローダウンと日次損失をチェックする。"""
        r = self.config.risk
        self.peak_equity = max(self.peak_equity, equity)

        if (
            self.peak_equity > 0
            and equity < self.peak_equity * (1.0 - r.max_drawdown)
            and self.cooldown_remaining == 0
        ):
            # サーキットブレーカー発動
            self.cooldown_remaining = r.cooldown_days
            # 発動時点をピークとして仕切り直す(連続発動を防ぐ)
            self.peak_equity = equity

        if (
            self.day_start_equity
            and equity < self.day_start_equity * (1.0 - r.daily_loss_limit)
        ):
            self._daily_blocked = True

    @property
    def circuit_breaker_active(self) -> bool:
        """クールダウン中(全決済 + 新規建て停止)かどうか。"""
        return self.cooldown_remaining > 0

    @property
    def can_open_new(self) -> bool:
        """新規建てが許可されているか。"""
        return not self.circuit_breaker_active and not self._daily_blocked

    def reset(self) -> None:
        self.peak_equity = 0.0
        self.cooldown_remaining = 0
        self.day_start_equity = None
        self._daily_blocked = False
