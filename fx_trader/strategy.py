"""複合シグナル戦略。

各ペアについて以下を合成し、-1〜+1 のスコアを算出する。

1. トレンド成分: EMA(fast/slow) の乖離率 + 中期モメンタムを tanh 正規化
2. キャリー成分: 政策金利差(対円)を tanh 正規化。金利差が大きいほどロング優位
3. 過熱フィルタ: RSI が買われ過ぎならロングを縮小、売られ過ぎならショートを縮小

さらにエントリー/イグジット閾値によるヒステリシスで小刻みな往復売買を防ぐ。
"""

from __future__ import annotations

import math

import pandas as pd

from fx_trader.carry import carry_score
from fx_trader.config import SystemConfig
from fx_trader.indicators import ema, momentum, rsi
from fx_trader.models import Signal


class CompositeStrategy:
    """トレンド + キャリー + 過熱フィルタの複合戦略。"""

    def __init__(self, config: SystemConfig):
        config.validate()
        self.config = config
        # ヒステリシス用: 直前バーで建玉ありと判断したペア
        self._active: set[str] = set()

    # ------------------------------------------------------------------
    # 成分計算
    # ------------------------------------------------------------------
    def trend_component(self, close: pd.Series) -> pd.Series:
        """EMAクロス乖離とモメンタムを平均した -1〜+1 のトレンドスコア。"""
        p = self.config.strategy
        fast = ema(close, p.ema_fast)
        slow = ema(close, p.ema_slow)
        # EMA乖離率を trend_scale で正規化
        cross = ((fast - slow) / slow / p.trend_scale).apply(math.tanh)
        # 中期モメンタム(±10%で概ね飽和)
        mom = (momentum(close, p.momentum_window) / 0.10).apply(
            lambda x: math.tanh(x) if pd.notna(x) else float("nan")
        )
        return (cross + mom.fillna(cross)) / 2.0

    def carry_component(self, pair: str) -> float:
        """金利差に基づくキャリースコア(-1〜+1)。"""
        p = self.config.strategy
        return carry_score(self.config.rate_diff(pair), p.carry_scale)

    @staticmethod
    def _overheat_multiplier(
        raw_score: float, rsi_value: float, overbought: float, oversold: float
    ) -> float:
        """RSI 過熱時にシグナルを減衰させる乗数(0.0〜1.0)。

        買われ過ぎ(RSI>overbought)ではロング方向のみ、
        売られ過ぎ(RSI<oversold)ではショート方向のみを線形に縮小する。
        RSI が極値(100/0)に達した時点で 0 になる。
        """
        if raw_score > 0 and rsi_value > overbought:
            return max(0.0, 1.0 - (rsi_value - overbought) / (100.0 - overbought))
        if raw_score < 0 and rsi_value < oversold:
            return max(0.0, 1.0 - (oversold - rsi_value) / oversold)
        return 1.0

    # ------------------------------------------------------------------
    # シグナル生成
    # ------------------------------------------------------------------
    def compute_scores(self, closes: dict[str, pd.Series]) -> dict[str, pd.DataFrame]:
        """全期間分の成分スコアを一括計算する(バックテスト用)。

        Returns:
            pair -> DataFrame(columns=[trend, carry, rsi, raw_score])
        """
        p = self.config.strategy
        out: dict[str, pd.DataFrame] = {}
        for pair in self.config.pairs:
            close = closes[pair]
            trend = self.trend_component(close)
            carry = self.carry_component(pair)
            rsi_series = rsi(close, p.rsi_window)
            raw = p.trend_weight * trend + p.carry_weight * carry
            df = pd.DataFrame(
                {"trend": trend, "carry": carry, "rsi": rsi_series, "raw_score": raw}
            )
            out[pair] = df
        return out

    def signal_at(
        self, pair: str, trend: float, carry: float, rsi_value: float
    ) -> Signal:
        """1時点分のシグナルを生成する(ヒステリシス込み)。"""
        p = self.config.strategy
        raw = p.trend_weight * trend + p.carry_weight * carry
        score = raw * self._overheat_multiplier(
            raw, rsi_value, p.rsi_overbought, p.rsi_oversold
        )

        # ヒステリシス: 未保有なら entry_threshold、保有中なら exit_threshold で判定
        threshold = p.exit_threshold if pair in self._active else p.entry_threshold
        if abs(score) < threshold:
            score = 0.0
            self._active.discard(pair)
        else:
            self._active.add(pair)

        return Signal(
            pair=pair, score=score, trend_score=trend, carry_score=carry, rsi=rsi_value
        )

    def reset(self) -> None:
        """ヒステリシス状態をリセットする。"""
        self._active.clear()
