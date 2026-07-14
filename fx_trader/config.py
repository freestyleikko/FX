"""システム設定。

通貨ペア・政策金利・レバレッジ上限・戦略パラメータを一元管理する。
政策金利は API 接続が無い前提のため静的な初期値を持つが、
`SystemConfig.policy_rates` を差し替えれば任意の時点の金利で動作する。
"""

from __future__ import annotations

from dataclasses import dataclass, field

# 対象通貨ペア(すべて対円)
PAIRS: tuple[str, ...] = ("USDJPY", "EURJPY", "GBPJPY")

# ペアごとのベース通貨(金利差計算に使用)
BASE_CURRENCY: dict[str, str] = {
    "USDJPY": "USD",
    "EURJPY": "EUR",
    "GBPJPY": "GBP",
}

# 政策金利の初期値(年率%)。実運用時は最新値に更新すること。
DEFAULT_POLICY_RATES: dict[str, float] = {
    "USD": 4.50,
    "EUR": 2.15,
    "GBP": 4.00,
    "JPY": 0.50,
}


@dataclass
class StrategyParams:
    """シグナル生成パラメータ。"""

    ema_fast: int = 20
    ema_slow: int = 60
    momentum_window: int = 90
    rsi_window: int = 14
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0
    # トレンド:キャリーの合成重み(合計1.0)
    trend_weight: float = 0.6
    carry_weight: float = 0.4
    # 金利差スコアの正規化スケール(この金利差[%pt]で carry スコアが飽和に近づく)
    carry_scale: float = 4.0
    # トレンド乖離の正規化スケール(価格に対する EMA 乖離率)
    trend_scale: float = 0.02
    # |score| がこの値未満ならノーポジション(エントリー閾値)
    entry_threshold: float = 0.15
    # 保有中はこの値までスコアが減衰するまで維持(ヒステリシス、entry より小さく)
    exit_threshold: float = 0.05


@dataclass
class RiskParams:
    """リスク管理パラメータ。"""

    atr_window: int = 14
    stop_loss_atr: float = 2.5     # ATR何倍で損切り
    take_profit_atr: float = 5.0   # ATR何倍で利確
    max_drawdown: float = 0.15     # 高値から15%下落で全決済+新規停止
    daily_loss_limit: float = 0.03  # 1日3%超の損失で当日の新規建て禁止
    # サーキットブレーカー発動後、再開までのクールダウン日数
    cooldown_days: int = 5


@dataclass
class PortfolioParams:
    """資金配分パラメータ。"""

    max_gross_leverage: float = 5.0   # 総建玉は自己資金の5倍まで(絶対上限)
    max_pair_leverage: float = 2.5    # 1ペアあたりの上限
    vol_window: int = 60              # 実現ボラ計測窓(日)
    target_pair_vol: float = 0.10     # 各ペアに割り当てる年率リスク目標
    min_trade_fraction: float = 0.02  # 目標との乖離がこの比率未満ならリバランスしない


@dataclass
class SystemConfig:
    """システム全体の設定。"""

    pairs: tuple[str, ...] = PAIRS
    policy_rates: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_POLICY_RATES)
    )
    strategy: StrategyParams = field(default_factory=StrategyParams)
    risk: RiskParams = field(default_factory=RiskParams)
    portfolio: PortfolioParams = field(default_factory=PortfolioParams)
    # 取引コスト(片道、価格に対する比率)。0.002% ≒ USDJPY 0.3銭相当
    transaction_cost: float = 0.00002
    # スワップの業者取り分(受取りは (1-h) 倍に減額、支払いは (1+h) 倍に増額)
    swap_haircut: float = 0.10

    @classmethod
    def aggressive(cls, max_gross_leverage: float = 5.0) -> "SystemConfig":
        """アグレッシブ設定。配分をレバレッジ上限近くまで使い、
        トレンドに長く乗る。

        Args:
            max_gross_leverage: グロスレバレッジ上限(既定5倍、法定上限25倍まで)。
                上限に比例してリスク目標・DD許容も自動スケールするため、
                10倍なら想定最大DDは50%規模になることに注意。
        """
        c = cls()
        scale = max_gross_leverage / 5.0
        c.strategy.entry_threshold = 0.10
        c.strategy.exit_threshold = 0.03
        # 各ペアに高いリスク目標を与え、グロス上限まで使わせる
        c.portfolio.max_gross_leverage = max_gross_leverage
        c.portfolio.target_pair_vol = 0.35 * scale
        c.portfolio.max_pair_leverage = 3.0 * scale
        # ストップを広げ、利確を外してトレンドに乗り続ける
        c.risk.stop_loss_atr = 3.5
        c.risk.take_profit_atr = 12.0
        # レバレッジに応じて損失許容も拡大(上限あり)
        c.risk.max_drawdown = min(0.60, 0.30 * scale)
        c.risk.daily_loss_limit = min(0.15, 0.06 * scale)
        c.risk.cooldown_days = 3
        c.validate()
        return c

    def rate_diff(self, pair: str) -> float:
        """ペアの金利差(ベース通貨金利 − 円金利、年率%)を返す。"""
        base = BASE_CURRENCY[pair]
        return self.policy_rates[base] - self.policy_rates["JPY"]

    def validate(self) -> None:
        if self.portfolio.max_gross_leverage > 25.0:
            raise ValueError(
                "レバレッジ上限は25倍(国内FXの法定上限)までです"
            )
        if self.strategy.exit_threshold >= self.strategy.entry_threshold:
            raise ValueError("exit_threshold は entry_threshold より小さくしてください")
        w = self.strategy.trend_weight + self.strategy.carry_weight
        if abs(w - 1.0) > 1e-9:
            raise ValueError("trend_weight + carry_weight は 1.0 にしてください")
