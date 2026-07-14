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
    # --- 強制ロスカット(証拠金維持率ベース) ---
    # 必要証拠金率(建玉総額に対する比率)。0.04 = 4% = 国内法定の25倍相当
    margin_requirement: float = 0.04
    # 証拠金維持率(有効証拠金 ÷ 必要証拠金)がこの水準を切ったら全建玉を強制決済
    forced_loscut_level: float = 0.50


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
    # 週末ノーポジション: 金曜17:00〜月曜7:00(JST)はポジションを一切持たない。
    # 日足バックテストでは「金曜の引けで全決済・金曜は新規建てなし」として扱う
    weekend_flat: bool = True
    friday_close_hour_jst: int = 17
    monday_open_hour_jst: int = 7

    @classmethod
    def aggressive(cls) -> "SystemConfig":
        """アグレッシブ設定。レバレッジ上限5倍は維持したまま、
        配分を上限近くまで使い、トレンドに長く乗る。

        リスクも比例して増える(想定最大DDは30%程度)ことに注意。
        """
        c = cls()
        c.strategy.entry_threshold = 0.10
        c.strategy.exit_threshold = 0.03
        # 各ペアに高いリスク目標を与え、グロス上限5倍まで使わせる
        c.portfolio.target_pair_vol = 0.35
        c.portfolio.max_pair_leverage = 3.0
        # ストップを広げ、利確を外してトレンドに乗り続ける
        c.risk.stop_loss_atr = 3.5
        c.risk.take_profit_atr = 12.0
        c.risk.max_drawdown = 0.30
        c.risk.daily_loss_limit = 0.06
        c.risk.cooldown_days = 3
        c.validate()
        return c

    def rate_diff(self, pair: str) -> float:
        """ペアの金利差(ベース通貨金利 − 円金利、年率%)を返す。"""
        base = BASE_CURRENCY[pair]
        return self.policy_rates[base] - self.policy_rates["JPY"]

    def validate(self) -> None:
        if self.portfolio.max_gross_leverage > 5.0:
            raise ValueError("レバレッジ上限は自己資金の5倍までです")
        if not 0.0 < self.risk.margin_requirement <= 1.0:
            raise ValueError("margin_requirement は 0 より大きく 1 以下にしてください")
        if self.strategy.exit_threshold >= self.strategy.entry_threshold:
            raise ValueError("exit_threshold は entry_threshold より小さくしてください")
        w = self.strategy.trend_weight + self.strategy.carry_weight
        if abs(w - 1.0) > 1e-9:
            raise ValueError("trend_weight + carry_weight は 1.0 にしてください")
