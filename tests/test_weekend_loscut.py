from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from fx_trader.backtest import Backtester
from fx_trader.config import SystemConfig
from fx_trader.data import generate_synthetic_ohlc
from fx_trader.risk import RiskManager
from fx_trader.schedule import in_weekend_flat_window, should_liquidate_for_weekend


# ----------------------------------------------------------------------
# 週末ノーポジション時間帯の判定
# ----------------------------------------------------------------------
def test_weekend_window_boundaries():
    # 2026-07-10 は金曜日
    assert not in_weekend_flat_window(datetime(2026, 7, 10, 16, 59))
    assert in_weekend_flat_window(datetime(2026, 7, 10, 17, 0))   # 金曜17:00〜
    assert in_weekend_flat_window(datetime(2026, 7, 11, 12, 0))   # 土曜
    assert in_weekend_flat_window(datetime(2026, 7, 12, 12, 0))   # 日曜
    assert in_weekend_flat_window(datetime(2026, 7, 13, 6, 59))   # 月曜7:00前
    assert not in_weekend_flat_window(datetime(2026, 7, 13, 7, 0))  # 月曜7:00〜
    assert not in_weekend_flat_window(datetime(2026, 7, 14, 12, 0))  # 火曜


def test_should_liquidate_lead_time():
    # 金曜16:40、30分前倒しなら 17:10 相当 → 決済すべき
    assert should_liquidate_for_weekend(datetime(2026, 7, 10, 16, 40))
    assert not should_liquidate_for_weekend(datetime(2026, 7, 10, 16, 20))


# ----------------------------------------------------------------------
# バックテストでの週末フラット
# ----------------------------------------------------------------------
def test_no_positions_held_over_weekend():
    config = SystemConfig()
    assert config.weekend_flat  # 既定で有効
    ohlc = generate_synthetic_ohlc(n_days=400, seed=5)
    result = Backtester(config, initial_equity=1_000_000).run(ohlc)
    fridays = result.daily_leverage[result.daily_leverage.index.dayofweek == 4]
    # 金曜の引け時点では建玉ゼロ(レバレッジ0)
    assert float(fridays.max()) == 0.0


def test_weekend_flat_can_be_disabled():
    config = SystemConfig()
    config.weekend_flat = False
    ohlc = generate_synthetic_ohlc(n_days=400, seed=5)
    result = Backtester(config, initial_equity=1_000_000).run(ohlc)
    fridays = result.daily_leverage[result.daily_leverage.index.dayofweek == 4]
    assert float(fridays.max()) > 0.0  # 持ち越しがある


# ----------------------------------------------------------------------
# 強制ロスカット
# ----------------------------------------------------------------------
def test_margin_ratio_and_trigger():
    config = SystemConfig()  # 必要証拠金4%、維持率50%未満で発動
    rm = RiskManager(config)
    # 資金100万・建玉500万 → 必要証拠金20万 → 維持率500%
    assert rm.margin_ratio(1_000_000, 5_000_000) == pytest.approx(5.0)
    assert not rm.forced_loscut_triggered(1_000_000, 5_000_000)
    # 維持率50%を下回る水準(有効証拠金9万円台)で発動
    assert rm.forced_loscut_triggered(90_000, 5_000_000)
    # 有効証拠金ゼロ以下は無条件で発動
    assert rm.forced_loscut_triggered(-1.0, 5_000_000)
    # 建玉がなければ発動しない
    assert not rm.forced_loscut_triggered(-1.0, 0.0)


def _crash_ohlc(n_days: int = 200, crash_day: int = 150) -> dict[str, pd.DataFrame]:
    """ノイズ付き上昇トレンドの後、1日で25%急落する人工データ。"""
    rng = np.random.default_rng(0)
    out = {}
    for pair, base in (("USDJPY", 150.0), ("EURJPY", 163.0), ("GBPJPY", 190.0)):
        dates = pd.bdate_range("2024-01-01", periods=n_days)
        ret = 0.0008 + rng.normal(0, 0.004, n_days)  # 上昇トレンド + 通常ボラ
        close = base * np.exp(np.cumsum(ret))
        close[crash_day:] *= 0.75  # 急落して戻らない(窓ではなく日中の連続下落)
        open_ = np.roll(close, 1)
        open_[0] = close[0]
        open_[crash_day] = close[crash_day - 1]  # 急落日は寄付き=前日終値
        high = np.maximum(open_, close) * 1.001
        low = np.minimum(open_, close) * 0.999
        out[pair] = pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close}, index=dates
        )
    return out


def test_forced_loscut_fires_on_crash():
    config = SystemConfig.aggressive()
    config.weekend_flat = False  # 週末決済に助けられないようにする
    # 必要証拠金を上げて(=許容レバレッジを下げて)発動しやすくする
    config.risk.margin_requirement = 0.15
    config.risk.forced_loscut_level = 1.0
    result = Backtester(config, initial_equity=1_000_000).run(_crash_ohlc())
    reasons = {t.reason for t in result.trades}
    assert "forced_loscut" in reasons
    # 強制決済後も残高はプラス(全損前に切られている)
    assert result.equity_curve.iloc[-1] > 0
