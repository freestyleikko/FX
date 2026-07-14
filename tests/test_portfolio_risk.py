import pytest

from fx_trader.config import SystemConfig
from fx_trader.models import Position
from fx_trader.portfolio import PortfolioAllocator
from fx_trader.risk import RiskManager


@pytest.fixture
def config():
    return SystemConfig()


# ----------------------------------------------------------------------
# ポートフォリオ
# ----------------------------------------------------------------------
def test_gross_leverage_never_exceeds_five(config):
    alloc = PortfolioAllocator(config)
    # 低ボラ + フルスコアで上限に張り付かせる
    scores = {"USDJPY": 1.0, "EURJPY": -1.0, "GBPJPY": 1.0}
    vols = {"USDJPY": 0.02, "EURJPY": 0.02, "GBPJPY": 0.02}
    weights = alloc.target_weights(scores, vols)
    gross = sum(abs(w) for w in weights.values())
    assert gross <= config.portfolio.max_gross_leverage + 1e-9
    # 比例縮小なので符号は維持される
    assert weights["USDJPY"] > 0
    assert weights["EURJPY"] < 0


def test_pair_leverage_cap(config):
    alloc = PortfolioAllocator(config)
    weights = alloc.target_weights(
        {"USDJPY": 1.0, "EURJPY": 0.0, "GBPJPY": 0.0},
        {"USDJPY": 0.01, "EURJPY": 0.10, "GBPJPY": 0.10},
    )
    assert abs(weights["USDJPY"]) <= config.portfolio.max_pair_leverage + 1e-9


def test_nan_vol_gives_zero_weight(config):
    alloc = PortfolioAllocator(config)
    weights = alloc.target_weights(
        {"USDJPY": 1.0}, {"USDJPY": float("nan")}
    )
    assert weights["USDJPY"] == 0.0


def test_small_diff_skips_rebalance(config):
    alloc = PortfolioAllocator(config)
    equity, price = 1_000_000.0, 150.0
    target_units = 0.5 * equity / price
    # 現在保有が目標とほぼ同じ → 注文なし
    orders = alloc.build_orders(
        {"USDJPY": 0.5}, equity, {"USDJPY": price},
        {"USDJPY": target_units * 0.99},
    )
    assert orders == []
    # 大きく乖離 → 注文あり
    orders = alloc.build_orders(
        {"USDJPY": 0.5}, equity, {"USDJPY": price}, {"USDJPY": 0.0}
    )
    assert len(orders) == 1


# ----------------------------------------------------------------------
# リスク管理
# ----------------------------------------------------------------------
def test_stops_attached_correct_side(config):
    rm = RiskManager(config)
    atr_value = 1.0
    long_pos = rm.attach_stops(
        Position("USDJPY", units=10_000, entry_price=150.0), atr_value
    )
    assert long_pos.stop_price == pytest.approx(150.0 - 2.5)
    assert long_pos.take_profit_price == pytest.approx(150.0 + 5.0)
    short_pos = rm.attach_stops(
        Position("USDJPY", units=-10_000, entry_price=150.0), atr_value
    )
    assert short_pos.stop_price == pytest.approx(150.0 + 2.5)
    assert short_pos.take_profit_price == pytest.approx(150.0 - 5.0)


def test_check_exit_detects_stop_and_tp(config):
    rm = RiskManager(config)
    pos = rm.attach_stops(
        Position("USDJPY", units=10_000, entry_price=150.0), 1.0
    )
    assert rm.check_exit(pos, high=150.5, low=147.0) == "stop_loss"
    assert rm.check_exit(pos, high=155.5, low=150.0) == "take_profit"
    assert rm.check_exit(pos, high=151.0, low=149.0) is None


def test_circuit_breaker_and_cooldown(config):
    rm = RiskManager(config)
    rm.start_day(1_000_000)
    rm.update_equity(1_000_000)
    assert not rm.circuit_breaker_active
    # 15%超のドローダウンで発動
    rm.update_equity(840_000)
    assert rm.circuit_breaker_active
    assert not rm.can_open_new
    # クールダウン消化後に再開
    for _ in range(config.risk.cooldown_days):
        rm.start_day(840_000)
    assert not rm.circuit_breaker_active


def test_daily_loss_limit_blocks_new_positions(config):
    rm = RiskManager(config)
    rm.start_day(1_000_000)
    rm.update_equity(960_000)  # -4% > 日次限度3%
    assert rm.can_open_new is False
    assert not rm.circuit_breaker_active  # DD遮断ではない
    rm.start_day(960_000)  # 翌日はリセット
    assert rm.can_open_new is True
