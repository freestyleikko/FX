import pytest

from fx_trader.backtest import Backtester
from fx_trader.config import SystemConfig
from fx_trader.data import generate_synthetic_ohlc


@pytest.fixture(scope="module")
def result():
    config = SystemConfig()
    bt = Backtester(config, initial_equity=1_000_000)
    ohlc = generate_synthetic_ohlc(n_days=500, seed=1)
    return bt.run(ohlc)


def test_backtest_runs_and_produces_curve(result):
    assert len(result.equity_curve) == 500
    assert result.equity_curve.iloc[-1] > 0


def test_leverage_never_exceeds_limit(result):
    # 建玉時点でグロス5倍以内(価格変動による日中の微超過は建玉基準では発生しない)
    assert result.daily_leverage.max() <= 5.0 + 0.1


def test_metrics_keys(result):
    m = result.metrics()
    for key in (
        "total_return", "cagr", "sharpe", "max_drawdown",
        "num_trades", "swap_pnl", "cost_paid", "max_leverage",
    ):
        assert key in m


def test_trades_recorded_with_reasons(result):
    assert len(result.trades) > 0
    reasons = {t.reason.split(" ")[0] for t in result.trades}
    assert any(r.startswith("rebalance") for r in reasons)


def test_deterministic_with_same_seed():
    config = SystemConfig()
    ohlc = generate_synthetic_ohlc(n_days=300, seed=7)
    r1 = Backtester(config).run(ohlc)
    r2 = Backtester(config).run(ohlc)
    assert r1.equity_curve.equals(r2.equity_curve)


def test_leverage_validation_rejects_over_five():
    config = SystemConfig()
    config.portfolio.max_gross_leverage = 10.0
    with pytest.raises(ValueError):
        Backtester(config)
