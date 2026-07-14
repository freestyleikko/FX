import numpy as np
import pandas as pd
import pytest

from fx_trader.config import SystemConfig
from fx_trader.strategy import CompositeStrategy


@pytest.fixture
def config():
    return SystemConfig()


@pytest.fixture
def strategy(config):
    return CompositeStrategy(config)


def test_trend_component_bounds_and_direction(strategy):
    # 一貫した上昇トレンドでは正、下落トレンドでは負のスコア
    up = pd.Series(100.0 * np.exp(np.linspace(0, 0.3, 300)))
    down = pd.Series(100.0 * np.exp(np.linspace(0, -0.3, 300)))
    t_up = strategy.trend_component(up)
    t_down = strategy.trend_component(down)
    assert t_up.iloc[-1] > 0.5
    assert t_down.iloc[-1] < -0.5
    assert t_up.abs().max() <= 1.0
    assert t_down.abs().max() <= 1.0


def test_carry_component_positive_for_yen_funding(strategy, config):
    # 現行金利では3ペアとも対円金利差プラス → キャリーはロング方向
    for pair in config.pairs:
        assert strategy.carry_component(pair) > 0


def test_overheat_filter_reduces_long_when_overbought():
    m_normal = CompositeStrategy._overheat_multiplier(0.5, 50.0, 70.0, 30.0)
    m_hot = CompositeStrategy._overheat_multiplier(0.5, 85.0, 70.0, 30.0)
    m_extreme = CompositeStrategy._overheat_multiplier(0.5, 100.0, 70.0, 30.0)
    assert m_normal == 1.0
    assert 0.0 < m_hot < 1.0
    assert m_extreme == 0.0
    # ショートシグナルは買われ過ぎの影響を受けない
    assert CompositeStrategy._overheat_multiplier(-0.5, 85.0, 70.0, 30.0) == 1.0


def test_hysteresis_prevents_flip_flop(strategy):
    p = strategy.config.strategy
    # entry_threshold 未満では建てない
    sig = strategy.signal_at("USDJPY", trend=0.1, carry=0.1, rsi_value=50.0)
    assert sig.score == 0.0
    # 閾値超えで建てる
    sig = strategy.signal_at("USDJPY", trend=0.5, carry=0.5, rsi_value=50.0)
    assert sig.score > p.entry_threshold
    # 保有中は entry 未満でも exit 以上なら維持
    weak = (p.entry_threshold + p.exit_threshold) / 2
    sig = strategy.signal_at("USDJPY", trend=weak, carry=weak, rsi_value=50.0)
    assert sig.score != 0.0
    # exit 未満まで減衰したら解消
    sig = strategy.signal_at("USDJPY", trend=0.01, carry=0.01, rsi_value=50.0)
    assert sig.score == 0.0


def test_compute_scores_shape(strategy, config):
    rng = np.random.default_rng(0)
    n = 200
    closes = {
        p: pd.Series(150.0 * np.exp(np.cumsum(rng.normal(0, 0.006, n))))
        for p in config.pairs
    }
    scores = strategy.compute_scores(closes)
    assert set(scores) == set(config.pairs)
    for df in scores.values():
        assert list(df.columns) == ["trend", "carry", "rsi", "raw_score"]
        assert len(df) == n
