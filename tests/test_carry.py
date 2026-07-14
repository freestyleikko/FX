import math

import pytest

from fx_trader.carry import carry_score, daily_swap, swap_for_positions
from fx_trader.config import SystemConfig


def test_carry_score_sign_and_bounds():
    assert carry_score(4.0, scale=4.0) == pytest.approx(math.tanh(1.0))
    assert carry_score(-2.0, scale=4.0) < 0
    assert -1.0 <= carry_score(-100.0, scale=4.0) <= carry_score(100.0, scale=4.0) <= 1.0
    assert carry_score(0.0, scale=4.0) == 0.0


def test_daily_swap_long_receives_short_pays():
    # 金利差 +4% のペア: ロングは受取り、ショートは支払い
    long_swap = daily_swap(units=10_000, price=150.0, rate_diff_pct=4.0)
    short_swap = daily_swap(units=-10_000, price=150.0, rate_diff_pct=4.0)
    assert long_swap > 0
    assert short_swap < 0
    assert long_swap == pytest.approx(10_000 * 150.0 * 0.04 / 365)


def test_daily_swap_haircut_asymmetry():
    # 業者取り分: 受取りは減額、支払いは増額
    receive = daily_swap(10_000, 150.0, 4.0, haircut=0.1)
    pay = daily_swap(-10_000, 150.0, 4.0, haircut=0.1)
    assert receive < daily_swap(10_000, 150.0, 4.0)
    assert abs(pay) > abs(daily_swap(-10_000, 150.0, 4.0))


def test_swap_for_positions_uses_config_rates():
    config = SystemConfig()
    config.policy_rates = {"USD": 4.5, "EUR": 2.0, "GBP": 4.0, "JPY": 0.5}
    positions = {"USDJPY": 10_000.0, "EURJPY": -10_000.0, "GBPJPY": 0.0}
    prices = {"USDJPY": 150.0, "EURJPY": 163.0, "GBPJPY": 190.0}
    total = swap_for_positions(positions, prices, config)
    h = config.swap_haircut
    expected = (
        10_000 * 150.0 * 0.04 / 365 * (1 - h)       # USDロング受取り
        - 10_000 * 163.0 * 0.015 / 365 * (1 + h)    # EURショート支払い
    )
    assert total == pytest.approx(expected)
