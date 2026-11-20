import numpy as np
import pytest

from bwalloc.allocators import (ABLATIONS, METHODS, max_min, max_min_full, proportional,
                                proportional_full, round_robin)
from bwalloc.metrics import jain_index, window_metrics


def test_round_robin_caps_at_demand_and_does_not_redistribute():
    a = round_robin(np.array([1.0, 10.0, 10.0]), C=12)
    np.testing.assert_allclose(a, [1, 4, 4])  # 3 units left unused


def test_max_min_textbook_example():
    # classic example: C=10, demands 2, 2.6, 4, 5 -> 2, 2.6, 2.7, 2.7
    a = max_min(np.array([2, 2.6, 4, 5]), C=10)
    np.testing.assert_allclose(a, [2, 2.6, 2.7, 2.7])


def test_max_min_gives_everything_when_uncongested():
    d = np.array([1.0, 2.0, 3.0])
    np.testing.assert_allclose(max_min(d, 100), d)


def test_proportional_eq1():
    np.testing.assert_allclose(proportional(np.array([2.0, 6.0]), C=4), [1, 3])
    np.testing.assert_allclose(proportional(np.array([2.0, 6.0]), C=100), [2, 6])


def test_proportional_full_eq2_uses_all_capacity():
    a = proportional_full(np.array([1.0, 3.0]), C=100)
    np.testing.assert_allclose(a, [25, 75])


def test_max_min_full_spreads_leftover():
    a = max_min_full(np.array([1.0, 3.0]), C=8)
    assert a.sum() == pytest.approx(8)
    assert a[1] > a[0]


@pytest.mark.parametrize("name", list(METHODS) + list(ABLATIONS))
def test_capacity_never_exceeded(name):
    fn = {**METHODS, **ABLATIONS}[name][0]
    rng = np.random.default_rng(0)
    for _ in range(200):
        d = rng.lognormal(0, 1.5, rng.integers(2, 60))
        C = rng.uniform(0.1, 2) * d.sum()
        a = fn(d, C)
        assert a.sum() <= C * (1 + 1e-9)
        assert (a >= -1e-12).all()


def test_oracle_prop_is_jain_optimal_and_beats_max_min_oracle():
    """Proportional on true demand gives equal satisfaction -> Jain = 1."""
    r = np.array([[1.0, 4.0, 10.0]])
    C = 9.0
    a_prop = proportional(r[0], C)[None, :]
    a_mm = max_min(r[0], C)[None, :]
    assert window_metrics(a_prop, r, C)["jain"][0] == pytest.approx(1.0)
    assert window_metrics(a_mm, r, C)["jain"][0] < 1.0


def test_stranded_counts_only_idle_capacity_while_someone_is_short():
    R = np.array([[1.0, 5.0]])
    A = np.array([[4.0, 2.0]])   # user 0 over-allocated by 3, user 1 short by 3
    m = window_metrics(A, R, C=6)
    assert m["wasted_frac"][0] == pytest.approx(3 / 6)
    assert m["stranded_frac"][0] == pytest.approx(3 / 6)
    m2 = window_metrics(np.array([[4.0, 5.0]]), R, C=9)  # nobody short
    assert m2["stranded_frac"][0] == 0


def test_jain_bounds():
    assert jain_index(np.ones(5)) == pytest.approx(1.0)
    assert jain_index(np.array([1, 0, 0, 0])) == pytest.approx(0.25)


def test_window_metrics_simple():
    R = np.array([[2.0, 2.0]])
    A = np.array([[2.0, 0.5]])
    m = window_metrics(A, R, C=4)
    assert m["utilization"][0] == pytest.approx(2.5 / 4)
    assert m["unmet"][0] == pytest.approx(1.5)
    assert m["starvation"][0] == pytest.approx(0.5)  # second user got 25%
    assert m["jain"][0] == pytest.approx((1.25) ** 2 / (2 * (1 + 0.0625)))
