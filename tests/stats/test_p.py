"""Test for synthx.stats.p."""

import pytest

import synthx as sx


class TestCalcPValue:
    def test_single_values(self) -> None:
        effects_test = 1.5
        effects_control = [1.2, 2.1]
        p_value = sx.stats.calc_p_value(effects_test, effects_control)
        assert isinstance(p_value, float)

    def test_list_values(self) -> None:
        effects_test = [1.2, 0.8, 1.5]
        effects_control = [0.9, 1.1, 1.3]
        p_value = sx.stats.calc_p_value(effects_test, effects_control)
        assert isinstance(p_value, float)

    def test_equal_means(self) -> None:
        effects_test = [1.0, 2.0, 3.0]
        effects_control = [1.0, 2.0, 3.0]
        p_value = sx.stats.calc_p_value(effects_test, effects_control)
        assert p_value == 1.0
