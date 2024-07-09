"""Module: synthx."""

from synthx import stats
from synthx.core.dataset import Dataset
from synthx.core.result import SyntheticControlResult
from synthx.core.sample import sample
from synthx.method import (
    placebo_sensitivity_check,
    placebo_test,
    synthetic_control,
    ttest_sensitivity_check,
)


__all__ = [
    'stats',
    'Dataset',
    'sample',
    'synthetic_control',
    'placebo_test',
    'placebo_sensitivity_check',
    'ttest_sensitivity_check',
    'SyntheticControlResult',
]
