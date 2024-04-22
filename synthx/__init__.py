"""Module: synthx."""

from synthx import stats
from synthx.core.dataset import Dataset
from synthx.core.result import SyntheticControlResult
from synthx.core.sample import sample
from synthx.method import placebo_test, sensitivity_check, synthetic_control


__all__ = [
    'stats',
    'Dataset',
    'sample',
    'synthetic_control',
    'placebo_test',
    'sensitivity_check',
    'SyntheticControlResult',
]
