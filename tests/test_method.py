"""Test for synthx.method."""

import numpy as np
import polars as pl
import pytest
from pytest_mock import MockerFixture
from scipy.optimize import OptimizeResult

import synthx as sx
from synthx.errors import NoFeasibleModelError
from synthx.method import synthetic_control


class TestSyntheticControl:
    @pytest.fixture
    def dummy_dataset(self) -> sx.Dataset:
        data = pl.DataFrame(
            {
                'unit': [1, 1, 1, 2, 2, 2, 3, 3, 3],
                'time': [1, 2, 3, 1, 2, 3, 1, 2, 3],
                'y': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                'cov1': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            }
        )
        return sx.Dataset(
            data=data,
            unit_column='unit',
            time_column='time',
            y_column='y',
            covariate_columns=['cov1'],
            intervention_units=[1],
            intervention_time=2,
        )

    def test_synthetic_control_single_unit(
        self, dummy_dataset: sx.Dataset, mocker: MockerFixture
    ) -> None:
        mocker.patch(
            'scipy.optimize.minimize', return_value=OptimizeResult(x=[0.5, 0.5], success=True)
        )
        result = synthetic_control(dummy_dataset)
        assert isinstance(result, sx.SyntheticControlResult)
        assert np.allclose(result.control_unit_weights, [0.5, 0.5])

    def test_synthetic_control_multiple_units(self, dummy_dataset: sx.Dataset) -> None:
        # TODO: Once implemented, update this test.
        dummy_dataset.intervention_units = [1, 2]
        with pytest.raises(NotImplementedError):
            synthetic_control(dummy_dataset)

    def test_synthetic_control_optimization_failure(
        self, dummy_dataset: sx.Dataset, mocker: MockerFixture
    ) -> None:
        mocker.patch('scipy.optimize.minimize', return_value=OptimizeResult(x=None, success=False))
        with pytest.raises(NoFeasibleModelError):
            synthetic_control(dummy_dataset)
