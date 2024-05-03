"""Test for synthx.core.result."""

import numpy as np
import polars as pl
import pytest
from pytest_mock import MockerFixture

import synthx as sx


class TestSyntheticControlResult:
    @pytest.fixture
    def dummy_dataset(self) -> sx.Dataset:
        data = pl.DataFrame(
            {
                'unit': [1, 1, 1, 2, 2, 2, 3, 3, 3],
                'time': [1, 2, 3, 1, 2, 3, 1, 2, 3],
                'y': [1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 3.0, 6.0, 9.0],
            }
        )
        return sx.Dataset(
            data=data,
            unit_column='unit',
            time_column='time',
            y_column='y',
            covariate_columns=[],
            intervention_units=[1],
            intervention_time=2,
        )

    @pytest.fixture
    def dummy_dataset_val(self) -> sx.Dataset:
        data = pl.DataFrame(
            {
                'unit': [1, 1, 1, 2, 2, 2, 3, 3, 3],
                'time': [1, 2, 3, 1, 2, 3, 1, 2, 3],
                'y': [1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 3.0, 6.0, 9.0],
            }
        )
        return sx.Dataset(
            data=data,
            unit_column='unit',
            time_column='time',
            y_column='y',
            covariate_columns=[],
            intervention_units=[1],
            intervention_time=3,
            validation_time=2,
        )

    @pytest.fixture
    def dummy_result(self, dummy_dataset: sx.Dataset) -> sx.SyntheticControlResult:
        control_unit_weights = np.array([[0.5, 0.5]])
        scales = np.array([2 / 5])
        return sx.SyntheticControlResult(
            dataset=dummy_dataset,
            control_unit_weights=control_unit_weights,
            scales=scales,
        )

    @pytest.fixture
    def dummy_result_val(self, dummy_dataset_val: sx.Dataset) -> sx.SyntheticControlResult:
        control_unit_weights = np.array([[0.5, 0.5]])
        scales = np.array([2 / 5])
        return sx.SyntheticControlResult(
            dataset=dummy_dataset_val,
            control_unit_weights=control_unit_weights,
            scales=scales,
        )

    def test_init(self, dummy_dataset: sx.Dataset) -> None:
        control_unit_weights = np.array([[0.5, 0.5]])
        scales = np.array([2 / 5])
        result = sx.SyntheticControlResult(
            dataset=dummy_dataset,
            control_unit_weights=control_unit_weights,
            scales=scales,
        )
        assert result.dataset == dummy_dataset
        assert np.allclose(result.control_unit_weights, control_unit_weights)

    def test_df_test(self, dummy_result: sx.SyntheticControlResult) -> None:
        expected_df_test = pl.DataFrame(
            {
                'unit': [1, 1, 1],
                'time': [1, 2, 3],
                'y': [1.0, 2.0, 3.0],
            }
        )
        assert dummy_result.df_test.equals(expected_df_test)

    def test_df_control(self, dummy_result: sx.SyntheticControlResult) -> None:
        expected_df_control = pl.DataFrame(
            {
                'unit': [2, 2, 2, 3, 3, 3],
                'time': [1, 2, 3, 1, 2, 3],
                'y': [2.0, 4.0, 6.0, 3.0, 6.0, 9.0],
            }
        )
        assert dummy_result.df_control.equals(expected_df_control)

    def test_y_test(self, dummy_result: sx.SyntheticControlResult) -> None:
        expected_y_test = np.array([1.0, 2.0, 3.0])
        assert np.allclose(dummy_result.y_test(1), expected_y_test)

    def test_y_control(self, dummy_result: sx.SyntheticControlResult) -> None:
        expected_y_control = np.array([1.0, 2.0, 3.0])
        assert np.allclose(dummy_result.y_control(1), expected_y_control)

    def test_estimate_effects(self, dummy_result: sx.SyntheticControlResult) -> None:
        expected_effect = 0
        assert dummy_result.estimate_effects()[0] == expected_effect

    def test_validation_differences_no_validation_time(
        self, dummy_result: sx.SyntheticControlResult
    ) -> None:
        assert dummy_result.validation_differences() is None

    def test_validation_differences_with_validation_time(
        self, dummy_result_val: sx.SyntheticControlResult
    ) -> None:
        expected_difference = 0
        val_result = dummy_result_val.validation_differences()
        assert val_result is not None
        assert val_result[0] == expected_difference

    def test_plot(self, dummy_result: sx.SyntheticControlResult, mocker: MockerFixture) -> None:
        mocker.patch('matplotlib.pyplot.show')
        dummy_result.plot()
        assert True  # Assert that the method runs without errors
