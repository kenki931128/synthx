"""Test for synthx.method."""

import numpy as np
import polars as pl
import pytest
from pytest_mock import MockerFixture
from scipy.optimize import OptimizeResult

import synthx as sx
from synthx.errors import NoFeasibleModelError
from synthx.method import sensitivity_check, synthetic_control


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


class TestPlaceboTest:
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

    def test_placebo_test_expected_type(self, dummy_dataset: sx.Dataset) -> None:
        effect_test, effects_placebo, sc_test, scs_placebo = sx.placebo_test(dummy_dataset)
        assert isinstance(effect_test, float)
        assert isinstance(effects_placebo, list)
        assert isinstance(sc_test, sx.SyntheticControlResult)
        assert isinstance(scs_placebo, list)

    def test_placebo_test_number_of_units(self, dummy_dataset: sx.Dataset) -> None:
        _, effects_placebo, _, scs_placebo = sx.placebo_test(dummy_dataset)
        control_units = (
            dummy_dataset.data.filter(
                ~dummy_dataset.data[dummy_dataset.unit_column].is_in(
                    dummy_dataset.intervention_units
                )
            )[dummy_dataset.unit_column]
            .unique()
            .to_list()
        )
        assert len(effects_placebo) == len(control_units)
        assert len(scs_placebo) == len(control_units)


class TestSensitivityCheck:
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

    def test_sensitivity_check_no_uplift_found(
        self, dummy_dataset: sx.Dataset, mocker: MockerFixture
    ) -> None:
        mocker.patch(
            'synthx.method.synthetic_control',
            return_value=mocker.Mock(estimate_effects=mocker.Mock(return_value=1.5)),
        )
        mocker.patch('synthx.stats.calc_p_value', return_value=0.1)

        uplift = sensitivity_check(
            dummy_dataset, effects_placebo=[1.0, 1.1, 1.2], p_value_target=0.03
        )

        assert uplift is None

    def test_sensitivity_check_optimization_failure(
        self, dummy_dataset: sx.Dataset, mocker: MockerFixture
    ) -> None:
        mocker.patch('synthx.method.synthetic_control', side_effect=NoFeasibleModelError)

        uplift = sensitivity_check(
            dummy_dataset, effects_placebo=[1.0, 1.1, 1.2], p_value_target=0.03
        )

        assert uplift is None

    def test_sensitivity_check_no_significant_difference(
        self, dummy_dataset: sx.Dataset, mocker: MockerFixture
    ) -> None:
        mocker.patch(
            'synthx.method.synthetic_control',
            return_value=mocker.Mock(estimate_effects=mocker.Mock(return_value=0.1)),
        )
        mocker.patch('synthx.stats.calc_p_value', return_value=0.01)

        uplift = sensitivity_check(
            dummy_dataset, effects_placebo=[1.0, 1.1, 1.2], p_value_target=0.03
        )

        assert uplift is None
