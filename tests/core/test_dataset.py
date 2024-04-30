"""Test for synthx.core.dataset."""

import polars as pl
import pytest
from pytest_mock import MockerFixture

import synthx as sx
from synthx.errors import (
    ColumnNotFoundError,
    InconsistentTimestampsError,
    InvalidColumnTypeError,
    InvalidInterventionTimeError,
    InvalidInterventionUnitError,
    InvalidNormalizationError,
)


class TestDataset:
    @pytest.fixture
    def sample_data(self) -> pl.DataFrame:
        df = pl.DataFrame(
            {
                'unit': [1, 1, 1, 2, 2, 2],
                'time': [1, 2, 3, 1, 2, 3],
                'y': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                'cov1': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                'cov2': [1, 2, 3, 4, 5, 6],
            }
        )
        return df

    def test_init(self, sample_data: pl.DataFrame) -> None:
        dataset = sx.Dataset(
            data=sample_data,
            unit_column='unit',
            time_column='time',
            y_column='y',
            covariate_columns=['cov1', 'cov2'],
            intervention_units=[1],
            intervention_time=2,
        )
        assert dataset.data.equals(sample_data)
        assert dataset.unit_column == 'unit'
        assert dataset.time_column == 'time'
        assert dataset.y_column == 'y'
        assert dataset.covariate_columns == ['cov1', 'cov2']
        assert dataset.intervention_units == [1]
        assert dataset.intervention_time == 2

    def test_init_single_intervention_unit(self, sample_data: pl.DataFrame) -> None:
        dataset = sx.Dataset(
            data=sample_data,
            unit_column='unit',
            time_column='time',
            y_column='y',
            covariate_columns=['cov1', 'cov2'],
            intervention_units=1,
            intervention_time=2,
        )
        assert dataset.intervention_units == [1]

    def test_init_with_validation_time(self, sample_data: pl.DataFrame) -> None:
        dataset = sx.Dataset(
            data=sample_data,
            unit_column='unit',
            time_column='time',
            y_column='y',
            covariate_columns=['cov1', 'cov2'],
            intervention_units=1,
            intervention_time=3,
            validation_time=2,
        )
        assert dataset.validation_time == 2

    def test_plot(self, sample_data: pl.DataFrame, mocker: MockerFixture) -> None:
        dataset = sx.Dataset(
            data=sample_data,
            unit_column='unit',
            time_column='time',
            y_column='y',
            covariate_columns=['cov1', 'cov2'],
            intervention_units=[1],
            intervention_time=2,
        )
        mocker.patch('matplotlib.pyplot.show')
        dataset.plot()
        assert True  # Assert that the method runs without errors

    def test_plot_with_save(self, sample_data: pl.DataFrame, mocker: MockerFixture) -> None:
        dataset = sx.Dataset(
            data=sample_data,
            unit_column='unit',
            time_column='time',
            y_column='y',
            covariate_columns=['cov1', 'cov2'],
            intervention_units=[1],
            intervention_time=2,
        )
        mocker.patch('matplotlib.pyplot.savefig')
        dataset.plot(save='test.png')
        assert True  # Assert that the method runs without errors

    def test_filtered_by_lift_and_moe(self, sample_data: pl.DataFrame) -> None:
        dataset = sx.Dataset(
            data=sample_data,
            unit_column='unit',
            time_column='time',
            y_column='y',
            covariate_columns=['cov1', 'cov2'],
            intervention_units=[1],
            intervention_time=2,
        )
        filtered_dataset = dataset.filtered_by_lift_and_moe(lift_threshold=2.0, moe_threshold=1.0)
        assert filtered_dataset.data.shape == (6, 5)  # Assuming no units are filtered out

    def test_filtered_by_lift_and_moe_invalid_lift_threshold(
        self, sample_data: pl.DataFrame
    ) -> None:
        dataset = sx.Dataset(
            data=sample_data,
            unit_column='unit',
            time_column='time',
            y_column='y',
            covariate_columns=['cov1', 'cov2'],
            intervention_units=[1],
            intervention_time=2,
        )
        with pytest.raises(ValueError):
            dataset.filtered_by_lift_and_moe(lift_threshold=-1.0)

    def test_filtered_by_lift_and_moe_invalid_moe_threshold(
        self, sample_data: pl.DataFrame
    ) -> None:
        dataset = sx.Dataset(
            data=sample_data,
            unit_column='unit',
            time_column='time',
            y_column='y',
            covariate_columns=['cov1', 'cov2'],
            intervention_units=[1],
            intervention_time=2,
        )
        with pytest.raises(ValueError):
            dataset.filtered_by_lift_and_moe(moe_threshold=0.0)

    def test_validate_missing_unit_column(self, sample_data: pl.DataFrame) -> None:
        with pytest.raises(ColumnNotFoundError):
            sx.Dataset(
                data=sample_data,
                unit_column='missing_column',
                time_column='time',
                y_column='y',
                covariate_columns=['cov1', 'cov2'],
                intervention_units=[1],
                intervention_time=2,
            )

    def test_validate_missing_intervention_unit(self, sample_data: pl.DataFrame) -> None:
        with pytest.raises(InvalidInterventionUnitError):
            sx.Dataset(
                data=sample_data,
                unit_column='unit',
                time_column='time',
                y_column='y',
                covariate_columns=['cov1', 'cov2'],
                intervention_units=[],
                intervention_time=1,
            )

    def test_validate_invalid_intervention_unit(self, sample_data: pl.DataFrame) -> None:
        with pytest.raises(InvalidInterventionUnitError):
            sx.Dataset(
                data=sample_data,
                unit_column='unit',
                time_column='time',
                y_column='y',
                covariate_columns=['cov1', 'cov2'],
                intervention_units=[3],
                intervention_time=2,
            )

    def test_validate_missing_time_column(self, sample_data: pl.DataFrame) -> None:
        with pytest.raises(ColumnNotFoundError):
            sx.Dataset(
                data=sample_data,
                unit_column='unit',
                time_column='missing_column',
                y_column='y',
                covariate_columns=['cov1', 'cov2'],
                intervention_units=[1],
                intervention_time=2,
            )

    def test_validate_invalid_time_column_type(self, sample_data: pl.DataFrame) -> None:
        sample_data = sample_data.with_columns([pl.col('time').cast(pl.Float64)])
        with pytest.raises(InvalidColumnTypeError):
            sx.Dataset(
                data=sample_data,
                unit_column='unit',
                time_column='time',
                y_column='y',
                covariate_columns=['cov1', 'cov2'],
                intervention_units=[1],
                intervention_time=2,
            )

    def test_validate_invalid_intervention_time(self, sample_data: pl.DataFrame) -> None:
        with pytest.raises(InvalidInterventionTimeError):
            sx.Dataset(
                data=sample_data,
                unit_column='unit',
                time_column='time',
                y_column='y',
                covariate_columns=['cov1', 'cov2'],
                intervention_units=[1],
                intervention_time=4,
            )

    def test_validate_invalid_intervention_time_vs_validation_time(
        self, sample_data: pl.DataFrame
    ) -> None:
        with pytest.raises(InvalidInterventionTimeError):
            sx.Dataset(
                data=sample_data,
                unit_column='unit',
                time_column='time',
                y_column='y',
                covariate_columns=['cov1', 'cov2'],
                intervention_units=[1],
                intervention_time=4,
                validation_time=5,
            )

    def test_validate_missing_y_column(self, sample_data: pl.DataFrame) -> None:
        with pytest.raises(ColumnNotFoundError):
            sx.Dataset(
                data=sample_data,
                unit_column='unit',
                time_column='time',
                y_column='missing_column',
                covariate_columns=['cov1', 'cov2'],
                intervention_units=[1],
                intervention_time=2,
            )

    def test_validate_invalid_y_column_type(self, sample_data: pl.DataFrame) -> None:
        sample_data = sample_data.with_columns([pl.col('y').cast(pl.Int64)])
        with pytest.raises(InvalidColumnTypeError):
            sx.Dataset(
                data=sample_data,
                unit_column='unit',
                time_column='time',
                y_column='y',
                covariate_columns=['cov1', 'cov2'],
                intervention_units=[1],
                intervention_time=2,
            )

    def test_validate_inconsistent_timestamps(self) -> None:
        inconsistent_data = pl.DataFrame(
            {
                'unit': [1, 1, 1, 2, 2],
                'time': [1, 2, 3, 1, 2],
                'y': [1.0, 2.0, 3.0, 4.0, 5.0],
                'cov1': [0.1, 0.2, 0.3, 0.4, 0.5],
                'cov2': [1, 2, 3, 4, 5],
            }
        )
        with pytest.raises(InconsistentTimestampsError):
            sx.Dataset(
                data=inconsistent_data,
                unit_column='unit',
                time_column='time',
                y_column='y',
                covariate_columns=['cov1', 'cov2'],
                intervention_units=[1],
                intervention_time=2,
            )

    def test_validate_missing_covariate_column(self, sample_data: pl.DataFrame) -> None:
        with pytest.raises(ColumnNotFoundError):
            sx.Dataset(
                data=sample_data,
                unit_column='unit',
                time_column='time',
                y_column='y',
                covariate_columns=['cov1', 'missing_column'],
                intervention_units=[1],
                intervention_time=2,
            )

    def test_validate_invalid_covariate_column_type(self, sample_data: pl.DataFrame) -> None:
        sample_data = sample_data.with_columns([pl.col('cov1').cast(pl.Utf8)])
        with pytest.raises(InvalidColumnTypeError):
            sx.Dataset(
                data=sample_data,
                unit_column='unit',
                time_column='time',
                y_column='y',
                covariate_columns=['cov1', 'cov2'],
                intervention_units=[1],
                intervention_time=2,
            )

    def test_normalization_z(self, sample_data: pl.DataFrame) -> None:
        dataset = sx.Dataset(
            data=sample_data,
            unit_column='unit',
            time_column='time',
            y_column='y',
            covariate_columns=['cov1', 'cov2'],
            intervention_units=[1],
            intervention_time=2,
            norm='z',
        )
        assert sample_data.columns == dataset.data.columns

    def test_normalization_cv(self, sample_data: pl.DataFrame) -> None:
        dataset = sx.Dataset(
            data=sample_data,
            unit_column='unit',
            time_column='time',
            y_column='y',
            covariate_columns=['cov1', 'cov2'],
            intervention_units=[1],
            intervention_time=2,
            norm='cv',
        )
        assert sample_data.columns == dataset.data.columns

    def test_normalization_yeo_johnson(self, sample_data: pl.DataFrame) -> None:
        dataset = sx.Dataset(
            data=sample_data,
            unit_column='unit',
            time_column='time',
            y_column='y',
            covariate_columns=['cov1', 'cov2'],
            intervention_units=[1],
            intervention_time=2,
            norm='yeo_johnson',
        )
        assert sample_data.columns == dataset.data.columns

    def test_normalization_invalid(self, sample_data: pl.DataFrame) -> None:
        with pytest.raises(InvalidNormalizationError):
            sx.Dataset(
                data=sample_data,
                unit_column='unit',
                time_column='time',
                y_column='y',
                covariate_columns=['cov1', 'cov2'],
                intervention_units=[1],
                intervention_time=2,
                norm='invalid_norm',
            )
