"""Test for synthx.core.sample."""

import polars as pl
import pytest

import synthx as sx


class TestSample:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.n_units = 100
        self.n_time = 20
        self.n_observed_covariates = 3
        self.n_unobserved_covariates = 2
        self.intervention_units = [1]
        self.intervention_time = 16
        self.intervention_effect = 1
        self.noise_effect = 0.1
        self.scale = 1
        self.seed = 42

    def test_sample_output_shape(self) -> None:
        df = sx.sample(
            n_units=self.n_units,
            n_time=self.n_time,
            n_observed_covariates=self.n_observed_covariates,
            n_unobserved_covariates=self.n_unobserved_covariates,
            intervention_units=self.intervention_units,
            intervention_time=self.intervention_time,
            intervention_effect=self.intervention_effect,
            noise_effect=self.noise_effect,
            seed=self.seed,
        )
        assert isinstance(df, pl.DataFrame)
        assert df.shape == (self.n_units * self.n_time, 3 + self.n_observed_covariates)

    def test_sample_column_names(self) -> None:
        df = sx.sample(
            n_units=self.n_units,
            n_time=self.n_time,
            n_observed_covariates=self.n_observed_covariates,
            n_unobserved_covariates=self.n_unobserved_covariates,
            intervention_units=self.intervention_units,
            intervention_time=self.intervention_time,
            intervention_effect=self.intervention_effect,
            noise_effect=self.noise_effect,
            scale=self.scale,
            seed=self.seed,
        )
        expected_columns = ['unit', 'time', 'y'] + [
            f'covariate_{i + 1}' for i in range(self.n_observed_covariates)
        ]
        assert list(df.columns) == expected_columns

    def test_sample_reproducibility(self) -> None:
        df1 = sx.sample(
            n_units=self.n_units,
            n_time=self.n_time,
            n_observed_covariates=self.n_observed_covariates,
            n_unobserved_covariates=self.n_unobserved_covariates,
            intervention_units=self.intervention_units,
            intervention_time=self.intervention_time,
            intervention_effect=self.intervention_effect,
            noise_effect=self.noise_effect,
            scale=self.scale,
            seed=self.seed,
        )
        df2 = sx.sample(
            n_units=self.n_units,
            n_time=self.n_time,
            n_observed_covariates=self.n_observed_covariates,
            n_unobserved_covariates=self.n_unobserved_covariates,
            intervention_units=self.intervention_units,
            intervention_time=self.intervention_time,
            intervention_effect=self.intervention_effect,
            noise_effect=self.noise_effect,
            scale=self.scale,
            seed=self.seed,
        )
        assert df1.equals(df2)

    def test_sample_invalid_arguments(self) -> None:
        with pytest.raises(ValueError):
            sx.sample(
                n_units=0,
                n_time=self.n_time,
                n_observed_covariates=self.n_observed_covariates,
                n_unobserved_covariates=self.n_unobserved_covariates,
                intervention_units=self.intervention_units,
                intervention_time=self.intervention_time,
                intervention_effect=self.intervention_effect,
                noise_effect=self.noise_effect,
                scale=self.scale,
                seed=self.seed,
            )
        with pytest.raises(ValueError):
            sx.sample(
                n_units=self.n_units,
                n_time=-1,
                n_observed_covariates=self.n_observed_covariates,
                n_unobserved_covariates=self.n_unobserved_covariates,
                intervention_units=self.intervention_units,
                intervention_time=self.intervention_time,
                intervention_effect=self.intervention_effect,
                noise_effect=self.noise_effect,
                scale=self.scale,
                seed=self.seed,
            )
        with pytest.raises(ValueError):
            sx.sample(
                n_units=self.n_units,
                n_time=self.n_time,
                n_observed_covariates=-1,
                n_unobserved_covariates=self.n_unobserved_covariates,
                intervention_units=self.intervention_units,
                intervention_time=self.intervention_time,
                intervention_effect=self.intervention_effect,
                noise_effect=self.noise_effect,
                scale=self.scale,
                seed=self.seed,
            )
        with pytest.raises(ValueError):
            sx.sample(
                n_units=self.n_units,
                n_time=self.n_time,
                n_observed_covariates=self.n_observed_covariates,
                n_unobserved_covariates=-1,
                intervention_units=self.intervention_units,
                intervention_time=self.intervention_time,
                intervention_effect=self.intervention_effect,
                noise_effect=self.noise_effect,
                scale=self.scale,
                seed=self.seed,
            )
        with pytest.raises(ValueError):
            sx.sample(
                n_units=self.n_units,
                n_time=self.n_time,
                n_observed_covariates=self.n_observed_covariates,
                n_unobserved_covariates=self.n_unobserved_covariates,
                intervention_units=[200],
                intervention_time=self.intervention_time,
                intervention_effect=self.intervention_effect,
                noise_effect=self.noise_effect,
                scale=self.scale,
                seed=self.seed,
            )
        with pytest.raises(ValueError):
            sx.sample(
                n_units=self.n_units,
                n_time=self.n_time,
                n_observed_covariates=self.n_observed_covariates,
                n_unobserved_covariates=self.n_unobserved_covariates,
                intervention_units=self.intervention_units,
                intervention_time=30,
                intervention_effect=self.intervention_effect,
                noise_effect=self.noise_effect,
                scale=self.scale,
                seed=self.seed,
            )
        with pytest.raises(ValueError):
            sx.sample(
                n_units=self.n_units,
                n_time=self.n_time,
                n_observed_covariates=self.n_observed_covariates,
                n_unobserved_covariates=self.n_unobserved_covariates,
                intervention_units=self.intervention_units,
                intervention_time=self.intervention_time,
                intervention_effect=self.intervention_effect,
                noise_effect=self.noise_effect,
                scale=-1,
                seed=self.seed,
            )
