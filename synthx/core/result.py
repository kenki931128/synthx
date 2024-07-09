"""Class containing synthetic control result."""

from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy import stats

import synthx as sx


class SyntheticControlResult:
    """Synthetic control result."""

    def __init__(
        self,
        *,
        dataset: sx.Dataset,
        control_unit_weights: np.ndarray,
        scales: np.ndarray,
    ) -> None:
        """Initialize SyntheticControlResult.

        Args:
            dataset (sx.Dataset): The dataset used for synthetic control analysis.
            control_unit_weights (np.ndarray): The weights of control units.
            scales (np.ndarray): The weights to get both graph close.
        """
        self.dataset = dataset
        self.control_unit_weights = control_unit_weights
        self.scales = scales

    @property
    def df_test(self) -> pl.DataFrame:
        """Get data in test group as a DataFrame.

        Returns:
            pl.DataFrame: dataframe of test group.
        """
        return self.dataset.data.filter(
            self.dataset.data[self.dataset.unit_column].is_in(self.dataset.intervention_units)
        )

    @property
    def df_control(self) -> pl.DataFrame:
        """Get data in control group as a DataFrame.

        Returns:
            pl.DataFrame: dataframe of control group.
        """
        return self.dataset.data.filter(
            ~self.dataset.data[self.dataset.unit_column].is_in(self.dataset.intervention_units)
        )

    @property
    def x_time(self) -> np.ndarray:
        """Get the series of timestamp for x axis

        Returns:
            np.ndarray: numpy array of timestamp.
        """
        return self.dataset.data.filter(
            self.dataset.data[self.dataset.unit_column] == self.dataset.intervention_units[0]
        )[self.dataset.time_column].to_numpy()

    def y_test(self, intervention_unit: Any) -> np.ndarray:
        """Get data in test group as an array.

        Args:
            intervention_unit (Any): intervented unit.

        Returns:
            np.ndarray: numpy array of test group.
        """
        df_test_pivoted = self.df_test.pivot(
            index=self.dataset.time_column,
            columns=self.dataset.unit_column,
            values=self.dataset.y_column,
        ).drop(self.dataset.time_column)
        # column type becomes string.
        return df_test_pivoted[str(intervention_unit)].to_numpy()

    def y_control(self, intervention_unit: Any) -> np.ndarray:
        """Get data in control group as an array for specified intervention unit.

        Args:
            intervention_unit (Any): intervented unit.

        Returns:
            np.ndarray: numpy array of control group.
        """
        df_control_pivoted = self.df_control.pivot(
            index=self.dataset.time_column,
            columns=self.dataset.unit_column,
            values=self.dataset.y_column,
        ).drop(self.dataset.time_column)
        arr_control_pivoted = df_control_pivoted.to_numpy()
        control_unit_weight = self.control_unit_weights[
            self.dataset.intervention_units.index(intervention_unit)
        ]
        y_control = np.sum(arr_control_pivoted * control_unit_weight, axis=1)
        scale = self.scales[self.dataset.intervention_units.index(intervention_unit)]
        return scale * y_control

    def estimate_effects(self) -> list[float]:
        """Estimate the effects of the intervention.

        Returns:
            list[float]: The estimated effect of the intervention.
        """
        # dataset in the training period
        training_time = (
            self.dataset.validation_time
            if self.dataset.validation_time is not None
            else self.dataset.intervention_time
        )
        pre_df = self.dataset.data.filter(
            self.dataset.data[self.dataset.time_column] < training_time
        )
        pre_result = SyntheticControlResult(
            dataset=sx.Dataset(
                pre_df,
                unit_column=self.dataset.unit_column,
                time_column=self.dataset.time_column,
                y_column=self.dataset.y_column,
                covariate_columns=self.dataset.covariate_columns,
                intervention_units=self.dataset.intervention_units,
                intervention_time=pre_df[self.dataset.time_column].min(),  # type: ignore
            ),
            control_unit_weights=self.control_unit_weights,
            scales=self.scales,
        )
        # dataset after intervention
        post_df = self.dataset.data.filter(
            self.dataset.data[self.dataset.time_column] >= self.dataset.intervention_time
        )
        post_result = SyntheticControlResult(
            dataset=sx.Dataset(
                post_df,
                unit_column=self.dataset.unit_column,
                time_column=self.dataset.time_column,
                y_column=self.dataset.y_column,
                covariate_columns=self.dataset.covariate_columns,
                intervention_units=self.dataset.intervention_units,
                intervention_time=post_df[self.dataset.time_column].min(),  # type: ignore
            ),
            control_unit_weights=self.control_unit_weights,
            scales=self.scales,
        )
        return [
            (
                np.mean(
                    post_result.y_test(intervention_unit) - post_result.y_control(intervention_unit)
                )
                - np.mean(
                    pre_result.y_test(intervention_unit) - pre_result.y_control(intervention_unit)
                )
            )
            / np.mean(pre_result.y_test(intervention_unit))
            for intervention_unit in self.dataset.intervention_units
        ]

    def validation_differences(self) -> Optional[list[float]]:
        """Calculate the difference between training and validation.

        Returns:
            Optional[list[float]]: The difference between training and validation.
        """
        if self.dataset.validation_time is None:
            return None

        # dataset in the training period
        pre_df = self.dataset.data.filter(
            self.dataset.data[self.dataset.time_column] < self.dataset.validation_time
        )
        pre_result = SyntheticControlResult(
            dataset=sx.Dataset(
                pre_df,
                unit_column=self.dataset.unit_column,
                time_column=self.dataset.time_column,
                y_column=self.dataset.y_column,
                covariate_columns=self.dataset.covariate_columns,
                intervention_units=self.dataset.intervention_units,
                intervention_time=pre_df[self.dataset.time_column].min(),  # type: ignore
            ),
            control_unit_weights=self.control_unit_weights,
            scales=self.scales,
        )
        # dataset in the validation period
        val_df = self.dataset.data.filter(
            (self.dataset.data[self.dataset.time_column] >= self.dataset.validation_time)
            & (self.dataset.data[self.dataset.time_column] < self.dataset.intervention_time)
        )
        val_result = SyntheticControlResult(
            dataset=sx.Dataset(
                val_df,
                unit_column=self.dataset.unit_column,
                time_column=self.dataset.time_column,
                y_column=self.dataset.y_column,
                covariate_columns=self.dataset.covariate_columns,
                intervention_units=self.dataset.intervention_units,
                intervention_time=val_df[self.dataset.time_column].min(),  # type: ignore
            ),
            control_unit_weights=self.control_unit_weights,
            scales=self.scales,
        )

        return [
            (
                np.mean(
                    val_result.y_test(intervention_unit) - val_result.y_control(intervention_unit)
                )
                - np.mean(
                    pre_result.y_test(intervention_unit) - pre_result.y_control(intervention_unit)
                )
            )
            / np.mean(pre_result.y_test(intervention_unit))
            for intervention_unit in self.dataset.intervention_units
        ]

    def paired_ttest(self) -> list[dict[str, float]]:
        """Perform paired t-tests for each intervention unit.

        This method compares y_test with the synthetic control (y_control)
        for each intervention unit using a paired t-test.

        Returns:
            list[dict[str, float]]: A list of p-values from the paired t-tests, one for each intervention unit.
        """
        p_values: list[dict[str, float]] = []
        training_time = (
            self.dataset.validation_time
            if self.dataset.validation_time is not None
            else self.dataset.intervention_time
        )
        pre_df = self.dataset.data.filter(
            self.dataset.data[self.dataset.time_column] < training_time
        )
        pre_result = SyntheticControlResult(
            dataset=sx.Dataset(
                pre_df,
                unit_column=self.dataset.unit_column,
                time_column=self.dataset.time_column,
                y_column=self.dataset.y_column,
                covariate_columns=self.dataset.covariate_columns,
                intervention_units=self.dataset.intervention_units,
                intervention_time=pre_df[self.dataset.time_column].min(),  # type: ignore
            ),
            control_unit_weights=self.control_unit_weights,
            scales=self.scales,
        )
        post_df = self.dataset.data.filter(
            self.dataset.data[self.dataset.time_column] >= self.dataset.intervention_time
        )
        post_result = SyntheticControlResult(
            dataset=sx.Dataset(
                post_df,
                unit_column=self.dataset.unit_column,
                time_column=self.dataset.time_column,
                y_column=self.dataset.y_column,
                covariate_columns=self.dataset.covariate_columns,
                intervention_units=self.dataset.intervention_units,
                intervention_time=post_df[self.dataset.time_column].min(),  # type: ignore
            ),
            control_unit_weights=self.control_unit_weights,
            scales=self.scales,
        )

        for intervention_unit in self.dataset.intervention_units:
            # p value in the training period
            _, pre_p_value = stats.ttest_rel(
                pre_result.y_test(intervention_unit), pre_result.y_control(intervention_unit)
            )

            # p value after intervention
            _, post_p_value = stats.ttest_rel(
                post_result.y_test(intervention_unit), post_result.y_control(intervention_unit)
            )

            # p value for the total
            _, p_value = stats.ttest_rel(
                self.y_test(intervention_unit), self.y_control(intervention_unit)
            )

            p_values.append(
                {
                    'intervention_unit': intervention_unit,
                    'p_value_in_training': pre_p_value,
                    'p_value_in_intervention': post_p_value,
                    'p_value': p_value,
                }
            )
        return p_values

    def plot(self, save: Optional[str] = None) -> None:
        """Plot the target variable over time for both test and control units.

        Args:
            save (Optional[str]): file path to save the plot. If None, the plot will be displayed.
        """
        num_plots = len(self.dataset.intervention_units)
        _, axs = plt.subplots(num_plots, 1, figsize=(10, 6 * num_plots))
        # Convert axs to list in case num_plots == 1
        axs = [axs] if num_plots == 1 else axs

        for i, intervention_unit in enumerate(self.dataset.intervention_units):
            axs[i].plot(self.x_time, self.y_test(intervention_unit), label='Test')
            axs[i].plot(self.x_time, self.y_control(intervention_unit), label='Control')
            if self.dataset.validation_time is not None:
                axs[i].axvline(
                    self.dataset.validation_time,
                    color='orange',
                    linestyle='--',
                    label='Validation Time',
                )
            axs[i].axvline(
                self.dataset.intervention_time,
                color='red',
                linestyle='--',
                label='Intervention Time',
            )
            axs[i].set_xlabel('Time')
            axs[i].set_ylabel('Target Variable')
            axs[i].set_title(f'Target Variable Over Time (Intervention Unit: {intervention_unit})')
            axs[i].legend()

        plt.tight_layout()
        if save is not None:
            plt.savefig(save)
        else:
            plt.show()
