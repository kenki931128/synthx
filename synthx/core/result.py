"""Class containing synthetic control result."""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

import synthx as sx


class SyntheticControlResult:
    """Synthetic control result."""

    def __init__(
        self,
        *,
        dataset: sx.Dataset,
        control_unit_weights: np.ndarray,
    ) -> None:
        """Initialize SyntheticControlResult.

        Args:
            dataset (sx.Dataset): The dataset used for synthetic control analysis.
            control_unit_weights (np.ndarray): The weights of control units.
        """
        self.dataset = dataset
        self.control_unit_weights = control_unit_weights

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
    def y_test(self) -> np.ndarray:
        """Get data in test group as an array.

        Returns:
            np.ndarray: numpy array of test group.
        """
        return self.df_test[self.dataset.y_column].to_numpy()

    @property
    def y_control(self) -> np.ndarray:
        """Get data in control group as an array.

        Returns:
            np.ndarray: numpy array of control group.
        """
        df_control_pivoted = self.df_control.pivot(
            index=self.dataset.time_column,
            columns=self.dataset.unit_column,
            values=self.dataset.y_column,
        ).drop(self.dataset.time_column)
        arr_control_pivoted = df_control_pivoted.to_numpy()
        return np.sum(arr_control_pivoted * self.control_unit_weights, axis=1)

    def estimate_effects(self) -> float:
        """Estimate the effects of the intervention.

        Returns:
            float: The estimated effect of the intervention.
        """
        # dataset before intervention
        pre_df = self.dataset.data.filter(
            self.dataset.data[self.dataset.time_column] < self.dataset.intervention_time
        )
        pre_result = SyntheticControlResult(
            dataset=sx.Dataset(
                pre_df,
                unit_column=self.dataset.unit_column,
                time_column=self.dataset.time_column,
                y_column=self.dataset.y_column,
                covariate_columns=self.dataset.covariate_columns,
                intervention_units=self.dataset.intervention_units,
                intervention_time=0,
            ),
            control_unit_weights=self.control_unit_weights,
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
                intervention_time=0,
            ),
            control_unit_weights=self.control_unit_weights,
        )
        return np.mean(post_result.y_test - post_result.y_control) - np.mean(
            pre_result.y_test - pre_result.y_control
        )

    def plot(self, save: Optional[str] = None) -> None:
        """Plot the target variable over time for both test and control units.

        Args:
            save (Optional[str]): file path to save the plot. If None, the plot will be displayed.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.df_test[self.dataset.time_column], self.y_test, label='Test')
        plt.plot(self.df_test[self.dataset.time_column], self.y_control, label='Control')
        plt.axvline(
            self.dataset.intervention_time, color='red', linestyle='--', label='Intervention Time'  # type: ignore
        )
        plt.xlabel('Time')
        plt.ylabel('Target Variable')
        plt.title('Target Variable Over Time')
        plt.legend()
        if save is not None:
            plt.savefig(save)
        else:
            plt.show()
