"""Class for dataset."""

from datetime import date
from typing import Any, Optional, Union

import matplotlib.pyplot as plt
import polars as pl

from synthx.errors import (
    ColumnNotFoundError,
    InconsistentTimestampsError,
    InvalidColumnTypeError,
    InvalidInterventionTimeError,
    InvalidInterventionUnitError,
    InvalidNormalizationError,
)
from synthx.stats import norm


class Dataset:
    """Dataset for synthetic control."""

    def __init__(
        self,
        data: pl.DataFrame,
        *,
        unit_column: str,
        time_column: str,
        y_column: str,
        covariate_columns: Optional[list[str]],
        intervention_units: Union[Any, list[Any]],
        intervention_time: Union[int, date],
        validation_time: Optional[Union[int, date]] = None,
        norm: Optional[str] = None,
    ) -> None:
        """Initialize the Dataset.

        Args:
            data (pl.DataFram): The input dataset as a Polars DataFrame.
            unit_column (str): The column representing the unit.
            time_column (str): The column representing the time period.
            y_column (str): The column representing the target variable.
            covariate_columns (Optional[list[str]]): The columns representing the covariates.
            intervention_units (Union[Any, list[Any]]): A list of intervented units
            intervention_time (Union[int, date]): When the intervention or event happens.
            validation_time (Optional[Union[int, date]]): validation time if needed.
            norm (Optional[str]): If not None, will normalize the y_column.
                it should be 'z_standardize'
        """
        self.data = data
        self.unit_column = unit_column
        self.time_column = time_column
        self.y_column = y_column
        self.covariate_columns = covariate_columns
        self.intervention_units = (
            intervention_units if isinstance(intervention_units, list) else [intervention_units]
        )
        self.intervention_time = intervention_time
        self.validation_time = validation_time
        self.norm = norm
        self.__validate()
        self.__normalization()

    def plot(self, units: Optional[list[str]] = None, save: Optional[str] = None) -> None:
        """Plot the dataset and display its characteristics.

        Args:
            units (Optional[list[str]]): only plot data of this variable if specified.
            save (Optional[str]): file path to save the plot. If None, the plot will be displayed.
        """
        # Plot the target variable over time for each unit
        units = units if units is not None else self.data[self.unit_column].unique().to_list()
        plt.figure(figsize=(10, 6))
        for unit in units:
            unit_data = self.data.filter(pl.col(self.unit_column) == unit)
            plt.plot(unit_data[self.time_column], unit_data[self.y_column], label=f'Unit {unit}')

        # Add vertical line for validation time and intervention time
        if self.validation_time is not None:
            plt.axvline(
                self.validation_time, color='orange', linestyle='--', label='Validation Time'  # type: ignore
            )
        plt.axvline(
            self.intervention_time, color='red', linestyle='--', label='Intervention Time'  # type: ignore
        )

        plt.xlabel('Time')
        plt.ylabel('Target Variable')
        plt.title('Target Variable Over Time')
        plt.legend()
        if save is not None:
            plt.savefig(save)
        else:
            plt.show()

    def __validate(self) -> None:
        """Validate the dataset and raise appropriate errors if any issues are found.

        This method checks for the presence and data types of the specified columns.

        Raises:
            ColumnNotFoundError if a required column is missing.
            InvalidColumnTypeError if a column has an invalid data type.
        """
        if self.unit_column not in self.data.columns:
            raise ColumnNotFoundError(self.unit_column)
        if len(self.intervention_units) == 0:
            raise InvalidInterventionUnitError(f'at least, need to specify one intervention unit.')
        for intervention_unit in self.intervention_units:
            if intervention_unit not in self.data[self.unit_column].unique():
                raise InvalidInterventionUnitError(f'{intervention_unit} does not exist.')

        if self.time_column not in self.data.columns:
            raise ColumnNotFoundError(self.time_column)
        if self.data[self.time_column].dtype not in [pl.Int64, pl.Date]:
            raise InvalidColumnTypeError(f'{self.time_column} should be int or date type.')
        if (
            self.data[self.time_column].dtype == pl.Date and isinstance(self.intervention_time, int)
        ) or (
            self.data[self.time_column].dtype == pl.Int64
            and isinstance(self.intervention_time, date)
        ):
            raise InvalidColumnTypeError(
                f'{self.time_column} and intervention_time should have the same type.'
            )
        if self.intervention_time > self.data[self.time_column].max():  # type: ignore
            raise InvalidInterventionTimeError(f'no date point at {self.intervention_time} time.')
        if self.validation_time is not None and type(self.intervention_time) != type(
            self.validation_time
        ):
            raise InvalidColumnTypeError(
                f'intervention_time and validation_time should have the same type.'
            )
        if self.validation_time is not None and self.intervention_time <= self.validation_time:  # type: ignore
            raise InvalidInterventionTimeError(
                f'intervention_time should be later than validation_time.'
            )

        if self.y_column not in self.data.columns:
            raise ColumnNotFoundError(self.y_column)
        if self.data[self.y_column].dtype != pl.Float64:
            raise InvalidColumnTypeError(f'{self.y_column} should be float.')

        units = self.data[self.unit_column].unique()
        timestamps = self.data[self.time_column].unique()
        for unit in units:
            unit_timestamps = self.data.filter(self.data[self.unit_column] == unit)[
                self.time_column
            ]
            if not timestamps.equals(unit_timestamps):
                raise InconsistentTimestampsError(f'Unit {unit} has inconsistent timestamps.')

        if self.covariate_columns is not None:
            for covariate_column in self.covariate_columns:
                if covariate_column not in self.data.columns:
                    raise ColumnNotFoundError(covariate_column)
                if self.data[covariate_column].dtype not in [pl.Int64, pl.Float64]:
                    raise InvalidColumnTypeError(f'{covariate_column} should be int or float.')

    def __normalization(self) -> None:
        """Perform normalization on the data based on the specified normalization method.

        This method applies the selected normalization technique to the data,
        grouping by the unit column and applying the corresponding standardization
        function to each group. The available normalization methods are:
        - 'z': Z-score standardization
        - 'cv': Coefficient of Variation (CV) standardization
        - 'yeo_johnson': Yeo-Johnson transformation followed by Z-score standardization

        If no normalization method is specified (i.e., `self.norm` is None), nothing is performed.

        Raises:
            InvalidNormalizationError: If an invalid normalization method is specified.
        """
        if self.norm is None:
            return

        if self.norm == 'z':
            self.data = self.data.group_by(self.unit_column).map_groups(
                lambda df: norm.z_standardize(df, self.y_column)
            )
        elif self.norm == 'cv':
            self.data = self.data.group_by(self.unit_column).map_groups(
                lambda df: norm.cv_standardize(df, self.y_column)
            )
        elif self.norm == 'yeo_johnson':
            self.data = self.data.group_by(self.unit_column).map_groups(
                lambda df: norm.yeo_johnson_standardize(df, self.y_column)
            )
        else:
            raise InvalidNormalizationError('norm should be `z`, `cv` or `yeo_johnson`.')
