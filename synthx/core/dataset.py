"""Class for dataset."""

import sys
from datetime import date
from typing import Any, Optional, Union

import matplotlib.pyplot as plt
import polars as pl
from tqdm import tqdm

from synthx.errors import (
    ColumnNotFoundError,
    InconsistentTimestampsError,
    InvalidColumnTypeError,
    InvalidInterventionTimeError,
    InvalidInterventionUnitError,
    StrictFilteringError,
)


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
        self.__validate()

    def plot(
        self,
        units: Optional[list[str]] = None,
        save: Optional[str] = None,
        show_label: bool = False,
    ) -> None:
        """Plot the dataset and display its characteristics.

        Args:
            units (Optional[list[str]]): only plot data of this variable if specified.
            save (Optional[str]): file path to save the plot. If None, the plot will be displayed.
            show_label (bool): show label on the plot or not.
        """
        # Plot the target variable over time for each unit
        units = units if units is not None else self.data[self.unit_column].unique().to_list()
        plt.figure(figsize=(10, 6))
        for unit in units:
            unit_data = self.data.filter(pl.col(self.unit_column) == unit)
            plt.plot(
                unit_data[self.time_column],
                unit_data[self.y_column],
                label=f'Unit {unit}' if show_label else None,
            )

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

    def filtered_by_lift_and_moe(
        self,
        *,
        lift_threshold: Optional[float] = None,
        moe_threshold: Optional[float] = None,
        write_progress: bool = False,
    ) -> 'Dataset':
        """Filter the dataset based on lift and margin of error thresholds.

        Args:
            lift_threshold (Optional[float]): The threshold for the lift. Units with lift values
                outside the range [1 / lift_threshold, lift_threshold] will be excluded.
                If None, no lift-based filtering is performed.
            moe_threshold (Optional[float]): The threshold for the margin of error. Units with
                margin of error values outside the range [-moe_threshold, moe_threshold] will be excluded.
                If None, no margin of error-based filtering is performed.
            write_progress (bool): Whether to write progress information to stderr.

        Returns:
            Dataset: A new Dataset object containing the filtered data.

        Raises:
            ValueError: If lift_threshold or moe_threshold is not positive.
            StrictFilteringError: If all units or all intervention units are filtered out.
                Consider loosening the thresholds in this case.
        """
        if lift_threshold is not None and lift_threshold <= 0:
            raise ValueError('lift_threshold should be positive.')
        if moe_threshold is not None and moe_threshold <= 0:
            raise ValueError('moe_threshold should be positive.')

        df = self.data
        units_excluded = []

        for unit in tqdm(df[self.unit_column].unique()):
            df_unit = df.filter(pl.col(self.unit_column) == unit)

            mean = df_unit[self.y_column].mean()
            std = df_unit[self.y_column].std()

            lift = df_unit[self.y_column] / mean
            moe = (df_unit[self.y_column] - mean) / std

            str_range = f'lift: {lift.min():.3f} ~ {lift.max():.3f}, moe: {moe.min():.3f} ~ {moe.max():.3f}'  # type: ignore
            if lift_threshold is not None and (
                (lift < 1 / lift_threshold).any() or (lift_threshold < lift).any()
            ):
                if write_progress:
                    tqdm.write(
                        f'unit {unit} out of lift threshold. {str_range}',
                        file=sys.stderr,
                    )
                units_excluded.append(unit)
            elif moe_threshold is not None and (
                (moe < -moe_threshold).any() or (moe_threshold < moe).any()
            ):
                if write_progress:
                    tqdm.write(
                        f'unit {unit} out of moe threshold. {str_range}',
                        file=sys.stderr,
                    )
                units_excluded.append(unit)
            elif write_progress:
                tqdm.write(
                    f'unit {unit} kept. {str_range}',
                    file=sys.stderr,
                )

        df = df.filter(~pl.col(self.unit_column).is_in(units_excluded))
        if len(df) == 0:
            raise StrictFilteringError('all units filterred out. Consider loosing thresholds.')
        intervention_units = [u for u in self.intervention_units if u not in units_excluded]
        if len(intervention_units) == 0:
            raise StrictFilteringError(
                'all intervention units filterred out. Consider loosing thresholds.'
            )

        return Dataset(
            df,
            unit_column=self.unit_column,
            time_column=self.time_column,
            y_column=self.y_column,
            covariate_columns=self.covariate_columns,
            intervention_units=intervention_units,
            intervention_time=self.intervention_time,
            validation_time=self.validation_time,
        )

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
