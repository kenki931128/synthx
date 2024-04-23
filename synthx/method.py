"""Synthetic Control Method."""

import sys
from typing import Optional

import numpy as np
import polars as pl
import scipy.optimize
from joblib import Parallel, delayed
from tqdm import tqdm

import synthx as sx
from synthx.errors import NoFeasibleModelError


def synthetic_control(dataset: sx.Dataset) -> sx.SyntheticControlResult:
    """Perform synthetic control analysis on the given dataset.

    This function applies the synthetic control method to estimate the effect of an intervention
    on a target variable. It optimizes the weights of control units to create a synthetic control
    unit that closely resembles the test unit in terms of pre-intervention characteristics.

    Args:
        dataset (sx.Dataset): The dataset containing the time series for the test and control units.

    Returns:
        sx.SyntheticControlResult: The result of synthetic control with optimized weights.

    Raises:
        NotImplementedError: If there are multiple units that received the intervention.
        NoFeasibleModelError: If the optimization of unit weights fails to find a feasible solution.

    Note:
        - The function currently supports only a single unit that received the intervention.
        - The weights for variables (y and covariates) are currently set to 1 for all variables.

    TODO:
        - Add validation period for evaluating the performance of the synthetic control.
        - Allow different weights for variables based on their importance or relevance.
        - Support multiple units that received the intervention. (also need to update result class)
    """

    df = dataset.data

    # condition
    # TODO: add validation period
    condition_pre_intervention_time = df[dataset.time_column] < dataset.intervention_time
    condition_test_units = df[dataset.unit_column].is_in(dataset.intervention_units)
    condition_control_units = ~df[dataset.unit_column].is_in(dataset.intervention_units)

    # weights for variables
    # TODO: use different weights for variables
    variables = [dataset.y_column]
    if dataset.covariate_columns is not None:
        variables += dataset.covariate_columns
    variable_weights = {variable: 1 for variable in variables}

    # TODO: multiple units intervention
    if len(dataset.intervention_units) > 1:
        raise NotImplementedError('multiple intervented units.')

    # dataframe for test & control
    df_test = df.filter(condition_pre_intervention_time & condition_test_units)
    df_control = df.filter(condition_pre_intervention_time & condition_control_units)

    # optimize unit weights
    def objective(unit_weights: np.ndarray) -> float:
        diff = 0
        for variable in variables:
            df_control_pivoted = df_control.pivot(
                index=dataset.time_column, columns=dataset.unit_column, values=variable
            ).drop(dataset.time_column)
            arr_control_pivoted = df_control_pivoted.to_numpy()
            arr_test = df_test[variable].to_numpy()
            variable_weight = variable_weights[variable]
            diff += np.sum(
                variable_weight
                * (arr_test - np.sum(arr_control_pivoted * unit_weights, axis=1)) ** 2
            )
        return diff

    control_units = df_control[dataset.unit_column].unique().to_list()
    initial_unit_weights = np.ones(len(control_units)) / len(control_units)
    bounds = [(0, 1)] * len(control_units)
    # The optimization of unit weights is performed using the 'minimize' function from scipy.
    solution = scipy.optimize.minimize(
        objective,
        initial_unit_weights,
        bounds=bounds,
        constraints={'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
    )

    if not solution.success:
        raise NoFeasibleModelError('synthetic control optimization failed.')

    return sx.SyntheticControlResult(dataset=dataset, control_unit_weights=solution.x)


def placebo_test(
    dataset: sx.Dataset, n_jobs: int = -1
) -> tuple[float, list[float], sx.SyntheticControlResult, list[sx.SyntheticControlResult]]:
    """Perform a placebo test to assess the significance of the intervention effect.

    This function applies the synthetic control method to the test area and each control area
    in the given dataset, estimates the placebo effects, and returns the results.

    Args:
        dataset (sx.Dataset): The dataset containing the time series data for the test and control areas.
        n_jobs (int): the number of cores used. -1 means using as many as possible.

    Returns:
        tuple: A tuple containing the following elements:
            - effect_test (float): The estimated effect of the intervention in the test area.
            - effects_placebo (List[float]): The estimated placebo effects for each control area.
            - sc_test (sx.SyntheticControlResult): The synthetic control result for the test area.
            - scs_placebo (List[sx.SyntheticControlResult]): The synthetic control results
                for each control area.
    """
    # placebo effect in test area
    try:
        sc_test = synthetic_control(dataset)
    except NoFeasibleModelError:
        raise NoFeasibleModelError('synthetic control optimization failed for test units.')
    effect_test = sc_test.estimate_effects()

    # placebo effects in control areas
    effects_placebo: list[float] = []
    scs_placebo: list[sx.SyntheticControlResult] = []
    control_units = (
        dataset.data.filter(~dataset.data[dataset.unit_column].is_in(dataset.intervention_units))[
            dataset.unit_column
        ]
        .unique()
        .to_list()
    )
    df_placebo = dataset.data.filter(dataset.data[dataset.unit_column].is_in(control_units))

    def process_placebo(
        test_unit_placebo: sx.Dataset,
    ) -> Optional[tuple[float, sx.SyntheticControlResult]]:
        """Apply synthetic control method to a single placebo unit and estimate the effect.

        Args:
            test_unit_placebo (sx.Dataset): The placebo unit to be used as the test unit.

        Returns:
            tuple[float, sx.SyntheticControlResult] or None: If the synthetic control optimization
                is successful, returns a tuple containing the following elements:
                - effect_placebo (float): The estimated placebo effect for the given control unit.
                - sc_placebo (sx.SyntheticControlResult): The synthetic control result for placebo.
                If the synthetic control optimization fails, returns None.

        Side Effects:
            Writes an error message to stderr using tqdm.write() if the synthetic control
            optimization fails for the given placebo unit.

        Note:
            This function is intended to be used as a helper function within the placebo_test()
            function for parallel processing of placebo tests. It assumes that the following
            variables are defined in the outer scope:
            - df_placebo (DataFrame): The subset dataset containing only the control units.
            - dataset (sx.Dataset): The original dataset object.
        """
        dataset_placebo = sx.Dataset(
            df_placebo,
            unit_column=dataset.unit_column,
            time_column=dataset.time_column,
            y_column=dataset.y_column,
            covariate_columns=dataset.covariate_columns,
            intervention_units=test_unit_placebo,
            intervention_time=dataset.intervention_time,
        )
        try:
            sc_placebo = synthetic_control(dataset_placebo)
            effect_placebo = sc_placebo.estimate_effects()
            return effect_placebo, sc_placebo
        except NoFeasibleModelError:
            tqdm.write(
                f'placebo synthetic control optimization failed: unit {test_unit_placebo}.',
                file=sys.stderr,
            )
            return None

    results = Parallel(n_jobs=n_jobs)(
        delayed(process_placebo)(test_unit_placebo) for test_unit_placebo in tqdm(control_units)
    )

    effects_placebo = []
    scs_placebo = []
    for result in results:
        if result is not None:
            effect_placebo, sc_placebo = result
            effects_placebo.append(effect_placebo)
            scs_placebo.append(sc_placebo)

    return effect_test, effects_placebo, sc_test, scs_placebo


def sensitivity_check(
    dataset: sx.Dataset, effects_placebo: list[float], p_value_target: float = 0.03
) -> Optional[float]:
    """Perform a sensitivity check on the synthetic control results.

    Args:
        dataset (sx.Dataset): The dataset for the synthetic control analysis.
        effects_placebo (list[float]): The list of placebo effects estimated.
        p_value_target (float, optional): The target p-value threshold for statistical significance.

    Returns:
        float or None: The uplift which becomes statistically significant.
    """
    df = dataset.data

    l, r = 1.0, 10.0
    while r - l > 0.001:
        uplift = (l + r) / 2

        df_sensitivity = df.with_columns(
            pl.when(
                pl.col(dataset.unit_column).is_in(dataset.intervention_units)
                & (pl.col(dataset.time_column) >= dataset.intervention_time)
            )
            .then(pl.col(dataset.y_column) * uplift)
            .otherwise(pl.col(dataset.y_column))
            .alias(dataset.y_column)
        )

        dataset_sensitivity = sx.Dataset(
            df_sensitivity,
            unit_column=dataset.unit_column,
            time_column=dataset.time_column,
            y_column=dataset.y_column,
            covariate_columns=dataset.covariate_columns,
            intervention_units=dataset.intervention_units,
            intervention_time=dataset.intervention_time,
        )

        try:
            sc = synthetic_control(dataset_sensitivity)
        except NoFeasibleModelError:
            r = uplift  # highly likely uplift was too big. TODO: think better algorithm.
            continue

        p_value = sx.stats.calc_p_value(sc.estimate_effects(), effects_placebo)
        if p_value <= p_value_target:
            r = uplift
        else:
            tqdm.write(f'uplift: {uplift:.4f}, p value: {p_value}.', file=sys.stderr)
            l = uplift

    # even 1000% uplift cannot be captured.
    if r == 10:
        return None
    # singnificant difference without actual uplift
    if l == 1:
        return None
    return r
