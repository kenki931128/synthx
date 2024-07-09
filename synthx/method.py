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

    TODO:
        - Allow different weights for variables based on their importance or relevance.
    """

    df = dataset.data

    # condition
    training_time = (
        dataset.validation_time
        if dataset.validation_time is not None
        else dataset.intervention_time
    )
    condition_training_time = df[dataset.time_column] < training_time
    condition_control_units = ~df[dataset.unit_column].is_in(dataset.intervention_units)

    # weights for variables and scale each variables
    # weights of y_column : weights of others = 8 : 2.
    # TODO: Update if there are better weights balance.
    variables = [dataset.y_column]
    variable_weights: dict[str, float] = {}
    if dataset.covariate_columns is not None:
        variables += dataset.covariate_columns
        variable_weights = {
            covariate: 2 / len(dataset.covariate_columns) for covariate in dataset.covariate_columns
        }
    variable_weights[dataset.y_column] = 8.0
    for variable in variables:
        df = df.with_columns(
            ((pl.col(variable) - pl.col(variable).min()) / pl.col(variable).std()).alias(variable)
        )

    # dataframe for control
    df_control = df.filter(condition_training_time & condition_control_units)

    # pre-create variables for objective function
    arrs_control_pivoted: dict[str, np.ndarray] = {}
    for variable in variables:
        df_control_pivoted = df_control.pivot(
            index=dataset.time_column, columns=dataset.unit_column, values=variable
        ).drop(dataset.time_column)
        arrs_control_pivoted[variable] = df_control_pivoted.to_numpy()

    control_unit_weights: list[np.ndarray] = []
    scales: list[np.ndarray] = []
    for intervention_unit in dataset.intervention_units:
        condition_test_unit = df[dataset.unit_column] == intervention_unit
        df_test = df.filter(condition_training_time & condition_test_unit)
        arrs_test: dict[str, np.ndarray] = {}
        for variable in variables:
            arrs_test[variable] = df_test[variable].to_numpy()

        # optimize unit weights
        def objective(unit_weights: np.ndarray) -> float:
            diff = 0
            for variable in variables:
                arr_control_pivoted = arrs_control_pivoted[variable]
                arr_test = arrs_test[variable]
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
            constraints=[
                {'type': 'ineq', 'fun': lambda w: np.sum(w) - 0.999},
                {'type': 'ineq', 'fun': lambda w: 1.001 - np.sum(w)},
            ],
            method='SLSQP',
            options={'ftol': 1e-4},
        )

        if not solution.success:
            raise NoFeasibleModelError(
                f'synthetic control optimization failed: test unit {intervention_unit}'
            )
        control_unit_weights.append(solution.x)

        # scale
        y_test = df_test[dataset.y_column].to_numpy()
        df_control_pivoted = df_control.pivot(
            index=dataset.time_column, columns=dataset.unit_column, values=dataset.y_column
        ).drop(dataset.time_column)
        arr_control_pivoted = df_control_pivoted.to_numpy()
        y_control = np.sum(arr_control_pivoted * solution.x, axis=1)
        scales.append(np.sum(y_test * y_control) / np.sum(y_control * y_control))

    return sx.SyntheticControlResult(
        dataset=dataset,
        control_unit_weights=np.asarray(control_unit_weights),
        scales=np.asarray(scales),
    )


def placebo_test(
    dataset: sx.Dataset, n_jobs: int = -1
) -> tuple[list[float], list[float], sx.SyntheticControlResult, list[sx.SyntheticControlResult]]:
    """Perform a placebo test to assess the significance of the intervention effect.

    This function applies the synthetic control method to the test area and each control area
    in the given dataset, estimates the placebo effects, and returns the results.

    Args:
        dataset (sx.Dataset): The dataset containing the time series data for the test and control areas.
        n_jobs (int): the number of cores used. -1 means using as many as possible.

    Returns:
        tuple: A tuple containing the following elements:
            - effects_test (list[float]): The estimated effect of the intervention in the test area.
            - effects_placebo (list[float]): The estimated placebo effects for each control area.
            - sc_test (sx.SyntheticControlResult): The synthetic control result for the test area.
            - scs_placebo (list[sx.SyntheticControlResult]): The synthetic control results
                for each control area.
    """
    # placebo effect in test area
    try:
        sc_test = synthetic_control(dataset)
    except NoFeasibleModelError:
        raise NoFeasibleModelError('synthetic control optimization failed for test units.')
    effects_test = sc_test.estimate_effects()

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
    ) -> Optional[tuple[list[float], sx.SyntheticControlResult]]:
        """Apply synthetic control method to a single placebo unit and estimate the effect.

        Args:
            test_unit_placebo (sx.Dataset): The placebo unit to be used as the test unit.

        Returns:
            tuple[list[float], sx.SyntheticControlResult] or None: If the synthetic control optimization
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
            validation_time=dataset.validation_time,
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
            effects_placebo.append(effect_placebo[0])
            scs_placebo.append(sc_placebo)

    return effects_test, effects_placebo, sc_test, scs_placebo


def placebo_sensitivity_check(
    dataset: sx.Dataset,
    effects_placebo: list[float],
    p_value_target: float = 0.03,
    l: float = 1.0,
    r: float = 10.0,
    write_progress: bool = False,
    equal_var: bool = True,
) -> Optional[float]:
    """Perform a sensitivity check on the synthetic control results with placebo.

    Args:
        dataset (sx.Dataset): The dataset for the synthetic control analysis.
        effects_placebo (list[float]): The list of placebo effects estimated.
        p_value_target (float, optional): The target p-value threshold for statistical significance.
        l (float), r (float): the range of uplift. If you have assumption, narrow down to be faster.
        write_progress (bool): Whether to write progress information to stderr.
        equal_var (bool):
            If True, perform a standard independent 2 sample test that assumes equal variances.
            If False, perform Welch's t-test, which does not assume equal variance.

    Returns:
        float or None: The uplift which becomes statistically significant.
    """
    df = dataset.data

    if l < 1.0:
        raise ValueError('l should be larger than or equal to 1.')
    if r <= l:
        raise ValueError('r should be larger than l.')

    progress_bar = tqdm()
    while r - l > 0.001:
        uplift = (l + r) / 2
        progress_bar.update(1)
        progress_bar.set_postfix(uplift=f'{uplift:.4f}')

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
            validation_time=dataset.validation_time,
        )

        try:
            sc = synthetic_control(dataset_sensitivity)
        except NoFeasibleModelError:
            r = uplift  # highly likely uplift was too big. TODO: think better algorithm.
            continue

        p_value = sx.stats.calc_p_value(sc.estimate_effects(), effects_placebo, equal_var=equal_var)
        if p_value <= p_value_target:
            r = uplift
        else:
            if write_progress:
                tqdm.write(f'uplift: {uplift:.4f}, p value: {p_value}.', file=sys.stderr)
            l = uplift

    # even 1000% uplift cannot be captured / singnificant difference without actual uplift.
    if r == 10 or l == 1:
        return None
    return r


def ttest_sensitivity_check(
    dataset: sx.Dataset,
    p_value_target: float = 0.03,
    l: float = 1.0,
    r: float = 10.0,
    write_progress: bool = False,
) -> Optional[float]:
    """Perform a sensitivity check on the synthetic control results with paired ttest.

    Args:
        dataset (sx.Dataset): The dataset for the synthetic control analysis.
        p_value_target (float, optional): The target p-value threshold for statistical significance.
        l (float), r (float): the range of uplift. If you have assumption, narrow down to be faster.
        write_progress (bool): Whether to write progress information to stderr.

    Returns:
        float or None: The uplift which becomes statistically significant.
    """
    df = dataset.data

    if l < 1.0:
        raise ValueError('l should be larger than or equal to 1.')
    if r <= l:
        raise ValueError('r should be larger than l.')

    progress_bar = tqdm()
    while r - l > 0.001:
        uplift = (l + r) / 2
        progress_bar.update(1)
        progress_bar.set_postfix(uplift=f'{uplift:.4f}')

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
            validation_time=dataset.validation_time,
        )

        try:
            sc = synthetic_control(dataset_sensitivity)
        except NoFeasibleModelError:
            r = uplift  # highly likely uplift was too big. TODO: think better algorithm.
            continue

        p_value = np.mean([p['p_value_in_intervention'] for p in sc.paired_ttest()])
        if p_value <= p_value_target:
            r = uplift
        else:
            if write_progress:
                tqdm.write(f'uplift: {uplift:.4f}, p value: {p_value}.', file=sys.stderr)
            l = uplift

    # even 1000% uplift cannot be captured / singnificant difference without actual uplift.
    if r == 10 or l == 1:
        return None
    return r
