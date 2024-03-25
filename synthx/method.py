"""Synthetic Control Method."""

import numpy as np
import scipy.optimize

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
    dataset: sx.Dataset,
) -> tuple[float, list[float], sx.SyntheticControlResult, list[sx.SyntheticControlResult]]:
    """Perform a placebo test to assess the significance of the intervention effect.

    This function applies the synthetic control method to the test area and each control area
    in the given dataset, estimates the placebo effects, and returns the results.

    Args:
        dataset (sx.Dataset): The dataset containing the time series data for the test and control areas.

    Returns:
        tuple: A tuple containing the following elements:
            - effect_test (float): The estimated effect of the intervention in the test area.
            - effects_placebo (List[float]): The estimated placebo effects for each control area.
            - sc_test (sx.SyntheticControlResult): The synthetic control result for the test area.
            - scs_placebo (List[sx.SyntheticControlResult]): The synthetic control results
                for each control area.
    """
    # placebo effect in test area
    sc_test = synthetic_control(dataset)
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
    for test_unit_placebo in control_units:
        dataset_placebo = sx.Dataset(
            df_placebo,
            unit_column=dataset.unit_column,
            time_column=dataset.time_column,
            y_column=dataset.y_column,
            covariate_columns=dataset.covariate_columns,
            intervention_units=test_unit_placebo,
            intervention_time=dataset.intervention_time,
        )
        sc_placebo = synthetic_control(dataset_placebo)
        effects_placebo.append(sc_placebo.estimate_effects())
        scs_placebo.append(sc_placebo)

    return effect_test, effects_placebo, sc_test, scs_placebo
