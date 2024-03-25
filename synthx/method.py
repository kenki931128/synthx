"""Synthetic Control Method."""

import numpy as np
from scipy.optimize import minimize

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
    solution = minimize(
        objective,
        initial_unit_weights,
        bounds=bounds,
        constraints={'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
    )

    if not solution.success:
        raise NoFeasibleModelError('synthetic control optimization failed.')

    return sx.SyntheticControlResult(dataset=dataset, control_unit_weights=solution.x)
