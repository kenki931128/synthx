"""Function for data sampling."""

from typing import Optional, Union

import numpy as np
import polars as pl


def sample(
    *,
    n_units: int,
    n_time: int,
    n_observed_covariates: int,
    n_unobserved_covariates: int,
    intervention_units: Union[int, list[int]],
    intervention_time: int,
    intervention_effect: int = 1,
    noise_effect: float = 1,
    scale: float = 1,
    seed: Optional[int] = None,
) -> pl.DataFrame:
    """Generates a synthetic dataset for causal inference tasks.

    Args:
        n_units (int): The number of units such as targeted regions for test and control.
        n_time (int): The number of timestamps. Weekly data in a year = 52.
        n_observed_covariates (int): The number of covariates such as the population in the region.
        n_unobserved_covariates (int): The number of covariates such as the history in the region.
        intervention_units (int | list[int]): A list of intervented units. Each less than n_units.
        intervention_time (int): When the intervention or event happens. less than n_time.
        intervention_effect (int): effect of the intervention. 1 (100%) means no effect.
        noise_effect (float): effect of the noise.
        scale (float): std of the distribution. Must be non-negative.
        seed (Optional[int]): for ramdom.

    Returns:
        pl.DataFrame: A polars DataFrame containing the generated dataset with columns
        for unit, time, outcome (y) and observed covariates.

    Example:
        >>> df = sample(n_units=100, n_time=10, n_observed_covariates=3, n_unobserved_covariates=1,
        ...             intervention_units=[1], intervention_time=5, intervention_effect=1, seed=42)
        >>> df.head()
        shape: (5, 6)
        ┌──────┬──────┬─────────┬────────────┬────────────┬────────────┐
        │ unit ┆ time ┆ y       ┆ covariate_1┆ covariate_2┆ covariate_3┆
        │ ---  ┆ ---  ┆ ---     ┆ ---        ┆ ---        ┆ ---        │
        │ i64  ┆ i64  ┆ f64     ┆ f64        ┆ f64        ┆ f64        │
        ╞══════╪══════╪═════════╪════════════╪════════════╪════════════╡
        │ 1    ┆ 1    ┆ 2.340096┆ 0.950088   ┆ 0.134298   ┆ 0.794324   │
        ├╌╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 1    ┆ 2    ┆ 2.370135┆ 0.950088   ┆ 0.134298   ┆ 0.794324   │
        ├╌╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 1    ┆ 3    ┆ 2.776434┆ 0.950088   ┆ 0.134298   ┆ 0.794324   │
        ├╌╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 1    ┆ 4    ┆ 3.140631┆ 0.950088   ┆ 0.134298   ┆ 0.794324   │
        ├╌╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 1    ┆ 5    ┆ 3.410797┆ 0.950088   ┆ 0.134298   ┆ 0.794324   │
        └──────┴──────┴─────────┴────────────┴────────────┴────────────┘
    """
    np.random.seed(seed)
    if isinstance(intervention_units, int):
        intervention_units = [intervention_units]

    # Error Handling
    if n_units <= 0:
        raise ValueError('n_units must be a positive integer.')
    if n_time <= 0:
        raise ValueError('n_time must be a positive integer.')
    if n_observed_covariates < 0:
        raise ValueError('n_observed_covariates must be a non-negative integer.')
    if n_unobserved_covariates < 0:
        raise ValueError('n_unobserved_covariates must be a non-negative integer.')
    if not all(1 <= unit <= n_units for unit in intervention_units):
        raise ValueError('all elements in intervention_units must be between 1 and n_units.')
    if not 1 <= intervention_time <= n_time:
        raise ValueError('intervention_time must be between 1 and n_time.')
    if scale <= 0:
        raise ValueError('scale must be positive.')

    # base value of units and time
    # actually units_base is not required here as it's also included in covariates.
    units_base = np.random.normal(0, 1, n_units)
    time_base = np.random.normal(0, 1, n_time)

    # coefficients of covariates
    observed_covariate_coefficients = (
        4 * np.random.rand(n_time, n_observed_covariates) / n_observed_covariates
    )
    unobserved_covariate_coefficients = (
        4 * np.random.rand(n_time, n_unobserved_covariates) / n_unobserved_covariates
    )

    # covariates data
    observed_covariates = np.random.rand(n_units, n_observed_covariates)
    unobserved_covariates = np.random.rand(n_units, n_unobserved_covariates)

    # shift the values to secure the value >= 0
    units_base = units_base - units_base.min()
    units_base /= units_base.max()
    time_base = time_base - time_base.min()
    time_base /= time_base.max()

    # time base added
    time_base += np.linspace(0, 1, n_time)

    # noise
    noise = np.random.normal(0, noise_effect, (n_units, n_time))
    noise -= noise.min()

    # generating data
    data = []
    for i in range(n_units):
        for t in range(n_time):
            # covariate effect of i unit
            observed_covariate_effect = np.dot(
                observed_covariate_coefficients[t], observed_covariates[i]
            )
            unobserved_covariate_effect = np.dot(
                unobserved_covariate_coefficients[t], unobserved_covariates[i]
            )

            # Y_it = δ_t + θ_t*Z_i + λ_t*μ_i + ε_it
            # units_base[i] is the base value. It is not explicitly included in Y_it.
            # time_base[t] corresponds to δ_t.
            # observed_covariate_effect corresponds to θ_t*Z_i.
            # unobserved_covariate_effect corresponds to λ_t*μ_i.
            # noise corresponds to ε_it
            y = (
                units_base[i]
                + time_base[t]
                + observed_covariate_effect
                + unobserved_covariate_effect
                + noise[i][t]
                + 1  # To secure y > 0.
            )

            # apply scale
            y *= scale

            # Add intervention effect if intervented.
            if ((i + 1) in intervention_units) and ((t + 1) >= intervention_time):
                y *= intervention_effect

            # create data point
            point = {'unit': i + 1, 'time': t + 1, 'y': y}
            for c in range(n_observed_covariates):
                point[f'covariate_{c+1}'] = observed_covariates[i][c]
            data.append(point)

    return pl.DataFrame(data)
