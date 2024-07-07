"""Function to calculate p value."""

from typing import Union

from scipy import stats


def calc_p_value(
    effects_test: Union[float, list[float]],
    effects_control: Union[float, list[float]],
    equal_var: bool = True,
) -> float:
    """Calculate the p-value using the independent t-test.

    This function performs an independent t-test to compare the means of two groups
    and returns the corresponding p-value. This is two sample T Test.

    Args:
        effects_test (Union[float, list[float]]): The effect sizes of the test group.
            Can be a single float value or a list of float values.
        effects_control (Union[float, list[float]]): The effect sizes of the control group.
            Can be a single float value or a list of float values.
        equal_var (bool):
            If True, perform a standard independent 2 sample test that assumes equal variances.
            If False, perform Welch's t-test, which does not assume equal variance.

    Returns:
        float: The calculated p-value.

    Example:
        >>> effects_test = 1.2
        >>> effects_control = [0.9, 1.1, 1.3]
        >>> calc_p_value(effects_test, effects_control)
        ...
    """
    _, p_value = stats.ttest_ind(effects_test, effects_control, equal_var=equal_var)
    return p_value
