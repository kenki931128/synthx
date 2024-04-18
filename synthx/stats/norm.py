"""Normalization methods."""

import polars as pl
from scipy import stats


def z_standardize(df: pl.DataFrame, column: str) -> pl.DataFrame:
    """Standardize the values in the column using Z-score normalization

    Args:
        df (pl.DataFrame): A Polars DataFrame.
        column (str): column name which will be standardized.

    Returns:
        pl.DataFrame: The input DataFrame containing the standardized values using Z-score normalization.
    """
    return df.with_columns((pl.col(column) - pl.col(column).mean()) / pl.col(column).std())


def cv_standardize(df: pl.DataFrame, column: str) -> pl.DataFrame:
    """Standardize the values in the column using the coefficient of variation (CV)

    Args:
        df (pl.DataFrame): A Polars DataFrame.
        column (str): column name which will be standardized.

    Returns:
        pl.DataFrame: The input DataFrame containing the standardized values using CV.
    """
    cv = df.select(pl.col(column).std() / pl.col(column).mean()).head(1).row(0)[0]
    return df.with_columns((pl.col(column) - pl.col(column).mean()) / (pl.col(column).mean() * cv))


def yeo_johnson_standardize(df: pl.DataFrame, column: str) -> pl.DataFrame:
    """Standardize the values in the column using Yeo-Johnson transformation and Z-score normalization.

    Args:
        df (pl.DataFrame): A Polars DataFrame.
        column (str): column name which will be standardized.

    Returns:
        pl.DataFrame: The input DataFrame containing the standardized values
        using Yeo-Johnson transformation and Z-score normalization.
    """
    _, lambda_ = stats.yeojohnson(df[column].to_numpy())
    df = df.with_columns(
        pl.Series(stats.yeojohnson(df[column].to_numpy(), lmbda=lambda_)).alias(column)
    )
    return z_standardize(df, column)
