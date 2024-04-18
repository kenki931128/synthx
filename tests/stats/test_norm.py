"""Test for synthx.stats.norm."""

import polars as pl

from synthx.stats import norm


class TestNorm:
    def test_z_standardize(self) -> None:
        # Test data
        df = pl.DataFrame({'x': [1, 2, 3, 4, 5]})

        # Expected output
        expected_output = pl.DataFrame(
            {
                'x': [
                    -1.2649110640673518,
                    -0.6324555320336759,
                    0.0,
                    0.6324555320336759,
                    1.2649110640673518,
                ]
            }
        )

        # Perform z-standardization
        result = norm.z_standardize(df, 'x')

        # Assert the result matches the expected output
        assert result.equals(expected_output)

    def test_cv_standardize(self) -> None:
        # Test data
        df = pl.DataFrame({'x': [1, 2, 3, 4, 5]})

        # Expected output
        expected_output = pl.DataFrame(
            {
                'x': [
                    -1.2649110640673518,
                    -0.6324555320336759,
                    0.0,
                    0.6324555320336759,
                    1.2649110640673518,
                ]
            }
        )

        # Perform CV standardization
        result = norm.cv_standardize(df, 'x')

        # Assert the result matches the expected output
        assert result.equals(expected_output)

    def test_yeo_johnson_standardize_negative_values(self) -> None:
        # Test data with negative values
        df = pl.DataFrame({'x': [-2, -1, 0, 1, 2]})

        # Perform Yeo-Johnson standardization
        result = norm.yeo_johnson_standardize(df, 'x')

        # Assert the result has no NaN or infinite values
        assert not result['x'].is_nan().any()
        assert not result['x'].is_infinite().any()
