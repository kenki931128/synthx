"""Custom Errors"""


class ColumnNotFoundError(Exception):
    """Exception raised when a specified column is not found."""


class InvalidColumnTypeError(Exception):
    """Exception raised when a specified column type mismatches."""


class InvalidInterventionUnitError(Exception):
    """Exception raised when a specified unit does not exist."""


class InvalidInterventionTimeError(Exception):
    """Exception raised when no data at a specified intervention time."""
