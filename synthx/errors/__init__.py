"""Custom Errors"""


class ColumnNotFoundError(Exception):
    """Exception raised when a specified column is not found."""


class InvalidColumnTypeError(Exception):
    """Exception raised when a specified column type mismatches."""


class InvalidInterventionUnitError(Exception):
    """Exception raised when a specified unit does not exist."""


class InvalidInterventionTimeError(Exception):
    """Exception raised when no data at a specified intervention time."""


class NoFeasibleModelError(Exception):
    """Exception raised when synthetic control does not work."""


class InconsistentTimestampsError(Exception):
    """Exception raised when timestamp is not inconcistent across units."""


class StrictFilteringError(Exception):
    """Exception raised when no data exists after filtering."""
