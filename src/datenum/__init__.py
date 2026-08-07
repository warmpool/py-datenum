"""py-datenum — MATLAB-style numeric dates for Python.

Convert between datenum (fractional days since epoch) and datetime,
YMD tuples, and formatted strings.  Supports lists and NumPy arrays.

Default epoch is 2000-01-01T00:00:00.  Use ``set_epoch`` or
``temporary_epoch`` to change it.
"""

from .core import (
    day,
    day_of_month,
    day_of_week,
    day_of_year,
    day_of_year_229,
    from_datetime,
    from_string,
    from_ymd,
    get_epoch,
    hour,
    is_leap_year,
    microsecond,
    minute,
    month,
    second,
    set_epoch,
    temporary_epoch,
    to_datetime,
    to_string,
    to_ymd,
    year,
)

__all__ = [
    "day",
    "day_of_month",
    "day_of_week",
    "day_of_year",
    "day_of_year_229",
    "from_datetime",
    "from_string",
    "from_ymd",
    "get_epoch",
    "hour",
    "is_leap_year",
    "microsecond",
    "minute",
    "month",
    "second",
    "set_epoch",
    "temporary_epoch",
    "to_datetime",
    "to_string",
    "to_ymd",
    "year",
]

try:
    from .core import from_datetime_array, to_datetime_array

    __all__ += ["from_datetime_array", "to_datetime_array"]
except ImportError:
    pass
