"""Core datenum conversion functions.

A *datenum* is a ``float`` counting fractional days since the module-wide
epoch (default ``2000-01-01T00:00:00``, timezone-naive, at midnight).
Values after the epoch are positive; values before it are negative.

Scalar helpers (:func:`from_datetime`, :func:`to_datetime`,
:func:`from_ymd`, :func:`to_ymd`, the component accessors and calendar
helpers) operate on single values and return Python builtins. Their
behaviour extends to arrays through the ``*_array`` pair and through the
vectorized paths of :func:`from_ymd`, :func:`to_ymd`, :func:`add_month`
and the accessors, which accept lists and NumPy arrays and return
``float64`` / ``int64`` / ``datetime64[ns]`` arrays.
"""

from __future__ import annotations

import calendar
import datetime as _dt
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, overload

from dateutil.parser import parse

try:
    import numpy as np  # ty: ignore

    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray  # ty: ignore

    DnScalar = int | float | np.integer | np.floating

_NS_PER_DAY: int = 86_400_000_000_000

# ---------------------------------------------------------------------------
# Epoch
# ---------------------------------------------------------------------------
_DEFAULT_EPOCH: _dt.datetime = _dt.datetime(2000, 1, 1)
_epoch: _dt.datetime = _DEFAULT_EPOCH


def _validate_epoch(epoch: _dt.datetime) -> None:
    """Reject epochs that would break the naive, midnight, timezone-less contract."""
    if epoch.tzinfo is not None:
        raise ValueError(f"epoch must be timezone-naive, got timezone {epoch.tzinfo!r}")
    if epoch.time() != _dt.time.min:
        raise ValueError(
            f"epoch must be at midnight (00:00:00), got {epoch.time().isoformat()}"
        )


def set_epoch(epoch: _dt.datetime) -> None:
    """Set the module-wide epoch used by all conversions.

    The change takes effect immediately and applies to every subsequent
    conversion in the process. Previously produced datenum values are *not*
    re-based, so mixing values from different epochs silently misinterprets
    them — set the epoch once at program start if you need a non-default.

    Args:
        epoch: The new reference point. Must be timezone-naive and fall
            exactly at midnight (``00:00:00.000000``).

    Raises:
        ValueError: If *epoch* is timezone-aware or not at midnight. The
            previously configured epoch is left unchanged.

    Example:
        >>> import datetime as dt
        >>> datenum.set_epoch(dt.datetime(1970, 1, 1))
        >>> datenum.get_epoch()
        datetime.datetime(1970, 1, 1)
    """
    _validate_epoch(epoch)
    global _epoch
    _epoch = epoch


def get_epoch() -> _dt.datetime:
    """Return the current module-wide epoch.

    Returns:
        The naive midnight :class:`datetime.datetime` that all conversions
        currently count fractional days from. Defaults to
        ``2000-01-01T00:00:00`` unless changed by :func:`set_epoch`.

    Example:
        >>> datenum.get_epoch()
        datetime.datetime(2000, 1, 1)
    """
    return _epoch


@contextmanager
def temporary_epoch(epoch: _dt.datetime) -> Generator[None, None, None]:
    """Temporarily set the epoch for the duration of a ``with`` block.

    The previous epoch is restored when the block exits, even if the body
    raises. Useful for converting legacy data recorded against a different
    reference point without mutating process-wide state.

    Args:
        epoch: The epoch to use inside the block; same validity rules as
            :func:`set_epoch` (naive, at midnight).

    Yields:
        None.

    Raises:
        ValueError: If *epoch* is invalid. The current epoch is left
            unchanged and the block body is never executed.

    Example:
        >>> import datetime as dt
        >>> with datenum.temporary_epoch(dt.datetime(1900, 1, 1)):
        ...     datenum.from_datetime(dt.datetime(1900, 1, 2))
        1.0
    """
    old = get_epoch()
    set_epoch(epoch)
    try:
        yield
    finally:
        set_epoch(old)


def _is_scalar_like(value: Any) -> bool:
    """Check if *value* should be treated as a single date."""
    if isinstance(value, _dt.datetime):
        return True
    if not _HAS_NUMPY:
        return isinstance(value, (int, float))
    if isinstance(value, np.ndarray):
        return value.ndim == 0
    return isinstance(value, (int, float, np.integer, np.floating))


# ---------------------------------------------------------------------------
# datetime <-> datenum
# ---------------------------------------------------------------------------
def from_datetime(dt: _dt.datetime) -> float:
    """Convert a naive :class:`datetime.datetime` to a datenum.

    The result is fractional days since the current epoch: a whole number
    at midnight, ``0.5`` at noon, negative before the epoch.

    Args:
        dt: A timezone-naive datetime. For lists or NumPy arrays of
            datetimes, use :func:`from_datetime_array`.

    Returns:
        Fractional days between *dt* and the epoch.

    Example:
        >>> import datetime as dt
        >>> datenum.from_datetime(dt.datetime(2000, 1, 1, 12))
        0.5
        >>> datenum.from_datetime(dt.datetime(1999, 12, 31))
        -1.0
    """
    delta = dt - _epoch
    days = delta.days + delta.seconds / 86400 + delta.microseconds / 86_400_000_000
    return days


def to_datetime(dn: DnScalar) -> _dt.datetime:
    """Convert a datenum to a naive :class:`datetime.datetime`.

    The inverse of :func:`from_datetime`. The day fraction is converted to
    whole microseconds (rounded to nearest), so a ``datetime -> datenum ->
    datetime`` roundtrip is exact at microsecond resolution for dates near
    the epoch.

    Args:
        dn: Fractional days since the epoch; may be negative. For arrays,
            use :func:`to_datetime_array`.

    Returns:
        The equivalent timezone-naive ``datetime`` with microsecond
        precision.

    Example:
        >>> datenum.to_datetime(0.5)
        datetime.datetime(2000, 1, 1, 12, 0)
        >>> datenum.to_datetime(-1.0)
        datetime.datetime(1999, 12, 31, 0, 0)
    """
    days_int = int(dn)
    remainder = dn - days_int
    total_us = round(remainder * 86_400_000_000)
    seconds, microseconds = divmod(total_us, 1_000_000)
    return _epoch + _dt.timedelta(
        days=days_int, seconds=int(seconds), microseconds=int(microseconds)
    )


@overload
def to_datetime_array(dn: DnScalar) -> _dt.datetime: ...
@overload
def to_datetime_array(dn: ArrayLike) -> NDArray[np.datetime64]: ...
def to_datetime_array(dn):
    """Convert datenum value(s) to datetime(s), scalar or array.

    Accepts both scalar and array input. A single number returns a Python
    :class:`datetime.datetime`; anything list-like or array-like is
    converted through vectorized nanosecond arithmetic and returned as a
    ``numpy.ndarray`` of dtype ``datetime64[ns]``.

    Args:
        dn: One number, or a list/tuple/:class:`numpy.ndarray` of numbers,
            each representing fractional days since the epoch. May be
            negative.

    Returns:
        A :class:`datetime.datetime` for scalar input; otherwise a
        ``datetime64[ns]`` ``ndarray`` with the input's shape.

    Raises:
        ImportError: If array input is given but NumPy is not installed.

    Example:
        >>> import numpy as np
        >>> datenum.to_datetime_array(np.array([0.0, 0.5]))
        array(['2000-01-01T00:00:00', '2000-01-01T12:00:00'],
              dtype='datetime64[ns]')
        >>> datenum.to_datetime_array(-1.0)
        datetime.datetime(1999, 12, 31, 0, 0)
    """
    if _is_scalar_like(dn):
        return to_datetime(dn)
    if not _HAS_NUMPY:
        raise ImportError(
            "numpy is required for array inputs (pip install datenum[numpy])"
        )
    arr = np.asarray(dn, dtype="float64")
    days_int = np.floor(arr).astype("int64")
    remainder = arr - days_int.astype("float64")
    total_us = np.round(remainder * 86_400_000_000).astype("int64")
    epoch64 = np.datetime64(_epoch, "ns")
    return (
        epoch64 + days_int.astype("timedelta64[D]") + total_us.astype("timedelta64[us]")
    )


@overload
def from_datetime_array(dt: _dt.datetime) -> float: ...
@overload
def from_datetime_array(
    dt: Sequence[_dt.datetime] | np.datetime64 | NDArray[np.datetime64],
) -> NDArray[np.float64]: ...
def from_datetime_array(dt):
    """Convert datetime value(s) to datenum value(s), scalar or array.

    Accepts both scalar and array input. A single datetime returns a Python
    ``float``; a list or tuple of datetimes, or a NumPy ``datetime64``
    array, is converted in bulk via nanosecond arithmetic into a
    ``float64`` array.

    Args:
        dt: A timezone-naive :class:`datetime.datetime`, a sequence of
            them, or a ``datetime64`` array.

    Returns:
        A ``float`` for scalar input; otherwise a ``float64`` ``ndarray``
        of fractional days since the epoch.

    Raises:
        TypeError: On timezone-aware input (cannot be mixed with the naive
            epoch).
        ImportError: If array input is given but NumPy is not installed.

    Example:
        >>> import numpy as np
        >>> dts = np.array(["2000-01-01", "2000-01-02"], dtype="datetime64[ns]")
        >>> datenum.from_datetime_array(dts)
        array([0., 1.])
    """
    if _is_scalar_like(dt):
        return from_datetime(dt)
    if not _HAS_NUMPY:
        raise ImportError(
            "numpy is required for array inputs (pip install datenum[numpy])"
        )
    arr = np.asarray(dt, dtype="datetime64[ns]")
    return (arr - np.datetime64(_epoch, "ns")).astype("timedelta64[ns]").astype(
        "float64"
    ) / _NS_PER_DAY


# ---------------------------------------------------------------------------
# NumPy helpers (vectorised internals)
# ---------------------------------------------------------------------------

_NS_HOUR: int = 3_600_000_000_000
_NS_MINUTE: int = 60_000_000_000
_NS_SECOND: int = 1_000_000_000


def _civil_to_days(
    year: ArrayLike, month: ArrayLike, day: ArrayLike
) -> NDArray[np.int64]:
    """Convert year/month/day arrays to days since 1970-01-01.

    Uses Howard Hinnant's algorithm (public domain).  All inputs are
    broadcast together; return dtype is ``int64``.
    """
    y = np.asarray(year, dtype="int64")
    m = np.asarray(month, dtype="int64")
    d = np.asarray(day, dtype="int64")
    m_adj = np.where(m <= 2, m + 12, m)
    y_adj = np.where(m <= 2, y - 1, y)
    era = y_adj // 400
    yoe = y_adj - era * 400
    doy = (153 * (m_adj - 3) + 2) // 5 + d - 1
    doe = yoe * 365 + yoe // 4 - yoe // 100 + doy
    return era * 146097 + doe - 719468


def _time_of_day_ns(dt64: NDArray[np.datetime64]) -> NDArray[np.int64]:
    """Extract time-of-day in nanoseconds from a datetime64[ns] array."""
    day_start = dt64.astype("datetime64[D]")
    return (dt64 - day_start).astype("timedelta64[ns]").astype("int64")


# ---------------------------------------------------------------------------
# YMD <-> datenum
# ---------------------------------------------------------------------------


def _to_int(v: DnScalar) -> int:
    if isinstance(v, float) or (_HAS_NUMPY and isinstance(v, np.floating)):
        if not v.is_integer():
            raise ValueError(f"expected a whole number, got {v}")
        return int(v)
    return int(v)


@overload
def from_ymd(
    year: DnScalar,
    month: DnScalar,
    day: DnScalar,
    hour: DnScalar = 0,
    minute: DnScalar = 0,
    second: DnScalar = 0,
    microsecond: DnScalar = 0,
) -> float: ...
@overload
def from_ymd(
    year: ArrayLike,
    month: ArrayLike,
    day: ArrayLike,
    hour: ArrayLike = 0,
    minute: ArrayLike = 0,
    second: ArrayLike = 0,
    microsecond: ArrayLike = 0,
) -> NDArray[np.float64]: ...
def from_ymd(
    year: Any,
    month: Any,
    day: Any,
    hour: Any = 0,
    minute: Any = 0,
    second: Any = 0,
    microsecond: Any = 0,
):
    """Convert calendar components to a datenum, scalar or array.

    Builds ``year-month-day hour:minute:second.microsecond`` and measures it
    against the current epoch. Uses the proleptic Gregorian calendar; there
    is no time zone and no DST handling.

    Whole-number floats are accepted for every component (``2024.0`` means
    2024), which keeps values loaded from text or CSV convenient to pass
    through. Non-whole floats such as ``2024.5`` are rejected rather than
    silently truncated.

    Args:
        year: Calendar year, e.g. ``2024``.
        month: Month of year, 1–12.
        day: Day of month, 1–31; must exist in the given month/year.
        hour: Hour of day, 0–23. Defaults to 0.
        minute: Minute, 0–59. Defaults to 0.
        second: Second, 0–59. Defaults to 0.
        microsecond: Microsecond, 0–999999. Defaults to 0.

    Returns:
        A ``float`` when all arguments are scalars; otherwise a ``float64``
        ``ndarray`` of fractional days since the epoch, with the arguments
        broadcast together.

    Raises:
        ValueError: If a scalar argument is a non-whole float, or a scalar
            component is outside its valid range (enforced by
            :class:`datetime.datetime`).
        ImportError: If any argument is an array but NumPy is not installed.

    Notes:
        Array inputs are converted with a vectorized civil-calendar
        algorithm and are *not* range-checked; invalid components (month
        13, day 32) produce undefined results rather than an error.

    Example:
        >>> import datetime as dt
        >>> datenum.from_ymd(2024, 7, 15) == \\
        ...     datenum.from_datetime(dt.datetime(2024, 7, 15))
        True
        >>> datenum.to_datetime(datenum.from_ymd(2024, 3, 15, 10, 30, 45))
        datetime.datetime(2024, 3, 15, 10, 30, 45)
    """
    if (
        _is_scalar_like(year)
        and _is_scalar_like(month)
        and _is_scalar_like(day)
        and _is_scalar_like(hour)
        and _is_scalar_like(minute)
        and _is_scalar_like(second)
        and _is_scalar_like(microsecond)
    ):
        return from_datetime(
            _dt.datetime(
                _to_int(year),
                _to_int(month),
                _to_int(day),
                _to_int(hour),
                _to_int(minute),
                _to_int(second),
                _to_int(microsecond),
            )
        )
    if not _HAS_NUMPY:
        raise ImportError(
            "numpy is required for array inputs (pip install datenum[numpy])"
        )
    days = _civil_to_days(year, month, day)
    dt64 = (
        days.astype("datetime64[D]")
        + np.asarray(hour, dtype="int64").astype("timedelta64[h]")
        + np.asarray(minute, dtype="int64").astype("timedelta64[m]")
        + np.asarray(second, dtype="int64").astype("timedelta64[s]")
        + np.asarray(microsecond, dtype="int64").astype("timedelta64[us]")
    ).astype("datetime64[ns]")
    return (dt64 - np.datetime64(_epoch, "ns")).astype("timedelta64[ns]").astype(
        "float64"
    ) / _NS_PER_DAY


@overload
def to_ymd(dn: DnScalar) -> tuple[int, int, int, int, int, int, int]: ...
@overload
def to_ymd(dn: ArrayLike) -> tuple[NDArray[np.int64], ...]: ...
def to_ymd(dn: Any):
    """Convert a datenum to ``(year, month, day, hour, minute, second,
    microsecond)``.

    The inverse of :func:`from_ymd`. Time components are derived from the
    day fraction after microsecond rounding, consistent with
    :func:`to_datetime`.

    Args:
        dn: Fractional days since the epoch — a single number, or an
            array/list of numbers. May be negative.

    Returns:
        For scalar input, a tuple of seven ``int``. For array input, a
        tuple of seven ``int64`` ``ndarray`` objects of the same shape.

    Example:
        >>> datenum.to_ymd(datenum.from_ymd(2024, 7, 15, 13, 45, 30))
        (2024, 7, 15, 13, 45, 30, 0)
    """
    if _is_scalar_like(dn):
        dt = to_datetime(dn)
        return (
            dt.year,
            dt.month,
            dt.day,
            dt.hour,
            dt.minute,
            dt.second,
            dt.microsecond,
        )
    if not _HAS_NUMPY:
        raise ImportError(
            "numpy is required for array inputs (pip install datenum[numpy])"
        )
    dt64: NDArray[np.datetime64] = to_datetime_array(dn)  # type: ignore[assignment]
    years = dt64.astype("datetime64[Y]").astype("int64") + 1970  # ty: ignore
    months = (dt64.astype("datetime64[M]") - dt64.astype("datetime64[Y]")).astype(  # ty: ignore
        "int64"
    ) + 1
    days = (dt64 - dt64.astype("datetime64[M]")).astype("timedelta64[D]").astype(  # ty: ignore
        "int64"
    ) + 1
    tod = _time_of_day_ns(dt64)
    hours = tod // _NS_HOUR
    minutes = (tod % _NS_HOUR) // _NS_MINUTE
    seconds = (tod % _NS_MINUTE) // _NS_SECOND
    microseconds = (tod % _NS_SECOND) // 1_000
    return (years, months, days, hours, minutes, seconds, microseconds)


# ---------------------------------------------------------------------------
# Month arithmetic
# ---------------------------------------------------------------------------


@overload
def add_month(dn: DnScalar, delta: int = 1) -> float: ...
@overload
def add_month(dn: DnScalar, delta: ArrayLike) -> NDArray[np.float64]: ...
@overload
def add_month(dn: ArrayLike, delta: int = 1) -> NDArray[np.float64]: ...
@overload
def add_month(dn: ArrayLike, delta: ArrayLike) -> NDArray[np.float64]: ...
def add_month(dn: Any, delta: Any = 1) -> Any:
    """Add *delta* calendar months to datenum *dn*, clamping the day.

    Moves the date forward or backward by whole months while preserving the
    time of day. When the target month has fewer days than the source day,
    the result clamps to the last valid day: January 31 plus one month is
    February 28, or February 29 in a leap year.

    Works element-wise: *dn* and *delta* may each be a scalar or an array
    and are broadcast against each other; two arrays must share the same
    shape.

    Args:
        dn: Fractional days since the epoch — a scalar or an array.
        delta: Number of months to add (negative to go back); a scalar
            integer or an integer array. Defaults to 1.

    Returns:
        A ``float`` when both operands are scalars; otherwise a ``float64``
        ``ndarray`` shaped like the broadcast result.

    Raises:
        ValueError: If *dn* and *delta* are both arrays with different
            shapes.
        ImportError: If array input is given but NumPy is not installed.

    Example:
        >>> dn = datenum.from_ymd(2024, 1, 31)
        >>> datenum.to_ymd(datenum.add_month(dn, 1))[:3]      # clamped
        (2024, 2, 29)
        >>> datenum.to_ymd(datenum.add_month(datenum.from_ymd(2025, 1, 31), 1))[:3]
        (2025, 2, 28)
        >>> datenum.to_ymd(datenum.add_month(dn, -2))[:3]
        (2023, 11, 30)
    """
    if _is_scalar_like(dn) and _is_scalar_like(delta):
        dt = to_datetime(dn)
        total_months = (dt.year * 12 + dt.month - 1) + int(delta)
        new_year = total_months // 12
        new_month = total_months % 12 + 1
        max_day = calendar.monthrange(new_year, new_month)[1]
        new_day = min(dt.day, max_day)
        return from_datetime(dt.replace(year=new_year, month=new_month, day=new_day))

    if not _HAS_NUMPY:
        raise ImportError(
            "numpy is required for array inputs (pip install datenum[numpy])"
        )

    if _is_scalar_like(dn):
        delta_arr = np.asarray(delta, dtype="int64")
        dn_arr = np.full(delta_arr.shape, float(dn))
    elif _is_scalar_like(delta):
        dn_arr = np.asarray(dn, dtype="float64")
        delta_arr = np.full(dn_arr.shape, int(delta))
    else:
        dn_arr = np.asarray(dn, dtype="float64")
        delta_arr = np.asarray(delta, dtype="int64")
        if dn_arr.shape != delta_arr.shape:
            raise ValueError(
                f"dn and delta must have the same shape, "
                f"got {dn_arr.shape} and {delta_arr.shape}"
            )

    y, m, d, h, mi, s, us = to_ymd(dn_arr)
    total_months = (y * 12 + m - 1) + delta_arr
    new_year = total_months // 12
    new_month = total_months % 12 + 1

    next_m = np.where(new_month == 12, 1, new_month + 1)
    next_y = np.where(new_month == 12, new_year + 1, new_year)
    days_in_target = _civil_to_days(next_y, next_m, 1) - _civil_to_days(
        new_year, new_month, 1
    )
    new_day = np.minimum(d, days_in_target)

    return from_ymd(new_year, new_month, new_day, h, mi, s, us)


# ---------------------------------------------------------------------------
# Individual component accessors
# ---------------------------------------------------------------------------


def _dt64_from_dn(dn: ArrayLike) -> NDArray[np.datetime64]:
    """Convert *dn* to a datetime64[ns] array (scalar callers must not use)."""
    return to_datetime_array(dn)


@overload
def year(dn: DnScalar) -> int: ...
@overload
def year(dn: ArrayLike) -> NDArray[np.int64]: ...
def year(dn):
    """Extract the calendar year from datenum value(s).

    Args:
        dn: Fractional days since the epoch — a scalar or an array.

    Returns:
        An ``int`` for scalar input; an ``int64`` ``ndarray`` for array
        input.

    Example:
        >>> datenum.year(datenum.from_ymd(2024, 7, 15))
        2024
    """
    if _is_scalar_like(dn):
        return to_datetime(dn).year
    dt64 = _dt64_from_dn(dn)
    return dt64.astype("datetime64[Y]").astype("int64") + 1970


@overload
def month(dn: DnScalar) -> int: ...
@overload
def month(dn: ArrayLike) -> NDArray[np.int64]: ...
def month(dn):
    """Extract the month of year (1–12) from datenum value(s).

    Args:
        dn: Fractional days since the epoch — a scalar or an array.

    Returns:
        An ``int`` for scalar input; an ``int64`` ``ndarray`` for array
        input.

    Example:
        >>> datenum.month(datenum.from_ymd(2024, 7, 15))
        7
    """
    if _is_scalar_like(dn):
        return to_datetime(dn).month
    dt64 = _dt64_from_dn(dn)
    return (dt64.astype("datetime64[M]") - dt64.astype("datetime64[Y]")).astype(
        "int64"
    ) + 1


@overload
def day(dn: DnScalar) -> int: ...
@overload
def day(dn: ArrayLike) -> NDArray[np.int64]: ...
def day(dn):
    """Extract the day of month (1–31) from datenum value(s).

    MATLAB-style name for the day-of-month component; see also
    :func:`day_of_month`, an explicit alias.

    Args:
        dn: Fractional days since the epoch — a scalar or an array.

    Returns:
        An ``int`` for scalar input; an ``int64`` ``ndarray`` for array
        input.

    Example:
        >>> datenum.day(datenum.from_ymd(2024, 7, 15))
        15
    """
    if _is_scalar_like(dn):
        return to_datetime(dn).day
    dt64 = _dt64_from_dn(dn)
    return (dt64 - dt64.astype("datetime64[M]")).astype("timedelta64[D]").astype(
        "int64"
    ) + 1


@overload
def hour(dn: DnScalar) -> int: ...
@overload
def hour(dn: ArrayLike) -> NDArray[np.int64]: ...
def hour(dn):
    """Extract the hour of day (0–23) from datenum value(s).

    Args:
        dn: Fractional days since the epoch — a scalar or an array.

    Returns:
        An ``int`` for scalar input; an ``int64`` ``ndarray`` for array
        input.

    Example:
        >>> datenum.hour(datenum.from_ymd(2024, 7, 15, 13))
        13
    """
    if _is_scalar_like(dn):
        return to_datetime(dn).hour
    return _time_of_day_ns(_dt64_from_dn(dn)) // _NS_HOUR


@overload
def minute(dn: DnScalar) -> int: ...
@overload
def minute(dn: ArrayLike) -> NDArray[np.int64]: ...
def minute(dn):
    """Extract the minute (0–59) from datenum value(s).

    Args:
        dn: Fractional days since the epoch — a scalar or an array.

    Returns:
        An ``int`` for scalar input; an ``int64`` ``ndarray`` for array
        input.

    Example:
        >>> datenum.minute(datenum.from_ymd(2024, 7, 15, 13, 45))
        45
    """
    if _is_scalar_like(dn):
        return to_datetime(dn).minute
    tod = _time_of_day_ns(_dt64_from_dn(dn))
    return (tod % _NS_HOUR) // _NS_MINUTE


@overload
def second(dn: DnScalar) -> int: ...
@overload
def second(dn: ArrayLike) -> NDArray[np.int64]: ...
def second(dn):
    """Extract the second (0–59) from datenum value(s).

    Args:
        dn: Fractional days since the epoch — a scalar or an array.

    Returns:
        An ``int`` for scalar input; an ``int64`` ``ndarray`` for array
        input.

    Example:
        >>> datenum.second(datenum.from_ymd(2024, 7, 15, 13, 45, 30))
        30
    """
    if _is_scalar_like(dn):
        return to_datetime(dn).second
    tod = _time_of_day_ns(_dt64_from_dn(dn))
    return (tod % _NS_MINUTE) // _NS_SECOND


@overload
def microsecond(dn: DnScalar) -> int: ...
@overload
def microsecond(dn: ArrayLike) -> NDArray[np.int64]: ...
def microsecond(dn):
    """Extract the microsecond component (0–999999) from datenum value(s).

    Args:
        dn: Fractional days since the epoch — a scalar or an array.

    Returns:
        An ``int`` for scalar input; an ``int64`` ``ndarray`` for array
        input.

    Example:
        >>> datenum.microsecond(datenum.from_ymd(2024, 7, 15, 13, 45, 30, 123456))
        123456
    """
    if _is_scalar_like(dn):
        return to_datetime(dn).microsecond
    tod = _time_of_day_ns(_dt64_from_dn(dn))
    return (tod % _NS_SECOND) // 1_000


# ---------------------------------------------------------------------------
# Calendar helpers
# ---------------------------------------------------------------------------


@overload
def is_leap_year(dn: DnScalar) -> bool: ...
@overload
def is_leap_year(dn: ArrayLike) -> NDArray[np.bool_]: ...
def is_leap_year(dn):
    """Report whether the year containing *dn* is a leap year.

    Uses the Gregorian rule: divisible by 4, except centuries not divisible
    by 400 — so 2000 is a leap year but 1900 is not.

    Args:
        dn: Fractional days since the epoch — a scalar or an array.

    Returns:
        A ``bool`` for scalar input; a boolean ``ndarray`` for array input.

    Example:
        >>> datenum.is_leap_year(datenum.from_ymd(2024, 1, 1))
        True
        >>> datenum.is_leap_year(datenum.from_ymd(1900, 1, 1))
        False
    """
    if _is_scalar_like(dn):
        return calendar.isleap(to_datetime(dn).year)
    dt64 = _dt64_from_dn(dn)
    years = dt64.astype("datetime64[Y]").astype("int64") + 1970
    return (years % 4 == 0) & ((years % 100 != 0) | (years % 400 == 0))


@overload
def day_of_year(dn: DnScalar) -> int: ...
@overload
def day_of_year(dn: ArrayLike) -> NDArray[np.int64]: ...
def day_of_year(dn):
    """Extract the ordinal day of year (1–366) from datenum value(s).

    Counts from January 1 (= 1) within the year containing *dn*; December
    31 is 365 in common years and 366 in leap years. Because the count
    drifts by one after February in non-leap years, prefer
    :func:`day_of_year_229` when aligning daily data across many years.

    Args:
        dn: Fractional days since the epoch — a scalar or an array.

    Returns:
        An ``int`` for scalar input; an ``int64`` ``ndarray`` for array
        input.

    Example:
        >>> datenum.day_of_year(datenum.from_ymd(2024, 1, 1))
        1
        >>> datenum.day_of_year(datenum.from_ymd(2024, 12, 31))
        366
    """
    if _is_scalar_like(dn):
        return to_datetime(dn).timetuple().tm_yday
    dt64 = _dt64_from_dn(dn)
    return (dt64 - dt64.astype("datetime64[Y]")).astype("timedelta64[D]").astype(
        "int64"
    ) + 1


@overload
def day_of_week(dn: DnScalar) -> int: ...
@overload
def day_of_week(dn: ArrayLike) -> NDArray[np.int64]: ...
def day_of_week(dn):
    """Extract the ISO day of week (1=Monday … 7=Sunday) from value(s).

    Follows ISO 8601. Note this differs from MATLAB's ``weekday``, which
    numbers Sunday as 1.

    Args:
        dn: Fractional days since the epoch — a scalar or an array.

    Returns:
        An ``int`` for scalar input; an ``int64`` ``ndarray`` for array
        input.

    Example:
        >>> datenum.day_of_week(datenum.from_ymd(2024, 7, 15))   # Monday
        1
        >>> datenum.day_of_week(datenum.from_ymd(2024, 7, 21))   # Sunday
        7
    """
    if _is_scalar_like(dn):
        return to_datetime(dn).isoweekday()
    dn_arr = np.asarray(dn, dtype="float64")
    epoch_dow = _epoch.isoweekday()
    return ((np.floor(dn_arr).astype("int64") + epoch_dow - 1) % 7 + 1).astype(int)


@overload
def day_of_month(dn: DnScalar) -> int: ...
@overload
def day_of_month(dn: ArrayLike) -> NDArray[np.int64]: ...
def day_of_month(dn):
    """Extract the day of month (1–31) from datenum value(s).

    Explicit-named alias for :func:`day`, for call sites where ``day``
    alone would be ambiguous (e.g. next to :func:`day_of_year`).

    Args:
        dn: Fractional days since the epoch — a scalar or an array.

    Returns:
        An ``int`` for scalar input; an ``int64`` ``ndarray`` for array
        input.

    Example:
        >>> datenum.day_of_month(datenum.from_ymd(2024, 7, 15))
        15
    """
    return day(dn)


@overload
def day_of_year_229(dn: DnScalar) -> int: ...
@overload
def day_of_year_229(dn: ArrayLike) -> NDArray[np.int64]: ...
def day_of_year_229(dn):
    """Day of year on a fixed 366-day calendar, as if Feb 29 always existed.

    Identical to :func:`day_of_year` in leap years. In non-leap years,
    dates after February are shifted up by one day — as though a virtual
    February 29 had been inserted ("229" is the mnemonic 2/29). Every date
    then occupies the same index in every year (March 1 is always 61,
    December 31 always 366), so daily time series such as climatologies can
    be stacked and compared across years without a shift-by-one drift at
    leap boundaries.

    Args:
        dn: Fractional days since the epoch — a scalar or an array.

    Returns:
        An ``int`` in 1–366 for scalar input; an ``int64`` ``ndarray`` for
        array input.

    Example:
        >>> datenum.day_of_year_229(datenum.from_ymd(2025, 3, 1))   # non-leap
        61
        >>> datenum.day_of_year_229(datenum.from_ymd(2024, 3, 1))   # leap
        61
        >>> datenum.day_of_year(datenum.from_ymd(2025, 3, 1))       # drifts
        60
    """
    if _is_scalar_like(dn):
        dt = to_datetime(dn)
        doy = dt.timetuple().tm_yday
        if not calendar.isleap(dt.year) and dt.month > 2:
            return doy + 1
        return doy
    dt64 = _dt64_from_dn(dn)
    years = dt64.astype("datetime64[Y]").astype("int64") + 1970
    is_leap = (years % 4 == 0) & ((years % 100 != 0) | (years % 400 == 0))
    months = (dt64.astype("datetime64[M]") - dt64.astype("datetime64[Y]")).astype(
        "int64"
    ) + 1
    doy = (dt64 - dt64.astype("datetime64[Y]")).astype("timedelta64[D]").astype(
        "int64"
    ) + 1
    return doy + ((~is_leap) & (months > 2)).astype("int64")


# ---------------------------------------------------------------------------
# String <-> datenum
# ---------------------------------------------------------------------------


def from_string(s: str, fmt: str | None = None) -> float:
    """Parse a date string into a datenum.

    Without *fmt*, the string is parsed flexibly via
    :func:`dateutil.parser.parse`, which handles most common layouts
    (ISO 8601, ``"2024 07 15"``, …). With *fmt*, the string must match the
    given Python :func:`strptime` pattern exactly. These are Python format
    codes — not MATLAB ``datestr`` syntax.

    Args:
        s: The date string to parse.
        fmt: Optional :func:`strptime` format, e.g. ``"%d/%m/%Y"``.
            Defaults to flexible dateutil parsing.

    Returns:
        Fractional days between the parsed instant and the epoch.

    Raises:
        ValueError: If the string cannot be parsed (dateutil's
            ``ParserError`` subclasses ``ValueError``).
        TypeError: If parsing yields a timezone-aware datetime, which
            cannot be mixed with the naive epoch.

    Example:
        >>> datenum.to_datetime(datenum.from_string("2024-07-15"))
        datetime.datetime(2024, 7, 15, 0, 0)
        >>> datenum.to_datetime(datenum.from_string("15/07/2024", fmt="%d/%m/%Y"))
        datetime.datetime(2024, 7, 15, 0, 0)
    """
    if fmt is None:
        return from_datetime(parse(s))
    else:
        return from_datetime(_dt.datetime.strptime(s, fmt))  # noqa: DTZ007


def to_string(dn: float, fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    """Format a datenum as a string using a :func:`strftime` pattern.

    Args:
        dn: Fractional days since the epoch (scalar).
        fmt: :func:`strftime` format pattern. Defaults to
            ``"%Y-%m-%d %H:%M:%S"``. Python format codes — not MATLAB
            ``datestr`` syntax.

    Returns:
        The formatted date string.

    Example:
        >>> datenum.to_string(datenum.from_ymd(2024, 7, 15, 13, 45))
        '2024-07-15 13:45:00'
        >>> datenum.to_string(datenum.from_ymd(2024, 7, 15), fmt="%d/%m/%Y")
        '15/07/2024'
    """
    return to_datetime(dn).strftime(fmt)
