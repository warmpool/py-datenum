"""Core datenum conversion functions.

Provides MATLAB-style numeric date representation: dates as fractional days
since a configurable epoch (default 2000-01-01T00:00:00).
"""

from __future__ import annotations

import calendar
import datetime as _dt
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, overload

from dateutil.parser import parse

try:
    import numpy as np

    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

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
    """Set the module-wide epoch for all subsequent conversions.

    The epoch must be a naive datetime at midnight (no timezone, no
    time-of-day); anything else raises :class:`ValueError`.
    """
    _validate_epoch(epoch)
    global _epoch
    _epoch = epoch


def get_epoch() -> _dt.datetime:
    """Return the current module-wide epoch."""
    return _epoch


@contextmanager
def temporary_epoch(epoch: _dt.datetime) -> Generator[None, None, None]:
    """Temporarily set the epoch for the duration of a with-block."""
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
    """Convert a datetime to a datenum (fractional days since epoch)."""
    delta = dt - _epoch
    days = delta.days + delta.seconds / 86400 + delta.microseconds / 86_400_000_000
    return days


def to_datetime(dn: DnScalar) -> _dt.datetime:
    """Convert a datenum to a datetime."""
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
    """Convert datenum value(s) to datetime(s).

    Returns a :class:`datetime` for scalar input, or a
    ``numpy.ndarray`` of ``datetime64[ns]`` for array input.
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
    """Convert datetime(s) to datenum value(s).

    Returns a :class:`float` for scalar input, or a
    ``numpy.ndarray`` of ``float64`` for array input.
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
    """Convert year/month/day (and optional time) to a datenum.

    Float arguments are accepted provided they are whole numbers (e.g. 2024.0);
    non-whole floats raise ValueError.  NumPy arrays are also accepted when
    NumPy is installed; the return value is then a ``float64`` array.
    """
    if _is_scalar_like(year):
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
    """Convert a datenum to (year, month, day, hour, minute, second, microsecond).

    Returns a tuple of seven integers for scalar input, or a tuple of seven
    ``int64`` arrays for array input.
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
    years = dt64.astype("datetime64[Y]").astype("int64") + 1970
    months = (dt64.astype("datetime64[M]") - dt64.astype("datetime64[Y]")).astype(
        "int64"
    ) + 1
    days = (dt64 - dt64.astype("datetime64[M]")).astype("timedelta64[D]").astype(
        "int64"
    ) + 1
    tod = _time_of_day_ns(dt64)
    hours = tod // _NS_HOUR
    minutes = (tod % _NS_HOUR) // _NS_MINUTE
    seconds = (tod % _NS_MINUTE) // _NS_SECOND
    microseconds = (tod % _NS_SECOND) // 1_000
    return (years, months, days, hours, minutes, seconds, microseconds)


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
    """Extract the year component from datenum value(s)."""
    if _is_scalar_like(dn):
        return to_datetime(dn).year
    dt64 = _dt64_from_dn(dn)
    return dt64.astype("datetime64[Y]").astype("int64") + 1970


@overload
def month(dn: DnScalar) -> int: ...
@overload
def month(dn: ArrayLike) -> NDArray[np.int64]: ...
def month(dn):
    """Extract the month component (1–12) from datenum value(s)."""
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
    """Extract the day-of-month component (1–31) from datenum value(s)."""
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
    """Extract the hour component (0–23) from datenum value(s)."""
    if _is_scalar_like(dn):
        return to_datetime(dn).hour
    return _time_of_day_ns(_dt64_from_dn(dn)) // _NS_HOUR


@overload
def minute(dn: DnScalar) -> int: ...
@overload
def minute(dn: ArrayLike) -> NDArray[np.int64]: ...
def minute(dn):
    """Extract the minute component (0–59) from datenum value(s)."""
    if _is_scalar_like(dn):
        return to_datetime(dn).minute
    tod = _time_of_day_ns(_dt64_from_dn(dn))
    return (tod % _NS_HOUR) // _NS_MINUTE


@overload
def second(dn: DnScalar) -> int: ...
@overload
def second(dn: ArrayLike) -> NDArray[np.int64]: ...
def second(dn):
    """Extract the second component (0–59) from datenum value(s)."""
    if _is_scalar_like(dn):
        return to_datetime(dn).second
    tod = _time_of_day_ns(_dt64_from_dn(dn))
    return (tod % _NS_MINUTE) // _NS_SECOND


@overload
def microsecond(dn: DnScalar) -> int: ...
@overload
def microsecond(dn: ArrayLike) -> NDArray[np.int64]: ...
def microsecond(dn):
    """Extract the microsecond component (0–999999) from datenum value(s)."""
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
    """Return True if the year of *dn* is a leap year."""
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
    """Extract the day-of-year component (1–366) from datenum value(s)."""
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
    """Extract the day-of-week (1=Monday … 7=Sunday) from datenum value(s)."""
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
    """Extract the day-of-month component (1–31) from datenum value(s).

    Semantic alias for :func:`day`.
    """
    return day(dn)


@overload
def day_of_year_229(dn: DnScalar) -> int: ...
@overload
def day_of_year_229(dn: ArrayLike) -> NDArray[np.int64]: ...
def day_of_year_229(dn):
    """Like :func:`day_of_year`, but adds 1 for dates after February in
    non-leap years, producing a continuous sequence as if every year had
    366 days.
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
    """Parse a date string into a datenum using *fmt* (default ISO-like)."""
    if fmt is None:
        return from_datetime(parse(s))
    else:
        return from_datetime(_dt.datetime.strptime(s, fmt))  # noqa: DTZ007


def to_string(dn: float, fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    """Format a datenum into a date string using *fmt*."""
    return to_datetime(dn).strftime(fmt)
