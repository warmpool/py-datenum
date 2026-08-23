# py-datenum

MATLAB-style numeric dates for Python. A thin, well-tested wrapper around the
standard `datetime` module that represents dates as **datenums** — fractional
days since a configurable epoch — so date arithmetic becomes plain arithmetic:

```python
import datetime as dt
import datenum

dn = datenum.from_datetime(dt.datetime(2024, 7, 15))
dn + 7                    # a week later, as a number
dn2 = datenum.from_datetime(dt.datetime(2024, 1, 31))
dn2 - dn                  # days between two dates: -166.0
```

## Why?

Comparing, sorting, differencing, and interpolating timestamps is simplest when
they are numbers. MATLAB users know this workflow as `datenum`. This library
brings it to Python while staying fully interoperable with `datetime`, and adds
NumPy-vectorized conversions for array workloads.

## Features

- **Scalar conversions** between `datetime`, `(year, month, day, ...)` tuples,
  strings, and datenums.
- **Vectorized conversions** accepting lists, tuples, and NumPy arrays — every
  conversion function accepts scalar *or* array input.
- **Component accessors** (`year`, `month`, `day`, `hour`, `minute`, `second`,
  `microsecond`) that work element-wise on arrays.
- **Calendar helpers**: `is_leap_year`, `day_of_year`, `day_of_week`,
  `day_of_year_229` (fixed 366-day calendar for climatology).
- **Month arithmetic** with end-of-month clamping (`add_month`).
- **Configurable epoch**, globally via `set_epoch` or scoped via
  `temporary_epoch`.
- Microsecond-precision roundtrips (`datetime -> datenum -> datetime` is exact
  at microsecond resolution for dates near the epoch).

## Installation

Requires Python ≥ 3.10.

```sh
pip install datenum
```

or with [uv](https://docs.astral.sh/uv/):

```sh
uv add datenum
```

Dependencies: `python-dateutil` and `numpy` (optional for vectorization) .

## Core concept: datenum

A **datenum** is a `float`: the number of days (including fractions) since the
**epoch**. Values after the epoch are positive; values before it are negative.

The default epoch is **2000-01-01T00:00:00** — naive (no timezone), at
midnight. `datenum.from_datetime(dt.datetime(2000, 1, 1))` is exactly `0.0`;
noon the same day is `0.5`; 1999-12-31 is `-1.0`.

> **Note:** this differs from MATLAB, whose `datenum` counts from year 0.
> Convert once at your system boundary, or set the epoch to match your data.

> **Timezones:** datenums are timezone-less by design. The epoch must be naive
> and at midnight; timezone interpretation is the caller's responsibility.

## Quick start

### Conversions

```python
import datetime as dt
import datenum

# datetime <-> datenum
dn = datenum.from_datetime(dt.datetime(2024, 7, 15, 13, 45, 30))
datenum.to_datetime(dn)          # datetime.datetime(2024, 7, 15, 13, 45, 30)

# YMD <-> datenum
dn = datenum.from_ymd(2024, 7, 15)                     # midnight assumed
dn = datenum.from_ymd(2024, 7, 15, 10, 30, 45)         # with time of day
datenum.to_ymd(dn)               # (2024, 7, 15, 10, 30, 45, 0)

# Strings <-> datenum (Python strptime/strftime formats, plus fuzzy parsing)
dn = datenum.from_string("15/07/2024", fmt="%d/%m/%Y")
datenum.to_string(dn, fmt="%Y-%m-%d")            # '2024-07-15'
```

### Arithmetic

Because datenums are floats, durations and shifts are ordinary numbers:

```python
dn = datenum.from_ymd(2024, 1, 1)
later = dn + 90                  # 90 days later
gap = later - dn                 # 90.0

dn = datenum.add_month(dn, 1)    # calendar-aware month addition
```

`add_month` clamps overflow days: Jan 31 + 1 month is Feb 29 (leap year) or
Feb 28 otherwise, and time-of-day is preserved.

### Components and calendar helpers

```python
dn = datenum.from_ymd(2024, 7, 15, 13, 45, 30, 123456)

datenum.year(dn)          # 2024
datenum.month(dn)         # 7
datenum.day(dn)           # 15        (day of month)
datenum.hour(dn)          # 13
datenum.minute(dn)        # 45
datenum.second(dn)        # 30
datenum.microsecond(dn)   # 123456

datenum.day_of_week(dn)   # 1         (ISO: 1=Monday … 7=Sunday)
datenum.day_of_year(dn)   # 197       (1–366)
datenum.is_leap_year(dn)  # True
```

### Arrays (NumPy)

Every conversion function accepts scalar *or* array input and returns a scalar
*or* array correspondingly:

```python
import numpy as np
import datenum

dns = datenum.from_datetime_array(
    np.array(["2000-01-01", "2000-01-02"], dtype="datetime64[ns]")
)
# array([0., 1.])

dt64 = datenum.to_datetime_array(np.array([0.0, 0.5]))
# array(['2000-01-01T00:00:00', '2000-01-01T12:00:00'], dtype='datetime64[ns]')

years = datenum.year([0.0, 400.0, 8000.0])          # array([2000, 2001, 2021])
leap = datenum.is_leap_year([0.0, 366.0])           # array([True, False])
shifted = datenum.add_month(np.array([0.0, 31.0]), 1)  # broadcast over dn
```

Rules for `add_month` broadcasting: scalar `dn` + array `delta`, array `dn` +
scalar `delta` (broadcast), or two arrays of identical shape (mismatched shapes
raise `ValueError`).

Array results use vectorized integer/nanosecond paths — no per-element Python
loops — and return `float64`, `int64`, or `datetime64[ns]` arrays.

### Configuring the epoch

```python
import datetime as dt
import datenum

datenum.set_epoch(dt.datetime(1970, 1, 1))     # module-wide, affects everything after
assert datenum.get_epoch() == dt.datetime(1970, 1, 1)

with datenum.temporary_epoch(dt.datetime(1900, 1, 1)):
    dn = datenum.from_datetime(dt.datetime(1900, 1, 2))   # 1.0 within the block
# previous epoch restored automatically
```

Both reject invalid epochs with `ValueError`:

- timezone-aware datetimes,
- datetimes with a time-of-day other than midnight.

A failed `set_epoch` leaves the current epoch unchanged.

## API summary

| Category | Functions |
| --- | --- |
| Epoch | `get_epoch`, `set_epoch`, `temporary_epoch` |
| datetime ↔ datenum | `from_datetime`, `to_datetime`, `from_datetime_array`, `to_datetime_array` |
| YMD ↔ datenum | `from_ymd`, `to_ymd` |
| Strings ↔ datenum | `from_string`, `to_string` |
| Components | `year`, `month`, `day`, `hour`, `minute`, `second`, `microsecond` |
| Calendar | `is_leap_year`, `day_of_year`, `day_of_week`, `day_of_month`, `day_of_year_229` |
| Arithmetic | `add_month` |

Full details live in the docstrings (`help(datenum)` / `datenum.from_ymd?`).

## Behavior notes

- **String parsing** follows Python conventions, not MATLAB `datestr` syntax:
  `fmt` is a `strptime`/`strftime` pattern. Without `fmt`, `from_string` uses
  `dateutil`'s flexible parser.
- **Day-of-week** is ISO (`1=Monday … 7=Sunday`); MATLAB's `weekday` instead
  uses `1=Sunday`.
- **Validation:** scalar `from_ymd` rejects non-whole floats and out-of-range
  components with `ValueError`; array inputs are assumed valid and are not
  range-checked.
- **Precision:** datenums are `float64` (~15–16 significant digits). Around the
  default epoch that comfortably covers microsecond resolution; dates many tens
  of thousands of years from the epoch lose sub-second precision.
- **Negative datenums** work throughout (before the epoch), including arrays.

## Development

```sh
uv sync                 # create venv + install deps
uv run pytest           # run the test suite
uv run ruff check .     # lint
```
