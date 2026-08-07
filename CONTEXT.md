# py-datenum

A Python module that wraps the standard `datetime` module to provide MATLAB-style numeric date arithmetic — dates as numbers for easy addition and subtraction.

## Language

**datenum**:
A numeric representation of a date/time as fractional days since a reference epoch. Positive values are after the epoch; negative values are before.
_Avoid_: serial date, date number, serial number

**epoch**:
The reference point from which datenum values are counted. Default is 2000-01-01T00:00:00. Naive, at midnight, timezone-less. Set module-wide with `set_epoch`; scoped to a `with` block with `temporary_epoch`. Epochs with a time-of-day or a timezone are rejected. Timezone semantics are the caller's responsibility — datenums are timezone-less.
_Avoid_: base date, origin, zero point

**from_datetime**:
Convert a Python `datetime` (or sequence of datetimes) to a datenum value (float, or float array). Accepts scalar or array input — every conversion function does.
_Avoid_: from_datetime_array

**to_datetime**:
Convert a datenum value (float, or float array) to a Python `datetime` (or datetime array). Accepts scalar or array input.
_Avoid_: to_datetime_array

**from_ymd**:
Convert year, month, day (and optional hour, minute, second, microsecond) to a datenum value. Whole-number floats are accepted; non-whole floats and out-of-range components raise `ValueError`. Validation applies to scalar and array input alike.

**to_ymd**:
Convert a datenum value to a `(year, month, day, hour, minute, second, microsecond)` tuple.

**from_string**:
Parse a date string into a datenum value using a Python `strptime` format pattern (e.g. `%Y-%m-%d`). Not MATLAB `datestr` syntax.

**to_string**:
Format a datenum value into a date string using a Python `strftime` format pattern (e.g. `%Y-%m-%d`). Not MATLAB `datestr` syntax.

**year**:
Extract the year component from a datenum value.

**month**:
Extract the month component (1–12) from a datenum value.

**day**:
Extract the day-of-month component (1–31) from a datenum value. MATLAB-style name.
_Avoid_: day_of_month, dom

**hour**:
Extract the hour component (0–23) from a datenum value.

**minute**:
Extract the minute component (0–59) from a datenum value.

**second**:
Extract the second component (0–59) from a datenum value.

**microsecond**:
Extract the microsecond component (0–999999) from a datenum value.

**is_leap_year**:
Return True if the year of a datenum value is a leap year (Gregorian calendar rule).
_Avoid_: is_leap

**day_of_year**:
Extract the day-of-year component (1–366) from a datenum value.

**day_of_week**:
Extract the day-of-week component (1=Monday … 7=Sunday, ISO convention) from a datenum value.

**day_of_year_229**:
Day-of-year on a fixed 366-day calendar, as if Feb 29 always exists — "229" is the mnemonic 2/29. Adds 1 to `day_of_year` for dates after February in non-leap years, so every date maps to the same index in every year. Used to align dates in daily climatology without shift-by-1 drift across year boundaries.
_Avoid_: day_of_year_virtual, day_of_year_366
