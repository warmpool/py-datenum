# 02 — Unify from_datetime/to_datetime for scalar and array input

**What to build:** `from_datetime` and `to_datetime` accept both scalar and array input, returning a scalar for a single value and an array for a list or numpy array — the same polymorphic contract as `from_ymd` and the component accessors. A single-element array comes back as an array, never a scalar. The public `from_datetime_array`/`to_datetime_array` names are retired from the package exports; their logic lives on privately. Microsecond-precision round-trips survive through the unified pair.

**Blocked by:** None — can start immediately.

**Status:** ready-for-agent

- [ ] `from_datetime` returns a float for a single datetime
- [ ] `from_datetime` returns a float array for a list/array of datetimes
- [ ] `to_datetime` returns a datetime for a scalar datenum
- [ ] `to_datetime` returns a datetime array for an array of datenums
- [ ] A single-element numpy array returns an array, not a scalar
- [ ] `from_datetime_array` and `to_datetime_array` are no longer importable from the package
- [ ] `from_datetime` → `to_datetime` round-trip preserves microsecond precision
