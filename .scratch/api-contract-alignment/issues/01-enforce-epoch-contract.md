# 01 — Enforce the naive, midnight, timezone-less epoch contract

**What to build:** `set_epoch` and `temporary_epoch` reject any epoch that is timezone-aware or has a non-midnight time-of-day, raising `ValueError` with a message that names the problem. Valid naive-midnight epochs still work as before, including the default 2000-01-01T00:00:00 and the round-trip through `get_epoch`/`temporary_epoch`. As a result, scalar and vectorized `day_of_week` agree for the same instant.

**Blocked by:** None — can start immediately.

**Status:** ready-for-agent

- [x] `set_epoch` raises `ValueError` for a timezone-aware epoch
- [x] `set_epoch` raises `ValueError` for a naive epoch with non-midnight time-of-day
- [x] `temporary_epoch` enforces the same validation as `set_epoch`
- [x] Valid naive-midnight epochs (including a non-2000 epoch) still work and round-trip
- [x] `day_of_week` agrees between scalar and vectorized calls for the same instant with a changed epoch

## Comments

Implemented: `_validate_epoch` in `src/datenum/core.py`, called from `set_epoch` before mutating the global (so `temporary_epoch`, which routes through `set_epoch`, enforces the same rules and a rejected epoch leaves the current epoch untouched). Errors name the problem: "timezone" for aware datetimes, "midnight" for a non-zero time-of-day. Tests added at the public-API seam in `tests/test_datenum.py` (epoch rejection, unchanged-on-reject, non-2000 midnight round-trip) and `tests/test_vectorize.py` (scalar/vectorized `day_of_week` agreement under a changed epoch).
