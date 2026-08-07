# 04 — Make vectorized from_ymd validation match the scalar path

**What to build:** The vectorized `from_ymd` path rejects non-whole float inputs and out-of-range components (e.g. month 13, day 32) with `ValueError`, exactly as the scalar path already does. No array input silently truncates a fractional value or converts an invalid date into a garbage datenum. Valid vectorized inputs (including broadcasts and leap-day dates) behave unchanged.

**Blocked by:** None — can start immediately.

**Status:** ready-for-agent

- [ ] `from_ymd` on arrays raises `ValueError` for a non-whole float component
- [ ] `from_ymd` on arrays raises `ValueError` for out-of-range month/day (and time components where applicable)
- [ ] Valid vectorized calls (leap years, scalar broadcasts) still produce correct datenums
- [ ] Scalar `from_ymd` behavior is unchanged
