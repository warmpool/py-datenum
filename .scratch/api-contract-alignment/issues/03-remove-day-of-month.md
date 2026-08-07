# 03 — Remove day_of_month from the public API

**What to build:** `day_of_month` is no longer exported from the package; `day` is the single canonical name for extracting the day-of-month component. Everything `day_of_month` used to do is still available through `day`.

**Blocked by:** None — can start immediately.

**Status:** ready-for-agent

- [ ] `day_of_month` is no longer importable from the package
- [ ] `day` still extracts the day-of-month component (1–31) for scalar and array input
