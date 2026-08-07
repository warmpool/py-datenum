Status: ready-for-agent

# Align the `datenum` API with the domain model

## Problem Statement

The `datenum` package's public API contradicts its own domain model. Several contracts are inconsistent or unstated:

- `from_datetime`/`to_datetime` are scalar-only while every sibling function auto-vectorizes, forcing users to learn the separate `from_datetime_array`/`to_datetime_array` pair.
- `set_epoch` accepts any datetime, including tz-aware and non-midnight values, even though the domain model says the epoch is naive, midnight, and timezone-less. A non-midnight epoch makes scalar and vectorized `day_of_week` disagree on the same input.
- `day_of_month` is a redundant alias for `day`, polluting the API with two names for one concept.
- Scalar and array input paths validate differently: `from_ymd(2024.5, 1, 1)` raises, but `from_ymd([2024.5], [1], [1])` silently truncates; invalid components (month 13, day 32) raise as scalars but silently produce garbage as arrays.

## Solution

From the user's perspective, one uniform, documented contract:

- Every conversion function accepts scalar or array input and returns scalar or array output accordingly.
- The epoch is naive, at midnight, and timezone-less. `set_epoch` and `temporary_epoch` reject anything that isn't, with a clear error.
- There is exactly one name for each concept: `day`, not `day_of_month`.
- Validation behaves the same for scalar and array input — non-whole floats and out-of-range components raise `ValueError` either way.
- String formats use Python `strftime`/`strptime` syntax, never MATLAB `datestr` syntax.

## User Stories

1. As a user converting a single datetime, I want `from_datetime` to accept it and return a float, so that scalar use stays simple.
2. As a user converting a list or numpy array of datetimes, I want `from_datetime` to accept it and return a float array, so that I don't need a separate function or a Python loop.
3. As a user converting a datenum to a datetime, I want `to_datetime` to accept a float or an array of floats and return the matching datetime or datetime array, so that round-trips work at any scale.
4. As a user of the scalar-returns-scalar, array-returns-array contract, I want a single-element numpy array to come back as an array, never a scalar, so that the output container type matches the input.
5. As a user, I want `from_datetime_array` and `to_datetime_array` gone from the public API, so that there is exactly one way to convert between datenums and datetimes.
6. As a user setting the epoch, I want `set_epoch` to reject a tz-aware datetime, so that I can't accidentally make the library timezone-dependent.
7. As a user setting the epoch, I want `set_epoch` to reject a non-midnight datetime, so that day boundaries stay aligned and datenums remain whole-day counts.
8. As a user using `temporary_epoch`, I want it to enforce the same rules as `set_epoch`, so that scoped epochs are as safe as module-wide ones.
9. As a user receiving an invalid epoch, I want a `ValueError` that names the problem (timezone or time-of-day), so that I can fix my call immediately.
10. As a vectorized user, I want `from_ymd` on arrays to raise `ValueError` for a non-whole float, so that I get the same failure I'd get as a scalar caller.
11. As a vectorized user, I want `from_ymd` on arrays to raise `ValueError` for out-of-range components (e.g. month 13, day 32), so that invalid dates can't silently produce garbage datenums.
12. As a scalar user, I want `from_ymd` to keep raising for non-whole floats and out-of-range components, so that my existing error handling keeps working.
13. As a user of the component accessors, I want `day_of_month` removed so that there is exactly one canonical name, `day`.
14. As a MATLAB refugee, I want `from_string`/`to_string` to accept Python `strftime`/`strptime` format patterns, so that I know the format language without trial and error.
15. As a climatology user, I want `day_of_year_229` to return the same index for a given date in any year, leap or not, so that my daily climatology series has no shift-by-1 drift across year boundaries.
16. As a user mixing scalar and vectorized calls, I want `day_of_week` to agree for the same instant regardless of how it's called, so that one code path doesn't contradict another.
17. As a user, I want a `from_datetime` → `to_datetime` round-trip to preserve microsecond precision, so that converting back and forth never loses data.
18. As a maintainer, I want the epoch contract (naive, midnight, timezone-less) enforced in code and stated in the glossary and ADR, so that future contributors can't silently break it.
19. As a future contributor, I want to know why tz-aware and fractional epochs are rejected, so that I don't reintroduce timezone support or half-day epochs in a well-meaning refactor.
20. As a user, I want the glossary to describe the uniform scalar-or-array and validation contracts, so that I know what to expect without reading source.

## Implementation Decisions

- **Unified conversion pair.** `from_datetime` and `to_datetime` become polymorphic over scalar and array input, dispatching through the existing `_is_scalar_like` check exactly like `from_ymd`/`to_ymd` and the component accessors. The current array logic moves into private helpers.
- **Remove the `_array` pair from the public API.** `from_datetime_array` and `to_datetime_array` are dropped from the package exports and `__all__`. Existing behavior is preserved behind private names for internal use by the now-unified public functions.
- **Epoch validation.** `set_epoch` and `temporary_epoch` raise `ValueError` for any epoch that is tz-aware or has a non-midnight time-of-day. This makes the vectorized day math correct by construction: with a midnight-only epoch, the floor-based vectorized `day_of_week` agrees with the scalar path.
- **Drop `day_of_month`.** Removed from the package exports; `day` is the single canonical name for the day-of-month component.
- **Uniform validation in `from_ymd`.** The vectorized path rejects non-whole float inputs and out-of-range components with `ValueError`, matching the scalar path's behavior. The scalar path is unchanged.
- **No change to rounding.** `to_datetime` keeps its existing microsecond rounding.
- **Glossary-aligned but already-correct behavior.** `from_string`/`to_string` (Python format syntax) and `day_of_year_229` need no code change; the decisions are recorded in `CONTEXT.md`.
- **Domain docs.** `CONTEXT.md` already reflects all decisions; `docs/adr/0001-naive-midnight-timezone-less-epoch.md` records the epoch contract.

## Testing Decisions

- **Good tests observe only external behavior** through the public `datenum` package — the single testing seam. No test reaches into module internals (helper functions, `__all__`, private names).
- **Modules tested.** The `datenum` package only, via the existing scalar and vectorized test files.
- **Prior art.** The existing suite is the template: `TestEpochIntegration` for epoch-rule tests, `TestScalarRegression` for scalar-return guarantees, `TestVectorized*` classes for array behavior, and the round-trip tests for precision. New tests extend these files at the same seam.
- **New tests to add:**
  - `set_epoch`/`temporary_epoch` raise `ValueError` for tz-aware and non-midnight epochs; valid midnight naive epochs still work.
  - `from_datetime`/`to_datetime` accept lists and numpy arrays and return arrays; single-element arrays return arrays.
  - `from_datetime_array`/`to_datetime_array` and `day_of_month` are no longer importable from the package.
  - Vectorized `from_ymd` raises `ValueError` for non-whole floats and out-of-range components.
  - Scalar/vectorized `day_of_week` agree for the same instants.
  - Round-trip precision tests still pass through the unified pair.

## Out of Scope

- MATLAB `datestr`/`datenum` format-string compatibility.
- Support for tz-aware datetimes or any timezone handling inside the library.
- Non-midnight epochs.
- Per-call `epoch=` parameters — epoch is per-module only.
- New calendar functions beyond the existing surface.
- Performance tuning of the vectorized paths.
- Changes to `from_string`/`to_string` or `day_of_year_229` code — already correct; documented only.

## Further Notes

- This is a breaking change to the public API (two removals, two functions gaining array support, stricter epoch validation) at version 0.1.0 — the package version should be bumped before release.
- The two existing test files are the only seams; the ideal one-seam structure is already in place.
- The vectorized `day_of_week` inconsistency noted during grilling is resolved by the midnight-only epoch contract rather than by changing the day math, which is why epoch validation is the load-bearing piece of this spec.
