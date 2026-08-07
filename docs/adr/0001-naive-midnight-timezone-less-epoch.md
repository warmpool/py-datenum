# Naive, midnight, timezone-less epoch

The epoch must be a naive datetime at midnight with no timezone — `set_epoch` enforces this and rejects anything else. A datenum is fractional days since this epoch; allowing a half-day or tz-aware epoch would make the scalar and vectorized day math disagree (day-of-week, day-of-year), and a tz-aware epoch would break the timezone-less contract. Timezone interpretation is the caller's responsibility.
