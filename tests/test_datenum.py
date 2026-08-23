"""Tests for datenum core conversions."""

import datetime as dt

import datenum

try:
    UTC = dt.UTC  # ty: ignore
except AttributeError:
    UTC = dt.timezone.utc  # ty: ignore


# ── Epoch basics ──────────────────────────────────────────────────────────


def test_epoch_default():
    assert datenum.get_epoch() == dt.datetime(2000, 1, 1)


def test_set_epoch_roundtrip():
    original = datenum.get_epoch()
    datenum.set_epoch(dt.datetime(1970, 1, 1))
    try:
        assert datenum.get_epoch() == dt.datetime(1970, 1, 1)
    finally:
        datenum.set_epoch(original)


def test_temporary_epoch_restores():
    original = datenum.get_epoch()
    with datenum.temporary_epoch(dt.datetime(1970, 1, 1)):
        assert datenum.get_epoch() == dt.datetime(1970, 1, 1)
    assert datenum.get_epoch() == original


def test_set_epoch_rejects_timezone_aware():
    import pytest

    with pytest.raises(ValueError, match="timezone"):
        datenum.set_epoch(dt.datetime(2000, 1, 1, tzinfo=UTC))


def test_set_epoch_rejects_non_midnight():
    import pytest

    with pytest.raises(ValueError, match="midnight"):
        datenum.set_epoch(dt.datetime(2000, 1, 1, 12, 0, 0))


def test_rejected_epoch_leaves_current_epoch_unchanged():
    import pytest

    original = datenum.get_epoch()
    with pytest.raises(ValueError):
        datenum.set_epoch(dt.datetime(2000, 1, 1, 6, 30, 0))
    assert datenum.get_epoch() == original


def test_temporary_epoch_rejects_timezone_aware():
    import pytest

    with (
        pytest.raises(ValueError, match="timezone"),
        datenum.temporary_epoch(dt.datetime(2000, 1, 1, tzinfo=UTC)),
    ):
        pass


def test_temporary_epoch_rejects_non_midnight():
    import pytest

    with (
        pytest.raises(ValueError, match="midnight"),
        datenum.temporary_epoch(dt.datetime(2000, 1, 1, 6, 30, 0)),
    ):
        pass


def test_non_2000_midnight_epoch_roundtrips():
    original = datenum.get_epoch()
    epoch = dt.datetime(2015, 7, 1)
    try:
        datenum.set_epoch(epoch)
        assert datenum.get_epoch() == epoch
        instant = dt.datetime(2016, 2, 29, 12, 30, 45, 123456)
        dn = datenum.from_datetime(instant)
        assert datenum.to_datetime(dn) == instant
    finally:
        datenum.set_epoch(original)


# ── from_datetime / to_datetime ───────────────────────────────────────────


def test_epoch_is_zero():
    assert datenum.from_datetime(dt.datetime(2000, 1, 1)) == 0.0


def test_one_day_later():
    dn = datenum.from_datetime(dt.datetime(2000, 1, 2))
    assert dn == 1.0


def test_fractional_day():
    dn = datenum.from_datetime(dt.datetime(2000, 1, 1, 12, 0, 0))
    assert abs(dn - 0.5) < 1e-10


def test_before_epoch_is_negative():
    dn = datenum.from_datetime(dt.datetime(1999, 12, 31))
    assert dn == -1.0


def test_roundtrip_datetime():
    original = dt.datetime(2024, 7, 15, 13, 45, 30, 123456)
    dn = datenum.from_datetime(original)
    recovered = datenum.to_datetime(dn)
    assert recovered == original


def test_roundtrip_microsecond_precision():
    """Microsecond-level roundtrip should survive."""
    original = dt.datetime(2024, 1, 1, 0, 0, 0, 1)
    dn = datenum.from_datetime(original)
    recovered = datenum.to_datetime(dn)
    assert recovered == original


# ── from_ymd / to_ymd ────────────────────────────────────────────────────


def test_from_ymd_matches_datetime():
    expected = datenum.from_datetime(dt.datetime(2024, 3, 15))
    assert datenum.from_ymd(2024, 3, 15) == expected


def test_from_ymd_with_time():
    expected = datenum.from_datetime(dt.datetime(2024, 3, 15, 10, 30, 45))
    assert datenum.from_ymd(2024, 3, 15, 10, 30, 45) == expected


def test_from_ymd_whole_floats():
    assert datenum.from_ymd(2024.0, 3.0, 15.0) == datenum.from_ymd(2024, 3, 15)


def test_from_ymd_non_whole_float_raises():
    import pytest

    with pytest.raises(ValueError, match="whole number"):
        datenum.from_ymd(2024.5, 1, 1)


def test_to_ymd_roundtrip():
    original = (2024, 7, 15, 13, 45, 30, 0)
    dn = datenum.from_ymd(*original)
    assert datenum.to_ymd(dn)[:3] == original[:3]
    assert datenum.to_ymd(dn)[3:6] == original[3:6]


# ── from_string / to_string ───────────────────────────────────────────────


def test_from_string_using_dateutil_parser():
    dn = datenum.from_string("2024-07-15 13:45:30")
    expected = datenum.from_datetime(dt.datetime(2024, 7, 15, 13, 45, 30))
    assert dn == expected

    dn = datenum.from_string("2024-07-15")
    expected = datenum.from_datetime(dt.datetime(2024, 7, 15))
    assert dn == expected

    dn = datenum.from_string("2024 07 15")
    expected = datenum.from_datetime(dt.datetime(2024, 7, 15))
    assert dn == expected


def test_from_string_custom_format():
    dn = datenum.from_string("15/07/2024", fmt="%d/%m/%Y")
    expected = datenum.from_datetime(dt.datetime(2024, 7, 15))
    assert dn == expected


def test_to_string_roundtrip():
    dn = datenum.from_ymd(2024, 7, 15, 13, 45, 30)
    s = datenum.to_string(dn)
    assert s == "2024-07-15 13:45:30"


def test_to_string_custom_format():
    dn = datenum.from_ymd(2024, 7, 15)
    s = datenum.to_string(dn, fmt="%d/%m/%Y")
    assert s == "15/07/2024"


# ── Arithmetic ────────────────────────────────────────────────────────────


def test_add_days():
    dn1 = datenum.from_ymd(2024, 1, 1)
    dn2 = dn1 + 7
    result = datenum.to_ymd(dn2)
    assert result[:3] == (2024, 1, 8)


def test_subtract_dates_gives_days():
    dn1 = datenum.from_ymd(2024, 1, 1)
    dn2 = datenum.from_ymd(2024, 1, 31)
    assert dn2 - dn1 == 30.0


# ── Individual component accessors ────────────────────────────────────────


def test_year():
    dn = datenum.from_ymd(2024, 7, 15, 13, 45, 30, 123456)
    assert datenum.year(dn) == 2024


def test_month():
    dn = datenum.from_ymd(2024, 7, 15, 13, 45, 30, 123456)
    assert datenum.month(dn) == 7


def test_day():
    dn = datenum.from_ymd(2024, 7, 15, 13, 45, 30, 123456)
    assert datenum.day(dn) == 15


def test_hour():
    dn = datenum.from_ymd(2024, 7, 15, 13, 45, 30, 123456)
    assert datenum.hour(dn) == 13


def test_minute():
    dn = datenum.from_ymd(2024, 7, 15, 13, 45, 30, 123456)
    assert datenum.minute(dn) == 45


def test_second():
    dn = datenum.from_ymd(2024, 7, 15, 13, 45, 30, 123456)
    assert datenum.second(dn) == 30


def test_microsecond():
    dn = datenum.from_ymd(2024, 7, 15, 13, 45, 30, 123456)
    assert datenum.microsecond(dn) == 123456


# ── Calendar helpers ──────────────────────────────────────────────────────


def test_is_leap_year_true():
    assert datenum.is_leap_year(datenum.from_ymd(2024, 1, 1)) is True


def test_is_leap_year_false():
    assert datenum.is_leap_year(datenum.from_ymd(2025, 1, 1)) is False


def test_is_leap_year_century_not_div_400():
    assert datenum.is_leap_year(datenum.from_ymd(1900, 1, 1)) is False


def test_is_leap_year_century_div_400():
    assert datenum.is_leap_year(datenum.from_ymd(2000, 1, 1)) is True


def test_day_of_year_jan_1():
    assert datenum.day_of_year(datenum.from_ymd(2024, 1, 1)) == 1


def test_day_of_year_dec_31_leap():
    assert datenum.day_of_year(datenum.from_ymd(2024, 12, 31)) == 366


def test_day_of_year_dec_31_non_leap():
    assert datenum.day_of_year(datenum.from_ymd(2025, 12, 31)) == 365


def test_day_of_week_monday():
    assert datenum.day_of_week(datenum.from_ymd(2024, 7, 15)) == 1


def test_day_of_week_sunday():
    assert datenum.day_of_week(datenum.from_ymd(2024, 7, 21)) == 7


def test_day_of_month():
    assert datenum.day_of_month(datenum.from_ymd(2024, 7, 15)) == 15


def test_day_of_year_229_non_leap_mar_1():
    assert datenum.day_of_year_229(datenum.from_ymd(2025, 3, 1)) == 61


def test_day_of_year_229_non_leap_feb_28():
    assert datenum.day_of_year_229(datenum.from_ymd(2025, 2, 28)) == 59


def test_day_of_year_229_leap_mar_1():
    assert datenum.day_of_year_229(datenum.from_ymd(2024, 3, 1)) == 61


# ── add_month ──────────────────────────────────────────────────────────────


def test_add_month_basic():
    dn = datenum.from_ymd(2024, 1, 15)
    result = datenum.to_ymd(datenum.add_month(dn, 1))
    assert result[:3] == (2024, 2, 15)


def test_add_month_clamp_leap():
    dn = datenum.from_ymd(2024, 1, 31)
    result = datenum.to_ymd(datenum.add_month(dn, 1))
    assert result[:3] == (2024, 2, 29)


def test_add_month_clamp_non_leap():
    dn = datenum.from_ymd(2025, 1, 31)
    result = datenum.to_ymd(datenum.add_month(dn, 1))
    assert result[:3] == (2025, 2, 28)


def test_add_month_subtract():
    dn = datenum.from_ymd(2024, 3, 15)
    result = datenum.to_ymd(datenum.add_month(dn, -1))
    assert result[:3] == (2024, 2, 15)


def test_add_month_wrap_year():
    dn = datenum.from_ymd(2024, 12, 15)
    result = datenum.to_ymd(datenum.add_month(dn, 1))
    assert result[:3] == (2025, 1, 15)


def test_add_month_preserves_time():
    dn = datenum.from_ymd(2024, 1, 15, 12, 30, 45)
    result = datenum.to_ymd(datenum.add_month(dn, 1))
    assert result[:3] == (2024, 2, 15)
    assert result[3:6] == (12, 30, 45)


def test_add_month_default_delta():
    dn = datenum.from_ymd(2024, 1, 15)
    result = datenum.to_ymd(datenum.add_month(dn))
    assert result[:3] == (2024, 2, 15)


def test_add_month_zero():
    dn = datenum.from_ymd(2024, 6, 15)
    result = datenum.to_ymd(datenum.add_month(dn, 0))
    assert result[:3] == (2024, 6, 15)


def test_add_month_scalar_dn_array_delta():
    import numpy as np

    dn = datenum.from_ymd(2024, 1, 15)
    deltas = np.array([1, 2, 3])
    results = datenum.add_month(dn, deltas)
    assert isinstance(results, np.ndarray)
    ymds = [datenum.to_ymd(r)[:3] for r in results]
    assert ymds == [(2024, 2, 15), (2024, 3, 15), (2024, 4, 15)]


def test_add_month_array_dn_scalar_delta():
    import numpy as np

    dns = np.array(
        [
            datenum.from_ymd(2024, 1, 15),
            datenum.from_ymd(2024, 6, 20),
            datenum.from_ymd(2024, 11, 1),
        ]
    )
    results = datenum.add_month(dns, 1)
    assert isinstance(results, np.ndarray)
    ymds = [datenum.to_ymd(r)[:3] for r in results]
    assert ymds == [(2024, 2, 15), (2024, 7, 20), (2024, 12, 1)]


def test_add_month_array_dn_array_delta():
    import numpy as np

    dns = np.array(
        [
            datenum.from_ymd(2024, 1, 31),
            datenum.from_ymd(2024, 3, 15),
            datenum.from_ymd(2024, 12, 1),
        ]
    )
    deltas = np.array([1, -1, 2])
    results = datenum.add_month(dns, deltas)
    assert isinstance(results, np.ndarray)
    ymds = [datenum.to_ymd(r)[:3] for r in results]
    assert ymds == [(2024, 2, 29), (2024, 2, 15), (2025, 2, 1)]


def test_add_month_mismatched_lengths():
    import numpy as np
    import pytest

    dns = np.array([datenum.from_ymd(2024, 1, 15), datenum.from_ymd(2024, 2, 15)])
    deltas = np.array([1, 2, 3])
    with pytest.raises(ValueError, match="same shape"):
        datenum.add_month(dns, deltas)
