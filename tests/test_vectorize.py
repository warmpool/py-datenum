"""Tests for NumPy-vectorised datenum operations."""

import datetime as dt

import numpy as np
import pytest

import datenum

# ── from_datetime_array / to_datetime_array ────────────────────────────────


class TestFromDatetimeArray:
    def test_single_datetime_returns_scalar(self):
        result = datenum.from_datetime_array(dt.datetime(2000, 1, 1))
        assert isinstance(result, float)
        assert result == 0.0

    def test_list_of_datetimes(self):
        dts = [
            dt.datetime(2000, 1, 1),
            dt.datetime(2000, 1, 2),
            dt.datetime(2000, 1, 3),
        ]
        result = datenum.from_datetime_array(dts)
        np.testing.assert_array_equal(result, [0.0, 1.0, 2.0])

    def test_numpy_datetime64_array(self):
        dt64 = np.array(
            ["2000-01-01", "2000-06-15", "2024-12-31"], dtype="datetime64[D]"
        )
        result = datenum.from_datetime_array(dt64)
        assert len(result) == 3
        assert result.dtype == np.float64
        assert result[0] == 0.0

    def test_fractional_days(self):
        dts = [dt.datetime(2000, 1, 1, 12, 0, 0)]
        result = datenum.from_datetime_array(dts)
        np.testing.assert_allclose(result, [0.5], atol=1e-10)

    def test_before_epoch(self):
        result = datenum.from_datetime_array([dt.datetime(1999, 12, 31)])
        np.testing.assert_array_equal(result, [-1.0])

    def test_empty_array(self):
        result = datenum.from_datetime_array(np.array([], dtype="datetime64[ns]"))
        assert len(result) == 0
        assert result.dtype == np.float64

    def test_microsecond_precision(self):
        dts = [dt.datetime(2024, 1, 1, 0, 0, 0, 123456)]
        result = datenum.from_datetime_array(dts)
        dn_scalar = datenum.from_datetime(dts[0])
        np.testing.assert_allclose(result, [dn_scalar], atol=1e-12)

    def test_single_element_array(self):
        result = datenum.from_datetime_array([dt.datetime(2024, 7, 15)])
        assert result.shape == (1,)
        assert result.ndim == 1


class TestToDatetimeArray:
    def test_scalar_returns_datetime(self):
        result = datenum.to_datetime_array(0.0)
        assert isinstance(result, dt.datetime)
        assert result == dt.datetime(2000, 1, 1)

    def test_array_of_floats(self):
        dn = np.array([0.0, 1.0, 2.0])
        result = datenum.to_datetime_array(dn)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.dtype("datetime64[ns]")
        assert len(result) == 3

    def test_roundtrip_array(self):
        dts = [
            dt.datetime(2024, 7, 15, 13, 45, 30, 123456),
            dt.datetime(2000, 1, 1, 0, 0, 0, 0),
            dt.datetime(1999, 12, 31, 23, 59, 59, 999999),
        ]
        dn = datenum.from_datetime_array(dts)
        recovered = datenum.to_datetime_array(dn)
        expected = np.array(dts, dtype="datetime64[ns]")
        np.testing.assert_array_equal(recovered, expected)

    def test_empty_array(self):
        result = datenum.to_datetime_array(np.array([], dtype="float64"))
        assert len(result) == 0

    def test_negative_datenum(self):
        result = datenum.to_datetime_array(-1.0)
        assert result == dt.datetime(1999, 12, 31)

    def test_single_element_array(self):
        result = datenum.to_datetime_array(np.array([0.0]))
        assert result.shape == (1,)
        assert result.dtype == np.dtype("datetime64[ns]")


# ── Vectorised component accessors ────────────────────────────────────────


@pytest.fixture()
def sample_dns():
    """Array of four datenums covering diverse edge cases."""
    return np.array(
        [
            datenum.from_ymd(2024, 7, 15, 13, 45, 30, 123456),  # Tue
            datenum.from_ymd(2024, 1, 1, 0, 0, 0, 0),  # Mon (leap year)
            datenum.from_ymd(2025, 12, 31, 23, 59, 59, 999999),  # Wed (non-leap)
            datenum.from_ymd(2000, 1, 1, 0, 0, 0, 0),  # Sat (epoch)
        ]
    )


class TestVectorizedComponents:
    def test_year(self, sample_dns):
        result = datenum.year(sample_dns)
        np.testing.assert_array_equal(result, [2024, 2024, 2025, 2000])

    def test_month(self, sample_dns):
        result = datenum.month(sample_dns)
        np.testing.assert_array_equal(result, [7, 1, 12, 1])

    def test_day(self, sample_dns):
        result = datenum.day(sample_dns)
        np.testing.assert_array_equal(result, [15, 1, 31, 1])

    def test_hour(self, sample_dns):
        result = datenum.hour(sample_dns)
        np.testing.assert_array_equal(result, [13, 0, 23, 0])

    def test_minute(self, sample_dns):
        result = datenum.minute(sample_dns)
        np.testing.assert_array_equal(result, [45, 0, 59, 0])

    def test_second(self, sample_dns):
        result = datenum.second(sample_dns)
        np.testing.assert_array_equal(result, [30, 0, 59, 0])

    def test_microsecond(self, sample_dns):
        result = datenum.microsecond(sample_dns)
        np.testing.assert_array_equal(result, [123456, 0, 999999, 0])

    def test_single_element_returns_array(self):
        dns = np.array([datenum.from_ymd(2024, 7, 15)])
        result = datenum.year(dns)
        assert isinstance(result, np.ndarray)
        assert result.shape == (1,)


# ── Vectorised calendar helpers ───────────────────────────────────────────


class TestVectorizedCalendarHelpers:
    def test_is_leap_year_array(self):
        dns = np.array(
            [
                datenum.from_ymd(2024, 1, 1),
                datenum.from_ymd(2025, 1, 1),
                datenum.from_ymd(1900, 1, 1),
                datenum.from_ymd(2000, 1, 1),
            ]
        )
        result = datenum.is_leap_year(dns)
        np.testing.assert_array_equal(result, [True, False, False, True])

    def test_day_of_year_array(self):
        dns = np.array(
            [
                datenum.from_ymd(2024, 1, 1),
                datenum.from_ymd(2024, 12, 31),
                datenum.from_ymd(2025, 12, 31),
            ]
        )
        result = datenum.day_of_year(dns)
        np.testing.assert_array_equal(result, [1, 366, 365])

    def test_day_of_week_array(self):
        dns = np.array(
            [
                datenum.from_ymd(2024, 7, 15),  # Monday
                datenum.from_ymd(2024, 7, 21),  # Sunday
                datenum.from_ymd(2000, 1, 1),  # Saturday (epoch)
            ]
        )
        result = datenum.day_of_week(dns)
        np.testing.assert_array_equal(result, [1, 7, 6])

    def test_day_of_month_array(self):
        dns = np.array(
            [
                datenum.from_ymd(2024, 7, 15),
                datenum.from_ymd(2024, 1, 1),
                datenum.from_ymd(2024, 12, 31),
            ]
        )
        result = datenum.day_of_month(dns)
        np.testing.assert_array_equal(result, [15, 1, 31])

    def test_day_of_year_229_array(self):
        dns = np.array(
            [
                datenum.from_ymd(2025, 3, 1),  # non-leap, month > 2 → +1
                datenum.from_ymd(2025, 2, 28),  # non-leap, month <= 2 → same
                datenum.from_ymd(2024, 3, 1),  # leap, month > 2 but leap → same
            ]
        )
        result = datenum.day_of_year_229(dns)
        np.testing.assert_array_equal(result, [61, 59, 61])


# ── Vectorised to_ymd ─────────────────────────────────────────────────────


class TestVectorizedToYmd:
    def test_basic(self):
        dns = np.array(
            [
                datenum.from_ymd(2024, 7, 15, 13, 45, 30, 0),
                datenum.from_ymd(2000, 1, 1, 0, 0, 0, 0),
            ]
        )
        y, m, d, h, mi, s, us = datenum.to_ymd(dns)
        np.testing.assert_array_equal(y, [2024, 2000])
        np.testing.assert_array_equal(m, [7, 1])
        np.testing.assert_array_equal(d, [15, 1])
        np.testing.assert_array_equal(h, [13, 0])
        np.testing.assert_array_equal(mi, [45, 0])
        np.testing.assert_array_equal(s, [30, 0])
        np.testing.assert_array_equal(us, [0, 0])

    def test_microsecond_precision(self):
        dns = np.array([datenum.from_ymd(2024, 1, 1, 0, 0, 0, 123456)])
        _y, _m, _d, _h, _mi, _s, us = datenum.to_ymd(dns)
        np.testing.assert_array_equal(us, [123456])

    def test_max_microsecond(self):
        dns = np.array([datenum.from_ymd(2025, 12, 31, 23, 59, 59, 999999)])
        y, m, d, h, mi, s, us = datenum.to_ymd(dns)
        np.testing.assert_array_equal(y, [2025])
        np.testing.assert_array_equal(m, [12])
        np.testing.assert_array_equal(d, [31])
        np.testing.assert_array_equal(h, [23])
        np.testing.assert_array_equal(mi, [59])
        np.testing.assert_array_equal(s, [59])
        np.testing.assert_array_equal(us, [999999])


# ── Vectorised from_ymd ───────────────────────────────────────────────────


class TestVectorizedFromYmd:
    def test_arrays(self):
        years = np.array([2024, 2000, 2025])
        months = np.array([7, 1, 12])
        days = np.array([15, 1, 31])
        result = datenum.from_ymd(years, months, days)
        expected = np.array(
            [
                datenum.from_ymd(2024, 7, 15),
                datenum.from_ymd(2000, 1, 1),
                datenum.from_ymd(2025, 12, 31),
            ]
        )
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_with_time(self):
        result = datenum.from_ymd(
            np.array([2024]),
            np.array([7]),
            np.array([15]),
            np.array([13]),
            np.array([45]),
            np.array([30]),
            np.array([123456]),
        )
        expected = datenum.from_ymd(2024, 7, 15, 13, 45, 30, 123456)
        np.testing.assert_allclose(result, [expected], atol=1e-6)

    def test_broadcast_scalar_month_day(self):
        result = datenum.from_ymd(np.array([2024, 2025]), 1, 1)
        expected = np.array(
            [datenum.from_ymd(2024, 1, 1), datenum.from_ymd(2025, 1, 1)]
        )
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_broadcast_all_scalars(self):
        """All scalar args should broadcast to a single-element result."""
        result = datenum.from_ymd(2024, 7, 15)
        expected = datenum.from_ymd(np.array([2024]), np.array([7]), np.array([15]))
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_leap_year_feb_29(self):
        result = datenum.from_ymd(
            np.array([2024, 2025]), np.array([2, 2]), np.array([29, 28])
        )
        expected = np.array(
            [datenum.from_ymd(2024, 2, 29), datenum.from_ymd(2025, 2, 28)]
        )
        np.testing.assert_allclose(result, expected, atol=1e-6)


# ── Negative datenums (before epoch) ──────────────────────────────────────


class TestNegativeDatenums:
    def test_component_extraction(self):
        dns = np.array(
            [
                datenum.from_ymd(1999, 12, 31),
                datenum.from_ymd(1970, 1, 1),
            ]
        )
        np.testing.assert_array_equal(datenum.year(dns), [1999, 1970])
        np.testing.assert_array_equal(datenum.month(dns), [12, 1])
        np.testing.assert_array_equal(datenum.day(dns), [31, 1])

    def test_day_of_week_before_epoch(self):
        dns = np.array(
            [
                datenum.from_ymd(1999, 12, 31),  # Friday
                datenum.from_ymd(1999, 12, 30),  # Thursday
            ]
        )
        result = datenum.day_of_week(dns)
        np.testing.assert_array_equal(result, [5, 4])

    def test_day_of_year_before_epoch(self):
        dns = np.array(
            [
                datenum.from_ymd(1999, 1, 1),
                datenum.from_ymd(1999, 12, 31),
            ]
        )
        result = datenum.day_of_year(dns)
        np.testing.assert_array_equal(result, [1, 365])


# ── Scalar regression ─────────────────────────────────────────────────────


class TestScalarRegression:
    """Every function must still return a plain Python scalar for scalar input."""

    def test_year(self):
        assert isinstance(datenum.year(datenum.from_ymd(2024, 7, 15)), int)

    def test_month(self):
        assert isinstance(datenum.month(datenum.from_ymd(2024, 7, 15)), int)

    def test_day(self):
        assert isinstance(datenum.day(datenum.from_ymd(2024, 7, 15)), int)

    def test_hour(self):
        assert isinstance(datenum.hour(datenum.from_ymd(2024, 7, 15, 13)), int)

    def test_minute(self):
        assert isinstance(datenum.minute(datenum.from_ymd(2024, 7, 15, 13, 45)), int)

    def test_second(self):
        assert isinstance(
            datenum.second(datenum.from_ymd(2024, 7, 15, 13, 45, 30)), int
        )

    def test_microsecond(self):
        assert isinstance(
            datenum.microsecond(datenum.from_ymd(2024, 7, 15, 0, 0, 0, 123456)), int
        )

    def test_is_leap_year(self):
        assert isinstance(datenum.is_leap_year(datenum.from_ymd(2024, 1, 1)), bool)

    def test_day_of_year(self):
        assert isinstance(datenum.day_of_year(datenum.from_ymd(2024, 7, 15)), int)

    def test_day_of_week(self):
        assert isinstance(datenum.day_of_week(datenum.from_ymd(2024, 7, 15)), int)

    def test_day_of_month(self):
        assert isinstance(datenum.day_of_month(datenum.from_ymd(2024, 7, 15)), int)

    def test_day_of_year_229(self):
        assert isinstance(datenum.day_of_year_229(datenum.from_ymd(2025, 3, 1)), int)

    def test_from_ymd(self):
        assert isinstance(datenum.from_ymd(2024, 7, 15), float)

    def test_to_ymd(self):
        result = datenum.to_ymd(datenum.from_ymd(2024, 7, 15))
        assert isinstance(result, tuple)
        assert len(result) == 7
        assert all(isinstance(v, int) for v in result)


# ── Epoch integration ─────────────────────────────────────────────────────


class TestEpochIntegration:
    """Vectorised functions must respect the current module-wide epoch."""

    def test_to_datetime_array_respects_epoch(self):
        original = datenum.get_epoch()
        try:
            datenum.set_epoch(dt.datetime(1970, 1, 1))
            result = datenum.to_datetime_array(0.0)
            assert result == dt.datetime(1970, 1, 1)
        finally:
            datenum.set_epoch(original)

    def test_from_datetime_array_respects_epoch(self):
        original = datenum.get_epoch()
        try:
            datenum.set_epoch(dt.datetime(1970, 1, 1))
            result = datenum.from_datetime_array(dt.datetime(1970, 1, 1))
            assert result == 0.0
        finally:
            datenum.set_epoch(original)

    def test_day_of_week_respects_epoch(self):
        original = datenum.get_epoch()
        try:
            datenum.set_epoch(dt.datetime(1970, 1, 1))  # Thursday
            dns = np.array([0.0, 1.0, 2.0])
            result = datenum.day_of_week(dns)
            np.testing.assert_array_equal(result, [4, 5, 6])  # Thu, Fri, Sat
        finally:
            datenum.set_epoch(original)

    def test_from_ymd_respects_epoch(self):
        original = datenum.get_epoch()
        try:
            datenum.set_epoch(dt.datetime(1970, 1, 1))
            result = datenum.from_ymd(np.array([1970]), np.array([1]), np.array([1]))
            np.testing.assert_allclose(result, [0.0], atol=1e-6)
        finally:
            datenum.set_epoch(original)

    def test_day_of_week_agrees_scalar_and_vectorized(self):
        original = datenum.get_epoch()
        try:
            datenum.set_epoch(dt.datetime(1970, 1, 1))  # Thursday
            instants = [
                dt.datetime(1969, 12, 31, 12, 0, 0),  # Wednesday
                dt.datetime(1970, 1, 1, 0, 0, 0),  # Thursday (epoch)
                dt.datetime(1970, 1, 1, 18, 0, 0),  # Thursday
                dt.datetime(2024, 7, 15, 13, 45, 30, 123456),  # Monday
            ]
            expected = [i.isoweekday() for i in instants]
            dns = np.array([datenum.from_datetime(i) for i in instants])
            scalar = [datenum.day_of_week(dn) for dn in dns]
            vectorized = datenum.day_of_week(dns)
            assert scalar == expected
            np.testing.assert_array_equal(vectorized, expected)
        finally:
            datenum.set_epoch(original)
