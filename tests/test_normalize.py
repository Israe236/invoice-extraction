"""Tests for the shared number / date / text normalisers.

These are the foundation of both the metric and the validation layer, so a bug
here silently corrupts every reported number.
"""

from datetime import date

import pytest

from invoice_extraction.normalize import (
    amounts_equal,
    normalize_date,
    normalize_ice,
    normalize_text,
    parse_amount,
    parse_date,
)


class TestParseAmount:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("1234.56", 1234.56),
            ("1234", 1234.0),
            ("0", 0.0),
            ("0.00", 0.0),
        ],
    )
    def test_plain_numbers(self, raw, expected):
        assert parse_amount(raw) == pytest.approx(expected)

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("1 234,56", 1234.56),  # French: space thousands, comma decimal
            ("1.234,56", 1234.56),  # dot thousands, comma decimal
            ("1,234.56", 1234.56),  # anglo
            ("1 234 567,89", 1234567.89),
            ("12,50", 12.50),
        ],
    )
    def test_separator_conventions(self, raw, expected):
        assert parse_amount(raw) == pytest.approx(expected)

    def test_three_digit_group_is_thousands_not_decimals(self):
        # "1,500" on an invoice is one thousand five hundred, not 1.5.
        assert parse_amount("1,500") == pytest.approx(1500.0)
        assert parse_amount("1.500") == pytest.approx(1500.0)

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("1234.56 MAD", 1234.56),
            ("MAD 1234.56", 1234.56),
            ("1 234,56 DH", 1234.56),
            ("€1.234,56", 1234.56),
            ("1234.56$", 1234.56),
        ],
    )
    def test_strips_currency(self, raw, expected):
        assert parse_amount(raw) == pytest.approx(expected)

    def test_non_breaking_space_thousands(self):
        assert parse_amount("1 234,56") == pytest.approx(1234.56)

    def test_negative(self):
        assert parse_amount("-1 234,56") == pytest.approx(-1234.56)

    @pytest.mark.parametrize("raw", ["", None, "   ", "abc", "N/A", "-", "MAD"])
    def test_unparseable_returns_none(self, raw):
        assert parse_amount(raw) is None

    def test_passes_through_real_numbers(self):
        assert parse_amount(12.5) == pytest.approx(12.5)
        assert parse_amount(12) == pytest.approx(12.0)

    def test_bool_is_not_a_number(self):
        # bool is a subclass of int; treating True as 1.0 would silently accept
        # garbage from a model that emitted a boolean where money was expected.
        assert parse_amount(True) is None


class TestAmountsEqual:
    def test_exact(self):
        assert amounts_equal("1234.56", "1 234,56")

    def test_cent_rounding_tolerated(self):
        assert amounts_equal("100.00", "100.01")

    def test_relative_tolerance_on_large_amounts(self):
        # A 1-cent absolute tolerance would be absurd at this magnitude.
        assert amounts_equal("1000000.00", "1000100.00")

    def test_clearly_different_amounts_are_not_equal(self):
        assert not amounts_equal("100.00", "110.00")

    def test_unparseable_is_never_equal(self):
        assert not amounts_equal("abc", "100.00")
        assert not amounts_equal(None, None)


class TestParseDate:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("2024-04-03", date(2024, 4, 3)),
            ("2024/04/03", date(2024, 4, 3)),
            ("03/04/2024", date(2024, 4, 3)),  # day-first
            ("03-04-2024", date(2024, 4, 3)),
            ("03.04.2024", date(2024, 4, 3)),
            ("3 April 2024", date(2024, 4, 3)),
        ],
    )
    def test_formats(self, raw, expected):
        assert parse_date(raw) == expected

    def test_iso_is_not_misread_as_day_first(self):
        # 2024-04-03 must not be parsed by the %d/%m/%Y branch.
        assert parse_date("2024-04-03") == date(2024, 4, 3)

    def test_day_first_disambiguation(self):
        # On a French/Moroccan invoice this is 5 March, not 3 May.
        assert parse_date("05/03/2024") == date(2024, 3, 5)

    @pytest.mark.parametrize("raw", ["", None, "not a date", "32/13/2024", "0000-00-00"])
    def test_unparseable_returns_none(self, raw):
        assert parse_date(raw) is None

    def test_normalize_date_renders_iso(self):
        assert normalize_date("03/04/2024") == "2024-04-03"
        assert normalize_date("garbage") == ""


class TestNormalizeText:
    def test_strips_accents_and_case(self):
        assert normalize_text("Développement") == normalize_text("developpement")

    def test_collapses_whitespace(self):
        assert normalize_text("  Prestation   de  conseil ") == "prestation de conseil"

    def test_none_is_empty(self):
        assert normalize_text(None) == ""


class TestNormalizeIce:
    def test_strips_formatting(self):
        assert normalize_ice("001 234 567 000 078") == "001234567000078"
        assert normalize_ice("ICE: 001234567000078") == "001234567000078"

    def test_none_is_empty(self):
        assert normalize_ice(None) == ""
