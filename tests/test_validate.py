"""Tests for the accounting business rules.

The important behaviour under test is the three-way outcome: a rule that cannot
run must report SKIPPED, not FAIL. Conflating the two was a real bug -- it made
every CORD-v2 receipt look like an arithmetic violation because receipts carry
no Moroccan ICE number.
"""

from datetime import date, timedelta

import pytest

from invoice_extraction.validate import (
    Severity,
    Status,
    check_currency_known,
    check_date_valid,
    check_ice_format,
    check_if_format,
    check_items_sum,
    check_required_fields,
    check_tax_arithmetic,
    check_tax_rate_plausible,
    validate,
)


def consistent_invoice(**overrides) -> dict:
    """A well-formed Moroccan invoice: 1000 HT + 20% TVA = 1200 TTC."""
    fields = {
        "items": [
            {"name": "Prestation de conseil", "qty": "2", "price": "600.00"},
            {"name": "Formation professionnelle", "qty": "1", "price": "400.00"},
        ],
        "subtotal": "1000.00",
        "tax": "200.00",
        "total": "1200.00",
        "ice": "001234567000078",
        "if_number": "12345678",
        "invoice_number": "FA-1234",
        "date": "2024-04-03",
        "currency": "MAD",
    }
    fields.update(overrides)
    return fields


class TestTaxArithmetic:
    def test_consistent_invoice_passes(self):
        assert check_tax_arithmetic(consistent_invoice()).status is Status.PASS

    def test_inconsistent_invoice_fails(self):
        result = check_tax_arithmetic(consistent_invoice(total="1500.00"))
        assert result.status is Status.FAIL
        assert result.on_fail is Severity.ERROR

    def test_cent_rounding_is_tolerated(self):
        assert check_tax_arithmetic(consistent_invoice(total="1200.01")).status is Status.PASS

    def test_french_number_format(self):
        fields = consistent_invoice(subtotal="1 000,00", tax="200,00", total="1 200,00")
        assert check_tax_arithmetic(fields).status is Status.PASS

    @pytest.mark.parametrize("missing", ["subtotal", "tax", "total"])
    def test_missing_field_is_skipped_not_failed(self, missing):
        result = check_tax_arithmetic(consistent_invoice(**{missing: ""}))
        assert result.status is Status.SKIPPED
        assert missing in result.detail

    def test_unparseable_field_is_skipped(self):
        assert check_tax_arithmetic(consistent_invoice(tax="N/A")).status is Status.SKIPPED


class TestItemsSum:
    def test_items_summing_to_subtotal_passes(self):
        assert check_items_sum(consistent_invoice()).status is Status.PASS

    def test_items_not_summing_to_subtotal_fails(self):
        fields = consistent_invoice(items=[{"name": "a", "qty": "1", "price": "50.00"}])
        assert check_items_sum(fields).status is Status.FAIL

    def test_no_items_is_skipped(self):
        assert check_items_sum(consistent_invoice(items=[])).status is Status.SKIPPED

    def test_missing_subtotal_is_skipped(self):
        assert check_items_sum(consistent_invoice(subtotal="")).status is Status.SKIPPED

    def test_unparseable_price_is_skipped(self):
        fields = consistent_invoice(
            items=[{"name": "a", "qty": "1", "price": "???"}],
        )
        assert check_items_sum(fields).status is Status.SKIPPED

    def test_tolerance_grows_with_line_count(self):
        # Ten lines each rounded to the cent can drift a few cents in total;
        # that must not be reported as a violation.
        items = [{"name": f"item {i}", "qty": "1", "price": "10.00"} for i in range(10)]
        fields = consistent_invoice(items=items, subtotal="100.05")
        assert check_items_sum(fields).status is Status.PASS


class TestTaxRatePlausible:
    @pytest.mark.parametrize(
        ("rate", "tax"),
        [(20, "200.00"), (14, "140.00"), (10, "100.00"), (7, "70.00")],
    )
    def test_standard_moroccan_rates_pass(self, rate, tax):
        fields = consistent_invoice(subtotal="1000.00", tax=tax)
        assert check_tax_rate_plausible(fields).status is Status.PASS

    def test_nonsense_rate_fails_as_warning(self):
        result = check_tax_rate_plausible(consistent_invoice(tax="374.50"))
        assert result.status is Status.FAIL
        assert result.on_fail is Severity.WARNING

    def test_zero_subtotal_is_skipped(self):
        assert check_tax_rate_plausible(consistent_invoice(subtotal="0")).status is Status.SKIPPED


class TestDateValid:
    def test_valid_past_date_passes(self):
        fields = consistent_invoice(date="2024-04-03")
        assert check_date_valid(fields, today=date(2024, 6, 1)).status is Status.PASS

    def test_printed_french_format_passes(self):
        fields = consistent_invoice(date="03/04/2024")
        assert check_date_valid(fields, today=date(2024, 6, 1)).status is Status.PASS

    def test_future_date_fails(self):
        fields = consistent_invoice(date="2030-01-01")
        result = check_date_valid(fields, today=date(2024, 6, 1))
        assert result.status is Status.FAIL
        assert "future" in result.detail

    def test_absurdly_old_date_fails(self):
        fields = consistent_invoice(date="1990-01-01")
        assert check_date_valid(fields, today=date(2024, 6, 1)).status is Status.FAIL

    def test_boundary_ten_years_is_accepted(self):
        today = date(2024, 6, 1)
        fields = consistent_invoice(date=(today - timedelta(days=365 * 10 - 1)).isoformat())
        assert check_date_valid(fields, today=today).status is Status.PASS

    def test_unparseable_date_fails(self):
        assert check_date_valid(consistent_invoice(date="soon")).status is Status.FAIL

    def test_missing_date_is_skipped(self):
        # CORD-v2 receipts have no date field at all.
        assert check_date_valid(consistent_invoice(date="")).status is Status.SKIPPED


class TestIdentifierFormats:
    def test_valid_ice_passes(self):
        assert check_ice_format(consistent_invoice()).status is Status.PASS

    def test_ice_with_separators_passes(self):
        assert check_ice_format(consistent_invoice(ice="001 234 567 000 078")).status is Status.PASS

    def test_short_ice_fails_as_warning(self):
        result = check_ice_format(consistent_invoice(ice="12345"))
        assert result.status is Status.FAIL
        assert result.on_fail is Severity.WARNING

    def test_missing_ice_is_skipped(self):
        assert check_ice_format(consistent_invoice(ice="")).status is Status.SKIPPED

    def test_valid_if_passes(self):
        assert check_if_format(consistent_invoice()).status is Status.PASS

    def test_overlong_if_fails(self):
        assert check_if_format(consistent_invoice(if_number="1234567890123")).status is Status.FAIL


class TestCurrency:
    def test_mad_passes(self):
        assert check_currency_known(consistent_invoice()).status is Status.PASS

    def test_lowercase_passes(self):
        assert check_currency_known(consistent_invoice(currency="mad")).status is Status.PASS

    def test_unknown_code_fails(self):
        assert check_currency_known(consistent_invoice(currency="XYZ")).status is Status.FAIL

    def test_missing_is_skipped(self):
        assert check_currency_known(consistent_invoice(currency="")).status is Status.SKIPPED


class TestRequiredFields:
    def test_total_present_passes(self):
        assert check_required_fields(consistent_invoice()).status is Status.PASS

    def test_missing_total_fails_as_error(self):
        result = check_required_fields(consistent_invoice(total=""))
        assert result.status is Status.FAIL
        assert result.on_fail is Severity.ERROR


class TestValidateReport:
    def test_consistent_invoice_is_valid_and_ok(self):
        report = validate(consistent_invoice())
        assert report.valid
        assert report.severity is Severity.OK
        assert not report.needs_review
        assert report.unverified == []

    def test_broken_arithmetic_is_an_error(self):
        report = validate(consistent_invoice(total="9999.00"))
        assert not report.valid
        assert report.severity is Severity.ERROR
        assert report.needs_review

    def test_only_format_problems_is_a_warning(self):
        report = validate(consistent_invoice(ice="123"))
        assert not report.valid
        assert report.severity is Severity.WARNING
        assert report.needs_review

    def test_receipt_without_moroccan_fields_is_not_penalised(self):
        """A CORD-style receipt: consistent totals, no ICE/IF/date/currency.

        This is the regression test for the skipped-vs-failed bug. It must come
        out OK, with the unrunnable rules listed as unverified.
        """
        receipt = {
            "items": [{"name": "Coffee", "qty": "2", "price": "50.00"}],
            "subtotal": "50.00",
            "tax": "10.00",
            "total": "60.00",
            "ice": "",
            "if_number": "",
            "invoice_number": "",
            "date": "",
            "currency": "",
        }
        report = validate(receipt)
        assert report.valid
        assert report.severity is Severity.OK
        assert set(report.unverified) == {"date_valid", "ice_format", "if_format", "currency_known"}

    def test_empty_extraction_is_an_error(self):
        report = validate({})
        assert not report.valid
        assert report.severity is Severity.ERROR

    def test_to_dict_is_json_serialisable(self):
        import json

        payload = validate(consistent_invoice()).to_dict()
        json.dumps(payload)  # must not raise
        assert payload["severity"] == "ok"
        assert {c["check"] for c in payload["checks"]} == {
            "required_fields",
            "tax_arithmetic",
            "items_sum",
            "tax_rate_plausible",
            "date_valid",
            "ice_format",
            "if_format",
            "currency_known",
        }
