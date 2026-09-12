"""Tests for CORD-v2 ground-truth mapping and synthetic invoice generation.

No network and no GPU: `extract_fields` is tested against hand-written
`gt_parse` payloads shaped like the real ones, and the synthetic generator is
tested without rendering (rendering pulls in WeasyPrint and is covered by
`test_synthetic_render.py`, which is skipped when those libraries are absent).
"""

import pytest

from invoice_extraction.data import EXTRACTION_PROMPT, extract_fields, to_chat_example
from invoice_extraction.evaluate import FIELD_NAMES
from invoice_extraction.normalize import parse_amount
from invoice_extraction.synthetic import generate_invoice_data, to_ground_truth
from invoice_extraction.validate import Severity, validate


class TestExtractFieldsFromCord:
    def test_typical_receipt(self):
        gt_parse = {
            "menu": [
                {"nm": "Cappuccino", "cnt": "2", "price": "50,000"},
                {"nm": "Croissant", "cnt": "1", "price": "25,000"},
            ],
            "sub_total": {"subtotal_price": "75,000", "tax_price": "7,500"},
            "total": {"total_price": "82,500"},
        }
        fields = extract_fields(gt_parse)
        assert len(fields["items"]) == 2
        assert fields["items"][0] == {"name": "Cappuccino", "qty": "2", "price": "50,000"}
        assert fields["subtotal"] == "75,000"
        assert fields["total"] == "82,500"

    def test_moroccan_fields_are_empty_not_guessed(self):
        fields = extract_fields({"total": {"total_price": "100"}})
        for name in ("ice", "if_number", "invoice_number", "date", "currency"):
            assert fields[name] == "", f"{name} must be empty for a CORD receipt"

    def test_schema_is_complete(self):
        """Every scored field must exist on the mapped record."""
        fields = extract_fields({})
        assert set(fields) == set(FIELD_NAMES)

    def test_single_dict_instead_of_list(self):
        # CORD sometimes stores a one-item menu as a dict rather than a list.
        gt_parse = {"menu": {"nm": "Tea", "cnt": "1", "price": "10,000"}}
        assert len(extract_fields(gt_parse)["items"]) == 1

    def test_repeated_sub_total_blocks_are_merged(self):
        gt_parse = {
            "sub_total": [{"subtotal_price": "75,000"}, {"tax_price": "7,500"}],
            "total": {"total_price": "82,500"},
        }
        fields = extract_fields(gt_parse)
        assert fields["subtotal"] == "75,000"
        assert fields["tax"] == "7,500"

    def test_list_valued_name_is_joined(self):
        gt_parse = {"menu": [{"nm": ["Iced", "Latte"], "cnt": "1", "price": "30,000"}]}
        assert extract_fields(gt_parse)["items"][0]["name"] == "Iced Latte"

    def test_item_without_a_name_is_dropped(self):
        gt_parse = {"menu": [{"cnt": "1", "price": "10,000"}]}
        assert extract_fields(gt_parse)["items"] == []

    def test_empty_gt_parse_does_not_raise(self):
        fields = extract_fields({})
        assert fields["items"] == []
        assert fields["total"] == ""


class TestChatExample:
    def test_structure(self):
        fields = extract_fields({"total": {"total_price": "100"}})
        messages = to_chat_example(image="<img>", fields=fields)
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert messages[0]["content"][1]["text"] == EXTRACTION_PROMPT

    def test_assistant_turn_is_valid_json_of_the_schema(self):
        import json

        fields = extract_fields({"total": {"total_price": "100"}})
        messages = to_chat_example(image="<img>", fields=fields)
        payload = json.loads(messages[1]["content"][0]["text"])
        assert set(payload) == set(FIELD_NAMES)


class TestSyntheticGenerator:
    def test_is_deterministic_for_a_seed(self):
        a = generate_invoice_data(seed=7)
        b = generate_invoice_data(seed=7)
        assert a == b

    def test_different_seeds_give_different_invoices(self):
        assert generate_invoice_data(seed=1) != generate_invoice_data(seed=2)

    @pytest.mark.parametrize("seed", range(20))
    def test_generated_invoices_are_internally_consistent(self, seed):
        """Ground truth must satisfy its own business rules, by construction.

        If the generator can emit an invoice that fails validation, then a
        model trained on it is being taught to produce invalid accounting.
        """
        invoice = generate_invoice_data(seed=seed)
        report = validate(dict(to_ground_truth(invoice)))
        assert report.severity is Severity.OK, report.to_dict()

    @pytest.mark.parametrize("seed", range(20))
    def test_totals_arithmetic(self, seed):
        invoice = generate_invoice_data(seed=seed)
        subtotal = parse_amount(invoice.subtotal)
        tax = parse_amount(invoice.tax)
        total = parse_amount(invoice.total)
        assert subtotal + tax == pytest.approx(total, abs=0.01)

    @pytest.mark.parametrize("seed", range(20))
    def test_line_items_sum_to_subtotal(self, seed):
        invoice = generate_invoice_data(seed=seed)
        line_sum = sum(parse_amount(item["price"]) for item in invoice.items)
        assert line_sum == pytest.approx(parse_amount(invoice.subtotal), abs=0.05)

    @pytest.mark.parametrize("seed", range(10))
    def test_unit_price_column_is_consistent_with_line_total(self, seed):
        """The P.U. distractor column has to be arithmetically real, or the
        model would be trained against a contradiction it cannot resolve."""
        invoice = generate_invoice_data(seed=seed)
        for item, unit_price in zip(invoice.items, invoice.unit_prices, strict=True):
            expected = parse_amount(unit_price) * float(item["qty"])
            assert expected == pytest.approx(parse_amount(item["price"]), abs=0.02)

    def test_ground_truth_covers_every_scored_field(self):
        fields = to_ground_truth(generate_invoice_data(seed=3))
        assert set(fields) == set(FIELD_NAMES)
        for name in ("ice", "if_number", "invoice_number", "date", "currency"):
            assert fields[name], f"{name} must be populated on a synthetic invoice"

    def test_ice_is_fifteen_digits(self):
        assert len(generate_invoice_data(seed=5).ice) == 15

    def test_ground_truth_date_is_iso(self):
        fields = to_ground_truth(generate_invoice_data(seed=5))
        assert len(fields["date"]) == 10 and fields["date"][4] == "-"

    def test_printed_date_is_day_first(self):
        invoice = generate_invoice_data(seed=5)
        day, month, year = invoice.invoice_date.split("/")
        assert f"{year}-{month}-{day}" == invoice.invoice_date_iso

    def test_currency_is_mad(self):
        assert generate_invoice_data(seed=5).currency == "MAD"

    @pytest.mark.parametrize("seed", range(20))
    def test_dates_come_from_the_fixed_window(self, seed):
        """Regression test: dates used to be drawn relative to today, so the same
        seed produced a different invoice on a different day."""
        from datetime import date

        from invoice_extraction.synthetic import DATE_WINDOW_END, DATE_WINDOW_START

        invoice_date = date.fromisoformat(generate_invoice_data(seed=seed).invoice_date_iso)
        assert DATE_WINDOW_START <= invoice_date <= DATE_WINDOW_END

    def test_held_out_seed_produces_a_pinned_date(self):
        """Pin the exact date for the first held-out seed.

        A window test alone would not have caught the original bug: a date
        shifted by one day still sits inside a two-year window. Pinning the value
        fails the moment a seed stops producing the same document.
        """
        assert generate_invoice_data(seed=900_000).invoice_date_iso == "2026-05-01"
