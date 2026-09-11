"""Tests for the per-field scoring metric.

The metric is the thing the whole project's claims rest on, so it gets tested
like production code. Two properties matter most:

1. A perfect prediction scores 1.0 -- including when it is formatted
   differently to the reference (`1 234,56` vs `1234.56`).
2. A field with no gold values reports support 0, so it can be printed as
   `n/a` instead of a misleading 0.000.
"""

import json

from invoice_extraction.data import ExtractedFields
from invoice_extraction.evaluate import (
    FIELD_NAMES,
    EvalReport,
    format_report,
    new_scores,
    score_dataset,
    score_example,
)


def gold(**overrides) -> ExtractedFields:
    fields = ExtractedFields(
        items=[
            {"name": "Prestation de conseil", "qty": "2", "price": "600.00"},
            {"name": "Formation professionnelle", "qty": "1", "price": "400.00"},
        ],
        subtotal="1000.00",
        tax="200.00",
        total="1200.00",
        ice="001234567000078",
        if_number="12345678",
        invoice_number="FA-1234",
        date="2024-04-03",
        currency="MAD",
    )
    fields.update(overrides)
    return fields


class TestPerfectPrediction:
    def test_identical_prediction_scores_one(self):
        truth = gold()
        scores = new_scores()
        score_example(dict(truth), truth, scores)
        for name in FIELD_NAMES:
            assert scores[name].f1 == 1.0, name

    def test_differently_formatted_prediction_still_scores_one(self):
        """The model writes French numbers and a printed date; both are right."""
        truth = gold()
        prediction = {
            "items": [
                {"name": "Prestation de conseil", "qty": "2", "price": "600,00"},
                {"name": "Formation professionnelle", "qty": "1", "price": "400,00"},
            ],
            "subtotal": "1 000,00",
            "tax": "200,00",
            "total": "1 200,00 MAD",
            "ice": "001 234 567 000 078",
            "if_number": "12345678",
            "invoice_number": "FA 1234",
            "date": "03/04/2024",
            "currency": "mad",
        }
        scores = new_scores()
        score_example(prediction, truth, scores)
        for name in FIELD_NAMES:
            assert scores[name].f1 == 1.0, f"{name} should tolerate formatting differences"

    def test_accented_item_name_matches_unaccented(self):
        truth = gold(items=[{"name": "Développement logiciel", "qty": "1", "price": "100.00"}])
        prediction = {"items": [{"name": "Developpement logiciel", "qty": "1", "price": "100.00"}]}
        scores = new_scores()
        score_example(prediction, truth, scores)
        assert scores["items"].f1 == 1.0


class TestWrongPrediction:
    def test_wrong_amount_is_a_miss(self):
        scores = new_scores()
        score_example({"total": "999.00"}, gold(), scores)
        assert scores["total"].tp == 0
        assert scores["total"].fp == 1
        assert scores["total"].fn == 1

    def test_missing_field_is_a_false_negative_only(self):
        scores = new_scores()
        score_example({}, gold(), scores)
        assert scores["total"].fn == 1
        assert scores["total"].fp == 0

    def test_hallucinated_field_is_a_false_positive_only(self):
        truth = gold(ice="")
        scores = new_scores()
        score_example({"ice": "999888777666555"}, truth, scores)
        assert scores["ice"].fp == 1
        assert scores["ice"].fn == 0

    def test_none_prediction_scores_all_false_negatives(self):
        scores = new_scores()
        score_example(None, gold(), scores)
        assert scores["total"].fn == 1
        assert scores["items"].fn == 2


class TestLineItems:
    def test_partial_item_match(self):
        truth = gold()
        prediction = {
            "items": [
                {"name": "Prestation de conseil", "qty": "2", "price": "600.00"},
                {"name": "Wrong item", "qty": "9", "price": "1.00"},
            ]
        }
        scores = new_scores()
        score_example(prediction, truth, scores)
        assert scores["items"].tp == 1
        assert scores["items"].fp == 1
        assert scores["items"].fn == 1

    def test_duplicate_lines_are_counted_twice(self):
        """Two identical lines on an invoice are two lines, not one.

        A set-based comparison would collapse them and silently award a perfect
        score to a prediction that dropped one.
        """
        truth = gold(
            items=[
                {"name": "Abonnement mensuel", "qty": "1", "price": "100.00"},
                {"name": "Abonnement mensuel", "qty": "1", "price": "100.00"},
            ]
        )
        prediction = {"items": [{"name": "Abonnement mensuel", "qty": "1", "price": "100.00"}]}
        scores = new_scores()
        score_example(prediction, truth, scores)
        assert scores["items"].tp == 1
        assert scores["items"].fn == 1

    def test_item_ordering_does_not_matter(self):
        truth = gold()
        prediction = {"items": list(reversed(gold()["items"]))}
        scores = new_scores()
        score_example(prediction, truth, scores)
        assert scores["items"].f1 == 1.0

    def test_non_list_items_scores_nothing(self):
        scores = new_scores()
        score_example({"items": "two coffees"}, gold(), scores)
        assert scores["items"].tp == 0
        assert scores["items"].fn == 2


class TestSupport:
    def test_field_absent_from_gold_has_zero_support(self):
        """CORD-v2 has no ICE. Support 0 means 'never tested', not 'always wrong'."""
        truth = gold(ice="", if_number="", invoice_number="", date="", currency="")
        scores = new_scores()
        score_example({"total": "1200.00"}, truth, scores)
        assert scores["ice"].support == 0
        assert scores["total"].support == 1

    def test_report_prints_na_for_unsupported_fields(self):
        truth = gold(ice="")
        report = score_dataset([json.dumps(dict(truth))], [truth])
        text = format_report(report)
        assert "n/a" in text
        assert "ice" in text


class TestScoreDataset:
    def test_parse_rate_counts_failures(self):
        truth = gold()
        predictions = [
            json.dumps(dict(truth)),
            "I cannot read this image.",
            '```json\n' + json.dumps(dict(truth)) + '\n```',
        ]
        report = score_dataset(predictions, [truth, truth, truth])
        assert report.n_examples == 3
        assert report.n_parsed == 2
        assert report.parse_rate == 2 / 3

    def test_micro_f1_on_perfect_predictions(self):
        truth = gold()
        report = score_dataset([json.dumps(dict(truth))] * 5, [truth] * 5)
        assert report.micro_f1 == 1.0

    def test_report_is_json_serialisable(self):
        truth = gold()
        report = score_dataset([json.dumps(dict(truth))], [truth])
        json.dumps(report.to_dict())  # must not raise

    def test_empty_report_does_not_divide_by_zero(self):
        report = EvalReport(scores=new_scores())
        assert report.parse_rate == 0.0
        assert report.micro_f1 == 0.0
        format_report(report)  # must not raise
