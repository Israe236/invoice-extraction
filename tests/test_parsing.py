"""Tests for recovering JSON from raw model output.

A VLM that has not been fine-tuned rarely returns clean JSON. It wraps the
object in a code fence, prefixes it with "Here is the extracted data:", or
leaves a trailing comma. Every one of those is recoverable, and recovering them
is the difference between a baseline of 0.0 and an honest baseline.
"""

from invoice_extraction.evaluate import parse_model_output


class TestParseModelOutput:
    def test_clean_json(self):
        assert parse_model_output('{"total": "100.00"}') == {"total": "100.00"}

    def test_code_fence(self):
        raw = '```json\n{"total": "100.00"}\n```'
        assert parse_model_output(raw) == {"total": "100.00"}

    def test_bare_code_fence(self):
        raw = '```\n{"total": "100.00"}\n```'
        assert parse_model_output(raw) == {"total": "100.00"}

    def test_prose_prefix(self):
        raw = 'Here is the extracted data:\n\n{"total": "100.00"}'
        assert parse_model_output(raw) == {"total": "100.00"}

    def test_prose_prefix_and_suffix(self):
        raw = 'Sure! {"total": "100.00"} Let me know if you need anything else.'
        assert parse_model_output(raw) == {"total": "100.00"}

    def test_trailing_comma_is_repaired(self):
        raw = '{"total": "100.00", "tax": "20.00",}'
        assert parse_model_output(raw) == {"total": "100.00", "tax": "20.00"}

    def test_nested_objects_and_arrays(self):
        raw = '{"items": [{"name": "a", "qty": "1", "price": "5.00"}], "total": "5.00"}'
        parsed = parse_model_output(raw)
        assert parsed["items"][0]["name"] == "a"

    def test_braces_inside_strings_do_not_confuse_the_scanner(self):
        raw = '{"items": [{"name": "widget {large}", "qty": "1", "price": "5.00"}]}'
        parsed = parse_model_output(raw)
        assert parsed["items"][0]["name"] == "widget {large}"

    def test_escaped_quote_inside_string(self):
        raw = '{"name": "say \\"hi\\"", "total": "1.00"}'
        assert parse_model_output(raw)["name"] == 'say "hi"'

    def test_truncated_output_returns_none(self):
        # Hit the max_new_tokens ceiling mid-object: unrecoverable, and must be
        # counted as a parse failure rather than silently scored as empty.
        raw = '{"items": [{"name": "a", "qty": "1"'
        assert parse_model_output(raw) is None

    def test_prose_only_returns_none(self):
        raw = "I cannot read this image clearly."
        assert parse_model_output(raw) is None

    def test_single_element_array_is_unwrapped(self):
        # The schema is an object, but a model that wraps it in an array is
        # still unambiguous, so recover it rather than scoring a parse failure.
        assert parse_model_output('[{"total": "1.00"}]') == {"total": "1.00"}

    def test_multi_element_array_is_a_parse_failure(self):
        # Which of the two is the invoice? Unanswerable, so don't guess.
        assert parse_model_output('[{"total": "1.00"}, {"total": "2.00"}]') is None

    def test_empty_and_none(self):
        assert parse_model_output("") is None
        assert parse_model_output(None) is None

    def test_empty_object_is_valid(self):
        assert parse_model_output("{}") == {}
