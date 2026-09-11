"""Per-field scoring for extracted JSON, tolerant of malformed model output.

Scoring is **field-aware**. Comparing raw strings punishes the model for things
no accountant would call a mistake -- `1 234,56` against `1234.56`, `03/04/2024`
against `2024-04-03`, `Développement` against `Developpement`. Every field is
routed through the normaliser appropriate to its type before comparison, and
the same normaliser is used for the base model and the fine-tuned model, so the
comparison between them stays fair.

Two quantities are reported that a plain F1 table hides:

- **parse rate** -- how often the model emitted JSON at all. A model that is
  accurate when it parses and emits prose the rest of the time is not usable,
  and this is exactly where a small untrained VLM fails.
- **support** -- how many gold values a field actually had. CORD-v2 receipts
  carry no ICE, so "ICE F1 = 0.000" on CORD means "never tested", not "always
  wrong". Fields with zero support are printed as `n/a`.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass

from invoice_extraction.data import ExtractedFields
from invoice_extraction.normalize import normalize_date, normalize_text, parse_amount

FIELD_NAMES = (
    "subtotal",
    "tax",
    "total",
    "items",
    "ice",
    "if_number",
    "invoice_number",
    "date",
    "currency",
)

# Fields compared as money, as dates, as bare digit strings, as free text.
_AMOUNT_FIELDS = ("subtotal", "tax", "total")
_DATE_FIELDS = ("date",)
_DIGIT_FIELDS = ("ice", "if_number")
_TEXT_FIELDS = ("invoice_number", "currency")

_CODE_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$")
_TRAILING_COMMA = re.compile(r",(\s*[}\]])")


@dataclass
class FieldScore:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def support(self) -> int:
        """Number of gold values for this field. Zero means the field was never
        present in the reference data, so precision/recall are meaningless."""
        return self.tp + self.fn

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0


@dataclass
class EvalReport:
    scores: dict[str, FieldScore]
    n_examples: int = 0
    n_parsed: int = 0

    @property
    def parse_rate(self) -> float:
        return self.n_parsed / self.n_examples if self.n_examples else 0.0

    @property
    def micro_f1(self) -> float:
        """Micro-average over every field that had any gold value."""
        tp = sum(s.tp for s in self.scores.values())
        fp = sum(s.fp for s in self.scores.values())
        fn = sum(s.fn for s in self.scores.values())
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def to_dict(self) -> dict:
        return {
            "n_examples": self.n_examples,
            "n_parsed": self.n_parsed,
            "parse_rate": round(self.parse_rate, 4),
            "micro_f1": round(self.micro_f1, 4),
            "fields": {
                name: {
                    "precision": round(s.precision, 4),
                    "recall": round(s.recall, 4),
                    "f1": round(s.f1, 4),
                    "support": s.support,
                }
                for name, s in self.scores.items()
            },
        }


def parse_model_output(text: str) -> dict | None:
    """Recover a JSON object from whatever the model produced.

    Tried in order: the text as-is, the text with a code fence stripped, the
    first balanced `{...}` span, and finally that span with trailing commas
    removed. Anything else counts as a parse failure and is reported.
    """
    if not text:
        return None

    stripped = _CODE_FENCE.sub("", text.strip()).strip()

    # If the output is valid JSON in its entirety, that reading wins outright.
    # Falling through to the brace scanner here would let a multi-object array
    # be silently reduced to its first element.
    for candidate in (text.strip(), stripped):
        if not candidate:
            continue
        try:
            return _as_invoice_object(json.loads(candidate))
        except json.JSONDecodeError:
            continue

    # Otherwise the JSON is embedded in prose, or lightly malformed.
    span = _first_balanced_object(stripped)
    if not span:
        return None
    for candidate in (span, _TRAILING_COMMA.sub(r"\1", span)):
        try:
            return _as_invoice_object(json.loads(candidate))
        except json.JSONDecodeError:
            continue
    return None


def _as_invoice_object(parsed: object) -> dict | None:
    """Accept an object, or a single-element array wrapping one.

    A model that wraps the invoice in an array is still unambiguous. An array
    of several objects is not -- which one is the invoice? -- so that counts as
    a parse failure rather than a guess.
    """
    if isinstance(parsed, dict):
        return parsed
    if isinstance(parsed, list) and len(parsed) == 1 and isinstance(parsed[0], dict):
        return parsed[0]
    return None


def _first_balanced_object(text: str) -> str | None:
    """The first `{...}` span with balanced braces, ignoring braces in strings."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _canonical(field: str, value: object) -> str:
    """Reduce a field's value to the form used for equality comparison."""
    if value is None:
        return ""
    if field in _AMOUNT_FIELDS:
        amount = parse_amount(value)
        # Round to cents so 1234.5 and 1234.50 are the same value.
        return f"{amount:.2f}" if amount is not None else ""
    if field in _DATE_FIELDS:
        return normalize_date(value)
    if field in _DIGIT_FIELDS:
        return "".join(ch for ch in str(value) if ch.isdigit())
    if field in _TEXT_FIELDS:
        # Invoice numbers differ only by separators and case in practice.
        return re.sub(r"[^a-z0-9]", "", normalize_text(value))
    return normalize_text(value)


def _field_bag(field: str, value: object) -> Counter:
    canonical = _canonical(field, value)
    return Counter([canonical]) if canonical else Counter()


def _items_bag(items: object) -> Counter:
    """Line items as a multiset, so two identical lines count twice."""
    if not isinstance(items, list):
        return Counter()
    bag: Counter = Counter()
    for item in items:
        if not isinstance(item, dict):
            continue
        name = normalize_text(item.get("name"))
        qty = _canonical("subtotal", item.get("qty")) if item.get("qty") not in (None, "") else ""
        price = _canonical("subtotal", item.get("price"))
        if name or qty or price:
            bag[(name, qty, price)] += 1
    return bag


def _update(score: FieldScore, predicted: Counter, truth: Counter) -> None:
    overlap = predicted & truth
    matched = sum(overlap.values())
    score.tp += matched
    score.fp += sum(predicted.values()) - matched
    score.fn += sum(truth.values()) - matched


def new_scores() -> dict[str, FieldScore]:
    return {name: FieldScore() for name in FIELD_NAMES}


def score_example(
    prediction: dict | None, truth: ExtractedFields, scores: dict[str, FieldScore]
) -> None:
    prediction = prediction or {}
    for name in FIELD_NAMES:
        if name == "items":
            _update(scores[name], _items_bag(prediction.get("items")), _items_bag(truth["items"]))
        else:
            _update(
                scores[name],
                _field_bag(name, prediction.get(name)),
                _field_bag(name, truth.get(name)),
            )


def score_dataset(raw_predictions: list[str], truths: list[ExtractedFields]) -> EvalReport:
    report = EvalReport(scores=new_scores())
    # strict=True: a length mismatch between predictions and references would
    # silently truncate the evaluation and report a number for fewer examples
    # than claimed. Better to crash.
    for raw_prediction, truth in zip(raw_predictions, truths, strict=True):
        parsed = parse_model_output(raw_prediction)
        report.n_examples += 1
        report.n_parsed += parsed is not None
        score_example(parsed, truth, report.scores)
    return report


def format_report(report: EvalReport | dict[str, FieldScore]) -> str:
    """Human-readable table. Fields with no gold values are shown as `n/a`."""
    if isinstance(report, dict):  # tolerate a bare score dict
        report = EvalReport(scores=report)

    lines = [f"{'field':15s} {'precision':>9s} {'recall':>9s} {'f1':>9s} {'support':>8s}"]
    for name, s in report.scores.items():
        if s.support == 0:
            lines.append(f"{name:15s} {'n/a':>9s} {'n/a':>9s} {'n/a':>9s} {0:8d}")
        else:
            lines.append(
                f"{name:15s} {s.precision:9.3f} {s.recall:9.3f} {s.f1:9.3f} {s.support:8d}"
            )
    if report.n_examples:
        lines.append("")
        lines.append(
            f"{'micro F1':15s} {report.micro_f1:9.3f}"
            f"   parse rate {report.n_parsed}/{report.n_examples} "
            f"({report.parse_rate * 100:.1f}%)"
        )
    return "\n".join(lines)
