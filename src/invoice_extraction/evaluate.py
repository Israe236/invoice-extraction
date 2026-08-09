"""Per-field scoring for extracted JSON, tolerant of malformed model output."""

import json
import re
from dataclasses import dataclass

from invoice_extraction.data import ExtractedFields

FIELD_NAMES = ("subtotal", "tax", "total", "items", "ice", "if_number")

_CODE_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$")


@dataclass
class FieldScore:
    tp: int = 0
    fp: int = 0
    fn: int = 0

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


def parse_model_output(text: str) -> dict | None:
    text = _CODE_FENCE.sub("", text.strip()).strip()
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    parsed = json.loads(text[start : i + 1])
                    return parsed if isinstance(parsed, dict) else None
                except json.JSONDecodeError:
                    return None
    return None


def _normalize(value: object) -> str:
    return str(value).strip() if value is not None else ""


def _field_set(value: object) -> set[str]:
    v = _normalize(value)
    return {v} if v else set()


def _items_set(items: object) -> set[tuple[str, str, str]]:
    if not isinstance(items, list):
        return set()
    result = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        name = _normalize(item.get("name"))
        qty = _normalize(item.get("qty"))
        price = _normalize(item.get("price"))
        if name or qty or price:
            result.add((name, qty, price))
    return result


def _update(score: FieldScore, predicted: set, truth: set) -> None:
    score.tp += len(predicted & truth)
    score.fp += len(predicted - truth)
    score.fn += len(truth - predicted)


def new_scores() -> dict[str, FieldScore]:
    return {name: FieldScore() for name in FIELD_NAMES}


def score_example(
    prediction: dict | None, truth: ExtractedFields, scores: dict[str, FieldScore]
) -> None:
    prediction = prediction or {}
    for name in ("subtotal", "tax", "total", "ice", "if_number"):
        _update(scores[name], _field_set(prediction.get(name)), _field_set(truth[name]))
    _update(scores["items"], _items_set(prediction.get("items")), _items_set(truth["items"]))


def score_dataset(raw_predictions: list[str], truths: list[ExtractedFields]) -> dict[str, FieldScore]:
    scores = new_scores()
    for raw_prediction, truth in zip(raw_predictions, truths):
        score_example(parse_model_output(raw_prediction), truth, scores)
    return scores


def format_report(scores: dict[str, FieldScore]) -> str:
    header = f"{'field':10s} {'precision':>9s} {'recall':>9s} {'f1':>9s}"
    rows = [f"{name:10s} {s.precision:9.3f} {s.recall:9.3f} {s.f1:9.3f}" for name, s in scores.items()]
    return "\n".join([header, *rows])
