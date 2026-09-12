"""Diagnose line-item errors on the synthetic test set.

    python scripts/inspect_items.py                                  # base model
    python scripts/inspect_items.py finetuned_synthetic_predictions  # adapter

The question is whether a wrong line price came from the P.U. (unit price)
column instead of the Montant (line total) column. That is a claim about the
data, so it is checked against the saved predictions rather than asserted.

Predicted lines are paired with gold lines one-to-one. An earlier version keyed
gold lines by product name, which silently collapsed two lines selling the same
product into one: the second predicted line was then compared against the
wrong gold line and counted as an error. That bug inflated "price wrong" for
both models, and produced 22 phantom errors on a prediction set whose line
items were in fact all correct. Pairing now picks, for each predicted line, the
best still-unused gold line with the same name.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from invoice_extraction.normalize import normalize_text, parse_amount  # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "results"
TOLERANCE = 0.02


def _close(a: float | None, b: float | None) -> bool:
    return a is not None and b is not None and abs(a - b) < TOLERANCE


def _price_outcome(predicted_price: float | None, gold: dict) -> str:
    gold_price = parse_amount(gold.get("price"))
    gold_qty = parse_amount(gold.get("qty"))
    if predicted_price is None:
        return "price unparseable"
    if _close(predicted_price, gold_price):
        return "price correct (Montant)"
    if gold_qty not in (None, 0) and gold_price is not None and _close(
        predicted_price, gold_price / gold_qty
    ):
        return "price = UNIT price (read P.U. column)"
    return "price wrong (other)"


def _match_score(item: dict, gold: dict) -> int:
    """Higher is a better pairing: exact price beats unit price beats qty only."""
    outcome = _price_outcome(parse_amount(item.get("price")), gold)
    score = {"price correct (Montant)": 3, "price = UNIT price (read P.U. column)": 2}.get(
        outcome, 0
    )
    if _close(parse_amount(item.get("qty")), parse_amount(gold.get("qty"))):
        score += 1
    return score


def classify(predicted_items: list, gold_items: list) -> Counter:
    """Pair each predicted line with one unused gold line, then bucket it."""
    tally: Counter = Counter()
    unused: dict[str, list[dict]] = {}
    for gold in gold_items:
        unused.setdefault(normalize_text(gold.get("name")), []).append(gold)

    for item in predicted_items:
        if not isinstance(item, dict):
            tally["not-an-object"] += 1
            continue
        candidates = unused.get(normalize_text(item.get("name")), [])
        if not candidates:
            tally["name not matched"] += 1
            continue
        gold = max(candidates, key=lambda g: _match_score(item, g))
        candidates.remove(gold)

        tally[_price_outcome(parse_amount(item.get("price")), gold)] += 1

        predicted_qty = parse_amount(item.get("qty"))
        gold_qty = parse_amount(gold.get("qty"))
        if predicted_qty is None or gold_qty is None:
            tally["qty missing"] += 1
        elif _close(predicted_qty, gold_qty):
            tally["qty correct"] += 1
        else:
            tally["qty wrong"] += 1

    tally["gold lines never predicted"] += sum(len(v) for v in unused.values())
    return tally


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "base_synthetic_predictions"
    path = RESULTS / f"{name}.json"
    print(f"diagnosing {path.name}")
    records = json.loads(path.read_text(encoding="utf-8"))

    tally: Counter = Counter()
    n_with_items = 0
    for record in records:
        parsed = record.get("parsed")
        if not parsed:
            tally["document unparseable"] += 1
            continue
        predicted_items = parsed.get("items")
        if not isinstance(predicted_items, list):
            tally["items not a list"] += 1
            continue
        n_with_items += 1
        tally += classify(predicted_items, record["ground_truth"]["items"])

    print(f"{len(records)} examples, {n_with_items} with a parseable items list\n")
    for label, count in tally.most_common():
        print(f"  {count:4d}  {label}")


if __name__ == "__main__":
    main()
