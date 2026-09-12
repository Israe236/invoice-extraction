"""Diagnose why line-item F1 is low on the synthetic test set.

The hypothesis is that the model reads the P.U. (unit price) column instead of
the Montant (line total) column. That is a claim about the data, so it gets
checked against the saved predictions rather than asserted.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from invoice_extraction.normalize import normalize_text, parse_amount  # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "results"


def classify(predicted_items: list, gold_items: list) -> Counter:
    """Bucket each predicted line against what it could have come from."""
    tally: Counter = Counter()
    gold_by_name = {normalize_text(g.get("name")): g for g in gold_items}

    for item in predicted_items:
        if not isinstance(item, dict):
            tally["not-an-object"] += 1
            continue
        gold = gold_by_name.get(normalize_text(item.get("name")))
        if gold is None:
            tally["name not matched"] += 1
            continue

        predicted_price = parse_amount(item.get("price"))
        gold_price = parse_amount(gold.get("price"))
        gold_qty = parse_amount(gold.get("qty"))
        if predicted_price is None:
            tally["price unparseable"] += 1
        elif gold_price is not None and abs(predicted_price - gold_price) < 0.02:
            tally["price correct (Montant)"] += 1
        elif gold_qty not in (None, 0) and abs(predicted_price - gold_price / gold_qty) < 0.02:
            tally["price = UNIT price (read P.U. column)"] += 1
        else:
            tally["price wrong (other)"] += 1

        predicted_qty = parse_amount(item.get("qty"))
        if predicted_qty is None or gold_qty is None:
            tally["qty missing"] += 1
        elif abs(predicted_qty - gold_qty) < 0.02:
            tally["qty correct"] += 1
        else:
            tally["qty wrong"] += 1
    return tally


def main() -> None:
    path = RESULTS / "base_synthetic_predictions.json"
    records = json.loads(path.read_text(encoding="utf-8"))

    tally: Counter = Counter()
    n_with_items = 0
    for record in records:
        parsed = record.get("parsed")
        if not parsed:
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

    print("\n--- one example ---")
    for record in records:
        parsed = record.get("parsed")
        if parsed and isinstance(parsed.get("items"), list) and parsed["items"]:
            print("predicted:", json.dumps(parsed["items"][:3], ensure_ascii=False))
            print("gold     :", json.dumps(record["ground_truth"]["items"][:3], ensure_ascii=False))
            break


if __name__ == "__main__":
    main()
