"""Does the validation flag actually predict whether an extraction is right?

    python scripts/validation_value.py

The validation layer's whole argument is that "severity: ok" is a signal you
can act on without a reference answer. That is a testable claim, so test it:
on the held-out sets we *do* have gold labels, so for every prediction compare
what validation said against whether the totals were actually correct.

"Totals correct" means subtotal, tax and total all match gold within a cent.
Only documents whose gold has all three are counted, since the rules cannot be
judged on a document that does not carry them.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from invoice_extraction.normalize import amounts_equal, parse_amount  # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "results"
AMOUNTS = ("subtotal", "tax", "total")


def totals_correct(parsed: dict, gold: dict) -> bool:
    return all(amounts_equal(parsed.get(f), gold.get(f)) for f in AMOUNTS)


def analyse(name: str) -> None:
    path = RESULTS / f"{name}.json"
    if not path.exists():
        print(f"{name}: missing")
        return
    records = json.loads(path.read_text(encoding="utf-8"))

    severity_counts: Counter = Counter()
    # (validation said ok?, totals actually correct?) -> count
    confusion: Counter = Counter()
    eligible = 0

    for record in records:
        parsed = record.get("parsed")
        if not parsed or not record.get("validation"):
            severity_counts["unparseable"] += 1
            continue
        severity = record["validation"]["severity"]
        severity_counts[severity] += 1

        gold = record["ground_truth"]
        if any(parse_amount(gold.get(f)) is None for f in AMOUNTS):
            continue
        eligible += 1
        # "Flagged" means validation raised an error: the arithmetic contradicts
        # itself. Warnings are format problems and do not bear on the totals.
        flagged = severity == "error"
        confusion[(flagged, totals_correct(parsed, gold))] += 1

    print(f"=== {name} ({len(records)} documents) ===")
    print("  severity:", dict(severity_counts))
    if not eligible:
        print("  no documents with all three gold amounts")
        return

    passed_ok = confusion[(False, True)]
    passed_wrong = confusion[(False, False)]
    flagged_wrong = confusion[(True, False)]
    flagged_ok = confusion[(True, True)]
    wrong = passed_wrong + flagged_wrong

    print(f"  eligible (gold has subtotal, tax, total): {eligible}")
    print(f"  not flagged, totals correct : {passed_ok}")
    print(f"  not flagged, totals WRONG   : {passed_wrong}")
    print(f"  flagged,     totals wrong   : {flagged_wrong}")
    print(f"  flagged,     totals correct : {flagged_ok}")
    if wrong:
        print(f"  wrong totals caught by the flag: {flagged_wrong}/{wrong}")
    if passed_ok + passed_wrong:
        print(f"  totals correct when NOT flagged: {passed_ok}/{passed_ok + passed_wrong}")
    print()


def main() -> None:
    names = sys.argv[1:] or [
        "base_cord_predictions",
        "finetuned_cord_predictions",
        "base_synthetic_predictions",
        "finetuned_synthetic_predictions",
    ]
    for name in names:
        analyse(name)


if __name__ == "__main__":
    main()
