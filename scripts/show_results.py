"""Print the saved evaluation reports as a readable table.

Reads whatever is in results/ so the README can be filled in from measured
numbers rather than from a terminal scrollback.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def show(path: Path) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    print(f"===== {path.stem} =====")
    print(
        f"model={data.get('model')}  dataset={data.get('dataset')}  "
        f"n={data['n_examples']}  parsed={data['n_parsed']}  "
        f"parse_rate={data['parse_rate']:.3f}  micro_f1={data['micro_f1']:.3f}"
    )
    for field, values in data["fields"].items():
        support = values["support"]
        if support == 0:
            print(f"  {field:16s} n/a (support 0)")
        else:
            print(
                f"  {field:16s} P {values['precision']:.3f}  R {values['recall']:.3f}  "
                f"F1 {values['f1']:.3f}  support {support}"
            )
    print()


def main() -> None:
    names = sys.argv[1:] or sorted(
        p.stem for p in RESULTS_DIR.glob("*.json") if "predictions" not in p.stem
    )
    for name in names:
        path = RESULTS_DIR / f"{name}.json"
        if path.exists():
            show(path)
        else:
            print(f"(missing: {path})")


if __name__ == "__main__":
    main()
