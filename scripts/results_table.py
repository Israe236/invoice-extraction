"""Print the base-vs-fine-tuned results as Markdown, straight from results/.

    python scripts/results_table.py

The README tables are pasted from this output rather than typed by hand, so a
number in the README is always a number some run actually wrote to disk.
Fields with zero support on a test set are shown as n/a: they were never
tested there, which is different from scoring zero.
"""

from __future__ import annotations

import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results"
FIELDS = (
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
DATASETS = (("cord", "CORD-v2 test split"), ("synthetic", "Synthetic held-out invoices"))


def load(model: str, dataset: str) -> dict | None:
    path = RESULTS / f"{model}_{dataset}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def cell(value: float) -> str:
    return f"{value:.3f}"


def delta(before: float, after: float) -> str:
    change = after - before
    return f"{'+' if change >= 0 else '−'}{abs(change):.3f}"


def table(dataset: str, title: str) -> str:
    base, tuned = load("base", dataset), load("finetuned", dataset)
    if base is None or tuned is None:
        return f"### {title}\n\n(missing results for {dataset})\n"

    lines = [
        f"### {title} — n={base['n_examples']}",
        "",
        "| Field | Base F1 | Fine-tuned F1 | Change | Support |",
        "|---|---|---|---|---|",
    ]
    for field in FIELDS:
        b, t = base["fields"][field], tuned["fields"][field]
        if b["support"] == 0:
            lines.append(f"| {field} | n/a | n/a | | 0 |")
            continue
        lines.append(
            f"| {field} | {cell(b['f1'])} | **{cell(t['f1'])}** | "
            f"{delta(b['f1'], t['f1'])} | {b['support']} |"
        )
    lines.append(
        f"| **micro-average** | {cell(base['micro_f1'])} | **{cell(tuned['micro_f1'])}** | "
        f"{delta(base['micro_f1'], tuned['micro_f1'])} | |"
    )
    lines.append(
        f"| **JSON parse rate** | {base['n_parsed']}/{base['n_examples']} "
        f"({base['parse_rate'] * 100:.0f}%) | **{tuned['n_parsed']}/{tuned['n_examples']} "
        f"({tuned['parse_rate'] * 100:.0f}%)** | | |"
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    for dataset, title in DATASETS:
        print(table(dataset, title))


if __name__ == "__main__":
    main()
