"""Render the README figures from measured files only.

    python scripts/plot_results.py

Reads `checkpoints/qwen2vl-2b-lora-v2/log_history.json` for the loss curve and
`results/{base,finetuned}_{cord,synthetic}.json` for the F1 comparison. A figure
whose inputs do not exist yet is skipped, never drawn from placeholders.

Each figure is written twice, for a light and a dark background, so the README
can serve the right one through GitHub's `<picture>` / prefers-color-scheme.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "docs" / "assets"
RESULTS = ROOT / "results"
# The committed copy, so the figure can be rebuilt from a fresh clone. Checkpoints
# themselves are gitignored.
HISTORY = RESULTS / "train_v2_log_history.json"

# Validated two-slot categorical palette (blue, orange), light and dark steps.
THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "text": "#0b0b0b",
        "text_secondary": "#52514e",
        "grid": "#e6e5e1",
        "series": ("#2a78d6", "#eb6834"),
    },
    "dark": {
        "surface": "#1a1a19",
        "text": "#ffffff",
        "text_secondary": "#c3c2b7",
        "grid": "#33332f",
        "series": ("#3987e5", "#d95926"),
    },
}

# Field order for the comparison: totals first, then identifiers, then items.
FIELDS = (
    "total",
    "subtotal",
    "tax",
    "items",
    "ice",
    "if_number",
    "invoice_number",
    "date",
    "currency",
)


def style_axes(ax, theme: dict) -> None:
    ax.set_facecolor(theme["surface"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(theme["grid"])
        ax.spines[side].set_linewidth(1)
    ax.tick_params(colors=theme["text_secondary"], labelsize=10, length=0)
    ax.grid(color=theme["grid"], linewidth=1, linestyle="-")
    ax.set_axisbelow(True)


def plot_loss(mode: str) -> Path | None:
    if not HISTORY.exists():
        return None
    theme = THEMES[mode]
    history = json.loads(HISTORY.read_text(encoding="utf-8"))
    points = [(e["step"], e["loss"]) for e in history if "loss" in e]
    if not points:
        return None
    steps, losses = zip(*points, strict=True)

    fig, ax = plt.subplots(figsize=(8, 3.6), dpi=150)
    fig.patch.set_facecolor(theme["surface"])
    style_axes(ax, theme)
    ax.grid(axis="x", visible=False)

    colour = theme["series"][0]
    ax.plot(steps, losses, color=colour, linewidth=2, solid_capstyle="round",
            solid_joinstyle="round")
    ax.scatter([steps[-1]], [losses[-1]], s=64, color=colour,
               edgecolors=theme["surface"], linewidths=2, zorder=3)
    # Label the two values the story is about: where it started and ended.
    ax.annotate(f"{losses[0]:.3f}", (steps[0], losses[0]), xytext=(6, 4),
                textcoords="offset points", color=theme["text"], fontsize=10)
    ax.annotate(f"{losses[-1]:.3f}", (steps[-1], losses[-1]), xytext=(-8, 10),
                textcoords="offset points", ha="right", color=theme["text"], fontsize=10)

    ax.set_ylim(0, max(losses) * 1.15)
    ax.set_xlabel("optimiser step", color=theme["text_secondary"], fontsize=10)
    ax.set_ylabel("training loss", color=theme["text_secondary"], fontsize=10)
    ax.set_title(
        "QLoRA training loss, 1 epoch, 800 CORD receipts + 200 synthetic invoices",
        color=theme["text"], fontsize=12, loc="left", pad=12,
    )
    fig.tight_layout()
    out = ASSETS / f"loss_curve_{mode}.png"
    fig.savefig(out, facecolor=theme["surface"])
    plt.close(fig)
    return out


def load_report(model: str, dataset: str) -> dict | None:
    path = RESULTS / f"{model}_{dataset}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def plot_f1(mode: str) -> Path | None:
    reports = {
        (model, dataset): load_report(model, dataset)
        for model in ("base", "finetuned")
        for dataset in ("cord", "synthetic")
    }
    if any(r is None for r in reports.values()):
        return None
    theme = THEMES[mode]

    panels = (("cord", "CORD-v2 receipts"), ("synthetic", "Synthetic Moroccan invoices"))
    # Fields with no gold values were never tested on a set; leave them out rather
    # than drawing a zero that reads as "always wrong".
    tested = {
        dataset: [f for f in FIELDS if reports[("base", dataset)]["fields"][f]["support"] > 0]
        for dataset, _ in panels
    }
    # Stack the panels and size each by its row count so a bar is the same
    # thickness in both. Side by side, the 4-field panel drew bars twice as fat.
    row_counts = [len(tested[dataset]) for dataset, _ in panels]
    fig, axes = plt.subplots(
        2, 1, figsize=(8.5, 0.42 * sum(row_counts) + 2.6), dpi=150, sharex=True,
        gridspec_kw={"height_ratios": row_counts},
    )
    fig.patch.set_facecolor(theme["surface"])
    bar_height = 0.36

    for ax, (dataset, title) in zip(axes, panels, strict=True):
        style_axes(ax, theme)
        ax.grid(axis="y", visible=False)
        base = reports[("base", dataset)]["fields"]
        tuned = reports[("finetuned", dataset)]["fields"]
        fields = tested[dataset]
        positions = range(len(fields))

        for offset, report, colour, label in (
            (-bar_height / 2, base, theme["series"][0], "Base model"),
            (bar_height / 2, tuned, theme["series"][1], "Fine-tuned"),
        ):
            values = [report[f]["f1"] for f in fields]
            ax.barh(
                [p + offset for p in positions], values, height=bar_height * 0.92,
                color=colour, label=label,
            )

        ax.set_yticks(list(positions))
        ax.set_yticklabels(fields, color=theme["text"], fontsize=10)
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        if ax is axes[-1]:
            ax.set_xlabel("F1", color=theme["text_secondary"], fontsize=10)
        micro_base = reports[("base", dataset)]["micro_f1"]
        micro_tuned = reports[("finetuned", dataset)]["micro_f1"]
        ax.set_title(
            f"{title}\nmicro-F1 {micro_base:.3f} → {micro_tuned:.3f}",
            color=theme["text"], fontsize=11, loc="left", pad=10,
        )

    handles, labels = axes[0].get_legend_handles_labels()
    legend = fig.legend(
        handles, labels, loc="upper right", ncol=2, frameon=False, fontsize=10,
        bbox_to_anchor=(0.99, 0.99),
    )
    for text in legend.get_texts():
        text.set_color(theme["text"])

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = ASSETS / f"f1_comparison_{mode}.png"
    fig.savefig(out, facecolor=theme["surface"])
    plt.close(fig)
    return out


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    for mode in THEMES:
        for render in (plot_loss, plot_f1):
            out = render(mode)
            name = render.__name__.removeprefix("plot_")
            where = out.relative_to(ROOT) if out else "skipped (inputs missing)"
            print(f"{name:5s} {mode:5s} -> {where}")


if __name__ == "__main__":
    main()
