"""Evaluation loop: run a model over a test set and score it.

The same entry point serves the zero-shot base model and the fine-tuned
adapter -- pass `--adapter` or don't. Keeping one code path is deliberate:
the "before" and "after" numbers in the README have to come from identical
preprocessing, identical decoding and identical scoring, or the comparison is
not a comparison.

Two test sets:

- `cord`      held-out CORD-v2 receipts. Real photographs, real noise.
- `synthetic` held-out generated French/Moroccan invoices, from a seed range
              disjoint from the training seeds. This is the only set that
              exercises ICE, IF, invoice number, date and currency.
"""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import dataclass
from pathlib import Path

import torch
import yaml

from invoice_extraction.data import (
    EXTRACTION_PROMPT,
    ExtractedFields,
    extract_fields,
    load_cord_split,
)
from invoice_extraction.evaluate import EvalReport, format_report, parse_model_output, score_dataset
from invoice_extraction.model import load_config, load_finetuned_model, load_model_and_processor
from invoice_extraction.validate import validate


@dataclass
class BaselineConfig:
    split: str
    num_eval_examples: int


def load_baseline_config(path: str) -> BaselineConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return BaselineConfig(split=raw["split"], num_eval_examples=raw["num_eval_examples"])


def generate(model, processor, image, prompt: str, max_new_tokens: int) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    trimmed = output_ids[:, inputs["input_ids"].shape[1] :]
    return processor.batch_decode(trimmed, skip_special_tokens=True)[0]


def _cord_examples(split: str, n: int) -> list[tuple[object, ExtractedFields]]:
    dataset = load_cord_split(split)
    examples = []
    for row in itertools.islice(dataset, n):
        gt_parse = json.loads(row["ground_truth"]).get("gt_parse", {})
        examples.append((row["image"], extract_fields(gt_parse)))
    return examples


def _synthetic_examples(n: int) -> list[tuple[object, ExtractedFields]]:
    # Imported lazily: the synthetic stack (WeasyPrint, PyMuPDF, Albumentations)
    # is not needed to evaluate on CORD, and is heavy to import.
    from invoice_extraction.render import build_synthetic_eval_set

    return build_synthetic_eval_set(n)


def load_eval_examples(
    dataset: str, config: BaselineConfig
) -> list[tuple[object, ExtractedFields]]:
    if dataset == "cord":
        return _cord_examples(config.split, config.num_eval_examples)
    if dataset == "synthetic":
        return _synthetic_examples(config.num_eval_examples)
    raise ValueError(f"unknown dataset {dataset!r}, expected 'cord' or 'synthetic'")


def run_evaluation(
    config_path: str,
    adapter_path: str | None = None,
    dataset: str = "cord",
    predictions_path: str | None = None,
) -> EvalReport:
    model_config = load_config(config_path)
    baseline_config = load_baseline_config(config_path)

    if adapter_path:
        model, processor = load_finetuned_model(model_config, adapter_path)
    else:
        model, processor = load_model_and_processor(model_config)

    examples = load_eval_examples(dataset, baseline_config)
    raw_predictions: list[str] = []
    truths: list[ExtractedFields] = []

    for index, (image, truth) in enumerate(examples, start=1):
        raw = generate(model, processor, image, EXTRACTION_PROMPT, model_config.max_new_tokens)
        raw_predictions.append(raw)
        truths.append(truth)
        print(f"  [{index}/{len(examples)}] {len(raw)} chars", flush=True)

    report = score_dataset(raw_predictions, truths)

    if predictions_path:
        # Saved so the README can quote a real example, and so a bad number can
        # be traced back to the text that produced it instead of re-running.
        records = []
        for raw, truth in zip(raw_predictions, truths, strict=True):
            parsed = parse_model_output(raw)
            records.append(
                {
                    "raw_output": raw,
                    "parsed": parsed,
                    "ground_truth": truth,
                    "validation": validate(parsed).to_dict() if parsed else None,
                }
            )
        Path(predictions_path).parent.mkdir(parents=True, exist_ok=True)
        Path(predictions_path).write_text(
            json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    return report


# Backwards-compatible alias for the earlier name used in docs/PLAN.md.
def run_baseline(config_path: str, adapter_path: str | None = None) -> EvalReport:
    return run_evaluation(config_path, adapter_path, dataset="cord")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--adapter", default=None, help="path to a trained LoRA adapter")
    parser.add_argument("--dataset", default="cord", choices=("cord", "synthetic"))
    parser.add_argument("--out", default=None, help="write the score report here as JSON")
    parser.add_argument("--predictions", default=None, help="write raw predictions here as JSON")
    args = parser.parse_args()

    label = f"{'fine-tuned' if args.adapter else 'base'} model on {args.dataset}"
    print(f"Evaluating {label}...", flush=True)

    report = run_evaluation(args.config, args.adapter, args.dataset, args.predictions)
    print()
    print(format_report(report))

    if args.out:
        payload = report.to_dict() | {
            "model": "fine-tuned" if args.adapter else "base",
            "adapter": args.adapter,
            "dataset": args.dataset,
            "config": args.config,
        }
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
