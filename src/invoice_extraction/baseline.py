"""Zero-shot / fine-tuned baseline evaluation loop."""

import itertools
import json
from dataclasses import dataclass

import torch
import yaml

from invoice_extraction.data import EXTRACTION_PROMPT, extract_fields, load_cord_split
from invoice_extraction.evaluate import FieldScore, format_report, score_dataset
from invoice_extraction.model import load_config, load_finetuned_model, load_model_and_processor


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


def run_baseline(config_path: str, adapter_path: str | None = None) -> dict[str, FieldScore]:
    model_config = load_config(config_path)
    baseline_config = load_baseline_config(config_path)
    if adapter_path:
        model, processor = load_finetuned_model(model_config, adapter_path)
    else:
        model, processor = load_model_and_processor(model_config)

    dataset = load_cord_split(baseline_config.split)
    raw_predictions = []
    truths = []
    for row in itertools.islice(dataset, baseline_config.num_eval_examples):
        gt_parse = json.loads(row["ground_truth"]).get("gt_parse", {})
        truths.append(extract_fields(gt_parse))
        raw_predictions.append(
            generate(model, processor, row["image"], EXTRACTION_PROMPT, model_config.max_new_tokens)
        )

    return score_dataset(raw_predictions, truths)


if __name__ == "__main__":
    import sys

    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/baseline.yaml"
    adapter_path = sys.argv[2] if len(sys.argv) > 2 else None
    scores = run_baseline(config_path, adapter_path)
    print(format_report(scores))
