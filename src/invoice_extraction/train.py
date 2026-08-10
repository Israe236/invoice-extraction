"""LoRA fine-tuning for Qwen2-VL on CORD-v2 receipts.

Meant to run on Kaggle (P100 16GB or 2xT4) — the local GPU is 6GB,
too tight for a full training run.
"""

import json
import os
from dataclasses import dataclass

# Must run before torch initializes CUDA. This is single-GPU QLoRA (batch
# size 1, no distributed setup) — on a multi-GPU box, leaving every GPU
# visible makes Trainer auto-wrap the model in torch.nn.DataParallel, which
# corrupts 4-bit quantized weights during replication (a submodule ends up
# with zero parameters, surfacing as a confusing StopIteration deep in
# forward()). Restricting to one GPU up front avoids that entirely.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch
import yaml
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import ConcatDataset, Dataset
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    PreTrainedModel,
    Qwen2VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
)

from invoice_extraction.data import extract_fields, load_cord_split, to_chat_example
from invoice_extraction.render import iter_synthetic_examples


@dataclass
class TrainConfig:
    model_id: str
    max_pixels: int
    min_pixels: int
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    lora_target_modules: list[str]
    learning_rate: float
    num_train_epochs: float
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    gradient_checkpointing: bool
    optim: str
    warmup_ratio: float
    logging_steps: int
    save_strategy: str
    output_dir: str
    num_synthetic_examples: int
    synthetic_seed: int


def load_train_config(path: str) -> TrainConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return TrainConfig(**raw)


def load_model_for_training(config: TrainConfig) -> tuple[PreTrainedModel, AutoProcessor]:
    processor = AutoProcessor.from_pretrained(
        config.model_id, min_pixels=config.min_pixels, max_pixels=config.max_pixels
    )
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        config.model_id,
        quantization_config=quantization_config,
        dtype=torch.bfloat16,
        device_map="cuda",
    )
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=config.gradient_checkpointing
    )
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, lora_config), processor


class CordDataset(Dataset):
    def __init__(self, split: str):
        self.rows = list(load_cord_split(split))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        gt_parse = json.loads(row["ground_truth"]).get("gt_parse", {})
        fields = extract_fields(gt_parse)
        return {"image": row["image"], "messages": to_chat_example(row["image"], fields)}


class SyntheticDataset(Dataset):
    def __init__(self, n: int, seed: int):
        self.examples = list(iter_synthetic_examples(n, seed=seed))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        messages = self.examples[idx]
        return {"image": messages[0]["content"][0]["image"], "messages": messages}


def make_collate_fn(processor: AutoProcessor):
    def collate(batch: list[dict]) -> dict:
        item = batch[0]  # batch size 1, per the 6GB/QLoRA training budget
        image = item["image"]
        messages = item["messages"]

        prompt_text = processor.apply_chat_template(
            messages[:1], tokenize=False, add_generation_prompt=True
        )
        full_text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        prompt_inputs = processor(text=[prompt_text], images=[image], return_tensors="pt")
        full_inputs = processor(text=[full_text], images=[image], return_tensors="pt")

        prompt_len = prompt_inputs["input_ids"].shape[1]
        labels = full_inputs["input_ids"].clone()
        labels[:, :prompt_len] = -100

        full_inputs["labels"] = labels
        return full_inputs

    return collate


def train(config_path: str) -> None:
    config = load_train_config(config_path)
    model, processor = load_model_for_training(config)
    cord_dataset = CordDataset("train")
    synthetic_dataset = SyntheticDataset(config.num_synthetic_examples, seed=config.synthetic_seed)
    dataset = ConcatDataset([cord_dataset, synthetic_dataset])

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_train_epochs,
        gradient_checkpointing=config.gradient_checkpointing,
        optim=config.optim,
        warmup_ratio=config.warmup_ratio,
        logging_steps=config.logging_steps,
        save_strategy=config.save_strategy,
        bf16=True,
        report_to=[],
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=make_collate_fn(processor),
    )
    trainer.train()
    trainer.save_model(config.output_dir)


if __name__ == "__main__":
    import sys

    train(sys.argv[1] if len(sys.argv) > 1 else "configs/train.yaml")
