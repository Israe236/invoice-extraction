"""Model and processor loading, with VRAM-safe defaults for a 6GB card."""

from dataclasses import dataclass

import torch
import yaml
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2VLForConditionalGeneration


@dataclass
class ModelConfig:
    model_id: str
    max_pixels: int
    min_pixels: int
    max_new_tokens: int
    load_in_4bit: bool


def load_config(path: str) -> ModelConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return ModelConfig(
        model_id=raw["model_id"],
        max_pixels=raw["max_pixels"],
        min_pixels=raw["min_pixels"],
        max_new_tokens=raw["max_new_tokens"],
        load_in_4bit=raw["load_in_4bit"],
    )


def load_model_and_processor(
    config: ModelConfig,
) -> tuple[Qwen2VLForConditionalGeneration, AutoProcessor]:
    processor = AutoProcessor.from_pretrained(
        config.model_id,
        min_pixels=config.min_pixels,
        max_pixels=config.max_pixels,
    )

    quantization_config = (
        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        if config.load_in_4bit
        else None
    )

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        config.model_id,
        quantization_config=quantization_config,
        dtype=torch.bfloat16,
        device_map="cuda",
    )
    model.eval()
    return model, processor
