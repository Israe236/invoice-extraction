"""LoRA fine-tuning for Qwen2-VL on CORD-v2 receipts.

Meant to run on Kaggle (P100 16GB or 2xT4) — the local GPU is 6GB,
too tight for a full training run.
"""

import json
import os
from dataclasses import dataclass
from functools import wraps
from pathlib import Path

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
from invoice_extraction.render import degrade, render_invoice
from invoice_extraction.synthetic import generate_invoice_data, to_ground_truth


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
    seed: int = 42
    # Set to a small number for an end-to-end smoke test before committing a
    # Kaggle session to a multi-hour run. 0 means "train the full epochs".
    max_steps: int = 0
    # Run the vision encoder outside autograd. See freeze_vision_tower().
    freeze_vision_tower: bool = True
    attn_implementation: str = "sdpa"
    # Fuse the final projection with cross-entropy so the full
    # [seq x 151936] logits tensor is never materialised. See apply_liger().
    use_liger_kernel: bool = False
    # Undo prepare_model_for_kbit_training's fp32 upcast on frozen tensors.
    cast_frozen_to_bf16: bool = True
    # Hard ceiling on PyTorch's share of VRAM, as a fraction of the card's
    # total. Must sit BELOW the physically free memory. 0 disables the cap.
    cuda_memory_fraction: float = 0.0


def load_train_config(path: str) -> TrainConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return TrainConfig(**raw)


def apply_liger(model_id: str) -> bool:
    """Swap in fused Triton kernels before the model is constructed.

    The memory that actually blocks training here is not the image. With the
    vision tower frozen, peak VRAM is flat at ~4.4 GB across every resolution
    from 150528 to 401408 pixels, because the sequence is dominated by text --
    945 tokens at the smallest setting, of which only 192 are visual.

    What is left is cross-entropy over Qwen's 151936-token vocabulary. A
    945-token sequence materialises a [945 x 151936] logits tensor, upcast to
    fp32 for the loss, plus its gradient: well over a gigabyte for a loss on
    ~200 supervised tokens. Liger's fused linear cross-entropy computes the
    projection and the loss in chunks and never holds the whole tensor.

    Must be called before `from_pretrained`, since it patches the classes the
    model is built from.
    """
    from liger_kernel.transformers import apply_liger_kernel_to_qwen2_vl

    apply_liger_kernel_to_qwen2_vl(fused_linear_cross_entropy=True)
    print(f"Liger fused kernels applied for {model_id}", flush=True)
    return True


def cast_frozen_params_to_bf16(model, dtype: torch.dtype = torch.bfloat16) -> tuple[int, float]:
    """Undo the fp32 upcast on parameters that are never trained.

    `prepare_model_for_kbit_training` casts every non-quantised parameter to
    fp32. For layer norms that is a real stability measure. For Qwen2-VL's
    token embedding it is 0.869 GB of VRAM spent on a frozen tensor: the
    vocabulary is 151936 and the hidden size 1536, so the embedding alone is
    233M parameters, and the model ties it to the output head -- which means
    the logits are produced in fp32 as well.

    Only frozen parameters are moved back. The LoRA adapters, the only things
    an optimiser touches, stay in fp32 where the precision matters.
    """
    moved_bytes = 0
    moved = 0
    for parameter in model.parameters():
        if not parameter.requires_grad and parameter.dtype == torch.float32:
            before = parameter.numel() * parameter.element_size()
            parameter.data = parameter.data.to(dtype)
            moved_bytes += before - parameter.numel() * parameter.element_size()
            moved += 1
    return moved, moved_bytes / 1024**3


def find_vision_tower(model) -> torch.nn.Module | None:
    """Locate the ViT inside a (possibly PEFT-wrapped, possibly nested) model."""
    for attribute_path in ("visual", "model.visual", "base_model.model.visual",
                           "base_model.model.model.visual"):
        current = model
        for attribute in attribute_path.split("."):
            current = getattr(current, attribute, None)
            if current is None:
                break
        if current is not None:
            return current
    return None


def freeze_vision_tower(model) -> bool:
    """Run the vision encoder as pure feature extraction, outside autograd.

    LoRA is applied to the language model only, so no vision parameter is ever
    updated. Despite that, the ViT was still being pulled into the autograd
    graph: `prepare_model_for_kbit_training` calls `enable_input_require_grads`,
    which makes the embedding output require grad, and because the image
    features are scattered into that same tensor the whole vision subgraph
    stays alive for the backward pass.

    Two costs followed from that. The activations of a ViT running attention
    over ~2000 patches were retained -- the backward pass asked for 12.21 GB on
    a 6 GB card. And with gradient checkpointing on, the entire ViT forward was
    *re-executed* during backward, which on WSL2's WDDM passthrough reliably
    ended in `CUDA driver error: device not ready` inside the ViT MLP, at every
    resolution tried, including ones far too small for memory to be the cause.

    Wrapping the encoder in `torch.no_grad` removes both: nothing to store,
    nothing to recompute. The gradients that reach the LoRA adapters are
    unchanged, because they never flowed through here in the first place.
    """
    visual = find_vision_tower(model)
    if visual is None:
        return False

    for parameter in visual.parameters():
        parameter.requires_grad_(False)
    # Checkpointing a module that needs no gradients only buys recomputation.
    visual.gradient_checkpointing = False

    if getattr(visual, "_frozen_no_grad", False):
        return True

    original_forward = visual.forward

    @wraps(original_forward)
    def no_grad_forward(*args, **kwargs):
        with torch.no_grad():
            return original_forward(*args, **kwargs)

    visual.forward = no_grad_forward
    visual._frozen_no_grad = True
    return True


def load_model_for_training(config: TrainConfig) -> tuple[PreTrainedModel, AutoProcessor]:
    if config.use_liger_kernel:
        apply_liger(config.model_id)

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
        attn_implementation=config.attn_implementation,
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
    peft_model = get_peft_model(model, lora_config)

    if config.freeze_vision_tower and not freeze_vision_tower(peft_model):
        print("WARNING: vision tower not found; it will train inside autograd", flush=True)

    if config.cast_frozen_to_bf16:
        moved, saved_gb = cast_frozen_params_to_bf16(peft_model)
        print(
            f"Cast {moved} frozen fp32 tensors back to bf16, saving {saved_gb:.2f} GB",
            flush=True,
        )

    return peft_model, processor


class CordDataset(Dataset):
    """CORD-v2 receipts, decoded one at a time.

    The HuggingFace `Dataset` is kept as-is rather than wrapped in `list(...)`.
    Materialising it decodes all 800 receipt images into memory at once --
    roughly 6 GB, since CORD photographs run up to 3020 px wide. That is fine
    on a Kaggle node with 30 GB of RAM and fatal on an 8 GB laptop, where the
    kernel OOM-killer terminates the run with no traceback right after the
    weights finish loading. Indexing the dataset instead decodes lazily.
    """

    def __init__(self, split: str):
        self.rows = load_cord_split(split)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        gt_parse = json.loads(row["ground_truth"]).get("gt_parse", {})
        fields = extract_fields(gt_parse)
        return {"image": row["image"], "messages": to_chat_example(row["image"], fields)}


class SyntheticDataset(Dataset):
    """Generated French/Moroccan invoices, rendered on demand.

    Rendering is deterministic in the seed, so producing example `i` at access
    time gives exactly the same document as pre-rendering the whole set would
    have -- at a fraction of the memory, and without a multi-minute stall
    before the first training step.
    """

    def __init__(self, n: int, seed: int):
        self.n = n
        self.seed = seed

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> dict:
        invoice = generate_invoice_data(seed=self.seed + idx)
        image = degrade(render_invoice(invoice), seed=self.seed + idx)
        return {"image": image, "messages": to_chat_example(image, to_ground_truth(invoice))}


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

    if config.cuda_memory_fraction:
        # Windows reserves part of the card for the desktop, so "6 GB" is not
        # 6 GB. If the allocator is allowed past what is actually free, WSL2's
        # WDDM passthrough spills into host RAM and the run dies with
        # "CUDA driver error: device not ready" rather than a clean OOM.
        torch.cuda.set_per_process_memory_fraction(config.cuda_memory_fraction)
        free_gb = torch.cuda.mem_get_info()[0] / 1024**3
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        cap_gb = total_gb * config.cuda_memory_fraction
        print(f"VRAM cap {cap_gb:.2f} GB (free {free_gb:.2f} GB of {total_gb:.2f} GB)", flush=True)
        if cap_gb > free_gb:
            print("WARNING: cap exceeds free VRAM; expect a driver fault, not an OOM", flush=True)

    model, processor = load_model_for_training(config)

    cord_dataset = CordDataset("train")
    synthetic_dataset = SyntheticDataset(config.num_synthetic_examples, seed=config.synthetic_seed)
    dataset = ConcatDataset([cord_dataset, synthetic_dataset])
    print(
        f"Training on {len(cord_dataset)} CORD receipts "
        f"+ {len(synthetic_dataset)} synthetic invoices = {len(dataset)} examples",
        flush=True,
    )

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_train_epochs,
        max_steps=config.max_steps or -1,
        gradient_checkpointing=config.gradient_checkpointing,
        optim=config.optim,
        warmup_ratio=config.warmup_ratio,
        logging_steps=config.logging_steps,
        save_strategy=config.save_strategy,
        seed=config.seed,
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

    # The loss curve is evidence. Persist it next to the adapter so the README
    # can show what training actually did instead of asserting that it worked.
    history_path = Path(config.output_dir) / "log_history.json"
    history_path.write_text(json.dumps(trainer.state.log_history, indent=2), encoding="utf-8")
    print(f"Saved adapter and loss history to {config.output_dir}", flush=True)


if __name__ == "__main__":
    import sys

    train(sys.argv[1] if len(sys.argv) > 1 else "configs/train.yaml")
