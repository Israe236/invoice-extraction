"""Measure peak training VRAM at one image resolution.

Run one resolution per process. An OOM leaves the CUDA context unusable --
subsequent allocations in the same process fail with "device not ready" -- so
sweeping inside a single process produces garbage after the first failure.
`scripts/sweep_train_memory.sh` drives this across a range.

    python scripts/probe_train_memory.py 401408
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
# At 6 GB, fragmentation alone can fail a 44 MB allocation while hundreds of MB
# sit reserved-but-unusable.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402
from transformers import AutoProcessor  # noqa: E402

from invoice_extraction.train import (  # noqa: E402
    CordDataset,
    SyntheticDataset,
    load_model_for_training,
    load_train_config,
    make_collate_fn,
)


def report_checkpointing(model) -> None:
    """Which submodules actually have gradient checkpointing switched on.

    The vision tower matters as much as the language model here: Qwen2-VL's ViT
    runs full attention over ~2000 patches, and `prepare_model_for_kbit_training`
    makes the embedding output require grad, which keeps the whole vision
    subgraph alive for backward even though none of its weights are trained.
    """
    on, off = [], []
    for name, module in model.named_modules():
        flag = getattr(module, "gradient_checkpointing", None)
        if flag is True:
            on.append(name or "<root>")
        elif flag is False:
            off.append(name or "<root>")
    print(f"  checkpointing ON : {len(on)} modules {on[:3]}")
    print(f"  checkpointing OFF: {len(off)} modules {off[:3]}")


def main() -> None:
    max_pixels = int(sys.argv[1]) if len(sys.argv) > 1 else 401408
    config = load_train_config("configs/train_smoke.yaml")

    model, _ = load_model_for_training(config)
    # prepare_model_for_kbit_training enables this, but being explicit costs
    # nothing and use_reentrant=False is required for it to work with PEFT.
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if hasattr(model, "base_model"):
        visual = getattr(model.base_model.model, "visual", None)
        if visual is not None:
            visual.gradient_checkpointing = True

    report_checkpointing(model)
    print(f"  VRAM after load: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    processor = AutoProcessor.from_pretrained(
        config.model_id, min_pixels=config.min_pixels, max_pixels=max_pixels
    )
    collate = make_collate_fn(processor)
    datasets = [CordDataset("train"), SyntheticDataset(2, seed=config.synthetic_seed)]

    torch.cuda.reset_peak_memory_stats()
    seq_len = 0
    try:
        for dataset in datasets:
            batch = collate([dataset[0]])
            seq_len = max(seq_len, batch["input_ids"].shape[1])
            batch = {k: v.to(model.device) for k, v in batch.items()}
            outputs = model(**batch)
            outputs.loss.backward()
            model.zero_grad(set_to_none=True)
            del outputs, batch
            torch.cuda.empty_cache()
    except torch.OutOfMemoryError:
        peak = torch.cuda.max_memory_allocated() / 1024**3
        print(f"RESULT max_pixels={max_pixels} seq_len={seq_len} peak={peak:.2f}G verdict=OOM")
        raise SystemExit(2) from None

    peak = torch.cuda.max_memory_allocated() / 1024**3
    print(f"RESULT max_pixels={max_pixels} seq_len={seq_len} peak={peak:.2f}G verdict=FITS")


if __name__ == "__main__":
    main()
