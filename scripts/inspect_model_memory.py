"""Break down where the model's VRAM actually goes, by dtype and by submodule.

`prepare_model_for_kbit_training` upcasts every non-quantised parameter to
fp32, and bitsandbytes leaves some modules unquantised. On a 152k-token
vocabulary and a 675M-parameter vision tower those exclusions are not a
rounding error, so it is worth seeing the actual numbers rather than assuming
"4-bit 2B model" means 1.3 GB.
"""

from __future__ import annotations

import os
from collections import defaultdict

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch  # noqa: E402

from invoice_extraction.train import load_model_for_training, load_train_config  # noqa: E402


def main() -> None:
    config = load_train_config("configs/train_smoke.yaml")
    model, _ = load_model_for_training(config)

    by_dtype: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # bytes, count
    by_top: dict[str, int] = defaultdict(int)
    trainable = 0

    for name, parameter in model.named_parameters():
        nbytes = parameter.numel() * parameter.element_size()
        key = str(parameter.dtype)
        by_dtype[key][0] += nbytes
        by_dtype[key][1] += 1
        # Group under the first meaningful path segment.
        parts = [p for p in name.split(".") if p not in ("base_model", "model", "weight")]
        by_top[parts[0] if parts else name] += nbytes
        if parameter.requires_grad:
            trainable += parameter.numel()

    print("\n--- parameters by dtype ---")
    for dtype, (nbytes, count) in sorted(by_dtype.items(), key=lambda kv: -kv[1][0]):
        print(f"  {dtype:20s} {nbytes / 1024**3:6.3f} GB  ({count} tensors)")

    print("\n--- parameters by top-level module ---")
    for name, nbytes in sorted(by_top.items(), key=lambda kv: -kv[1])[:10]:
        print(f"  {name:20s} {nbytes / 1024**3:6.3f} GB")

    total = sum(v[0] for v in by_dtype.values())
    print(f"\n  total parameter bytes: {total / 1024**3:.3f} GB")
    print(f"  torch.cuda.memory_allocated: {torch.cuda.memory_allocated() / 1024**3:.3f} GB")
    print(f"  trainable parameters: {trainable / 1e6:.1f} M")

    print("\n--- largest individual tensors ---")
    biggest = sorted(
        ((p.numel() * p.element_size(), n, p) for n, p in model.named_parameters()),
        reverse=True,
        key=lambda t: t[0],
    )[:8]
    for nbytes, name, parameter in biggest:
        print(f"  {nbytes / 1024**3:6.3f} GB  {str(parameter.dtype):15s} {name}")


if __name__ == "__main__":
    main()
