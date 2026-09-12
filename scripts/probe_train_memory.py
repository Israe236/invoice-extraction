"""Measure peak training VRAM for one configuration.

Run one configuration per process. A CUDA OOM leaves the context unusable, so
sweeping inside a single process produces garbage after the first failure.
`scripts/sweep_train_memory.sh` drives this across a range.

    python scripts/probe_train_memory.py --max-pixels 200704
    python scripts/probe_train_memory.py --max-pixels 200704 --reentrant

Two things this controls that turned out to matter more than the resolution:

**Memory fraction.** WSL2's WDDM passthrough lets CUDA allocate past the
physical 6 GB into host RAM. PyTorch then happily reports "12.21 GiB
allocated" on a 6 GB card, the VM starts thrashing, and the failure surfaces
as `CUDA driver error: device not ready` or as the kernel OOM-killer -- never
as the out-of-memory error it actually is. Capping the fraction makes the
allocator refuse instead of spill, so a configuration that does not fit says
so cleanly in a tenth of the time.

**use_reentrant.** With PEFT, reentrant gradient checkpointing silently does
nothing when the checkpointed inputs do not require grad. The flag is on, the
memory saving is absent. `--reentrant` exists to measure that difference
rather than argue about it.
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-pixels", type=int, default=401408)
    parser.add_argument("--config", default="configs/train_smoke.yaml")
    parser.add_argument(
        "--reentrant",
        action="store_true",
        help="use reentrant gradient checkpointing (the old default; usually a no-op with PEFT)",
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=0.92,
        help="cap PyTorch at this fraction of total VRAM so it cannot spill into host RAM",
    )
    parser.add_argument("--attn", default=None, choices=("sdpa", "eager"))
    parser.add_argument("--liger", action="store_true", help="fused linear cross-entropy")
    parser.add_argument("--forward-only", action="store_true", help="skip the backward pass")
    parser.add_argument("--dataset", default="both", choices=("both", "cord", "synthetic"))
    return parser.parse_args()


def checkpointing_summary(model) -> tuple[int, int]:
    on = off = 0
    for module in model.modules():
        flag = getattr(module, "gradient_checkpointing", None)
        if flag is True:
            on += 1
        elif flag is False:
            off += 1
    return on, off


def main() -> None:
    args = parse_args()
    config = load_train_config(args.config)
    if args.attn:
        config.attn_implementation = args.attn
    if args.liger:
        config.use_liger_kernel = True

    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    free_before = torch.cuda.mem_get_info()[0] / 1024**3
    # The hard stop that turns a host-RAM spill into an honest CUDA OOM.
    torch.cuda.set_per_process_memory_fraction(args.fraction)

    model, _ = load_model_for_training(config)
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": args.reentrant}
    )
    # Essential, and easy to leave out: HuggingFace's GradientCheckpointingLayer
    # only checkpoints when `self.training` is true. Probing an eval-mode model
    # measures a configuration with checkpointing silently disabled, which is
    # strictly worse than what Trainer actually runs.
    model.train()
    on, off = checkpointing_summary(model)

    print(f"  device total {total:.2f}G, free before load {free_before:.2f}G")
    print(f"  memory fraction cap {args.fraction} -> {total * args.fraction:.2f}G")
    print(f"  gradient checkpointing: {on} modules on, {off} off, reentrant={args.reentrant}")
    print(f"  VRAM after load: {torch.cuda.memory_allocated() / 1024**3:.2f}G")

    processor = AutoProcessor.from_pretrained(
        config.model_id, min_pixels=config.min_pixels, max_pixels=args.max_pixels
    )
    collate = make_collate_fn(processor)
    datasets = {
        "cord": CordDataset("train"),
        "synthetic": SyntheticDataset(2, seed=config.synthetic_seed),
    }
    if args.dataset != "both":
        datasets = {args.dataset: datasets[args.dataset]}

    seq_len = 0
    verdict = "FITS"
    peak = 0.0
    try:
        for name, dataset in datasets.items():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            batch = collate([dataset[0]])
            this_seq = batch["input_ids"].shape[1]
            seq_len = max(seq_len, this_seq)
            batch = {k: v.to(model.device) for k, v in batch.items()}

            outputs = model(**batch)
            after_forward = torch.cuda.max_memory_allocated() / 1024**3
            # Liger's fused linear cross-entropy returns a loss without ever
            # building the logits tensor. If logits comes back with a shape,
            # the fused path did NOT engage, whatever the patch call reported.
            logits = getattr(outputs, "logits", None)
            if logits is None:
                print("    logits: not materialised (fused CE active)")
            else:
                print(
                    f"    logits: {tuple(logits.shape)} {logits.dtype} = "
                    f"{logits.numel() * logits.element_size() / 1024**3:.2f}G"
                )
            if not args.forward_only:
                outputs.loss.backward()
                model.zero_grad(set_to_none=True)
            after_backward = torch.cuda.max_memory_allocated() / 1024**3
            peak = max(peak, after_backward)
            print(
                f"  {name:10s} seq={this_seq:5d}  fwd_peak={after_forward:.2f}G  "
                f"total_peak={after_backward:.2f}G"
            )
            del outputs, batch
    except torch.OutOfMemoryError as exc:
        verdict = "OOM"
        peak = max(peak, torch.cuda.max_memory_allocated() / 1024**3)
        # The size of the allocation that failed is the most informative
        # number available: it says whether we are slightly over budget or
        # asking for something structurally wrong.
        first_line = str(exc).split(". ")[0]
        print(f"  OOM DETAIL: {first_line}")
        print(
            f"  at failure: allocated={torch.cuda.memory_allocated() / 1024**3:.2f}G "
            f"reserved={torch.cuda.memory_reserved() / 1024**3:.2f}G"
        )

    print(
        f"RESULT max_pixels={args.max_pixels} attn={config.attn_implementation} "
        f"liger={config.use_liger_kernel} fwd_only={args.forward_only} "
        f"seq_len={seq_len} peak={peak:.2f}G verdict={verdict}"
    )
    raise SystemExit(0 if verdict == "FITS" else 2)


if __name__ == "__main__":
    main()
