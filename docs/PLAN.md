# Project plan

Living status document. The *reasoning* behind each choice lives in
[DECISIONS.md](DECISIONS.md); this file tracks what is done and what is next.

## Goal

Fine-tune a small VLM to extract structured JSON from invoices and receipts, with a
validation layer that flags documents whose arithmetic does not close. Portfolio project —
the README and the honesty of the metrics matter as much as the model.

## Constraints

- Free compute only. Local RTX 4050 (6 GB, measured 6141 MiB) + Kaggle free tier
  (P100 16 GB or 2×T4, ~30 GPU-hours/week, 12 h sessions).
- Local = code, debugging, and **all evaluation** (4-bit inference uses ~1.7 GB).
- Kaggle = training runs only.
- No real client data, ever. Public datasets or synthetic only.

## Schema (9 fields, stable)

`items[{name, qty, price}]`, `subtotal`, `tax`, `total`, `ice`, `if_number`,
`invoice_number`, `date`, `currency`.

Started as 4 fields taken from CORD-v2's actual field frequencies, extended with `ice` and
`if_number` once the synthetic generator made clear those are the project's differentiator,
then extended again with `invoice_number`, `date` and `currency` — the generator already
produced all three but never exposed them as ground truth, so the fields an accountant most
needs were untestable.

## Status

### Done

- [x] CORD-v2 loading and chat-format conversion
- [x] Synthetic French/Moroccan invoice generator (Faker → Jinja → WeasyPrint → PyMuPDF →
      Albumentations), with content cropping and a unit-price distractor column
- [x] Per-field P/R/F1 with field-aware normalisation, JSON parse rate, and per-field support
- [x] Robust JSON recovery from malformed model output
- [x] Validation layer: 8 rules, three outcomes (pass/fail/skipped), severity levels
- [x] **238 tests**, none requiring a GPU or a download
- [x] FastAPI `/extract`, `/validate`, `/health` + React/Vite frontend
- [x] Kaggle training notebook with a 3-step smoke test
- [x] **Zero-shot baseline measured** on both test sets — CORD-v2 *test* split and the
      held-out synthetic set, 50 examples each
- [x] End-to-end API smoke test against a real image (22 s/document)
- [x] README, DECISIONS.md

### Measured so far

Zero-shot `Qwen2-VL-2B-Instruct`, 4-bit, `max_pixels=401408`, CORD-v2 test split, n=50:

| field | precision | recall | F1 | support |
|---|---|---|---|---|
| subtotal | 0.650 | 0.419 | 0.510 | 31 |
| tax | 0.150 | 0.150 | 0.150 | 20 |
| total | 0.800 | 0.340 | 0.478 | 47 |
| items | 0.522 | 0.278 | 0.363 | 126 |
| ice / if_number / invoice_number / date / currency | — | — | n/a | 0 |

micro-F1 **0.346**, JSON parse rate **22/50 (44 %)**.

The parse rate is the headline: the base model failed to return parseable JSON on more than
half the documents, *after* a repair pass. Fields with support 0 are not in CORD-v2 and are
measured on the synthetic test set instead.

> These numbers are **not comparable** to the ones in this file's earlier revisions. The
> metric changed (field-aware normalisation instead of string equality), the split changed
> (test instead of validation), and n changed (50 instead of 20). Both the base and the
> fine-tuned model are being re-measured under the current metric via the same code path.

Zero-shot on the held-out **synthetic** test set, same settings, n=50:

| field | precision | recall | F1 | support |
|---|---|---|---|---|
| subtotal | 0.688 | 0.660 | 0.673 | 50 |
| tax | 0.688 | 0.660 | 0.673 | 50 |
| total | 0.833 | 0.800 | 0.816 | 50 |
| items | 0.097 | 0.093 | 0.095 | 161 |
| ice | 0.854 | 0.820 | 0.837 | 50 |
| if_number | 1.000 | 0.960 | 0.980 | 50 |
| invoice_number | 1.000 | 0.680 | 0.809 | 50 |
| date | 1.000 | 0.680 | 0.809 | 50 |
| currency | 1.000 | 0.340 | 0.507 | 50 |

micro-F1 **0.567**, parse rate **49/50 (98 %)**.

Three things this changes about the plan:

1. **The parse rate gap (44 % on photos vs 98 % on renders) is the main baseline weakness**,
   not raw reading accuracy. Fine-tuning should be judged mostly on whether it closes that.
2. **ICE and IF are already strong zero-shot** (0.837 / 0.980) — the opposite of the v1
   expectation that they were the big gap. The earlier "ice F1 0.000" was a *support-0
   artefact of measuring on CORD*, not evidence the model could not read an ICE. Worth
   remembering: it was a reporting bug masquerading as a model finding.
3. **Line items are the real weakness** (0.095), and the cause is measured: 104 of 137
   matched lines took the unit price from the P.U. distractor column instead of the line
   total. See `scripts/inspect_items.py`.

### Local training: attempted, does not work

Training was tried on the RTX 4050 rather than assumed impossible. Measured:

- only **4.88 GB of 6.00 GB** VRAM is free with the Windows desktop running
- model + LoRA loads in 1.98 GB; the backward pass at `max_pixels=401408` wants **12.21 GB**
- at `max_pixels=100352` it fails with `CUDA driver error: device not ready`, which is a
  driver fault rather than an OOM, inside the bitsandbytes 4-bit backward kernel
- with `CUDA_LAUNCH_BLOCKING=1` the **WSL2 VM itself crashed**

Gradient checkpointing was verified on for all 62 checkpointable modules including the vision
tower, and expandable segments were enabled. Inference on the same GPU is completely stable
(100+ generations, a live API). Full write-up in
[DECISIONS §10a](DECISIONS.md#10a-why-training-does-not-run-on-the-local-gpu). Re-runnable
with `bash scripts/sweep_train_memory.sh`.

**Kaggle is therefore required for the training run, not merely preferred.**

### In progress

- [ ] v2 training run on Kaggle: CORD + 200 synthetic. This is what should move
      `ice` / `if_number` / `date` / `currency` off zero — the v1 adapter was trained on
      CORD only, which contains none of those fields.
- [ ] Fine-tuned evaluation on both test sets, and the before/after table in the README

### Next, in priority order

1. More synthetic layouts (3–4 templates) — biggest limitation, zero GPU cost
2. More epochs with a validation loss and early stopping
3. Re-run the pipeline on `Qwen3-VL-2B-Instruct`, which now exists (config change only)
4. Constrained decoding for a 100 % parse rate by construction
5. Per-field confidence scores

## History worth keeping

The v1 adapter (CORD-only, 1 epoch, ~2 h 04 m on Kaggle, loss 0.144 → 0.045) showed a large
improvement on every field CORD actually contains, measured under the *old* string-equality
metric on 20 validation examples. Those numbers are superseded and are not reported in the
README, but the run confirmed the pipeline works end to end.

Bugs found and fixed along the way are written up in
[DECISIONS.md §10](DECISIONS.md#10-things-that-broke-and-the-fixes).
