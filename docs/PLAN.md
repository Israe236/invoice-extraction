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
- [x] **259 tests**, none requiring a GPU or a download
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

| field | F1 | support |
|---|---|---|
| subtotal | 0.722 | 50 |
| tax | 0.722 | 50 |
| total | 0.825 | 50 |
| items | 0.083 | 161 |
| ice | 0.825 | 50 |
| if_number | 0.949 | 50 |
| invoice_number | 0.809 | 50 |
| date | 0.809 | 50 |
| currency | 0.507 | 50 |

micro-F1 **0.569**, parse rate **48/50 (96 %)**. These are the numbers from the re-score on the
date-stable test set (DECISIONS §10); the first synthetic baseline, on dates that drifted by a
day, read micro-F1 0.567 and is superseded.

Three things this changed about the plan:

1. **The parse rate gap (44 % on photos vs 96 % on renders) was the main baseline weakness**,
   not raw reading accuracy. Fine-tuning closed it: 50/50 on both sets.
2. **ICE and IF were already strong zero-shot** (0.825 / 0.949) — the opposite of the v1
   expectation that they were the big gap. The earlier "ice F1 0.000" was a *support-0
   artefact of measuring on CORD*, not evidence the model could not read an ICE. Worth
   remembering: it was a reporting bug masquerading as a model finding.
3. **Line items were the real weakness** (0.083), and the cause is measured: 124 of the 137
   lines the base model matched by name took the unit price from the P.U. distractor column.
   After fine-tuning, 161 of 161 lines are correct. See `scripts/inspect_items.py`.

### Local training: made to fit, v2 trained

The v2 run (800 CORD + 200 synthetic, 1 epoch) **trained locally on the RTX 4050 in 25
minutes**, peak VRAM 4.16 GB, loss 0.113 → 0.030. It first failed with driver faults; four
fixes made it fit — a VRAM cap below physically free memory, the vision tower moved outside
autograd, frozen fp32 tensors cast back to bf16, and gradient checkpointing actually active.
Full write-up in
[DECISIONS §10a](DECISIONS.md#10a-how-training-was-made-to-fit-on-the-local-gpu).
Config: `configs/train_local.yaml`. The Kaggle notebook still works for machines without a GPU.

### Fine-tuned results (done)

Adapter `qwen2vl-2b-lora-v2`, same prompt, same decoding, same scoring code as the baseline,
50 documents per set:

| | Base micro-F1 | Fine-tuned micro-F1 | Base parse rate | Fine-tuned parse rate |
|---|---|---|---|---|
| CORD-v2 test | 0.346 | **0.827** | 22/50 | **50/50** |
| Synthetic held-out | 0.569 | **1.000** | 48/50 | **50/50** |

Per-field tables are in the README, generated by `scripts/results_table.py`. The synthetic
1.000 is an upper bound, not a production estimate: the test invoices come from the same single
template the model was trained on. CORD line items (0.743) are the weakest real-world field.

The synthetic numbers here supersede the first fine-tuned run, which was scored on a test set
whose dates had drifted by a day (see DECISIONS §10). Both models were re-scored on the fixed,
byte-identical set.

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
