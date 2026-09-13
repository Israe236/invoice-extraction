# Project plan

Living status document. The *reasoning* behind each choice lives in
[DECISIONS.md](DECISIONS.md); this file tracks what is done and what is next.

## Goal

Fine-tune a small VLM to extract structured JSON from invoices and receipts, with a
validation layer that flags documents whose arithmetic does not close. Portfolio project —
the README and the honesty of the metrics matter as much as the model.

## Constraints

- Free compute only. Local RTX 4050 laptop GPU (6 GB, measured 6141 MiB, ~4.9 GB actually free
  under Windows).
- Everything runs locally: code, evaluation (4-bit inference ~1.7 GB) and, since the memory
  fixes in DECISIONS §10a, training (peak 4.16 GB, 25 min per epoch).
- Kaggle's free tier (P100 16 GB or 2×T4) remains the documented path for anyone without a
  local GPU, via `notebooks/train_kaggle.ipynb`.
- No real client data, ever. Public datasets or synthetic only.

## Schema (9 fields, stable)

`items[{name, qty, price}]`, `subtotal`, `tax`, `total`, `ice`, `if_number`,
`invoice_number`, `date`, `currency`.

Started as 4 fields taken from CORD-v2's actual field frequencies, extended with `ice` and
`if_number` once the synthetic generator made clear those are the project's differentiator,
then extended again with `invoice_number`, `date` and `currency` — the generator already
produced all three but never exposed them as ground truth, so the fields an accountant most
needs were untestable.

## Status: complete

### Done

- [x] CORD-v2 loading and chat-format conversion
- [x] Synthetic French/Moroccan invoice generator (Faker → Jinja → WeasyPrint → PyMuPDF →
      Albumentations), with content cropping, a unit-price distractor column, and dates from a
      fixed window so a seed produces the same invoice on any day
- [x] Per-field P/R/F1 with field-aware normalisation, JSON parse rate, and per-field support
- [x] Robust JSON recovery from malformed model output
- [x] Validation layer: 8 rules, three outcomes (pass/fail/skipped), severity levels
- [x] **259 tests**, none requiring a GPU or a download
- [x] FastAPI `/extract`, `/validate`, `/health` + React/Vite frontend
- [x] Zero-shot baseline on both test sets — CORD-v2 *test* split and held-out synthetic, 50 each
- [x] QLoRA fine-tuning on the 6 GB laptop GPU (v2 adapter: 800 CORD + 200 synthetic, 1 epoch)
- [x] Fine-tuned evaluation on both test sets, through the same code path as the baseline
- [x] Validation flag tested against gold labels (`scripts/validation_value.py`)
- [x] Live API requests against a real image, base model and fine-tuned adapter
- [x] README with results tables and figures generated from `results/`, DECISIONS.md
- [x] Kaggle training notebook with a 3-step smoke test, as the no-GPU fallback

### Results

Same prompt, same decoding, same scoring code for both models, 50 documents per set:

| | Base micro-F1 | Fine-tuned micro-F1 | Base valid JSON | Fine-tuned valid JSON |
|---|---|---|---|---|
| CORD-v2 test | 0.346 | **0.827** | 22/50 | **50/50** |
| Synthetic held-out | 0.569 | **1.000** | 48/50 | **50/50** |

Per-field tables are in the README, printed by `scripts/results_table.py`. The synthetic 1.000
is an upper bound, not a production estimate: the test invoices come from the same single
template the model was trained on. CORD line items (0.743) are the weakest real-world field.

Validation flag, across all four runs: all 23 documents with wrong totals were flagged, and no
unflagged document carried a wrong total. On real receipts it is conservative — 9 of the 10
fine-tuned CORD receipts it flagged had correct totals, and 15 of the 50 CORD gold answers fail
the rules on their own.

> These numbers are **not comparable** to earlier revisions of this file. The metric changed
> (field-aware normalisation instead of string equality), the CORD split changed (test instead
> of validation), n changed (50 instead of 20), and the synthetic set was regenerated once dates
> stopped drifting between days. Both models were measured under the final setup.

### What the baseline taught, and what fine-tuning did about it

1. **The parse-rate gap (44 % on photos vs 96 % on renders) was the main baseline weakness**,
   not raw reading accuracy. Fine-tuning closed it: 50/50 on both sets.
2. **ICE and IF were already strong zero-shot** (0.825 / 0.949) — the opposite of the v1
   expectation that they were the big gap. The earlier "ice F1 0.000" was a *support-0 artefact
   of measuring on CORD*, not evidence the model could not read an ICE. A reporting bug had been
   masquerading as a model finding.
3. **Line items were the real weakness** (0.083 on synthetic). 124 of the 137 lines the base
   model matched by name took the unit price from the P.U. distractor column. After
   fine-tuning, 161 of 161 lines are correct.

### Training

The v2 run (800 CORD + 200 synthetic, 1 epoch) trained on the RTX 4050 in 25 minutes, peak
VRAM 4.16 GB, loss 0.113 → 0.030. It first failed with driver faults; four fixes made it fit —
a VRAM cap below physically free memory, the vision tower moved outside autograd, frozen fp32
tensors cast back to bf16, and gradient checkpointing actually active. Full write-up in
[DECISIONS §10a](DECISIONS.md#10a-how-training-was-made-to-fit-on-the-local-gpu).
Config: `configs/train_local.yaml`.

## Next, in priority order

1. More synthetic layouts (3–4 templates) — the biggest limitation, zero GPU cost, and what
   would make the synthetic score worth trusting
2. A richer schema — a tax-inclusive flag and an "other charges" field — to remove most of the
   validation false alarms on real receipts
3. Several seeds and a validation loss, for error bars and an evidence-based epoch count
4. Re-run the pipeline on `Qwen3-VL-2B-Instruct`, which now exists (config change only)
5. Faster inference: merge the adapter and batch requests
6. Per-field confidence scores

## History worth keeping

The v1 adapter (CORD-only, 1 epoch, ~2 h 04 m on a Kaggle T4, loss 0.144 → 0.045) showed a
large improvement on every field CORD contains, measured under the *old* string-equality metric
on 20 validation examples. Those numbers are superseded and are not reported in the README, but
the run confirmed the pipeline worked end to end.

Bugs found and fixed along the way are written up in
[DECISIONS.md §10](DECISIONS.md#10-things-that-broke-and-the-fixes).
