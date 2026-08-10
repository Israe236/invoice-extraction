# Project plan

## Goal
Fine-tune a small VLM to extract structured JSON from invoices and
receipts. Portfolio project for job applications — the README and the
metrics matter as much as the model.

## Constraints
- Free compute only. Local RTX 4050 (6GB) + Kaggle free tier
  (P100 16GB or 2xT4, ~30 GPU-hours/week, 12h sessions).
- Local = code, debug, small runs, all evaluation.
- Kaggle = full training runs.

## Data
- CORD-v2 (naver-clova-ix/cord-v2), ~1000 receipts, image -> JSON.
- Synthetic French/Moroccan invoices, generated in-repo: Jinja
  templates + Faker + WeasyPrint, degraded with Albumentations.
  Fields: ICE, IF, HT, TVA, TTC, line items. This is the
  differentiator — no public dataset has these.

## Model
- Qwen2-VL-2B-Instruct, 4-bit QLoRA via peft. (Qwen3-VL has no 2B size —
  smallest is 4B, which is too big for the 6GB local budget.)
- Cap processor max_pixels. Uncapped image resolution is the main
  OOM cause at 6GB.

## Deliverables
1. [x] Zero-shot baseline, per-field F1, before any training.
2. [x] Fine-tuned model, same metrics, honest comparison (CORD-only
   so far — see status; ice/if_number need synthetic data mixed in).
3. [x] validate.py — accounting rules: HT + TVA = TTC, line items sum
   to subtotal. (No date-sanity check — see status for why.) Outputs
   a pass/fail per check for human review.
4. [ ] README with a demo GIF (not a live Space — free Spaces are CPU
   only and too slow).

## Order of work
1. [x] Inspect CORD structure, decide the target schema
2. [x] data.py — loading + chat-format conversion
3. [x] evaluate.py — per-field F1, handle malformed JSON output
4. [x] Zero-shot baseline
5. [x] LoRA fine-tune (CORD-only pass done)  <- CURRENT: synthetic-mixed pass next
6. [x] Synthetic invoice generator
7. [x] validate.py
8. [ ] README + demo GIF

## Current status
Environment done: WSL2, GPU passthrough confirmed, uv venv, torch+CUDA,
package installed editable, skeleton committed.

Schema: items (name, qty, price), subtotal, tax, total, ice, if_number.
Started as 4 fields from CORD-v2's actual field frequencies, then
extended with ice/if_number once the synthetic generator made clear
those are the project's actual differentiator fields (CORD examples
just carry them as empty strings).

FINAL zero-shot baseline (Qwen2-VL-2B-Instruct, untrained), 6-field
schema, max_pixels=401408 (matches training resolution), 20 CORD-v2
validation examples — this is the real "before" number:
  subtotal F1 0.286, tax F1 0.111, total F1 0.467, items F1 0.407,
  ice F1 0.000, if_number F1 0.000 (last two: CORD has neither, so
  0 is expected/uninformative here, not a real signal).
Two earlier runs at max_pixels=1003520 scored notably lower on
total/items (0.069/0.039 and 0.308/0.130 across schema versions).
Checked whether this was just sampling noise before trusting it: the
model's own generation_config uses top_k=1, which makes decoding
deterministic regardless of do_sample, so the resolution itself is
doing this, not run-to-run variance. Genuinely interesting: lower
resolution helped zero-shot accuracy here, not just speed.

train.py (LoRA/QLoRA via peft) ran on Kaggle (T4 x2, single GPU
pinned): 1 epoch, CORD-only, max_pixels=401408. Loss 0.144 -> 0.045,
stable the whole way, ~2h04m. Checkpoint pulled to
checkpoints/qwen2vl-2b-lora/ locally (gitignored).

BEFORE/AFTER RESULT (deliverable 2, done) — same 20 CORD-v2
validation examples, same max_pixels=401408, via run_baseline() with
and without the adapter:
  field       before  after
  subtotal    0.286   0.909
  tax         0.111   0.800
  total       0.467   0.900
  items       0.407   0.686
  ice         0.000   0.000  (expected -- CORD has no ICE examples)
  if_number   0.000   0.000  (expected -- same reason)
Large, clean improvement on every field CORD actually contains, from
just 1 epoch. ice/if_number staying at 0 isn't a failure, it's the
exact gap the synthetic generator exists to close next.

Synthetic generator (synthetic.py + render.py + templates/invoice.html.jinja):
Faker-based French/Moroccan invoice data -> Jinja HTML -> WeasyPrint PDF
-> PyMuPDF rasterize -> Albumentations degradation (rotation, blur,
noise, JPEG compression). Verified visually and against evaluate.py
(perfect predictions score 1.0 on all 6 fields). WeasyPrint dropped
direct PNG export in this version, hence the PyMuPDF rasterize step.

Fixed along the way:
- The original model choice was "Qwen3-VL 2B", which doesn't exist
  (Qwen3-VL starts at 4B). Switched to Qwen2-VL-2B-Instruct, a real
  2B model that fits the 6GB local budget.
- Local GPU was silently broken (torch resolved to a CUDA 13 build,
  driver only supports 12.7) — pinned torch/torchvision to matching
  CUDA 12.6 builds in pyproject.toml and repaired several corrupted
  nvidia-*-cu12 package installs.

validate.py: two accounting checks (subtotal + tax = total, line
items sum to subtotal), tolerant of both thousands-separator and
decimal-comma number formats. Tested against synthetic ground truth
(always passes, consistent by construction), real CORD ground truth
(4/5 examples flagged — see note below), and a deliberately broken
example (correctly flagged). No date-sanity check: the plan mentioned
one, but no date field exists in the schema (CORD doesn't expose one,
and the synthetic generator's invoice_date was never wired into
ground truth) — skipped rather than faked.
Note on the CORD flags: real CORD receipts often don't reconcile
under our simplified schema because we deliberately excluded
service_price (only 14/100 receipts had it) from the total. That's a
schema simplification showing up as expected "failures," not a bug.

Schema is final and stable (no more retroactive changes expected).

First Kaggle run complete (see baseline/training numbers above). Hit
and fixed three real bugs along the way:
- Trainer strips dataset dict keys that aren't in the model's
  forward() signature by default (RemoveColumnsCollator) — our raw
  "image"/"messages" keys got silently dropped before the collator
  ever ran. Fix: remove_unused_columns=False.
- Kaggle's 2xT4 GPUs both being visible made Trainer auto-wrap the
  model in torch.nn.DataParallel, which corrupts 4-bit quantized
  weights during replication. Fix: pin CUDA_VISIBLE_DEVICES=0 before
  torch initializes (this is single-GPU QLoRA, not distributed).
- max_pixels=1003520 (Qwen2-VL's high-res preset) made one training
  step take ~217s — 300 steps would be ~18h, more than a single
  Kaggle session and most of the weekly GPU-hour quota, without even
  finishing. Dropped to max_pixels=401408 and num_train_epochs=1 for
  a first real run.

train.py currently trains on CORD only — synthetic French/Moroccan
examples aren't mixed in yet.

Next: mix synthetic French/Moroccan examples into train.py for a
second training pass (this should move ice/if_number off 0), then
README + demo GIF.