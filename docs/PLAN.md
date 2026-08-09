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
1. Zero-shot baseline, per-field F1, before any training.
2. Fine-tuned model, same metrics, honest comparison.
3. validate.py — accounting rules: HT + TVA = TTC, line items sum
   to subtotal, date sanity. Outputs a confidence flag for human
   review. This is the part that makes it a product, not a demo.
4. README with a demo GIF (not a live Space — free Spaces are CPU
   only and too slow).

## Order of work
1. [x] Inspect CORD structure, decide the target schema
2. [x] data.py — loading + chat-format conversion
3. [x] evaluate.py — per-field F1, handle malformed JSON output
4. [x] Zero-shot baseline
5. [ ] LoRA fine-tune  <- CURRENT (script ready, run pending on Kaggle)
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

Zero-shot baseline (Qwen2-VL-2B-Instruct, untrained), final 6-field
schema, 20 CORD-v2 validation examples:
  subtotal F1 0.000, tax F1 0.118, total F1 0.069, items F1 0.039,
  ice F1 0.000, if_number F1 0.000 (last two: CORD has neither, so
  0 is expected/uninformative here, not a real signal).
This is meaningfully lower than the earlier 4-field run (subtotal
0.353, tax 0.286, total 0.308, items 0.130). Confirmed why by
diffing raw outputs on the same images across both prompts: the
dominant failure mode (model returns a flat list instead of the
{items, subtotal, ...} dict) pre-dates the schema change, but adding
ice/if_number visibly confuses the untrained model further — e.g. one
example hallucinated a field that was never requested ("morocco": "0")
apparently triggered by the word "Moroccan" in the prompt, and got
subtotal/tax/total all wrong on the same turn. This is the honest
"before" number: same schema/prompt the fine-tuned model will be
trained and judged against, so it's a fair comparison point, not an
apples-to-oranges one.

train.py (LoRA/QLoRA via peft) is written and its trickiest part (the
label-masking collator) verified correct against real data — but not
executed. Training runs on Kaggle, watched manually — local is for
code, debugging, and evaluation only, per the project's compute plan.

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

Only remaining gap before the fine-tune: schema is now final and
stable (this was the last change), so no more baseline re-runs should
be needed after training.

Next: take train.py + configs/train.yaml to Kaggle and run the actual
fine-tune, mixing CORD + synthetic examples. After that, re-run
evaluate.py on the fine-tuned model for the real before/after
comparison, then README + demo GIF.