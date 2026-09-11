# Decisions

Every important choice in this project, in plain language: what the options were, what was
picked, and why. Written to be read out loud in an interview.

---

## 1. Why a vision-language model instead of OCR plus rules

The obvious approach is Tesseract (or a cloud OCR) followed by regular expressions and
positional heuristics. It is cheaper and it needs no GPU.

It also breaks constantly. OCR gives you a bag of words and boxes; turning that into
"this number is the TVA and that number is the unit price of line 3" means writing layout
rules, and every supplier has a different layout. The rules become a per-supplier
maintenance burden, which is exactly the manual work the project is supposed to remove.

A VLM reads the document the way a person does — it sees that a number sits in a column
headed *Montant* on the row labelled *Prestation de conseil*. Layout understanding comes for
free instead of being hand-coded.

**The honest counter-argument:** a VLM will occasionally invent a number, which OCR will
never do. That is precisely why this project has a validation layer (§12). The combination —
a flexible reader plus a strict arithmetic checker — is stronger than either alone.

---

## 2. Why Qwen2-VL-2B-Instruct

The original plan named **Qwen3-VL 2B**. When the project started, that model did not exist:
Qwen3-VL's smallest size was 4B, which does not fit a 6 GB card for training. So the choice
became Qwen2-VL-2B-Instruct — a real 2B vision-language model with a mature `max_pixels`
control, which is the single most important knob on a small GPU.

`Qwen/Qwen3-VL-2B-Instruct` **has since been released**. It was not adopted because the
measured before/after results in this repository were produced on Qwen2-VL, and switching
would mean discarding them and re-running a fresh baseline plus a fresh ~2.5 h training run
for a comparison nobody has asked for yet. Re-running the same pipeline on Qwen3-VL-2B is the
most obvious next step, and the code needs only a config change to do it.

Sizes considered:

| Model | Verdict |
|---|---|
| Qwen2-VL-7B | Will not train in 6 GB, and too slow on free Kaggle hours. |
| **Qwen2-VL-2B** | **Chosen.** ~1.7 GB in 4-bit at inference; trains on a free P100. |
| SmolVLM-500M | Would certainly fit, but the accuracy ceiling on dense tables is lower. Kept as a fallback that was never needed. |

---

## 3. Why QLoRA and not full fine-tuning

Full fine-tuning of a 2B model needs roughly: 4 GB for bf16 weights, 4 GB for gradients and
around 16 GB for Adam's optimiser state. That is ~24 GB before a single activation. It is not
happening on free compute.

**QLoRA** does three things at once:

1. **4-bit NF4 quantisation** of the frozen base weights — 2B parameters in ~1.3 GB instead
   of ~4.4 GB. NF4 is information-theoretically matched to the roughly-normal distribution of
   neural network weights, so it loses less than plain int4.
2. **Double quantisation** — the quantisation constants are themselves quantised. A small win
   (~0.4 bits per parameter), free.
3. **LoRA adapters** — instead of updating a weight matrix `W`, learn a low-rank correction
   `BA` where `B` and `A` are thin. At `r=16` that is about 1 % of the parameters, so the
   optimiser state is ~1 % of the size.

The result trains on a free Kaggle GPU and produces a **180 MB** artefact instead of a 4.4 GB
one. The base model is unchanged and re-downloadable, so the adapter is the only thing that
has to move between machines.

**Cost:** 4-bit introduces some quantisation error, and LoRA cannot express every update that
full fine-tuning could. For teaching a model a fixed output format on a narrow domain, that
ceiling is far above what is needed — which the results bear out.

---

## 4. Why these hyperparameters

| Setting | Value | Reasoning |
|---|---|---|
| `lora_r` | 16 | The task is "always emit this JSON schema", which is a formatting behaviour, not new world knowledge. Low rank is enough. r=8 risked underfitting the table-reading; r=64 spends memory on capacity the task does not need. |
| `lora_alpha` | 32 | The conventional `2 × r`. Alpha/r is the effective scale on the adapter; 2 is the standard starting point and there was no evidence to move it. |
| `lora_dropout` | 0.05 | Light regularisation. With only ~1000 training examples, overfitting is a genuine risk. |
| `target_modules` | all attention + MLP projections | Attention-only (`q,k,v,o`) is the cheaper classic recipe, but including the MLP projections (`gate,up,down`) consistently helps on structured-output tasks, and the memory cost at r=16 is small. |
| `learning_rate` | 2e-4 | The standard QLoRA learning rate. LoRA tolerates a rate ~10× higher than full fine-tuning because only the small adapters move. |
| `batch_size` | 1 | Not a choice. One image's visual tokens dominate memory; two do not fit. |
| `grad_accum` | 8 | Recovers an effective batch of 8 for gradient-noise purposes at the memory cost of 1. |
| `optim` | `paged_adamw_8bit` | 8-bit optimiser state, and *paged* means a transient memory spike gets moved to CPU RAM rather than crashing the run. |
| `epochs` | 1 | A first honest pass. On 1000 examples one epoch already moved every field substantially; more epochs is the next experiment, not a claim. |
| `gradient_checkpointing` | on | Trades ~30 % speed for a large activation-memory saving. Mandatory here. |

**The vision tower is frozen.** LoRA is applied to the language model only. The visual encoder
already knows how to see text and tables; what the model lacks is the habit of emitting *our*
JSON schema, which is a language-side behaviour. Training the vision tower would cost memory
for the part of the problem that is not broken.

---

## 5. The 6 GB constraint, and what it forced

`nvidia-smi` reports **6141 MiB** on the RTX 4050 laptop GPU. Every decision below traces
back to that number.

- **Inference is local, training is not.** 4-bit inference uses ~1.7 GB measured, so all
  evaluation runs on the laptop. A training step adds activations, gradients and optimiser
  state; that goes to Kaggle's free tier (P100 16 GB or 2×T4, ~30 GPU-hours/week).
- **`max_pixels` is capped.** This is the single most important line in the config. Left
  uncapped, Qwen2-VL will happily turn a phone photo into several thousand visual tokens and
  OOM instantly.
- **Everything is batch size 1.**

### Why `max_pixels = 401408`

This is ~512 visual tokens. It was chosen by measurement, not by taste:

- At `max_pixels=1003520` (Qwen2-VL's high-resolution preset) one training step took **~217
  seconds**. One epoch would have been ~18 hours — longer than a Kaggle session and most of a
  week's quota.
- At `401408` a step is roughly 2.5× faster and the epoch fits comfortably.

The genuinely surprising part: **the lower resolution also scored better zero-shot.** Two
earlier runs at `1003520` scored notably worse on `total` and `items`. Before trusting that,
the obvious alternative explanation — run-to-run sampling noise — was checked and ruled out:
Qwen2-VL's own `generation_config` sets `top_k=1`, which makes decoding deterministic
regardless of `do_sample`. Identical inputs give identical outputs, so the resolution itself
was responsible. Receipts are low-information images; more pixels mostly bought more tokens
for the model to lose track in.

### Why the synthetic invoices are cropped to their content

A generated invoice with two line items fills the top third of an A4 page and leaves the rest
white. Because the processor is capped at `max_pixels`, the whole page — blank space included
— gets downscaled by about 9×, and the text ends up near-illegible while most of the token
budget is spent on empty paper.

Cropping to the printed area before degradation recovers **3.3–4.5× more effective resolution
on the text at the same token cost** (measured across seeds). This is free accuracy, and it is
the kind of preprocessing detail that matters far more on a small model than on a large one.

---

## 6. Why synthetic data at all

CORD-v2 is 1000 real receipt photographs, and it is genuinely valuable: crumpled paper, odd
angles, motion blur, thermal-printer artefacts. No generator produces that kind of noise
convincingly.

But CORD is Indonesian retail receipts. It contains **no ICE, no IF, no invoice number, no
date and no currency field** — which is to say, none of the fields that make this project
about Moroccan invoices rather than a CORD reimplementation. Training on CORD alone produces
a model that scores 0 on exactly the fields the project exists for, and no public dataset of
Moroccan invoices exists to fill the gap.

So the invoices are generated: Faker → Jinja HTML → WeasyPrint PDF → PyMuPDF raster →
Albumentations degradation (rotation, brightness/contrast, Gaussian noise, blur, JPEG
compression). The degradation step is what stops the model from learning "clean render =
invoice"; without it, synthetic training data teaches a model that falls apart on a photo.

**Nothing here touches real client data.** Every company name, address, ICE and IF is
generated. That is a hard project constraint, and synthetic data is how it is satisfied
without giving up the domain.

### The unit-price column is a deliberate trap

The invoice template prints both a **P.U.** (unit price) and a **Montant** (line total)
column, and ground truth is the *Montant*. Real invoices always show both, and picking the
wrong column is the most common real-world extraction failure. An easier synthetic set that
omitted the distractor would produce a higher number in the results table that would not
survive contact with a real document.

The generator asserts that `unit_price × quantity = line_total`, so the distractor is
arithmetically real rather than a random decoy.

### Train and test synthetic invoices use disjoint seeds

Faker is deterministic: the same seed produces a byte-identical invoice. Training draws from
`TRAIN_SEED_BASE = 42`, evaluation from `EVAL_SEED_BASE = 900_000`. A shared seed would put
an identical document in both train and test and make every synthetic number in the README
meaningless. There is a test that asserts the ranges cannot collide.

---

## 7. Why per-field precision / recall / F1

A single accuracy number would hide everything interesting. "82 % accurate" does not tell an
accountant whether the model is reliable on totals and hopeless on dates, which is exactly
what they need to know before deciding what to automate and what to review.

- **Precision** — of the values the model produced, how many were right. Low precision means
  it invents numbers, which is the dangerous failure.
- **Recall** — of the values on the document, how many it found. Low recall means it misses
  fields, which is annoying but safe.
- **F1** — the balance, for ranking fields against each other.

Line items are matched as a **multiset**, not a set. Two identical lines on an invoice are two
lines; a set comparison would silently collapse them and award a perfect score to a prediction
that dropped one. There is a test for this.

### Comparison is field-aware, not string equality

The first version of the metric compared raw strings. That punished the model for things no
accountant would call a mistake:

| Gold | Prediction | String equality | Field-aware |
|---|---|---|---|
| `1234.56` | `1 234,56` | wrong | correct |
| `2024-04-03` | `03/04/2024` | wrong | correct |
| `Développement` | `Developpement` | wrong | correct |
| `001234567000078` | `001 234 567 000 078` | wrong | correct |

Every field now goes through the normaliser for its type — money, date, identifier, free text
— before comparison. Amounts match within a cent (or 0.5 % on large invoices), dates are
parsed day-first because that is how French and Moroccan invoices are written, and text is
accent-folded and case-folded.

**This is a metric change, so old numbers are not comparable to new ones.** Both the base
model and the fine-tuned model are re-measured under the current metric, using the same code
path, so the comparison between the two columns stays fair.

### Two numbers a plain F1 table hides

**JSON parse rate.** A model that is accurate when it parses but emits prose the rest of the
time is not deployable, and averaging that away would be dishonest. On the CORD test split
the base model returned parseable JSON in only **22 of 50 cases (44 %)**. That single number
says more about why fine-tuning is needed than any F1 does.

**Support.** CORD-v2 has no ICE. Under the old reporting, "ice F1 = 0.000" looked like the
model always got it wrong, when in fact it was never tested. Fields with zero gold values now
print as `n/a`, and support is shown next to every row.

### Why the test split, and 50 examples

Earlier runs used 20 examples from the `validation` split. The current configuration uses
**50 examples from the `test` split** — which training never touches — because 20 examples is
too few for a per-field F1 to mean much when a field appears on only half the documents.

---

## 8. Why JSON output needs a repair layer

An untrained VLM asked for JSON will frequently return it wrapped in a markdown fence, prefaced
with "Here is the extracted data:", or with a trailing comma. Scoring all of that as total
failure would understate the base model and make the fine-tuned improvement look bigger than
it is.

`parse_model_output` tries, in order: the raw text, the text with a code fence stripped, the
first brace-balanced span (with a scanner that ignores braces inside strings), and that span
with trailing commas removed. A single-element array wrapping the object is unwrapped, because
that is unambiguous. An array of *several* objects is **not** unwrapped — which invoice is it?
— and counts as a parse failure rather than a guess.

Recovering the base model's output as generously as possible is what makes the "before"
column trustworthy.

---

## 9. Why the validation layer has three outcomes, not two

This started as pass/fail and it was **wrong**, in a way worth explaining because it is a good
example of a bug that produces plausible-looking output.

A missing field returned `passed=False`. So every CORD receipt was reported as an *arithmetic
violation* — not because the arithmetic was wrong, but because receipts have no Moroccan ICE
number. The report was full of red flags that meant nothing, which is worse than no report:
it trains the user to ignore it.

The fix is a third state:

- **PASS** — the rule ran and held.
- **FAIL** — the rule ran and was violated.
- **SKIPPED** — the rule could not run, because the document does not carry the fields it
  needs.

Skipped checks never raise severity. They are listed separately as *unverified*, because
"we could not check this" is a genuinely different message to an accountant than "this is
wrong".

### Severity

Not every failure is equal. Broken arithmetic means *do not post this*. A 14-digit ICE means
*someone should glance at this*. So each rule carries the severity it raises when it fails,
and the report's overall severity is the worst one that actually fired:

- `ok` → every applicable check passed; safe to post automatically.
- `warning` → formatting or plausibility problem; human glance.
- `error` → the document contradicts itself; human review required.

### The rules

| Rule | Check | On failure |
|---|---|---|
| `required_fields` | a total is present at all | error |
| `tax_arithmetic` | Total HT + TVA = Total TTC | error |
| `items_sum` | line items sum to Total HT | error |
| `tax_rate_plausible` | implied VAT rate is 20/14/10/7 % | warning |
| `date_valid` | parses, not in the future, not >10 years old | warning |
| `ice_format` | exactly 15 digits | warning |
| `if_format` | 7–9 digits | warning |
| `currency_known` | a recognised code | warning |

Tolerances are both absolute (2 cents) and relative (0.5 %), because a fixed 1-cent tolerance
is absurd on an invoice for 400 000 MAD. The `items_sum` tolerance grows with the number of
lines, since each line was rounded independently before being summed.

**This layer is the product argument.** A model output you cannot trust is not worth much. A
model output that comes with "the arithmetic closes, and here is which checks were run" is
something a finance team can actually wire into a workflow.

### Why real CORD receipts get flagged

Roughly one CORD receipt in five fails `items_sum`. That is expected, not a bug: CORD's schema
has a `service_price` field that appears on only 14 of 100 receipts, and it was deliberately
excluded from the schema as a simplification. Those receipts genuinely do not reconcile under
*our* schema. This is written down rather than quietly tolerated, because it is the kind of
thing that looks like a model failure in a results table and is not.

---

## 10. Things that broke, and the fixes

These are the real ones, kept because the debugging is the interesting part.

**The model in the original plan did not exist.** "Qwen3-VL 2B" was specified before checking;
Qwen3-VL's smallest size at the time was 4B. Fixed by verifying against Hugging Face and
switching to Qwen2-VL-2B-Instruct. Lesson: check that the artefact exists before planning
around it.

**The local GPU was silently broken.** `torch` had resolved to a CUDA 13 build while the
driver only supports 12.7, and several `nvidia-*-cu12` packages were half-installed. CUDA was
simply unavailable with no useful error. Fixed by pinning torch and torchvision to matching
CUDA 12.6 builds through an explicit `[[tool.uv.index]]` in `pyproject.toml`.

**`Trainer` silently deleted the data before the collator saw it.** HuggingFace's `Trainer`
wraps the dataset in a `RemoveColumnsCollator` that drops any key not in the model's
`forward()` signature. Our raw `image` and `messages` keys were exactly that, so the custom
collator received empty dictionaries. Nothing errored; the batches were just wrong. Fixed with
`remove_unused_columns=False`. This is a good example of a default that is helpful for text
models and actively harmful for a custom multimodal collator.

**Two visible GPUs corrupted the 4-bit weights.** Kaggle's 2×T4 made `Trainer` auto-wrap the
model in `torch.nn.DataParallel`, which replicates modules across devices — and replicating
4-bit quantised weights corrupts them. It surfaced as a baffling `StopIteration` deep inside
`forward()` because a submodule ended up with zero parameters. Fixed by setting
`CUDA_VISIBLE_DEVICES=0` *before* torch initialises CUDA. This is single-GPU QLoRA; there was
never a reason for a second device to be visible.

**A resolution setting nearly burned a week's GPU quota.** Covered in §5: `max_pixels=1003520`
meant ~217 s per step, ~18 h per epoch. Caught by watching the first few steps rather than
launching and leaving. The smoke-test cell in the Kaggle notebook exists because of this.

**The validation layer flagged everything.** Covered in §9.

**The metric punished correct answers.** Covered in §7.

---

## 11. Why `uv`, and why the repo lives where it does

`uv` resolves and installs an order of magnitude faster than pip, and `uv.lock` makes the
environment reproducible — which matters when the CUDA build of torch is load-bearing.

The repository lives on the **WSL2 ext4 filesystem**, not under `/mnt/c`. `/mnt/c` is a DrvFs
mount: file I/O is slow, and Python virtualenvs on it break in subtle ways. Given that all the
ML tooling is Linux-first (bitsandbytes, triton), WSL is where the work happens.

---

## 12. Known limitations

Stated plainly, because a portfolio project that claims no weaknesses is not credible.

- **The test sets are small.** 50 CORD receipts and 50 synthetic invoices. Enough to see a
  large effect; not enough for tight confidence intervals on a per-field F1.
- **Synthetic invoices come from one template.** Degradation varies the pixels, but the layout
  does not vary. The model has seen one way of arranging a Moroccan invoice, and real
  suppliers use many. This is the most important gap.
- **No real Moroccan invoices have been tested.** By design — no real client data — but it
  means the ICE/IF/date numbers are measured against documents from the same generator that
  produced the training data. They should be read as an upper bound.
- **One epoch.** No learning-rate sweep, no early stopping on a validation loss, no seed
  variance. The numbers are one run.
- **CORD's `service_price` is excluded**, so some real receipts legitimately fail `items_sum`
  (§9).
- **Latency is not production-grade.** Roughly 50 s per document on a 6 GB laptop GPU in
  4-bit. Fine for a demo, too slow for bulk processing without batching and better hardware.

---

## 13. What comes next, in priority order

1. **More synthetic layouts.** Three or four distinct invoice templates would attack the
   biggest limitation directly and cost no GPU time.
2. **More epochs, with a validation loss.** One epoch was a first honest pass; the obvious
   next experiment is 2–3 epochs with early stopping.
3. **Qwen3-VL-2B-Instruct.** Now that it exists, re-run the same pipeline on it. The code
   needs a config change and nothing else, which is the point of keeping hyperparameters in
   YAML.
4. **Constrained decoding.** A grammar that forces valid JSON would take the parse rate to
   100 % by construction rather than by training.
5. **Confidence per field**, so the validation layer can say *which* number it doubts, not
   just that the document does not reconcile.
