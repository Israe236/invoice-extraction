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

The result trains on the 6 GB laptop GPU in 25 minutes (once the memory fixes in §10a were in
place) and produces a **74 MB** adapter — 18.5 M trainable parameters — instead of a 4.4 GB
model. The base model is unchanged and re-downloadable, so the adapter is the only thing that
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

- **Everything runs on the laptop — but training only after four fixes.** 4-bit inference
  uses ~1.7 GB. A training step also holds activations, gradients and optimiser state, and
  out of the box that did not fit. With the fixes in §10a it peaks at 4.16 GB and one epoch
  takes 25 minutes. The Kaggle notebook stays as the path for anyone without a GPU.
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

**It worked, and the effect is large.** On the synthetic test set the base model scores 0.083
F1 on line items while scoring 0.825 on `total` and 0.949 on `if_number`. Pairing every
predicted line with a gold line and classifying where its price came from
([`scripts/inspect_items.py`](../scripts/inspect_items.py)):

| Outcome | Base model | Fine-tuned |
|---|---|---|
| price correct (read the Montant column) | 13 | **161** |
| price = **unit price** (read the P.U. column) | **124** | 0 |
| predicted line with no matching gold line | 15 | 0 |
| gold line the model never predicted | 15 | 0 |

Of the 137 lines the base model matched by name, 124 — nine in ten — took the unit price.
Fine-tuning removed the mistake entirely: 161 of 161 lines correct. A synthetic set without the
distractor would have reported base line-item F1 somewhere near the scalar fields and hidden
the single most useful thing fine-tuning taught the model.

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

### Does the flag actually work? Measured, not asserted

The claim is "if validation does not raise an error, the totals can be trusted". On the test
sets there *are* gold labels, so the claim can be checked
([`scripts/validation_value.py`](../scripts/validation_value.py)). A document counts if its
gold has a subtotal, tax and total; "totals correct" means all three match gold within a cent.

| Run | Wrong totals | …flagged | Not flagged | …totals correct |
|---|---|---|---|---|
| Base, CORD | 5 | **5** | 2 | **2** |
| Fine-tuned, CORD | 1 | **1** | 10 | **10** |
| Base, synthetic | 17 | **17** | 0 | — |
| Fine-tuned, synthetic | 0 | — | 50 | **50** |

**Across all four runs, 23 documents had wrong totals and the flag caught all 23. Not one
document with wrong totals passed unflagged.** That is the property that matters for posting
invoices automatically: an unflagged document never carried a wrong total in any of these runs.

The flag is not free, though. On fine-tuned CORD it raised an error on 10 receipts, and 9 of
those had subtotal, tax and total all *correct*. Those are not model mistakes. Running the same
rules over the **gold answers** shows that 15 of the 50 CORD receipts fail validation on their
own. In 8 of the 9 cases `tax_arithmetic` fails because the receipt does not close under this
schema: on some, the printed total equals the subtotal, so tax was already included in the
prices; on others the total carries a charge the schema does not capture, such as the
`service_price` field CORD records on a minority of receipts and that was deliberately left out.

So on real receipts the flag is conservative — it sends some correct documents to a human — and
that is the right way round for accounting. The fix belongs in the schema (a tax-inclusive flag
and an "other charges" field), not in loosening the rules. This is written down rather than
quietly tolerated, because it is exactly the kind of thing that looks like a model failure in a
results table and is not.

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

**The dataset classes loaded everything into RAM.** `CordDataset.__init__` did
`list(load_cord_split(split))`, which decodes all 800 CORD receipt photographs at once —
around 6 GB, since they run up to 3020 px wide — and `SyntheticDataset` pre-rendered all 200
invoices. On a Kaggle node with 30 GB of RAM this is invisible. On an 8 GB laptop the kernel
OOM-killer terminates the process immediately after the weights finish loading, with **no
traceback at all**, because SIGKILL does not raise. Fixed by indexing the HuggingFace dataset
lazily and rendering synthetic invoices on demand; rendering is deterministic in the seed, so
the examples are identical either way. Worth fixing regardless of machine — it also removed a
multi-minute stall before the first training step on Kaggle.

**The synthetic test set changed from one day to the next.** The generator drew invoice dates
with `date_between(start_date="-2y", end_date="today")` — relative to the day it ran. The
base model was scored on 11 September and the fine-tuned model on 12 September, so every one
of the 50 held-out invoices came out with its date shifted by exactly one day. It was caught
by noticing that seed 900000's gold date read `2026-05-11` in one results file and
`2026-05-12` in the other, then comparing all 50 records field by field: only `date`
differed, and nothing else. Each model had still been scored against the date printed on its
own image, so no score was wrong, but "base and fine-tuned were measured on the same test
set" was no longer literally true. **Fix:** dates come from a fixed window
(`DATE_WINDOW_START`/`DATE_WINDOW_END`), a test pins seed 900000 to one exact date, and both
models were re-scored on the now byte-identical set. The lesson: a seed only makes data
reproducible if nothing else the generator reads — here, the clock — changes between runs.

**The line-item diagnosis script invented errors.** `scripts/inspect_items.py` classifies each
predicted line as correct, unit-price mistake, or other. It looked gold lines up in a
dictionary keyed by product name, and invoices often sell the same product on two lines, so
the second line was compared against the wrong gold line. On the fine-tuned predictions it
reported 22 wrong prices while line-item F1 was exactly 1.000 — two numbers that cannot both be
true, which is how it was caught. **Fix:** pair each predicted line with one unused gold line,
preferring the best match. On the base model the corrected count *strengthened* the finding
rather than weakening it: the "wrong for some other reason" bucket had been entirely
duplicate-name artefacts, and those lines were in fact unit-price mistakes too.

Both of these are measurement bugs, not model bugs. They are the reason every analysis script
in this project is checked against a second number that has to agree with it.

---

## 10a. How training was made to fit on the local GPU

This is the best debugging story in the project, partly because the first conclusion was
wrong. An earlier version of this section said local training was impossible. It is not:
the final run trained **one full epoch in 25 minutes on the 6 GB laptop card**.

### The symptoms, which pointed the wrong way

| Attempt | What happened |
|---|---|
| Model loaded, 4-bit, LoRA attached | 1.98 GB — fine |
| Forward+backward at `max_pixels=401408` | PyTorch reported **12.21 GB allocated** on a 6 GB card |
| Same at 128 visual tokens | `CUDA driver error: device not ready` |
| Same with `CUDA_LAUNCH_BLOCKING=1` | the WSL2 VM itself crashed |

`device not ready` is not an out-of-memory message, so for a while it looked like a driver or
kernel bug. Standalone SDPA and matmul forward+backward on the same GPU worked perfectly,
and so did over a hundred inference generations. That was the clue that the problem was the
*model setup*, not the hardware.

### The four real causes

**1. Allocations were spilling past the card into host RAM.** Windows keeps about 1.1 GB of
the 6 GB for the desktop, so only ~4.9 GB is free. WSL2's GPU passthrough (WDDM) does not
refuse an allocation past that — it oversubscribes into system RAM. That is how PyTorch could
"allocate 12.21 GB" on a 6 GB card, and why the failure showed up as a driver fault or the
host OOM-killer instead of a readable error. **Fix:** `torch.cuda.set_per_process_memory_fraction`
set *below* the physically free memory (`cuda_memory_fraction: 0.82`). After that, every
over-allocation failed as a clean, fast, honest OOM, and the real debugging could start.

**2. The frozen vision tower was inside the autograd graph.** No vision weight is ever
trained, but `prepare_model_for_kbit_training` calls `enable_input_require_grads`, which
makes the embedding output require grad. The image features are written into that same
tensor, so the whole vision encoder stayed in the graph — its activations kept for backward,
and with gradient checkpointing, its entire forward re-run during backward. **Fix:** run the
vision encoder under `torch.no_grad()` (`freeze_vision_tower`). The gradients reaching the
LoRA adapters are identical, because they never flowed through the vision tower anyway.

**3. A frozen 0.87 GB tensor was stored in fp32.** The same helper upcasts every
non-quantised parameter to fp32. For layer norms that helps stability. For Qwen's token
embedding (151,936 × 1,536 = 233M parameters) it is 0.869 GB of waste on a tensor that never
changes — found with `scripts/inspect_model_memory.py`, which breaks memory down by dtype.
**Fix:** cast frozen fp32 parameters back to bf16, keeping the trainable LoRA weights in fp32.
Saved 0.44 GB.

**4. My own measurement script had gradient checkpointing switched off.** HuggingFace only
checkpoints a layer when the module is in training mode, and the probe never called
`model.train()`. Every number measured before that fix was worse than what `Trainer` actually
runs. This is worth admitting in an interview: a measurement bug cost more time than any
model bug.

Two things tried that did **not** help, recorded so nobody repeats them: lowering resolution
(the sequence is mostly text — 945 tokens at the smallest setting, only 192 of them visual)
and Liger's fused cross-entropy (the patch reported success but never engaged on this
transformers version — the probe checks whether logits were still materialised).

### Result

| Measurement | Value |
|---|---|
| Peak VRAM, full forward+backward, seq_len 1255 | **4.16 GB** (of ~4.9 GB free) |
| Training time, 1 epoch, 1000 examples, 125 optimiser steps | **25 min** (1501 s) |
| Throughput | 0.67 examples/s, ~10 s per step once warm |
| Training loss, step 10 → step 120 | 0.113 → 0.030 (mean over the epoch 0.044) |

For comparison, the earlier CORD-only run on a Kaggle T4 took about 2 h 04 m for 800
examples. The laptop is faster, because with the fixes above it no longer wastes memory
bandwidth on a graph it never uses.

Also needed: WSL's default memory cap is half the host RAM (7.8 GB here), and loading the
model peaked at 5.3 GB of host RAM. `~/.wslconfig` raises it to 11 GB with 8 GB of swap —
a copy is in `docs/wslconfig.example`.

The settings live in `configs/train_local.yaml`. `configs/train.yaml` and the Kaggle notebook
still work unchanged for anyone without a local GPU.

**The general lesson:** when an error message does not match the obvious cause, first make
failures honest (here, the memory cap), then measure — and check that the measuring tool
itself runs the same code path as the real thing.

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

- **The synthetic 1.000 measures one layout.** Every synthetic invoice, train and test, comes
  from one template; degradation varies the pixels, not the arrangement. The fine-tuned model
  getting all 50 held-out invoices exactly right shows it learned *this* layout, not that it
  reads any Moroccan invoice. This is the most important gap, and the reason that number is
  labelled an upper bound everywhere it appears.
- **No real Moroccan invoices have been tested.** By design — no real client data — but it
  means the ICE, IF, date and currency numbers come from the same generator as the training
  data. CORD-v2 is the only real-world evidence, and it contains none of those fields.
- **Small test sets, one run.** 50 documents per set, one epoch, one seed, no learning-rate
  sweep. How noisy that is was measured by accident: re-scoring the base model after a change
  that only altered the date printed on each synthetic invoice moved individual fields by up to
  0.049 (micro-F1 by 0.002). Differences smaller than that should not be read as real.
- **The validation flag is conservative on real receipts.** 15 of the 50 CORD *gold* answers
  fail the rules on their own — tax-inclusive totals, and charges such as `service_price` that
  the schema leaves out. On the fine-tuned model, 9 of the 10 flagged receipts had correct
  totals (§9). Safe, but it sends correct documents to a human.
- **CORD line items are the weakest real-world field** at 0.743 F1.
- **Latency is demo-grade.** On the 6 GB laptop GPU in 4-bit: about 12 s per CORD receipt in
  the evaluation loop, 26 s per synthetic invoice including rendering and model loading, and
  39.5 s of generation for one live API request. Fine for a demo, too slow for bulk processing
  without batching, a merged adapter and better hardware.

---

## 13. What comes next, in priority order

1. **More synthetic layouts.** Three or four distinct invoice templates attack the biggest
   limitation directly and cost no GPU time. This is also what would turn the synthetic
   1.000 into a number worth trusting.
2. **A richer schema.** A tax-inclusive flag and an "other charges" field would remove most of
   the validation false alarms on real receipts, because the rules would then describe how
   those receipts actually add up.
3. **Several seeds and a validation loss.** To put error bars on every number here, and to pick
   the epoch count by evidence rather than by "one epoch fitted in 25 minutes".
4. **Qwen3-VL-2B-Instruct.** Now that it exists, re-run the same pipeline on it. The code needs
   a config change and nothing else, which is the point of keeping hyperparameters in YAML.
5. **Faster inference.** Merge the adapter into the weights and batch requests.
6. **Confidence per field**, so the validation layer can say *which* number it doubts, not just
   that the document does not reconcile.
