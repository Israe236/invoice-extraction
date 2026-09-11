# invoice-extraction

Turn an invoice or receipt **image** into **structured JSON** an accountant can use directly,
and flag the documents that do not add up.

A small vision-language model (Qwen2-VL-2B) fine-tuned with 4-bit QLoRA on free compute,
plus a deterministic validation layer that checks the extracted numbers against accounting
rules — `Total HT + TVA = Total TTC`, line items summing to the subtotal, dates that are real.

---

## The problem

Entering supplier invoices by hand is slow, expensive and error-prone. The usual automation is
OCR plus regular expressions, which breaks on the first supplier whose layout is different —
and every supplier's layout is different.

A vision-language model reads the document the way a person does: it sees that a number sits
in the column headed *Montant*, on the row labelled *Prestation de conseil*. Layout
understanding comes for free instead of being hand-coded per supplier.

The catch is that a language model will occasionally invent a number. **That is what the
validation layer is for.** The output is not "here is a number, trust me" — it is "here is a
number, and here is whether the document's own arithmetic agrees with it".

---

## Architecture

```mermaid
flowchart TB
    subgraph training ["Training — free tier, Kaggle P100"]
        CORD["CORD-v2<br/>800 real receipt photos<br/>public dataset"]
        SYNTH["Synthetic generator<br/>Faker → Jinja → WeasyPrint<br/>→ raster → degrade<br/>200 FR/MA invoices"]
        CORD --> MIX["Unified 9-field schema"]
        SYNTH --> MIX
        MIX --> QLORA["4-bit QLoRA<br/>NF4 + double quant<br/>r=16, vision tower frozen"]
        QLORA --> ADAPTER["LoRA adapter<br/>~180 MB"]
    end

    subgraph serving ["Serving — local, RTX 4050 6 GB"]
        UPLOAD["Invoice image"] --> API["FastAPI /extract"]
        API --> VLM["Qwen2-VL-2B<br/>4-bit, max_pixels capped"]
        ADAPTER -.loaded at runtime.-> VLM
        VLM --> RAW["Raw model text"]
        RAW --> PARSE["JSON recovery<br/>fence / prose / trailing comma"]
        PARSE --> FIELDS["Extracted fields"]
        FIELDS --> RULES["Validation layer<br/>pure Python, no GPU"]
        RULES --> OUT["JSON + per-rule verdict<br/>ok / warning / error"]
    end

    OUT --> UI["React frontend"]

    style QLORA fill:#e8f0fe,stroke:#2f5fd0
    style RULES fill:#e8f5ee,stroke:#157347
    style OUT fill:#e8f5ee,stroke:#157347
```

The two halves are deliberately separable. The validation layer needs no GPU and no model —
`POST /validate` runs it standalone, so it can sit in front of an existing OCR pipeline.

---

## Results

> **Status:** the zero-shot baseline is measured. The fine-tuned column is pending a Kaggle
> training run — see [Reproducing](#reproducing). Only measured numbers appear here; no cell
> is filled in by estimation.

### Zero-shot baseline — CORD-v2 test split, 50 examples

`Qwen2-VL-2B-Instruct`, 4-bit, no fine-tuning, `max_pixels=401408`.

| Field | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| subtotal | 0.650 | 0.419 | **0.510** | 31 |
| tax | 0.150 | 0.150 | **0.150** | 20 |
| total | 0.800 | 0.340 | **0.478** | 47 |
| items | 0.522 | 0.278 | **0.363** | 126 |
| ice | — | — | n/a | 0 |
| if_number | — | — | n/a | 0 |
| invoice_number | — | — | n/a | 0 |
| date | — | — | n/a | 0 |
| currency | — | — | n/a | 0 |
| **micro-average** | | | **0.346** | |

**JSON parse rate: 22/50 (44 %).** The base model failed to return parseable JSON on more
than half the documents — after a repair pass that strips code fences, prose prefixes and
trailing commas. This single number is the clearest argument for fine-tuning: a model that is
sometimes accurate and often unparseable cannot be put in a workflow.

`n/a` means **support 0** — CORD-v2 receipts carry no ICE, IF, invoice number, date or
currency, so those fields were never tested here, rather than always wrong. They are measured
on the synthetic test set instead, which is the whole reason the synthetic data exists.

Raw per-example outputs: [`results/base_cord_predictions.json`](results/base_cord_predictions.json).

---

## Example

Input — a generated Moroccan invoice (seed 900000, held-out test range), after photographic
degradation:

<img src="docs/assets/example_invoice.png" alt="Synthetic Moroccan invoice" width="520">

Target output:

```json
{
  "items": [
    { "name": "Services de nettoyage", "qty": "6", "price": "3871.80" }
  ],
  "subtotal": "3871.80",
  "tax": "542.05",
  "total": "4413.85",
  "ice": "040978053964059",
  "if_number": "80947950",
  "invoice_number": "FA-7314",
  "date": "2026-05-11",
  "currency": "MAD"
}
```

Validation result attached by the API:

```json
{
  "valid": true,
  "severity": "ok",
  "needs_review": false,
  "unverified": [],
  "checks": [
    { "check": "tax_arithmetic", "status": "pass",
      "detail": "subtotal 3871.80 + tax 542.05 = 4413.85, stated total 4413.85 (difference 0.00)" },
    { "check": "items_sum", "status": "pass",
      "detail": "1 line item(s) sum to 3871.80, subtotal 3871.80 (difference 0.00)" },
    { "check": "tax_rate_plausible", "status": "pass", "detail": "implied VAT rate 14.00%" },
    { "check": "ice_format", "status": "pass", "detail": "15 digits (expected 15)" }
  ]
}
```

Note the unit-price column (`645.30`) on the rendered invoice. It is a **deliberate
distractor**: real invoices always print one next to the line total, and grabbing the wrong
column is the most common real-world extraction failure. Ground truth is the *Montant* column.

---

## The validation layer

Business rules run on the model's output, with no reference answer — which is the point,
because in production there is no gold label. The only automatic signal that an extraction is
wrong is that it contradicts itself.

| Rule | Check | On failure |
|---|---|---|
| `required_fields` | a total is present at all | error |
| `tax_arithmetic` | Total HT + TVA = Total TTC | error |
| `items_sum` | line items sum to Total HT | error |
| `tax_rate_plausible` | implied VAT rate is 20 / 14 / 10 / 7 % | warning |
| `date_valid` | parses, not in the future, not >10 years old | warning |
| `ice_format` | exactly 15 digits | warning |
| `if_format` | 7–9 digits | warning |
| `currency_known` | a recognised currency code | warning |

Each rule returns **pass**, **fail** or **skipped** — not just pass/fail. A rule that cannot
run because the document does not carry the fields it needs is *unverified*, not *violated*.
Without that distinction every CORD receipt got reported as an arithmetic error purely for
having no Moroccan ICE number, and a report full of meaningless red flags is worse than no
report. See [DECISIONS §9](docs/DECISIONS.md#9-why-the-validation-layer-has-three-outcomes-not-two).

---

## Setup

Requires Linux or WSL2, and an NVIDIA GPU for the model (the validation layer and tests run
anywhere).

```bash
git clone https://github.com/Israe236/invoice-extraction.git
cd invoice-extraction

# Python environment (uv resolves the CUDA-matched torch build from pyproject.toml)
uv venv --python 3.11
uv pip install -e ".[dev,api]"

# Tests need no GPU and no downloads
pytest
```

### Run the demo

```bash
# Terminal 1 — API (add INVOICE_ADAPTER=checkpoints/... to serve the fine-tuned model)
uvicorn invoice_extraction.api:app --port 8000

# Terminal 2 — frontend
cd frontend && npm install && npm run dev
# open http://localhost:5173
```

The first `/extract` request loads the model onto the GPU and takes about a minute; after that
it is roughly 50 s per document on a 6 GB laptop card.

### Evaluate

```bash
python -m invoice_extraction.baseline --dataset cord      --out results/base_cord.json
python -m invoice_extraction.baseline --dataset synthetic --out results/base_synthetic.json

# with a trained adapter
python -m invoice_extraction.baseline --dataset cord --adapter checkpoints/qwen2vl-2b-lora-v2 \
    --out results/finetuned_cord.json
```

---

## Reproducing the training

Training does not fit in 6 GB, so it runs on Kaggle's free tier.

1. Upload [`notebooks/train_kaggle.ipynb`](notebooks/train_kaggle.ipynb) to Kaggle.
2. Set **Accelerator → GPU P100** and **Internet → On**.
3. Run the smoke-test cell first (3 steps, ~5 minutes). It catches the failures that would
   otherwise waste a two-hour session.
4. Run the full cell (~2.5 h for one epoch over 1000 examples).
5. Download `qwen2vl-2b-lora-v2.zip` from the Output panel and unzip it into `checkpoints/`.

The notebook clones this repository rather than carrying its own copy of the training code, so
what runs on Kaggle is exactly what is in version control.

---

## Data

| Source | What | Licence |
|---|---|---|
| [CORD-v2](https://huggingface.co/datasets/naver-clova-ix/cord-v2) | 1000 real receipt photographs (800 train / 100 val / 100 test) | public research dataset |
| Synthetic FR/MA invoices | generated in-repo: Faker → Jinja → WeasyPrint → PyMuPDF → Albumentations degradation | generated, no licence issue |

**No real client data is used anywhere in this project.** Every company name, address, ICE and
IF number in the synthetic set is generated. CORD-v2 receipts have no ICE, IF, invoice number,
date or currency, which is precisely why the synthetic invoices exist — without them the model
cannot learn the fields the project is actually about.

Train and test synthetic invoices are drawn from **disjoint seed ranges** (42+ vs 900 000+).
Faker is deterministic, so a shared seed would put a byte-identical document in both sets;
there is a test asserting the ranges cannot collide.

---

## Project layout

```
src/invoice_extraction/
  data.py            CORD-v2 → the unified 9-field schema, chat formatting
  synthetic.py       generated French/Moroccan invoice data
  render.py          HTML → PDF → raster → photographic degradation, content cropping
  normalize.py       money / date / text normalisation, shared by metric and rules
  evaluate.py        JSON recovery, per-field P/R/F1, parse rate, support
  validate.py        the business rules
  model.py           4-bit loading, VRAM-safe defaults
  train.py           QLoRA fine-tuning
  baseline.py        evaluation entry point (base and fine-tuned, same code path)
  api.py             FastAPI service
frontend/            React + Vite demo
tests/               238 tests, no GPU required
configs/             all hyperparameters
docs/DECISIONS.md    why every choice was made
notebooks/           Kaggle training notebook
results/             measured evaluation output (committed — it is the evidence)
```

---

## Limitations

Stated plainly, because a portfolio project claiming no weaknesses is not credible.

- **Small test sets.** 50 CORD receipts, 50 synthetic invoices. Enough to see a large effect,
  not enough for tight confidence intervals per field.
- **One synthetic template.** Degradation varies the pixels; the layout does not vary. The
  model has seen one way of arranging a Moroccan invoice. This is the biggest gap.
- **No real Moroccan invoices tested.** By design (no real client data), but it means the
  ICE/IF/date numbers are measured against the same generator that produced the training data
  and should be read as an upper bound.
- **One epoch, one seed.** No learning-rate sweep, no early stopping, no variance estimate.
- **~50 s per document** on a 6 GB laptop GPU in 4-bit. Fine for a demo, too slow for bulk
  processing.
- **CORD's `service_price` is excluded** from the schema, so some real receipts legitimately
  fail `items_sum`. Documented rather than quietly tolerated.

## Next steps

1. **More synthetic layouts** — three or four distinct templates. Attacks the biggest
   limitation and costs no GPU time.
2. **More epochs with a validation loss** and early stopping.
3. **Qwen3-VL-2B-Instruct** — it did not exist when this project started (Qwen3-VL began at
   4B) and has since been released. Re-running the same pipeline needs only a config change.
4. **Constrained decoding** — a JSON grammar would take the parse rate to 100 % by
   construction rather than by training.
5. **Per-field confidence**, so the validation layer can say *which* number it doubts.

---

## Licence

MIT. CORD-v2 is subject to its own licence.
