"""FastAPI service: upload an invoice image, get validated JSON back.

The response deliberately carries three separate things:

- `extraction`  what the model read off the document
- `validation`  whether that reading is internally consistent
- `raw_output`  the model's literal text, so a failure can be diagnosed

An accountant does not want a number they cannot trust. The validation block is
what makes the output actionable: `severity: "ok"` means the arithmetic closes
and the document can be posted automatically, anything else means a human
should look at it. That is the whole product argument for this project.

The model is loaded once, lazily, on the first request rather than at import,
so that `/health` answers immediately and the container starts fast.
"""

from __future__ import annotations

import io
import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

from invoice_extraction.data import EXTRACTION_PROMPT
from invoice_extraction.evaluate import parse_model_output
from invoice_extraction.validate import validate

# Configuration comes from the environment so the same image can serve the base
# model or a fine-tuned adapter without a rebuild.
CONFIG_PATH = os.environ.get("INVOICE_CONFIG", "configs/baseline.yaml")
ADAPTER_PATH = os.environ.get("INVOICE_ADAPTER") or None
MAX_UPLOAD_BYTES = int(os.environ.get("INVOICE_MAX_UPLOAD_BYTES", 10 * 1024 * 1024))

_ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP", "BMP", "TIFF"}


@dataclass
class _Engine:
    """Lazily-initialised model holder."""

    model: Any = None
    processor: Any = None
    config: Any = None

    @property
    def ready(self) -> bool:
        return self.model is not None

    def load(self) -> None:
        if self.ready:
            return
        # Imported here, not at module scope: torch and transformers take ~10s
        # to import, and the health endpoint should not wait for them.
        from invoice_extraction.model import (
            load_config,
            load_finetuned_model,
            load_model_and_processor,
        )

        self.config = load_config(CONFIG_PATH)
        if ADAPTER_PATH:
            self.model, self.processor = load_finetuned_model(self.config, ADAPTER_PATH)
        else:
            self.model, self.processor = load_model_and_processor(self.config)


_engine = _Engine()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Nothing to warm up eagerly; the first /extract pays the load cost.
    yield
    _engine.model = None
    _engine.processor = None


app = FastAPI(
    title="Invoice Extraction",
    description="Turn an invoice or receipt image into validated structured JSON.",
    version="0.1.0",
    lifespan=lifespan,
)

# The React dev server runs on a different port; without this the browser
# blocks every request.
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("INVOICE_CORS_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    """Liveness probe. Answers without touching the GPU."""
    return {
        "status": "ok",
        "model_loaded": _engine.ready,
        "config": CONFIG_PATH,
        "adapter": ADAPTER_PATH,
    }


def _read_image(payload: bytes) -> Image.Image:
    try:
        image = Image.open(io.BytesIO(payload))
        image.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail="File is not a readable image") from exc

    if image.format and image.format.upper() not in _ALLOWED_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image format {image.format}. "
            f"Expected one of {sorted(_ALLOWED_FORMATS)}.",
        )
    return image.convert("RGB")


@app.post("/extract")
async def extract(file: UploadFile = File(...)) -> dict:
    """Extract structured fields from one uploaded invoice or receipt image."""
    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="Empty upload")
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Image is larger than the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit",
        )

    image = _read_image(payload)

    try:
        _engine.load()
    except Exception as exc:  # noqa: BLE001 - surfaced to the client as a 503
        raise HTTPException(status_code=503, detail=f"Model unavailable: {exc}") from exc

    from invoice_extraction.baseline import generate

    started = time.perf_counter()
    raw_output = generate(
        _engine.model, _engine.processor, image, EXTRACTION_PROMPT, _engine.config.max_new_tokens
    )
    elapsed = time.perf_counter() - started

    extraction = parse_model_output(raw_output)
    if extraction is None:
        # Not a server error: the model genuinely failed to produce JSON, and
        # the caller needs to know that rather than receive an empty object
        # that looks like a document with no fields on it.
        return {
            "ok": False,
            "error": "The model did not return parseable JSON for this image.",
            "extraction": None,
            "validation": None,
            "raw_output": raw_output,
            "elapsed_seconds": round(elapsed, 2),
        }

    report = validate(extraction)
    return {
        "ok": True,
        "extraction": extraction,
        "validation": report.to_dict(),
        "raw_output": raw_output,
        "elapsed_seconds": round(elapsed, 2),
        "filename": file.filename,
    }


@app.post("/validate")
def validate_only(fields: dict) -> dict:
    """Run the business rules against JSON that was extracted elsewhere.

    Useful on its own: the validation layer needs no GPU and no model, so it
    can run as a cheap standalone service over an existing OCR pipeline.
    """
    return validate(fields).to_dict()
