"""HTML invoice rendering and photographic degradation."""

import io
from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path

import albumentations as A
import numpy as np
import pymupdf
from jinja2 import Environment, FileSystemLoader
from PIL import Image
from weasyprint import HTML

from invoice_extraction.data import ExtractedFields, to_chat_example
from invoice_extraction.synthetic import InvoiceData, generate_invoice_data, to_ground_truth

_TEMPLATE_DIR = Path(__file__).parent / "templates"
_env = Environment(loader=FileSystemLoader(_TEMPLATE_DIR))

_DEGRADE_TRANSFORMS = [
    A.Rotate(limit=3, border_mode=0, fill=(255, 255, 255), p=0.7),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.GaussNoise(std_range=(0.02, 0.08), p=0.4),
    A.GaussianBlur(blur_limit=(1, 3), p=0.3),
    A.ImageCompression(quality_range=(40, 85), p=0.6),
]


def crop_to_content(image: Image.Image, margin: int = 40, threshold: int = 245) -> Image.Image:
    """Trim the blank part of the page below the invoice.

    An invoice with two line items fills the top third of an A4 sheet and
    leaves the rest white. That matters more than it looks: the processor is
    capped at `max_pixels`, so a full page is downscaled ~9x and the text ends
    up near-illegible, with most of the visual token budget spent on blank
    paper. Cropping to the printed area roughly triples the effective
    resolution of the text at the same token cost.

    A margin is kept so the crop never shaves a character, and a page that is
    entirely blank is returned unchanged rather than collapsing to nothing.
    """
    grayscale = image.convert("L")
    # Invert so that ink is bright, then let Pillow find its bounding box.
    ink = grayscale.point(lambda value: 255 if value < threshold else 0)
    bbox = ink.getbbox()
    if bbox is None:
        return image

    left, top, right, bottom = bbox
    return image.crop(
        (
            max(0, left - margin),
            max(0, top - margin),
            min(image.width, right + margin),
            min(image.height, bottom + margin),
        )
    )


def render_invoice(invoice: InvoiceData, dpi: int = 200, crop: bool = True) -> Image.Image:
    template = _env.get_template("invoice.html.jinja")
    html_str = template.render(**asdict(invoice))
    pdf_bytes = HTML(string=html_str).write_pdf()
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    pixmap = doc[0].get_pixmap(dpi=dpi)
    image = Image.open(io.BytesIO(pixmap.tobytes("png"))).convert("RGB")
    doc.close()
    # Crop before degrading: rotation fills the corners with white, which would
    # otherwise defeat the content-bounding-box detection.
    return crop_to_content(image) if crop else image


def degrade(image: Image.Image, seed: int | None = None) -> Image.Image:
    transform = A.Compose(_DEGRADE_TRANSFORMS, seed=seed)
    result = transform(image=np.array(image))["image"]
    return Image.fromarray(result)


# Training and evaluation draw from disjoint seed ranges so that no invoice the
# model was trained on can reappear in the test set. Faker is deterministic per
# seed, so identical seeds would produce byte-identical invoices -- the exact
# leak that makes a synthetic benchmark meaningless.
TRAIN_SEED_BASE = 42
EVAL_SEED_BASE = 900_000


def iter_synthetic_records(
    n: int, seed: int | None = None
) -> Iterator[tuple[Image.Image, ExtractedFields]]:
    """Yield `(degraded image, ground truth)` pairs for n generated invoices."""
    for i in range(n):
        example_seed = None if seed is None else seed + i
        invoice = generate_invoice_data(seed=example_seed)
        image = degrade(render_invoice(invoice), seed=example_seed)
        yield image, to_ground_truth(invoice)


def iter_synthetic_examples(n: int, seed: int | None = None) -> Iterator[list[dict]]:
    """Training-format chat examples."""
    for image, fields in iter_synthetic_records(n, seed=seed):
        yield to_chat_example(image, fields)


def build_synthetic_eval_set(n: int) -> list[tuple[Image.Image, ExtractedFields]]:
    """The held-out synthetic test set, fixed and reproducible.

    Uses `EVAL_SEED_BASE`, far from the `TRAIN_SEED_BASE` range, so a training
    run of any realistic size cannot collide with it.
    """
    return list(iter_synthetic_records(n, seed=EVAL_SEED_BASE))
