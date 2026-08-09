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

from invoice_extraction.data import to_chat_example
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


def render_invoice(invoice: InvoiceData, dpi: int = 200) -> Image.Image:
    template = _env.get_template("invoice.html.jinja")
    html_str = template.render(**asdict(invoice))
    pdf_bytes = HTML(string=html_str).write_pdf()
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    pixmap = doc[0].get_pixmap(dpi=dpi)
    image = Image.open(io.BytesIO(pixmap.tobytes("png"))).convert("RGB")
    doc.close()
    return image


def degrade(image: Image.Image, seed: int | None = None) -> Image.Image:
    transform = A.Compose(_DEGRADE_TRANSFORMS, seed=seed)
    result = transform(image=np.array(image))["image"]
    return Image.fromarray(result)


def iter_synthetic_examples(n: int, seed: int | None = None) -> Iterator[list[dict]]:
    for i in range(n):
        example_seed = None if seed is None else seed + i
        invoice = generate_invoice_data(seed=example_seed)
        image = degrade(render_invoice(invoice), seed=example_seed)
        yield to_chat_example(image, to_ground_truth(invoice))
