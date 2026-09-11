"""Tests for invoice rendering and the train/eval seed split.

Skipped when the rendering stack (WeasyPrint, PyMuPDF, Albumentations) is not
installed, so the core test suite still runs on a machine that only needs the
validation layer -- the API container, for instance.
"""

import pytest

pytest.importorskip("weasyprint", reason="rendering stack not installed")
pytest.importorskip("pymupdf", reason="rendering stack not installed")

from PIL import Image  # noqa: E402

from invoice_extraction.render import (  # noqa: E402
    EVAL_SEED_BASE,
    TRAIN_SEED_BASE,
    build_synthetic_eval_set,
    crop_to_content,
    degrade,
    render_invoice,
)
from invoice_extraction.synthetic import generate_invoice_data  # noqa: E402
from invoice_extraction.validate import Severity, validate  # noqa: E402


class TestCropToContent:
    def test_removes_blank_margin(self):
        image = Image.new("RGB", (500, 500), "white")
        image.paste(Image.new("RGB", (50, 20), "black"), (100, 100))
        cropped = crop_to_content(image, margin=10)
        assert cropped.size == (70, 40)

    def test_keeps_a_margin_around_the_ink(self):
        image = Image.new("RGB", (500, 500), "white")
        image.paste(Image.new("RGB", (50, 20), "black"), (100, 100))
        assert crop_to_content(image, margin=0).size == (50, 20)

    def test_margin_is_clamped_at_the_page_edge(self):
        image = Image.new("RGB", (100, 100), "white")
        image.paste(Image.new("RGB", (100, 100), "black"), (0, 0))
        assert crop_to_content(image, margin=50).size == (100, 100)

    def test_blank_page_is_returned_unchanged(self):
        # getbbox() returns None here; collapsing to a zero-size image would
        # crash the processor downstream.
        blank = Image.new("RGB", (200, 200), "white")
        assert crop_to_content(blank).size == (200, 200)

    def test_light_grey_is_treated_as_background(self):
        image = Image.new("RGB", (200, 200), (250, 250, 250))
        image.paste(Image.new("RGB", (20, 20), "black"), (50, 50))
        assert crop_to_content(image, margin=0).size == (20, 20)


class TestRenderInvoice:
    def test_renders_an_image(self):
        image = render_invoice(generate_invoice_data(seed=1))
        assert image.width > 0 and image.height > 0

    def test_cropping_reduces_the_page(self):
        invoice = generate_invoice_data(seed=1)
        full = render_invoice(invoice, crop=False)
        cropped = render_invoice(invoice, crop=True)
        assert cropped.height < full.height

    def test_degrade_preserves_size(self):
        image = render_invoice(generate_invoice_data(seed=1))
        assert degrade(image, seed=1).size[0] > 0

    def test_degradation_is_deterministic_for_a_seed(self):
        image = render_invoice(generate_invoice_data(seed=1))
        assert degrade(image, seed=3).tobytes() == degrade(image, seed=3).tobytes()


class TestSeedSplit:
    def test_train_and_eval_seed_ranges_cannot_overlap(self):
        """A shared seed would put a byte-identical invoice in train and test.

        Faker is deterministic per seed, so this is the leak that would make
        every synthetic number in the README meaningless.
        """
        largest_plausible_training_set = 100_000
        assert TRAIN_SEED_BASE + largest_plausible_training_set < EVAL_SEED_BASE

    def test_eval_set_is_reproducible(self):
        first = build_synthetic_eval_set(2)
        second = build_synthetic_eval_set(2)
        assert [fields for _, fields in first] == [fields for _, fields in second]

    def test_eval_set_ground_truth_is_internally_valid(self):
        for _, fields in build_synthetic_eval_set(3):
            assert validate(dict(fields)).severity is Severity.OK
