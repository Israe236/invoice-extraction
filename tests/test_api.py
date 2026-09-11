"""Tests for the FastAPI service.

The model is never loaded here. `/health` and `/validate` genuinely need no
GPU, and `/extract` is tested with the generation step stubbed out, so the
request handling -- upload limits, bad files, unparseable model output -- is
covered on any machine and in CI.
"""

import io

import pytest

pytest.importorskip("fastapi", reason="API extra not installed")

from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402

from invoice_extraction import api  # noqa: E402


@pytest.fixture
def client():
    return TestClient(api.app)


def png_bytes(size=(64, 64), colour="white") -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, colour).save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def stub_model(monkeypatch):
    """Replace model loading and generation with a canned response."""

    def _install(raw_output: str):
        monkeypatch.setattr(api._engine, "model", object())
        monkeypatch.setattr(api._engine, "processor", object())
        monkeypatch.setattr(api._engine, "config", type("C", (), {"max_new_tokens": 512})())
        monkeypatch.setattr(
            "invoice_extraction.baseline.generate",
            lambda *args, **kwargs: raw_output,
        )

    yield _install
    api._engine.model = None
    api._engine.processor = None


CONSISTENT_JSON = """{
  "items": [{"name": "Prestation de conseil", "qty": "2", "price": "1000.00"}],
  "subtotal": "1000.00", "tax": "200.00", "total": "1200.00",
  "ice": "001234567000078", "if_number": "12345678",
  "invoice_number": "FA-1234", "date": "2024-04-03", "currency": "MAD"
}"""


class TestHealth:
    def test_reports_ok_without_loading_the_model(self, client):
        body = client.get("/health").json()
        assert body["status"] == "ok"
        assert body["model_loaded"] is False


class TestValidateEndpoint:
    def test_consistent_invoice_is_ok(self, client):
        payload = {
            "items": [{"name": "a", "qty": "1", "price": "1000.00"}],
            "subtotal": "1000.00",
            "tax": "200.00",
            "total": "1200.00",
        }
        body = client.post("/validate", json=payload).json()
        assert body["severity"] == "ok"
        assert body["valid"] is True

    def test_broken_arithmetic_is_an_error(self, client):
        payload = {"subtotal": "1000.00", "tax": "200.00", "total": "9999.00"}
        body = client.post("/validate", json=payload).json()
        assert body["severity"] == "error"
        assert body["needs_review"] is True

    def test_needs_no_gpu(self, client):
        # Regression guard: /validate must never trigger a model load.
        client.post("/validate", json={"total": "1.00"})
        assert api._engine.ready is False


class TestExtractEndpoint:
    def test_successful_extraction(self, client, stub_model):
        stub_model(CONSISTENT_JSON)
        response = client.post("/extract", files={"file": ("inv.png", png_bytes(), "image/png")})
        assert response.status_code == 200
        body = response.json()
        assert body["ok"] is True
        assert body["extraction"]["total"] == "1200.00"
        assert body["validation"]["severity"] == "ok"

    def test_validation_is_attached_to_the_extraction(self, client, stub_model):
        stub_model(CONSISTENT_JSON.replace('"total": "1200.00"', '"total": "8888.00"'))
        body = client.post(
            "/extract", files={"file": ("inv.png", png_bytes(), "image/png")}
        ).json()
        assert body["ok"] is True
        assert body["validation"]["severity"] == "error"

    def test_code_fenced_output_is_recovered(self, client, stub_model):
        stub_model(f"Here is the data:\n```json\n{CONSISTENT_JSON}\n```")
        body = client.post(
            "/extract", files={"file": ("inv.png", png_bytes(), "image/png")}
        ).json()
        assert body["ok"] is True

    def test_unparseable_output_is_reported_not_hidden(self, client, stub_model):
        """A model that returns prose must not look like an empty invoice."""
        stub_model("I am unable to read this document.")
        response = client.post("/extract", files={"file": ("inv.png", png_bytes(), "image/png")})
        assert response.status_code == 200
        body = response.json()
        assert body["ok"] is False
        assert body["extraction"] is None
        assert "I am unable" in body["raw_output"]

    def test_non_image_upload_is_rejected(self, client):
        response = client.post(
            "/extract", files={"file": ("notes.txt", b"this is not an image", "text/plain")}
        )
        assert response.status_code == 400

    def test_empty_upload_is_rejected(self, client):
        response = client.post("/extract", files={"file": ("empty.png", b"", "image/png")})
        assert response.status_code == 400

    def test_oversized_upload_is_rejected(self, client, monkeypatch):
        monkeypatch.setattr(api, "MAX_UPLOAD_BYTES", 128)
        response = client.post(
            "/extract", files={"file": ("big.png", png_bytes((256, 256)), "image/png")}
        )
        assert response.status_code == 413

    def test_missing_file_field_is_a_422(self, client):
        assert client.post("/extract").status_code == 422
