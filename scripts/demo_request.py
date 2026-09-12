"""Send one image to a running API and print the response.

Used as the end-to-end smoke test, and to produce the worked example in the
README from a real request rather than a hand-written one.

    python scripts/demo_request.py docs/assets/example_invoice.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx

DEFAULT_URL = "http://127.0.0.1:8000/extract"


def main() -> None:
    image_path = Path(sys.argv[1] if len(sys.argv) > 1 else "docs/assets/example_invoice.png")
    url = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_URL

    if not image_path.exists():
        raise SystemExit(f"no such image: {image_path}")

    with image_path.open("rb") as handle:
        files = {"file": (image_path.name, handle, "image/png")}
        # The first request loads the model onto the GPU, so the timeout has to
        # be generous or the client gives up before the server has started.
        response = httpx.post(url, files=files, timeout=600.0)

    response.raise_for_status()
    print(json.dumps(response.json(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
