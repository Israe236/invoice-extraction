"""CORD-v2 loading and chat-format conversion."""

import json
from collections.abc import Iterator
from typing import TypedDict

from datasets import Dataset, load_dataset
from PIL import Image

CORD_DATASET = "naver-clova-ix/cord-v2"

EXTRACTION_PROMPT = (
    "Extract the following fields from this document as JSON: "
    "items (list of objects with name, qty, price), subtotal, tax, total, "
    "ice (Moroccan company ID number), if_number (Moroccan tax ID number). "
    "Return only the JSON. Use an empty string for any field that is "
    "missing from the document."
)


class LineItem(TypedDict):
    name: str
    qty: str
    price: str


class ExtractedFields(TypedDict):
    items: list[LineItem]
    subtotal: str
    tax: str
    total: str
    ice: str
    if_number: str


def load_cord_split(split: str) -> Dataset:
    return load_dataset(CORD_DATASET, split=split)


def _as_dicts(value: object) -> list[dict]:
    if isinstance(value, list):
        return [v for v in value if isinstance(v, dict)]
    if isinstance(value, dict):
        return [value]
    return []


def _as_str(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return " ".join(_as_str(v) for v in value)
    return str(value)


def _merge_dicts(dicts: list[dict]) -> dict:
    merged: dict = {}
    for d in dicts:
        merged.update(d)
    return merged


def extract_fields(gt_parse: dict) -> ExtractedFields:
    items = [
        LineItem(
            name=_as_str(item.get("nm")),
            qty=_as_str(item.get("cnt")),
            price=_as_str(item.get("price")),
        )
        for item in _as_dicts(gt_parse.get("menu"))
        if item.get("nm") is not None
    ]
    sub_total = _merge_dicts(_as_dicts(gt_parse.get("sub_total")))
    total = _merge_dicts(_as_dicts(gt_parse.get("total")))
    return ExtractedFields(
        items=items,
        subtotal=_as_str(sub_total.get("subtotal_price")),
        tax=_as_str(sub_total.get("tax_price")),
        total=_as_str(total.get("total_price")),
        ice="",
        if_number="",
    )


def to_chat_example(image: Image.Image, fields: ExtractedFields) -> list[dict]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": EXTRACTION_PROMPT},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": json.dumps(fields, ensure_ascii=False)}],
        },
    ]


def iter_chat_examples(split: str) -> Iterator[list[dict]]:
    for row in load_cord_split(split):
        gt_parse = json.loads(row["ground_truth"]).get("gt_parse", {})
        yield to_chat_example(row["image"], extract_fields(gt_parse))
