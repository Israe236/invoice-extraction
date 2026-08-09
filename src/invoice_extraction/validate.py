"""Accounting sanity checks on extracted invoice fields.

Runs on model predictions (not ground truth) to catch internally
inconsistent output and flag it for human review, independent of
whether the extraction happens to match a reference answer.
"""

from dataclasses import dataclass

_TOLERANCE = 0.02  # relative tolerance for rounding noise


def _to_float(value: str) -> float | None:
    if not value:
        return None
    cleaned = value.strip().replace(" ", "")
    if "," in cleaned and "." not in cleaned:
        int_part, _, frac_part = cleaned.rpartition(",")
        if len(frac_part) == 2:
            cleaned = f"{int_part}.{frac_part}"
        cleaned = cleaned.replace(",", "")
    else:
        cleaned = cleaned.replace(",", "")
    try:
        return float(cleaned)
    except ValueError:
        return None


def _within_tolerance(a: float, b: float) -> bool:
    reference = max(abs(a), abs(b))
    if reference == 0:
        return True
    return abs(a - b) / reference <= _TOLERANCE


@dataclass
class ValidationResult:
    check: str
    passed: bool
    detail: str


def check_tax_arithmetic(fields: dict) -> ValidationResult:
    subtotal = _to_float(fields.get("subtotal", ""))
    tax = _to_float(fields.get("tax", ""))
    total = _to_float(fields.get("total", ""))
    if subtotal is None or tax is None or total is None:
        return ValidationResult(
            "tax_arithmetic", False, "subtotal, tax, or total missing or unparseable"
        )
    expected_total = subtotal + tax
    passed = _within_tolerance(expected_total, total)
    return ValidationResult(
        "tax_arithmetic",
        passed,
        f"subtotal + tax = {expected_total:.2f}, total = {total:.2f}",
    )


def check_items_sum(fields: dict) -> ValidationResult:
    items = fields.get("items") or []
    if not items:
        return ValidationResult("items_sum", False, "no line items")
    prices = [_to_float(item.get("price", "")) for item in items]
    if any(p is None for p in prices):
        return ValidationResult("items_sum", False, "one or more item prices unparseable")
    items_total = sum(prices)
    subtotal = _to_float(fields.get("subtotal", ""))
    if subtotal is None:
        return ValidationResult("items_sum", False, "subtotal missing or unparseable")
    passed = _within_tolerance(items_total, subtotal)
    return ValidationResult(
        "items_sum", passed, f"sum(items) = {items_total:.2f}, subtotal = {subtotal:.2f}"
    )


def validate(fields: dict) -> list[ValidationResult]:
    return [check_tax_arithmetic(fields), check_items_sum(fields)]


def needs_review(results: list[ValidationResult]) -> bool:
    return any(not r.passed for r in results)
