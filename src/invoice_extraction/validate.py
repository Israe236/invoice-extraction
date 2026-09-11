"""Business rules applied to extracted invoice fields.

These run on *model predictions*, never on ground truth, and they do not need a
reference answer. That is the point: in production there is no gold label, so
the only automatic signal that an extraction is wrong is that it contradicts
itself. An invoice where HT + TVA does not equal TTC is either a bad extraction
or a bad invoice, and both need a human.

Three outcomes per check, not two:

- ``PASS``    the rule was evaluated and held
- ``FAIL``    the rule was evaluated and was violated
- ``SKIPPED`` the rule could not be evaluated, because the document genuinely
              does not carry the fields it needs

The distinction matters. An earlier version of this module returned
``passed=False`` when a field was missing, which meant every CORD-v2 receipt
was reported as an arithmetic violation purely for having no ICE number. A
check that cannot run is unverified, not failed, and the report says so.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from enum import Enum

from invoice_extraction.normalize import amounts_equal, parse_amount, parse_date

# Rounding noise on a real invoice is cents, but line items are rounded
# individually before being summed, so the tolerance has to absorb n roundings.
_ABS_TOLERANCE = 0.02
_REL_TOLERANCE = 0.005

_MAX_INVOICE_AGE = timedelta(days=365 * 10)
_ICE_DIGITS = 15
_IF_DIGIT_RANGE = (7, 9)  # Moroccan Identifiant Fiscal, in practice 7-9 digits
_KNOWN_CURRENCIES = frozenset({"MAD", "EUR", "USD", "GBP", "IDR"})


class Status(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    SKIPPED = "skipped"


class Severity(str, Enum):
    OK = "ok"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class CheckResult:
    """One business rule, evaluated."""

    check: str
    status: Status
    detail: str
    # How bad it is when this particular rule fails. Arithmetic contradictions
    # block automatic posting; a malformed ICE is worth a human glance.
    on_fail: Severity = Severity.ERROR

    @property
    def failed(self) -> bool:
        return self.status is Status.FAIL


@dataclass
class ValidationReport:
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def severity(self) -> Severity:
        """Worst severity among the rules that actually failed.

        Skipped rules never raise severity -- they are reported separately via
        `unverified`, because "we could not check this" is a different message
        to the accountant than "this is wrong".
        """
        if any(c.failed and c.on_fail is Severity.ERROR for c in self.checks):
            return Severity.ERROR
        if any(c.failed for c in self.checks):
            return Severity.WARNING
        return Severity.OK

    @property
    def valid(self) -> bool:
        return not any(c.failed for c in self.checks)

    @property
    def needs_review(self) -> bool:
        """True if a human should look at this document before it is posted."""
        return self.severity is not Severity.OK

    @property
    def unverified(self) -> list[str]:
        return [c.check for c in self.checks if c.status is Status.SKIPPED]

    def to_dict(self) -> dict:
        return {
            "valid": self.valid,
            "severity": self.severity.value,
            "needs_review": self.needs_review,
            "unverified": self.unverified,
            "checks": [
                {
                    "check": c.check,
                    "status": c.status.value,
                    "detail": c.detail,
                    "on_fail": c.on_fail.value,
                }
                for c in self.checks
            ],
        }


def _skip(name: str, reason: str, on_fail: Severity = Severity.ERROR) -> CheckResult:
    return CheckResult(name, Status.SKIPPED, reason, on_fail)


def check_tax_arithmetic(fields: dict) -> CheckResult:
    """Total HT + TVA = Total TTC. The single most useful rule on an invoice."""
    subtotal = parse_amount(fields.get("subtotal"))
    tax = parse_amount(fields.get("tax"))
    total = parse_amount(fields.get("total"))

    missing = [
        name
        for name, value in (("subtotal", subtotal), ("tax", tax), ("total", total))
        if value is None
    ]
    if missing:
        return _skip("tax_arithmetic", f"not checkable: missing {', '.join(missing)}")

    expected = subtotal + tax
    ok = amounts_equal(expected, total, abs_tol=_ABS_TOLERANCE, rel_tol=_REL_TOLERANCE)
    return CheckResult(
        "tax_arithmetic",
        Status.PASS if ok else Status.FAIL,
        f"subtotal {subtotal:.2f} + tax {tax:.2f} = {expected:.2f}, "
        f"stated total {total:.2f} (difference {abs(expected - total):.2f})",
    )


def check_items_sum(fields: dict) -> CheckResult:
    """The line items have to add up to the subtotal."""
    items = fields.get("items") or []
    if not isinstance(items, list) or not items:
        return _skip("items_sum", "not checkable: no line items extracted")

    subtotal = parse_amount(fields.get("subtotal"))
    if subtotal is None:
        return _skip("items_sum", "not checkable: subtotal missing or unparseable")

    prices = [parse_amount(item.get("price")) for item in items if isinstance(item, dict)]
    unparseable = sum(1 for p in prices if p is None)
    if unparseable:
        return _skip("items_sum", f"not checkable: {unparseable} item price(s) unparseable")

    items_total = sum(prices)
    # Each line was rounded independently, so allow the tolerance to grow with
    # the number of lines rather than flagging every long invoice.
    ok = amounts_equal(
        items_total,
        subtotal,
        abs_tol=_ABS_TOLERANCE * max(1, len(prices)),
        rel_tol=_REL_TOLERANCE,
    )
    return CheckResult(
        "items_sum",
        Status.PASS if ok else Status.FAIL,
        f"{len(prices)} line item(s) sum to {items_total:.2f}, subtotal {subtotal:.2f} "
        f"(difference {abs(items_total - subtotal):.2f})",
    )


def check_tax_rate_plausible(fields: dict) -> CheckResult:
    """The implied VAT rate should be one Morocco actually uses.

    Rates are 20% (standard), 14%, 10% and 7%. A rate outside that set usually
    means the model picked up the wrong number as the tax amount.
    """
    subtotal = parse_amount(fields.get("subtotal"))
    tax = parse_amount(fields.get("tax"))
    if subtotal is None or tax is None:
        return _skip(
            "tax_rate_plausible",
            "not checkable: subtotal or tax missing",
            on_fail=Severity.WARNING,
        )
    if subtotal == 0:
        return _skip("tax_rate_plausible", "not checkable: subtotal is zero", Severity.WARNING)

    implied = tax / subtotal
    known = (0.20, 0.14, 0.10, 0.07, 0.0)
    closest = min(known, key=lambda r: abs(r - implied))
    ok = abs(closest - implied) <= 0.005
    return CheckResult(
        "tax_rate_plausible",
        Status.PASS if ok else Status.FAIL,
        f"implied VAT rate {implied * 100:.2f}%"
        + ("" if ok else f", nearest standard Moroccan rate {closest * 100:.0f}%"),
        on_fail=Severity.WARNING,
    )


def check_date_valid(fields: dict, *, today: date | None = None) -> CheckResult:
    """The date must parse, and must not be in the future or absurdly old."""
    raw = fields.get("date")
    if not raw or not str(raw).strip():
        return _skip("date_valid", "not checkable: no date extracted", on_fail=Severity.WARNING)

    parsed = parse_date(raw)
    if parsed is None:
        return CheckResult(
            "date_valid", Status.FAIL, f"{raw!r} is not a recognisable date", Severity.WARNING
        )

    reference = today or date.today()
    if parsed > reference:
        return CheckResult(
            "date_valid",
            Status.FAIL,
            f"{parsed.isoformat()} is in the future (today is {reference.isoformat()})",
            Severity.WARNING,
        )
    if reference - parsed > _MAX_INVOICE_AGE:
        return CheckResult(
            "date_valid",
            Status.FAIL,
            f"{parsed.isoformat()} is more than 10 years old",
            Severity.WARNING,
        )
    return CheckResult("date_valid", Status.PASS, f"{parsed.isoformat()}", Severity.WARNING)


def check_ice_format(fields: dict) -> CheckResult:
    """A Moroccan ICE is exactly 15 digits."""
    raw = fields.get("ice")
    if not raw or not str(raw).strip():
        return _skip("ice_format", "not checkable: no ICE on this document", Severity.WARNING)

    digits = "".join(ch for ch in str(raw) if ch.isdigit())
    ok = len(digits) == _ICE_DIGITS
    return CheckResult(
        "ice_format",
        Status.PASS if ok else Status.FAIL,
        f"{len(digits)} digits (expected {_ICE_DIGITS})",
        Severity.WARNING,
    )


def check_if_format(fields: dict) -> CheckResult:
    """A Moroccan Identifiant Fiscal is a short run of digits (7-9 in practice)."""
    raw = fields.get("if_number")
    if not raw or not str(raw).strip():
        return _skip("if_format", "not checkable: no IF on this document", Severity.WARNING)

    digits = "".join(ch for ch in str(raw) if ch.isdigit())
    low, high = _IF_DIGIT_RANGE
    ok = low <= len(digits) <= high
    return CheckResult(
        "if_format",
        Status.PASS if ok else Status.FAIL,
        f"{len(digits)} digits (expected {low}-{high})",
        Severity.WARNING,
    )


def check_currency_known(fields: dict) -> CheckResult:
    raw = fields.get("currency")
    if not raw or not str(raw).strip():
        return _skip("currency_known", "not checkable: no currency extracted", Severity.WARNING)

    code = str(raw).strip().upper()
    ok = code in _KNOWN_CURRENCIES
    return CheckResult(
        "currency_known",
        Status.PASS if ok else Status.FAIL,
        f"{code}" + ("" if ok else " is not a currency code we recognise"),
        Severity.WARNING,
    )


def check_required_fields(fields: dict) -> CheckResult:
    """An extraction with no total is not usable, whatever else it got right."""
    required = ("total",)
    missing = [name for name in required if parse_amount(fields.get(name)) is None]
    return CheckResult(
        "required_fields",
        Status.PASS if not missing else Status.FAIL,
        "all required fields present" if not missing else f"missing: {', '.join(missing)}",
    )


_CHECKS = (
    check_required_fields,
    check_tax_arithmetic,
    check_items_sum,
    check_tax_rate_plausible,
    check_date_valid,
    check_ice_format,
    check_if_format,
    check_currency_known,
)


def validate(fields: dict) -> ValidationReport:
    """Run every business rule against one extracted invoice."""
    return ValidationReport(checks=[check(fields) for check in _CHECKS])


def needs_review(report: ValidationReport) -> bool:
    """Kept as a function for callers that prefer it to the property."""
    return report.needs_review
