"""Normalisation of the raw strings a VLM produces into comparable values.

Both the evaluation metric and the validation layer need to answer "are these
two amounts the same?" and "is this a real date?". Before this module they each
had their own half-answer, which meant a prediction could pass validation and
still be scored wrong for a purely cosmetic difference.

Invoices in scope are French/Moroccan, so the formats seen in the wild include
`1 234,56` (space thousands, comma decimal), `1.234,56` (dot thousands),
`1,234.56` (anglo) and bare `1234.56`. Dates are day-first (`03/04/2024` is
3 April, not 3 March).
"""

from __future__ import annotations

import re
import unicodedata
from datetime import date, datetime

# Currency symbols and codes that show up glued to amounts on Moroccan invoices.
_CURRENCY_NOISE = re.compile(r"(?i)\b(mad|dh|dhs|dirhams?|eur|euros?|usd)\b|[€$£₣]")
# Anything that is not a digit, separator or sign once currency noise is gone.
_AMOUNT_ALLOWED = re.compile(r"[^0-9,.\-]")
_WHITESPACE = re.compile(r"\s+")

# Day-first first: on a French/Moroccan invoice 03/04/2024 means 3 April.
_DATE_FORMATS = (
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%d/%m/%Y",
    "%d-%m-%Y",
    "%d.%m.%Y",
    "%d/%m/%y",
    "%d-%m-%y",
    "%d.%m.%y",
    "%d %B %Y",
    "%d %b %Y",
    "%B %d, %Y",
    "%b %d, %Y",
)

_ISO_LIKE = re.compile(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}$")


def normalize_text(value: object) -> str:
    """Casefold, strip accents and collapse whitespace, for string comparison.

    Accent stripping matters because the model transcribing `Developpement`
    where the invoice reads `Développement` is an OCR artefact, not an
    extraction error.
    """
    if value is None:
        return ""
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return _WHITESPACE.sub(" ", text).strip().casefold()


def parse_amount(value: object) -> float | None:
    """Parse a money string into a float, or None if it is not a number.

    Handles the four separator conventions listed in the module docstring by
    treating whichever of `.` / `,` appears **last** as the decimal separator,
    which is the only rule that is correct for all of them.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)

    text = _CURRENCY_NOISE.sub(" ", str(value))
    # Non-breaking and thin spaces are common in rendered invoices.
    text = text.replace(" ", " ").replace(" ", " ")
    text = _AMOUNT_ALLOWED.sub("", text)
    if not text or not any(ch.isdigit() for ch in text):
        return None

    negative = text.startswith("-")
    text = text.lstrip("-").replace("-", "")

    last_dot, last_comma = text.rfind("."), text.rfind(",")
    if last_dot == -1 and last_comma == -1:
        cleaned = text
    else:
        sep_index = max(last_dot, last_comma)
        integer_part = text[:sep_index]
        decimal_part = text[sep_index + 1 :]
        # A group of exactly 3 digits after the final separator with no other
        # separator before it is thousands, not decimals: `1,500` is 1500.
        if len(decimal_part) == 3 and not re.search(r"[.,]", integer_part):
            cleaned = text.replace(",", "").replace(".", "")
        else:
            cleaned = re.sub(r"[.,]", "", integer_part) + "." + decimal_part

    try:
        result = float(cleaned)
    except ValueError:
        return None
    return -result if negative else result


def amounts_equal(a: object, b: object, *, abs_tol: float = 0.01, rel_tol: float = 0.005) -> bool:
    """True if two amounts agree within rounding noise.

    Both an absolute and a relative tolerance: `abs_tol` covers cent-level
    rounding on small numbers, `rel_tol` covers invoices in the hundreds of
    thousands where a 1-cent absolute tolerance would be unreasonably strict.
    """
    x, y = parse_amount(a), parse_amount(b)
    if x is None or y is None:
        return False
    difference = abs(x - y)
    return difference <= abs_tol or difference <= rel_tol * max(abs(x), abs(y))


def parse_date(value: object) -> date | None:
    """Parse a date string in any of the formats invoices actually use."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value

    text = _WHITESPACE.sub(" ", str(value)).strip()
    if not text:
        return None

    for fmt in _DATE_FORMATS:
        # Don't let %d/%m/%Y claim an unambiguously ISO string.
        if fmt.startswith("%d") and _ISO_LIKE.match(text):
            continue
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def normalize_date(value: object) -> str:
    """Render a date as ISO `YYYY-MM-DD`, or `""` if it cannot be parsed."""
    parsed = parse_date(value)
    return parsed.isoformat() if parsed else ""


def normalize_ice(value: object) -> str:
    """Reduce an ICE to its digits. Invoices print it as `001234567000078`,
    `001 234 567 000 078` or `ICE: 001234567000078`."""
    if value is None:
        return ""
    return re.sub(r"\D", "", str(value))
