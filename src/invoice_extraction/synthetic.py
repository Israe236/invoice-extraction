"""Synthetic French/Moroccan invoice data generation."""

from dataclasses import dataclass

from faker import Faker

from invoice_extraction.data import ExtractedFields, LineItem

_fake = Faker("fr_FR")

TAX_RATES = (0.20, 0.14, 0.10, 0.07)  # standard + reduced Moroccan VAT rates

PRODUCTS = (
    "Prestation de conseil",
    "Développement logiciel",
    "Maintenance informatique",
    "Fournitures de bureau",
    "Formation professionnelle",
    "Location de matériel",
    "Transport de marchandises",
    "Services de nettoyage",
    "Impression de documents",
    "Abonnement mensuel",
    "Réparation véhicule",
    "Installation électrique",
)


@dataclass
class InvoiceData:
    vendor_name: str
    vendor_address: str
    ice: str
    if_number: str
    client_name: str
    client_address: str
    invoice_number: str
    invoice_date: str
    items: list[LineItem]
    subtotal: str
    tax: str
    total: str
    tax_rate_pct: str


def generate_invoice_data(seed: int | None = None) -> InvoiceData:
    if seed is not None:
        _fake.seed_instance(seed)
    rng = _fake.random

    n_items = rng.randint(1, 6)
    items: list[LineItem] = []
    subtotal = 0.0
    for _ in range(n_items):
        qty = rng.randint(1, 10)
        unit_price = round(rng.uniform(50, 2000), 2)
        line_total = round(qty * unit_price, 2)
        subtotal += line_total
        items.append(LineItem(name=rng.choice(PRODUCTS), qty=str(qty), price=f"{line_total:.2f}"))

    tax_rate = rng.choice(TAX_RATES)
    tax = round(subtotal * tax_rate, 2)
    total = round(subtotal + tax, 2)

    return InvoiceData(
        vendor_name=_fake.company(),
        vendor_address=_fake.address().replace("\n", ", "),
        ice="".join(str(rng.randint(0, 9)) for _ in range(15)),
        if_number="".join(str(rng.randint(0, 9)) for _ in range(8)),
        client_name=_fake.company(),
        client_address=_fake.address().replace("\n", ", "),
        invoice_number=f"FA-{rng.randint(1000, 9999)}",
        invoice_date=_fake.date_between(start_date="-2y", end_date="today").strftime("%d/%m/%Y"),
        items=items,
        subtotal=f"{subtotal:.2f}",
        tax=f"{tax:.2f}",
        total=f"{total:.2f}",
        tax_rate_pct=f"{tax_rate * 100:.0f}",
    )


def to_ground_truth(invoice: InvoiceData) -> ExtractedFields:
    return ExtractedFields(
        items=invoice.items,
        subtotal=invoice.subtotal,
        tax=invoice.tax,
        total=invoice.total,
        ice=invoice.ice,
        if_number=invoice.if_number,
    )
