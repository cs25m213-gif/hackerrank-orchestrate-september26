"""Shared exact-money and dataset utilities; no network dependencies."""
import csv
from datetime import date
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COLUMNS = 'request_id amount_safe_to_pay affordability_status recommended_payment_method payment_plan earliest_date_for_full_payment spending_changes_needed decision_explanation'.split()
ZERO = Decimal('0')

def money(value):
    if value is None or str(value).strip() == '':
        raise ValueError('Missing monetary amount; linked image evidence is required')
    result = Decimal(str(value))
    if not result.is_finite():
        raise ValueError('Non-finite monetary amount')
    return result.quantize(Decimal('.01'), rounding=ROUND_HALF_UP)

def fmt(value):
    return format(money(value), 'f').rstrip('0').rstrip('.') if money(value) else '0'

def day(value):
    return date.fromisoformat(value)

def rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))
