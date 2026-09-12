#!/usr/bin/env python3
"""Explain one existing prediction using its audited cash forecast; no API calls."""
import argparse
import json
from pathlib import Path
from common import ROOT, rows, money, fmt


def explain(request_id, output, audit_path):
    predictions = {r['request_id']: r for r in rows(output)}
    audits = {a['request_id']: a for a in json.loads(audit_path.read_text())}
    if request_id not in predictions or request_id not in audits:
        raise ValueError('Request not found. Generate predictions/audit first with code/main.py.')
    p, a = predictions[request_id], audits[request_id]
    currency = a['currency']
    print(f'BUY OR WAIT? — {request_id}\n')
    print(f'1. Start with {currency} {a["opening_balance"]} on {a["request_date"]}.')
    print(f'2. Keep {currency} {a["minimum_balance"]} untouched as the minimum balance.')
    incoming = sum((money(f['amount']) for f in a['flows'] if money(f['amount']) > 0), money(0))
    outgoing = -sum((money(f['amount']) for f in a['flows'] if money(f['amount']) < 0), money(0))
    print(f'3. Forecast 90 days: {currency} {fmt(incoming)} income and {fmt(outgoing)} outgoings.')
    print('   Dates matter: money arriving next month cannot pay a bill due tomorrow.')
    print(f'4. The lowest projected amount above the reserve is {currency} {a["minimum_baseline_headroom"]}.')
    print(f'   Safe to pay today, capped at the requested amount: {currency} {p["amount_safe_to_pay"]}.')
    print(f'5. Recommendation: {p["recommended_payment_method"].replace("_", " ")}.')
    print(f'   Payment schedule: {p["payment_plan"]}.')
    print(f'   Earliest safe single full payment: {p["earliest_date_for_full_payment"] or "not found within 90 days"}.')
    print(f'   Spending changes: {p["spending_changes_needed"]}.')
    print('\nWhy: ' + p['decision_explanation'])
    if a.get('images'):
        print('\nEvidence read from images:')
        for fact in a['images']:
            print(f'  {fact["image_id"]}: {fact["currency"]} {fact["amount"]}; {fact["reason"]}')
    if a['warnings']:
        print('\nEvidence to review: ' + '; '.join(a['warnings']))
    print('\nUpcoming cash events (first 10):')
    for f in a['flows'][:10]:
        print(f'  {f["date"]}  {currency} {f["amount"]:>12}  {f["category"]} ({f["event_id"]})')
    print('\nThese are forecasts, not guarantees. The API reads evidence; Python checks the payment maths.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('request_id')
    parser.add_argument('--output', type=Path, default=ROOT/'output.csv')
    parser.add_argument('--audit', type=Path, default=Path(__file__).resolve().parent/'evaluation/full_audit.json')
    args = parser.parse_args()
    explain(args.request_id, args.output, args.audit)
