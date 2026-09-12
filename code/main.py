#!/usr/bin/env python3
"""Run from any working directory: python /path/to/code/main.py."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
from common import ROOT, COLUMNS, money, rows
from engine import solve
from evidence import image_facts
from vision import VisionExtractor, Usage, DEFAULT_MODEL, DEFAULT_FALLBACK


def run(args):
    usage = Usage()
    try:
        return run_with_usage(args, usage)
    except Exception:
        args.evaluation_dir.mkdir(parents=True, exist_ok=True)
        usage.report(args.evaluation_dir / 'failed_run_usage.md', getattr(usage, 'request_count', 0), 'none', success=False)
        raise


def run_with_usage(args, usage):
    dataset = args.dataset.resolve()
    requests = rows(dataset / ('sample_requests.csv' if args.samples else 'requests.csv'))
    usage.request_count = len(requests)
    if len({r['request_id'] for r in requests}) != len(requests):
        raise ValueError('Duplicate request IDs')
    profiles = {p['user_id']: p for p in rows(dataset / 'financial_profiles.csv')}
    all_events = rows(dataset / 'financial_events.csv')
    extractor = VisionExtractor(args.vision_cache, model=args.vision_model, fallback=args.vision_fallback,
                                max_calls=args.max_api_calls, usage=usage,
                                refresh=args.vision == 'refresh', offline=args.vision == 'offline')
    evaluation = args.evaluation_dir
    evaluation.mkdir(parents=True, exist_ok=True)
    user_ids = {r['user_id'] for r in requests}
    facts, unresolved = image_facts(dataset, args.image_cache, all_events, extractor,
                        {r['request_id'] for r in requests}, user_ids,
                        allow_legacy=args.allow_legacy_cache)
    # A missing image file, an extraction failure, or two images disagreeing on
    # one event leaves that event's amount unresolved. No amount is invented for
    # it: the event is left out of its user's history, and every request for
    # that user is routed to manual review below instead of stopping the whole
    # run, so one bad image cannot prevent producing output for every other
    # request (a prior unresolved image used to abort the entire batch).
    blocked_users = {}
    events, messages, options = defaultdict(list), defaultdict(list), defaultdict(list)
    for e in all_events:
        if e['user_id'] not in user_ids:
            continue
        if not e['amount']:
            fact = facts.get(e['event_id'])
            if not fact:
                reason = unresolved.get(e['event_id'], 'missing linked image amount')
                blocked_users.setdefault(e['user_id'], []).append(f'{e["event_id"]}: {reason}')
                continue
            e['amount'] = fact['amount']
        money(e['amount'])
        events[e['user_id']].append(e)
    for m in rows(dataset / 'messages.csv'):
        messages[m['user_id']].append(m)
    for o in rows(dataset / 'request_payment_options.csv'):
        options[o['request_id']].append(o)
    rates = rows(dataset / 'exchange_rates.csv')
    predictions, audits = [], []
    for r in requests:
        if r['user_id'] in blocked_users:
            reasons = blocked_users[r['user_id']]
            result = dict(request_id=r['request_id'], amount_safe_to_pay='0', affordability_status='not_affordable',
                          recommended_payment_method='not_recommended', payment_plan='none',
                          earliest_date_for_full_payment='', spending_changes_needed='none',
                          decision_explanation='Evidence could not be resolved for a linked image needed for this '
                                                'user; no amount was invented. This request needs manual review '
                                                'before a decision can be made; see the audit warnings.')
            audit = {'request_id': r['request_id'], 'request_date': r['request_date'], 'evidence_unresolved': True,
                     'warnings': list(reasons)}
            predictions.append(result)
            audits.append(audit)
            continue
        result, audit = solve(r, profiles[r['user_id']], events[r['user_id']], messages[r['user_id']], options[r['request_id']], rates)
        audit['images'] = []
        for fact in facts.values():
            if fact['user_id'] == r['user_id']:
                audit['images'].append({k: fact[k] for k in ('image_id', 'related_event_id', 'amount', 'currency', 'source_quote', 'confidence', 'reason')})
            if fact['user_id'] == r['user_id'] and fact['confidence'] != 'high':
                audit['warnings'].append(f'{fact["image_id"]}: {fact["reason"]}')
        assert 0 <= money(result['amount_safe_to_pay']) <= money(r['requested_amount'])
        predictions.append(result)
        audits.append(audit)
    output = args.output or ROOT / ('sample_predictions.csv' if args.samples else 'output.csv')
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix('.tmp')
    with temporary.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, COLUMNS)
        writer.writeheader()
        writer.writerows(predictions)
    temporary.replace(output)
    name = 'samples' if args.samples else 'full'
    (evaluation / f'{name}_audit.json').write_text(json.dumps(audits, indent=2) + '\n')
    if args.samples:
        scores = {c: sum(p[c] == r[c] for p, r in zip(predictions, requests)) for c in COLUMNS[1:-1]}
        scores['amount_safe_to_pay'] = sum(money(p['amount_safe_to_pay']) == money(r['amount_safe_to_pay']) for p, r in zip(predictions, requests))
        relative_errors = [abs(money(p['amount_safe_to_pay']) - money(r['amount_safe_to_pay'])) / max(money(r['requested_amount']), 1)
                           for p, r in zip(predictions, requests)]
        report = {'sample_count': len(requests), 'exact_matches': scores,
                  'mean_safe_amount_error_fraction_of_request': float(sum(relative_errors) / len(requests)),
                  'note': 'Diagnostic only; sample outputs are never used to fit forecasts.'}
        (evaluation / 'sample_metrics.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report, indent=2))
        usage.report(evaluation / 'sample_usage.md', len(requests), hashlib.sha256(output.read_bytes()).hexdigest())
    else:
        digest = hashlib.sha256(output.read_bytes()).hexdigest()
        usage.report(evaluation / 'usage_report.md', len(requests), digest)
    warnings = sum(bool(a['warnings']) for a in audits)
    print(f'Wrote {len(predictions)} predictions to {output}; {warnings} requests have evidence review notes in {name}_audit.json.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=ROOT / 'dataset')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--image-cache', type=Path, default=Path(__file__).resolve().parent / 'cache/image_facts.json')
    parser.add_argument('--samples', action='store_true')
    parser.add_argument('--evaluation-dir', type=Path, default=Path(__file__).resolve().parent / 'evaluation')
    parser.add_argument('--vision', choices=['auto', 'offline', 'refresh'], default='auto')
    parser.add_argument('--vision-cache', type=Path, default=Path(__file__).resolve().parent / 'cache/vision')
    parser.add_argument('--vision-model', default=os.environ.get('OPENAI_VISION_MODEL', DEFAULT_MODEL))
    parser.add_argument('--vision-fallback', default=os.environ.get('OPENAI_VISION_FALLBACK', DEFAULT_FALLBACK))
    parser.add_argument('--max-api-calls', type=int, default=64)
    parser.add_argument('--allow-legacy-cache', action='store_true', help='Allow original development image facts, including review notes')
    run(parser.parse_args())
