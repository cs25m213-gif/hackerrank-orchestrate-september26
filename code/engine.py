"""Deterministic 90-day cash-flow reconstruction and plan verification."""
from calendar import monthrange
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import timedelta
from decimal import Decimal
from itertools import combinations, product
from statistics import median
import re
from common import ZERO, day, money, fmt
from evidence import interpret
from estimation import estimate_expense, first_safe_offset


@dataclass(frozen=True)
class Flow:
    date: object
    amount: Decimal  # positive credit, negative debit
    event_id: str
    category: str
    projected: bool = False


def month_after(d, offset=1, due_day=None):
    n = d.year * 12 + d.month - 1 + offset
    y, m = divmod(n, 12)
    return d.replace(year=y, month=m + 1, day=min(due_day or d.day, monthrange(y, m + 1)[1]))


def recurrence(history):
    """Require >=3 observations and a stable monthly or short-interval cadence."""
    dates = sorted(set(day(e['event_date']) for e in history))
    if len(dates) < 3:
        return None
    days = Counter(d.day for d in dates)
    due, count = days.most_common(1)[0]
    gaps = [(b - a).days for a, b in zip(dates, dates[1:])]
    if count / len(dates) >= .7 and 25 <= median(gaps) <= 35:
        return ('monthly', due)
    interval = round(median(gaps))
    if 2 <= interval <= 35 and sum(abs(g - interval) <= 2 for g in gaps) / len(gaps) >= .7:
        return ('days', interval)
    return None


def income_streams(history):
    """Split settled salary-category credit history into independent income
    streams when the combined history does not itself show a stable cadence.

A user can genuinely have two (or more) separate recurring pay cycles, e.g. two
client/gig payouts landing near the 7th and near the 20th of every month.
Interleaving both into one series produces irregular 13-18 day gaps that pass
neither the monthly nor the short-interval recurrence test, silently dropping
real, evidenced future income. If the whole history already recurs on its own
(the common single-income case), it is returned unsplit. Otherwise history is
grouped by day-of-month proximity and only groups that independently recur are
kept, so this can only recover evidence the combined check missed, never
invent a cadence that isn't there.
"""
    if not history:
        return []
    if recurrence(history):
        return [history]
    days = sorted(set(day(e['event_date']).day for e in history))
    groups = [[days[0]]]
    for d in days[1:]:
        if d - groups[-1][-1] <= 8:
            groups[-1].append(d)
        else:
            groups.append([d])
    clustered = [[e for e in history if day(e['event_date']).day in g] for g in groups]
    return [c for c in clustered if recurrence(c)]


def forecast(request, profile, events, messages, rates):
    start = day(request['request_date'])
    end = start + timedelta(days=90)
    evidence = interpret(messages, request)
    final_payroll = [e for e in events if 'final employer payroll' in e['description'].lower()
                     and e['status'] == 'settled' and day(e['settlement_date']) <= start]
    if final_payroll and not evidence.get('salary'):
        evidence['salary_ended'] = True
    flows, recurring, warnings = [], {}, []
    evidence['expense_estimates'] = []
    conversion = {(r['rate_date'], r['from_currency'], r['to_currency']): Decimal(r['rate']) for r in rates}

    def convert(amount, currency, when):
        value = money(amount)
        if currency == profile['home_currency']:
            return value
        key = (when.isoformat(), currency, profile['home_currency'])
        if key not in conversion:
            raise ValueError(f'Missing dated exchange rate: {key}')
        return money(value * conversion[key])

    # Explicit settled/cancelled successors suppress only prior representations of
    # the SAME debit, not an investment purchase followed by a sale or valuation.
    by_id = {e['event_id']: e for e in events}
    superseded = set()
    for e in events:
        prev = by_id.get(e['linked_event_id'])
        if (prev and e['status'] in {'settled', 'cancelled'}
                and prev['status'] in {'pending', 'scheduled'}
                and prev['direction'] == e['direction']
                and e['event_type'] == prev['event_type']):
            superseded.add(prev['event_id'])
    valid = [e for e in events if e['status'] not in {'failed', 'cancelled', 'unrealized'}
             and e['event_id'] not in superseded]
    # A purchase later reimbursed or reversed is a one-off, not evidence of a
    # recurring commitment in its category; description wording varies, so this
    # is keyed off the structural refund/reversal link rather than text matching.
    refunded_purchases = {e['linked_event_id'] for e in events
                          if e['event_type'] == 'refund' and e['linked_event_id']}
    explicit_salary = []
    seen = set()
    for e in valid:
        when = day(e['settlement_date'] or e['event_date'])
        if e['direction'] == 'credit' and e['status'] != 'settled':
            if e['status'] != 'scheduled' or e['category'] != 'salary':
                continue
        if e['status'] == 'settled' and when <= start:
            continue  # Opening available balance already contains this cash.
        if e['category'] == 'salary' and e['direction'] == 'credit':
            if evidence.get('salary_ended'):
                continue
            explicit_salary.append(e)
            if evidence.get('salary_delay'):
                when = day(evidence['salary_delay'])
        conversion_date = when
        when = start if e['status'] == 'pending' and e['direction'] == 'debit' else max(start, when)
        if when > end:
            continue
        amount = convert(e['amount'], e['currency'], conversion_date)
        amendment = evidence.get('salary')
        if e['category'] == 'salary' and e['direction'] == 'credit' and amendment:
            amount = convert(amendment['amount'], amendment['currency'], when)
        key = (e['description'], e['direction'], amount, when, e['status'])
        if key in seen:
            continue
        seen.add(key)
        flows.append(Flow(when, amount if e['direction'] == 'credit' else -amount, e['event_id'], e['category']))

    groups = defaultdict(list)
    salary_history = []
    for e in valid:
        if e['status'] != 'settled' or day(e['settlement_date'] or e['event_date']) > start:
            continue
        if e['category'] == 'salary' and e['direction'] == 'credit':
            # A former/other-household income source is excluded by description
            # (it is not this user's current, ongoing income); everything else
            # in the salary category is judged by the same recurrence() cadence
            # test used for every other category below, not by wording. Gig,
            # freelance, commission and platform-payout income is real income
            # once the user's own history shows a consistent cadence for it;
            # requiring specific payroll-style wording silently dropped all
            # future income for non-payroll workers.
            if not re.search(r'previous employer|final employer|second household', e['description'], re.I):
                salary_history.append(e)
        elif e['direction'] == 'debit' and not e['linked_event_id'] and e['event_id'] not in refunded_purchases:
            if re.search(r'authorization|card charge|card purchase|one.time|investment|taxi fare|airline|wallet payment|tote bag|bulk groceries|delivered grocery|pharmacy purchase|outstanding', e['description'], re.I):
                continue
            groups[(e['category'], e['currency'])].append(e)

    for (category, currency), hist in groups.items():
        hist.sort(key=lambda e: e['event_date'])
        cadence = recurrence(hist)
        if not cadence:
            continue
        latest = hist[-1]
        last_date = day(latest['event_date'])
        if (start - last_date).days > 62:
            continue
        # Collect the future occurrence dates once, both to size the
        # occurrence-calibrated quantile and to place the projected flows.
        occurrence_dates = []
        when = last_date
        while True:
            when = month_after(when, due_day=cadence[1]) if cadence[0] == 'monthly' else when + timedelta(days=cadence[1])
            if when > end:
                break
            if when < start:
                continue
            occurrence_dates.append(when)
        amount, estimation_method = estimate_expense(hist, len(occurrence_dates))
        if category == 'rent' and 'rent_increase_percent' in evidence:
            amount = money(amount * (1 + Decimal(evidence['rent_increase_percent']) / 100))
        recurring[latest['event_id']] = {**latest, 'forecast_amount': amount}
        evidence['expense_estimates'].append({'event_id': latest['event_id'], 'category': category,
                                             'amount': fmt(amount), 'currency': currency,
                                             'method': estimation_method, 'observations': len(hist),
                                             'cadence': list(cadence), 'horizon_occurrences': len(occurrence_dates)})
        for when in occurrence_dates:
            if any(f.category == category and f.date == when and f.amount < 0 for f in flows):
                continue
            flows.append(Flow(when, -convert(amount, currency, when), latest['event_id'], category, True))

    amendment = evidence.get('salary')
    if not evidence.get('salary_ended'):
        salary_history.sort(key=lambda e: e['event_date'])
        # A message amendment or an explicitly scheduled credit names a single
        # source and overrides everything else, so it is projected once as
        # before. Otherwise each independently-evidenced income stream (see
        # income_streams()) is projected on its own cadence and summed.
        streams = [salary_history] if (explicit_salary or amendment) else income_streams(salary_history)
        for stream in streams:
            salary_cadence = recurrence(stream)
            if not (explicit_salary or amendment or salary_cadence):
                continue
            source = explicit_salary[0] if explicit_salary else (stream[-1] if stream else None)
            if not (source or (amendment and amendment['date'])):
                continue
            normal_day = Counter(day(e['event_date']).day for e in stream).most_common(1)[0][0] if stream else (day(amendment['date']).day if amendment and amendment['date'] else day(source['settlement_date']).day)
            amount = amendment['amount'] if amendment else source['amount']
            currency = amendment['currency'] if amendment else source['currency']
            if amendment and amendment['date']:
                when = day(amendment['date'])
            elif explicit_salary:
                when = day(explicit_salary[0]['settlement_date'])
            else:
                when = start.replace(day=min(normal_day, monthrange(start.year, start.month)[1]))
                if when <= start:
                    when = month_after(when, due_day=normal_day)
            if evidence.get('salary_delay'):
                when = day(evidence['salary_delay'])
                normal_day = when.day
            index = 0
            while when <= end:
                duplicate = any(f.category == 'salary' and f.date == when
                                and (source is None or f.event_id == source['event_id']) for f in flows)
                if when > start and not duplicate:
                    v = amount
                    if amendment and amendment['next_only'] and index > 0 and stream:
                        v = median(money(e['amount']) for e in stream)
                        currency = stream[-1]['currency']
                    flows.append(Flow(when, convert(v, currency, when), source['event_id'] if source else evidence['sources'][-1], 'salary', True))
                when = month_after(when, due_day=normal_day)
                index += 1
            if explicit_salary or amendment:
                break
    invoice = evidence.get('confirmed_invoice')
    if invoice:
        when = day(invoice['date'])
        if start < when <= end:
            value = convert(invoice['amount'], invoice['currency'], when)
            if not any(f.date == when and f.amount == value for f in flows):
                flows.append(Flow(when, value, 'confirmed_invoice', 'salary'))
    for event_id in evidence.get('retry_events', []):
        e = by_id.get(event_id)
        if e and e['status'] == 'failed' and not any(x['linked_event_id'] == event_id and x['status'] in {'scheduled', 'pending', 'settled'} for x in events):
            flows.append(Flow(start, -convert(e['amount'], e['currency'], start), event_id, e['category']))
    if evidence['unhandled']:
        warnings.append('Messages needing review: ' + ', '.join(evidence['unhandled']))
    return flows, recurring, evidence, warnings


def trajectory(start, opening, minimum, flows, changes=None):
    changes = changes or {}
    daily = defaultdict(lambda: ZERO)
    for f in flows:
        value = f.amount
        if f.projected and f.event_id in changes and value < 0:
            value = -changes[f.event_id]
        daily[f.date] += value
    balance = money(opening)
    result = []
    for i in range(91):
        when = start + timedelta(days=i)
        balance += daily[when]
        result.append(balance - money(minimum))
    return result


def feasible(headroom, payments, start):
    debits = defaultdict(lambda: ZERO)
    for when, amount in payments:
        offset = (when - start).days
        if not 0 <= offset < len(headroom) or amount < 0:
            return False
        debits[offset] += amount
    paid = ZERO
    for i, capacity in enumerate(headroom):
        paid += debits[i]
        if capacity - paid < 0:
            return False
    return True


def solve(request, profile, events, messages, options, rates):
    start, deadline = day(request['request_date']), day(request['desired_completion_date'])
    total = money(request['requested_amount'])
    flows, recurring, evidence, warnings = forecast(request, profile, events, messages, rates)
    base = trajectory(start, profile['current_available_balance'], profile['minimum_balance_to_keep'], flows)
    safe = max(ZERO, min(total, min(base)))
    offset = first_safe_offset(base, total)
    earliest = start + timedelta(days=offset) if offset is not None else None
    methods = set(profile['payment_methods_user_will_consider'].split('|'))
    candidates = []

    def consider(method, payments, headroom, actions=(), option_id=''):
        if not payments or payments[-1][0] > deadline or not feasible(headroom, payments, start):
            return
        status = ('affordable_with_plan' if actions or method in {'partial_payment', 'installments'}
                  else 'affordable_now' if payments[0][0] == start else 'affordable_later')
        rank = (bool(actions), sum(a for _, a in payments), payments[0][0], len(payments), option_id, actions)
        candidates.append((rank, method, payments, status, actions, headroom))

    def enumerate_plans(headroom, actions=()):
        if 'full_payment' in methods:
            offset = first_safe_offset(headroom, total)
            when = start + timedelta(days=offset) if offset is not None else None
            if when:
                consider('full_payment' if when == start or actions else 'wait', [(when, total)], headroom, actions)
        if ('partial_payment' in methods and request['allows_partial_payment'].lower() == 'true'
                and ZERO < safe < total and earliest and earliest > start):
            consider('partial_payment', [(start, safe), (earliest, total - safe)], headroom, actions)
        if 'installments' in methods and profile['max_installment_months']:
            for o in options:
                if o['payment_method'] != 'installments':
                    continue
                n = int(o['number_of_payments'])
                first, interval = day(o['first_payment_date']), int(o['payment_frequency_days'] or 0)
                if n < 1 or (n > 1 and interval <= 0):
                    continue
                plan = [(first + timedelta(days=i * interval), money(o['payment_amount'])) for i in range(n)]
                months = int(profile['max_installment_months'])
                if n > months or plan[-1][0] > month_after(first, months):
                    continue
                # Allow cent rounding in the supplied equal-payment schedule.
                if abs(sum(a for _, a in plan) - money(o['total_payable_amount'])) > Decimal('.01') * n:
                    continue
                consider('installments', plan, headroom, actions, o['payment_option_id'])
    enumerate_plans(base)
    # No-change plans always outrank changes, so avoid unnecessary search.
    if not candidates:
        change_options = []
        protect = set(profile['expense_categories_to_protect'].split('|'))
        stop = set(profile['expense_categories_user_is_willing_to_stop'].split('|'))
        reduce = set(profile['expense_categories_user_is_willing_to_reduce'].split('|'))
        for event_id, e in sorted(recurring.items()):
            if e['category'] in protect or e['currency'] != profile['home_currency']:
                continue
            choices = []
            if e['category'] in stop and 'stoppable' in e['flexibility']:
                choices.append((event_id, ZERO, 'stop:' + event_id))
            if e['category'] in reduce and 'reducible' in e['flexibility'] and e['minimum_allowed_amount']:
                value = money(e['minimum_allowed_amount'])
                if value < e['forecast_amount']:
                    choices.append((event_id, value, f'reduce_to:{event_id}:{fmt(value)}'))
            if choices:
                change_options.append(choices)
        for count in range(1, min(3, len(change_options)) + 1):
            for selected in combinations(change_options, count):
                for combo in product(*selected):
                    changes = {e: v for e, v, _ in combo}
                    actions = tuple(a for _, _, a in combo)
                    headroom = trajectory(start, profile['current_available_balance'], profile['minimum_balance_to_keep'], flows, changes)
                    enumerate_plans(headroom, actions)
    explanation = f'90-day forecast in {profile["home_currency"]}; safe today {fmt(safe)}, minimum balance {profile["minimum_balance_to_keep"]}.'
    result = dict(request_id=request['request_id'], amount_safe_to_pay=fmt(safe), affordability_status='not_affordable',
                  recommended_payment_method='not_recommended', payment_plan='none',
                  earliest_date_for_full_payment=earliest.isoformat() if earliest else '', spending_changes_needed='none',
                  decision_explanation=explanation + ' No eligible plan safely completes the request by its deadline.')
    if candidates:
        _, method, plan, status, actions, headroom = min(candidates, key=lambda c: c[0])
        result.update(affordability_status=status, recommended_payment_method=method,
                      payment_plan='|'.join(f'{d.isoformat()}:{fmt(a)}' for d, a in plan),
                      spending_changes_needed='|'.join(actions) or 'none',
                      decision_explanation=explanation + f' {method} completes on {plan[-1][0]}; all projected balances preserve the reserve.' + (' Requires the listed flexible spending changes.' if actions else ' No spending changes required.'))
        assert feasible(headroom, plan, start)
    audit = {'request_id': request['request_id'], 'request_date': request['request_date'],
             'opening_balance': profile['current_available_balance'],
             'minimum_balance': profile['minimum_balance_to_keep'], 'currency': profile['home_currency'],
             'evidence': evidence, 'warnings': warnings,
             'minimum_baseline_headroom': fmt(min(base)), 'flows': [dict(date=f.date.isoformat(), amount=fmt(f.amount), event_id=f.event_id, category=f.category, projected=f.projected) for f in sorted(flows, key=lambda f: (f.date, f.event_id))]}
    return result, audit
