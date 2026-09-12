"""Source-bound image facts and constrained English/Indonesian message extraction.

Messages are parsed as data only. No message text is executed or used as a prompt.
Unknown messages remain visible in the audit; they cannot introduce instructions.
"""
import hashlib
import json
import re
from pathlib import Path
from common import day, money


# Phrases that mark a mentioned figure as NOT yet confirmed (proposal, pending
# approval, awaiting sign-off). A message containing one of these is still
# read and marked handled, but must not create a confirmed-income or
# confirmed-payment fact; inventing unsupported future income/payments is
# explicitly disallowed by the challenge rules.
NEGATED_CONFIRMATION = (
    'not approved', 'not yet approved', 'not been approved', 'pending approval',
    'not confirmed', 'not yet confirmed', 'not been confirmed', 'yet to be confirmed',
    'has not been confirmed', 'is not confirmed', 'not finalized', 'not yet finalized',
    'belum disetujui', 'belum dikonfirmasi', 'belum final',
)


def negated_near(text, *keywords):
    """True only if a non-confirmation phrase shares a clause with one of these
    keywords, not merely the same message. A single message can report one
    confirmed fact alongside an unrelated pending one (e.g. "Salary USD 2000
    is confirmed; commission is pending approval."); checking the whole
    message for a negation phrase would wrongly cancel the confirmed fact.
    """
    for clause in re.split(r'[.;!?\n]', text):
        if any(k in clause for k in keywords) and any(n in clause for n in NEGATED_CONFIRMATION):
            return True
    return False


def event_context_hash(event):
    fields = ('description', 'category', 'direction', 'amount', 'currency',
              'event_date', 'settlement_date', 'status')
    return hashlib.sha256(json.dumps({k: event.get(k, '') for k in fields}, sort_keys=True).encode()).hexdigest()


def image_facts(dataset, cache_path, events=None, extractor=None, request_ids=None,
                user_ids=None, allow_legacy=False):
    """Extract only images relevant to this run; cache by bytes AND financial context.

Returns (facts, unresolved). A malformed dataset row (an unsafe image identifier,
or a link with no valid same-user event) is a data-integrity problem and still
raises immediately. Everything downstream of that -- a missing image file, no
configured/available extraction, an ambiguous or invalid extraction result, or
two images disagreeing on one event's amount -- is instead recorded in
`unresolved` (event_id -> reason) rather than raised, so one unresolvable image
does not stop the whole run from producing output for every other request. No
amount is invented for these; the caller must not use fact-less events and
should route the affected requests to manual review instead.
"""
    from common import rows
    by_id = {e['event_id']: e for e in (events if events is not None else rows(dataset / 'financial_events.csv'))}
    cache = json.loads(Path(cache_path).read_text()) if allow_legacy and Path(cache_path).exists() else {}
    facts, unresolved = {}, {}
    for link in rows(dataset / 'images.csv'):
        if user_ids is not None and link['user_id'] not in user_ids:
            continue
        if request_ids is not None and link['request_id'] and link['request_id'] not in request_ids:
            continue
        image_id = link['image_id']
        if not re.fullmatch(r'image_[A-Za-z0-9_-]+', image_id):
            raise ValueError('Unsafe image identifier')
        event = by_id.get(link['related_event_id'])
        if not event or event['user_id'] != link['user_id']:
            raise ValueError(f'{image_id}: needs a valid same-user financial event association')
        try:
            path = dataset / 'media' / 'images' / (image_id + '.png')
            if not path.is_file():
                raise ValueError(f'Missing image file: {image_id}')
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            fact = cache.get(image_id)
            legacy_valid = (fact and fact.get('sha256') == digest
                            and fact.get('event_context_sha256') == event_context_hash(event)
                            and fact.get('related_event_id') == event['event_id']
                            and fact.get('currency') == event['currency'])
            if legacy_valid and not (extractor and extractor.refresh):
                if money(fact['amount']) < 0 or not fact.get('source_quote'):
                    raise ValueError(f'{image_id}: invalid legacy evidence')
                if extractor:
                    extractor.usage.legacy_hits += 1
            else:
                if extractor is None:
                    raise ValueError(f'{image_id}: no reusable evidence; configure vision extraction')
                fact = extractor.extract(path, event)
            if fact['currency'] != event['currency']:
                raise ValueError(f'{image_id}: image/event currency conflict')
        except ValueError as exc:
            unresolved[event['event_id']] = str(exc)
            facts.pop(event['event_id'], None)
            continue
        if event['event_id'] in facts and money(facts[event['event_id']]['amount']) != money(fact['amount']):
            unresolved[event['event_id']] = f'{image_id}: conflicting image amounts need review'
            facts.pop(event['event_id'], None)
            continue
        facts[event['event_id']] = {**fact, **link}
    return facts, unresolved


def interpret(messages, request):
    result = {'sources': [], 'unhandled': []}
    for m in sorted(messages, key=lambda x: (x['sent_at'], x['message_id'])):
        if m['request_id'] and m['request_id'] != request['request_id']:
            continue
        if m['sent_at'][:10] > request['request_date']:
            continue
        t = m['message_text'].lower()
        # Amounts may use thousands grouping, either Western (USD 1,500) or
        # Indian lakh/crore style (INR 1,50,000); commas can fall at any
        # digit width, so match any run of comma-grouped digits and strip
        # the commas after matching rather than assuming fixed 3-digit groups.
        amounts = [(cur, num.replace(',', '')) for cur, num in re.findall(
            r'\b(INR|IDR|USD|EUR|ZAR)\s+([0-9]+(?:,[0-9]+)*(?:\.[0-9]+)?)',
            m['message_text'])]
        dates = re.findall(r'\b\d{4}-\d{2}-\d{2}\b', t)
        handled = False
        if m['source_type'] == 'employer':
            if any(s in t for s in ['employment has ended', 'seasonal contract has ended', 'kontrak musiman saat ini telah berakhir', 'pekerjaan anda telah berakhir']):
                result['salary_ended'] = True
                handled = True
            if amounts and any(s in t for s in ['salary', 'monthly pay', 'gaji']):
                if negated_near(t, 'salary', 'monthly pay', 'gaji'):
                    # A proposed or not-yet-approved figure is not confirmed income.
                    handled = True
                else:
                    result['salary'] = {'amount': amounts[0][1], 'currency': amounts[0][0],
                                        'date': dates[0] if dates else None,
                                        'next_only': 'unpaid leave' in t or 'cuti tanpa' in t}
                    # A later explicit confirmation supersedes an earlier termination.
                    result['salary_ended'] = False
                    handled = True
            if dates and ('now expected' in t or 'menggantikan' in t or 'revised date' in t):
                result['salary_delay'] = dates[0]
                handled = True
            if 'one-time arrears' in t or 'tunggakan satu kali' in t:
                # Arrears are not extrapolated or counted as unreceived cash.
                handled = True
        if ('rent' in t or 'sewa' in t) and '%' in t and any(s in t for s in ['increases', 'menaikkan']):
            match = re.search(r'(\d+(?:\.\d+)?)%', t)
            if match:
                result['rent_increase_percent'] = match[1]
                handled = True
        if amounts and dates and ('approved an invoice' in t or 'menyetujui pembayaran faktur' in t):
            if negated_near(t, 'approved an invoice', 'menyetujui pembayaran faktur'):
                # Not yet approved; do not treat as confirmed future cash.
                handled = True
            else:
                result['confirmed_invoice'] = {'amount': amounts[0][1], 'currency': amounts[0][0], 'date': dates[0]}
                handled = True
        if m['related_event_id'] and ('bill is still outstanding' in t or 'tagihan masih' in t):
            result.setdefault('retry_events', []).append(m['related_event_id'])
            handled = True
        if any(s in t for s in ['pending', 'belum', 'still processing', 'still in payment processing', 'no cash', 'no units', 'between your two accounts', 'antara dua rekening', 'claim is now closed', 'payment was received', 'order was paid', 'reimbursement', 'penggantian']):
            handled = True  # The structured cash state remains authoritative.
        result['sources' if handled else 'unhandled'].append(m['message_id'])
    return result
