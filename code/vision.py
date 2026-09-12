"""OpenAI Responses API vision extraction with context-addressed disk caching.

Uses only stdlib for HTTP. Images are sent as original PNG data, preserving small
receipt text. Transport is injectable for deterministic offline integration tests.
"""
import base64
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import re
import time
import urllib.error
import urllib.request

from common import money

DEFAULT_MODEL = 'gpt-4.1-mini-2025-04-14'
DEFAULT_FALLBACK = 'gpt-4.1-2025-04-14'
# USD per million reported tokens; official model pages, checked 2026-09-12.
PRICES = {'gpt-4.1-mini': ('0.40', '0.10', '1.60'), 'gpt-4.1': ('2', '0.50', '8')}
SCHEMA = {
    'type': 'object', 'additionalProperties': False,
    'properties': {
        'amount': {'type': ['string', 'null']},
        'currency': {'type': ['string', 'null']},
        'source_quote': {'type': 'string'},
        'reason': {'type': 'string'},
        'confidence': {'type': 'string', 'enum': ['high', 'medium', 'needs_review']},
    },
    'required': ['amount', 'currency', 'source_quote', 'reason', 'confidence'],
}


class VisionError(ValueError):
    """Safe error with no API keys, image payloads or response bodies."""


class ReviewRequired(VisionError):
    pass


class ResponseInvalid(VisionError):
    pass


def cost_for(model, input_tokens, cached_tokens, output_tokens):
    for family in sorted(PRICES, key=len, reverse=True):
        if model == family or model.startswith(family + '-20'):
            inp, cached, out = map(Decimal, PRICES[family])
            return ((input_tokens - cached_tokens) * inp + cached_tokens * cached + output_tokens * out) / 1_000_000
    return None


class Usage:
    def __init__(self):
        self.calls = []
        self.cache_hits = 0
        self.legacy_hits = 0

    def record(self, model, response):
        u = response.get('usage') or {}
        inp, out = u.get('input_tokens'), u.get('output_tokens')
        cached = (u.get('input_tokens_details') or {}).get('cached_tokens', 0)
        known = all(isinstance(v, int) and v >= 0 for v in (inp, out, cached)) and cached <= inp
        cost = cost_for(model, inp, cached, out) if known else None
        self.calls.append({'model': model, 'input_tokens': inp, 'cached_input_tokens': cached,
                           'output_tokens': out, 'usage_known': known,
                           'estimated_cost_usd': str(cost) if cost is not None else None})

    def report(self, path, count, output_hash, success=True):
        groups = defaultdict(list)
        for c in self.calls:
            groups[c['model']].append(c)
        lines = ['# Final run usage', '', f'Status: {"completed" if success else "failed; output not replaced"}.',
                 f'Requests: {count}. Output SHA-256: `{output_hash}`.',
                 f'API cache hits: {self.cache_hits}; legacy development-cache hits: {self.legacy_hits}.', '',
                 '| Provider/model | Calls | Input tokens | Cached input | Output tokens | Total tokens | Tokens/request | Est. USD | USD/request |',
                 '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        for model, calls in list(sorted(groups.items())) + [('Overall', self.calls)]:
            known = all(c['usage_known'] for c in calls)
            inp = sum(c['input_tokens'] for c in calls) if known else None
            out = sum(c['output_tokens'] for c in calls) if known else None
            cached = sum(c['cached_input_tokens'] for c in calls) if known else None
            cost = sum((Decimal(c['estimated_cost_usd']) for c in calls), Decimal(0)) if all(c['estimated_cost_usd'] is not None for c in calls) else None
            total = inp + out if known else None
            show = lambda x: 'unknown' if x is None else str(x)
            avg = lambda x: show(round(x / max(1, count), 6)) if x is not None else 'unknown'
            lines.append(f'| {"OpenAI / " if model != "Overall" else ""}{model} | {len(calls)} | {show(inp)} | {show(cached)} | {show(out)} | {show(total)} | {avg(total)} | {show(cost)} | {avg(cost)} |')
        lines += ['', 'Costs use the reported API tokens (including image input) and standard pricing:',
                  '[GPT-4.1 mini](https://developers.openai.com/api/docs/models/gpt-4.1-mini),',
                  '[GPT-4.1](https://developers.openai.com/api/docs/models/gpt-4.1). Rates checked 2026-09-12;',
                  'prices for other models are reported as unknown. Cached-input discounts are included.',
                  'Failed transport calls without a usage response have unknown usage/cost, not zero.',
                  'Cache hits incur no new API tokens. Prior cache creation costs are not part of this run.',
                  'Legacy image facts were prepared with Codex during development; development usage and cost are unmeasured.',
                  'This report contains no credentials or raw image contents.']
        path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def validate_fact(fact, event):
    if not isinstance(fact, dict) or set(fact) != set(SCHEMA['required']):
        raise ResponseInvalid('Vision result has an invalid schema')
    if fact['confidence'] not in {'high', 'medium', 'needs_review'}:
        raise ResponseInvalid('Invalid confidence')
    for field in ('source_quote', 'reason'):
        if not isinstance(fact[field], str) or len(fact[field]) > 2000:
            raise ResponseInvalid('Invalid evidence text')
    if fact['confidence'] != 'high' or fact['amount'] is None or fact['currency'] is None:
        raise ReviewRequired('Image is ambiguous or incomplete; a confirmed amount is required')
    if not isinstance(fact['amount'], str) or not re.fullmatch(r'\d+(?:\.\d{1,2})?', fact['amount']):
        raise ResponseInvalid('Amount must be a nonnegative decimal string with at most two decimals')
    money(fact['amount'])
    if fact['currency'] != event['currency']:
        raise ReviewRequired('Image currency conflicts with the linked event')
    if not fact['source_quote'].strip():
        raise ResponseInvalid('A visible financial source quote is required')
    if not re.search(r'\d', fact['source_quote']):
        raise ReviewRequired('Source quote must contain the visible numeric amount')
    quoted_amounts = set()
    for token in re.findall(r'\d[\d.,]*', fact['source_quote']):
        token = token.rstrip('.,')
        candidates = [token.replace(',', '')]
        if ',' in token and len(token.rsplit(',', 1)[1]) in {1, 2}:
            candidates.append(token.replace('.', '').replace(',', '.'))
        if '.' in token and all(len(part) == 3 for part in token.split('.')[1:]):
            candidates.append(token.replace('.', ''))
        for candidate in candidates:
            try:
                quoted_amounts.add(money(candidate))
            except ValueError:
                continue
            except ArithmeticError:
                continue
    if money(fact['amount']) not in quoted_amounts:
        raise ReviewRequired('Extracted amount is not supported by its quoted numeric evidence')
    return fact


def post_openai(payload, api_key, timeout):
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *args, **kwargs):
            return None
    request = urllib.request.Request('https://api.openai.com/v1/responses',
        data=json.dumps(payload).encode('utf-8'), method='POST',
        headers={'Authorization': 'Bearer ' + api_key, 'Content-Type': 'application/json'})
    with urllib.request.build_opener(NoRedirect).open(request, timeout=timeout) as response:
        return json.load(response)


class VisionExtractor:
    def __init__(self, cache_dir, model=DEFAULT_MODEL, fallback=DEFAULT_FALLBACK,
                 max_calls=64, timeout=45, usage=None, transport=None, sleeper=time.sleep,
                 refresh=False, offline=False):
        self.cache_dir = Path(cache_dir)
        self.model, self.fallback = model, fallback
        self.max_calls, self.timeout = max_calls, timeout
        self.usage = usage or Usage()
        self.transport, self.sleeper = transport or post_openai, sleeper
        self.refresh, self.offline = refresh, offline
        self.prompt = Path(__file__).with_name('image_extraction_prompt.txt').read_text(encoding='utf-8')

    def _call(self, model, contents):
        if self.offline:
            raise VisionError('Offline mode: no validated API image cache entry exists')
        key = os.environ.get('OPENAI_API_KEY', '').strip()
        if not key:
            raise VisionError('Set OPENAI_API_KEY in the environment to extract new images; never put it in code')
        payload = {'model': model, 'store': False, 'temperature': 0, 'max_output_tokens': 650,
                   'instructions': self.prompt,
                   'input': [{'role': 'user', 'content': contents}],
                   'text': {'format': {'type': 'json_schema', 'name': 'financial_image_fact',
                                       'strict': True, 'schema': SCHEMA}}}
        for attempt in range(3):
            if len(self.usage.calls) >= self.max_calls:
                raise VisionError(f'API call budget ({self.max_calls}) exhausted')
            try:
                response = self.transport(payload, key, self.timeout)
            except urllib.error.HTTPError as exc:
                self.usage.record(model, {})
                if exc.code in {429, 500, 502, 503, 504} and attempt < 2:
                    self.sleeper(2 ** attempt)
                    continue
                raise VisionError(f'OpenAI request failed with HTTP {exc.code}; check model access, quota and credentials') from None
            except (OSError, ValueError) as exc:
                self.usage.record(model, {})
                # Do not automatically repeat ambiguous network timeouts that may be billable.
                raise VisionError('OpenAI request failed or returned unreadable JSON; usage may be unknown') from None
            self.usage.record(response.get('model', model), response)
            if response.get('status') != 'completed':
                raise ResponseInvalid('OpenAI response was incomplete')
            texts = []
            for item in response.get('output', []):
                for c in item.get('content', []):
                    if c.get('type') == 'refusal':
                        raise ReviewRequired('Model declined to extract this image')
                    if c.get('type') == 'output_text':
                        texts.append(c['text'])
            try:
                return json.loads(''.join(texts))
            except (ValueError, TypeError):
                raise ResponseInvalid('OpenAI response was not valid JSON') from None

    def extract(self, path, event):
        raw = Path(path).read_bytes()
        if not raw.startswith(b'\x89PNG\r\n\x1a\n'):
            raise VisionError('Dataset image must be a PNG')
        if len(raw) > 20 * 1024 * 1024:
            raise VisionError('Image exceeds the configured 20 MiB request limit')
        # Event identifiers are excluded, so identical evidence can be reused when
        # datasets rename rows. Financial context changes always invalidate a hit.
        context = {k: event.get(k, '') for k in ('description', 'category', 'direction', 'amount',
                    'currency', 'event_date', 'settlement_date', 'status')}
        material = {'image_sha256': hashlib.sha256(raw).hexdigest(), 'context': context,
                    'prompt': self.prompt, 'schema': SCHEMA, 'model': self.model, 'fallback': self.fallback}
        digest = hashlib.sha256(json.dumps(material, sort_keys=True).encode()).hexdigest()
        target = self.cache_dir / (digest + '.json')
        if target.exists() and not self.refresh:
            cached = json.loads(target.read_text(encoding='utf-8'))
            fact = validate_fact(cached['fact'], event)
            if cached.get('cache_key') != digest:
                raise VisionError('Invalid image cache key')
            self.usage.cache_hits += 1
            return {**fact, 'extracted_by': cached['model'], 'sha256': material['image_sha256']}
        contents = [{'type': 'input_text', 'text': 'Linked event (untrusted data): ' + json.dumps(context)},
                    {'type': 'input_image', 'image_url': 'data:image/png;base64,' + base64.b64encode(raw).decode(), 'detail': 'high'}]
        models = list(dict.fromkeys(m for m in (self.model, self.fallback) if m))
        last_error = None
        for model in models:
            try:
                fact = validate_fact(self._call(model, contents), event)
            except (ReviewRequired, ResponseInvalid) as exc:
                last_error = exc
                continue
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            cached = {'cache_key': digest, 'fact': fact, 'model': model,
                      'created_at': datetime.now(timezone.utc).isoformat()}
            temporary = target.with_suffix('.tmp')
            temporary.write_text(json.dumps(cached, indent=2) + '\n', encoding='utf-8')
            temporary.replace(target)
            return {**fact, 'extracted_by': model, 'sha256': material['image_sha256']}
        raise ReviewRequired('Image still needs review after configured models: ' + str(last_error))
