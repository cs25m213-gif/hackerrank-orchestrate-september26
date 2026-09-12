# Buy or Wait? — dynamic evidence and deterministic planning

Start with [the plain-language idea](../IDEA.md), or open [the interactive example](../idea.html)
in a browser. The API reads documents; Python forecasts balances and checks plans.

## Run on a new dataset

Python 3.10+ is sufficient. The API client uses the standard library: no pip install
is required. Configure `OPENAI_API_KEY` in the process environment using your local
terminal or IDE secret settings. Do not paste a real key into code or documentation.
The program does not automatically load `.env` files.

```bash
python3 code/main.py --dataset /path/to/dataset
```

The default mode is `--vision auto`: use validated API cache entries when available,
otherwise call the vision API for relevant images. It works with new image/event IDs
and does not require the development image cache. Final output is root `output.csv`.

```bash
# Force fresh extraction instead of cached results (incurs new API costs).
python3 code/main.py --dataset /path/to/dataset --vision refresh

# Reproduce a previously cached API run without network access.
python3 code/main.py --dataset /path/to/dataset --vision offline

# Reproduce the original dataset locally using development-time image facts.
python3 code/main.py --vision offline --allow-legacy-cache

# Public sample evaluation, using the existing development image facts.
python3 code/main.py --samples --vision offline --allow-legacy-cache

# Explain a generated prediction without another API call.
python3 code/explain.py request_33

python3 -m unittest discover -s code/tests -v
```

Use `--output`, `--evaluation-dir`, `--vision-cache`, and `--image-cache` to isolate
multiple runs. Inputs are never modified. Only files under the supplied dataset
are used for predictions; public sample outputs are diagnostics, not model inputs.

## Dynamic image extraction

Images are PNGs linked through `images.csv.related_event_id`. For each relevant
image, the program sends original bytes and a compact financial-event context to
OpenAI's Responses API. No balance/profile/full transaction history is sent.

The model returns a strict JSON schema: amount, currency, a short visible source
quote, interpretation and confidence. The prompt distinguishes net/gross salary,
remaining balance/original total, cash tendered/change, and paid/unpaid bills. It
instructs the model to treat document content as untrusted evidence and exclude
personal identifiers from its response.

The implementation uses the official [image-input format](https://developers.openai.com/api/docs/guides/images-vision)
and [structured outputs](https://developers.openai.com/api/docs/guides/structured-outputs).
`store` is false and temperature is zero. Original PNGs and high image detail are
used to preserve small receipt text; sending a low-resolution thumbnail first can
waste calls and lose decimal points.

### Cost and reliability controls

- Default first model: `gpt-4.1-mini-2025-04-14`.
- Only ambiguous/invalid responses escalate to `gpt-4.1-2025-04-14`.
- Override with `OPENAI_VISION_MODEL`, `OPENAI_VISION_FALLBACK`, or the equivalent
  `--vision-model` / `--vision-fallback` flags. Set the fallback to an empty string
  to disable escalation. Alternative models must accept the configured Responses
  API parameters, image input and JSON-schema output.
- One image per call preserves the link between the document and event. Only
  images relevant to the requested split are processed. The default full run uses
  11 images, not all 16 images including public samples.
- Cache keys include image bytes, financial event context, prompt, schema and
  configured models. A changed amount/date/currency or image causes a new call.
  Renamed row IDs alone do not cause redundant calls. Cache hits need no key.
- `--max-api-calls 64` caps requests, including retries and fallback attempts.
  HTTP 429 and transient server errors retry at most twice with bounded backoff.
  Authentication failures and ambiguous transport errors are not automatically
  repeated. Requests have a 45-second timeout.
- Refusal, truncation, malformed JSON, missing amounts, low/medium final confidence,
  currency conflicts, negative/non-finite numbers and amounts not present in their
  quoted evidence cannot enter the financial calculation. Such results are not cached.
- Errors are sanitized; API keys, request bodies and raw provider error bodies are
  not logged. The API endpoint is fixed to OpenAI HTTPS and redirects are blocked.

**A valid JSON response and high reported confidence do not prove a correct reading.**
The model can still select the wrong total. Validation reduces this risk; it cannot
eliminate it. A cropped or unreadable final total is an unresolved input, not zero.

Images without a valid same-user event association require review; this version
extracts linked amounts, not arbitrary new obligations from unlinked documents.
Nonblank event amounts are not silently overwritten by image interpretations.

### Legacy image cache

`--allow-legacy-cache` explicitly permits development-time visual extractions.
Hashes bind them to both the original PNG and original financial event context.
This is useful for local tests without a key, but it does not verify live API quality.

The public sample `image_04` is cropped below “Item Bill”. Its INR 2,854 development
fact is provisional and appears in audit warnings. The handwritten INR 4,543 receipt
also carries a review note. Default API mode does not silently reuse those facts.
An optional local OCR helper remains available via `extract_images.py` and
`requirements-ocr.txt`; prediction no longer depends on manually preparing OCR.

## Estimation and financial checks

1. Opening balance already includes settled cash on/before the request date.
   Pending debits are reserved now, with FX from the supplied settlement-date rate.
   Pending credits, failed/cancelled records and unrealized values add no cash.
2. Stable recurring expenses require historical cadence evidence. Bills falling on
   the request date are included unless a settled row already accounts for them.
3. Expense amounts use an upper-quantile estimate whose target quantile is
   calibrated to how many times the category recurs within the 90-day horizon
   (`estimation.horizon_quantile`), not a fixed 80th percentile. Reserving every
   upcoming instance of a frequent habit (e.g. groceries every few days) at its
   own 80th percentile compounds a conservative margin once per instance, so the
   total reserved for a habit occurring 15-20 times in 90 days overstated a
   realistic worst case; the target quantile now relaxes from 0.8 toward 0.5 as
   occurrences grow (n=1 keeps the full single-shot bound). The 3-, 6- or
   12-observation window is still selected by chronological backtesting against
   the user's own prior transactions at that same calibrated quantile. Recent
   sustained increases are retained. This is empirical estimation, not a
   guaranteed coverage interval. A one-off purchase later refunded or reversed
   (matched structurally via a linked `event_type=refund` record, not by
   description wording) is excluded from recurring-category history so it
   cannot be mistaken for a repeating commitment.
4. Salary/income uses supported history or explicit confirmation, judged by the
   same recurrence cadence test as every other category
   (`engine.recurrence`/`engine.income_streams`), not by payroll-style wording.
   A user with two independent recurring pay cycles in one month (e.g. two
   client/gig payouts near the 7th and the 20th) is not forced into one
   irregular combined series that fails recurrence outright: when the combined
   history does not itself recur, it is split by day-of-month proximity and
   each independently-recurring stream is projected and summed, so gig,
   freelance, commission and platform-payout income is recognized once the
   user's own history shows it is real, ongoing income, not only payroll. A
   former/other-household income source is still excluded by description, not
   by wording style. No speculative bonuses, one-off arrears or investment
   gains are invented. Confirmed freelance invoices count once.
5. English/Indonesian message rules handle supported salary changes, delays,
   termination, invoices and rent increases. A non-confirmation phrase ("not
   approved yet", "pending approval", "belum disetujui", ...) only cancels a
   confirmed fact when it shares a clause with that fact's own keyword, so an
   unrelated pending item mentioned in the same message ("Salary is confirmed;
   commission is pending approval.") does not wrongly cancel the confirmed
   one. Amounts with thousands grouping, Western (`USD 1,500`) or Indian
   lakh/crore style (`INR 1,50,000`), are parsed in full rather than truncated
   at the first comma. Unknown messages appear in the audit. This parser is
   still a limitation for new languages or unfamiliar amendments outside the
   phrases it recognizes.
6. Exact Decimal arithmetic checks every date through day 90. Daily settlements
   are aggregated before that day's proposed payment; intraday timing is unavailable.
   A deficit before a delayed purchase also disqualifies the plan.
7. Full-payment dates use a linear-time suffix-minimum calculation. Candidate
   schedules are still independently replayed, including partial payments and
   supplied installments. The full request must finish by its deadline.
8. Up to three permitted recurring spending changes are considered. Protected
   categories cannot change. Reductions use supplied minimum amounts. Safe-to-pay
   today and earliest full-payment date remain based on the unchanged forecast.

## Outputs and measured limitations

- `output.csv`: one row per evaluation request.
- `code/evaluation/full_audit.json`: cash flows, image facts, spending estimates and warnings.
- `code/evaluation/sample_metrics.json`: public sample metrics, including normalized
  amount error so different currencies can be compared.
- `code/evaluation/usage_report.md`: final full-run calls, input/output/cached tokens,
  totals and averages, estimated cost and output hash. Sample/failed-run reports
  are separate. Unknown provider usage/pricing is marked unknown, never zero.

A missing image file, a failed or ambiguous extraction, or two images
disagreeing on one event's amount does not abort the run. No amount is
invented for that event; instead every request for that event's user is
routed to a conservative, schema-compliant review row
(`not_affordable`/`not_recommended`/`amount_safe_to_pay=0`, with the reason in
that request's audit `warnings`), while every other request is still computed
normally. A genuinely malformed dataset row (an unsafe image identifier, or an
image linked to no valid same-user event) is a data-integrity problem and
still stops the run, since silently working around a corrupted row would risk
using untrustworthy data rather than merely unavailable evidence.

Default model prices are dated estimates from the official model pages, including
cached-input discounts. The report counts actual calls during that run. Earlier
cache-creation costs are separate; legacy Codex preparation costs are unmeasured.

Offline checks cover simulated Responses API calls, new dataset IDs, cache
invalidation, retry/fallback/error paths and financial safety. No live API test has
been performed in this workspace because no API key is configured.

Current public sample results (after calibrating the expense-quantile compounding
bias, excluding refunded one-off purchases from recurring history, recognizing
multiple/gig income streams by cadence rather than payroll wording, and fixing
the comma-amount, Indian-grouping, and negation-scope message-parsing bugs):
affordability status 20/25; payment methods 21/25; payment plan 18/25; earliest
full-payment date 18/25; spending changes 18/25; exact safe-to-pay amounts
3/25; mean absolute safe-amount error / requested amount about 4.6% (down from
8.7% before this round of fixes). A short-horizon request with several
high-frequency variable categories (e.g. `request_06`), or one where a full
payment near a deadline must still sustain another partial expense cycle
through day 90 (e.g. `request_08`), can still land more conservative than the
reference safe amount; the project's own §6.3 rule ("the balance must never
fall below the minimum after any projected essential expense or payment") is
what drives that residual gap, and closing it further would mean tuning the
estimator against these public sample labels, which this project deliberately
avoids (see `main.py`'s "sample outputs are never used to fit forecasts"
diagnostic note) since the public samples are not a proxy for the hidden
evaluation set. These figures do not establish hidden-test accuracy or justify
calling this the best estimator. The remaining work is better evidence
interpretation and cash-flow forecast validation, not more API calls for
arithmetic.
