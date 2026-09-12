"""Small, deterministic estimators selected using each user's own cash history.

No request labels, model calls, or cross-user currency mixing are involved.
"""
from decimal import Decimal
from math import ceil
from common import money


def upper_quantile(values, q=Decimal('.8')):
    ordered = sorted(values)
    return ordered[max(0, ceil(len(ordered) * q) - 1)]


def horizon_quantile(occurrences_in_horizon):
    """Per-instance quantile calibrated to the SUM over the forecast horizon.

Reserving every upcoming instance of a recurring cost at its own 80th-percentile
value is only a single-shot bound: it is the right level of caution when a
category occurs once in the horizon. Once it repeats many times, applying that
same high quantile to every instance independently compounds a conservative
margin n times instead of once, and the reserved total grows far faster than a
realistic worst case for the SUM. By the central limit theorem the confidence
margin of a sum of n similar, roughly independent draws grows with sqrt(n), not
n, so the marginal per-instance conservatism needed to keep an ~80th-percentile
bound on the total should shrink toward the median as n grows. q=0.8 at n=1
matches the single-shot case exactly and relaxes toward 0.5 as occurrences
increase.
"""
    return Decimal('.5') + Decimal('.3') / Decimal(max(1, occurrences_in_horizon)).sqrt()


def estimate_expense(history, occurrences_in_horizon=1):
    """Choose a recent-history window via rolling one-step quantile loss.

Window selection uses past-only predictions on at most 12 observations. The
target quantile is calibrated by horizon_quantile() using how many times this
category is expected to recur within the forecast horizon, so a frequent habit
(e.g. groceries every few days) is not reserved at the same per-instance
tail-risk level as a one-off monthly bill. This is empirical calibration, not a
guarantee of future coverage.
"""
    values = [money(e['amount']) for e in sorted(history, key=lambda e: e['event_date'])]
    q = horizon_quantile(occurrences_in_horizon)
    if len(values) < 6:
        predicted = upper_quantile(values, q)
        return money(predicted), f'upper_quantile_{q:.3f} (limited history, n={len(values)}, horizon_occurrences={occurrences_in_horizon})'
    scores = []
    for window in (3, 6, 12):
        loss = Decimal(0)
        for i in range(max(3, len(values) - 12), len(values)):
            prediction = upper_quantile(values[max(0, i-window):i], q)
            error = values[i] - prediction
            loss += q * error if error >= 0 else (q - 1) * error
        scores.append((loss, -window, window))
    window = min(scores)[2]
    predicted = upper_quantile(values[-window:], q)
    # Do not average away an established recent increase in a commitment.
    recent = values[-3:]
    if recent[0] <= recent[1] <= recent[2]:
        predicted = max(predicted, recent[-1])
    return money(predicted), f'upper_quantile_{q:.3f}, window={window}, horizon_occurrences={occurrences_in_horizon}, rolling_history_selection'


def first_safe_offset(headroom, total):
    """O(horizon) full-payment date; also rejects pre-payment baseline deficits."""
    if not headroom or min(headroom) < 0:
        return None
    suffix_min = headroom[-1]
    first = None
    for i in range(len(headroom) - 1, -1, -1):
        suffix_min = min(suffix_min, headroom[i])
        if suffix_min >= total:
            first = i
    return first
