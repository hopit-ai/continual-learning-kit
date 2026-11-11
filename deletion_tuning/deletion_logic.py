"""Torch-free reference arithmetic and the fixed smoke reservation."""
import math
import hashlib
import json


def sharpness(step, total=40, start=100., end=10000.):
    if not isinstance(step, int) or step < 0 or total <= 0 or start <= 0 or end <= 0:
        raise ValueError('invalid mask schedule')
    return start + (end - start) * min(step / total, 1.)


def wanda_deletions(scores, eta=1 / 2048):
    if not scores or not 0 <= eta <= 1 or not all(math.isfinite(x) for x in scores):
        raise ValueError('invalid Wanda scores')
    threshold = sorted(scores)[max(1, int(len(scores) * eta)) - 1]
    return [x < threshold for x in scores]


def calibration_batches(document):
    """Validate a hash-bound, task-only ordered calibration artifact."""
    payload = {k: v for k, v in document.items() if k != 'sha256'}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if document.get('sha256') != digest or document.get('schema') != 'deletion-calibration.v1':
        raise ValueError('calibration identity differs')
    ids = document['task_ids']
    batches = document['batches']
    if len(ids) != 32 or len(batches) != 4 or not document.get('schedule_sha256'):
        raise ValueError('calibration requires first 32 bound task prompts')
    for batch in batches:
        if set(batch) != {'input_ids', 'attention_mask'} or len(batch['input_ids']) != 8:
            raise ValueError('calibration requires four batches of eight')
        if len(batch['attention_mask']) != 8:
            raise ValueError('calibration attention shape differs')
        widths = {len(row) for row in batch['input_ids'] + batch['attention_mask']}
        if len(widths) != 1 or not 0 < next(iter(widths)) <= 1024:
            raise ValueError('calibration width differs')
    return batches


def smoke_price():
    # Modal Sandbox non-preemptible rates, observed 2026-11-11 (A10 list $0.000306/GPU-s; L4 had no capacity twice).
    # Sandbox GPU is 3x the task price. CPU/memory have published rounded rates.
    rate = 2 * .000306 * 3 + 4 * .00003942 + 32 * .00000667
    return dict(gpu='A10G', count=2, cores=4, memory_gib=32, seconds=3900, startup_seconds=480,
                rate_usd_second=rate, runtime_max_usd=round(rate * 3900, 6),
                image_build_seconds=0, image_build_usd=0., startup_reserve_usd=1., receipt_egress_usd=.001,
                total_cap_usd=round(rate * (3900 + 480) + .001, 6),
                expected_low_usd=round(rate * 1400 + .001, 6),
                expected_high_usd=round(rate * 3900 + .001, 6))
