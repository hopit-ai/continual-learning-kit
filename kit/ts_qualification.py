"""Deterministic update-one request qualification before any optimizer update."""
from kit import ts_arms as ts
from kit.nm_common import atomic, need, digest


class ProbeComplete(Exception):
    """Stop an explicitly requested generation-only qualification."""


def capture(batch, native):
    slots = []
    data = batch.non_tensor_batch
    for i, extra in enumerate(data['extra_info']):
        if extra.get('m3_replay'):
            continue
        need(extra['nm_update'] == 1, 'identity probe must precede update one')
        mask = batch.batch['response_mask'][i].bool()
        slot = dict(position=int(extra['nm_position']), question_id=str(extra['index']),
                    request=data.get('ts_attempt0_request', [None] * len(data['extra_info']))[i],
                    token_ids=batch.batch['responses'][i][mask].tolist())
        if native['ts']['arm'] == 'M':
            records = data.get('ts_attempts', [None] * len(data['extra_info']))[i]
            slot['finish_reasons'] = [data.get('ts_finish_reason', [None] * len(data['extra_info']))[i]]
            slot['finish_reasons'] += [r.get('finish_reason') for r in records] if records is not None else []
            slot['masked_reason'] = data.get('ts_mask_reason', ['missing_mask_reason_transport'] * len(data['extra_info']))[i]
            slot['masked_detail'] = data.get('ts_mask_detail', [None] * len(data['extra_info']))[i]
        slots.append(slot)
    slots.sort(key=lambda r: r['position'])
    return dict(schema='ts-attempt0-probe.v2', arm=native['ts']['arm'], binding=native['ts']['binding'],
                seed=native['seed'], update=1, optimizer_updates=0,
                incoming_sha256=native['incoming']['sha256'], schedule_sha256=native['ts']['schedule_sha256'],
                slots=slots, technical_synthetic=False)


def token_agreement(left, right):
    matched = sum(x['token_ids'] == y['token_ids'] for x, y in zip(left, right))
    return dict(identical_slots=matched, total_slots=len(left), identical_fraction=matched / len(left))


def compare(left, right, control=None):
    need(left['arm'] == 'D' and right['arm'] == 'M', 'D/M qualification pair required')
    for key in ('binding', 'seed', 'update', 'optimizer_updates', 'incoming_sha256', 'schedule_sha256', 'technical_synthetic'):
        need(left[key] == right[key], 'identity probe differs in ' + key)
    need(left['update'] == 1 and left['optimizer_updates'] == 0, 'probe weights already updated')
    expected = ts.settings(left['binding'], 'finqa')['task_rows']
    a, b = left['slots'], right['slots']
    need(len(a) == len(b) == expected, 'identity probe must cover every scheduled task slot')
    positions = [r['position'] for r in a]
    width = ts.settings(left['binding'], 'finqa')['rows_per_update']
    need(positions == sorted(set(positions)) and all(1 <= p <= width for p in positions), 'identity probe slot geometry differs')
    need([(r['position'], r['question_id']) for r in a] == [(r['position'], r['question_id']) for r in b],
         'identity probe task identities differ')
    for row in a + b:
        request = row.get('request')
        need(isinstance(request, dict) and set(request) == {'prompt_token_ids', 'sampling_params'},
             'attempt-zero request observation missing or invalid')
        ids, params = request['prompt_token_ids'], request['sampling_params']
        need(isinstance(ids, list) and ids and all(type(t) is int and t >= 0 for t in ids),
             'attempt-zero prompt token IDs invalid')
        need(isinstance(params, dict) and 'seed' not in params, 'attempt-zero sampling must have no seed')
    matched = sum(x['request'] == y['request'] for x, y in zip(a, b))
    need(matched == expected, 'attempt-zero request identity below 100 percent')
    reasons = []
    if any(len(r.get('finish_reasons', [])) != 8 or
           any(value not in ('stop', 'length') for value in r.get('finish_reasons', [])) for r in b):
        reasons.append('finish_reason_transport_invalid')
    for reason in ('unknown_finish_reason', 'invalid_model_output_or_checker_error', 'missing_mask_reason_transport'):
        if any(r.get('masked_reason', 'missing_mask_reason_transport') == reason for r in b):
            reasons.append(reason)
    exception_types = sorted({str(r['masked_detail']) for r in b if r.get('masked_detail') and
                              r.get('masked_reason') == 'invalid_model_output_or_checker_error'})
    passed = 'TECHNICAL-ONLY' if left['technical_synthetic'] else 'PASS'
    status = 'UNVERIFIED' if reasons else passed
    # A shared M transport failure stops M/S/H. D keeps its independent chain.
    arms = {arm: dict(status=passed if arm == 'D' else status,
                      reasons=[] if arm == 'D' else reasons,
                      exception_types=[] if arm == 'D' else exception_types) for arm in ts.ARMS}
    advisory = dict(D_vs_M=token_agreement(a, b), D_vs_D=None)
    if control is not None:
        # Optional existing control evidence never adds a GPU row or a gate.
        keys = ('arm', 'binding', 'seed', 'update', 'optimizer_updates', 'incoming_sha256',
                'schedule_sha256', 'technical_synthetic')
        if (all(control.get(k) == left[k] for k in keys) and
                [(r.get('position'), r.get('question_id'), r.get('request')) for r in control.get('slots', [])] ==
                [(r['position'], r['question_id'], r['request']) for r in a]):
            advisory['D_vs_D'] = token_agreement(a, control['slots'])
    return dict(status=status, arms=arms, reasons=reasons, exception_types=exception_types,
                request_identity=dict(identical_slots=matched, total_slots=expected, identical_fraction=1.),
                token_identity_advisory=advisory,
                incoming_sha256=left['incoming_sha256'], probe_sha256=digest([left, right]))


def write_probe(batch, native):
    atomic(native['receipt'], capture(batch, native))
    raise ProbeComplete()
