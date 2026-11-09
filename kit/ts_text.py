"""Path-free numbered TS text returns. Detailed raw evidence stays local."""
from kit.nm_common import SEEDS, need


def return_parts(result, limit=11000):
    lines = ['DRAFT - NOT FOR SEND',
             'CPU-STANDIN TECHNICAL-ONLY' if result.get('technical_synthetic') is True or result.get('status') == 'TECHNICAL-ONLY' else 'PARTNER-COMPUTED',
             'Replay ' + result['binding'], 'Status ' + result['status']]
    qualification = result.get('qualification', {})
    for arm, check in qualification.get('arms', {}).items():
        lines.append('|'.join(['Q', arm, check['status'], *check['reasons'],
                               *('exception=' + name for name in check.get('exception_types', []))]))
    identity = qualification.get('request_identity')
    if identity:
        lines.append('REQUEST-IDENTITY|' + str(identity['identical_fraction']))
    for pair, advisory in qualification.get('token_identity_advisory', {}).items():
        lines.append('|'.join(['TOKEN-IDENTITY-ADVISORY', pair,
                              str(advisory['identical_fraction']) if advisory else 'UNAVAILABLE']))
    for name, contrast in result.get('contrasts', {}).items():
        if contrast['status'] == 'PENDING':
            lines.append('|'.join(['R', name, contrast['priority'], 'PENDING']))
            continue
        primary = contrast['primary']
        lines.append('|'.join(['R', name, contrast['priority'], contrast['status'],
            *[f'{s}={v:.8g}' for s, v in zip(SEEDS, primary['values'])],
            f"mean={primary['mean']:.8g}", f"SD={primary['sd']:.8g}",
            f"lower={primary['lower']:.8g}", f"upper={primary['upper']:.8g}"]))
        for guard, value in contrast['guards'].items():
            lines.append('|'.join(['G', name, guard, *[f'{v:.8g}' for v in value['values']],
                                  f"lower={value['lower']:.8g}", 'PASS' if value['pass'] else 'INCONCLUSIVE']))
    for name, history in result.get('training_telemetry', {}).items():
        fields = ('attempted_rows', 'attempted_trajectories', 'eligible_rows', 'loss_bearing_tokens',
                  'actual_task_loss_mass', 'singleton_success_count', 'empty_batches', 'capped_outputs',
                  'nondistinct_success_context', 'model_output_masked_rows', 'near_duplicate_pairs', 'sibling_pairs',
                  'near_duplicate_excluded_pairs')
        values = []
        for field in fields:
            key = field if field in history[0] else 'ts/' + field
            observed = [row.get(key) for row in history]
            values.append(field + '=' + (f'{sum(observed):.8g}' if all(v is not None for v in observed) else 'UNMEASURED'))
        lines.append('|'.join(['T', name, *values]))
        attempted = sum(r.get('attempted_rows', r.get('ts/attempted_rows', 0)) for r in history)
        eligible = sum(r.get('eligible_rows', r.get('ts/eligible_rows', 0)) for r in history)
        if attempted and eligible < attempted:
            lines.append('|'.join(['W', 'MASKED-RATE', name, 'warn-only',
                                  f'masked={attempted - eligible}', f'attempted={attempted}',
                                  f'rate={(attempted - eligible) / attempted:.8g}']))
        pair_count = sum(r.get('sibling_pairs', r.get('ts/sibling_pairs', 0)) for r in history)
        near_count = sum(r.get('near_duplicate_pairs', r.get('ts/near_duplicate_pairs', 0)) for r in history)
        lines.append('|'.join(['NEAR-DUPLICATE', name, str(near_count / pair_count if pair_count else None)]))
        reasons = sorted({k for r in history for k in r if k.startswith(('masked_reason_', 'ts/masked_reason_'))})
        for reason in reasons:
            lines.append('|'.join(['MASKED', name, reason.removeprefix('ts/'), str(sum(r.get(reason, 0) for r in history))]))
    for name, costs in result.get('costs', {}).items():
        known = all(c['output_tokens'] is not None for c in costs)
        tokens = sum(c['output_tokens'] for c in costs) if known else None
        correct = sum(c['correct'] for c in costs)
        per_correct = tokens / correct if tokens is not None and correct else None
        lines.append('|'.join(['C', name, 'tokens=' + str(tokens), 'correct=' + str(correct),
                              'tokens_per_correct=' + str(per_correct)]))
    for name, overlap in result.get('mask_overlap_S_M', {}).items():
        lines.append('|'.join(['MASK-S-M', name, *[key + '=' + str(value) for key, value in overlap.items()]]))
    for warning in result.get('utilisation_warnings', []):
        lines.append('W|' + warning)
    lines += ['Accounting ' + result.get('accounting_status', 'PENDING scheduler allocation-end accounting'),
              'Mechanical qualification and measured sibling throughput remain PENDING.']
    chunks, current = [], []
    for line in lines:
        need(len(line) < limit - 80, 'TS return line exceeds part limit')
        if sum(len(v) + 1 for v in current) + len(line) + 1 > limit - 80:
            chunks.append(current); current = []
        current.append(line)
    chunks.append(current)
    return [f'PART {i + 1} OF {len(chunks)}\n' + '\n'.join(chunk) + '\n' for i, chunk in enumerate(chunks)]
