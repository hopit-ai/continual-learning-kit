"""Per-slot eligibility ledgers for own-policy mask-overlap diagnostics."""
from pathlib import Path
import uuid

from kit.nm_common import need, ready, seal, load, atomic, digest


def record(root, update, extras, signals, groups=None):
    rows = [dict(position=int(extra['nm_position']), question_id=str(extra['index']),
                 task_mask=signal.mask, success_mask=int(any(signal.labels)))
            for extra, signal in zip(extras, signals)]
    need(len(extras) == len(signals) == len(rows) and len({r['position'] for r in rows}) == len(rows),
         'mask ledger schedule differs')
    for index, (row, signal) in enumerate(zip(rows, signals)):
        row.update(route=signal.route, reason=signal.reason)
        if groups is not None:
            row['attempts'] = [dict(finish_reason=a.finish_reason, truncated=a.truncated) for a in groups[index]]
    path = Path(root) / 'mask-history' / f'update-{update:02d}.json'
    seal(path, rows)
    return path


def supersede(root, checkpoint_step):
    """Archive active entries newer than the restored native checkpoint.

    Write the archive and its audit before removing active entries. A crash at
    any point is safe to retry. The per-training-row lock excludes concurrent
    writers. Entries at/before the checkpoint keep their original bytes.
    """
    need(type(checkpoint_step) is int and 0 <= checkpoint_step <= 40, 'invalid resume checkpoint')
    history = Path(root) / 'mask-history'
    paths = [p for p in sorted(history.glob('update-*.json'))
             if int(p.stem.split('-')[-1]) > checkpoint_step]
    if not paths:
        return None
    archive = history / 'superseded' / uuid.uuid4().hex
    entries = {}
    for path in paths:
        rows = load(path)
        atomic(archive / path.name, rows)
        entries[path.name] = digest(rows)
    atomic(archive / 'superseded.json', dict(reason='resume_from_checkpoint',
           checkpoint_step=checkpoint_step, entries=entries))
    for path in paths:
        path.unlink()
    return archive


def overlap(left, right):
    counts = dict(both=0, left_only=0, right_only=0, neither=0, attempted_rows=0)
    for update in range(1, 41):
        name = f'update-{update:02d}.json'
        paths = [Path(root) / 'mask-history' / name for root in (left, right)]
        ready(all(path.is_file() for path in paths), 'mask-overlap ledger missing')
        a, b = [{(r['position'], r['question_id']): r['task_mask'] for r in load(path)} for path in paths]
        need(a.keys() == b.keys(), 'mask-overlap scheduled task identity differs')
        for key in a:
            need(a[key] in (0, 1) and b[key] in (0, 1), 'mask-overlap labels must be binary')
            category = 'both' if a[key] and b[key] else 'left_only' if a[key] else 'right_only' if b[key] else 'neither'
            counts[category] += 1
            counts['attempted_rows'] += 1
    union = counts['both'] + counts['left_only'] + counts['right_only']
    return dict(counts, jaccard=counts['both'] / union if union else None)
