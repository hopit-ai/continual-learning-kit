"""Sealed shared gold/27B membership after the four registered attempts."""
from collections import Counter
from kit.trace_common import *
from kit.nm_common import ready

NOTES = (
    'TRACE v7i3 departure B/C: 27B answer format is normalised; NumGLUE-ds agreement uses 1% relative tolerance (gold zero requires exact zero). Registered exact-match scores are unchanged.',
    'TRACE v7i3 departure C: rows unverified after four attempts are dropped from BOTH gold and 27B; floor 400/task; each shared cohort cycles in registered hash order to 1280 exposures = 40 updates x 32. See sealed cohort.json for kept/dropped IDs and reasons.',
)


def finish(root, prepared, accepted, decisions):
    tasks = {}
    for task, rows in prepared['selected'].items():
        kept, dropped = [], []
        for row in rows:
            history = decisions.get(row['id'], [])
            if row['id'] in accepted:
                kept.append(dict(id=row['id'], reason='verified', attempt=accepted[row['id']]['attempt']))
            else:
                dropped.append(dict(id=row['id'], reason='attempts_exhausted', attempts=history))
        reasons = Counter(d['attempts'][-1]['reason'] if d['attempts'] else 'no_attempt' for d in dropped)
        tasks[task] = dict(kept=kept, dropped=dropped, accepted=len(kept), rejected=len(dropped),
                           rejected_by_reason=dict(reasons))
        print('TRACE cohort', task, 'accepted', len(kept), 'rejected', len(dropped), dict(reasons))
    status = 'COMPLETE' if all(v['accepted'] >= 400 for v in tasks.values()) else 'PENDING'
    doc = dict(schema='trace-shared-cohort.v1', status=status,
               pending=None if status=='COMPLETE' else 'TRACE_COHORT_FLOOR_PENDING',
               prepared_sha256=sha(Path(root)/'prepared.json'), tasks=tasks,
               targets_sha256=digest(accepted), dose=dict(exposures=1280, updates=40, batch=32),
               departure_notes=list(NOTES))
    seal(Path(root)/'cohort.json', doc)
    for note in NOTES:
        print(note)
    ready(status=='COMPLETE', 'TRACE_COHORT_FLOOR_PENDING: '+__import__('json').dumps(
        {task:{k:v[k] for k in ('accepted','rejected','rejected_by_reason')} for task,v in tasks.items()},
        sort_keys=True))
    return doc


def shared_rows(root, prepared, demos):
    doc = load(Path(root)/'cohort.json')
    ready(doc['status']=='COMPLETE', 'TRACE_COHORT_FLOOR_PENDING')
    need(doc['prepared_sha256']==sha(Path(root)/'prepared.json') and
         doc['targets_sha256']==digest(demos), 'shared cohort identity differs')
    result = {}
    for task, rows in prepared['selected'].items():
        kept = [r['id'] for r in doc['tasks'][task]['kept']]
        dropped = [r['id'] for r in doc['tasks'][task]['dropped']]
        need(len(set(kept+dropped))==len(rows) and set(kept+dropped)=={r['id'] for r in rows}
             and 400 <= len(kept) <= 500, 'shared cohort partition/floor differs')
        result[task] = [r for r in rows if r['id'] in set(kept)]
    need(set(demos)=={r['id'] for rows in result.values() for r in rows},
         'gold/27B membership differs')
    return result
