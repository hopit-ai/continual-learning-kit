#!/usr/bin/env python3
"""CPU job: recompute admission and show the next registered step."""
from pathlib import Path
import sys
if __package__ in (None, ''): sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kit.p3a_common import *


def job(a):
    """Decision only; never launches a model or schedules a job."""
    freeze = frozen(a.out, a.standin)
    for name in s.ORDER:
        out = a.out / OUTPUTS / name; base = a.out / INPUTS / 'prepared' / name
        previous = decision(a.out, 'code', 2, freeze, a.standin) if name == 'math_screen' else None
        step1 = decision(a.out, name, 1, freeze, a.standin)
        reason = None; status = 'INCOMPLETE'
        if previous and previous.get('complete') and previous['status'] == 'PASS':
            status, reason = 'SKIPPED', 'LCB qualifies; registered order ends here'
        elif name == 'math_screen' and not (previous and previous.get('complete') and previous['status'] == 'FAIL'):
            reason = 'LCB incomplete; selection unresolved'
        elif not step1.get('complete'): reason = 'step 1 not complete'
        elif step1['status'] == 'FAIL': status, reason = 'FAIL', 'step 1 headroom stop'
        if reason:
            row = dict(status=status, complete=status != 'INCOMPLETE', reason=reason,
                step2_executed=False, selected_for_partner_qualification=False)
        else:
            def rows(phase, file):
                path = out / phase / file
                return t.read_jsonl(path) if path.exists() else []
            row = s.admission(t.read_jsonl(base / 'sample.jsonl'), rows('teacher', 'demonstrations.jsonl'),
                rows('rewrite', 'rewrites.jsonl'), complete=all((out / (p + '-complete.json')).exists() for p in ('teacher', 'rewrite')),
                synthetic=a.standin, pool=t.read_jsonl(base / 'train.jsonl'))
        row.update(candidate=name, step=2, freeze_sha256=freeze, technical_synthetic=a.standin)
        atomic(out / 'decision.json', row)
        print(name, row['status'])
    print('Only complete LCB step-2 FAIL permits MATH baseline; only its PASS permits MATH teacher/rewrite.')

if __name__ == '__main__': raise SystemExit(entry(job, lambda p: None, 'decide'))
