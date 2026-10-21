#!/usr/bin/env python3
"""Print the hard-MATH teacher-coverage decision from a finished (or partial) job directory; CPU only.

    python kit/math_hard_teacher_read.py --source "$OUT/math-hard-teacher" [--json]

Checks the frozen kit resources (the frozen registration and amendment 1 bound separately, by hash), the job's
manifest hashes and its inputs against those resources, then
recomputes the amendment-1 decision from the journalled demonstrations and rejections (re-reading every
demonstration's box with the frozen verifier) and refuses a written decision that differs. Exit 0 with the
decision printed (QUALIFIES, DOES_NOT_QUALIFY or INCOMPLETE), 2 on a correctness stop.
"""
from pathlib import Path
import sys
if __package__ in (None, ''): sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import json
from kit import math_hard_teacher as h

KEYS = ('status', 'complete', 'teacher_solved', 'teachable_gap', 'student_failures', 'third_task_qualifies', 'technical_synthetic')


def read(source):
    source = Path(source)
    f = h.frozen()
    inputs = json.loads((source / 'inputs.json').read_text())
    h.require(inputs.get('schema') == h.SCHEMA and inputs.get('resources') == f['hashes'],
              'job inputs differ from these frozen resources')
    finished = (source / 'manifest.json').exists()
    if finished: h.frozen_files(source)
    demonstrations, rejections, raws = h.read_outputs(source)
    result = h.decide(f['panel'], f['student'], demonstrations, rejections,
                      technical_synthetic=inputs['technical_synthetic'] is True, raws=raws)
    if finished:
        written = json.loads((source / 'decision.json').read_text())
        h.require({k: written.get(k) for k in KEYS} == {k: result[k] for k in KEYS}, 'written decision differs from its evidence')
        result['measured'] = written.get('measured')
    elif result['status'] != 'INCOMPLETE':
        result.update(status='INCOMPLETE', third_task_qualifies=False,
                      problems=['job manifest missing: the job did not finish'] + result['problems'])
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--source', type=Path, required=True, help='OUT/math-hard-teacher (or an unpacked return)')
    p.add_argument('--json', action='store_true')
    a = p.parse_args(argv)
    try: result = read(a.source)
    except h.CorrectnessStop as exc:
        print('STOP: correctness:', exc); return 2
    print(json.dumps(result, indent=2) if a.json else h.summary(result))
    if result.get('measured'): print('measured:', json.dumps(result['measured']))
    return 0


if __name__ == '__main__': raise SystemExit(main())
