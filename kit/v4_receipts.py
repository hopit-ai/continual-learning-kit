"""Completed optimizer dose, measured from explicit runtime counters, never checkpoint names."""
import json
import os
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kit.v4_teacher import atomic_json
from kit.v4_run import REGISTERED


def completed_updates(out):
    out = Path(out)
    counter = out / 'env/optimizer-updates.json'
    if counter.is_file(): return int(json.loads(counter.read_text())['completed_optimizer_updates'])
    completed = 0
    path = out / 'metrics.jsonl'
    if path.is_file():
        for line in path.read_text().split('\n'):
            if line.strip():
                record = json.loads(line)
                value = record.get('data', record).get('v4/completed_optimizer_updates', 0)
                completed = max(completed, int(value))
    return completed


def write_receipt(out, arm, requested, profile_name):
    out = Path(out)
    completed = completed_updates(out)
    doc = json.loads((out / 'run-summary.json').read_text())
    registered = {**REGISTERED, 'rollout_n':8 if arm == 'S' else 1,
                  'teacher_update_rate':.05 if arm == 'S' else .01 if arm == 'D' else None}
    doc.update(profile=profile_name, technical_smoke=profile_name == 'technical-smoke',
               requested_optimizer_updates=int(requested), completed_optimizer_updates=completed,
               dose_complete=completed == int(requested), registered=registered)
    data = {}
    if os.environ.get('DATA_MANIFEST'):
        data = json.loads(Path(os.environ['DATA_MANIFEST']).read_text())
    from kit.v4_contract import technical_synthetic
    doc['technical_synthetic'] = technical_synthetic(data)
    doc['scientific_phase_allowed'] = not doc['technical_synthetic'] and (profile_name == 'scientific' and data.get('scientific_phase_allowed') is True and doc['dose_complete'])
    atomic_json(out / 'env/optimizer-updates.json', {'technical_synthetic':doc['technical_synthetic'], 'requested_optimizer_updates':int(requested), 'completed_optimizer_updates':completed})
    atomic_json(out / 'env/v4-settings.json', {**registered, 'technical_synthetic':doc['technical_synthetic']})
    from kit.v4_qualification import departures_hash, check_metrics
    doc['departures_sha256']=departures_hash()
    if doc.get('dose_complete') and not os.environ.get('V4_RESTORE_PROBE_OUT'):
        from kit.v4_teacher import read_jsonl
        doc.update(check_metrics(read_jsonl(out/'metrics.jsonl'), arm, int(requested)))
    atomic_json(out / 'run-summary.json', doc)
    return doc


if __name__ == '__main__':
    if sys.argv[1:] in (["--help"], ["-h"]):
        print("usage: v4_receipts.py OUT ARM REQUESTED_UPDATES PROFILE")
        raise SystemExit(0)
    result = write_receipt(sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4])
    if result.get('returncode') == 0 and not result['dose_complete']:
        raise SystemExit('v4 optimizer dose incomplete; requested and completed counts are in run-summary.json')
