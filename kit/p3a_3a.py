#!/usr/bin/env python3
"""One GPU job: score one Chemistry model on the four frozen panels."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
if __package__ in (None, ''): sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kit.p3a_common import *
from kit import chemistry_shortcut_run as chem
from kit.chemistry_shortcut_panels import item_id, rows


def complete(folder, panel):
    score, raw = folder / 'bed-score.json', folder / 'responses.jsonl'
    if not score.exists() or not raw.exists(): return False
    try:
        result = json.loads(score.read_text()); responses = t.read_jsonl(raw)
        ids = {item_id(r) for r in rows(panel)}
        return (set(result['per_item']) == ids == {r['id'] for r in responses}
                and len(responses) == len(ids) and result['responses_sha256'] == s.file_hash(raw))
    except (ValueError, KeyError): return False


def job(a):
    """Score all panels; reruns reuse complete scores and keep unfinished evidence."""
    if not a.standin:
        observed = t.local_identity(model_path(a.model), 'Chemistry-' + a.lane)
        record = a.out / '3a' / (a.lane + '-model.json')
        if record.exists() and json.loads(record.read_text())['model_file_hashes'] != observed['model_file_hashes']:
            raise CorrectnessStop('3A model input hashes changed: ' + a.lane)
        atomic(record, observed)
    from kit.chemistry_shortcut_report import EXPECTED_MANIFEST_SHA256
    check(chem.DEFAULT / 'manifest.json', EXPECTED_MANIFEST_SHA256)
    manifest = json.loads((chem.DEFAULT / 'manifest.json').read_text())
    for name, expected in manifest['code'].items(): check(KIT.parent / name, expected)
    target = a.out / '3a/report-inputs'
    if not target.exists(): shutil.copytree(chem.DEFAULT, target)
    check(target / 'manifest.json', s.file_hash(chem.DEFAULT / 'manifest.json'))
    for name, expected in manifest['files'].items(): check(target / name, expected)
    os.environ.update(CUDA_VISIBLE_DEVICES=a.gpu, KIT_DETERMINISTIC='1', V4_PHASE0_INFERENCE='0')
    if a.model: os.environ[chem.MODEL_ENV[a.lane]] = a.model
    failed = []
    for panel in chem.PANELS:
        folder = a.out / '3a/eval' / (a.lane + '-' + panel)
        if complete(folder, target / panel / 'test.jsonl'):
            print('SKIP:', a.lane, panel); continue
        retain_partial(folder)
        command = [sys.executable, str(KIT / 'chemistry_shortcut_run.py'), 'score', '--work', str(a.out), '--model', a.lane, '--panel', panel, '--out', str(folder)]
        if a.standin: command.append('--standin')
        if subprocess.run(command).returncode: print('FAILED: scoring', a.lane, panel); failed.append(panel)
    # Later panels still run; the lane then exits non-zero so the failure is visible.
    if failed: raise RuntimeError('3A scoring failed for ' + a.lane + ': ' + ', '.join(failed))


def configure(p):
    p.add_argument('--lane', choices=chem.MODEL_ENV, required=True)
    p.add_argument('--model', help='local HF checkpoint directory')
    p.add_argument('--gpu', default=os.environ.get('CUDA_VISIBLE_DEVICES', '0'))

if __name__ == '__main__': raise SystemExit(entry(job, configure, '3a-{lane}'))
