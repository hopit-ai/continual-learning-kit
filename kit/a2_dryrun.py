#!/usr/bin/env python3
"""Every A2 row via the real shell/allocation/child wrapper with CPU stand-ins."""
from pathlib import Path
import sys
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import os
import subprocess
from kit.a2_common import *
from kit.a2_jobs import inventory, completed
from kit.a2_sim import fixture, confirmation_fixture


def rehearse(root, arm):
    root = Path(root).resolve()
    require(not root.exists(), 'fresh rehearsal output directory required')
    cfg = fixture(root, arm)
    rows = inventory(arm)['rows']
    fail_task = next(r['id'] for r in rows if r['phase'] == 'train')
    cfg['fail_task'] = fail_task
    config = root/'config.json'
    atomic(config, cfg)
    binary = root/'bin'
    binary.mkdir()
    smi = binary/'nvidia-smi'
    smi.write_text('#!'+sys.executable+'\nimport sys,os\n'
                   'q=next((x for x in sys.argv if x.startswith("--query-")), "")\n'
                   'if q=="--query-gpu=uuid": print("GPU-CPU-STANDIN-0")\n'
                   'elif "utilization.gpu" in q: print("\\n".join(f"{i}, GPU-CPU-STANDIN-{i}, 10, 100" for i in range(8)))\n'
                   'else: print("\\n".join(f"{i}, GPU-CPU-STANDIN-{i}" for i in range(8)))\n')
    smi.chmod(0o755)
    env = {k: v for k, v in os.environ.items() if not k.startswith(('SLURM_', 'CUDA_', 'M2_', 'M3_', 'UTIL_', 'KIT_LANE_'))}
    env.update(KIT=str(KIT), A2_CONFIG=str(config), JOB_PY=sys.executable,
               PATH=str(binary)+os.pathsep+os.environ['PATH'])
    calls = []
    def call(task, expected=0, environment=env):
        done = subprocess.run(['bash', str(KIT/'jobs/a2/run-task.sh'), task],
                              env=environment, capture_output=True, text=True, timeout=180)
        with (root/'wrapper.log').open('a') as stream:
            stream.write(task+'\n'+done.stdout+done.stderr+'\n')
        require(done.returncode == expected, f'{task} expected {expected}, got {done.returncode}: '+done.stdout[-1500:]+done.stderr[-1500:])
        calls.append(dict(task=task, code=done.returncode))
    call('A2-'+arm+'-arrival-m1', PENDING)
    inputs = checked(cfg['inputs'])
    saved = inputs.read_bytes()
    inputs.unlink()
    call('A2-'+arm+'-bind', PENDING)
    inputs.write_bytes(saved)
    for row in rows:
        if row['id'] == fail_task:
            call(row['id'], 2)
            export = row['id'].removesuffix('-train')+'-export'
            call(export, PENDING)
        if row['phase'] == 'confirmation-bind':
            call(row['id'], PENDING)
            cfg = confirmation_fixture(root, cfg)
            atomic(config, cfg)
        call(row['id'])
    out = Path(cfg['out'])
    # Complete rows replay from verified artifacts, without training/scoring again.
    hashes = {str(p.relative_to(out)): s.file_hash(p) for p in (out/'runs').rglob('*') if p.is_file()}
    call(fail_task)
    require(hashes == {str(p.relative_to(out)): s.file_hash(p) for p in (out/'runs').rglob('*') if p.is_file()}, 'completed student changed on resume')
    for row in rows:
        completed(out, row['id'])
    # Archive every row without touching nonexistent inputs, even GPU-labelled rows.
    archived = dict(cfg, out=str(root/'archived'), disposition='archived',
                    archive=str(root/'archived-return.tar.gz'), text_out=str(root/'archived-text'),
                    inputs=dict(path='/absent', sha256='0'*64))
    archive_cfg = root/'archived-config.json'
    atomic(archive_cfg, archived)
    archive_env = dict(env, A2_CONFIG=str(archive_cfg))
    for row in rows:
        call(row['id'], environment=archive_env)
    require(load(root/'archived/reading.json')['status'] == 'ARCHIVED UNREAD', 'archive opened evidence')
    reading = load(out/'reading.json')
    returned = load(out/'returned-reading.json')
    require(reading['contrasts'] == returned['contrasts'], 'archive exploratory bars changed')
    summary = dict(arm=arm, status='TECHNICAL-ONLY', rows=len(rows), all_rows_through_real_wrapper=True,
                   training_stages=len(list((out/'runs').glob('*/summary.json'))),
                   score_cells=len(load(completed(out, 'A2-'+arm+'-comparison')/'cells.json')),
                   missing_input='PENDING', archived_arm='ARCHIVED UNREAD',
                   confirmation_unreleased='PENDING', interrupted_training='FAILED then same-seed resume',
                   archive_readback_equal=True, text_readback_equal=True, completed_exports_unchanged=True,
                   scientific_verdict=None, gpu_hours_spent=0, calls=calls)
    atomic(root/'rehearsal.json', summary)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm', choices=('J', 'P', 'R', 'RP'), required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    result = rehearse(a.out, a.arm)
    print(json.dumps({k: v for k, v in result.items() if k != 'calls'}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
