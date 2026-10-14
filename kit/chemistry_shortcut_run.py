"""Small 3A campaign operations, reusing eval_bed, runner and collector."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kit import eval_bed
from kit.chemistry_shortcut_panels import DEFAULT, PANELS, item_id, rows
from kit.v4_contract import INITIAL_8B_REVISION

KIT = Path(__file__).resolve().parent
MODEL_ENV = {'base': 'CHEM3A_BASE', 'r1': 'CHEM3A_R1', 'r2': 'CHEM3A_R2'}


def prepare(work, panels=DEFAULT):
    destination = work / '3a/report-inputs'
    if not destination.exists():
        shutil.copytree(panels, destination)
    manifest = json.loads((destination / 'manifest.json').read_text())
    warnings = []
    for name, digest in manifest['files'].items():
        path = destination / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            warnings.append('Frozen file differs: ' + name)
    for name, digest in manifest['code'].items():
        path = KIT.parent / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            warnings.append('Source differs from freeze: ' + name)
    # Programme collector includes JSON/text under report folders.
    (destination / 'prepare-receipt.json').write_text(json.dumps({'warnings': warnings, 'base_revision': INITIAL_8B_REVISION}, indent=2) + '\n')
    for warning in warnings:
        print('WARNING:', warning)
    return destination


def score(work, model_key, panel, out, standin=False):
    frozen = work / '3a/report-inputs'
    start = time.monotonic()
    manifest_hash = hashlib.sha256((frozen / 'manifest.json').read_bytes()).hexdigest()
    os.environ.pop('KIT_V4_ARM', None)
    os.environ['SDPO_DIR'] = str(frozen / 'scorer')
    model = os.environ.get(MODEL_ENV[model_key], '')
    identity = {'key': model_key, 'path': model, 'expected_base_revision': INITIAL_8B_REVISION if model_key == 'base' else None}
    warnings = []
    if standin:
        # Gold-aware fixture for plumbing only; never a language-model result.
        out.mkdir(parents=True)
        response_rows = []
        for row in rows(frozen / panel / 'test.jsonl'):
            info = row['extra_info']; gold = row['reward_model']['ground_truth']
            number = int(info['question_id'].rsplit('-', 1)[1])
            correct = (number % 3 == 0) if model_key == 'base' else (number % 3 == 0 or info['affected'])
            if model_key != 'base' and panel == 'balanced':
                correct = number % 3 == 0 or info['gold_rank'] == 3
            answer = gold if correct else 'ABCD'[('ABCD'.index(gold) + 1) % 4]
            response_rows.append({'id': item_id(row), 'response': f'<reasoning>CPU fixture.</reasoning><answer>{answer}</answer>',
                                  'output_tokens': 20 + number % 7, 'finish_reason': 'stop', 'technical_standin': True})
        responses = out / 'responses.jsonl'
        responses.write_text(''.join(json.dumps(r) + '\n' for r in response_rows))
        scored = out / 'scored'
        eval_bed.main(['score', '--bed', 'chemistry', '--root', str(frozen / panel), '--responses', str(responses), '--out', str(scored)])
        shutil.move(scored / 'bed-score.json', out / 'bed-score.json'); scored.rmdir()
    else:
        if not model:
            raise ValueError('set ' + MODEL_ENV[model_key] + ' to the kept/pinned local HF model')
        model_path = Path(model)
        identity['resolved_path'] = str(model_path.resolve())
        config = json.loads((model_path / 'config.json').read_text())
        identity['config_commit_hash'] = config.get('_commit_hash')
        if model_key == 'base' and INITIAL_8B_REVISION not in str(model_path.resolve()) and config.get('_commit_hash') != INITIAL_8B_REVISION:
            warnings.append('Base revision not established by snapshot path/config; return local download provenance')
        identity['config_sha256'] = hashlib.sha256((model_path / 'config.json').read_bytes()).hexdigest()
        identity['files'] = {p.name: p.stat().st_size for p in model_path.iterdir() if p.is_file()}
        os.environ['KIT_DETERMINISTIC'] = '1'
        eval_bed.main(['generate', '--bed', 'chemistry', '--root', str(frozen / panel), '--model', model,
                      '--max-new-tokens', '2048', '--max-model-len', '12288', '--out', str(out)])
    result_path = out / 'bed-score.json'
    result = json.loads(result_path.read_text())
    result.update({'panel_manifest_sha256': manifest_hash, 'panel': panel,
                   'model_identity': identity, 'technical_standin': standin, 'warnings': warnings,
                   'timing': {'elapsed_seconds': time.monotonic() - start, 'gpu_count': 0 if standin else 1,
                              'slurm_job_id': os.environ.get('SLURM_JOB_ID'), 'allocated_gpus': os.environ.get('SLURM_GPUS_ON_NODE')}})
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')


def campaign(out, standin=False):
    name = '3a-shortcut-cpu' if standin else '3a-shortcut'
    common = [sys.executable if standin else 'python', '{kit}/chemistry_shortcut_run.py']
    scoring = []
    rows_out = [{'id': 'inputs', 'command': common + ['prepare', '--work', '{work}'], 'timeout_seconds': 36000}]
    for model in MODEL_ENV:
        for panel in PANELS:
            rid = model + '-' + panel
            scoring.append(rid)
            rows_out.append({'id': rid, 'needs': ['inputs'], 'timeout_seconds': 36000,
                             'command': common + ['score', '--work', '{work}', '--model', model, '--panel', panel,
                                                 '--out', '{work}/3a/eval/' + rid + '-a{attempt}'] + (['--standin'] if standin else [])})
    rows_out.append({'id': 'report', 'needs': ['inputs'], 'wants': scoring, 'timeout_seconds': 36000,
                     'command': [common[0], '{kit}/chemistry_shortcut_report.py', '--archive', '{work}',
                                 '--out', '{work}/3a/report-reading/a{attempt}']})
    result = {'schema': 'kit-campaign.v1', 'name': name, 'workdir_env': 'WORK', 'rows': rows_out}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + '\n')


def rehearse(out):
    out.mkdir(parents=True)
    work = out / 'work'; work.mkdir()
    file = out / '3a-shortcut-cpu.json'
    campaign(file, standin=True)
    env = {**os.environ, 'WORK': str(work.resolve())}
    for action in ('plan', 'prepare', 'run'):
        argv = [sys.executable, str(KIT / 'runner.py'), action, str(file)]
        if action != 'plan':
            argv += ['--all']
        subprocess.run(argv, env=env, check=True)
    archive = out / '3a-cpu-return.tar.gz'
    subprocess.run([sys.executable, str(KIT / 'collect.py'), '--work', str(work), '--out', str(archive), '--campaign', str(file)], check=True)
    subprocess.run([sys.executable, '-m', 'kit.chemistry_shortcut_report', '--archive', str(archive), '--out', str(out / 'archive-reading')], check=True)
    result = json.loads((out / 'archive-reading/reading.json').read_text())
    direct = json.loads((work / '3a/report-reading/a1/reading.json').read_text())
    assert result == direct, 'archive-only readout differs from direct WORK readout'
    print('CPU rehearsal: 12 scoring rows, direct/archive readers identical; NO MODEL OUTPUT')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    p = sub.add_parser('prepare'); p.add_argument('--work', type=Path, required=True)
    p.add_argument('--panels', type=Path, default=DEFAULT)
    p = sub.add_parser('score'); p.add_argument('--work', type=Path, required=True)
    p.add_argument('--model', choices=MODEL_ENV, required=True); p.add_argument('--panel', choices=PANELS, required=True)
    p.add_argument('--out', type=Path, required=True); p.add_argument('--standin', action='store_true')
    p = sub.add_parser('campaign'); p.add_argument('--out', type=Path, required=True); p.add_argument('--standin', action='store_true')
    p = sub.add_parser('rehearse'); p.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.action == 'prepare': prepare(args.work, args.panels)
    elif args.action == 'score': score(args.work, args.model, args.panel, args.out, args.standin)
    elif args.action == 'campaign': campaign(args.out, args.standin)
    else: rehearse(args.out)


if __name__ == '__main__':
    main()
