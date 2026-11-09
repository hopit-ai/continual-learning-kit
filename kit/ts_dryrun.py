#!/usr/bin/env python3
"""CPU-only TS rehearsal through the unchanged campaign runner and real children."""
from pathlib import Path
import sys
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from contextlib import redirect_stdout
import os

from kit import runner, ts_jobs, ts_arms as ts
from kit.nm_common import atomic, load, need


def run(out):
    out = Path(out).resolve()
    need(not out.exists(), 'choose a fresh TS rehearsal directory')
    out.mkdir(parents=True)
    previous = {key: os.environ.get(key) for key in ('TS_WORK', 'CUDA_VISIBLE_DEVICES', 'HF_HUB_OFFLINE', 'HF_DATASETS_OFFLINE')}
    os.environ.update(TS_WORK=str(out), CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1')
    manifest = ts_jobs.inventory()
    for row in manifest['rows']:
        row['env']['TS_PYTHON'] = sys.executable
    campaign_path = out / 'campaign.json'
    atomic(campaign_path, manifest)
    config_path = out / 'ts-config.json'
    cfg = dict(out=str(out / 'results'), technical_synthetic=True, replay_binding=None)
    atomic(config_path, cfg)
    failures = []
    try:
        with (out / 'runner.log').open('w') as log, redirect_stdout(log):
            campaign = runner.load_campaign(campaign_path)
            need(not runner.implicit_edges(campaign), 'TS dependencies must be fully explicit')
            # The real runner refuses an export before its training dependency.
            export = next(r for r in campaign['rows'] if r['phase'] == 'export')
            need(runner.cmd_run(campaign, [export], False, None) == runner.EXIT_REFUSED, 'missing dependency launched')
            failures.append('missing dependency refused before launch')
            # A structurally invalid dependency is rejected while loading the DAG.
            broken = ts_jobs.inventory(('I0',))
            broken['rows'][0]['needs'] = ['does-not-exist']
            invalid = out / 'invalid-dependency.json'; atomic(invalid, broken)
            try:
                runner.load_campaign(invalid)
            except runner.CampaignError:
                failures.append('invalid dependency rejected')
            else:
                raise AssertionError('invalid dependency accepted')
            # Exercise a genuine checker-output failure through a runner child.
            prefix = 'TS-I0-D-s101-finqa'
            cfg['missing_checker_task'] = prefix + '-train'; atomic(config_path, cfg)
            selected = [r for r in campaign['rows'] if r['id'] == 'TS-I0-bind' or r['id'].startswith('TS-I0-qualify-')
                        or r['id'] in (prefix + '-prepare', prefix + '-train')]
            need(runner.cmd_run(campaign, selected, True, None) == runner.EXIT_FAILED, 'missing checker output accepted')
            failures.append('missing checker output failed without update')
            cfg.pop('missing_checker_task'); atomic(config_path, cfg)
            need(runner.cmd_run(campaign, campaign['rows'], True, None) == 0, 'complete TS runner rehearsal failed')
            need(runner.cmd_run(campaign, campaign['rows'], True, None) == 0, 'runner idempotent replay failed')
        complete = sum((runner.latest_verdict(campaign, r['id']) or {}).get('verdict') == 'PASS' for r in campaign['rows'])
        empty_batches = 0
        for binding in ts.BINDINGS:
            for arm in ('M', 'S'):
                for seed in ts.SEEDS:
                    for stage in ts.STAGES:
                        receipt = load(out / 'results' / binding / 'runs' / binding / f'{arm}-s{seed}' / stage / 'native-receipt.json')
                        need(receipt['history'][0]['empty_batches'] == 1 and receipt['steps'] == dict(optimizer=40, scheduler=40, ema=40),
                             'all-failure replay/step path failed')
                        empty_batches += sum(r['empty_batches'] for r in receipt['history'])
            need(load(out / 'results' / binding / 'reading.json')['status'] == 'TECHNICAL-ONLY', 'synthetic reading lost provenance')
        failures.append('all-failure batches retained replay scheduler and EMA updates')
        disagreements = 0
        fixtures = [('finqa', 'Answer: 42', 'Answer: 41', 'malformed', '42'),
                    ('contract_nli', 'Answer: entailment', 'Answer: contradiction', 'Answer: ENTAILMENT', 'entailment'),
                    ('math_hard', r'\boxed{42}', r'\boxed{41}', r'\boxed{42', '42')]
        for stage, correct, incorrect, malformed, gold in fixtures:
            observed = [ts.check_attempt(stage, ts.Attempt(text, (1, 8), cut), gold) for text, cut in
                        [(correct, False), (incorrect, False), (malformed, False), (correct, True)]]
            expected = [1, 0, 0, 0 if stage == 'finqa' else 1]
            disagreements += sum(a != b for a, b in zip(observed, expected))
        need(disagreements == 0, 'frozen checker fixture disagreement')
        result = dict(status='TECHNICAL-ONLY', rows=len(manifest['rows']), complete=complete,
                      training_rows=sum(r['phase'] == 'train' for r in manifest['rows']),
                      scoring_rows=sum(r['phase'] == 'score' for r in manifest['rows']),
                      empty_batches=empty_batches, failure_paths=failures,
                      replay_binding=None, paid_services_used=False,
                      checker_hashes=ts.checker_hashes(), checker_disagreements=disagreements)
        atomic(out / 'result.json', result)
        return result
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    import json
    print(json.dumps(run(args.out), sort_keys=True))


if __name__ == '__main__':
    main()
