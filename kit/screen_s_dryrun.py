#!/usr/bin/env python3
"""Every Screen S row, failed domain audit, durable requeue and archive readback; CPU only."""
from pathlib import Path
import sys
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kit.s_common import *
from types import SimpleNamespace as NS
import contextlib
import shutil
import time
from unittest.mock import patch


def fixtures(root):
    """Closed synthetic ancestry with dev exposure invalidating FinQA only."""
    from kit.s_inputs import pins
    root.mkdir(parents=True, exist_ok=True)
    source = {}
    for domain, pin in pins().items():
        rr = []
        for i in range(216):
            unique = hashlib.sha256(f'{domain}:{i}'.encode()).hexdigest()
            row = dict(id=f'{domain}-{i}', prompt=unique, source=domain, fake_response='Answer: 7', gold='7')
            if domain == 'ifeval':
                # Frozen checker: one real executable constraint, not a synthetic verdict.
                row['ground_truth'] = [{'func_name': 'validate_no_commas'}]
            if domain == 'arc':
                row.update(labels='AB', gold='A', fake_response='Answer: A')
            rr.append(row)
        path = root/(domain+'.jsonl')
        write_rows(path, rr)
        source[domain] = dict(path=str(path), sha256=sha(path), **{k: pin[k] for k in ('repo', 'revision', 'split')})
    sources = root/'sources.json'
    atomic(sources, source)
    pools = []
    for role in ('b-ifeval-candidates', 'broad-gsm8k-candidates', 'v4-finqa-training', 'v4-finqa-dev', 'b-second-task-training'):
        path = root/(role+'.jsonl')
        write_rows(path, [dict(id=role, prompt=hashlib.sha256(role.encode()).hexdigest())])
        pools.append(dict(role=role, path=str(path), sha256=sha(path)))
    panels = {}
    for name in registration()['panel_inventory']:
        path = root/(name+'.jsonl')
        write_rows(path, [dict(id=name, prompt=hashlib.sha256(name.encode()).hexdigest())])
        panels[name] = dict(path=str(path), sha256=sha(path))
    nodes = {}
    def node(name, parent, step):
        checkpoint = dict(technical_synthetic=True, identity=name)
        proof = root/(name+'-exposure.json')
        atomic(proof, dict(complete=True, parent=parent, checkpoint_identity=digest(checkpoint), pools=pools))
        nodes[name] = dict(parent=parent, root=parent is None, step=step, checkpoint=checkpoint,
                           exposure=dict(path=str(proof), sha256=sha(proof)))
    node('base', None, 0)
    stages = []
    for tier in (1, 2):
        for recipe in ('D', 'F'):
            for seed in SEEDS:
                ident = f't{tier}-{recipe}-s{seed}'
                incoming = 'base' if tier == 1 else f't1-{recipe}-s{seed}-40'
                checkpoints = {}
                for step in (10, 20, 40):
                    name = ident+'-'+str(step)
                    node(name, incoming, step)
                    checkpoints[str(step)] = name
                stages.append(dict(id=ident, tier=tier, recipe=recipe, seed=seed,
                    stage='base-to-finqa' if tier == 1 else 'finqa-to-second-task', incoming=incoming, checkpoints=checkpoints))
    exports, inventory = [], []
    for slot in inventory_slots():
        primary = slot['arm'] in ('D', 'F')
        prefix = f"t1-{slot['arm']}-s{slot['seed']}"
        exports.append(dict(slot=slot['id'], checkpoint=nodes[prefix+'-20']['checkpoint'] if primary else dict(technical_synthetic=True, identity=slot['id']+'-20')))
        inventory.append(dict(slot=slot['id'], checkpoint=nodes[prefix+'-10']['checkpoint'] if primary else None))
    spec = root/'spec.json'
    atomic(spec, dict(nodes=nodes, stages=stages, panels=panels, v4_step20=exports, v4_step10=inventory))
    # The add-on validates the same existing binding reader. No message-2 row is modified.
    from kit.m2_common import protocol
    message2 = root/'message2'
    atomic(message2/'binding.json', dict(protocol=protocol(), technical_synthetic=True))
    return sources, spec, message2


def run(root):
    from kit import s_inputs, s_bind, s_audit, s_freeze, s_inventory, s_generate, s_drift, s_probe, s_score, s_collect, s_price
    from kit import message_handoff as handoff
    from scripts import screen_s_analysis as analysis
    sources, spec, message2 = fixtures(root/'fixtures')
    out, durable = root/'out', root/'durable'
    common = dict(out=out, standin=True)
    counts = {}
    def row(name, func, **kw):
        func(NS(**common, **kw))
        counts[name] = counts.get(name, 0)+1
    with patch.dict(os.environ, SLURM_JOB_ID='screen-s-standin', SLURM_RESTART_COUNT='0', SLURM_JOB_START_TIME=str(time.time()-1)):
        handoff.record_job_start(durable, True)
    row('S1-inputs', s_inputs.job, sources=sources)
    row('S2-bind', s_bind.job, message2=message2, sources=sources, spec=spec)
    row('S3-inventory', s_inventory.job)
    row('S4-v4-step20', s_score.job, action='v4-step20', finqa=None, gpu='0')
    row('S5-audit', s_audit.job)
    require(read(out/'audit.json')['domains']['finqa']['passed'] is False, 'failed audit domain not exercised')
    row('S6-freeze', s_freeze.job)
    frozen = context(out, True)
    for incoming in sorted({c['incoming'] for c in frozen['spec']['stages']}):
        row('S7-generate', s_generate.job, incoming=incoming, gpu='0')
    handoff.save(out, durable)
    before = {str(p.relative_to(out)): sha(p) for p in out.rglob('*') if p.is_file()}
    shutil.rmtree(out)
    with patch.dict(os.environ, SLURM_JOB_ID='screen-s-standin', SLURM_RESTART_COUNT='1', SLURM_JOB_START_TIME=str(time.time()-1)):
        record = handoff.record_job_start(durable, True)
        require(record['slurm_restart_count'] == 1, 'requeue receipt missing')
    handoff.restore(durable, out)
    require(all(sha(out/name) == value for name, value in before.items()), 'requeue changed frozen evidence')
    row('S7-generate-resume', s_generate.job, incoming='base', gpu='0')
    for stage in frozen['spec']['stages']:
        row('S8-drift', s_drift.job, stage=stage['id'], gpu='0')
    row('S9-boundaries', s_score.job, action='boundaries', finqa=None, gpu='0')
    for incoming in sorted({c['incoming'] for c in frozen['spec']['stages']}):
        for domain in s_generate.domains_for(frozen['spec'], incoming):
            row('S10-probes', s_probe.job, incoming=incoming, domain=domain, gpus='0,1,2,3,4,5,6,7', score_gpu='0',
                trainer_python=sys.executable, inference_python=sys.executable)
    row('S10-probe-resume', s_probe.job, incoming='base', domain='gsm8k', gpus='0,1,2,3,4,5,6,7', score_gpu='0')
    require(not list((out/'probes').rglob('hf-probe')), 'probe output retained')
    row('S11-price', s_price.job)
    archive = root/'screen-s-return.tar.gz'
    row('S12-collect', s_collect.job, archive=archive)
    row('S12-collect', s_collect.job, archive=archive)
    # gzip timestamp headers can differ; contents and registered readback must not.
    doc = analysis.load_archive(archive, True)
    result = analysis.analyze(doc, True)
    require(len(doc['association']) == 24 and len(doc['forward']) == 12, 'archive cell counts wrong')
    try:
        analysis.load_archive(archive)
    except ValueError as exc:
        require('technical_synthetic' in str(exc), 'wrong archive refusal')
    else:
        raise ValueError('synthetic archive accepted without --standin')
    counts['S13-archive-readback'] = 1
    counts['requeue'] = 1
    counts['failed-audit-domains'] = 1
    counts['probe-receipts'] = len(list((out/'probes').glob('*/*/receipt.json')))
    counts['completed-probes'] = sum(read(p)['status'] == 'COMPLETE' for p in (out/'probes').glob('*/*/receipt.json'))
    atomic(root/'dryrun-summary.json', dict(counts=counts, analysis=result))
    return counts


def main():
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--standin', action='store_true', required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    with (a.out/'dryrun.log').open('w') as log, contextlib.redirect_stdout(log):
        counts = run(a.out)
    print('Screen S dry run: '+json.dumps(counts, sort_keys=True))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
