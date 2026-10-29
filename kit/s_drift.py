#!/usr/bin/env python3
"""EOS-inclusive a4_nll scoring, always against the exact incoming checkpoint."""
from pathlib import Path
import sys
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kit.s_common import *
from kit.s_generate import selected, domains_for
import time


def nll(model_path, targets, standin, receipt=None):
    require(bool(targets) and all(r['verified'] is True for r in targets), 'NLL requires verified own responses')
    from kit.a4_nll import measure
    if standin:
        return measure(model_path, targets, True)
    # A process per NLL releases all CUDA allocations before training/scoring next.
    import tempfile
    import subprocess
    with tempfile.TemporaryDirectory(prefix='screen-s-nll-') as temporary:
        root = Path(temporary)
        atomic(root/'reference.json', dict(model=model_path, receipt=receipt))
        write_rows(root/'targets.jsonl', targets)
        subprocess.run([sys.executable, str(ROOT/'kit/s_nll.py'), '--reference', str(root/'reference.json'),
                        '--schedule', str(root/'targets.jsonl'), '--output', str(root/'nll.json')], check=True)
        return read(root/'nll.json')


def job(a):
    doc = context(a.out, a.standin)
    from kit.m2_handoff import scoring_identity
    scoring_machine=scoring_identity(a.out, a.gpu, a.standin)
    os.environ['CUDA_VISIBLE_DEVICES'] = a.gpu
    stage = next(c for c in doc['spec']['stages'] if c['id'] == a.stage)
    proof=scoring_machine.get('assignment')
    if proof:require(proof['family']=='screen-s' and proof['seed']==stage['seed'],'UNVERIFIED: drift assignment differs from stage')
    incoming = stage['incoming']
    checkpoints = {'0': incoming, **stage['checkpoints']}
    for domain in domains_for(doc['spec'], incoming):
        own = selected(a.out, incoming, domain)
        for step, ident in checkpoints.items():
            checkpoint = doc['spec']['nodes'][ident]['checkpoint']
            identity = model(checkpoint, a.standin)
            require(identity == doc['identities'][ident], 'scoring checkpoint changed')
            path = a.out/'nll'/stage['id']/domain/(step+'.json')
            inputs = dict(assignment=scoring_machine.get('assignment'),checkpoint=identity, targets=digest(own['monitor']), incoming=incoming,
                          registration_sha256=sha(RESOURCE), technical_synthetic=a.standin)
            bind(path.with_suffix('.inputs.json'), inputs)
            if path.exists():
                require(read(path)['inputs'] == inputs, 'NLL identity mismatch')
                continue
            start = time.monotonic()
            value = nll(checkpoint.get('model', 'standin'), own['monitor'], a.standin, checkpoint.get('receipt')) if len(own['monitor']) >= 64 else None
            if a.standin and value is not None:
                # Technical fixtures retain a4_nll's reductions and counts.
                value = dict(value, token_mean=1+DOMAINS.index(domain)*.2+int(step)/100,
                             sequence_mean=1+DOMAINS.index(domain)*.2+int(step)/100)
            atomic(path, dict(inputs=inputs, nll=value, seconds=time.monotonic()-start,
                              gpu_hours=0 if a.standin else (time.monotonic()-start)/3600))
    print('Drift scoring:', stage['id'], '0/'+ '/'.join(stage['checkpoints']))


def configure(p):
    p.add_argument('--stage', required=True)
    p.add_argument('--gpu', default='0')
if __name__ == '__main__':
    raise SystemExit(entry(job, configure))
