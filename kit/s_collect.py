#!/usr/bin/env python3
"""Collect compact Screen S evidence, precomputed cells and hash-indexed bulk journals."""
from pathlib import Path
import sys
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kit.s_common import *
from kit.s_generate import selected, domains_for
import io
import gzip
import tarfile


def change(out, stage, step, domain, forward=False):
    lo = stage['checkpoints']['20'] if forward else stage['incoming']
    hi = stage['checkpoints']['40'] if forward else stage['checkpoints'][str(step)]
    records = []
    for ident in (lo, hi):
        namespace=out/'accuracy'/f"screen-s-s{stage['seed']}"
        path = (namespace if (out/'scoring-assignment.json').exists() else out/'accuracy')/(ident+'.json')
        if not path.exists():
            return None, None
        record = read(path)
        records.append(record['panels'].get(PANELS[domain]))
    before, after = records
    if not before or not after:
        return None, None
    require(set(before) == set(after), 'accuracy panel membership changed')
    total = len(before)
    loss = 100*(sum(before.values())-sum(after.values()))/total
    transitions = dict(n=total, incoming_correct=sum(before.values()),
        lost=sum(before[k] == 1 and after[k] == 0 for k in before),
        learned=sum(before[k] == 0 and after[k] == 1 for k in before),
        never_correct=sum(before[k] == after[k] == 0 for k in before))
    return loss, transitions


def measurements(out, doc):
    audit = read(out/'audit.json')
    association, forward = [], []
    for stage in doc['spec']['stages']:
        domain_rows = {}
        forward_rows = {}
        for domain in domains_for(doc['spec'], stage['incoming']):
            own = selected(out, stage['incoming'], domain)
            from kit.s_probe import receipt_path
            path=receipt_path(out,stage['incoming'],domain,family='screen-s',seed=stage['seed']) if (out/'scoring-assignment.json').exists() else out/'probes'/stage['incoming']/domain/'receipt.json'
            probe = read(path) if path.exists() else {}
            if probe:
                require(probe['inputs']['incoming'] == doc['identities'][stage['incoming']] and probe['inputs']['targets'] == {k: digest(v) for k, v in own.items()}, 'probe evidence changed')
            observed = {}
            for step in ('0', '10', '20', '40'):
                path = out/'nll'/stage['id']/domain/(step+'.json')
                if path.exists():
                    value = read(path)
                    require(value['inputs']['targets'] == digest(own['monitor']) and value['inputs']['incoming'] == stage['incoming'], 'drift evidence changed')
                    observed[step] = value['nll']
                else:
                    observed[step] = None
            baseline = observed['0']['token_mean'] if observed['0'] else None
            common = dict(audit_passed=audit['domains'][domain]['passed'], monitor_n=len(own['monitor']), train_n=len(own['train']),
                          start_nll=baseline, plasticity=probe.get('drop'), final_probe_nll=probe.get('final_nll'))
            domain_rows[domain] = (common, observed)
            loss, transitions = change(out, stage, 40, domain, forward=True)
            forward_rows[domain] = dict(common, forward=observed['10']['token_mean']-baseline if observed['10'] and baseline is not None else None,
                                         loss=loss, transitions=transitions)
        for step in (20, 40):
            values = {}
            for domain, (common, observed) in domain_rows.items():
                loss, transitions = change(out, stage, step, domain)
                drift = observed[str(step)]['token_mean']-common['start_nll'] if observed[str(step)] and common['start_nll'] is not None else None
                values[domain] = dict(common, drift=drift, loss=loss, transitions=transitions,
                                      nll=observed[str(step)], stage_start_nll=observed['0'])
            association.append(dict(tier=stage['tier'], recipe=stage['recipe'], stage=stage['stage'], seed=stage['seed'], step=step, domains=values))
        # Entirely missing tier-1 step10 support is permitted; tier2 still needs all three seeds.
        if stage['tier'] == 2 or '10' in stage['checkpoints']:
            forward.append(dict(tier=stage['tier'], recipe=stage['recipe'], stage=stage['stage'], seed=stage['seed'], step=40, domains=forward_rows))
    return dict(association=association, forward=forward, technical_synthetic=doc['technical_synthetic'])


def recompute(files,doc):
    """Run the frozen calculator on compact, checked manifests and measurements."""
    import tempfile
    from types import FunctionType
    class Rows(list):
        def __init__(self,count,checksum):super().__init__([None]*count);self.checksum=checksum
    with tempfile.TemporaryDirectory(prefix='s-recompute-') as folder:
        root=Path(folder)
        for name,raw in files.items():
            if name.endswith('.json'):
                target=root/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
        def compact_selected(out,incoming,domain):
            own=json.loads(files[f'own/{incoming}/{domain}/manifest.json'])
            require(own['incoming']==doc['identities'][incoming],'compact own checkpoint changed')
            return {part:Rows(own[part+'_n'],own['selected_digests'][part]) for part in ('train','monitor')}
        namespace=dict(measurements.__globals__,selected=compact_selected,
            digest=lambda value:value.checksum if isinstance(value,Rows) else digest(value))
        calculate=FunctionType(measurements.__code__,namespace)
        return calculate(root,doc)


def job(a):
    doc = context(a.out, a.standin)
    from kit.m2_assignment import validate_files
    validate_files({str(p.relative_to(a.out)):p.read_bytes() for p in a.out.rglob('*') if p.is_file() and p.name!='measurements.json' and p.suffix in ('.json','.csv') and not p.is_symlink()}, require_assignment=not a.standin)
    data = measurements(a.out, doc)
    atomic(a.out/'measurements.json', data)
    # Include receipts, original binding, audit findings and compact accuracy evidence.
    # Own targets/raw responses remain durable and are SHA-indexed, never weights.
    require((a.out/'scoring-assignment.json').exists() or not any((a.out/'probes').glob('*/*/work')), 'unfinished probe outputs remain; score/clean before collection')
    evidence = {}
    bulk = []
    cost = []
    for path in sorted(a.out.rglob('*')):
        if not path.is_file() or path.is_symlink() or path == a.archive:
            continue
        relative = str(path.relative_to(a.out))
        if path.name.endswith('.tmp'):
            raise ValueError('unfinished atomic write')
        if path.name in ('cost.json','training-cost.json') or path.name.startswith('receipt') and path.suffix=='.json' or relative.startswith('nll/') and path.suffix == '.json' and not path.name.endswith('.inputs.json'):
            value = read(path)
            if 'gpu_hours' in value:
                cost.append(dict(path=relative, gpu_hours=value['gpu_hours'], seconds=value.get('seconds')))
        if relative in ('scoring-assignment.json','measurements.json', 'binding.json', 'v4-step10-inventory.json', 'pricing.json', 'staged-inputs.json', 'frozen/manifest.json') or relative.startswith(('v4-step20/', 'accuracy/','nll/','node-preflight/','execution/')) or relative.startswith('panel-scoring/scores/') and path.name in ('scores.csv','identity.json','assignment-identity.json','manifest.json') or relative.startswith('own/') and path.name == 'manifest.json' or relative.startswith('probes/') and (path.name.startswith('receipt') or path.name=='training-cost.json') and path.suffix=='.json':
            key='screen-s/'+relative
            evidence[key+'.gz' if path.name=='scores.csv' else key] = gzip.compress(path.read_bytes(),mtime=0) if path.name=='scores.csv' else path.read_bytes()
        else:
            bulk.append(dict(path=relative, sha256=sha(path), bytes=path.stat().st_size))
    # Full audit can be large; retain eligible IDs and all source/exclusion proof hashes.
    audit = read(a.out/'audit.json')
    evidence['screen-s/audit.json'] = json.dumps(dict(audit, domains={d: dict(v, eligible=[r['id'] for r in v['eligible']]) for d, v in audit['domains'].items()}), sort_keys=True).encode()
    evidence['screen-s/bulk.json'] = json.dumps(bulk, sort_keys=True).encode()
    evidence['screen-s/costs.json'] = json.dumps(cost, sort_keys=True).encode()
    evidence['screen-s/registration.json'] = RESOURCE.read_bytes()
    manifest = dict(files={name: hashlib.sha256(raw).hexdigest() for name, raw in evidence.items()}, technical_synthetic=a.standin)
    evidence['screen-s/manifest.json'] = json.dumps(manifest, sort_keys=True).encode()
    a.archive.parent.mkdir(parents=True, exist_ok=True)
    temp = a.archive.with_suffix(a.archive.suffix+'.tmp')
    with tarfile.open(temp, 'w:gz') as tar:
        for name, raw in sorted(evidence.items()):
            member = tarfile.TarInfo(name)
            member.size = len(raw)
            member.mtime = 0
            tar.addfile(member, io.BytesIO(raw))
    require(temp.stat().st_size < 28*1024*1024, 'compact archive exceeds 28 MiB')
    temp.replace(a.archive)
    print('Screen S archive:', a.archive.stat().st_size, 'bytes')

if __name__ == '__main__':
    raise SystemExit(entry(job, lambda p: p.add_argument('--archive', type=Path, required=True)))
