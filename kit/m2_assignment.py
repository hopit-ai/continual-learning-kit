"""Frozen scoring authority. UUIDs, environments and family IDs are never inferred from ordinals."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from kit.v4_finqa_policy import CorrectnessStop


class Unverified(CorrectnessStop):
    pass


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def require(ok, reason):
    if not ok: raise Unverified('UNVERIFIED: '+reason)


def inventory(gpu, standin=False):
    if standin:
        return dict(node=os.environ.get('M2_STANDIN_NODE','CPU-STANDIN'), uuid='CPU-STANDIN', environment='CPU-STANDIN')
    from kit import v4_teacher as t
    return dict(node=platform.node(), uuid=subprocess.check_output(
        ['nvidia-smi','-i',str(gpu),'--query-gpu=uuid','--format=csv,noheader'], text=True).strip(),
        environment=__import__('kit.second_task_screen',fromlist=['digest']).digest(dict(python=sys.executable, versions=t.runtime_versions())))


def load(path):
    checksum=Path(path).with_suffix(Path(path).suffix+'.sha256')
    if checksum.exists():require(checksum.read_text().strip()==sha(path),'assignment checksum changed')
    doc = json.loads(Path(path).read_text())
    require(doc.get('schema') == 'message2-scoring-assignment.v2', 'assignment schema')
    require(bool(doc.get('assignments')), 'empty scoring assignment')
    for key, value in doc['assignments'].items():
        require(key.count('|') == 1 and key.rsplit('|',1)[1].isdigit(), 'assignment family/seed key')
        require(all(isinstance(value.get(k),str) and value[k] for k in ('node','uuid','environment')), 'incomplete assignment '+key)
    return doc


def freeze(path, doc):
    path = Path(path)
    raw = (json.dumps(doc, indent=2, sort_keys=True)+'\n').encode()
    from kit.m2_snapshot import immutable
    immutable(path,raw)
    load(path)
    checksum=path.with_suffix(path.suffix+'.sha256')
    immutable(checksum,(sha(path)+'\n').encode())
    return sha(path)


def path_for(out=None):
    value=os.environ.get('M2_ASSIGNMENT')
    if value: return Path(value)
    if out is not None and (Path(out)/'scoring-assignment.json').exists(): return Path(out)/'scoring-assignment.json'
    return None


def active(out, gpu, standin, *, family=None, seed=None):
    path=path_for(out)
    if path is None:
        require(standin, 'frozen scoring-assignment.json required before scoring')
        return None  # Historical CPU rehearsal only, never scientific evidence.
    family=family or os.environ.get('M2_FAMILY')
    seed=seed if seed is not None else os.environ.get('M2_FAMILY_SEED')
    require(family and seed is not None, 'explicit contrast family and seed required')
    doc=load(path);key=f'{family}|{int(seed)}'
    require(key in doc['assignments'], 'unassigned contrast '+key)
    if not standin:
        from kit.m2_gpu import prepare
        gpu=prepare(gpu,doc['assignments'][key]['uuid'])['index']
    observed=inventory(gpu,standin)
    require(observed == doc['assignments'][key], 'off-assignment scoring '+key)
    return dict(assignment_sha256=sha(path), family=family, seed=int(seed), **observed)


def verify(meta, assignment):
    """Reader-side check; constituent identity must be hash-bound by its manifest."""
    proof=meta.get('assignment')
    require(isinstance(proof,dict), 'constituent lacks assignment identity')
    key=f"{proof.get('family')}|{proof.get('seed')}"
    require(proof.get('assignment_sha256') == digest_file_bytes(assignment), 'stale assignment hash')
    doc=json.loads(assignment) if isinstance(assignment,(bytes,str)) else assignment
    expected=doc['assignments'].get(key)
    require(expected is not None, 'foreign family '+key)
    require({k:proof.get(k) for k in ('node','uuid','environment')} == expected, 'off-assignment constituent '+key)
    gpu=meta.get('gpu_identity',proof)
    require(gpu.get('uuid')==expected['uuid'] and gpu.get('environment')==expected['environment'] and
            gpu.get('node',gpu.get('host',expected['node']))==expected['node'], 'constituent GPU/environment mismatch')
    return key


def digest_file_bytes(value):
    if isinstance(value,str): value=value.encode()
    require(isinstance(value,bytes), 'reader must supply original assignment bytes')
    return hashlib.sha256(value).hexdigest()


def cache_identity(out=None):
    path=path_for(out)
    if path is None: return None
    require(os.environ.get('M2_FAMILY') and os.environ.get('M2_FAMILY_SEED'), 'cache requires family/seed')
    return dict(assignment_sha256=sha(path),family=os.environ['M2_FAMILY'],seed=int(os.environ['M2_FAMILY_SEED']))


def validate_files(files, *, require_assignment=False):
    """Validate available scores, aliases and NLL/probe evidence before reduction/collection/readback."""
    import gzip
    files=dict(files)
    for name,value in list(files.items()):
        if name.startswith('panel-scoring/') and name.endswith('/scores.csv.gz'):files[name[:-3]]=gzip.decompress(value)
    raw=files.get('scoring-assignment.json')
    if raw is None:
        require(not require_assignment, 'returned assignment missing')
        return
    binding=json.loads(files.get('binding.json',b'{}'))
    for name, value in files.items():
        if name.startswith(('accuracy/','v4-step20/')) and name.endswith('.json'):
            record=json.loads(value)
            if require_assignment:require(isinstance(record.get('assignment'),dict),'missing Screen S accuracy assignment '+name)
            if record.get('assignment'):
                proof=record['assignment']
                expected='s-inventory.'+Path(name).stem if name.startswith('v4-step20/') else 'screen-s'
                require(proof['family']==expected,'foreign accuracy family '+name)
                if name.startswith('accuracy/'):
                    require('/screen-s-s'+str(proof['seed'])+'/' in '/'+name,'foreign accuracy seed '+name)
                else:require(proof['seed']==0,'foreign inventory seed')
                require(record.get('source_files'),'unlinked Screen S accuracy '+name)
                actual={};format_failures={}
                import csv,io
                for panel,source in record['source_files'].items():
                    prefix=source['prefix']
                    for child,expected in source['files'].items():
                        require(prefix+'/'+child in files and hashlib.sha256(files[prefix+'/'+child]).hexdigest()==expected,'unlinked Screen S constituent '+name)
                    identity=json.loads(files[prefix+'/identity.json'])
                    require(verify(identity,raw)==verify(record,raw),'Screen S paired assignment changed')
                    for row in csv.DictReader(io.StringIO(files[prefix+'/scores.csv'].decode())):
                        actual.setdefault(row['panel'],{})[row['question_id']]=int(row['correct'])
                        if row['panel']=='knowledge':
                            require(row.get('incorrect_format') not in (None,''),'strict-format diagnostic missing')
                            format_failures['knowledge']=format_failures.get('knowledge',0)+int(row['incorrect_format'])
                require(actual==record['panels'],'Screen S accuracy differs from scored rows')
                require(format_failures==record.get('strict_format_failures'),'Screen S format diagnostics differ')
        if name.endswith('/manifest.json'):
            prefix=name.rsplit('/',1)[0]+'/'
            for child, expected in json.loads(value).get('files',{}).items():
                if prefix+child in files:
                    require(hashlib.sha256(files[prefix+child]).hexdigest()==expected, 'hash-bound receipt changed: '+prefix+child)
    for name,value in files.items():
        if name.endswith('.json') and name != 'scoring-assignment.json':
            record=json.loads(value)
            if isinstance(record,dict):
                proof=record.get('assignment') or record.get('inputs',{}).get('assignment') if isinstance(record.get('inputs',{}),dict) else record.get('assignment')
                if isinstance(proof,dict) and 'uuid' in proof:
                    verify(dict(assignment=proof),raw)
        if name.endswith(('m2-identity.json','assignment-identity.json')):
            prefix=name.rsplit('/',1)[0]
            manifest=json.loads(files.get(prefix+'/manifest.json',b'{}'))
            require(manifest.get('files',{}).get(name.rsplit('/',1)[1])==hashlib.sha256(value).hexdigest(), 'unbound constituent identity '+name)
            meta=json.loads(value)
            verify(meta,raw)
            protocol=binding.get('scoring_protocol') or binding.get('protocol')
            if protocol and 'protocol' in meta:require(meta['protocol']==protocol,'UNVERIFIED: changed scoring protocol '+name)
            if 'seed' in meta:require(meta['seed']==meta['assignment']['seed'],'score seed differs from assigned seed')
            if meta.get('panel')=='hellaswag' and not meta.get('technical_synthetic'):
                from kit.m2_format import POLICY,validation,selected
                method=meta.get('format_free',{})
                require(method.get('policy_sha256')==sha(POLICY),'missing/changed format-free HellaSwag scorer')
                likelihood_path=prefix+'/likelihoods.json'
                require(likelihood_path in files and manifest.get('files',{}).get('likelihoods.json')==hashlib.sha256(files[likelihood_path]).hexdigest(),'unbound likelihood evidence')
                likelihood=json.loads(files[likelihood_path]);proof=meta['assignment']
                check_name=f"format-validation/{proof['family']}-s{proof['seed']}.json"
                require(check_name in files and hashlib.sha256(files[check_name]).hexdigest()==method.get('validation_sha256'),'technical validation missing/changed')
                report=json.loads(files[check_name]);verify(report,raw)
                require(report['status']=='PASS' and report['policy_sha256']==sha(POLICY),'technical validation not passed')
                panel=files.get('hellaswag/panel.jsonl')
                require(panel is not None,'HellaSwag frozen items missing')
                panel_rows=[json.loads(line) for line in panel.splitlines() if line.strip()]
                expected={r['id'] for r in panel_rows}
                require(len(expected)==100 and len(likelihood['rows'])==100 and {r['id'] for r in likelihood['rows']}==expected,'likelihood item coverage changed')
                for check_record in report['checks']:
                    require(validation(check_record.get('production',[]),check_record.get('reference',[]),expected)['status']=='PASS','technical agreement/finite coverage changed')
                require({c['checkpoint'] for c in report['checks']}==set(report['checkpoints']),'technical reference checkpoint coverage changed')
                gold={r['id']:'ABCD'.index(r['gold']) for r in panel_rows}
                wanted={r['id']:int(selected(r['scores'])==gold[r['id']]) for r in likelihood['rows']}
                import csv,io
                actual={r['question_id']:int(r['correct']) for r in csv.DictReader(io.StringIO(files[prefix+'/scores.csv'].decode()))}
                require(actual==wanted,'HellaSwag accuracy differs from format-free likelihoods')

            if name.startswith('scores/') and 'arm' in meta:
                arm=meta['arm'];family=meta['assignment']['family']
                if arm.startswith('x-t2-'):expected='x-tier2.'+arm[len('x-t2-'):].rsplit('-',1)[0]
                elif arm.startswith('x-'):expected='x-tier1'
                elif binding.get('purpose')=='message2-four-arm-reprobe.v2':expected='reprobe'
                elif 'c_exports' in binding:expected='core'
                else:expected=family
                require(family==expected,'foreign contrast family '+name)
        if name.startswith('nll/') and name.endswith('.json') and not name.endswith('.inputs.json'):
            record=json.loads(value);proof=record.get('inputs',{}).get('assignment')
            if require_assignment:require(isinstance(proof,dict),'missing NLL assignment '+name)
            if proof:
                verify(dict(assignment=proof),raw)
                stage=next((s for s in binding.get('spec',{}).get('stages',[]) if s.get('id')==name.split('/')[1]),None)
                require(stage is not None and proof['family']=='screen-s' and proof['seed']==stage['seed'],'foreign Screen S NLL family/seed')
                step=Path(name).stem;node=stage['incoming'] if step=='0' else stage['checkpoints'][step]
                require(record['inputs']['checkpoint']==binding['identities'][node] and record['inputs']['incoming']==stage['incoming'],'NLL checkpoint changed')
                own=json.loads(files[f"own/{stage['incoming']}/{name.split('/')[2]}/manifest.json"])
                require(record['inputs']['targets']==own['selected_digests']['monitor'],'NLL monitoring targets changed')
        if name.startswith('probes/') and Path(name).name.startswith('receipt') and name.endswith('.json'):
            record=json.loads(value);proof=record.get('assignment')
            if require_assignment:require(isinstance(proof,dict),'missing probe assignment '+name)
            if proof:
                verify(record,raw)
                require(Path(name).name==f"receipt-{proof['family']}-s{proof['seed']}.json",'probe family alias changed')
                incoming=Path(name).parts[1]
                if proof['family']=='screen-s':
                    require(any(s['incoming']==incoming and s['seed']==proof['seed'] for s in binding.get('spec',{}).get('stages',[])),'foreign Screen S probe family/seed')
                else:require(proof['family']=='x-tier1' or proof['family'].startswith('x-tier2.'),'foreign probe family')
                if incoming.startswith('b-s'):require(incoming=='b-s'+str(proof['seed']),'foreign X probe seed')
        if name.startswith(('signals/','tier2/signals/')) and name.endswith('.json') and not name.endswith('.inputs.json'):
            record=json.loads(value);proof=record.get('inputs',{}).get('assignment')
            if require_assignment:require(isinstance(proof,dict),'missing signal assignment '+name)
            if proof:
                verify(dict(assignment=proof),raw)
                parts=Path(name).parts
                if name.startswith('signals/'):
                    require(proof['family']=='x-tier1' and parts[1]=='s'+str(proof['seed']),'foreign tier1 signal family/seed')
                else:
                    comparison=parts[2].removeprefix('t2-').rsplit('-',1)[0]
                    require(proof['family']=='x-tier2.'+comparison and parts[3]=='s'+str(proof['seed']),'foreign tier2 signal family/seed')
        if name.startswith('scores/') and name.endswith('/scores.csv'):
            prefix=name.rsplit('/',1)[0]
            ident=next((files[prefix+'/'+n] for n in ('m2-identity.json','assignment-identity.json') if prefix+'/'+n in files),None)
            require(ident is not None, 'score missing assignment receipt '+prefix)
        if name.endswith('/alias.json'):
            alias=json.loads(value);prefix=name.rsplit('/',1)[0]
            source=alias['source']
            for child,field in (('scores.csv','scores_sha256'),('m2-identity.json','identity_sha256')):
                if field in alias:
                    require(source+'/'+child in files and hashlib.sha256(files[source+'/'+child]).hexdigest()==alias[field], 'stale alias '+name)
            if prefix+'/m2-identity.json' in files and source+'/m2-identity.json' in files:
                require(verify(json.loads(files[prefix+'/m2-identity.json']),raw)==verify(json.loads(files[source+'/m2-identity.json']),raw), 'cross-assignment alias '+name)
    if 'measurements.json' in files:
        from kit.s_collect import recompute
        require(json.loads(files['measurements.json'])==recompute(files,binding),'Screen S measurements differ from checked constituents')


def build_inventory_assignment(nodes, return1=None, tier_b=False, screen_spec=None):
    """Build once from the allocated hardware inventory, before any scoring."""
    from kit.x_tier2 import COMPARISONS
    families=[('reprobe',s) for s in (101,102,103)]+[('core',s) for s in (101,102,103)]+[('x-tier1',s) for s in (101,102,103)]
    families += [('x-tier2.'+name,s) for name in COMPARISONS for s in (101,102,103)]
    from kit.s_common import SEEDS as SCREEN_SEEDS, inventory_slots
    families += [('screen-s',s) for s in SCREEN_SEEDS]
    families += [('x-generation',s) for s in (101,102,103)]
    if screen_spec:
        families += [('s-generation.'+incoming,0) for incoming in sorted({s['incoming'] for s in screen_spec['stages']})]
    families += [('s-inventory.'+row['id'],0) for row in inventory_slots()]
    require(nodes and all(n.get('node') and n.get('gpus') for n in nodes),'hardware inventory incomplete')
    require(len({n['node'] for n in nodes})==len(nodes),'duplicate node')
    values={}
    for index,(family,seed) in enumerate(families):
        node=nodes[index%len(nodes)]
        gpu=node['gpus'][(index//len(nodes))%len(node['gpus'])]
        values[f'{family}|{seed}']=dict(node=node['node'],uuid=gpu['uuid'],environment=gpu['environment'])
    if return1 is not None:
        for package in return1['rescoring']['packages']:
            if package['tier']=='B' and not tier_b:continue
            locations=[n['node'] for n in nodes if any(g['uuid']==package['assigned_gpu_uuid'] for g in n['gpus'])]
            require(len(locations)<=1,'physical UUID appears on multiple nodes')
            values[f"return1.{package['id']}|{package['seed']}"]=dict(node=locations[0] if locations else 'UNAVAILABLE',uuid=package['assigned_gpu_uuid'],environment=package['compatible_environment'])
    return dict(schema='message2-scoring-assignment.v2',assignments=values)


def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--return1-decisions',type=Path);p.add_argument('--enable-tier-b',action='store_true')
    p.add_argument('--screen-spec',type=Path)
    p.add_argument('--inventory',type=Path);p.add_argument('--output',type=Path)
    p.add_argument('--gpu',help='print this allocated GPU UUID and scoring environment')
    a=p.parse_args()
    if a.gpu is not None:print(json.dumps(inventory(a.gpu),sort_keys=True));return
    require(a.inventory is not None and a.output is not None,'inventory and output required')
    print(freeze(a.output,build_inventory_assignment(json.loads(a.inventory.read_text()),json.loads(a.return1_decisions.read_text()) if a.return1_decisions else None,a.enable_tier_b,json.loads(a.screen_spec.read_text()) if a.screen_spec else None)))


if __name__=='__main__':main()
