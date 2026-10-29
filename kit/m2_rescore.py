#!/usr/bin/env python3
"""Independent return-1 rescoring. Exact Tier A calls; optional Tier B is off by default."""
from pathlib import Path
import sys
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kit.m2_common import *
from kit.m2_reprobe import clone
from kit import m2_assignment as assignment
from types import SimpleNamespace as NS
from contextlib import contextmanager
import shutil


@contextmanager
def scope(family,seed):
    values=dict(M2_FAMILY=family,M2_FAMILY_SEED=str(seed))
    previous={k:os.environ.get(k) for k in values};os.environ.update(values)
    try:yield
    finally:
        for k,v in previous.items():
            if v is None:os.environ.pop(k,None)
            else:os.environ[k]=v


def prepare(a):
    from kit.m2_return1 import decisions
    core=binding(a.message2,a.standin);ruling=decisions(a.decisions)
    require(core['decisions_sha256']==s.file_hash(a.decisions),'rescoring decisions differ from send binding')
    packages=[p for p in ruling['rescoring']['packages'] if p['tier']=='A' or a.enable_tier_b]
    require(sum(len(p['calls']) for p in packages if p['tier']=='A')==31 and sum(p['tier']=='A' for p in packages)==3,'reviewed Tier A matrix changed')
    for package in packages:
        for call in package['calls']:
            require(call['checkpoint'] in core['rescore_refs'],'checkpoint paths must come from send binding: '+call['checkpoint'])
    doc=dict(core,purpose='return1-exact-rescoring.v2',packages=packages,enable_tier_b=a.enable_tier_b,
        source_binding_sha256=s.file_hash(a.message2/'binding.json'),protocol=protocol())
    bind(a.out/'binding.json',doc)
    if not (a.out/'scoring-assignment.json').exists():shutil.copyfile(a.assignment or a.message2/'scoring-assignment.json',a.out/'scoring-assignment.json')
    authority=assignment.load(a.out/'scoring-assignment.json')
    from kit.p3b_read import load
    _,original_files=load(Path(core['c_source']))
    for package in packages:
        for call in package.get('retained_calls',[]):
            prefix=call['score_set']+'/'
            require(call['identity_path'] in original_files and t.sha(original_files[call['identity_path']])==call['identity_sha256'],'retained original identity missing/changed')
            companion=json.loads(original_files[call['companion_path']])
            require(companion['gpu_identity']['uuid']==package['assigned_gpu_uuid'] and companion['gpu_identity']['environment']==package['compatible_environment'],'retained original scoring machine differs')
            for name,raw in original_files.items():
                if name.startswith(prefix) and name.rsplit('/',1)[1] in ('identity.json','scores.csv','manifest.json'):
                    path=a.out/'retained'/package['id']/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
            bind(a.out/'retained'/package['id']/call['companion_path'],companion)
        key=f"return1.{package['id']}|{package['seed']}"
        require(key in authority['assignments'],'missing frozen rescoring assignment '+key)
        expected=authority['assignments'][key]
        require(expected['uuid']==package['assigned_gpu_uuid'] and expected['environment']==package['compatible_environment'],
                'Tier A/B scoring UUID/environment differs from reviewed decisions')


def score(a):
    from kit import m2_score
    doc=binding(a.out,a.standin);packages=[p for p in doc['packages'] if p['id']==a.package]
    require(len(packages)==1,'unbound/disabled return-1 package')
    package=packages[0];root=a.out/'packages'/package['id']
    bind(root/'binding.json',doc)
    if not (root/'scoring-assignment.json').exists():shutil.copyfile(a.out/'scoring-assignment.json',root/'scoring-assignment.json')
    with scope('return1.'+package['id'],package['seed']):
        proof=assignment.active(root,a.gpu,a.standin)
        require(proof['uuid']==package['assigned_gpu_uuid'] and proof['environment']==package['compatible_environment'], 'UNVERIFIED: return-1 off-assignment')
        for index,call in enumerate(package['calls']):
            if a.call is not None and a.call!=index:continue
            ref=doc['rescore_refs'][call['checkpoint']]
            from kit.m2_bind import tier_a_base
            observed=checked_model(ref,a.standin,base=tier_a_base(call))
            require(a.standin or observed['model_file_hashes']==call['model_hashes'],'wrong return-1 checkpoint export')
            arm_seed=Path(call['score_set']).name.rsplit('-s',1);arm=arm_seed[0]
            args=NS(out=root,standin=a.standin,arm=arm,seed=package['seed'],step=call['step'],panel_set=call['panel_set'],gpu=a.gpu,panel_root=a.panel_root)
            clone(m2_score.job,binding=lambda out,standin:doc,record=lambda *args,ref=ref:ref)(args)
            folder=root/call['score_set']
            bind(folder/'return1-source.json',dict(package=package['id'],call_index=index,source=call,decisions_sha256=doc['decisions_sha256']))
            manifest=frozen_files(folder);manifest['files']['return1-source.json']=s.file_hash(folder/'return1-source.json');atomic(folder/'manifest.json',manifest)


def read(source,standin):
    from kit.p3b_read import load
    _,files=load(source);doc=json.loads(files['binding.json'])
    require(doc['technical_synthetic'] is standin and doc['purpose']=='return1-exact-rescoring.v2','rescoring return scope differs')
    assignment.validate_files(files,require_assignment=True)
    missing=[];verified=[]
    for package in doc['packages']:
        for index,call in enumerate(package['calls']):
            prefix='packages/'+package['id']+'/'+call['score_set']+'/'
            if not all(prefix+n in files for n in ('m2-identity.json','scores.csv','manifest.json','return1-source.json')):missing.append(dict(package=package['id'],call=index));continue
            meta=json.loads(files[prefix+'m2-identity.json'])
            require(meta['protocol']==doc['protocol'],'UNVERIFIED: rescoring protocol changed')
            proof=meta['assignment']
            require(proof['family']=='return1.'+package['id'] and proof['seed']==package['seed'] and
                proof['uuid']==package['assigned_gpu_uuid'] and proof['environment']==package['compatible_environment'],'UNVERIFIED: rescoring assignment changed')
            require(meta['step']==call['step'] and meta['panel']==call['panel_set'],'rescoring call changed')
            require(call['score_set']==f"scores/{meta['arm']}-s{meta['seed']}-k{meta['step']}-{meta['panel']}",'score folder arm/seed/step differs')
            require(standin or meta['model']['model_file_hashes']==call['model_hashes'],'rescoring export changed')
            original=json.loads(files[prefix+'return1-source.json'])
            require(original['source']==call and original['decisions_sha256']==doc['decisions_sha256'],'rescoring original identity changed')
            import csv,io
            from collections import defaultdict
            cells=defaultdict(set)
            for row in csv.DictReader(io.StringIO(files[prefix+'scores.csv'].decode())):
                require(int(row['seed'])==package['seed'] and int(row['step'])==call['step'],'rescoring row identity differs')
                require(row['question_id'] not in cells[row['panel']],'duplicate rescoring question')
                cells[row['panel']].add(row['question_id'])
            panels=('ifeval','knowledge','math') if call['panel_set']=='general-v1' else (call['panel_set'],)
            from kit.m2_score import items_for
            items=items_for(call['panel_set'],True)
            expected={p:{r['id'] for r in items if r.get('panel',call['panel_set'])==p} for p in panels}
            if set(cells)!=set(panels) or any(len(cells[p])!=1147 if p=='finqa' else cells[p]!=expected[p] for p in panels):
                missing.append(dict(package=package['id'],call=index,reason='panel coverage incomplete'));continue
            verified.append(dict(package=package['id'],call=index))
    for package in doc['packages']:
        for call in package.get('retained_calls',[]):
            root='retained/'+package['id']+'/'
            required=[call['identity_path'],call['companion_path'],call['score_set']+'/scores.csv',call['score_set']+'/manifest.json']
            if not all(root+n in files for n in required):
                missing.append(dict(package=package['id'],retained=call['score_set']));continue
            require(t.sha(files[root+call['identity_path']])==call['identity_sha256'],'retained original identity changed')
            companion=json.loads(files[root+call['companion_path']])
            require(companion['gpu_identity']['uuid']==package['assigned_gpu_uuid'] and companion['gpu_identity']['environment']==package['compatible_environment'],'retained machine differs')
            manifest=json.loads(files[root+call['score_set']+'/manifest.json'])
            require(manifest['files']['scores.csv']==t.sha(files[root+call['score_set']+'/scores.csv']),'retained scores changed')
        for call in package.get('reused_calls',[]):
            source=next((p for p in doc['packages'] if p['id']==call['from_package']),None)
            require(source is not None,'reused source package missing')
            target='packages/'+source['id']+'/'+call['score_set']+'/m2-identity.json'
            if target not in files:
                missing.append(dict(package=package['id'],reused=call['score_set']));continue
            meta=json.loads(files[target]);assignment.verify(meta,files['scoring-assignment.json'])
            require(meta['assignment']['family']=='return1.'+source['id'] and meta['assignment']['uuid']==package['assigned_gpu_uuid'] and meta['assignment']['environment']==package['compatible_environment'],'reused source assignment differs')
            require(standin or meta['model']['model_file_hashes']==call['model_hashes'],'reused checkpoint differs')
            require(call['score_set']==f"scores/{meta['arm']}-s{meta['seed']}-k{meta['step']}-{meta['panel']}",'reused folder differs')
    return dict(status='PENDING' if missing else 'COMPLETE',verified=verified,missing=missing,enabled_tier_b=doc['enable_tier_b'],
        decisions_sha256=doc['decisions_sha256'],technical_synthetic=standin)


def job(a):
    if a.action=='prepare':return prepare(a)
    if a.action=='score':return score(a)
    if a.action=='collect':
        from kit.m2_collect import job as collect
        return collect(a)
    result=read(a.source or a.out,a.standin);atomic(a.out/'rescoring-reading.json',result);print(result['status'],len(result['verified']),'verified calls')


def configure(p):
    from kit.m2_return1 import DECISIONS
    p.add_argument('--action',choices=('prepare','score','read','collect'),required=True)
    p.add_argument('--message2',type=Path);p.add_argument('--decisions',type=Path,default=DECISIONS)
    p.add_argument('--assignment',type=Path);p.add_argument('--enable-tier-b',action='store_true')
    p.add_argument('--package');p.add_argument('--call',type=int);p.add_argument('--gpu',default='0');p.add_argument('--panel-root',type=Path)
    p.add_argument('--source',type=Path);p.add_argument('--archive',type=Path)
if __name__=='__main__':raise SystemExit(entry(job,configure,'return1-rescore-{action}'))
