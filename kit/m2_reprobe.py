#!/usr/bin/env python3
"""Fresh four-arm C-route re-probe and frozen on-node nomination; no job submission."""
from pathlib import Path
import sys
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kit.m2_common import *
from types import SimpleNamespace as NS, FunctionType
from kit import m2_assignment as assignment, m2_snapshot as snapshots
import shutil

REPAIR='message2-four-arm-reprobe.v2'


def clone(function,**overrides):
    return FunctionType(function.__code__,dict(function.__globals__,**overrides),function.__name__,function.__defaults__,function.__closure__)


def prepare(a):
    original=binding(a.message2,a.standin)
    assets=json.loads((a.message2/'assets.json').read_text())
    root=Path(assets['inputs']['pools']['path']);pool=frozen_files(root)
    from kit.next_pools import POOL_PROTOCOL
    require(pool.get('pool_protocol')==POOL_PROTOCOL,'re-probe requires regenerated fixed-prompt broad pool')
    if not a.standin:
        from kit.next_hellaswag import strict_label
        labels={r['id']:r.get('labels','ABCD') for r in t.read_jsonl(root/'candidates.jsonl')}
        require(all(strict_label(r['response'],labels[r['id']]) is not None for r in t.read_jsonl(root/'arc.jsonl')),
                'ARC replay answers must satisfy the unchanged strict final-line parser')
    starts=dict(original['starts'])
    for seed in SEEDS:
        ref=original.get('reprobe_starts',{}).get(str(seed),original['starts'].get(f'finqa-plain:{seed}'))
        require(ref is not None,'C after-FinQA incoming missing')
        checked_model(ref,a.standin)
        starts[f'finqa-sibling:{seed}']=ref
    doc=dict(original,starts=starts,purpose=REPAIR,source_binding_sha256=s.file_hash(a.message2/'binding.json'),
        repair=dict(pool_protocol=POOL_PROTOCOL,pool_manifest=assets['inputs']['pools']['manifest'],
            hellaswag_manifest_sha256=s.file_hash(a.message2/'hellaswag/manifest.json'),
            arc_prompt=__import__('kit.next_pools',fromlist=['ARC_PROMPT']).ARC_PROMPT,strict_parser=s.file_hash(KIT/'next_hellaswag.py')),
        protocol=protocol(),out=str(a.out.resolve()))
    bind(a.out/'binding.json',doc);bind(a.out/'assets.json',assets)
    if not (a.out/'hellaswag').exists():shutil.copytree(a.message2/'hellaswag',a.out/'hellaswag')
    check(a.out/'hellaswag/manifest.json',s.file_hash(a.message2/'hellaswag/manifest.json'))


def checkpoint(out,doc,arm,seed,step):
    if arm=='base':return doc['starts']['base']
    require(arm in [legal_arm(c) for c in CONFIGS],'unregistered re-probe arm')
    if step==0:return doc['starts'][f'finqa-sibling:{seed}']
    root=out/'runs'/f'{arm}-s{seed}';frozen_files(root)
    return dict(model=str(root/f'hf-step{step}'),receipt=str(root/f'export-{step}.json'))


def score(a):
    from kit import m2_score
    doc=binding(a.out,a.standin);require(doc.get('purpose')==REPAIR,'fresh re-probe binding required')
    proof=assignment.active(a.out,a.gpu,a.standin)
    if proof:require(proof['family']=='reprobe' and proof['seed']==a.seed,'UNVERIFIED: re-probe family/seed differs')
    clone(m2_score.job,record=checkpoint)(NS(**dict(vars(a),arm='base' if a.arm=='base' else legal_arm(a.arm))))


def train(a):
    from kit import m2_train
    doc=binding(a.out,a.standin);require(doc.get('purpose')==REPAIR,'fresh re-probe binding required')
    args=NS(**dict(vars(a),stage='legal',configuration=a.arm,trajectory='plain',durable=None))
    stage=clone(m2_train.stage,permitted_legal=lambda out,doc,config:config in CONFIGS,record=checkpoint,PANEL_SETS=('finqa','test','general-v1'))
    clone(m2_train.job,stage=stage)(args)


def adjudicate(rows,complete=True):
    """Same probe endpoints and two-sided seed-t bars. Inconclusive is resolved non-nomination."""
    from kit.next_read import analyse,benefit,protects
    pending=dict(status='PENDING',broad=None,earlier=None,earlier_comparison=None,admission='Pending connected finals')
    if not complete and not coverage(rows,('generic','broad')):return pending
    mapped=[dict(r,arm='v4-'+r['arm'].removeprefix('legal-') if r['arm'].startswith('legal-') else r['arm'],
        panel='contract_nli' if r['panel']=='test' else r['panel']) for r in rows if r['panel']!='panel']
    result=analyse(mapped,1,acquisition='contract_nli')
    comparisons={(c['candidate'],c['control']):c['endpoints'] for c in result['v4_comparisons']}
    def valid(e,primary,other):
        required=(primary,other,'hellaswag','gain','contract_nli')
        return all(e[k]['n']==3 and e[k]['lower'] is not None and e[k]['mean'] is not None for k in required)
    def passes(e,primary,other):return benefit(e[primary]) and all(protects(e[k]) for k in (other,'hellaswag','gain','contract_nli'))
    from kit.m2_format import letter_cells
    diagnostics=letter_cells(rows)
    broad=comparisons['broad','generic']
    if not valid(broad,'math','survival'):return pending
    nominated=passes(broad,'math','survival');candidate,control=('both','broad') if nominated else ('earlier','generic')
    earlier=comparisons[candidate,control]
    if not valid(earlier,'survival','math'):return dict(pending,broad=nominated,broad_status='NOMINATED' if nominated else 'INCONCLUSIVE')
    earlier_yes=passes(earlier,'survival','math')
    if candidate=='both':
        safeguard=comparisons['both','generic']
        if any(safeguard[k]['lower'] is None for k in ('gain','contract_nli')):return dict(pending,broad=nominated)
        earlier_yes=earlier_yes and all(protects(safeguard[k]) for k in ('gain','contract_nli'))
    return dict(status='COMPLETE',broad=nominated,earlier=earlier_yes,earlier_comparison=f'{candidate} vs {control}',
        broad_status='NOMINATED' if nominated else 'INCONCLUSIVE',earlier_status='NOMINATED' if earlier_yes else 'INCONCLUSIVE',
        admission='Pending connected finals',format_cells=diagnostics,comparisons={f'{c} vs {k}':v for (c,k),v in comparisons.items()})


def coverage(rows,configs=CONFIGS):
    from collections import defaultdict
    cells=defaultdict(set)
    for r in rows:
        key=(r['arm'],int(r['seed']),int(r['step']),r['panel'])
        require(r['question_id'] not in cells[key],'duplicate re-probe question')
        cells[key].add(r['question_id'])
    expected={p:{r['id'] for r in legal_panels()[p]} for p in ('test','panel')}
    general=t.read_jsonl(KIT/'panels/general-v1.jsonl')
    expected.update({p:{r['id'] for r in general if r['panel']==p} for p in ('ifeval','math','knowledge')})
    def covered(key):
        ids=cells.get(key,set());p=key[-1]
        return ids==expected[p] if p in expected else len(ids)==(1147 if p=='finqa' else 100)
    panels=('test','panel','finqa','ifeval','math','knowledge','hellaswag')
    needed=[('base',seed,0,p) for seed in SEEDS for p in panels]
    needed += [(legal_arm(c),seed,k,p) for c in configs for seed in SEEDS for p in panels
        for k in (STEPS if p in ('test','panel') else RETENTION_STEPS)]
    return all(covered(k) for k in needed)


def verify_record(record,authority,core_sha):
    require(record['schema']==REPAIR and record['binding_sha256']==core_sha,'nomination bound to another core')
    require(record['assignment_sha256']==t.sha(authority),'nomination assignment changed')
    doc=record['reprobe_binding']
    require(doc['source_binding_sha256']==core_sha and doc['purpose']==REPAIR and doc['repair']==record['repair'],'re-probe source/repair changed')
    require(doc['repair']['strict_parser']==s.file_hash(KIT/'next_hellaswag.py'),'re-probe strict parser changed')
    require(record['repair']['pool_protocol']==__import__('kit.next_pools',fromlist=['POOL_PROTOCOL']).POOL_PROTOCOL,'re-probe pool repair changed')
    cells=set()
    from kit.m2_score import items_for
    panel_ids={p:s.digest(items_for(p,True)) for p in ('test','panel','general-v1')}
    panel_ids['hellaswag']=doc['repair']['hellaswag_manifest_sha256']
    finqa_panels=set()
    for meta in record['identities'].values():
        assignment.verify(meta,authority)
        require(meta['assignment']['family']=='reprobe' and meta['assignment']['seed']==meta['seed'],'UNVERIFIED: wrong re-probe contrast')
        require(meta['protocol']==doc['protocol'] and meta['binding_sha256']==record['reprobe_binding_sha256'],'re-probe scoring protocol changed')
        require(meta['decoding']==dict(seed=0,temperature=0,thinking=False,max_new_tokens=2048,eager=True,batch_invariant=True),'re-probe decoding changed')
        arm,seed,step,panel=meta['arm'],meta['seed'],meta['step'],meta['panel']
        if panel in panel_ids:require(meta['panel_sha256']==panel_ids[panel],'UNVERIFIED: re-probe panel/prompt changed')
        if panel=='finqa':finqa_panels.add(meta['panel_sha256'])
        require(seed in SEEDS and (arm=='base' and step==0 or arm in [legal_arm(c) for c in CONFIGS] and step in STEPS),'foreign re-probe cell')
        if arm=='base':ref=doc['starts']['base']
        elif step==0:ref=doc['starts'][f'finqa-sibling:{seed}']
        else:ref=dict(identity=record['export_receipts'][f'runs/{arm}-s{seed}/export-{step}.json'])
        if not record['technical_synthetic']:
            require(meta['model']['model_file_hashes']==ref['identity']['model_file_hashes'],'UNVERIFIED: re-probe export changed')
        for p in ('ifeval','math','knowledge') if panel=='general-v1' else (panel,):cells.add((arm,seed,step,p))
    require(all((r['arm'],int(r['seed']),int(r['step']),r['panel']) in cells for r in record['rows']),'UNVERIFIED: re-probe rows lack constituent identities')
    require(len(finqa_panels)<=1,'UNVERIFIED: unmatched FinQA scoring panels')
    computed=adjudicate_scope(record['rows'],record.get('scope','full'))
    require(computed==record['nomination'],'nomination differs from frozen four-arm reader')
    return computed


def adjudicate_scope(rows,scope):
    value=adjudicate(rows,coverage(rows))
    if scope=='broad' and value.get('broad') is not None:
        return dict(value,status='BROAD_COMPLETE',earlier=None,earlier_comparison=None,earlier_status='PENDING')
    return value


def nominate(a):
    from kit.p3b_read import load
    doc=binding(a.out,a.standin);require(doc.get('purpose')==REPAIR,'fresh re-probe binding required')
    rows,files=load(a.out)
    scope=getattr(a,'scope','full')
    # An uncommitted cell is incomplete, not an identity refusal.
    prefixes={'/'.join(Path(n).parts[:2]) for n in files if n.startswith('scores/') and len(Path(n).parts)>2}
    incomplete=set()
    for prefix in prefixes:
        manifest=files.get(prefix+'/manifest.json')
        try:declared=json.loads(manifest).get('files',{}) if manifest else {}
        except json.JSONDecodeError:declared={}
        if 'm2-identity.json' not in declared:incomplete.add(prefix)
    for prefix in incomplete:
        parts=Path(prefix).name
        cells=[r for r in rows if parts==f"{r['arm']}-s{r['seed']}-k{r['step']}-"+('general-v1' if r['panel'] in ('ifeval','math','knowledge') else r['panel'])]
        rows=[r for r in rows if r not in cells]
    files={n:v for n,v in files.items() if not any(n.startswith(prefix+'/') for prefix in incomplete)}
    if scope=='broad':
        rows=[r for r in rows if r['arm'] in ('base','legal-generic','legal-broad')]
        files={n:v for n,v in files.items() if not any('/'+arm+'-' in '/'+n for arm in ('legal-earlier','legal-both'))}
    assignment.validate_files(files,require_assignment=not a.standin)
    identities={n:json.loads(v) for n,v in files.items() if n.startswith('scores/') and n.endswith('/m2-identity.json')}
    if not a.standin:
        required={str(p.parent) for p in map(Path,files) if p.parts[0]=='scores' and p.name=='scores.csv'}
        require({str(Path(n).parent) for n in identities}==required,'UNVERIFIED: missing re-probe constituent identity')
    result=adjudicate_scope(rows,scope)
    authority=(a.out/'scoring-assignment.json').read_bytes()
    rows=sorted(rows,key=lambda r:(r['arm'],int(r['seed']),int(r['step']),r['panel'],r['question_id']))
    record=dict(schema=REPAIR,scope=scope,binding_sha256=doc['source_binding_sha256'],assignment_sha256=t.sha(authority),
        repair=doc['repair'],nomination=result,rows=rows,identities=identities,
        reprobe_binding=doc,reprobe_binding_sha256=s.file_hash(a.out/'binding.json'),export_receipts={n:json.loads(v) for n,v in files.items() if n.startswith('runs/') and Path(n).name.startswith('export-') and n.endswith('.json')},
        evidence_sha256=s.digest({n:t.sha(v) for n,v in files.items() if n.startswith('scores/')}),technical_synthetic=a.standin)
    verify_record(record,authority,doc['source_binding_sha256'])
    path=a.out/'nomination.json'
    if path.exists() and json.loads(path.read_text())['nomination']['status']=='COMPLETE':bind(path,record)
    snapshots.atomic(path,record)
    ready='BROAD_COMPLETE' if scope=='broad' else 'COMPLETE'
    if result['status']!=ready:raise NotReady('re-probe nomination still pending; no earlier-only fallback')
    commit_nomination(a.out,a.durable,record,scope)


def commit_nomination(out,durable,record,scope='full'):
    path=out/'nomination.json'
    require(durable is not None,'durable nomination root required')
    destination=durable/('nomination-broad' if scope=='broad' else 'nomination')
    key='reprobe-broad-nomination' if scope=='broad' else 'reprobe-nomination'
    if (destination/'COMMIT.json').exists():
        commit=json.loads((destination/'COMMIT.json').read_text())
        require(commit['record_sha256']==s.file_hash(path) and (destination/'record.json').read_bytes()==path.read_bytes(),'immutable nomination changed')
        pointer=snapshots.read_pointer(durable,commit['task'],completed=True)
        require(pointer['sha256']==commit['snapshot_sha256'],'nomination durable snapshot differs')
        snapshots.immutable(out/'nomination-commit.json',(json.dumps(commit,indent=2,sort_keys=True)+'\n').encode());return
    payload=out.parent/'nomination-payload';payload.mkdir(exist_ok=True)
    snapshots.atomic(payload/'nomination.json',record)
    snapshots.atomic(payload/'identities.json',dict(binding=record['binding_sha256'],assignment=record['assignment_sha256']))
    try:
        published=snapshots.read_pointer(durable,key,completed=True)
        previous=snapshots.header(durable/'snapshots'/key/published['tar'],published['manifest_sha256'])
        require(previous['files']['nomination.json']['sha256']==s.file_hash(path),'immutable nomination snapshot changed')
    except snapshots.Pending:
        published=snapshots.publish(payload,durable,key,binding=record['binding_sha256'],assignment=record['assignment_sha256'])
    commit=dict(record_sha256=s.file_hash(path),snapshot_sha256=published['sha256'],task=key,
        assignment_sha256=record['assignment_sha256'])
    with snapshots.lock(destination):
        snapshots.immutable(destination/'record.json',path.read_bytes())
        snapshots.immutable(destination/'COMMIT.json',(json.dumps(commit,indent=2,sort_keys=True)+'\n').encode())
    # Last: a crash before this receipt reuses the dedicated immutable object.
    snapshots.immutable(out/'nomination-commit.json',(json.dumps(commit,indent=2,sort_keys=True)+'\n').encode())


def collect_attachment(a):
    """Return pending/refused receipts for review; never turn them into a decision."""
    import tarfile,io
    files={str(p.relative_to(a.out)):p.read_bytes() for p in a.out.rglob('*') if p.is_file() and not p.is_symlink() and
        p.suffix in ('.json','.csv','.md','.log') and 'raw' not in p.parts and p.name!='reprobe-return-index.json'}
    status='PENDING'
    try:assignment.validate_files(files,require_assignment=not a.standin)
    except (assignment.Unverified,ValueError,KeyError,json.JSONDecodeError) as error:
        status='UNVERIFIED';files['refusal.json']=json.dumps(dict(status=status,reason=str(error))).encode()
    index=dict(status=status,attachment_only=True,scientific_evidence=False,files={n:t.sha(v) for n,v in files.items()})
    files['reprobe-return-index.json']=json.dumps(index,sort_keys=True).encode()
    archive=a.archive or a.out/'reprobe-return.tar.gz';temporary=archive.with_suffix('.tmp');archive.parent.mkdir(parents=True,exist_ok=True)
    with tarfile.open(temporary,'w:gz') as tar:
        for name,raw in sorted(files.items()):
            info=tarfile.TarInfo('return/'+name);info.size=len(raw);tar.addfile(info,io.BytesIO(raw))
    require(temporary.stat().st_size<28*1024*1024,'re-probe attachment exceeds compact return cap')
    temporary.replace(archive);atomic(a.out/'reprobe-return-index.json',dict(index,archive_sha256=s.file_hash(archive)))


def job(a):
    if a.action=='collect':return collect_attachment(a)
    if a.action=='prepare':return prepare(a)
    if a.action=='train':return train(a)
    if a.action=='score':return score(a)
    return nominate(a)


def configure(p):
    p.add_argument('--scope',choices=('broad','full'),default='full')
    p.add_argument('--action',choices=('prepare','train','score','nominate','collect'),required=True)
    p.add_argument('--archive',type=Path)
    p.add_argument('--message2',type=Path);p.add_argument('--durable',type=Path)
    p.add_argument('--arm',choices=('base',*CONFIGS));p.add_argument('--seed',type=int,choices=SEEDS)
    p.add_argument('--step',type=int,choices=STEPS,default=0);p.add_argument('--panel-set',choices=(*PANEL_SETS,'panel','hellaswag'))
    p.add_argument('--gpu',default='0');p.add_argument('--panel-root',type=Path)
    p.add_argument('--tokenizer');p.add_argument('--sdpo',type=Path);p.add_argument('--trainer-python',default=sys.executable)
if __name__=='__main__':raise SystemExit(entry(job,configure,'reprobe-{action}-{arm}-{seed}'))
