"""Offline registered 4A protocol, real row functions and public-only fixtures."""
import argparse
import copy
import importlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS
import pytest
from kit import a4_common as c, a4_bind, a4_sim, a4_score, a4_ledger, a4_memory, a4_train, a4_read, a4_run, a4_collect
from kit.c_sim import Tokenizer


def args(out,**kw):return NS(out=out,standin=True,**kw)


def setup(root,winner='F'):
    sl,b,rp=a4_sim.fixture(root/'fixtures',winner);out=root/'A4'
    a=a4_bind_args(out,sl,b,rp)
    a.approved_ceiling=600 if winner in ('F','F+replay') else 800
    a4_bind.job(a)
    c.start_4a(out, True)
    return out,a


def a4_bind_args(out,sl,b,rp):
    return args(out,slice=sl,b_source=b,base='synthetic-base',finqa=None,replay=rp,stage_gpu_hours=3,
        approved_ceiling=600,approval_reference=c.APPROVAL_REFERENCE,gpu='0')


def score(out,kind,arm,seed,n,p,mode='off'):
    a4_score.job(args(out,kind=kind,arm=arm,seed=seed,month=n,panel=p,mode=mode,gpu='0'))


def arrive(out,n=1):
    doc=c.binding(out,True);mo=c.MONTHS[n-1];a4_memory.archive(out,doc,mo,True)
    for arm,seed,mode in c.arrival_specs():score(out,'arrival',arm,seed,n,mo,mode)
    a4_ledger.job(args(out,month=mo))


@pytest.mark.parametrize('recipe',('F','D','F+replay','D+replay'))
def test_bind_c_rule_and_resumption(tmp_path,recipe):
    out,a=setup(tmp_path,recipe)
    doc=c.binding(out,True)
    assert doc['W']==recipe
    assert doc['approved_ceiling_gpu_hours']==(600 if recipe in ('F','F+replay') else 800)
    assert doc['ceiling_gpu_hours']==350
    before=(out/'binding.json').read_bytes();a4_bind.job(a)
    assert (out/'binding.json').read_bytes()==before
    a.stage_gpu_hours=3.1
    with pytest.raises(c.CorrectnessStop,match='changed'):a4_bind.job(a)


@pytest.mark.parametrize('hours,expected',[(2.7,350),(3,350),(3.3,400),(6.3,600),(6.7,600),(7.73,700)])
def test_registered_cost_formula(hours,expected):assert a4_bind.ceiling(hours)==expected


def test_scientific_slice_and_legacy_pins():
    legacy=json.loads((c.KIT/'resources/next/legacy-sha256.json').read_text())
    assert len(legacy)==53
    for name,digest in legacy.items():c.check(c.KIT/name,digest)
    pins=json.loads((c.RES/'slice.json').read_text())
    assert pins['manifest']=='ed5a8b17357e3e9679f12b653e7a782dfab5af7ac9ddf14452183f26ae4cbcad'
    assert len(pins['files'])==15
    source=c.KIT.parent/'data/p4_sec/slice-4a-primary'
    if source.exists():
        for name,digest in pins['files'].items():c.check(source/name,digest)
    for line in (c.RES/'SHA256SUMS').read_text().splitlines():
        digest,name=line.split();c.check(c.RES/name,digest)
    assert 'Registration final' in (c.RES/'registration.md').read_text()


def test_no_feedback_until_all_arms_sealed(tmp_path):
    out,a=setup(tmp_path);a4_memory.archive(out,c.binding(out,True),c.MONTHS[0],True)
    score(out,'arrival','frozen',0,1,c.MONTHS[0])
    with pytest.raises(c.NotReady):a4_ledger.job(args(out,month=c.MONTHS[0]))
    assert not (out/'ledger'/c.MONTHS[0]/'manifest.json').exists()
    arrive(out)
    traces=c.ledger(out,1)
    assert len(traces)==9 and all(not r['item']['sec']['issuer_holdout'] for r in traces)
    assert all(r['response_sha256']==c.t.sha(r['response'].encode()) for r in traces)
    # No-history generation is identical for every arm/mode/seed, and physically generated once.
    roots=[c.score_root(out,'arrival',arm,seed,1,c.MONTHS[0],mode) for arm,seed,mode in c.arrival_specs()]
    assert len({(p/'raw/responses.jsonl').read_bytes() for p in roots})==1
    assert len({json.loads((p/'generated.json').read_text())['cache_key'] for p in roots})==1
    before=(out/'ledger'/c.MONTHS[0]/'manifest.json').read_bytes();a4_ledger.job(args(out,month=c.MONTHS[0]))
    assert before==(out/'ledger'/c.MONTHS[0]/'manifest.json').read_bytes()


@pytest.mark.parametrize('change,reason',[
    ('holdout','holdout_or_training_excluded'),('permission','holdout_or_training_excluded'),
    ('finish','unfinished'),('tokens','response_cap'),('long','character_cap'),
    ('unit','explicit_unit'),('wrong','historical'),('source','source_unit_provenance')])
def test_admission_both_scores_and_source_caps(tmp_path,change,reason):
    sl,_,_=a4_sim.fixture(tmp_path);i=json.loads((sl/'batch-2026-01.json').read_text())[0]
    r=dict(response='Calculation checked.\nAnswer: 100.0',output_tokens=10,finish_reason='stop')
    assert not a4_ledger.admission(i,r,Tokenizer())
    if change=='holdout':i['sec']['issuer_holdout']=True
    if change=='permission':i['sec']['training_allowed']=False
    if change=='finish':r['finish_reason']='length'
    if change=='tokens':r['output_tokens']=2049
    if change=='long':r['response']='x'*2501+'\nAnswer: 100.0'
    if change=='unit':r['response']='Answer: 100.0%'
    if change=='wrong':r['response']='Answer: 7654321.0'
    if change=='source':i['sec']['operands'][0]['cell']='999'
    assert reason in a4_ledger.admission(i,r,Tokenizer())


@pytest.mark.parametrize('text,expected',[('Answer: 100.0',True),('Answer: 1.0',False),('Answer: 100.0%',False),
    ('Answer: 100.0 percentage points',True),('Answer: $100.0',False),('Answer: 100.1',False)])
def test_explicit_units(tmp_path,text,expected):
    sl,_,_=a4_sim.fixture(tmp_path);i=json.loads((sl/'batch-2026-01.json').read_text())[0]
    assert c.sec.v4_explicit_correct(i,text)==expected


def test_memory_exclusions_cap_and_recurrence(tmp_path):
    sl,_,_=a4_sim.fixture(tmp_path);i=json.loads((sl/'batch-2026-02.json').read_text())[0]
    cases=json.loads((sl/'batch-2026-01.json').read_text())[:4]
    pool=[dict(id=r['id'],item=r,response='Answer: 100.0',month=c.MONTHS[0]) for r in cases]
    built=a4_memory.build(i,pool,Tokenizer())
    assert len(built['case_ids'])==3
    assert a4_memory.recurrence(i,pool)==dict(recurrence=1,stratum='low')
    more=[dict(r,month=c.MONTHS[1]) for r in pool]
    assert a4_memory.recurrence(i,pool+more)==dict(recurrence=2,stratum='high')
    for key,value in [('issuer_ciks',i['sec']['issuer_ciks']),('accession',i['sec']['accession']),
        ('filing_group',i['sec']['filing_group']),('required_accessions',i['sec']['required_accessions']),
        ('required_filing_groups',i['sec']['required_filing_groups'])]:
        modified=copy.deepcopy(cases[0]);modified['sec'][key]=value
        assert not a4_memory.eligible(i,modified)
    class Large(Tokenizer):
        def encode(self,text,**kw):
            return [1]*(9000+800*text.count('Example '))
    dropped=a4_memory.build(i,pool,Large())
    assert len(dropped['case_ids'])==1 and dropped['cases_dropped']==2
    class Over(Tokenizer):
        def encode(self,text,**kw):return [1]*10241
    with pytest.raises(c.CorrectnessStop,match='current documents'):a4_memory.build(i,pool,Over())


@pytest.mark.parametrize('recipe',('F','D','F+replay','D+replay'))
def test_w_recipe_and_eos_masks(tmp_path,recipe):
    a=NS(sdpo=tmp_path,seed=101,gpu='0,1,2,3,4,5,6,7',tokenizer='tokenizer',trainer_python='python')
    cmd,resolve,env=a4_train.command(a,recipe,'base',tmp_path/'data',tmp_path/'run')
    assert 'trainer.total_training_steps=40' in cmd and 'data.train_batch_size=32' in cmd
    assert 'trainer.save_freq=40' in cmd
    assert ('actor_rollout_ref.actor.optim.lr=1e-5' if recipe.startswith('D') else 'optim.lr=1e-5') in cmd
    assert env['P3B_REPLAY']==str(int('+replay' in recipe))
    rows=[dict(id='train',prompt=[dict(role='user',content='Question')],response='An original verified trace.\nAnswer: 100.0')]
    converted,counts=a4_train.training_rows(rows,Tokenizer(),recipe)
    assert converted[0]['data_source']=='a4_sec' and converted[0]['extra_info']['demonstration']==rows[0]['response']
    from kit.c_data import encode
    tok=Tokenizer();p,t=c.t.tokenize_student(tok,rows[0]['prompt'],rows[0]['response'])
    enc=encode(tok,rows[0]['prompt'],rows[0]['response'])
    assert sum(enc['loss_mask'])==len(t) and t==tok.encode(rows[0]['response']+tok.eos_token,add_special_tokens=False) and not any(enc['loss_mask'][:len(p)])


def test_primary_all_bars_and_inconclusive():
    matrix={s:[6+(s-103)*.1]*5 for s in c.SEEDS}
    assert a4_read.primary(matrix)['established']
    assert not a4_read.primary({s:[4.9]*5 for s in c.SEEDS})['established']
    assert not a4_read.primary({s:[30,30,-1,-1,-1] for s in c.SEEDS})['established']
    assert not a4_read.primary({s:[10 if s<104 else -1]*5 for s in c.SEEDS})['established']
    assert a4_read.interval([3]*5,True)['upper']==3
    assert a4_read.interval([0,0,0,0,9],True)['upper']>3
    with pytest.raises(c.CorrectnessStop):a4_read.interval([1]*4)


def test_paired_transitions_reject_unpaired():
    before=[dict(panel='x',question_id=str(i),correct=x) for i,x in enumerate([0,1,1,0])]
    after=[dict(r,correct=x) for r,x in zip(before,[1,0,1,0])]
    assert a4_read.pair(before,after)==dict(n=4,correct_to_wrong=1,wrong_to_correct=1,delta_points=0)
    with pytest.raises(c.CorrectnessStop):a4_read.pair(before,after[:-1])


def dispatch(command,**kwargs):
    """Run each actual entry-point job in process; only process creation is replaced."""
    name=Path(command[1]).stem
    if not name.startswith('a4_'):raise AssertionError(command)
    module=importlib.import_module('kit.'+name);p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path,required=True);p.add_argument('--standin',action='store_true');module.configure(p)
    module.job(p.parse_args(command[2:]));return NS(returncode=0)


def test_full_graph_resume_every_row_and_return(tmp_path,monkeypatch):
    out,a=setup(tmp_path,'D+replay')
    monkeypatch.setattr(subprocess,'run',dispatch)
    monkeypatch.setattr(a4_train,'gpu_run',dispatch)
    run=args(out,through_month=6,score_gpu='0',gpu='0,1,2,3,4,5,6,7',sdpo=None,tokenizer=None,
        trainer_python=sys.executable,inference_python=sys.executable,fail_once=True)
    with pytest.raises(RuntimeError,match='interrupted'):a4_run.job(run)
    assert (out/'runs/direct-s101-m1/failure-injected.json').exists()
    run.fail_once=False;a4_run.job(run)
    records,missing=a4_read.load(out)
    assert not missing and len(records)==len(list(a4_run.score_plan()))
    assert len(list((out/'runs').glob('*/summary.json')))==45
    assert all(len(c.t.read_jsonl(p))==1280 for p in (out/'data').glob('*/schedule.jsonl'))
    snapshots={str(p):c.s.file_hash(p) for p in (out/'runs').rglob('*') if p.is_file()}
    a4_run.job(run)
    assert snapshots=={str(p):c.s.file_hash(p) for p in (out/'runs').rglob('*') if p.is_file()}
    result=json.loads((out/'reading/summary.json').read_text())
    assert len(result['primary']['per_seed'])==5
    assert len(result['retention'])>50 and result['scientific_verdict'] is None
    assert not any(r['panel']=='finqa' for r in result['retention'])
    assert result['retention_established']==all(r['established'] for r in result['retention'] if r['arm']=='consolidation' and r['is_registered_bar'])
    assert result['recurrence']['v6_trigger'] is None and 'EXPLORATORY' in result['recurrence']['status']
    # Reused early consolidation generations still cost the same in an independent deployment.
    g1=records[('arrival','frozen',0,1,c.MONTHS[0],'off')][0]['generation_id']
    g2=records[('arrival','memory',0,2,c.MONTHS[1],'on')][0]['generation_id']
    costs=[dict(kind='scoring',arm='frozen',seed=0,gpu_hours=5,generation_id=g1,generated=True,kind_row='arrival'),
           dict(kind='scoring',arm='memory',seed=0,gpu_hours=7,generation_id=g2,generated=True,kind_row='arrival'),
           dict(kind='training',arm='consolidation',seed=101,month=2,gpu_hours=10)]
    summaries={p.parent.name:json.loads(p.read_text()) for p in (out/'runs').glob('*/summary.json')}
    priced=a4_read.analyse(records,summaries,costs,20)
    assert next(r for r in priced['all_in'] if r['arm']=='memory')['gpu_hours']==12
    assert next(r for r in priced['all_in'] if r['arm']=='consolidation' and r['seed']==101)['gpu_hours']==22
    assert priced['shared_study_gpu_hours']==22
    archive=tmp_path/'return.tar.gz';a4_collect.job(args(out,archive=archive,durable_root=None))
    assert archive.stat().st_size<28*1024*1024
    a4_read.job(args(out,source=archive,bootstrap=2000))
    assert json.loads((out/'returned-reading/summary.json').read_text())==result


def test_empty_ledgers_only_skip(tmp_path,monkeypatch):
    out,a=setup(tmp_path);arrive(out)
    root=out/'ledger'/c.MONTHS[0];(root/'traces.jsonl').write_text('');c.freeze(root,['traces.jsonl','admission.jsonl','sealed.json'])
    for panel in ('general-v1',):score(out,'incoming','direct',101,1,panel)
    train=args(out,arm='direct',seed=101,month=1,gpu='0,1,2,3,4,5,6,7',score_gpu='0',tokenizer=None,sdpo=None,
        trainer_python=sys.executable,inference_python=sys.executable,fail_once=False)
    a4_train.job(train)
    summary=json.loads((out/'runs/direct-s101-m1/summary.json').read_text())
    assert summary['skipped'] and summary['reason']=='empty_monthly_ledger'
    assert c.state(out,c.binding(out,True),'direct',101,1)['model']=='synthetic-base'


def test_wrong_gpu_implementation_and_ledger_stop(tmp_path):
    out,a=setup(tmp_path);arrive(out)
    physical=out/'next-scoring-machine.json';c.atomic(physical,dict(uuid='different',environment='CPU-STANDIN'))
    with pytest.raises(c.CorrectnessStop):score(out,'base','frozen',0,0,'general-v1')
    trace=out/'ledger'/c.MONTHS[0]/'traces.jsonl';trace.write_text(trace.read_text()+'\n')
    with pytest.raises(c.CorrectnessStop):c.ledger(out,1)
    binding=json.loads((out/'binding.json').read_text());binding['protocol']={};c.atomic(out/'binding.json',binding)
    with pytest.raises(c.CorrectnessStop,match='implementation'):c.binding(out,True)


@pytest.mark.parametrize('code,exception',[(2,c.CorrectnessStop),(3,c.NotReady),(1,subprocess.CalledProcessError)])
def test_parent_preserves_child_stop_semantics(tmp_path,monkeypatch,code,exception):
    out,a=setup(tmp_path)
    monkeypatch.setattr(subprocess,'run',lambda *a,**kw:NS(returncode=code))
    run=args(out,through_month=1,score_gpu='0',gpu='0,1,2,3,4,5,6,7',sdpo=None,tokenizer=None,
        trainer_python=sys.executable,inference_python=sys.executable,fail_once=False)
    with pytest.raises(exception):a4_run.job(run)


def test_monthly_and_cumulative_pools_and_empty_boundary(tmp_path):
    out,a=setup(tmp_path);doc=c.binding(out,True)
    for n,month in enumerate(c.MONTHS[:2]):
        ii=c.items(doc,month)[:2]
        rows=[dict(id=i['id'],month=month,item=i,response='Original calculation.\nAnswer: 100.0') for i in ii]
        root=out/'ledger'/month;root.mkdir(parents=True)
        c.t.write_jsonl(root/'traces.jsonl',rows);c.freeze(root,['traces.jsonl'])
    direct,da=a4_train.schedule(out,doc,'direct',101,2,Tokenizer())
    cumulative,ca=a4_train.schedule(out,doc,'consolidation',101,2,Tokenizer())
    assert da['unique_traces']==2 and ca['unique_traces']==4
    assert all(r['id'].startswith('fixture-2-') for r in direct)
    assert any(r['id'].startswith('fixture-1-') for r in cumulative)
    root=out/'ledger'/c.MONTHS[1];(root/'traces.jsonl').write_text('');c.freeze(root,['traces.jsonl'])
    assert a4_train.schedule(out,doc,'direct',101,2,Tokenizer())[0]==[]
    assert a4_train.schedule(out,doc,'consolidation',101,2,Tokenizer())[1]['unique_traces']==2


@pytest.mark.parametrize('role',('trainer','inference'))
def test_different_python_environments_are_bound_separately(tmp_path,role):
    out,a=setup(tmp_path);doc=json.loads((out/'binding.json').read_text())
    doc['software'][role]={'changed':'environment'};c.atomic(out/'binding.json',doc)
    if role=='inference':
        with pytest.raises(c.CorrectnessStop,match='inference environment'):
            score(out,'base','frozen',0,0,'general-v1')
    else:
        with pytest.raises(c.CorrectnessStop,match='trainer environment'):
            a4_train.job(args(out,arm='direct',seed=101,month=1,fail_once=False))


def test_rendered_questions_and_teacher_request_are_gold_blind(tmp_path):
    sl,_,_=a4_sim.fixture(tmp_path);i=json.loads((sl/'batch-2026-01.json').read_text())[0]
    other=copy.deepcopy(i);other['qa'].update(answer='unavailable',exe_ans=999,program='private gold rationale')
    assert c.finqa.render_prompt(i)==c.finqa.render_prompt(other)
    from kit.sdft_objective import teacher_messages
    prompt=[dict(role='user',content=c.finqa.render_prompt(i))];response='Collector original reasoning.\nAnswer: 100.0'
    request=teacher_messages(prompt,response)
    rendered='\n'.join(r['content'] for r in request)
    assert response in rendered and 'private gold rationale' not in rendered


def test_reader_rejects_a_validly_hashed_foreign_scoring_epoch(tmp_path):
    out,a=setup(tmp_path);score(out,'base','frozen',0,0,'general-v1')
    root=c.score_root(out,'base','frozen',0,0,'general-v1','off')
    observed=json.loads((root/'identity.json').read_text());observed['physical']['uuid']='foreign GPU'
    c.atomic(root/'identity.json',observed)
    manifest=json.loads((root/'manifest.json').read_text());manifest['files']['identity.json']=c.s.file_hash(root/'identity.json')
    c.atomic(root/'manifest.json',manifest)
    with pytest.raises(c.CorrectnessStop,match='mixed scoring GPU'):a4_read.load(out,False)


def test_root_response_write_without_receipt_recovers(tmp_path):
    out,a=setup(tmp_path);root=c.score_root(out,'base','frozen',0,0,'general-v1','off')
    raw=root/'raw/responses.jsonl';raw.parent.mkdir(parents=True);raw.write_text('interrupted copy')
    score(out,'base','frozen',0,0,'general-v1')
    assert len(c.t.read_jsonl(raw))==30 and len(list(raw.parent.glob('responses.partial-*')))==1
    c.frozen_files(root)


def test_incomplete_canonical_generation_is_retained_and_retried(tmp_path,monkeypatch):
    out,a=setup(tmp_path);original=a4_score.freeze
    def fail(root,*args,**kw):
        if root.parent==out/'generation-cache':raise RuntimeError('interrupted generation commit')
        return original(root,*args,**kw)
    monkeypatch.setattr(a4_score,'freeze',fail)
    with pytest.raises(RuntimeError,match='generation commit'):score(out,'base','frozen',0,0,'general-v1')
    monkeypatch.setattr(a4_score,'freeze',original)
    score(out,'base','frozen',0,0,'general-v1')
    assert list((out/'generation-cache/partials').glob('*/responses.jsonl'))


@pytest.mark.parametrize('recipe',('F','D','F+replay','D+replay'))
def test_original_b_export_identities_required_and_pinned(recipe):
    files={}
    for route in ('standalone','sequence'):
        for seed in c.SEEDS:
            name=f'scores/{route}-{recipe}-s{seed}-k40-math_screen/identity.json'
            files[name]=json.dumps(dict(model=dict(model_dir=f'/durable/B/runs/{route}-{recipe}-s{seed}/hf-step40',
                model_file_hashes={'model.safetensors':'a'*64}),technical_synthetic=False)).encode()
    assert len(a4_bind.selected_exports(files,recipe,False))==10
    name=next(iter(files));original=files.pop(name)
    with pytest.raises(c.NotReady,match='identity pending'):a4_bind.selected_exports(files,recipe,False)
    changed=json.loads(original);changed['model']['model_dir']='/durable/B/resets/hf-reset';files[name]=json.dumps(changed).encode()
    with pytest.raises(c.CorrectnessStop,match='unreset final'):a4_bind.selected_exports(files,recipe,False)


def test_scoring_message_list_uses_message_renderer(tmp_path,monkeypatch):
    from kit import eval_bed, score_forgetting
    tok=Tokenizer()
    # general-v1 retains its original system/user messages and order.
    rows=[dict(prompt=[dict(role='system',content='Follow instructions.'),dict(role='user',content='Name a flower.')]),
          dict(prompt='Plain SEC question')]
    expected=[eval_bed.render_messages(tok,rows[0]['prompt']),score_forgetting.render(tok,rows[1]['prompt'])]
    assert a4_score.render_prompts(tok,rows)==expected
    out,a=setup(tmp_path)
    with pytest.raises(c.CorrectnessStop,match='unregistered'):score(out,'base','frozen',0,0,'finqa')
    assert all(r[4]!='finqa' for r in a4_run.score_plan())


@pytest.mark.parametrize('recipe',('D+replay','F+replay'))
def test_replay_schedule_nll_uses_training_prompt_conversion(tmp_path,recipe,monkeypatch):
    from kit import a4_nll
    out,a=setup(tmp_path,recipe);arrive(out)
    rows,audit=a4_train.schedule(out,c.binding(out,True),'direct',101,1,Tokenizer())
    assert audit['replay_exposures']==160
    assert all(isinstance(r['prompt'],str) for r in rows if r.get('p3b_replay'))
    converted,_=a4_train.training_rows(rows,Tokenizer(),recipe)
    expected={r['id']:r['prompt'] for r in converted};calls=[];original=c.t.tokenize_student
    def tokenize(tok,prompt,response):
        assert isinstance(prompt,list) and all(isinstance(m,dict) for m in prompt)
        calls.append(prompt)
        return original(tok,prompt,response)
    monkeypatch.setattr(c.t,'tokenize_student',tokenize)
    schedule=tmp_path/'schedule.jsonl';c.t.write_jsonl(schedule,rows)
    output=tmp_path/'nll.json'
    a4_nll.job(args(out,schedule=schedule,model='synthetic-base',receipt=None,output=output,gpu='0'))
    result=json.loads(output.read_text())
    assert result['exposures']==1280 and result['response_tokens']>1280
    assert len(calls)==result['unique']
    assert {json.dumps(p,sort_keys=True) for p in calls}=={json.dumps(p,sort_keys=True) for p in expected.values()}


def test_registration_hash_and_removed_recurrence_option(tmp_path,monkeypatch):
    sl,b,rp=a4_sim.fixture(tmp_path/'fixtures')
    res=tmp_path/'resources';res.mkdir()
    for name in ('slice.json','registration.md'):(res/name).write_bytes((c.RES/name).read_bytes())
    (res/'registration.md').write_text('tampered registration')
    monkeypatch.setattr(a4_bind,'RES',res)
    with pytest.raises(c.CorrectnessStop):a4_bind.job(a4_bind_args(tmp_path/'A4',sl,b,rp))
    parser=argparse.ArgumentParser();a4_bind.configure(parser)
    assert not any('--recurrence-freeze' in a.option_strings for a in parser._actions)
    source=c.KIT.parent/'docs/phase4/4a-registration-20261020.md'
    if source.exists():assert source.read_bytes()==(c.RES/'registration.md').read_bytes()
    assert c.s.file_hash(c.RES/'registration.md')==json.loads((c.RES/'slice.json').read_text())['registration_sha256']


def budget_doc(ceiling=600):
    return dict(technical_synthetic=False,allocation_gpus=8,ceiling_gpu_hours=.01,
        W='F' if ceiling == 600 else 'D', approved_ceiling_gpu_hours=ceiling,
        approval_reference=c.APPROVAL_REFERENCE,
        job_limits=dict(scoring=600,nll=600,training=1800),handoff_seconds=300)


original_slurm_times=c._slurm_times


@pytest.fixture
def held_job(monkeypatch):
    monkeypatch.setenv('SLURM_JOB_ID','101')
    monkeypatch.delenv('SLURM_JOB_START_TIME',raising=False)
    monkeypatch.delenv('SLURM_RESTART_COUNT',raising=False)
    monkeypatch.setattr(c,'_slurm_times',lambda job, allocation_start=None: {})


@pytest.mark.parametrize('kind',('scoring','nll','training'))
def test_budget_refuses_worst_case_before_attempt(tmp_path,kind,held_job):
    doc=budget_doc();doc['ceiling_gpu_hours']=1000
    doc['job_limits'][kind]=600*3600/8
    c.start_4a(tmp_path, False)
    with pytest.raises(c.CorrectnessStop,match='worst-case'):c.start_attempt(tmp_path,doc,kind,False)
    assert not list((tmp_path/'attempts').glob('*.json'))


def test_budget_charges_all_reserved_gpus_idle_and_handoff(tmp_path,monkeypatch,held_job):
    now=[1000.];monkeypatch.setattr(c.time,'time',lambda:now[0]);doc=budget_doc()
    monkeypatch.setenv('SLURM_JOB_START_TIME','940')
    c.start_4a(tmp_path, False)
    attempt=c.start_attempt(tmp_path,doc,'scoring',False)
    now[0]+=60;c.finish_attempt(tmp_path,attempt,completed=True)
    now[0]+=120
    assert c.budget(tmp_path,doc)==pytest.approx(8*180/3600)
    handoff=c.start_attempt(tmp_path,doc,'handoff',False)
    now[0]+=30;c.finish_attempt(tmp_path,handoff,completed=True)
    assert c.budget(tmp_path,doc)==pytest.approx(8*210/3600)


@pytest.mark.parametrize('source',('scontrol','job_end_marker'))
def test_24_hour_queue_gap_is_never_charged(tmp_path,monkeypatch,held_job,source):
    now=[1000.];monkeypatch.setattr(c.time,'time',lambda:now[0]);doc=budget_doc()
    c.start_4a(tmp_path, False)
    attempt=c.start_attempt(tmp_path,doc,'scoring',False)
    now[0]=1060;c.finish_attempt(tmp_path,attempt,completed=True)
    now[0]=1120
    if source=='job_end_marker':c.end_allocation(tmp_path,doc,'101',1120)
    else:monkeypatch.setattr(c,'_slurm_times',lambda job, allocation_start=None:dict(allocation_ended_at=1120))
    now[0]+=24*3600
    monkeypatch.setenv('SLURM_JOB_ID','102')
    monkeypatch.setenv('SLURM_JOB_START_TIME',str(now[0]))
    c.start_4a(tmp_path, False)
    assert c.budget(tmp_path,doc)==pytest.approx(8*120/3600)
    assert c.budget(tmp_path,doc)==pytest.approx(8*120/3600)
    allocation=json.loads((tmp_path/'allocations/101-r0.json').read_text())
    assert allocation['allocation_ended_at']==1120 and allocation['end_source']==source
    receipt=json.loads((tmp_path/'costs'/(attempt['id']+'.json')).read_text())
    assert receipt['slurm_job_id']=='101' and receipt['allocation_ended_at']==1120
    assert receipt['approved_ceiling_gpu_hours']==600


def test_unknown_previous_job_end_refuses_instead_of_charging_queue(tmp_path,monkeypatch,held_job):
    now=[1000.];monkeypatch.setattr(c.time,'time',lambda:now[0]);doc=budget_doc()
    c.start_4a(tmp_path, False)
    c.budget(tmp_path,doc)
    now[0]+=24*3600;monkeypatch.setenv('SLURM_JOB_ID','102')
    with pytest.raises(c.NotReady,match='end required.*101'):c.budget(tmp_path,doc)
    # Supply the previous job's own marker after a failed resume has recorded job 102.
    c.end_allocation(tmp_path,doc,'101',1120)
    assert c.budget(tmp_path,doc)==pytest.approx(8*120/3600)


def test_term_killed_job_without_cost_is_charged_its_held_time(tmp_path,monkeypatch,held_job):
    import os, signal
    root=Path(__file__).resolve().parents[1]
    script='''import json,sys,time
from pathlib import Path
from kit import a4_common as c
out=Path(sys.argv[1]);doc=json.loads(sys.argv[2])
c.start_4a(out,False)
c.start_attempt(out,doc,'training',False)
print('STARTED',flush=True)
time.sleep(60)
'''
    doc=budget_doc();proc=subprocess.Popen([sys.executable,'-u','-c',script,str(tmp_path),json.dumps(doc)],
        cwd=root,stdout=subprocess.PIPE,text=True)
    try:
        assert proc.stdout.readline().strip()=='STARTED'
        os.kill(proc.pid,signal.SIGTERM);assert proc.wait(timeout=10)==-signal.SIGTERM
    finally:
        if proc.poll() is None:proc.kill();proc.wait()
    assert not list((tmp_path/'costs').glob('*.json')) or all(
        p.name.startswith('allocation-idle-') for p in (tmp_path/'costs').glob('*.json'))
    allocation=json.loads((tmp_path/'allocations/101-r0.json').read_text())
    attempt=json.loads(next((tmp_path/'attempts').glob('*.json')).read_text())
    end=allocation['allocation_started_at']+120
    monkeypatch.setattr(c,'_slurm_times',lambda job, allocation_start=None:dict(allocation_ended_at=end))
    monkeypatch.setattr(c.time,'time',lambda:end+24*3600)
    monkeypatch.setenv('SLURM_JOB_ID','102')
    monkeypatch.setenv('SLURM_JOB_START_TIME',str(end+24*3600))
    c.start_4a(tmp_path, False)
    assert c.budget(tmp_path,doc)==pytest.approx(8*120/3600)
    assert c.budget(tmp_path,doc)==pytest.approx(8*120/3600)
    record=json.loads((tmp_path/'costs'/(attempt['id']+'.json')).read_text())
    assert record['reconciled'] and not record['completed']
    assert record['seconds']==pytest.approx(end-attempt['started_at'])
    assert record['allocation_ended_at']==end and record['slurm_job_id']=='101'


@pytest.mark.parametrize('approved',(600,800))
def test_approved_ceiling_controls_stop_and_not_formula(tmp_path,monkeypatch,held_job,approved):
    now=[1000.];monkeypatch.setattr(c.time,'time',lambda:now[0]);doc=budget_doc(approved)
    c.start_4a(tmp_path, False)
    attempt=c.start_attempt(tmp_path,doc,'scoring',False)
    now[0]+=60;c.finish_attempt(tmp_path,attempt,completed=True)
    assert c.budget(tmp_path,doc)>doc['ceiling_gpu_hours']
    now[0]=1000+approved*3600/8
    with pytest.raises(c.CorrectnessStop,match='ceiling exhausted'):c.budget(tmp_path,doc)
    assert sum(json.loads(p.read_text())['gpu_hours'] for p in (tmp_path/'costs').glob('*.json'))==approved


@pytest.mark.parametrize('value',(None,0,400,599,601,800,True,float('nan'),float('inf')))
def test_binding_refuses_missing_or_unapproved_ceiling(tmp_path,value):
    sl,b,rp=a4_sim.fixture(tmp_path/'fixtures')
    a=a4_bind_args(tmp_path/'A4',sl,b,rp)
    a.approved_ceiling=value
    with pytest.raises(c.CorrectnessStop,match='approved ceiling required'):a4_bind.job(a)
    assert not (a.out/'binding.json').exists()


@pytest.mark.parametrize('recipe,approved', [('F',600),('F+replay',600),('D',800),('D+replay',800)])
def test_binding_accepts_only_pinned_approval(tmp_path,recipe,approved):
    sl,b,rp=a4_sim.fixture(tmp_path/'fixtures',recipe)
    a=a4_bind_args(tmp_path/'A4',sl,b,rp);a.approved_ceiling=approved
    a4_bind.job(a);doc=c.binding(a.out,True)
    assert doc['approved_ceiling_gpu_hours']==approved
    assert doc['approval_reference']==c.APPROVAL_REFERENCE
    assert doc['approval_registration']==c.APPROVAL_REGISTRATION


@pytest.mark.parametrize('recipe', ('F','F+replay','D','D+replay'))
def test_binding_pins_approval_without_caller_inputs(tmp_path,recipe):
    sl,b,rp=a4_sim.fixture(tmp_path/'fixtures',recipe)
    a=a4_bind_args(tmp_path/'A4',sl,b,rp)
    del a.approved_ceiling;del a.approval_reference
    a4_bind.job(a)
    assert c.approved_ceiling(c.binding(a.out,True))==c.APPROVED_CEILINGS[recipe]


@pytest.mark.parametrize('reference', [None,'','   ',600,'different approval'])
def test_binding_requires_approval_reference(tmp_path,reference):
    sl,b,rp=a4_sim.fixture(tmp_path/'fixtures')
    a=a4_bind_args(tmp_path/'A4',sl,b,rp);a.approval_reference=reference
    with pytest.raises(c.CorrectnessStop,match='approval reference required'):a4_bind.job(a)
    assert not (a.out/'binding.json').exists()


@pytest.mark.parametrize('recipe', ['D','D+replay'])
def test_binding_refuses_below_d_minimum(tmp_path,recipe):
    sl,b,rp=a4_sim.fixture(tmp_path/'fixtures',recipe)
    a=a4_bind_args(tmp_path/'A4',sl,b,rp);a.approved_ceiling=799
    with pytest.raises(c.CorrectnessStop,match='pinned approved ceiling'):a4_bind.job(a)
    assert not (a.out/'binding.json').exists()


def test_80_hours_before_4a_are_excluded_and_marker_is_idempotent(tmp_path,monkeypatch,held_job):
    now=[81*3600.];monkeypatch.setattr(c.time,'time',lambda:now[0]);doc=budget_doc(600)
    monkeypatch.setenv('SLURM_JOB_START_TIME',str(3600))
    assert c.budget(tmp_path,doc)==0  # Return 1 alone never starts the 4A clock.
    marker=c.start_4a(tmp_path,False)
    assert marker['slurm_job_id']=='101' and marker['a4_started_at']==81*3600
    assert marker['a4_started_at_utc'].endswith('+00:00')
    assert c.budget(tmp_path,doc)==0
    now[0]+=3600
    assert c.start_4a(tmp_path,False)==marker
    assert c.budget(tmp_path,doc)==8
    idle=json.loads((tmp_path/'costs/allocation-idle-101-r0.json').read_text())
    assert idle['held_seconds']==3600 and idle['a4_started_at']==marker['a4_started_at']


def test_section_start_fsyncs_the_durable_copy_before_binding(tmp_path,monkeypatch,held_job):
    from kit import a4_allocation, message_handoff
    parent=tmp_path/'message';out=parent/'A4';durable=tmp_path/'durable'
    synced=set();original=c.os.fsync
    def fsync(fd):
        stat=c.os.fstat(fd);synced.add((stat.st_dev,stat.st_ino));return original(fd)
    monkeypatch.setattr(c.os,'fsync',fsync)
    a4_allocation.job(NS(out=out,start=True,parent_out=parent,durable=durable,standin=False))
    marker=message_handoff.committed(durable)/'tree/A4/4a-starts/101-r0.json'
    stat=marker.stat()
    assert (stat.st_dev,stat.st_ino) in synced
    assert marker.read_bytes()==(out/'4a-starts/101-r0.json').read_bytes()
    assert not (out/'binding.json').exists()


def test_job_without_marker_charges_zero_even_on_continuation(tmp_path,monkeypatch,held_job):
    now=[81*3600.];monkeypatch.setattr(c.time,'time',lambda:now[0]);doc=budget_doc(600)
    monkeypatch.setenv('SLURM_JOB_START_TIME',str(3600))
    assert c.budget(tmp_path,doc)==0
    with pytest.raises(c.CorrectnessStop,match='start marker required'):
        c.start_attempt(tmp_path,doc,'scoring',False)
    monkeypatch.setenv('SLURM_JOB_ID','102')
    c.start_4a(tmp_path,False)
    now[0]+=3600
    assert c.budget(tmp_path,doc)==8
    assert json.loads((tmp_path/'costs/allocation-idle-101-r0.json').read_text())['gpu_hours']==0


def test_continuation_uses_each_jobs_own_4a_marker(tmp_path,monkeypatch,held_job):
    now=[81*3600.];monkeypatch.setattr(c.time,'time',lambda:now[0]);doc=budget_doc(600)
    monkeypatch.setenv('SLURM_JOB_START_TIME',str(3600))
    c.start_4a(tmp_path,False);c.budget(tmp_path,doc)
    now[0]+=3600;c.end_allocation(tmp_path,doc,'101',now[0])
    now[0]+=24*3600;job2_start=now[0]
    monkeypatch.setenv('SLURM_JOB_ID','102');monkeypatch.setenv('SLURM_JOB_START_TIME',str(job2_start))
    now[0]+=2*3600  # Restore/identity/SKIP work before reaching the payload.
    c.start_4a(tmp_path,False)
    assert c.budget(tmp_path,doc)==24
    now[0]+=2*3600
    assert c.budget(tmp_path,doc)==40
    assert c.budget(tmp_path,doc)==40


@pytest.mark.parametrize('state', ('RUNNING','COMPLETING'))
def test_still_running_previous_job_is_pending(tmp_path,monkeypatch,held_job,state):
    now=[1000.];monkeypatch.setattr(c.time,'time',lambda:now[0]);doc=budget_doc()
    c.start_4a(tmp_path,False)
    attempt=c.start_attempt(tmp_path,doc,'scoring',False)
    now[0]+=60;c.finish_attempt(tmp_path,attempt,completed=True)
    now[0]+=24*3600
    monkeypatch.setenv('SLURM_JOB_ID','102');monkeypatch.setenv('SLURM_JOB_START_TIME',str(now[0]))
    from datetime import datetime
    stamp=lambda n:datetime.fromtimestamp(n).isoformat()
    monkeypatch.setattr(c,'_slurm_times',original_slurm_times)
    monkeypatch.setattr(c.subprocess,'run',lambda *a,**kw:NS(stdout=
        f'JobId={a[0][-1]} JobState={state} StartTime={stamp(1000)} EndTime={stamp(now[0]+3600)}'))
    c.start_4a(tmp_path,False)
    with pytest.raises(c.NotReady,match='end required.*101'):
        c.start_attempt(tmp_path,doc,'scoring',False)
    assert len(list((tmp_path/'attempts').glob('*.json')))==1


def test_4a_window_never_starts_before_allocation(tmp_path,monkeypatch,held_job):
    now=[1000.];monkeypatch.setattr(c.time,'time',lambda:now[0]);doc=budget_doc()
    c.start_4a(tmp_path,False)
    now[0]+=120;monkeypatch.setenv('SLURM_JOB_START_TIME','1060')
    assert c.budget(tmp_path,doc)==pytest.approx(8*60/3600)


def test_missing_slurm_start_still_charges_from_4a_entry(tmp_path,monkeypatch,held_job):
    now=[1000.];monkeypatch.setattr(c.time,'time',lambda:now[0]);doc=budget_doc()
    c.start_4a(tmp_path,False)
    now[0]+=120
    assert c.budget(tmp_path,doc)==pytest.approx(8*120/3600)


@pytest.mark.parametrize('source', ['scontrol','job_end_marker'])
def test_durable_marker_without_allocation_receipt_retains_charge(tmp_path,monkeypatch,held_job,source):
    now=[81*3600.];monkeypatch.setattr(c.time,'time',lambda:now[0]);doc=budget_doc(600)
    # A hard kill before the next snapshot leaves only the fsynced section marker.
    c.start_4a(tmp_path,False)
    assert not (tmp_path/'allocations/101-r0.json').exists()
    now[0]+=3600;end=now[0]
    now[0]+=24*3600;monkeypatch.setenv('SLURM_JOB_ID','102')
    monkeypatch.setenv('SLURM_JOB_START_TIME',str(now[0]))
    c.start_4a(tmp_path,False)
    if source=='scontrol':
        monkeypatch.setattr(c,'_slurm_times',lambda job, allocation_start=None:dict(allocation_started_at=3600,allocation_ended_at=end))
    else:c.end_allocation(tmp_path,doc,'101',end)
    assert c.budget(tmp_path,doc)==8
    assert c.budget(tmp_path,doc)==8


def test_binding_formula_is_forecast_even_above_approval(tmp_path):
    out,a=setup(tmp_path)
    (out/'binding.json').unlink();a.stage_gpu_hours=10
    a4_bind.job(a);doc=c.binding(out,True)
    assert doc['ceiling_gpu_hours']==850 and doc['approved_ceiling_gpu_hours']==600


def test_budget_requires_bound_approved_ceiling(tmp_path):
    doc=budget_doc();del doc['approved_ceiling_gpu_hours']
    with pytest.raises(c.CorrectnessStop,match='ceiling required'):c.budget(tmp_path,doc)


@pytest.mark.parametrize('state,ended',(('RUNNING',None),('COMPLETING',None),('COMPLETED',2000),('CANCELLED',2000)))
def test_scontrol_uses_only_actual_job_end(monkeypatch,state,ended):
    from datetime import datetime
    stamp=lambda n:datetime.fromtimestamp(n).isoformat()
    monkeypatch.setattr(c.subprocess,'run',lambda *a,**kw:NS(stdout=
        f'JobId=101 JobState={state} StartTime={stamp(1000)} EndTime={stamp(2000)}'))
    assert c._slurm_times('101',1000)==dict(allocation_started_at=1000,allocation_ended_at=ended)


@pytest.mark.parametrize('missing', ('error', 'empty'))
@pytest.mark.parametrize('state,ended', (('RUNNING',None),('COMPLETING',None),
    ('COMPLETED',2000),('CANCELLED by 1234',2000)))
def test_sacct_fallback_after_scontrol_forgets_job(monkeypatch,missing,state,ended):
    from datetime import datetime
    stamp=lambda n:datetime.fromtimestamp(n).isoformat()
    calls=[]
    def run(command,**kw):
        calls.append(command)
        if command[0]=='scontrol':
            if missing=='error':raise subprocess.CalledProcessError(1,command)
            return NS(stdout='')
        return NS(stdout=f'101|{stamp(1000)}|{stamp(2000)}|{state}\n')
    monkeypatch.setattr(c.subprocess,'run',run)
    assert c._slurm_times('101',1000)==dict(allocation_started_at=1000,
        allocation_ended_at=ended,source='sacct')
    assert calls==[['scontrol','show','job','-o','101'],
        ['sacct','-D','-j','101','-X','-n','-P','-o','JobIDRaw,Start,End,State']]


@pytest.mark.parametrize('missing', ('error', 'empty'))
def test_sacct_missing_history_leaves_manual_end_required(tmp_path,monkeypatch,held_job,missing):
    now=[1000.];monkeypatch.setattr(c.time,'time',lambda:now[0]);doc=budget_doc()
    c.start_4a(tmp_path,False);c.budget(tmp_path,doc)
    now[0]=10000;monkeypatch.setenv('SLURM_JOB_ID','102')
    monkeypatch.setenv('SLURM_JOB_START_TIME','10000');c.start_4a(tmp_path,False)
    monkeypatch.setattr(c,'_slurm_times',original_slurm_times)
    def run(command,**kw):
        if missing=='error':raise FileNotFoundError(command[0])
        return NS(stdout='')
    monkeypatch.setattr(c.subprocess,'run',run)
    with pytest.raises(c.NotReady,match='actual allocation end required.*101'):
        c.budget(tmp_path,doc)


def test_sacct_recovered_end_excludes_chained_job_queue_gap(tmp_path,monkeypatch,held_job):
    from datetime import datetime
    stamp=lambda n:datetime.fromtimestamp(n).isoformat()
    now=[1000.];monkeypatch.setattr(c.time,'time',lambda:now[0]);doc=budget_doc()
    c.start_4a(tmp_path,False);c.start_attempt(tmp_path,doc,'training',False)
    now[0]=10000;monkeypatch.setenv('SLURM_JOB_ID','102')
    monkeypatch.setenv('SLURM_JOB_START_TIME','10000');c.start_4a(tmp_path,False)
    monkeypatch.setattr(c,'_slurm_times',original_slurm_times)
    monkeypatch.setattr(c.subprocess,'run',lambda command,**kw:NS(
        stdout='' if command[0]=='scontrol' else f'101|{stamp(1000)}|{stamp(2000)}|COMPLETED\n'))
    assert c.budget(tmp_path,doc)==pytest.approx(8*1000/3600)
    allocation=json.loads((tmp_path/'allocations/101-r0.json').read_text())
    assert allocation['allocation_ended_at']==2000 and allocation['end_source']=='sacct'


def test_hard_duration_limit_terminates_child_tree(tmp_path):
    # Run in a child so alarm/signal handling is exercised exactly as on the partner.
    script='''from kit.a4_common import *
import sys
try:
    with job_deadline(.15):gpu_run([sys.executable,'-c','import time; time.sleep(60)'])
except CorrectnessStop:
    print('HARD STOP')
'''
    result=subprocess.run([sys.executable,'-c',script],capture_output=True,text=True,timeout=10)
    assert result.returncode==0 and result.stdout.strip()=='HARD STOP'


def test_v0_all_395_frozen_and_memory_prompts_match():
    evidence=c.KIT.parent/'docs/phase4/evidence/4a0-readouts-4a-primary-v0'
    source=c.KIT.parent/'data/p4_sec/slice-4a-primary'
    if not evidence.exists() or not source.exists():
        pytest.skip('full v0 evidence is checked in the research repository; absent from standalone public export')
    readout={r['id']:r for r in c.t.read_jsonl(evidence/'runs/readout-questions.jsonl')}
    pool=[];checked=admitted=0;tok=Tokenizer()
    for month in c.MONTHS:
        batch=json.loads((source/f'batch-{month}.json').read_text())
        memory_path=evidence/f'memory/batch-{month}.json'
        memory={i['id']:i for i in json.loads(memory_path.read_text())} if memory_path.exists() else {}
        assert memory or month==c.MONTHS[0]
        frozen={r['id']:r for r in c.t.read_jsonl(evidence/f'runs/frozen-{month}/responses.jsonl')}
        next_pool=[]
        for item in batch:
            i=item['id'];ref=readout[i]['arms']
            off=a4_memory.build(item,[],tok);on=a4_memory.build(item,pool,tok)
            assert off['prompt'].encode()==ref['frozen']['prompt'].encode()
            reference=ref.get('memory',ref['frozen'])
            assert on['prompt'].encode()==reference['prompt'].encode()
            assert on['case_ids']==reference['case_ids']
            if memory:
                assert on['prompt'].encode()==memory[i]['messages'][0]['content'].encode()
                assert on['case_ids']==memory[i]['memory']['case_ids']
            else:assert on['prompt']==off['prompt'] and not on['case_ids']
            if not a4_ledger.admission(item,frozen[i],tok):
                next_pool.append(dict(id=i,item=item,response=frozen[i]['response'],month=month));admitted+=1
            checked+=1
        pool+=next_pool
    assert checked==len(readout)==395
    assert admitted>0


def test_scoring_deadline_includes_response_commit(tmp_path,monkeypatch):
    import time
    out,a=setup(tmp_path);doc=json.loads((out/'binding.json').read_text())
    doc['job_limits']['scoring']=.1;c.atomic(out/'binding.json',doc)
    original=a4_score.freeze
    def slow_commit(root,*args,**kwargs):
        if root.parent==out/'generation-cache':time.sleep(.3)
        return original(root,*args,**kwargs)
    monkeypatch.setattr(a4_score,'freeze',slow_commit)
    with pytest.raises(c.CorrectnessStop,match='hard duration'):score(out,'base','frozen',0,0,'general-v1')
    attempt=json.loads(next((out/'attempts').glob('*.json')).read_text())
    cost=json.loads((out/'costs'/(attempt['id']+'.json')).read_text())
    assert not cost['completed']
    assert not (c.score_root(out,'base','frozen',0,0,'general-v1','off')/'manifest.json').exists()


@pytest.mark.parametrize('record_end', (False, True))
def test_requeue_has_fresh_marker_and_excludes_queue_gap(tmp_path,monkeypatch,held_job,record_end):
    now=[1000.];monkeypatch.setattr(c.time,'time',lambda:now[0]);doc=budget_doc()
    first=c.start_4a(tmp_path,False)
    attempt=c.start_attempt(tmp_path,doc,'scoring',False)
    now[0]=1060;c.finish_attempt(tmp_path,attempt,completed=True)
    if record_end:c.end_allocation(tmp_path,doc,'101',now[0])
    now[0]=10000  # Same job ID, a new held allocation after the queue gap.
    monkeypatch.setenv('SLURM_RESTART_COUNT','1')
    monkeypatch.setenv('SLURM_JOB_START_TIME',str(now[0]))
    second=c.start_4a(tmp_path,False)
    assert second['slurm_job_id']==first['slurm_job_id']=='101'
    assert first['slurm_restart_count']==0 and second['slurm_restart_count']==1
    assert first['a4_started_at']==1000 and second['a4_started_at']==10000
    second_attempt=c.start_attempt(tmp_path,doc,'scoring',False)
    now[0]+=120;c.finish_attempt(tmp_path,second_attempt,completed=True)
    assert c.budget(tmp_path,doc)==pytest.approx(8*(1060-1000)/3600+8*(10120-10000)/3600)
    assert c.budget(tmp_path,doc)==pytest.approx(8*180/3600)
    assert json.loads((tmp_path/'allocations/101-r0.json').read_text())['allocation_ended_at']==1060
    assert json.loads((tmp_path/'allocations/101-r1.json').read_text())['allocation_started_at']==10000
    assert json.loads((tmp_path/'costs'/f"{attempt['id']}.json").read_text())['slurm_restart_count']==0
    assert json.loads((tmp_path/'costs'/f"{second_attempt['id']}.json").read_text())['slurm_restart_count']==1



def test_requeued_run_after_4a_start_charges_from_its_allocation_start(tmp_path,monkeypatch,held_job):
    now=[1000.];monkeypatch.setattr(c.time,'time',lambda:now[0]);doc=budget_doc()
    c.start_4a(tmp_path,False)
    attempt=c.start_attempt(tmp_path,doc,'scoring',False)
    now[0]=1060;c.finish_attempt(tmp_path,attempt,completed=True);c.end_allocation(tmp_path,doc,'101',now[0])
    monkeypatch.setenv('SLURM_RESTART_COUNT','1');monkeypatch.setenv('SLURM_JOB_START_TIME','9900')
    now[0]=10000  # Restore and SKIP passes after the requeue serve 4A: 9900 -> 10000 is charged.
    second=c.start_4a(tmp_path,False)
    assert second['continuation']
    now[0]=10120;assert c.budget(tmp_path,doc)==pytest.approx(8*(60+220)/3600)


@pytest.mark.parametrize('next_start', (5500., 10000.))
@pytest.mark.parametrize('correction', ('job_end_marker', 'scontrol', 'sacct'))
def test_hard_kill_mid_training_requeue_bounds_and_corrects_end(tmp_path,monkeypatch,held_job,next_start,correction):
    from kit import a4_allocation
    now=[1000.];monkeypatch.setattr(c.time,'time',lambda:now[0]);doc=budget_doc()
    doc['job_limits']['training']=5400
    c.start_4a(tmp_path,False)
    attempt=c.start_attempt(tmp_path,doc,'training',False)
    # No finish_attempt: the node dies 70 minutes into a 90-minute attempt.
    actual_end=1000+70*60
    now[0]=next_start
    monkeypatch.setenv('SLURM_RESTART_COUNT','1')
    monkeypatch.setenv('SLURM_JOB_START_TIME',str(next_start))
    c.start_4a(tmp_path,False)
    expected_end=min(1000+5400,next_start)
    assert c.budget(tmp_path,doc)==pytest.approx(8*(expected_end-1000)/3600)
    path=tmp_path/'allocations/101-r0.json'
    allocation=json.loads(path.read_text())
    assert allocation['allocation_ended_at']==expected_end
    assert allocation['end_source']=='last_4a_record_before_requeue'
    # A terminal record belonging to the later run must not replace this end.
    monkeypatch.setattr(c,'_slurm_times',lambda job, allocation_start=None:dict(slurm_restart_count=1,
        allocation_started_at=next_start,allocation_ended_at=next_start))
    assert c.budget(tmp_path,doc)==pytest.approx(8*(expected_end-1000)/3600)
    if correction=='job_end_marker':
        monkeypatch.setattr(a4_allocation,'binding',lambda *a:doc)
        monkeypatch.setattr(sys,'argv',['a4_allocation.py','--out',str(tmp_path),
            '--job-id','101','--restart-count','0','--ended-at',str(actual_end)])
        assert a4_allocation.main()==0
    else:
        monkeypatch.setattr(c,'_slurm_times',lambda job, allocation_start=None:dict(allocation_started_at=1000,
            allocation_ended_at=actual_end,source=correction))
    assert c.budget(tmp_path,doc)==pytest.approx(8*(actual_end-1000)/3600)
    allocation=json.loads(path.read_text())
    assert allocation['allocation_ended_at']==actual_end and allocation['end_source']==correction
    cost=json.loads((tmp_path/'costs'/(attempt['id']+'.json')).read_text())
    assert cost['seconds']==70*60 and cost['allocation_ended_at']==actual_end
    with pytest.raises(c.CorrectnessStop,match='allocation end changed'):
        c.end_allocation(tmp_path,doc,'101',actual_end+1)


def test_requeue_guard_stops_on_missing_restart_count(tmp_path,monkeypatch,held_job,capsys):
    from kit import a4_allocation
    now=[1000.];monkeypatch.setattr(c.time,'time',lambda:now[0]);doc=budget_doc()
    monkeypatch.setenv('SLURM_JOB_START_TIME','1000')
    c.start_4a(tmp_path/'A4',False);c.budget(tmp_path/'A4',doc)
    now[0]=10000;monkeypatch.setenv('SLURM_JOB_START_TIME','10000')
    # The unchanged r0 key must not reuse its allocation start across a queue gap.
    with pytest.raises(c.CorrectnessStop,match='SLURM_JOB_START_TIME differs.*SLURM_RESTART_COUNT'):
        c.budget(tmp_path/'A4',doc)
    monkeypatch.setattr(sys,'argv',['a4_allocation.py','--out',str(tmp_path/'A4'),
        '--start','--parent-out',str(tmp_path),'--durable',str(tmp_path/'durable')])
    assert a4_allocation.main()==2
    assert 'SLURM_RESTART_COUNT' in capsys.readouterr().out
    assert json.loads((tmp_path/'A4/allocations/101-r0.json').read_text())['allocation_started_at']==1000


def test_job_start_before_first_restore_and_save(tmp_path,monkeypatch,held_job):
    from kit import message_handoff
    import shutil
    parent=tmp_path/'message';durable=tmp_path/'durable'
    def command(action):
        return subprocess.run([sys.executable,str(c.KIT/'message_handoff.py'),
            '--out',str(parent),'--durable',str(durable),'--action',action,
            *(['--move-aside'] if action=='restore' else [])],capture_output=True,text=True)
    assert command('job-start').returncode==0
    result=command('restore')
    assert result.returncode==0 and 'FIRST JOB' in result.stdout
    (parent/'evidence.json').write_text('{"first_job": true}\n')
    assert command('save').returncode==0
    shutil.rmtree(parent);monkeypatch.setenv('SLURM_JOB_ID','102')
    assert command('job-start').returncode==0
    assert command('restore').returncode==0
    assert (parent/'evidence.json').read_text()=='{"first_job": true}\n'
    assert message_handoff.committed(durable) is not None
    assert len(list((durable/'job-starts').glob('*.json')))==2


@pytest.mark.parametrize('slurm_start', (False, True))
def test_continuation_uses_durable_first_job_record_before_restore(tmp_path,monkeypatch,held_job,slurm_start):
    from kit import a4_allocation, message_handoff
    import shutil
    parent=tmp_path/'message';out=parent/'A4';durable=tmp_path/'durable'
    now=[1000.];monkeypatch.setattr(c.time,'time',lambda:now[0]);doc=budget_doc()
    c.start_4a(out,False);c.budget(out,doc)
    now[0]=1060;c.end_allocation(out,doc,'101',now[0]);message_handoff.save(parent,durable)
    now[0]=10000;monkeypatch.setenv('SLURM_JOB_ID','102')
    if slurm_start:monkeypatch.setenv('SLURM_JOB_START_TIME','9940')
    recorded=message_handoff.record_job_start(durable)
    assert recorded['allocation_started_at']==(9940 if slurm_start else 10000)
    now[0]+=60
    assert message_handoff.record_job_start(durable)==recorded
    shutil.rmtree(parent);message_handoff.restore(durable,parent,move_aside=True)
    now[0]+=60  # Restore, identity and SKIP work precede the payload marker.
    a4_allocation.job(NS(out=out,start=True,parent_out=parent,durable=durable,standin=False))
    assert c.budget(out,doc)==pytest.approx(8*(60+now[0]-recorded['allocation_started_at'])/3600)
    marker=json.loads((out/'4a-starts/102-r0.json').read_text())
    assert marker['a4_started_at']==now[0] and marker['continuation']
    assert marker['allocation_started_at']==recorded['allocation_started_at']
    record_path=durable/'job-starts/102-r0.json'
    assert json.loads(record_path.read_text())==recorded


@pytest.mark.parametrize('field,value', (
    ('approved_ceiling_gpu_hours',601),('approved_ceiling_gpu_hours',800),
    ('approved_ceiling_gpu_hours',None),('approved_ceiling_gpu_hours',True),
    ('approval_reference','different approval'),('approval_reference',None),('W','unknown')))
def test_runtime_rejects_hand_edited_binding_before_attempt(tmp_path,field,value):
    out,a=setup(tmp_path)
    doc=json.loads((out/'binding.json').read_text());doc[field]=value
    c.atomic(out/'binding.json',doc)
    # A JSON edit can preserve every implementation hash, even for direct entry points.
    for call in (lambda:c.binding(out,True),
                 lambda:a4_run.job(args(out,fail_once=False)),
                 lambda:a4_train.job(args(out,fail_once=False)),
                 lambda:c.start_attempt(out,doc,'training',True)):
        with pytest.raises(c.CorrectnessStop,match='pinned'):call()
    assert not list((out/'attempts').glob('*.json'))


@pytest.mark.parametrize('folder', ('allocations', '4a-starts', 'job-starts'))
@pytest.mark.parametrize('receipt_exists', (False, True))
def test_manual_end_refuses_overlap_before_saving(tmp_path,monkeypatch,held_job,capsys,folder,receipt_exists):
    from kit import a4_allocation
    out=tmp_path/'message/A4';durable=tmp_path/'durable';doc=budget_doc()
    now=[1000.];monkeypatch.setattr(c.time,'time',lambda:now[0])
    c.start_4a(out,False)
    if receipt_exists:c.budget(out,doc)
    now[0]=20000
    root=durable if folder=='job-starts' else out
    for job,restart,started in (('101',1,5500),('101',2,10000),('102',9,1500)):
        c.atomic(root/folder/f'{job}-r{restart}.json',dict(slurm_job_id=job,
            slurm_restart_count=restart,allocation_started_at=started))
    before={p:p.read_bytes() for p in tmp_path.rglob('*.json')}
    monkeypatch.setattr(a4_allocation,'binding',lambda *a:doc)
    command=['a4_allocation.py','--out',str(out),'--job-id','101',
        '--restart-count','0','--durable',str(durable),'--ended-at']
    monkeypatch.setattr(sys,'argv',command+['20000'])
    assert a4_allocation.main()==2
    assert 'allocation end exceeds a later run allocation start' in capsys.readouterr().out
    assert {p:p.read_bytes() for p in tmp_path.rglob('*.json')}==before
    # An end exactly at the earliest later allocation start is allowed.
    monkeypatch.setattr(sys,'argv',command+['5500'])
    assert a4_allocation.main()==0
    assert json.loads((out/'allocations/101-r0.json').read_text())['allocation_ended_at']==5500


@pytest.mark.parametrize('job,started', (('101',3000),('101_1',1000),('101.batch',1000)))
def test_sacct_wrong_run_or_job_leaves_end_pending(tmp_path,monkeypatch,held_job,job,started):
    from datetime import datetime
    stamp=lambda n:datetime.fromtimestamp(n).isoformat()
    doc=budget_doc();now=[1000.];monkeypatch.setattr(c.time,'time',lambda:now[0])
    c.start_4a(tmp_path,False);c.start_attempt(tmp_path,doc,'training',False)
    # Requeued run r1 fails before 4A entry, so there is no later 4A marker.
    now[0]=10000;monkeypatch.setenv('SLURM_JOB_ID','102')
    monkeypatch.setenv('SLURM_JOB_START_TIME','10000');c.start_4a(tmp_path,False)
    monkeypatch.setattr(c,'_slurm_times',original_slurm_times)
    monkeypatch.setattr(c.subprocess,'run',lambda command,**kw:NS(stdout=
        '' if command[0]=='scontrol' else f'{job}|{stamp(started)}|{stamp(3600)}|FAILED\n'))
    with pytest.raises(c.NotReady,match='actual allocation end required.*101-r0'):
        c.budget(tmp_path,doc)
    assert json.loads((tmp_path/'allocations/101-r0.json').read_text())['allocation_ended_at'] is None


@pytest.mark.parametrize('reverse', (False, True))
def test_sacct_matches_exact_job_and_recorded_start(monkeypatch,reverse):
    from datetime import datetime
    stamp=lambda n:datetime.fromtimestamp(n).isoformat()
    rows=[f'101|{stamp(1000)}|{stamp(2000)}|COMPLETED',
          f'101|{stamp(3000)}|{stamp(3600)}|FAILED',
          f'101_1|{stamp(1000)}|{stamp(4000)}|COMPLETED']
    if reverse:rows.reverse()
    monkeypatch.setattr(c.subprocess,'run',lambda command,**kw:NS(
        stdout='' if command[0]=='scontrol' else '\n'.join(rows)))
    assert c._slurm_times('101',1000)==dict(allocation_started_at=1000,
        allocation_ended_at=2000,source='sacct')
    assert c._slurm_times('101',900)=={}
    assert c._slurm_times('101')=={}


@pytest.mark.parametrize('end', ('Unknown','None',''))
def test_sacct_unknown_end_stays_unknown(monkeypatch,end):
    from datetime import datetime
    start=datetime.fromtimestamp(1000).isoformat()
    monkeypatch.setattr(c.subprocess,'run',lambda command,**kw:NS(
        stdout='' if command[0]=='scontrol' else f'101|{start}|{end}|FAILED\n'))
    assert c._slurm_times('101',1000)['allocation_ended_at'] is None


def test_manual_commands_record_all_missing_ends_before_accounting(tmp_path,monkeypatch,held_job):
    from kit import a4_allocation
    doc=budget_doc();now=[1000.];monkeypatch.setattr(c.time,'time',lambda:now[0])
    c.start_4a(tmp_path,False);c.start_attempt(tmp_path,doc,'scoring',False)
    for job,started in (('102',5000),('103',10000)):
        now[0]=started;monkeypatch.setenv('SLURM_JOB_ID',job)
        monkeypatch.setenv('SLURM_JOB_START_TIME',str(started));c.start_4a(tmp_path,False)
        with pytest.raises(c.NotReady,match='actual allocation end required.*101'):
            c.start_attempt(tmp_path,doc,'scoring',False)
    monkeypatch.setattr(a4_allocation,'binding',lambda *a:doc)
    for job,ended in (('101',1060),('102',5060)):
        monkeypatch.setattr(sys,'argv',['a4_allocation.py','--out',str(tmp_path),
            '--job-id',job,'--restart-count','0','--ended-at',str(ended)])
        assert a4_allocation.main()==0
        assert json.loads((tmp_path/f'allocations/{job}-r0.json').read_text())['allocation_ended_at']==ended
    assert c.budget(tmp_path,doc)==pytest.approx(8*120/3600)


@pytest.mark.parametrize('action', ('start_4a','job-start'))
@pytest.mark.parametrize('existing', (False, True))
@pytest.mark.parametrize('env_restart,scheduler_restart', ((None,1),('0',1),('1',2),('2',1)))
def test_scontrol_restart_mismatch_stops_before_new_or_stale_record(tmp_path,monkeypatch,held_job,capsys,action,existing,env_restart,scheduler_restart):
    from kit import a4_allocation, message_handoff
    now=[1000.];monkeypatch.setattr(c.time,'time',lambda:now[0])
    if env_restart is not None:monkeypatch.setenv('SLURM_RESTART_COUNT',env_restart)
    out=tmp_path/'message/A4';durable=tmp_path/'durable'
    if existing:
        if action=='start_4a':c.start_4a(out,False)
        else:message_handoff.record_job_start(durable)
    before={p:p.read_bytes() for p in tmp_path.rglob('*.json')}
    now[0]=10000  # No SLURM_JOB_START_TIME: only scontrol exposes the requeue.
    monkeypatch.setattr(c,'_slurm_times',original_slurm_times)
    monkeypatch.setattr(c.subprocess,'run',lambda *a,**kw:NS(stdout=
        f'JobId=101 JobState=RUNNING Restarts={scheduler_restart}'))
    if action=='start_4a':
        monkeypatch.setattr(sys,'argv',['a4_allocation.py','--out',str(out),'--start',
            '--parent-out',str(out.parent),'--durable',str(durable)])
        result=a4_allocation.main()
    else:
        monkeypatch.setattr(sys,'argv',['message_handoff.py','--out',str(out.parent),
            '--durable',str(durable),'--action','job-start'])
        result=message_handoff.main()
    assert result==2
    assert 'SLURM_RESTART_COUNT differs from scontrol Restarts' in capsys.readouterr().out
    assert {p:p.read_bytes() for p in tmp_path.rglob('*.json')}==before


@pytest.mark.parametrize('action', ('start_4a','job-start'))
@pytest.mark.parametrize('scheduler_restart', (None,1))
def test_matching_or_unavailable_scontrol_restart_allows_record(tmp_path,monkeypatch,held_job,action,scheduler_restart):
    from kit import message_handoff
    monkeypatch.setenv('SLURM_RESTART_COUNT','1')
    monkeypatch.setattr(c,'_slurm_times',original_slurm_times)
    restarts='' if scheduler_restart is None else f' Restarts={scheduler_restart}'
    monkeypatch.setattr(c.subprocess,'run',lambda *a,**kw:NS(stdout=f'JobId=101 JobState=RUNNING{restarts}'))
    record=c.start_4a(tmp_path,False) if action=='start_4a' else message_handoff.record_job_start(tmp_path)
    assert record['slurm_restart_count']==1


def test_job_start_refuses_stale_allocation_start(tmp_path,monkeypatch,held_job):
    from kit import message_handoff
    monkeypatch.setenv('SLURM_JOB_START_TIME','1000')
    record=message_handoff.record_job_start(tmp_path)
    monkeypatch.setenv('SLURM_JOB_START_TIME','10000')
    with pytest.raises(c.CorrectnessStop,match='SLURM_JOB_START_TIME differs'):
        message_handoff.record_job_start(tmp_path)
    assert json.loads((tmp_path/'job-starts/101-r0.json').read_text())==record
