"""Offline C protocol tests and a full plain-job technical rehearsal, all rows."""
import copy
import csv
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS
import pytest
from kit import c_common as c, c_bind, c_targets, c_train, c_score, c_read, c_collect
from kit.c_sim import Tokenizer, b_rows
from kit.p3b_read import analyse as b_analyse
from kit.p3b_score import write_csv
from kit.m2_seed_policy import SEEDS as RETURN1_SEEDS


def args(out, **kw): return NS(out=out,standin=True,**kw)


def prepare_b(root, winner='F+replay'):
    root.mkdir(parents=True); source=root/'b'; source.mkdir()
    rows=b_rows(winner); write_csv(source/'scores.csv',rows)
    result=dict(b_analyse(rows,20),technical_synthetic=True)
    c.atomic(root/'reading.json',result)
    starts={'base':dict(model='synthetic-base')}
    for recipe in c.RECIPES:
        for seed in c.SEEDS: starts[f'{recipe}:{seed}']=dict(model=f'synthetic-{recipe}-{seed}')
    c.atomic(root/'starts.json',starts)
    return root/'reading.json', source, root/'starts.json'


def bind(out, source, winner='F+replay'):
    reading, bsource, starts=prepare_b(source,winner)
    a=args(out,b_reading=reading,b_source=bsource,checkpoints=starts)
    c_bind.job(a)
    return a


def targets(out, **kw):
    a=args(out,source=None,tokenizer=None,model=None,gpu='0,1',prepare_only=False,standin_zero=False)
    for k,v in kw.items(): setattr(a,k,v)
    c_targets.job(a); return a


def score(out, arm, seed, step, panel, gpu='0'):
    a=args(out,arm=arm,seed=seed,step=step,panel_set=panel,panel_root=None,gpu=gpu)
    c_score.job(a); return a


def train(out, recipe, route, seed, replay):
    a=args(out,recipe=recipe,route=route,seed=seed,replay=replay,tokenizer=None,sdpo=None,trainer_python=sys.executable,gpu='0,1,2,3,4,5,6,7',model=None)
    c_train.job(a); return a


def replay_fixture(root):
    root.mkdir(); c.t.write_jsonl(root/'replay.jsonl',[dict(id=f'generic-{i}',prompt=f'Repeat the unrelated generic phrase number {i}.',response='A frozen generic reply.') for i in range(12)])
    c.atomic(root/'manifest.json',c.files_manifest(root,['replay.jsonl'],technical_synthetic=True))
    return root


@pytest.mark.parametrize('winner',c.RECIPES)
def test_bind_each_recipe_and_no_duplicate_f(tmp_path,winner):
    out=tmp_path/'out'; a=bind(out,tmp_path/'evidence',winner)
    doc=c.binding(out,True)
    assert doc['W']==winner
    assert len(c.arms(doc))==(2 if winner=='F' else 3)
    assert c.start(doc,'answer-only','sequence',101)['model']=='synthetic-F-101'
    assert c.start(doc,winner,'sequence',105)['model']==f'synthetic-{winner[0]}-105'
    c_bind.job(a)  # immutable replay of the same binding


def test_pending_wrong_reading_and_exact_ties(tmp_path):
    a=args(tmp_path/'out',b_reading=tmp_path/'missing',b_source=tmp_path/'missing',checkpoints=tmp_path/'missing')
    with pytest.raises(c.NotReady): c_bind.job(a)
    a=bind(tmp_path/'ready',tmp_path/'evidence','F')
    reading=json.loads(a.b_reading.read_text()); reading['curves'][0]['final']+=1; c.atomic(a.b_reading,reading)
    with pytest.raises(c.CorrectnessStop,match='differs'): c_bind.job(a)
    reading=b_analyse(b_rows('none'),20)
    assert c_bind.choose(reading)['W']=='F'
    for seq in reading['sequence']: seq['passes']=False
    assert c_bind.choose(reading)['provisional']
    reading['curves']=[r for r in reading['curves'] if r['seed']!=105]
    with pytest.raises(c.NotReady): c_bind.choose(reading)


def test_targets_protocol_resumption_hashes_and_zero(tmp_path):
    out=tmp_path/'out'; a=targets(out,prepare_only=True)
    assert not (out/'targets/manifest.json').exists()
    a.prepare_only=False; c_targets.job(a)
    manifest=c.frozen_files(out/'targets')
    assert manifest['valid']==24
    pool=c.t.read_jsonl(out/'targets/pool.jsonl')
    assert len(pool)==24
    raw=list((out/'targets/teacher/raw').glob('*.json'))
    assert len(raw)==30
    before={p.name:p.read_bytes() for p in raw}; c_targets.job(a)
    assert before=={p.name:p.read_bytes() for p in raw}
    for seed in c.SEEDS: assert len(c.t.read_jsonl(out/f'targets/schedule-{seed}.jsonl'))==1280
    a.standin=False
    with pytest.raises(c.CorrectnessStop): c_targets.job(a)
    with pytest.raises(c.NotReady,match='zero valid'): targets(tmp_path/'zero',standin_zero=True)
    assert len(list((tmp_path/'zero/targets/teacher/raw').glob('*.json')))==96
    assert not (tmp_path/'zero/targets/manifest.json').exists()
    schedule=out/'targets/schedule-101.jsonl'; schedule.write_text('changed')
    with pytest.raises(c.CorrectnessStop): c.frozen_files(out/'targets')


@pytest.mark.parametrize('text,finish,reason',[
    ('Answer: entailment','stop','incomplete_worked_reasoning'),
    ('Here is the sufficiently detailed legal reasoning for this legal example.\nAnswer: contradiction','stop','label_mismatch_or_invalid'),
    ('Here is the sufficiently detailed legal reasoning for this legal example.\n**Answer: entailment**','stop','strict_terminal_line_required'),
    ('Here is the sufficiently detailed legal reasoning for this legal example.\nAnswer: entailment','length','unfinished'),
    ('<think>Here is the sufficiently detailed legal reasoning for this legal example.</think>\nAnswer: entailment','stop','incomplete_worked_reasoning'),
    ('word '*9000+'\nAnswer: entailment','stop','response_over_2048_including_eos'),
])
def test_verifier_rejects(text,finish,reason):
    row=c_targets.synthetic_rows()['train'][0]
    checked=c_targets.verify(row,dict(text=text,finish_reason=finish,attempt=1),Tokenizer())
    assert not checked['verified'] and reason in checked['reasons']


def test_gold_blind_request_and_eos_mask():
    from kit.c_data import encode
    row=c_targets.synthetic_rows()['train'][0]; changed=dict(row,gold='contradiction')
    assert c_targets.messages(row)==c_targets.messages(changed)
    tok=Tokenizer(); msg=[dict(role='user',content='unaided')]; response='Answer: entailment'
    encoded=encode(tok,msg,response); pids,tids=c.t.tokenize_student(tok,msg,response)
    assert encoded['loss_mask'][len(pids)-1]==0
    assert encoded['loss_mask'][len(pids):len(pids)+len(tids)]==[1]*len(tids)
    assert sum(encoded['loss_mask'])==len(tids)
    with pytest.raises(c.CorrectnessStop): encode(tok,msg,'x'*9000)
    with pytest.raises(c.CorrectnessStop): encode(tok,[dict(role='user',content='x'*45000)],response)


def test_document_round_robin_exclusions_and_caps():
    splits=c_targets.synthetic_rows()
    rows=splits['train']; more=[]
    for r in rows[:3]:
        for j in range(6): more.append(dict(r,id=r['id']+f'-{j}',content=dict(r['content'],hypothesis=f'unique {j}')))
    splits['train']=more
    selected,excluded=c_targets.select_pool(splits,[],Tokenizer())
    counts={g:sum(r['group']==g for r in selected) for g in {r['group'] for r in selected}}
    assert len(selected)==12 and set(counts.values())=={4}
    assert len({r['group'] for r in selected[:3]})==3
    splits['dev']=[more[0]]
    selected,excluded=c_targets.select_pool(splits,[],Tokenizer())
    assert len(selected)==8 and any(x['reason'].startswith('cross_split') for x in excluded)
    huge=dict(rows[3],prompt='x'*45000); splits['train']=[huge]
    assert c_targets.select_pool(splits,[],Tokenizer())[1][0]['reason']=='input_over_10240'
    assert c_targets.select_pool(dict(train=[rows[0]],dev=[],test=[]),[rows[0]],Tokenizer())[0]==[]


def test_bounded_journal_resume_after_interruption(tmp_path,monkeypatch):
    original=c_targets.verify; count=0
    def crash(*a):
        nonlocal count
        count+=1
        if count==2: raise RuntimeError('interrupted verification')
        return original(*a)
    monkeypatch.setattr(c_targets,'verify',crash)
    out=tmp_path/'out'
    with pytest.raises(RuntimeError): targets(out)
    raw={p.name:p.read_bytes() for p in (out/'targets/teacher/raw').glob('*.json')}
    assert len(raw)==24
    monkeypatch.setattr(c_targets,'verify',original); targets(out)
    assert all((out/'targets/teacher/raw'/n).read_bytes()==b for n,b in raw.items())
    assert len(list((out/'targets/teacher/raw').glob('*.json')))==30


def test_parser_freeze_and_missing_invalid_outputs():
    assert c.s.file_hash(c.C/'legal_headroom.py')==c.protocol()['files']['legal_headroom.py']
    items=c.panels()['test']; raw=[dict(id=items[0]['id'],response='**Answer: '+items[0]['gold']+'**',output_tokens=12,finish_reason='length')]
    rows=c_score.legal_scores(items,raw,'standalone-F',101,0,'test')
    assert len(rows)==100 and rows[0]['tolerant_correct']==1 and rows[0]['strict_correct']==0 and rows[0]['finished']==0
    assert sum(r['tolerant_correct'] for r in rows)==1
    with pytest.raises(c.CorrectnessStop): c_score.legal_scores(items,raw+raw,'standalone-F',101,0,'test')
    assert c.legal.parse_answer_tolerant('contract_nli','Answer: entailment\nAnswer: invalid') is None


def test_context_adapter_and_training_command(tmp_path):
    from kit.c_sdft_entry import legacy_config
    config=dict(data=dict(max_prompt_length=10240),actor_rollout_ref=dict(actor=dict(self_distillation=dict(max_reprompt_len=14336))))
    legacy=legacy_config(config)
    assert config['data']['max_prompt_length']==10240 and legacy['data']['max_prompt_length']==2048
    config['data']['max_prompt_length']=4096
    with pytest.raises(c.CorrectnessStop): legacy_config(config)
    a=NS(sdpo=tmp_path,seed=101,gpu='0,1,2,3,4,5,6,7',trainer_python=sys.executable,tokenizer='base')
    for recipe in (*c.RECIPES,'answer-only'):
        cmd,resolve,env=c_train.command(a,recipe,'model',tmp_path/'data',tmp_path/'run')
        assert env['P3B_REPLAY']==('1' if '+replay' in recipe else '0')
        assert env['P3B_SFT']=='1' and env['KIT_SFT_ARM_F']=='0'
        assert 'data.max_prompt_length=10240' in [x.lstrip('+') for x in cmd]
        if recipe.startswith('D'):
            assert any('max_reprompt_len=14336' in x for x in cmd)
            assert any('c_reward.py' in x for x in cmd)
        else:
            assert 'data.max_length=12288' in cmd and 'data.custom_cls.name=CSFTDataset' in cmd
        assert '--resolve' in resolve


@pytest.mark.parametrize('winner',c.RECIPES)
def test_full_standin_every_core_row_and_roundtrip(tmp_path,winner):
    """29 Oct decisions §registered amendment: return-1 rule §Statistics/Decision map (C), 101–103.

    docs/phase3/three-seed-rule-return1-20261028.md, applied 29 Oct in
    return1-three-seed-decisions-20261029.md §Three-seed registered decisions.
    B selection retains all five original seeds; only C rows/counts change here.
    """
    out=tmp_path/'out'; bind(out,tmp_path/'b-evidence',winner); targets(out)
    replay=replay_fixture(tmp_path/'replay'); doc=c.binding(out,True)
    for p in ('panel','test','finqa','general-v1'): score(out,'base',101,0,p)
    for recipe in c.arms(doc):
        for route in ('standalone','sequence'):
            for seed in RETURN1_SEEDS:
                arm=route+'-'+recipe
                for p in ('panel','test','finqa','general-v1'): score(out,arm,seed,0,p)
                a=train(out,recipe,route,seed,replay); c_train.job(a)
                for step in (10,20,30,40):
                    for p in ('panel','test'): score(out,arm,seed,step,p)
                for step in (20,40):
                    for p in ('finqa','general-v1'): score(out,arm,seed,step,p)
    assert len(list((out/'runs').glob('*/complete.json')))==6*len(c.arms(doc))
    read=tmp_path/'read'; c_read.job(args(read,source=out,bootstrap=20))
    result=json.loads((read/'reading.json').read_text())
    assert result['complete'] and result['goal1']['status']=='PASS'
    assert len(result['seed_tables'])==24*len(c.arms(doc)) and len(result['retention'])==len(c.arms(doc))
    assert result['cost']['recorded_gpu_hours']==0
    assert len(list((out/'scores').glob('*/manifest.json')))==96*len(c.arms(doc))+4
    # Copy manager summaries to the partner tree before collection.
    import shutil
    shutil.copytree(read,out/'reading')
    archive=tmp_path/'return.tar.gz'; c_collect.job(args(out,archive=archive)); c_collect.job(args(out,archive=archive))
    assert archive.stat().st_size<28*1024*1024
    returned=tmp_path/'returned'; c_read.job(args(returned,source=archive,bootstrap=20))
    assert json.loads((returned/'reading.json').read_text())==result
    import tarfile
    with tarfile.open(archive) as tar:
        names=tar.getnames()
        assert any('bulk-evidence.json' in n for n in names)
        assert not any('/raw/' in n or '/data/' in n or 'schedule-' in n for n in names)
    c.atomic(tmp_path/'dry-run-summary.json',dict(stages=6*len(c.arms(doc)),score_jobs=96*len(c.arms(doc))+4,seed_table_rows=24*len(c.arms(doc)),goal1='technical PASS',archive_bytes=archive.stat().st_size,gpu_hours=0))


def test_train_scoring_and_collection_failure_paths(tmp_path):
    out=tmp_path/'out'
    with pytest.raises(c.NotReady): score(out,'base',101,0,'test')
    bind(out,tmp_path/'b-evidence','D'); targets(out)
    with pytest.raises(c.NotReady,match='score incoming'): train(out,'D','standalone',101,None)
    with pytest.raises(c.CorrectnessStop,match='one physical'): score(out,'base',101,0,'test','0,1')
    with pytest.raises(c.CorrectnessStop,match='0/20/40'): score(out,'base',101,10,'finqa')
    with pytest.raises(c.CorrectnessStop): score(out,'sequence-F+replay',101,0,'test')
    with pytest.raises(c.NotReady): score(out,'standalone-D',101,40,'test')
    for p in ('panel','test','finqa','general-v1'): score(out,'standalone-D',101,0,p)
    train(out,'D','standalone',101,None)
    a=score(out,'standalone-D',101,40,'test'); c_score.job(a)
    with pytest.raises(c.CorrectnessStop): score(out,'standalone-D',101,40,'test','1')
    bad=tmp_path/'bad'; c.atomic(bad/'binding.json',json.loads((out/'binding.json').read_text()))
    with pytest.raises(c.NotReady): c_collect.job(args(bad,archive=tmp_path/'nothing.tar.gz'))


def test_goal_mean_alone_is_insufficient_and_question_pair_cluster(tmp_path):
    """29 Oct return-1 amendment §Decision map/C Goal 1: same 69.7 bar, three-seed t bound."""
    out=tmp_path/'out'; bind(out,tmp_path/'b-evidence','F'); doc=c.binding(out,True)
    items=c.panels()['test']; rows=[]
    for seed,final in zip(RETURN1_SEEDS,(60,80,94)):
        for step in (0,40):
            raw=[dict(id=r['id'],response='Answer: '+(r['gold'] if i<(42 if step==0 else final) else 'invalid'),output_tokens=10,finish_reason='stop') for i,r in enumerate(items)]
            rows+=c_score.legal_scores(items,raw,'standalone-F',seed,step,'test')
    result=c_read.analyse(rows,doc,20)
    assert result['goal1']['final_seed_bound']['mean']==78
    assert result['goal1']['final_seed_bound']['lower']<69.7
    assert result['goal1']['status']=='INCONCLUSIVE'
    pair=result['question_pairs'][0]
    assert pair['clusters']==73 and pair['resampling']=='document'
    assert not result['complete']
    with pytest.raises(c.CorrectnessStop): c_read.analyse(rows+rows[:1],doc,20)
    assert c_read.status(dict(mean=70,lower=69.7,upper=72),69.7)=='PASS'
    assert c_read.status(dict(mean=69,lower=65,upper=69.6),69.7)=='INCONCLUSIVE'


def test_cli_failure_exit_codes_and_help(tmp_path):
    for name in ('bind','targets','train','score','read','collect'):
        done=subprocess.run([sys.executable,str(c.KIT/f'c_{name}.py'),'--help'],capture_output=True,text=True)
        assert done.returncode==0
    done=subprocess.run([sys.executable,str(c.KIT/'c_score.py'),'--out',str(tmp_path/'pending'),'--standin','--arm','base','--seed','101','--step','0','--panel-set','test'],capture_output=True,text=True)
    assert done.returncode==3 and 'NOT READY' in done.stdout


def test_real_teacher_engine_configuration_and_gold_blind_requests(monkeypatch):
    import types
    seen={}
    class Engine:
        def __init__(self,**kw): seen['engine']=kw
        def get_tokenizer(self): return Tokenizer()
        def generate(self,prompts,params,**kw):
            seen['prompts']=prompts; seen['params']=params
            return [NS(outputs=[NS(text='An answer',finish_reason='stop',stop_reason=None,token_ids=[1,2])]) for p in prompts]
    monkeypatch.setitem(sys.modules,'vllm',NS(__version__='0.18.0',LLM=Engine,SamplingParams=lambda **kw:kw))
    gen=c_targets.Teacher('pinned-local-teacher')
    requests=[dict(id='x',attempt=2,seed=27102,temperature=.3,top_p=1,messages=[dict(role='user',content='gold-blind agreement')])]
    result=gen.generate_many(requests)
    assert seen['engine']['tensor_parallel_size']==2 and seen['engine']['max_model_len']==16384
    assert seen['engine']['gdn_prefill_backend']=='triton' and seen['engine']['enforce_eager']
    assert seen['params'][0]==dict(seed=27102,temperature=.3,top_p=1,top_k=-1,repetition_penalty=1,n=1,max_tokens=2560)
    assert result[0]['teacher_tokens']==2
    monkeypatch.setitem(sys.modules,'vllm',NS(__version__='0.17.0'))
    with pytest.raises(c.CorrectnessStop,match='0.18.0'): c_targets.Teacher('pinned-local-teacher')


def test_ray_adapter_preserves_other_checks_and_restores_them(monkeypatch):
    import importlib.util
    import kit
    from kit import sdft_objective, v4_contract
    calls=[]
    def check_obj(config):
        assert config['data']['max_prompt_length']==2048
        assert config['actor_rollout_ref']['actor']['self_distillation']['max_reprompt_len']==6144
        assert config['other_setting']=='must survive'
        calls.append('objective')
    def check_contract(config,*a,**kw): calls.append('contract')
    monkeypatch.setattr(sdft_objective,'check_trainer_config',check_obj)
    monkeypatch.setattr(v4_contract,'check_resolved',check_contract)
    class Parent:
        def run(self,config):
            sdft_objective.check_trainer_config(config)
            v4_contract.check_resolved(config,'D',profile_name='scientific')
            raise RuntimeError('stand-in Ray runner failure')
    monkeypatch.setattr(kit,'sdft_verl',NS(SdftTaskRunner=Parent),raising=False)
    spec=importlib.util.spec_from_file_location('c_test_runner',c.KIT/'c_sdft_runner.py')
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    config=dict(data=dict(max_prompt_length=10240),actor_rollout_ref=dict(actor=dict(self_distillation=dict(max_reprompt_len=14336))),other_setting='must survive')
    with pytest.raises(RuntimeError,match='Ray runner'): module.CTaskRunner().run(config)
    assert calls==['objective','contract'] and config['data']['max_prompt_length']==10240
    assert sdft_objective.check_trainer_config is check_obj and v4_contract.check_resolved is check_contract


def test_full_documents_longer_than_b_caps_are_trainable():
    row=dict(id='long',prompt=[dict(role='user',content='x'*30000)],response='Complete sufficiently detailed legal reasoning is given for this example.\nAnswer: entailment',gold='entailment')
    for recipe in c.RECIPES:
        converted,tokens=c_train.training_rows([row],Tokenizer(),recipe)
        assert 4096<tokens['input']<10240
        assert converted[0]['data_source']=='contract_nli' and not converted[0]['extra_info']['p3b_replay']
    from kit.c_reward import compute_score
    assert compute_score('contract_nli','Answer: entailment','entailment')['score']==1
    assert compute_score('p3b_replay','anything','',dict(p3b_replay=True))['score']==0


def test_goal_needs_three_positive_seeds_and_undefined_cost(tmp_path):
    """29 Oct return-1 amendment §Decision map/C Goal 1: 3/3 positive; undefined costs unchanged."""
    out=tmp_path/'out'; bind(out,tmp_path/'b','F'); doc=c.binding(out,True); items=c.panels()['test']
    rows=[]
    for seed,start in zip(RETURN1_SEEDS,(42,42,95)):
        for step,num in ((0,start),(40,90)):
            raw=[dict(id=r['id'],response='Answer: '+(r['gold'] if i<num else 'invalid'),output_tokens=100,finish_reason='stop') for i,r in enumerate(items)]
            rows.extend(c_score.legal_scores(items,raw,'standalone-F',seed,step,'test'))
    result=c_read.analyse(rows,doc,10)
    assert result['goal1']['final_seed_bound']['lower']==90
    assert result['goal1']['positive_seeds']==2 and result['goal1']['status']=='INCONCLUSIVE'
    for r in rows: r['correct']=r['tolerant_correct']=r['strict_correct']=0
    result=c_read.analyse(rows,doc,10)
    assert all(r['tokens_per_correct'] is None and r['serving_status']=='UNDEFINED' for r in result['summaries'])
    assert c_read.analyse(rows[:-100],doc,10)['goal1']['status']=='PENDING'


def test_w_ranking_uses_joint_then_margin_then_math_then_cost():
    reading=b_analyse(b_rows('none'),20)
    for r in reading['sequence']:
        if r['recipe']=='F': r['passes']=False
    assert c_bind.choose(reading)['W']=='D'
    for r in reading['sequence']:
        if r['recipe']=='D': r['stage2_retention']['finqa']['lower']=-2
    assert c_bind.choose(reading)['W']=='D+replay'
    for r in reading['curves']:
        if r['recipe']=='F+replay' and r['route']=='sequence': r['final']+=1
    assert c_bind.choose(reading)['W']=='F+replay'
    for r in reading['curves']:
        if r['recipe']=='F+replay' and r['route']=='sequence': r['final']-=1
    for r in reading['summaries']:
        if r['arm']=='sequence-D+replay' and r['step']==40 and r['panel']=='finqa': r['serving_ratio']=100
    assert c_bind.choose(reading)['W']=='F+replay'
    for r in reading['summaries']:
        if r['arm']=='sequence-F+replay' and r['step']==40 and r['panel']=='finqa': r['serving_ratio']=None
    assert c_bind.choose(reading)['W']=='D+replay'  # completed undefined cost ranks last; never blocks C


def test_wrong_seed_original_receipt_and_synthetic_binding_rejected(tmp_path):
    receipt=tmp_path/'export.json'; c.atomic(receipt,dict(model_dir='/durable/finqa-D+replay-s102/hf-step40'))
    with pytest.raises(c.CorrectnessStop,match='wrong original'):
        c_bind.check_start(dict(model='unused',receipt=str(receipt)),'D+replay',101,False)
    out=tmp_path/'out'; bind(out,tmp_path/'b','F')
    with pytest.raises(c.CorrectnessStop,match='synthetic'): c.binding(out,False)
    a=args(tmp_path/'no-source',source=None,tokenizer=None,finqa_panel=None)
    a.standin=False
    with pytest.raises(c.CorrectnessStop,match='FinQA'): c_targets.evaluation_items(a)


def test_frozen_parser_and_prompt_function_bytes_and_unicode_jsonl(tmp_path):
    import ast
    import hashlib
    source=(c.C/'legal_headroom.py').read_text()
    functions={n.name:hashlib.sha256(ast.get_source_segment(source,n).encode()).hexdigest() for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name in ('parse_answer','parse_answer_tolerant','prompt_for')}
    assert functions==json.loads((c.C/'freeze.json').read_text())['frozen_function_sha256']
    path=tmp_path/'unicode.jsonl'; value=dict(document='one'+chr(0x2028)+'two'+chr(0x85)+'three')
    path.write_text(json.dumps(value,ensure_ascii=False)+'\n')
    assert c.legal.read_jsonl(path)==[value]


@pytest.mark.parametrize('winner', c.RECIPES)
def test_probe_starts_and_shared_generic_identity(tmp_path,winner):
    from kit import next_common as n
    out=tmp_path/'C';bind(out,tmp_path/'B',winner);targets(out)
    doc=c.binding(out,True);replay=replay_fixture(tmp_path/'replay')
    root=tmp_path/'pools';root.mkdir()
    for source in ('gsm8k','arc','earlier'):
        c.t.write_jsonl(root/(source+'.jsonl'),[dict(id=source,prompt='An unrelated replay question',response='A frozen correct response.')])
    n.freeze(root,[source+'.jsonl' for source in ('gsm8k','arc','earlier')],True)
    for probe in ('generic','broad','earlier','both'):
        arm=c.probe_arm(doc,probe)
        for panel in ('panel','test','finqa','general-v1'):score(out,arm,101,0,panel)
        a=args(out,recipe='W',route='sequence',seed=101,replay=replay,tokenizer=None,sdpo=None,trainer_python=sys.executable,
            gpu='0,1,2,3,4,5,6,7',model=None,probe_arm=probe,pool_root=root)
        c_train.job(a)
        summary=json.loads((out/'runs'/f'{arm}-s101/summary.json').read_text())
        assert summary['incoming']['model']==f'synthetic-{winner[0]}-101'
        assert summary['task_exposures']==dict(generic=1120,broad=960,earlier=960,both=800)[probe]
        if probe=='generic' and '+replay' in winner:
            before=(out/'identities'/f'{arm}-s101.json').read_bytes()
            train(out,winner,'sequence',101,replay)
            assert (out/'identities'/f'{arm}-s101.json').read_bytes()==before


def test_replay_prevention_receipt_cannot_enter_c(tmp_path,monkeypatch):
    receipt=tmp_path/'receipt.json'
    c.atomic(receipt,dict(model_dir='/durable/finqa-D+replay-s101/hf-step40'))
    with pytest.raises(c.CorrectnessStop,match='wrong original'):
        c_bind.check_start(dict(model='unused',receipt=str(receipt)),'D+replay',101,False)
    c.atomic(receipt,dict(model_dir='/durable/D-finqa-alone-s101/hf-step40'))
    monkeypatch.setattr(c_bind,'identity',lambda *a,**kw:dict(weights='v4-D'))
    assert c_bind.check_start(dict(model='unused',receipt=str(receipt)),'D+replay',101,False)['identity']['weights']=='v4-D'


@pytest.mark.parametrize('reference_guards',[True,False],ids=['reference-guards','kit-only-pending'])
def test_probe_contractnli_final_and_gain_common_rule_document_pairs(monkeypatch,tmp_path,reference_guards):
    """1 Nov mandatory-reference ruling: pinned guards admit; kit-only remains PENDING."""
    from kit.next_read import analyse
    from kit import m2_assignment
    from kit.ref_guard import METHOD
    assignment=tmp_path/'ref-scoring-assignment.json'
    authority={f'core.{panel}|{seed}':dict(node='CPU-STANDIN',uuid=f'CPU-STANDIN-{seed}-{panel}',environment='CPU-STANDIN')
               for seed in c.SEEDS for panel in ('hellaswag','knowledge-ll')}
    assignment_sha=m2_assignment.freeze(assignment,dict(schema='message2-scoring-assignment.v2',assignments=authority))
    assignment_bytes=assignment.read_bytes()
    # Observe actual draw populations, not just the result's metadata.
    choices=c_read.random.Random.choices;populations=[]
    def observed(rng,population,**kwargs):
        populations.append((len(population),sum(n for _,n in population)))
        return choices(rng,population,**kwargs)
    monkeypatch.setattr(c_read.random.Random,'choices',observed)
    rows=[]
    def add(arm,seed,step,panel,n):
        rows.extend(dict(arm=arm,seed=seed,step=step,panel=panel,cap=2048,question_id=str(i),correct=int(i<n),
            output_tokens=10,finished=1,group='doc-'+str(i//2)) for i in range(100))
    add('base',0,0,'finqa',50)
    for arm in ('generic','broad','earlier','both'):
        for seed in c.SEEDS:
            for step in c.STEPS:add('v4-'+arm,seed,step,'contract_nli',42 if step==0 else 71 if arm in ('broad','both') else 72)
            for step in c.RETENTION_STEPS:
                add('v4-'+arm,seed,step,'math',53 if step and arm in ('broad','both') else 50)
                add('v4-'+arm,seed,step,'hellaswag',50)
                add('v4-'+arm,seed,step,'knowledge',50)
                add('v4-'+arm,seed,step,'finqa',70 if step==0 else 62 if arm in ('earlier','both') else 60)
    for row in rows:
        if row['panel'] not in ('hellaswag','knowledge') or row['step'] not in (0,40):continue
        panel='knowledge-ll' if row['panel']=='knowledge' else row['panel']
        family='core.'+panel
        proof=dict(assignment_sha256=assignment_sha,family=family,seed=row['seed'],**authority[f"{family}|{row['seed']}"])
        assert m2_assignment.verify(dict(assignment=proof),assignment_bytes)==f"{family}|{row['seed']}"
        row.update(_guard_method=METHOD if reference_guards else 'KIT-SCORED',_guard_validation='PASS',
                   _guard_uuid=proof['uuid'],_guard_environment=proof['environment'])
    result=analyse(rows,20,acquisition='contract_nli')
    if not reference_guards:
        assert all(c['status']=='PENDING' and c['reason']=='REFERENCE-SCORED required for decision checkpoint guards'
                   for c in result['v4_comparisons'])
        assert not result['v4_nomination']['broad'] and not result['v4_nomination']['earlier']
        return
    assert all(c['method']==METHOD for c in result['v4_comparisons'])
    assert result['v4_nomination']['broad'] and result['v4_nomination']['earlier']
    ep=result['v4_comparisons'][0]['endpoints']
    assert ep['contract_nli']['lower']==ep['gain']['lower']==-1
    legal=[r for r in result['v4_comparisons'][0]['question_pairs'] if r['panel']=='contract_nli']
    assert all(r['resampling']=='document' and r['clusters']==50 for r in legal)
    legal += [r for r in result['v4_transitions'] if r['panel']=='contract_nli']
    legal += [r['question_interval'] for r in result['summaries'] if r['panel']=='contract_nli']
    legal += [r for comparison in result['v4_comparisons'] for key in ('auc_question_pairs','gain_question_pairs') for r in comparison[key]]
    for curve in result['v4_curves']:
        legal += curve['question_intervals']['points']
        legal += [curve['question_intervals'][key] for key in ('gain','gain_auc')]
    assert all(r['resampling']=='document' and r['clusters']==50 and r['n']==100 for r in legal)
    assert populations and set(populations)=={(50,100)}
    for r in rows:
        if r['arm']=='v4-both' and r['panel']=='contract_nli' and r['step']>0:r['correct']=int(int(r['question_id'])<70)
    assert not analyse(rows,20,acquisition='contract_nli')['v4_nomination']['earlier']


def test_combined_message_full_standin_and_job_handoff(tmp_path):
    """29 Oct return-1 amendment §C/probe and plan §Combined message 2+3: three C seeds, B unchanged."""
    from kit.message_dryrun import job
    job(NS(out=tmp_path/'message',standin=True))
    result=json.loads((tmp_path/'message/message-summary.json').read_text())
    assert result['all_message_rows'] and result['core_and_probe_stages']==24
    assert result['screen']['probe_stages']==0
    assert result['resume_across_jobs'] and result['restored_hashes']>0
    assert result['mid_save_interruption']
    assert result['storage']['bytes_per_save']==[16777221,7,0]
    assert result['storage']['unchanged_skipped'] and len(result['bytes_per_save'])==4
    assert result['changed_gpu_full_C_rescore'] and result['changed_gpu_full_screen_rescore']
    assert len(result['failure_paths'])==3
    assert all(n<28*1024*1024 for n in result['compact_archive_bytes'].values())


@pytest.mark.parametrize('exit_code', [0, 7])
def test_message_readme_exit_trap_captures_parent_with_c_out(tmp_path, exit_code):
    import os
    readme=(c.KIT/'README-message.md').read_text()
    boundary=readme.split('Job boundary commands (',1)[1].split('```bash\n',1)[1].split('```',1)[0]
    printer=tmp_path/'print-only'
    printer.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$CALLS"\n')
    printer.chmod(0o755)
    env=dict(os.environ,TRAINER_PY=str(printer),KIT=str(c.KIT),OUT=str(tmp_path/'parent'),
             C_OUT=str(tmp_path/'parent/C'),DURABLE=str(tmp_path/'durable'),SCORE_GPU='0',CALLS=str(tmp_path/'calls'))
    # README-c uses OUT=C_OUT and commands ending with || exit $?.
    script=boundary+'\nOUT="$C_OUT"\n'+f'(exit {exit_code}) || exit $?\n'
    done=subprocess.run(['bash','-c',script],env=env,capture_output=True,text=True)
    assert done.returncode==exit_code
    calls=(tmp_path/'calls').read_text().splitlines()
    assert len(calls)==4
    assert calls[0]==f'{c.KIT}/message_handoff.py --out {tmp_path}/parent --durable {tmp_path}/durable --action job-start'
    assert calls[-1]==f'{c.KIT}/message_handoff.py --out {tmp_path}/parent --durable {tmp_path}/durable --action save'


def test_message_atomic_generations_interrupted_copy_and_latest_restore(tmp_path):
    from kit import message_handoff as h
    from kit.message_dryrun import interrupt_save
    out=tmp_path/'out';durable=tmp_path/'durable';out.mkdir();(out/'C').mkdir()
    (out/'C/evidence').write_text('generation one')
    (out/'NEXT').mkdir();(out/'NEXT/evidence').write_text('screen one')
    (out/'link').symlink_to('C/evidence')
    first=h.save(out,durable);first_tree=h.committed(durable)/'tree'
    # Restore reproduces the recorded source metadata (nanosecond mtimes), not the durable copy's precision.
    old_manifest=json.loads((h.committed(durable)/'manifest.json').read_text())['entries']
    old_tree=h.inventory(first_tree)
    (out/'C/evidence').write_text('generation two')
    interrupt_save(out,durable)
    restored=tmp_path/'restored';h.restore(durable,restored)
    assert h.inventory(restored)['entries']==old_manifest
    assert (restored/'C/evidence').read_text()=='generation one'
    assert not (restored/'C/interrupted-save-only').exists()
    second=h.save(out,durable)
    assert first['generation']!=second['generation']
    assert h.inventory(first_tree)==old_tree  # prior generation is retained byte-for-byte
    assert json.loads((durable/'last-job-end.json').read_text())['generation']==second['generation']
    # A newer-looking partial/unpublished generation is never selected.
    (durable/'generations/999999999999999999999').mkdir()
    h.restore(durable,restored)
    assert (restored/'C/evidence').read_text()=='generation two'
    assert (restored/'NEXT/evidence').read_text()=='screen one'
    assert (restored/'link').is_symlink()
    assert not (durable/'generations/999999999999999999999/COMMITTED.json').exists()


@pytest.mark.parametrize('failure_point', ['verification', 'publication', 'end-marker'])
def test_message_generation_failure_boundaries(tmp_path, monkeypatch, failure_point):
    from kit import message_handoff as h
    out=tmp_path/'out';out.mkdir();durable=tmp_path/'durable'
    (out/'evidence').write_text('old');h.save(out,durable)
    old=(durable/'CURRENT.json').read_bytes();end=(durable/'last-job-end.json').read_bytes()
    (out/'evidence').write_text('new')
    atomic=h.atomic;inventory=h.inventory
    def interrupted_atomic(path, value):
        if path.name==('CURRENT.json' if failure_point=='publication' else 'last-job-end.json'):
            raise RuntimeError('simulated interruption')
        return atomic(path,value)
    def bad_inventory(root, *args, **kwargs):
        value=inventory(root, *args, **kwargs)
        if root.name=='tree': value['entries']['evidence']['bytes']+=1
        return value
    with monkeypatch.context() as m:
        if failure_point=='verification': m.setattr(h,'inventory',bad_inventory)
        else: m.setattr(h,'atomic',interrupted_atomic)
        with pytest.raises((c.CorrectnessStop,RuntimeError)): h.save(out,durable)
    assert (durable/'last-job-end.json').read_bytes()==end
    restored=tmp_path/'restored';h.restore(durable,restored)
    assert (restored/'evidence').read_text()==('new' if failure_point=='end-marker' else 'old')
    assert ((durable/'CURRENT.json').read_bytes()==old)==(failure_point!='end-marker')


def test_message_restore_rejects_corrupt_committed_generation(tmp_path):
    from kit import message_handoff as h
    out=tmp_path/'out';out.mkdir();durable=tmp_path/'durable'
    (out/'evidence').write_text('verified');h.save(out,durable)
    tree=h.committed(durable)/'tree';(tree/'evidence').write_text('corrupt')
    with pytest.raises(c.CorrectnessStop,match='verification differs'): h.restore(durable,tmp_path/'restore')
    assert not (tmp_path/'restore').exists()


def test_c_document_question_intervals_keep_each_pair_together(monkeypatch):
    rows={str(i):dict(group='doc-'+str(i//2),correct=i%2) for i in range(100)}
    choices=c_read.random.Random.choices;populations=[]
    def observed(rng,population,**kwargs):
        populations.append((len(population),sum(n for _,n in population)))
        return choices(rng,population,**kwargs)
    monkeypatch.setattr(c_read.random.Random,'choices',observed)
    zero={q:dict(r,correct=0) for q,r in rows.items()}
    intervals=c_read.curve_question_intervals([zero,rows,rows,rows,rows],'correct',40)
    assert intervals['points'][-1]['question_bootstrap_95']==[50,50]
    assert intervals['gain']['question_bootstrap_95']==[50,50]
    assert intervals['gain_auc']['question_bootstrap_95']==[43.75,43.75]
    assert c_read.paired(zero,rows,'correct',40,True)['question_bootstrap_95']==[50,50]
    assert set(populations)=={(50,100)}


def test_message_incremental_hash_copy_skip_prune_and_partials(tmp_path, monkeypatch):
    from kit import message_handoff as h
    out=tmp_path/'out';out.mkdir();durable=tmp_path/'durable'
    (out/'large').write_bytes(b'x'*(8*1024*1024));(out/'small').write_text('one')
    first=h.save(out,durable);one=h.committed(durable)/'tree'
    hashes=[];original=h.s.file_hash
    def hashed(path):
        hashes.append(Path(path));return original(path)
    monkeypatch.setattr(h.s,'file_hash',hashed)
    # Same-size, same-second edit: ns diff must bypass rsync's quick check.
    st=(out/'small').stat();(out/'small').write_text('two')
    import os
    os.utime(out/'small',ns=(st.st_atime_ns,st.st_mtime_ns+1))
    second=h.save(out,durable);two=h.committed(durable,verify=False)/'tree'
    assert second['bytes_copied']==second['changed_bytes']==3
    assert not any(p.name=='large' for p in hashes)
    assert (one/'large').stat().st_ino==(two/'large').stat().st_ino
    assert (one/'small').read_text()=='one' and (two/'small').read_text()=='two'
    before=(durable/'CURRENT.json').read_bytes()
    hashes.clear();skipped=h.save(out,durable)
    assert skipped['skipped'] and skipped['bytes_copied']==0
    assert not any(p.name in ('small','large') for p in hashes)
    assert (durable/'CURRENT.json').read_bytes()==before
    (durable/'generations/.partial-dead').mkdir();(durable/'generations/orphan').mkdir()
    (out/'small').write_text('three');third=h.save(out,durable)
    assert third['bytes_copied']==5
    assert {p.name for p in (durable/'generations').iterdir()}=={second['generation'],third['generation']}
    assert not one.exists()


def test_message_restore_preserves_local_newer_extra_and_type_conflicts(tmp_path):
    from kit import message_handoff as h
    import os, time
    out=tmp_path/'out';out.mkdir();durable=tmp_path/'durable'
    (out/'stage').write_text('committed');(out/'type').write_text('file')
    (out/'touched').write_text('same bytes')
    h.save(out,durable)
    (out/'stage').write_text('new local stage')
    os.utime(out/'stage',(time.time()+10,time.time()+10))
    os.utime(out/'touched',(time.time()+10,time.time()+10))
    (out/'extra').mkdir();(out/'extra/checkpoint').write_text('precious')
    (out/'type').unlink();(out/'type').mkdir();(out/'type/worker').write_text('worker')
    with pytest.raises(c.CorrectnessStop,match='--move-aside'): h.restore(durable,out,move_aside=False)
    assert (out/'stage').read_text()=='new local stage'
    result=h.restore(durable,out);backup=Path(result['backup'])
    assert {'stage','touched'} <= set(result['newer_local'])
    assert (backup/'touched').read_text()=='same bytes'
    assert (backup/'stage').read_text()=='new local stage'
    assert (backup/'extra/checkpoint').read_text()=='precious'
    assert (backup/'type/worker').read_text()=='worker'
    assert (out/'stage').read_text()=='committed' and (out/'type').read_text()=='file'


def test_message_lock_excludes_save_and_restore_processes(tmp_path):
    from kit import message_handoff as h
    out=tmp_path/'out';out.mkdir();durable=tmp_path/'durable'
    (out/'file').write_text('data')
    with h.locked(durable):
        for action in ('save','restore'):
            done=subprocess.run([sys.executable,str(c.KIT/'message_handoff.py'),'--action',action,
                '--out',str(out),'--durable',str(durable)],capture_output=True,text=True)
            assert done.returncode!=0 and 'handoff busy' in done.stderr
        assert not (durable/'CURRENT.json').exists()
    assert h.save(out,durable)['verified']


def test_message_stale_pointer_tmp_first_save_and_recovery(tmp_path, capsys):
    from kit import message_handoff as h
    durable=tmp_path/'durable';durable.mkdir();out=tmp_path/'out'
    (durable/'CURRENT.json.tmp').write_text('{dead writer')
    assert h.restore(durable,out).startswith('FIRST JOB')
    (out/'evidence').write_text('current');h.save(out,durable)
    assert not (durable/'CURRENT.json.tmp').exists()
    (durable/'CURRENT.json.tmp-123-dead').write_text('{dead writer')
    assert h.restore(durable,tmp_path/'restored')['verified']
    assert (tmp_path/'restored/evidence').read_text()=='current'
    assert 'ignoring stale CURRENT' in capsys.readouterr().err


@pytest.mark.parametrize('cooperative',[True,False])
def test_message_term_forwards_to_background_child_then_saves(tmp_path, cooperative):
    import os, signal, time
    printer=tmp_path/'save'
    printer.write_text('#!/bin/sh\nprintf "save\\n" >> "$EVENTS"\n');printer.chmod(0o755)
    worker=tmp_path/'worker.py'
    grandchild=('import os,signal,time\nfrom pathlib import Path\n'
        'def term(*args):\n with open(os.environ["EVENTS"],"a") as h: h.write("grandchild-term\\n")\n raise SystemExit(0)\n'
        'signal.signal(signal.SIGTERM,term)\nPath(os.environ["READY"]+"-grandchild").write_text("ready")\nwhile True: time.sleep(.1)\n')
    worker.write_text('import os,signal,time,subprocess,sys\nfrom pathlib import Path\n'
        'def term(*args):\n with open(os.environ["EVENTS"],"a") as h: h.write("term\\n")\n'+
        (' raise SystemExit(0)\n' if cooperative else ' return\n')+
        'signal.signal(signal.SIGTERM,term)\n'+
        'subprocess.Popen([sys.executable,"-c",'+repr(grandchild)+'])\n'+
        'while not Path(os.environ["READY"]+"-grandchild").exists(): time.sleep(.01)\n'+
        'Path(os.environ["READY"]).write_text("ready")\nwhile True: time.sleep(0.1)\n')
    env=dict(os.environ,TRAINER_PY=str(printer),KIT=str(c.KIT),PARENT_OUT=str(tmp_path/'out'),
             DURABLE_ROOT=str(tmp_path/'durable'),TERM_LEAD_SECONDS='2',SAVE_ALLOWANCE_SECONDS='1',
             EVENTS=str(tmp_path/'events'),READY=str(tmp_path/'ready'))
    script='source "$KIT/message_job.sh"\nrun_stage '+sys.executable+' '+str(worker)+'\n'
    shell=subprocess.Popen(['bash','-c',script],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    try:
        deadline=time.monotonic()+10
        while not (tmp_path/'ready').exists():
            assert shell.poll() is None and time.monotonic()<deadline
            time.sleep(.02)
        start=time.monotonic();shell.send_signal(signal.SIGTERM)
        shell.communicate(timeout=5)
        assert shell.returncode==143 and time.monotonic()-start<4
        events=(tmp_path/'events').read_text().splitlines()
        assert sorted(events[:-1])==['grandchild-term','term'] and events[-1]=='save'
    finally:
        if shell.poll() is None: shell.kill();shell.wait()


def test_message_save_after_restore_reuses_digests_and_hard_links(tmp_path):
    # 21 Oct (Opus re-check): restored files carried whole-second mtimes, so the first save after a restore
    # re-copied and re-hashed all of OUT. Restore now puts back the recorded nanosecond mtimes.
    import os
    from kit import message_handoff as h
    out=tmp_path/'out';out.mkdir();durable=tmp_path/'durable'
    (out/'large').write_bytes(b'y'*(4*1024*1024));(out/'small').write_text('a')
    st=(out/'large').stat();os.utime(out/'large',ns=(st.st_atime_ns,st.st_mtime_ns+123456789))
    h.save(out,durable);gen=h.committed(durable)/'tree'
    fresh=tmp_path/'fresh'
    h.restore(durable,fresh)
    # Simulate a filesystem that truncated mtimes to whole seconds during copy, then re-apply via restore logic.
    assert (fresh/'large').stat().st_mtime_ns==(out/'large').stat().st_mtime_ns
    again=h.save(fresh,durable)
    assert again.get('skipped') and again['bytes_copied']==0


def test_message_lock_falls_back_to_mkdir_when_flock_unsupported(tmp_path, monkeypatch):
    from kit import message_handoff as h
    def no_flock(*a, **k): raise OSError(38, 'Function not implemented')
    monkeypatch.setattr(h.fcntl, 'flock', no_flock)
    durable=tmp_path/'durable'
    with h.locked(durable):
        assert (durable/'.handoff.lock.d'/'owner.json').exists()
        with pytest.raises(h.CorrectnessStop):
            with h.locked(durable): pass
    assert not (durable/'.handoff.lock.d').exists()
    # A stale lock from a dead process on this host is cleared.
    (durable/'.handoff.lock.d').mkdir()
    import json, os
    (durable/'.handoff.lock.d'/'owner.json').write_text(json.dumps(dict(host=os.uname().nodename,pid=999999)))
    with h.locked(durable): pass


@pytest.mark.parametrize('first_action', ('save','restore'))
def test_message_first_handoff_with_mkdir_lock(tmp_path,monkeypatch,first_action):
    from kit import message_handoff as h
    def no_flock(*a,**kw):raise OSError(38,'Function not implemented')
    monkeypatch.setattr(h.fcntl,'flock',no_flock)
    durable=tmp_path/'durable';out=tmp_path/'out'
    if first_action=='restore':
        assert h.restore(durable,out)=='FIRST JOB: no committed durable OUT yet'
    else:out.mkdir()
    (out/'evidence').write_text('first generation')
    assert h.save(out,durable)['verified']
    restored=tmp_path/'restored'
    assert h.restore(durable,restored)['verified']
    assert (restored/'evidence').read_text()=='first generation'
    assert not (durable/'.handoff.lock.d').exists()
