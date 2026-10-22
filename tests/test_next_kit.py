"""Offline fixture tests for next-message scripts, scientific rules and legacy isolation."""
import argparse
from collections import Counter
import copy
import importlib.util
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
from types import SimpleNamespace as NS

import pytest
import torch
from kit import next_common as c, next_blend as blend, next_slots as slots, next_pools as pools
from kit import next_read as reader, next_hellaswag as hs


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setenv('HF_HUB_OFFLINE','1');monkeypatch.setenv('HF_DATASETS_OFFLINE','1')
    import urllib.request
    monkeypatch.setattr(urllib.request,'urlopen',lambda *a,**kw:pytest.fail('network forbidden'))


def pool_fixture(root):
    for source in ('gsm8k','arc','earlier'):
        c.write_rows(root/(source+'.jsonl'),[dict(id=source+str(i),prompt='q'+str(i),response='target',source=source) for i in range(3)])
    c.freeze(root,[source+'.jsonl' for source in ('gsm8k','arc','earlier')],True)
    return root


def test_frozen_and_legacy_sources_byte_identical():
    c.frozen_next()
    for name,digest in json.loads((c.N/'legacy-sha256.json').read_text()).items():c.check(c.KIT/name,digest)
    # The SEC vendoring freeze predates later private EDGAR transport fixes.
    # Pin that exact source; the movable private script is not a reference.
    assert c.s.file_hash(c.KIT/'next_sec_core.py')=='faaa699c8c60d678d29d4ecf675277e321a50843d4148b839a0e30462c2909da'
    assert c.s.file_hash(c.N/'sec/manifest.json')=='3e98419636a327c004464cb31b921db54acf78a06cc2e92f873c6a5e83b13782'
    assert sum(len(json.loads(p.read_text())) for p in (c.N/'sec').glob('batch-*.json'))==193


@pytest.mark.parametrize('arm,math_count,added',[('generic',1120,160),('broad',960,320),('earlier',960,320),('both',800,480)])
@pytest.mark.parametrize('seed',range(101,106))
def test_registered_maps_and_balanced_sources(tmp_path,arm,math_count,added,seed):
    root=pool_fixture(tmp_path/'pools')
    task=[dict(id='t'+str(i),prompt='math',response='math target') for i in range(1280)]
    generic=[dict(id='g'+str(i),prompt='generic',response='base target') for i in range(3)]
    rows,audit=slots.compose(task,generic,root,arm,seed)
    assert len(rows)==1280 and sum(not r.get('p3b_replay') for r in rows)==math_count
    assert sum(bool(r.get('p3b_replay')) for r in rows)==added
    mapping={p:src for src,positions in slots.MAPS[arm].items() for p in positions}
    for i,r in enumerate(rows):
        if i%32+1 not in mapping:assert r==task[i]
        else:
            assert r['p3b_replay'] and r['data_source']=='p3b_replay'
            assert r['extra_info']['p3b_task_source']=='math_screen'
    if arm in ('broad','both'):
        broad=[r['source'] for r in rows if r.get('source') in ('gsm8k','arc')]
        assert broad==['gsm8k','arc']*80
    assert audit['exposures']['math']==math_count


def test_d_skips_in_place_per_source_and_preserves_full_pool_shuffle(tmp_path):
    from test_p3b_mixed import Tokenizer
    root=pool_fixture(tmp_path/'pools');generic=[dict(id=str(i),prompt='x'*(2049 if i==0 else 4),response='target') for i in range(5)]
    task=[dict(id=str(i)) for i in range(1280)]
    rows,audit=slots.compose(task,generic,root,'both',101,Tokenizer())
    order=list(generic);random.Random(101).shuffle(order);eligible=[r['id'] for r in order if r['id']!='0']
    observed=[r['id'] for i,r in enumerate(rows) if i%32+1 in (8,16,24,32)]
    assert observed==(eligible*40)[:160]
    assert audit['skipped_draw_count']>0 and all(r['id']=='0' and r['source']=='generic' for r in audit['skipped_draws'])


@pytest.mark.parametrize('dtype',[torch.bfloat16,torch.float16,torch.float32])
@pytest.mark.parametrize('lam',[0,.25,.5,.75,1])
def test_interpolate_fp32_and_exact_endpoints(dtype,lam):
    x=torch.tensor([1.0078125,-100.5,7.25],dtype=dtype);y=torch.tensor([5.375,12.625,-3.5],dtype=dtype)
    result=blend.interpolate(x,y,lam)
    expected=x if lam==0 else y if lam==1 else ((1-lam)*x.float()+lam*y.float()).to(dtype)
    assert result.dtype==dtype and torch.equal(result,expected)
    assert blend.tensor_hash(result)==blend.tensor_hash(expected)


@pytest.mark.parametrize('kind',['keys','shape','dtype','buffer','tokenizer'])
def test_real_safetensor_blend_rejects_mismatches(tmp_path,kind):
    from safetensors.torch import save_file
    start=tmp_path/'a';final=tmp_path/'b';start.mkdir();final.mkdir()
    x=dict(weight=torch.tensor([1,2],dtype=torch.bfloat16),count=torch.tensor([3]))
    y={k:v.clone() for k,v in x.items()}
    if kind=='keys':y['other']=torch.ones(1)
    if kind=='shape':y['weight']=torch.ones(3,dtype=torch.bfloat16)
    if kind=='dtype':y['weight']=y['weight'].float()
    if kind=='buffer':y['count']+=1
    for root,values in ((start,x),(final,y)):
        save_file(values,str(root/'model.safetensors'));c.atomic(root/'config.json',{})
        c.atomic(root/'tokenizer.json',dict(version=2 if kind=='tokenizer' and root==final else 1))
    with pytest.raises(c.CorrectnessStop):blend.blend(start,final,tmp_path/'blend',.5)


def test_real_blend_sharding_hashes_endpoints_and_cleanup_on_scoring_failure(tmp_path,monkeypatch):
    from safetensors.torch import save_file,load_file
    start=tmp_path/'start';final=tmp_path/'final'
    for root,v in ((start,1),(final,3)):
        root.mkdir();c.atomic(root/'config.json',{});c.atomic(root/'tokenizer.json',{})
        save_file(dict(weight=torch.full((2,),v,dtype=torch.bfloat16)),str(root/'model.safetensors'))
    hashes=blend.blend(start,final,tmp_path/'blend',.5)
    assert load_file(str(tmp_path/'blend/model-00001-of-00001.safetensors'))['weight'].tolist()==[2,2]
    assert hashes['weight']['blend']==blend.tensor_hash(torch.tensor([2,2],dtype=torch.bfloat16))
    parents={}
    for key,root in [('start',start),('final',final)]:
        identity=c.identity(root);receipt=tmp_path/(key+'.json');c.atomic(receipt,identity)
        parents[key]=dict(path=str(root),receipt=str(receipt),identity=identity)
    out=tmp_path/'out';c.atomic(out/'next-inputs/inputs.json',dict(finals=[dict(name='sequence-F-s101',recipe='F',seed=101,route='sequence',**parents)]))
    c.freeze(out/'next-inputs',['inputs.json'],False)
    monkeypatch.setattr(blend,'machine',lambda *a:None);monkeypatch.setattr(blend,'local_nvme',lambda *a:None)
    def fail(command):raise RuntimeError('simulated scorer crash')
    monkeypatch.setattr(blend,'run',fail)
    with pytest.raises(RuntimeError,match='scorer crash'):
        blend.job(NS(out=out,standin=False,gpu='0',nvme_root=tmp_path,slot='sequence-F-s101',lam=.5,package_a=None,finqa=None))
    assert not list(tmp_path.glob('next-blend-*'))
    assert json.loads((out/'blends/sequence-F-s101-l0.5/storage.json').read_text())['deleted']
    assert (out/'blends/sequence-F-s101-l0.5/tensor-hashes.json').exists()


@pytest.mark.parametrize('text,expected',[('Answer: A','A'),('reasoning\nAnswer: D','D'),('Answer: a',None),('Answer: A.',None),('Answer: A\nextra',None),('<answer>A</answer>',None),('',None)])
def test_hellaswag_frozen_strict_label(text,expected):assert hs.strict_label(text)==expected


def hellaswag_fixture(n=120):
    return [dict(ind=i,source_id=f'activitynet~fixture-{i}',ctx=f'context {i}',
                 endings=['a','b','c','d'],label=i%4) for i in range(n)]


def test_hellaswag_activitynet_population_and_duplicate_check():
    import hashlib
    raw=hellaswag_fixture()
    # Real validation has cross-population ind collisions; filter before checking IDs.
    mixed=raw+[dict(r,source_id=f'wikihow~fixture-{r["ind"]}',ctx='excluded') for r in raw]
    panel=hs.panel_rows(mixed)
    expected=sorted(raw,key=lambda r:(hashlib.sha256(('next-v4-holdout-v1|'+str(r['ind'])).encode()).hexdigest(),str(r['ind'])))[:100]
    assert len(panel)==100 and len({r['id'] for r in panel})==100
    assert [r['id'] for r in panel]==['hellaswag-val-'+str(r['ind']) for r in expected]
    assert all(r['prompt']==source['ctx']+'\nA. a\nB. b\nC. c\nD. d\nEnd with Answer: <letter>'
               and r['gold']=='ABCD'[source['label']] and r['licence_notice']==hs.NOTICE
               for r,source in zip(panel,expected))
    with pytest.raises(c.CorrectnessStop,match='duplicate HellaSwag source IDs'):
        hs.panel_rows(raw+[raw[0]])


def test_hellaswag_wikihow_row_cannot_fill_panel_stops_exit_2(tmp_path):
    raw=hellaswag_fixture(100);raw[0]['source_id']='wikihow~excluded'
    rows=tmp_path/'rows.jsonl';c.write_rows(rows,raw)
    result=subprocess.run([sys.executable,str(c.KIT/'next_hellaswag.py'),'--out',str(tmp_path/'out'),
                           '--standin','--action','prepare','--rows',str(rows)],capture_output=True,text=True)
    assert result.returncode==2,result.stdout+result.stderr
    assert 'STOP: correctness: HellaSwag panel requires 100 ActivityNet items' in result.stdout
    assert not (tmp_path/'out/hellaswag/manifest.json').exists()
    with pytest.raises(c.CorrectnessStop,match='100 ActivityNet items'):
        hs.panel_rows([dict(raw[0])])


def test_hellaswag_wrong_parquet_hash_stops_exit_2(tmp_path,monkeypatch,capfd):
    import huggingface_hub
    import pyarrow.parquet as pq
    cache=tmp_path/'cache';path=cache/next(iter(hs.VALIDATION_FILES))
    path.parent.mkdir(parents=True);path.write_bytes(b'wrong parquet hash')
    monkeypatch.setattr(huggingface_hub,'snapshot_download',lambda *a,**kw:str(cache))
    monkeypatch.setattr(pq,'read_table',lambda *a,**kw:pytest.fail('hash must be checked before reading parquet'))
    out=tmp_path/'out'
    monkeypatch.setattr(sys,'argv',['next_hellaswag.py','--out',str(out),'--action','prepare'])
    assert hs.entry(hs.job,hs.configure,'next-hellaswag-{action}')==2
    assert 'STOP: correctness: pinned HellaSwag validation parquet files/SHA256 mismatch' in capfd.readouterr().out
    assert not (out/'hellaswag/manifest.json').exists()


def test_hellaswag_local_parquet_panel_and_freeze(tmp_path,monkeypatch,record_property):
    parquet=Path(os.environ.get('HELLASWAG_VALIDATION_PARQUET',''))
    if not os.environ.get('HELLASWAG_VALIDATION_PARQUET') or not parquet.is_file():pytest.skip('set HELLASWAG_VALIDATION_PARQUET to a staged validation parquet')
    import huggingface_hub
    import pyarrow.parquet as pq
    # A cache-shaped symlink exercises the real download/hash/prepare path without copying data.
    cache=tmp_path/'cache';path=cache/next(iter(hs.VALIDATION_FILES))
    path.parent.mkdir(parents=True);path.symlink_to(parquet)
    def snapshot(repo,**kw):
        assert repo=='Rowan/hellaswag'
        assert kw==dict(repo_type='dataset',revision='218ec52e09a7e7462a5400043bb9a69a41d06b76',local_files_only=True)
        return str(cache)
    monkeypatch.setattr(huggingface_hub,'snapshot_download',snapshot)
    raw=pq.read_table(parquet).to_pylist()
    panel=hs.panel_rows(raw)
    assert len(raw)==10042 and sum(r['source_id'].startswith('activitynet~') for r in raw)==3243
    activitynet={f'hellaswag-val-{r["ind"]}':r for r in raw if r['source_id'].startswith('activitynet~')}
    assert len(panel)==100 and len({r['id'] for r in panel})==100
    assert all(activitynet[r['id']]['source_id'].startswith('activitynet~') for r in panel)
    out=tmp_path/'out';args=NS(out=out,standin=False,rows=None)
    hs.prepare(args)
    root=out/'hellaswag';manifest=c.frozen_files(root)
    assert c.t.read_jsonl(root/'panel.jsonl')==panel
    assert manifest['technical_synthetic'] is False and manifest['notice']==hs.NOTICE
    assert json.loads((root/'inputs.json').read_text())['source_files']==hs.VALIDATION_FILES
    digest=c.s.file_hash(root/'panel.jsonl')
    assert manifest['panel_sha256']==digest
    record_property('hellaswag_panel_sha256',digest)
    print('Real parquet frozen HellaSwag-100 panel SHA256:',digest)
    before={p.name:p.read_bytes() for p in root.iterdir()}
    hs.prepare(args)
    assert before=={p.name:p.read_bytes() for p in root.iterdir()}


def test_pool_prompts_namespaces_order_checker_and_chat_dedup():
    rows=pools.candidates([dict(question='What is 2 plus 2?',answer='work #### 4')],[dict(id='id',question='Question?',choices=dict(label=['1','2','3','4'],text=['a','b','c','d']),answerKey='2')])
    assert rows[0]['id']=='gsm8k-train-0' and rows[1]['id']=='arc-c-train-id'
    assert rows[0]['prompt'].endswith('\n\nSolve step by step, then end your response with "Answer: <number>".')
    assert pools.correct(rows[0],'Steps\nAnswer: 4') and not pools.correct(rows[0],'Answer: 5')
    assert pools.correct(rows[1],'Answer: 2') and not pools.correct(rows[1],'Answer: B')
    kept,excluded=pools.exclude([dict(id='earlier',prompt=[dict(role='user',content='ＨＥＬＬＯ world')])],[dict(id='eval',prompt='hello WORLD')])
    assert kept==[] and excluded[0]['overlaps']=='eval'


@pytest.mark.parametrize('recipe',['D','F','D+replay','F+replay'])
@pytest.mark.parametrize('route',['standalone','sequence'])
def test_ordinary_b_training_outputs_byte_identical_to_previous_implementation(tmp_path,recipe,route):
    from kit import p3b_train
    from kit.next_dryrun import fixture
    f=fixture(tmp_path/'fixture');out=f['b']
    # The legacy implementation is a frozen source copy, run against exactly the same OUT.
    ns={'__name__':'kit.legacy_train'};exec(compile((c.N/'legacy/p3b_train.py').read_text(),'<legacy>', 'exec'),ns)
    args=NS(out=out,standin=True,recipe=recipe,route=route,seed=101,gpu='0',model='unused',incoming_receipt=None,
        task_schedule=out/'targets/schedule-101.jsonl',schedule_manifest=out/'targets/manifest.json')
    run=out/f'runs/{route}-{recipe}-s101';shutil.rmtree(run)
    ns['job'](args)
    paths=[out/f'identities/{route}-{recipe}-s101.json',*sorted(run.rglob('*.json'))]
    expected={str(p.relative_to(out)):p.read_bytes() for p in paths}
    shutil.rmtree(run);(out/f'identities/{route}-{recipe}-s101.json').unlink()
    p3b_train.job(args)
    assert expected=={str(p.relative_to(out)):p.read_bytes() for p in [out/f'identities/{route}-{recipe}-s101.json',*sorted(run.rglob('*.json'))]}


def test_reader_amendments_and_undefined_denominators():
    assert reader.interval([1]*5)['lower']==1
    assert reader.protects(reader.interval([-1]*5))
    assert not reader.protects(reader.interval([-1.01]*5))
    assert reader.benefit(reader.interval([2]*5))
    assert not reader.benefit(reader.interval([1.99]*5))
    assert reader.interval([1,2,3,4])['lower'] is None
    assert reader.interval([None]*5)['mean'] is None
    result=reader.analyse([],20)
    assert not result['v4_nomination']['broad'] and not result['v4_nomination']['earlier']
    assert result['v3_proper_lambda']==.5 and result['v4_nomination']['admission']=='Pending connected finals'


def test_public_cli_failure_codes_and_traceback(tmp_path):
    def cli(script,*args,code):
        result=subprocess.run([sys.executable,str(c.KIT/script),'--out',str(tmp_path/'out'),'--standin',*map(str,args)],capture_output=True,text=True,
            env={**os.environ,'HF_HUB_OFFLINE':'1','HF_DATASETS_OFFLINE':'1'})
        assert result.returncode==code,result.stdout+result.stderr
        return result
    cli('next_collect.py','--archive',tmp_path/'empty.tar.gz',code=3)
    cli('next_score.py','--arm','base','--seed',0,'--step',0,'--panel-set','finqa','--gpu','0,1',code=2)
    malformed=tmp_path/'broken.jsonl';malformed.write_text('{broken\n')
    result=cli('next_hellaswag.py','--action','prepare','--rows',malformed,code=1)
    assert 'Traceback' in result.stderr+result.stdout


def test_full_runbook_standin(tmp_path):
    from kit.next_dryrun import job
    job(NS(out=tmp_path,standin=True))
    result=json.loads((tmp_path/'dryrun-summary.json').read_text())
    assert result['screen_finals']==60 and result['blend_points']==300 and result['probe_stages']==0
    assert result['probe_score_jobs']==0 and result['sec_checkpoints']==6
    assert result['question_rows']>0
    assert result['unique_score_jobs']==663 and result['unique_shared_endpoints']==183
    assert len(result['failure_paths'])==7
    assert not list(tmp_path.glob('next-blend-*'))


def test_hellaswag_real_scorer_path_with_fixture_vllm(tmp_path,monkeypatch):
    """Execute eval_bed generation+grade plumbing with synthetic engine outputs, no GPU/network."""
    import types
    rows=hs.panel_rows([dict(ind=i,source_id=f'activitynet~fixture-{i}',ctx=f'context {i}',endings=['a','b','c','d'],label=i%4) for i in range(100)])
    c.write_rows(tmp_path/'hellaswag/panel.jsonl',rows)
    c.freeze(tmp_path/'hellaswag',['panel.jsonl'],False,panel_sha256=c.s.file_hash(tmp_path/'hellaswag/panel.jsonl'))
    model=tmp_path/'model';model.mkdir();c.atomic(model/'config.json',{})
    class Tokenizer:
        def apply_chat_template(self,messages,**kw):return '\n'.join(m['content'] for m in messages)
    class LLM:
        def __init__(self,**kw):assert kw['max_model_len']==12288 and kw['seed']==0
        def generate(self,prompts,params):
            return [NS(prompt_token_ids=[1,2],outputs=[NS(text='Answer: '+r['gold'],token_ids=[3,4],finish_reason='stop')]) for r in rows]
    monkeypatch.setitem(sys.modules,'transformers',NS(AutoTokenizer=NS(from_pretrained=lambda path:Tokenizer()),__version__='fixture'))
    monkeypatch.setitem(sys.modules,'vllm',NS(LLM=LLM,SamplingParams=lambda **kw:kw,__version__='fixture'))
    monkeypatch.setattr(hs,'machine',lambda *a:None);monkeypatch.setattr(hs,'checkpoint',lambda *a,**kw:dict(fixture=True))
    hs.score(NS(out=tmp_path,standin=False,gpu='0',model=str(model),checkpoint_receipt=None,arm='base',seed=0,step=0))
    import csv
    scores=list(csv.DictReader((tmp_path/'scores/base-s0-k0-hellaswag/scores.csv').open()))
    assert len(scores)==100 and all(r['correct']=='1' for r in scores)


def test_factorial_admission_uses_one_point_acquisition_exception_and_surviving_incumbent():
    rows=[]
    def add(arm,seed,step,panel,ncorrect):
        rows.extend(dict(arm=arm,seed=seed,step=step,panel=panel,task=panel,cap=2048,question_id=str(i),
            correct=int(i<ncorrect),output_tokens=10,finished=1) for i in range(100))
    add('base',0,0,'finqa',50)
    for arm in ('generic','broad','earlier','both'):
        for seed in c.SEEDS:
            for step in (0,10,20,30,40):
                add('v4-'+arm,seed,step,'math_screen',50 if step==0 or arm in ('generic','earlier') else 49)
            for step in (0,20,40):
                add('v4-'+arm,seed,step,'math',53 if step and arm in ('broad','both') else 50)
                add('v4-'+arm,seed,step,'hellaswag',50)
                add('v4-'+arm,seed,step,'finqa',70 if step==0 else 62 if arm in ('earlier','both') else 60)
    result=reader.analyse(rows,20)
    assert result['v4_nomination']['broad'] is True
    assert result['v4_nomination']['earlier'] is True
    assert result['v4_nomination']['earlier_comparison']==['both','broad']
    both=next(r for r in result['v4_comparisons'] if r['candidate']=='both' and r['control']=='generic')
    assert both['endpoints']['math_screen']['lower']==-1
    assert both['endpoints']['gain_auc']['lower']>=-1
    assert len(result['hellaswag_transfer'])==20
    assert len(result['v4_survival'])==20 and all(r['n']==20 for r in result['v4_survival'])
    for r in rows:
        if r['arm']=='v4-both' and r['panel']=='math_screen' and r['step']>0:r['correct']=int(int(r['question_id'])<48)
    assert not reader.analyse(rows,20)['v4_nomination']['earlier']


def test_both_d_actor_uses_twenty_kl_and_twelve_eos_ce_items(monkeypatch):
    from test_p3b_plain import _mixed_actor_update
    from kit import sdft_actor,p3b_loss
    monkeypatch.setenv('P3B_REPLAY','1')
    kl=sdft_actor.projected_distillation;ce=p3b_loss.replay_loss;counts=Counter()
    def task(*args,**kw):counts['KL']+=1;return kl(*args,**kw)
    def replay(model,inputs,target_ids):
        counts['CE']+=1;assert target_ids==[6,5,4,3]  # All frozen target tokens, including its final token.
        return ce(model,inputs,target_ids)
    monkeypatch.setattr(sdft_actor,'projected_distillation',task);monkeypatch.setattr(p3b_loss,'replay_loss',replay)
    positions={p for values in slots.MAPS['both'].values() for p in values}
    _mixed_actor_update([i+1 in positions for i in range(32)])
    assert counts=={'KL':20,'CE':12}


def test_real_base_export_metadata_blend_and_explicit_recreation(tmp_path,monkeypatch):
    """Real safetensors, with the same encoding/metadata differences as base vs export."""
    from safetensors.torch import save_file,load_file
    start=tmp_path/'base';final=tmp_path/'export'
    for root,value in ((start,1),(final,5)):
        root.mkdir();c.atomic(root/'config.json',{})
        for name in ('tokenizer.json','vocab.json'):c.atomic(root/name,dict(tokens=['a','b']))
        (root/'merges.txt').write_text('#version: 0.2\na b\n')
        c.atomic(root/'tokenizer_config.json',dict(export=root==final))
        save_file(dict(weight=torch.full((2,),value,dtype=torch.bfloat16),count=torch.tensor([2])),str(root/'model.safetensors'))
    for name in ('chat_template.jinja','added_tokens.json','special_tokens_map.json'):(final/name).write_text('{}')
    parents={}
    for key,root in [('start',start),('final',final)]:
        identity=c.identity(root);receipt=tmp_path/(key+'.json');c.atomic(receipt,identity)
        parents[key]=dict(path=str(root),receipt=str(receipt),identity=identity)
    out=tmp_path/'out';slot=dict(name='sequence-F-s101',recipe='F',seed=101,route='sequence',**parents)
    c.atomic(out/'next-inputs/inputs.json',dict(finals=[slot]));c.freeze(out/'next-inputs',['inputs.json'],False)
    monkeypatch.setattr(blend,'machine',lambda *a:None);monkeypatch.setattr(blend,'local_nvme',lambda *a:None)
    commands=[];monkeypatch.setattr(blend,'run',lambda command:commands.append(command))
    args=NS(out=out,standin=False,gpu='0',nvme_root=tmp_path,slot=slot['name'],lam=.5,package_a=None,finqa=None,recreate=None)
    blend.job(args)
    folder=out/'blends/sequence-F-s101-l0.5';record=json.loads((folder/'tensor-hashes.json').read_text())
    assert len(commands)==3 and not list(tmp_path.glob('next-blend-*'))
    differences=record['tokenizer_metadata_differences']
    assert set(differences)=={'tokenizer_config.json','chat_template.jinja','added_tokens.json','special_tokens_map.json'}
    assert differences['chat_template.jinja']['start'] is None
    # Run the recorded CLI through its real parser; completed blends must rebuild, not SKIP.
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path);parser.add_argument('--standin',action='store_true');blend.configure(parser)
    recreated=parser.parse_args(record['recreate'][2:]);blend.job(recreated)
    values=load_file(str(recreated.recreate/'model-00001-of-00001.safetensors'))
    assert values['weight'].tolist()==[3,3] and len(commands)==3
    assert {k:blend.tensor_hash(v) for k,v in values.items()}=={k:v['blend'] for k,v in record['tensors'].items()}
    # A frozen but incorrect expected hash must fail verification and remove the rebuilt copy.
    record['tensors']['weight']['blend']='0'*64
    c.atomic(folder/'tensor-hashes.json',record)
    c.freeze(folder,['inputs.json','tensor-hashes.json'],False)
    recreated.recreate=tmp_path/'bad-hash'
    with pytest.raises(c.CorrectnessStop,match='recreated tensor hashes differ'):blend.job(recreated)
    assert not recreated.recreate.exists()
    # Alter a parent: the original parent receipt still rejects it.
    save_file(dict(weight=torch.full((2,),7,dtype=torch.bfloat16),count=torch.tensor([2])),str(final/'model.safetensors'))
    recreated.recreate=tmp_path/'tampered'
    with pytest.raises(c.CorrectnessStop):blend.job(recreated)
    assert not recreated.recreate.exists()


def test_native_finqa_exclusions_use_b_validation_and_preserve_chat(tmp_path,monkeypatch):
    from kit import v4_contract as contract
    rows=[dict(data_source='finqa',prompt=[dict(role='user',content=f'Question from filing {i} with table')],
               reward_model=dict(ground_truth='2'),extra_info=dict(index=f'test-{i}',split='test')) for i in range(1147)]
    frozen={'finqa|'+r['extra_info']['index']:contract.prompt_hash(r['prompt']) for r in rows}
    monkeypatch.setattr(contract,'frozen_heldout',lambda:frozen)
    converted=pools.finqa_exclusions(rows)
    assert converted[0]==dict(id='test-0',prompt=rows[0]['prompt']) and len(converted)==1147
    kept,excluded=pools.exclude([dict(id='replay',prompt=rows[0]['prompt'])],converted)
    assert not kept and excluded[0]['overlaps']=='test-0'
    leaked=copy.deepcopy(rows);leaked[0]['extra_info']['demonstration']='leaked answer'
    with pytest.raises(c.CorrectnessStop):pools.finqa_exclusions(leaked)
    changed=copy.deepcopy(rows);changed[0]['prompt'][0]['content']='changed prompt'
    with pytest.raises(c.CorrectnessStop):pools.finqa_exclusions(changed)


@pytest.mark.parametrize('kind',['id','normalized','shingle','edit','within'])
def test_inherited_generic_audit_stops_and_keeps_exclusion_receipt(tmp_path,kind):
    panel=dict(id='hellaswag-test-1',prompt='a person carefully puts many clean plates into a large kitchen cupboard today')
    candidate=dict(id='generic',prompt='an unrelated harmless instruction')
    if kind=='id':candidate['id']=panel['id']
    if kind=='normalized':candidate['prompt']=panel['prompt'].upper()
    if kind=='shingle':candidate['prompt']=panel['prompt']+' now'
    if kind=='edit':candidate['prompt']=panel['prompt'].replace('plates','platez')
    generic=[candidate]
    if kind=='within':generic.append(dict(candidate,id='another'))
    with pytest.raises(c.CorrectnessStop,match='inherited generic replay overlaps'):
        pools.audit_generic(tmp_path,generic,[panel],True)
    receipt=json.loads((tmp_path/'generic-exclusion-audit.json').read_text())
    assert receipt['passed'] is False and receipt['exclusions']


def test_v3_sequence_finqa_retention_can_supply_signal_without_math_acquisition_loss():
    rows=[]
    for seed in c.SEEDS:
        for lam in (1,.5):
            for panel in ('finqa','math_screen','ifeval','knowledge','math'):
                ncorrect=52 if lam==.5 and panel=='finqa' else 50
                rows += [dict(arm=f'v3-sequence-F-l{lam:g}',seed=seed,step=40,task=panel,panel=panel,cap=2048,
                              question_id=str(i),correct=int(i<ncorrect),output_tokens=10,finished=1) for i in range(100)]
    result=reader.analyse(rows,20)
    rule=next(r for r in result['v3_rules'] if r['route']=='sequence' and r['recipe']=='F' and r['lam']==.5)
    assert rule['acquisition']==['math_screen'] and 'finqa' in rule['retention'] and rule['exploratory_signal']
    finqa=next(r for r in result['v3_rules'] if r['route']=='finqa')
    assert finqa['acquisition']==['finqa'] and 'finqa' not in finqa['retention']


def test_probe_log_names_preserve_b_and_separate_four_arms():
    from kit.p3b_replay_train import ReplayLogName
    name=ReplayLogName('p3b-{route}-{recipe}-{seed}')
    values=dict(route='sequence',recipe='F+replay',seed=101)
    assert name.format(**values,slot_map=None)=='p3b-sequence-F+replay-101'
    assert len({name.format(**values,slot_map=a) for a in ('generic','broad','earlier','both')})==4


def test_realistic_return_sizes_omit_bulk_and_keep_gzipped_question_scores(tmp_path):
    """560k responses (~560 MB uncompressed) must stay durable; all scores return."""
    import csv,gzip,hashlib,tarfile
    from kit import next_collect as collect
    out=tmp_path/'out';bulk=out/'scores/run/raw/responses.jsonl';bulk.parent.mkdir(parents=True)
    # Nontrivial response bytes: compression of easy repeated fixture text cannot hide inclusion bugs.
    rng=random.Random(7);alphabet='abcdefghijklmnopqrstuvwxyz0123456789 '
    chunks=[''.join(rng.choices(alphabet,k=950)) for _ in range(1000)]
    with bulk.open('wb') as f:
        for i in range(560000):f.write((json.dumps(dict(id=i,response=chunks[i%1000]))+'\n').encode())
    assert bulk.stat().st_size>500_000_000
    for job in range(700):
        path=out/f'scores/job-{job}/scores.csv';path.parent.mkdir(parents=True)
        with path.open('w') as f:
            w=csv.writer(f);w.writerow('arm seed step task panel cap question_id correct output_tokens finished'.split())
            w.writerows([f'v3-finqa-F-l0.5',101,40,'finqa','finqa',2048,f'test-{i}',i%2,100+i%1900,1] for i in range(800))
    # Qwen-sized tensor-hash receipts: 180 interior blends, ~500 named tensors each.
    for job in range(180):
        tensors={f'model.layers.{i//12}.weight_{i%12}':dict(
            start=hashlib.sha256(f'start-{job//3}-{i}'.encode()).hexdigest(),
            final=hashlib.sha256(f'final-{job//3}-{i}'.encode()).hexdigest(),
            blend=hashlib.sha256(f'blend-{job}-{i}'.encode()).hexdigest(),dtype='torch.bfloat16',shape=[4096,4096]) for i in range(500)}
        c.atomic(out/f'blends/blend-{job}/tensor-hashes.json',dict(lam=.5,tensors=tensors))
    large=out/'reading/large.json';large.parent.mkdir();large.write_bytes(b'0'*(collect.LIMIT+1))
    archive=tmp_path/'return.tar.gz';collect.job(NS(out=out,archive=archive,standin=True))
    assert archive.stat().st_size<collect.CAP
    with tarfile.open(archive) as tar:
        names=tar.getnames();manifest=json.load(tar.extractfile('collect-manifest.json'))
        assert len([n for n in names if n.endswith('/scores.csv.gz')])==700
        assert not any('/raw/' in n or n.endswith('/responses.jsonl') for n in names)
        assert all(m.size<=collect.LIMIT for m in tar if m.name!='collect-manifest.json')
        assert sum(len(gzip.decompress(tar.extractfile(n).read()).splitlines())-1 for n in names if n.endswith('/scores.csv.gz'))==560000
    omissions={r['path']:r for r in manifest['omitted']}
    assert omissions['scores/run/raw/responses.jsonl']['sha256']==c.s.file_hash(bulk)
    assert omissions['reading/large.json']['bytes']==collect.LIMIT+1


def test_collect_verifies_bulk_durable_copy_before_return(tmp_path):
    from kit import next_collect as collect
    from kit.p3b_score import write_csv
    out=tmp_path/'out';durable=tmp_path/'durable'
    write_csv(out/'scores/job/scores.csv',[dict(arm='base',seed=0,step=0,task='finqa',panel='finqa',cap=2048,question_id='q',correct=1,output_tokens=10,finished=1)])
    bulk=out/'scores/job/raw/responses.jsonl';bulk.parent.mkdir();bulk.write_text('full evidence')
    durable.mkdir();archive=tmp_path/'return.tar.gz'
    args=NS(out=out,durable_root=durable,archive=archive,standin=False)
    with pytest.raises((c.CorrectnessStop,FileNotFoundError)):collect.job(args)
    assert not archive.exists()
    shutil.copytree(out,durable,dirs_exist_ok=True)
    collect.job(args)
    assert archive.exists()
    (durable/'scores/job/raw/responses.jsonl').write_text('corrupted evidence')
    with pytest.raises(c.CorrectnessStop):collect.job(args)


def test_shared_endpoint_cache_accepts_identical_export_copies(tmp_path):
    from kit.next_dryrun import model
    out=tmp_path/'out';left=tmp_path/'left';right=tmp_path/'right'
    parents=[]
    for root in (left,right):
        receipt=model(root,'same weights')
        parents.append(dict(path=str(root),receipt=str(receipt),identity=json.loads(receipt.read_text())))
    c.atomic(out/'next-inputs/inputs.json',dict(base=parents[0]));c.freeze(out/'next-inputs',['inputs.json'],True)
    calls=[]
    def score(command):
        calls.append(command)
        arm=command[command.index('--arm')+1]
        folder=out/f'scores/{arm}-s0-k0-finqa'
        (folder).mkdir(parents=True);(folder/'scores.csv').write_text('fixture scores')
        c.freeze(folder,['scores.csv'],True)
    args=NS(out=out,standin=True,gpu='0',package_a=None,finqa=None)
    for seed,parent in zip((101,102),parents):c.shared_score(args,parent,'v3-finqa-F-l0',seed,40,'finqa',score)
    assert len(calls)==1
    aliases=[json.loads(p.read_text()) for p in (out/'score-aliases').glob('*.json')]
    assert len(aliases)==2 and len({a['source'] for a in aliases})==1


def test_registration_retains_original_pin_and_appends_owner_amendment():
    mirror = c.N/'registration.md'
    text = mirror.read_text()
    original_line = text.splitlines()[44]
    assert '90242dd8c0c481249d6e24cbe6c6f2cefaf7444d' in original_line
    assert hs.REVISION not in original_line
    heading = '## Amendment, 22 October (owner: "Yes, ActivityNet-only fix is OK."; before any HellaSwag output)'
    amendment = text.split(heading)[1]
    assert hs.REVISION in amendment
    assert next(iter(hs.VALIDATION_FILES.values())) in amendment
    assert '433' in amendment and '3,243' in amendment
    assert '2b0fc85c6e80854bb47982d35a18d15bc83a5655de6202634547071eccbe487d' in amendment
    assert "partner's agent at row M2" in amendment
    original = c.KIT.parent/'docs/phase3/next-message-registration-20261020.md'
    if original.exists():
        assert mirror.read_bytes() == original.read_bytes()
