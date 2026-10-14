"""CPU-only package-B jobs; no model downloads, CUDA or trainer dependency."""
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import types
import pytest
from kit import p3b_common as c, p3b_read as reader, p3b_replay_data as replay
from kit.p3b_reset import indices, reset_tensors


def run(script, out, *args, code=0, stub=None):
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'','HF_HUB_OFFLINE':'1','HF_DATASETS_OFFLINE':'1'}
    if stub: env['PYTHONPATH']=str(stub)+os.pathsep+str(c.KIT.parent)
    result=subprocess.run([sys.executable,str(c.KIT / ('p3b_'+script+'.py')),'--out',str(out),'--standin',*map(str,args)],env=env,text=True,capture_output=True)
    assert result.returncode==code,result.stdout+result.stderr
    return result.stdout+result.stderr


def source(tmp_path):
    rng=random.Random(2); rows=[]
    for i in range(70):
        rows.append(dict(id=str(i),prompt=''.join(rng.choices('abcdefghijklmnopqrstuvwxyz ',k=100)),ground_truth=json.dumps([dict(func_name='validate_uppercase',N=None)]),fake_response='HELLO'))
    file=tmp_path / 'source.jsonl';c.t.write_jsonl(file,rows)
    # langdetect is absent on this CPU host. Exercise real non-language checker
    # functions; language dispatch deliberately raises if accidentally reached.
    stub=tmp_path / 'deps';stub.mkdir();(stub / 'langdetect.py').write_text('def detect(text): raise RuntimeError("language detector stand-in must not be called")\n')
    return file,stub


def test_end_to_end_and_skip(tmp_path):
    out=tmp_path / 'out';file,stub=source(tmp_path)
    args=['--rows',file]
    run('replay_data',out,*args,stub=stub)
    assert 'SKIP' in run('replay_data',out,*args,stub=stub)
    c.frozen_files(out / 'replay')
    for arm in ('D','F'):
        args=['--arm',arm,'--size',64,'--seed',101]
        run('recover',out,*args);assert 'SKIP' in run('recover',out,*args)
    schedule=tmp_path / 'schedule.jsonl';c.t.write_jsonl(schedule,[dict(id=str(i),prompt='task',response='target') for i in range(1280)])
    manifest=tmp_path / 'schedule.json';c.atomic(manifest,dict(seed=101,files={schedule.name:c.s.file_hash(schedule)}))
    for recipe in ('D+replay','F+replay'):
        args=['--recipe',recipe,'--route','finqa','--seed',101,'--task-schedule',schedule,'--schedule-manifest',manifest]
        run('replay_train',out,*args);assert 'SKIP' in run('replay_train',out,*args)
    for panel in ('finqa','general-v1'):
        for arm,step in [('base',0),('D',40),('recover-D-n64',4)]:
            args=['--arm',arm,'--seed',101,'--step',step,'--panel-set',panel]
            run('score',out,*args);assert 'SKIP' in run('score',out,*args)
    reading=tmp_path / 'reading';args=['--source',out,'--bootstrap',20]
    run('read',reading,*args);assert 'SKIP' in run('read',reading,*args)
    archive=tmp_path / 'return.tar.gz';args=['--archive',archive]
    run('collect',out,*args);assert 'SKIP' in run('collect',out,*args)
    run('read',tmp_path / 'returned','--source',archive,'--bootstrap',20)
    renamed=tmp_path / 'renamed.csv.gz';renamed.write_bytes((out / 'return/scores.csv.gz').read_bytes())
    run('read',tmp_path / 'csv-reading','--source',renamed,'--bootstrap',20)
    assert json.loads((reading / 'reading.json').read_text())['summaries']==json.loads((tmp_path / 'csv-reading/reading.json').read_text())['summaries']
    assert json.loads((reading / 'reading.json').read_text())==json.loads((tmp_path / 'returned/reading.json').read_text())
    trigger=tmp_path / 'trigger.json';c.atomic(trigger,dict(reset_recipe='D'))
    args=['--trigger',trigger,'--recipe','D','--seed',101]
    run('reset',out,*args);assert 'SKIP' in run('reset',out,*args)
    assert list((out / 'logs').glob('p3b-*.log'))


def test_targets_reuses_a_journal(tmp_path,monkeypatch):
    from test_p3a_partner import fixture, run as arun
    source=fixture(tmp_path / 'fixture'); folder=source / 'prepared/code'
    train=c.t.read_jsonl(folder / 'train.jsonl'); extra=dict(train[0],id='code-easy-100',prompt='Extra training-only successor problem')
    gold=json.loads(extra['gold']);gold['problem_id']=extra['id'];extra['gold']=json.dumps(gold)
    c.t.write_jsonl(folder / 'train.jsonl',train+[extra])
    tests=c.t.read_jsonl(folder / 'tests.jsonl');c.t.write_jsonl(folder / 'tests.jsonl',tests+[dict(tests[0],problem_id=extra['id'])])
    seal=json.loads((source / 'frozen-annex.json').read_text())
    seal['manifest']['candidates']['code']['files'].update({n:c.s.file_hash(folder/n) for n in ('train.jsonl','tests.jsonl')})
    seal['manifest_sha256']=c.s.digest(seal['manifest']);c.atomic(source / 'frozen-annex.json',seal);c.atomic(source / 'prepared/frozen.json',seal['manifest'])
    step=json.loads((source/'code-step1.json').read_text());step['freeze_sha256']=seal['manifest_sha256'];c.atomic(source/'code-step1.json',step)
    package=tmp_path / 'a'; arun('p3a_inputs.py',package,'--sources',source)
    run('targets',tmp_path / 'pending','--package-a',package,code=3)
    for phase in ('teacher','rewrite'): arun('p3a_'+phase+'.py',package)
    arun('p3a_decide.py',package)
    out=tmp_path / 'b';args=['--package-a',package]
    raw=package / c.OUTPUTS / 'code/teacher/raw' if hasattr(c,'OUTPUTS') else package / '3c/report-screen/code/teacher/raw'
    before={p.name:p.read_bytes() for p in raw.glob('*.json')}
    run('targets',out,*args);assert 'SKIP' in run('targets',out,*args)
    assert before and all((out / 'targets/teacher/raw' / n).read_bytes()==v for n,v in before.items())
    assert len(c.t.read_jsonl(out / 'targets/schedule-101.jsonl'))==1280
    info=json.loads((out/'targets/inputs.json').read_text())
    assert info['tests_sha256']==c.s.file_hash(Path(info['tests_path']))
    monkeypatch.setenv('CODE_TESTS',info['tests_path'])
    from kit.p3b_reward import compute_score
    assert compute_score('livecodebench','```python\nprint(int(input()) + 1)\n```',extra['gold'])['score']==1
    assert len(c.t.read_jsonl(out / 'targets/pool.jsonl'))==101
    assert len(list((out / 'targets/teacher/raw').glob('*.json')))==len(before)+4


def test_exit_codes(tmp_path):
    out=tmp_path / 'out';file,stub=source(tmp_path)
    run('recover',out,'--arm','D','--size',16,'--seed',101,code=3)
    run('replay_data',out,'--rows',file,'--source-revision','wrong',code=2,stub=stub)
    rows=c.t.read_jsonl(file);c.t.write_jsonl(file,rows[:64])
    run('replay_data',out,'--rows',file,code=2,stub=stub)
    verdicts=(out/'replay/candidate-checks.jsonl').read_bytes()
    run('replay_data',out,'--rows',file,code=2,stub=stub)
    assert (out/'replay/candidate-checks.jsonl').read_bytes()==verdicts
    bad=tmp_path / 'bad.jsonl';bad.write_text('{bad json\n')
    result=run('replay_data',tmp_path / 'crash','--rows',bad,code=1,stub=stub)
    assert 'Traceback' in result
    run('score',out,'--arm','base','--seed',101,'--step',0,'--panel-set','finqa','--gpu','0,1',code=2)
    run('collect',out,'--archive',tmp_path / 'empty.tar.gz',code=3)


def test_composition_loss_reset_and_dedup(monkeypatch):
    import torch
    from kit.p3b_loss import mixed_loss
    task=list(range(1280));generic=['a','b','c']
    rows=c.mixed(task,generic,101);order=(['b','a','c']*54)[:160]
    assert [r for i,r in enumerate(rows) if i%32 in (7,15,23,31)]==order
    assert all(r==i for i,r in enumerate(rows) if i%32 not in (7,15,23,31))
    x=torch.tensor([2.,4.],requires_grad=True);y=torch.tensor([10.,14.],requires_grad=True)
    loss=mixed_loss(x,y);assert loss.item()==pytest.approx(4.125)
    loss.backward();assert x.grad.tolist()==pytest.approx([.4375,.4375]);assert y.grad.tolist()==pytest.approx([.0625,.0625])
    names={'gate_proj':(200,3),'up_proj':(200,3),'down_proj':(3,200)}
    state={f'model.layers.0.mlp.{n}.weight':torch.ones(shape) for n,shape in names.items()}
    state['model.layers.0.self_attn.q_proj.weight']=torch.ones(3,3)
    base={k:torch.zeros_like(v) for k,v in state.items()};record=reset_tensors(state,base)
    assert indices('0',200)==[34,159] and indices('1',200)==[52,177]
    assert set(record)=={f'model.layers.0.mlp.{n}.weight' for n in names}
    for k,v in state.items():
        changed=torch.zeros_like(v,dtype=torch.bool)
        if k in record:
            if '.down_proj.' in k:changed[:,[34,159]]=True
            else:changed[[34,159],:]=True
        assert torch.equal(v==0,changed)
    assert replay.duplicate('Ａ B  c','a b c') and replay.duplicate('cat dogsss','cat dogss')
    assert replay.duplicate('one two three four five six seven eight nine ten','one two three four five six seven eight nine ten eleven')
    assert not replay.duplicate('cat','a completely different prompt')
    kept,excluded=replay.clean([dict(id='x',prompt='a b c'),dict(id='y',prompt='Ａ B C'),dict(id='panel',prompt='unrelated')],[dict(id='panel',prompt='held out')])
    assert len(kept)==1 and len(excluded)==2
    monkeypatch.setitem(sys.modules,'langdetect',types.SimpleNamespace(detect=lambda text: 'en'))
    assert all(r['passed'] for r in replay.checker('HELLO',json.dumps([dict(func_name='validate_uppercase',N=None),dict(func_name='verify_keywords',keyword_list=['hello'])])))
    assert not replay.checker('has, comma',dict(func_name='validate_no_commas'))[0]['passed']
    with pytest.raises(c.CorrectnessStop):replay.checker('x',dict(func_name='unknown'))


def test_reader_bars_and_trigger():
    rows=[]
    def add(arm,seed,step,panel,n,correct):
        for q in range(n):rows.append(dict(arm=arm,seed=seed,step=step,panel=panel,task=panel,cap=2048,question_id=str(q),correct=int(q<correct),output_tokens=10,finished=1))
    for p in ('ifeval','finqa','knowledge','math'):add('base',0,0,p,100,80)
    for seed in range(101,104):
        for p in ('ifeval','finqa','knowledge','math'):
            add('D',seed,40,p,100,60 if p=='ifeval' else 80)
            add('recover-D-n64',seed,4,p,100,78 if p=='ifeval' else 80)
    for seed in range(101,106):
        for step in (0,10,20,30,40):
            add('standalone-D',seed,step,'code',51,step//2)
            add('sequence-D',seed,step,'code',51,step//5)
    result=reader.analyse(rows,20)
    assert next(x for x in result['recovery'] if x['arm']=='D' and x['size']==64)['near_base']
    assert result['reset_recipe'] is None and result['reset_pending']
    # The higher-priority selected-task comparison must be complete and ruled out.
    for seed in range(101,106):
        for step in (0,10,20,30,40):
            add('standalone-D+replay',seed,step,'code',51,step//2)
            add('sequence-D+replay',seed,step,'code',51,step//2)
    pending=reader.analyse([r for r in rows if not (r['arm']=='sequence-D+replay' and r['seed']==105)],20)
    assert pending['reset_recipe'] is None and pending['reset_pending']
    result=reader.analyse(rows,20)
    assert result['reset_recipe']=='D'
    assert not result['reset_pending']
    assert next(x for x in result['reset_deficits'] if x['recipe']=='D' and x['panel']=='code')['both_wrong'][0]['n']==51
    higher=[dict(r,correct=int(int(r['question_id'])<r['step']//5)) if r['arm']=='sequence-D+replay' else r for r in rows]
    assert reader.analyse(higher,20)['reset_recipe']=='D+replay'
    for row in rows:
        if row['arm']=='recover-D-n64' and row['panel']=='knowledge':row['correct']=0
    failed=reader.analyse(rows,20)
    assert not next(x for x in failed['recovery'] if x['arm']=='D' and x['size']==64)['useful']
    assert reader.bound([0,0,0],3)['lower']==0
    assert reader.bound([1,2],3)['mean'] is None


def test_replay_ce_matches_full_cross_entropy():
    import torch
    from kit.p3b_loss import replay_loss
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__();self.embed=torch.nn.Embedding(7,3);self.head=torch.nn.Linear(3,7,bias=False)
        def get_output_embeddings(self):return self.head
        def forward(self,input_ids,**kwargs):return types.SimpleNamespace(logits=self.head(self.embed(input_ids)))
    torch.manual_seed(1);model=Tiny()
    inputs=dict(input_ids=torch.tensor([[1,2,3,4]]),responses=torch.tensor([[3,4]]),attention_mask=torch.ones(1,4,dtype=torch.long))
    target=[5,6]  # EOS is the last supervised label, with no first-three-token mask.
    loss=replay_loss(model,inputs,target)
    full=model(torch.tensor([[1,2,5,6]])).logits[:,1:3]
    expected=torch.nn.functional.cross_entropy(full.reshape(-1,7),torch.tensor(target))
    assert torch.allclose(loss,expected)
    actual=torch.autograd.grad(loss,tuple(model.parameters()),retain_graph=True)
    reference=torch.autograd.grad(expected,tuple(model.parameters()))
    assert all(torch.allclose(a,b) for a,b in zip(actual,reference))


def test_replay_teacher_padding_uses_task_context_only():
    import torch
    from kit.p3b_loss import mixed_teacher_inputs
    class Tokenizer:
        pad_token_id=0
        def apply_chat_template(self,messages,**kwargs):return ' '.join(m['content'] for m in messages)
        def __call__(self,text,**kwargs):return {'input_ids':[1]*len(text)}
    prompts=[[dict(role='user',content='Solve the task')],[dict(role='user',content='Repeat HELLO')]]
    built=mixed_teacher_inputs(Tokenizer(),prompts,['worked solution','HELLO'],torch.ones(2,4,dtype=torch.long),
        torch.ones(2,4,dtype=torch.long),replay=[False,True],max_reprompt_len=1024)
    assert built['teacher_attention_mask'][0].sum()>0
    for key in ('teacher_input_ids','teacher_attention_mask','teacher_position_ids'):
        assert torch.equal(built[key][0],built[key][1])
    assert built['teacher_prompt_lengths'][1]==built['teacher_prompt_lengths'][0]
    assert built['self_distillation_mask'].tolist()==[1.,1.]


def test_generic_training_text_and_eos_only_targets():
    from kit.p3b_train import training_rows
    class Tokenizer:
        pad_token_id=0; eos_token='<eos>'
        def encode(self,text,**kwargs):return [1]*len(text)
        def apply_chat_template(self,messages,**kwargs):return ' '.join(m['content'] for m in messages)
    rows=training_rows([dict(id='r',prompt='[This is plain prompt text]',response='',p3b_replay=True)],Tokenizer(),'D')
    assert rows[0]['prompt'][0]['content']=='[This is plain prompt text]'
    assert rows[0]['extra_info']['p3b_target_ids'] and rows[0]['extra_info']['demonstration']==''
    with pytest.raises(c.CorrectnessStop):
        training_rows([dict(id='bad',prompt=[dict(role='user',content='Answer')],response='Answer')],Tokenizer(),'D')


def test_language_checker_validates_without_detection(monkeypatch):
    class LangDetectException(Exception): pass
    calls=[]
    def detect(text):
        calls.append(text)
        if not text or text=='12345': raise LangDetectException('No features in text')
        return 'en'
    monkeypatch.setitem(sys.modules,'langdetect',types.SimpleNamespace(detect=detect,LangDetectException=LangDetectException))
    gt=dict(func_name='validate_response_language',language='en')
    assert replay.checker(None,gt,validate_only=True)==[] and calls==[]
    for bad in (dict(gt,extra=True),dict(func_name='validate_response_language')):
        with pytest.raises(c.CorrectnessStop,match='malformed checker arguments'):
            replay.checker(None,bad,validate_only=True)
    assert calls==[]
    assert replay.checker('English words',gt)[0]['passed']
    assert not replay.checker('',gt)[0]['passed']
    assert not replay.checker('12345',gt)[0]['passed']
    c.check(c.R/'if_functions.py',replay.CHECKER)


@pytest.mark.parametrize('incoming,refreshed',[(75,80),(70,80)])
def test_nonpositive_loss_recovery_is_undefined(incoming,refreshed):
    rows=[]
    for panel in ('ifeval','finqa','knowledge','math'):
        arms=[('base',0,0,70)]+[(arm,seed,step,score) for seed in range(101,104)
            for arm,step,score in [('D',40,incoming if panel=='ifeval' else 70),
                                  ('recover-D-n64',4,refreshed if panel=='ifeval' else 70)]]
        for arm,seed,step,score in arms:
            rows.extend(dict(arm=arm,seed=seed,step=step,panel=panel,cap=2048,question_id=str(q),
                             correct=int(q<score),output_tokens=10,finished=1) for q in range(100))
    result=next(r for r in reader.analyse(rows,20)['recovery'] if r['arm']=='D' and r['size']==64)
    assert result['useful'] is None and result['near_base'] is None
    assert result['status']=='UNDEFINED_RECOVERY' and result['protection']
    assert result['bars']['ifeval']['mean']==refreshed-incoming
    assert all(s['recovery_fraction'] is None for s in result['seeds'])


@pytest.mark.parametrize('recipe,steps',[('D',40),('F',4)])
@pytest.mark.parametrize('evidence',['metrics','checkpoint','checkpoint_before_metrics'])
def test_training_completion_without_parent_receipt(tmp_path,recipe,steps,evidence):
    from kit.p3b_train import completed_training
    root=tmp_path/('sdft' if recipe=='D' else 'train')
    checkpoint=root/f'global_step_{steps}'
    if recipe=='D': checkpoint/='actor'
    checkpoint.mkdir(parents=True)
    c.atomic(tmp_path/'started.json',dict(seed=101))
    if evidence=='metrics':
        c.t.write_jsonl(tmp_path/'metrics.jsonl',[dict(data={'v4/completed_optimizer_updates':steps})])
        (root/'latest_checkpointed_iteration.txt').write_text('')  # Interrupted tracker write cannot hide the final metrics.
    else:
        (root/'latest_checkpointed_iteration.txt').write_text(str(steps))
        if evidence=='checkpoint_before_metrics':
            c.t.write_jsonl(tmp_path/'metrics.jsonl',[dict(data={'v4/completed_optimizer_updates':steps-1})])
    assert completed_training(tmp_path,recipe,steps)
    assert json.loads((tmp_path/'trained.json').read_text())['seconds'] is None
    assert checkpoint.exists() and not (tmp_path/'partials').exists()


@pytest.fixture
def trainer_job(tmp_path,monkeypatch):
    from kit import p3b_train as train, v4_sft
    out=tmp_path/'out'; root=out/'replay'; root.mkdir(parents=True)
    c.t.write_jsonl(root/'refresher-64.jsonl',[dict(id=str(i),prompt='q',response='a') for i in range(64)])
    c.atomic(root/'manifest.json',c.files_manifest(root,['refresher-64.jsonl'],technical_synthetic=False))
    args=types.SimpleNamespace(out=out,seed=101,arm='base',size=64,standin=False,
        gpu='0,1,2,3,4,5,6,7',sdpo=tmp_path,tokenizer=str(tmp_path),model=str(tmp_path),
        incoming_receipt=None,trainer_python=sys.executable)
    monkeypatch.setattr(train,'frozen_hashes',lambda:None)
    monkeypatch.setattr(train,'identity',lambda *a,**k:dict(model_file_hashes={'stub':'hash'}))
    monkeypatch.setattr(train.subprocess,'check_output',lambda *a,**k:'7c457fc1b1f636ae794eb0362ba37d4743b06fbc')
    monkeypatch.setitem(sys.modules,'transformers',types.SimpleNamespace(AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *a,**k:None)))
    monkeypatch.setattr(train,'training_rows',lambda rows,*a:rows)
    pa=types.ModuleType('pyarrow');pa.Table=types.SimpleNamespace(from_pylist=lambda values:values)
    pq=types.ModuleType('pyarrow.parquet');pq.write_table=lambda values,path:path.write_bytes(b'CPU test parquet')
    monkeypatch.setitem(sys.modules,'pyarrow',pa);monkeypatch.setitem(sys.modules,'pyarrow.parquet',pq)
    monkeypatch.setattr(v4_sft,'complete_export',lambda path:(path/'test-export').exists())
    run=out/'runs/recover-base-n64-s101'; calls=[]; mode=['complete']
    def invoke(command,**kwargs):
        if '--resolve' in command:
            kwargs['stdout'].write('resolved: CPU test stub\n'); return types.SimpleNamespace(returncode=0)
        if 'verl.model_merger' in command:
            export=Path(command[command.index('--target_dir')+1]); export.mkdir();(export/'test-export').write_text('ok')
            return types.SimpleNamespace(returncode=0)
        calls.append(command)
        completed=4 if mode[0] in ('complete','missing-final') else 3
        c.t.write_jsonl(run/'metrics.jsonl',[dict(data={'p3b/completed_optimizer_updates':completed})])
        if mode[0] in ('complete','bad-dose'):
            (run/'train/global_step_4').mkdir(parents=True)
            (run/'train/latest_checkpointed_iteration.txt').write_text('4')
        return types.SimpleNamespace(returncode=1 if mode[0]=='interrupted' else 0)
    monkeypatch.setattr(train.subprocess,'run',invoke)
    return train,args,run,calls,mode


@pytest.mark.parametrize('receipt',['trainer-exit.json','trained.json'])
def test_completed_trainer_crash_proceeds_to_exports(trainer_job,monkeypatch,receipt):
    train,args,run,calls,mode=trainer_job
    atomic=train.atomic
    def crash(path,value):
        if path.name==receipt: raise RuntimeError('parent died after final checkpoint')
        atomic(path,value)
    monkeypatch.setattr(train,'atomic',crash)
    with pytest.raises(RuntimeError,match='parent died'): train.job(args,recovery=True)
    assert len(calls)==1
    monkeypatch.setattr(train,'atomic',atomic)
    train.job(args,recovery=True)
    assert len(calls)==1 and (run/'complete.json').exists()
    assert not (run/'partials').exists()
    summary=json.loads((run/'summary.json').read_text())
    if receipt=='trainer-exit.json': assert summary['seconds'] is None and summary['gpu_hours'] is None


def test_successful_bad_dose_stays_correctness_stop(trainer_job,monkeypatch):
    train,args,run,calls,mode=trainer_job; mode[0]='bad-dose'
    for _ in range(2):
        with pytest.raises(c.CorrectnessStop,match='registered optimizer dose'):
            train.job(args,recovery=True)
    assert len(calls)==1 and not (run/'partials').exists()
    # Exercise the public exit mapping as well as the in-memory job path.
    monkeypatch.setattr(sys,'argv',['p3b-recover','--out',str(args.out)])
    assert c.entry(lambda a:train.job(args,recovery=True),lambda p:None,'test-dose-stop')==2
    assert len(calls)==1


@pytest.mark.parametrize('failure',['interrupted','missing-final'])
def test_interrupted_training_restarts_original_input(trainer_job,failure):
    train,args,run,calls,mode=trainer_job; mode[0]=failure
    with pytest.raises(subprocess.CalledProcessError if failure=='interrupted' else c.NotReady):
        train.job(args,recovery=True)
    mode[0]='complete'; train.job(args,recovery=True)
    assert len(calls)==2 and calls[0]==calls[1]
    kept=list((run/'partials').iterdir()); assert len(kept)==1
    assert (kept[0]/'metrics.jsonl').exists() and (kept[0]/'trainer-exit.json').exists()
    assert (run/'complete.json').exists()


class _MixedBatch:
    def __init__(self,flags):
        import torch
        n=len(flags);ids=torch.tensor([[1,2,3,4,5,6]]).expand(n,-1).clone()
        self.batch=dict(input_ids=ids,attention_mask=torch.ones_like(ids),position_ids=torch.arange(6).expand(n,-1),
            responses=ids[:,-4:],response_mask=torch.ones(n,4),self_distillation_mask=torch.ones(n),
            teacher_input_ids=ids.clone(),teacher_attention_mask=torch.ones_like(ids),teacher_position_ids=torch.arange(6).expand(n,-1),
            old_log_probs=torch.zeros(n,4),rollout_log_probs=torch.zeros(n,4))
        self.non_tensor_batch={'extra_info':[dict(p3b_replay=f,p3b_target_ids=[6,5,4,3]) for f in flags]}
        self.meta_info={'temperature':1.}
    def split(self,size):
        if size==len(self.non_tensor_batch['extra_info']): return [self]
        assert size==1
        batches=[]
        for i,extra in enumerate(self.non_tensor_batch['extra_info']):
            batch=_MixedBatch([extra['p3b_replay']]); batch.batch={k:v[i:i+1] for k,v in self.batch.items()}
            batches.append(batch)
        return batches
    def to(self,device): return self


def _mixed_actor_update(flags,distributed=False):
    import torch
    from kit import sdft_actor
    calls=[]
    class Tiny(torch.nn.Module):
        def __init__(self,label):
            super().__init__(); self.embed=torch.nn.Embedding(8,3);self.head=torch.nn.Linear(3,8,bias=False);self.label=label
        def get_output_embeddings(self): return self.head
        def forward(self,input_ids,**kwargs):
            calls.append(self.label)
            if distributed:
                tag=torch.tensor([int(self.label=='actor')]); gathered=[torch.empty_like(tag) for _ in range(2)]
                torch.distributed.all_gather(gathered,tag)
                assert all(torch.equal(tag,t) for t in gathered),'ranks entered different model forwards'
            return types.SimpleNamespace(logits=self.head(self.embed(input_ids)))
    torch.manual_seed(1); actor=Tiny('actor');teacher=Tiny('teacher')
    optimizer=torch.optim.SGD(actor.parameters(),lr=.1)
    sd=dict(full_logit_distillation=True,alpha=0.,is_clip=None,teacher_regularization='ema')
    owner=types.SimpleNamespace(actor_module=actor,teacher_module=teacher,actor_optimizer=optimizer,scaler=None,
        config=types.SimpleNamespace(self_distillation=sd,loss_agg_mode='seq-mean-token-mean',ppo_epochs=1,ppo_mini_batch_size=len(flags)),
        ulysses_sequence_parallel_size=1,_update_teacher=lambda:None)
    def step():
        norm=torch.nn.utils.clip_grad_norm_(actor.parameters(),1.); optimizer.step(); return norm
    owner._optimizer_step=step
    metrics=sdft_actor.update_policy(owner,_MixedBatch(flags))
    assert calls==['teacher','actor']*len(flags)
    assert all(p.grad is None for p in teacher.parameters())
    assert metrics['p3b/replay_sft_loss']>0 and metrics['actor/pg_loss']>0
    return calls


def test_mixed_actor_calls_teacher_and_actor_once_per_item(monkeypatch):
    from kit import sdft_actor
    monkeypatch.setenv('P3B_REPLAY','1')
    project=sdft_actor.projected_distillation; objectives=[]
    def counted(*args,**kwargs):
        objectives.append(1); return project(*args,**kwargs)
    monkeypatch.setattr(sdft_actor,'projected_distillation',counted)
    flags=[i in (7,15,23,31) for i in range(32)]
    assert len(_mixed_actor_update(flags))==64
    assert len(objectives)==28  # Four replay projections never enter the KL objective.


def _gloo_mixed_worker(rank,root):
    import datetime
    import torch
    os.environ['P3B_REPLAY']='1'
    if sys.platform=='darwin': os.environ.setdefault('GLOO_SOCKET_IFNAME','lo0')
    try:
        torch.distributed.init_process_group('gloo',init_method='file://'+str(Path(root)/'rendezvous'),
            rank=rank,world_size=2,timeout=datetime.timedelta(seconds=20))
    except RuntimeError as exc:
        if 'Operation not permitted' not in str(exc): raise
        (Path(root)/f'blocked-{rank}').write_text(str(exc));return
    try:
        # Different replay placement across ranks exercises the former divergence.
        flags=[i in ((0,1) if rank==0 else (14,15)) for i in range(16)]
        calls=_mixed_actor_update(flags,distributed=True)
        (Path(root)/f'calls-{rank}.json').write_text(json.dumps(calls))
    finally: torch.distributed.destroy_process_group()


def test_two_process_gloo_mixed_actor_sequence(tmp_path):
    import socket
    import torch
    if not torch.distributed.is_gloo_available(): pytest.skip('CPU gloo backend unavailable')
    # macOS gloo/libuv aborts the process on a denied bind instead of raising.
    try:
        with socket.socket() as probe: probe.bind(('127.0.0.1',0))
    except PermissionError as exc: pytest.skip('sandbox denies gloo local sockets: '+str(exc))
    torch.multiprocessing.spawn(_gloo_mixed_worker,args=(str(tmp_path),),nprocs=2,join=True)
    blocked=list(tmp_path.glob('blocked-*'))
    if blocked: pytest.skip('sandbox denies gloo local sockets: '+blocked[0].read_text())
    calls=[json.loads((tmp_path/f'calls-{rank}.json').read_text()) for rank in range(2)]
    assert calls[0]==calls[1]==['teacher','actor']*16


def test_empty_checkpoint_tracker_is_an_interruption_not_a_crash(tmp_path):
    from kit import p3b_train as train
    run=tmp_path/'run'; (run/'train').mkdir(parents=True)
    (run/'train'/'latest_checkpointed_iteration.txt').write_text('')   # write interrupted
    assert train.completed_training(run,'F',4) is False
