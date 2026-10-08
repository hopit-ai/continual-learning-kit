"""Executable phase-0 closure in the public export, with no private inputs."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import pytest

ROOT=Path(__file__).resolve().parents[1]
PRIVATE_INPUTS=('V4_REFERENCE_ROOT','SDPO_DIR','FINQA_ROOT','QWEN3_8B_TOKENIZER')
EVAL_SOURCE={'pre_text': ['.'], 'post_text': ['.'], 'filename': 'STT/2014/page_54.pdf', 'table_ori': [['', '2009', '2010', '2011', '2012', '2013', '2014'], ['State Street Corporation', '$100', '$107', '$114', '$101', '$120', '$190'], ['S&P 500 Index', '100', '115', '132', '135', '157', '208'], ['S&P Financial Index', '100', '112', '126', '104', '135', '183'], ['KBW Bank Index', '100', '123', '152', '117', '153', '211']], 'table': [['', '2009', '2010', '2011', '2012', '2013', '2014'], ['state street corporation', '$ 100', '$ 107', '$ 114', '$ 101', '$ 120', '$ 190'], ['s&p 500 index', '100', '115', '132', '135', '157', '208'], ['s&p financial index', '100', '112', '126', '104', '135', '183'], ['kbw bank index', '100', '123', '152', '117', '153', '211']], 'qa': {'question': 'what is the roi of an investment in state street corporation from 20011 to 2012?', 'answer': '-11.4%', 'explanation': '', 'ann_table_rows': [1], 'ann_text_rows': [], 'steps': [{'op': 'minus2-1', 'arg1': '101', 'arg2': '114', 'res': '13'}, {'op': 'divide2-2', 'arg1': '#0', 'arg2': '114', 'res': '-11.4%'}], 'program': 'subtract(101, 114), divide(#0, 114)', 'gold_inds': {'table_1': 'the state street corporation of 2009 is $ 100 ; the state street corporation of 2010 is $ 107 ; the state street corporation of 2011 is $ 114 ; the state street corporation of 2012 is $ 101 ; the state street corporation of 2013 is $ 120 ; the state street corporation of 2014 is $ 190 ;'}, 'exe_ans': -0.11404, 'tfidftopn': {'table_2': 'The s&p 500 index of 2009 is 100 ; The s&p 500 index of 2010 is 115 ; The s&p 500 index of 2011 is 132 ; The s&p 500 index of 2012 is 135 ; The s&p 500 index of 2013 is 157 ; The s&p 500 index of 2014 is 208 ;', 'table_3': 'The s&p financial index of 2009 is 100 ; The s&p financial index of 2010 is 112 ; The s&p financial index of 2011 is 126 ; The s&p financial index of 2012 is 104 ; The s&p financial index of 2013 is 135 ; The s&p financial index of 2014 is 183 ;'}, 'program_re': 'divide(subtract(101, 114), 114)', 'model_input': [['table_1', 'the state street corporation of 2009 is $ 100 ; the state street corporation of 2010 is $ 107 ; the state street corporation of 2011 is $ 114 ; the state street corporation of 2012 is $ 101 ; the state street corporation of 2013 is $ 120 ; the state street corporation of 2014 is $ 190 ;'], ['table_2', 'the s&p 500 index of 2009 is 100 ; the s&p 500 index of 2010 is 115 ; the s&p 500 index of 2011 is 132 ; the s&p 500 index of 2012 is 135 ; the s&p 500 index of 2013 is 157 ; the s&p 500 index of 2014 is 208 ;'], ['table_4', 'the kbw bank index of 2009 is 100 ; the kbw bank index of 2010 is 123 ; the kbw bank index of 2011 is 152 ; the kbw bank index of 2012 is 117 ; the kbw bank index of 2013 is 153 ; the kbw bank index of 2014 is 211 ;']]}, 'id': 'STT/2014/page_54.pdf-2', 'table_retrieved': [{'score': 3.0496225357055664, 'ind': 'table_1'}, {'score': -2.087226390838623, 'ind': 'table_4'}, {'score': -2.1553492546081543, 'ind': 'table_2'}, {'score': -2.4109296798706055, 'ind': 'table_3'}, {'score': -2.6947011947631836, 'ind': 'table_0'}], 'text_retrieved': [], 'table_retrieved_all': [{'score': 3.0496225357055664, 'ind': 'table_1'}, {'score': -2.087226390838623, 'ind': 'table_4'}, {'score': -2.1553492546081543, 'ind': 'table_2'}, {'score': -2.4109296798706055, 'ind': 'table_3'}, {'score': -2.6947011947631836, 'ind': 'table_0'}], 'text_retrieved_all': [{'score': -3.1654834747314453, 'ind': 'text_0'}, {'score': -3.1654834747314453, 'ind': 'text_1'}]}


def test_eval_admission_in_v4_finqa(tmp_path,monkeypatch):
    """A scoring row must execute held-out admission without private docs or reference inputs."""
    from kit import eval_bed
    from kit.beds import finqa
    for key in PRIVATE_INPUTS:monkeypatch.delenv(key,raising=False)
    root=tmp_path/'v4_finqa';root.mkdir();(root/'test.json').write_text(json.dumps([EVAL_SOURCE]))
    monkeypatch.setitem(finqa.SPLIT_SIZES,'test',1)  # owned single-row admission panel
    old_text=Path.read_text
    def read_text(path,*a,**kw):
        assert not path.is_relative_to(ROOT/'docs'),'admission must not read private docs'
        return old_text(path,*a,**kw)
    monkeypatch.setattr(Path,'read_text',read_text)
    args=SimpleNamespace(root=str(root),bed='finqa',split='test',limit=0)
    items=eval_bed.items_of(finqa,args)
    assert len(items)==1 and items[0]['id']==EVAL_SOURCE['id']
    from kit.v4_contract import check_eval_items
    with pytest.raises(ValueError,match='frozen question-only'):
        check_eval_items([{**items[0],'prompt':items[0]['prompt']+'changed'}],'finqa','test')


@pytest.mark.parametrize('role',['generate','rewrite'])
def test_standin_teacher_and_rewrite_manifest(tmp_path,monkeypatch,role):
    """Both real generation CLI paths must construct manifests without Chemistry/private resources for FinQA."""
    from kit import v4_teacher as t
    for key in PRIVATE_INPUTS:monkeypatch.delenv(key,raising=False)
    class Tokenizer:
        eos_token_id=1
        eos_token="<eos>"
        def encode(self,text,**kw):return list(range(len(text.split())))
        def apply_chat_template(self,messages,tokenize=False,**kw):
            text=' '.join(m['content'] for m in messages)
            return self.encode(text) if tokenize else text
    model=tmp_path/('a'*40);model.mkdir()
    for name in ('config.json','tokenizer.json','tokenizer_config.json'):(model/name).write_text('{}')
    item={'id':'standin-train','task':'finqa','question':'Compute 5 minus 2.', 'messages':[{'role':'user','content':'Compute 5 minus 2.'}],'gold':'3','idx':'standin-train','split':'train'}
    row={'prompt':item['messages'],'reward_model':{'ground_truth':'3'},'extra_info':{'index':item['id']}}
    # Stand in only for external data/model/tokenizer; provenance, manifests,
    # verification, generation journal and rewrite construction remain real.
    monkeypatch.setattr(t,'authors_rows',lambda *a,**kw:[row])
    monkeypatch.setattr(t,'load_student_tokenizer',lambda *a:Tokenizer())
    monkeypatch.setattr(t,'VLLMGenerator',lambda *a,**kw:lambda messages,attempt,**options:{'text':'Read the figures and subtract the old amount from the new amount.\nAnswer: 3','attempt':attempt,'finish_reason':'stop'})
    # On the private tree, simulate the public tree's absent docs/scripts.
    old_bytes=Path.read_bytes;old_text=Path.read_text
    def external(path):return path.is_relative_to(ROOT/'docs') or path.is_relative_to(ROOT/'scripts')
    def read_bytes(path,*a,**kw):
        assert not external(path),'runtime read outside public kit: '+str(path)
        return old_bytes(path,*a,**kw)
    def read_text(path,*a,**kw):
        assert not external(path),'runtime read outside public kit: '+str(path)
        return old_text(path,*a,**kw)
    monkeypatch.setattr(Path,'read_bytes',read_bytes);monkeypatch.setattr(Path,'read_text',read_text)
    pool=tmp_path/'pool.jsonl';t.write_jsonl(pool,[item])
    teacher=tmp_path/'teacher'
    common=['--model',str(model),'--task','finqa','--pool',str(pool),'--tokenizer-8b',str(model),'--stand-in-teacher']
    assert t.main(['generate',*common,'--out',str(teacher)])==0
    out=teacher
    if role=='rewrite':
        out=tmp_path/'rewrite';assert t.main(['rewrite',*common,'--out',str(out),'--demonstrations',str(teacher)])==0
    manifest=json.loads((out/'manifest.json').read_text())
    assert manifest['taxonomy_sha256'] is None and manifest['stand_in_teacher']
    assert manifest['generator_role']==('frozen_initial_student' if role=='rewrite' else 'external_teacher')
    assert json.loads((out/'coverage.json').read_text())['complete']


def test_frozen_resource_hashes_and_public_runtime_closure():
    """Every frozen runtime resource must ship with its byte identity under kit/."""
    from kit.v4_resources import RESOURCE_HASHES,resource
    assert RESOURCE_HASHES
    for name,digest in RESOURCE_HASHES.items():
        path=resource(name)
        assert path.is_relative_to(ROOT/'kit') and hashlib.sha256(path.read_bytes()).hexdigest()==digest
    for name in ('v4_contract.py','v4_teacher.py','v4_campaign_ops.py','v4_datasets.py'):
        text=(ROOT/'kit'/name).read_text()
        assert not any(s in text for s in ("ROOT/'docs",'ROOT / "docs',"ROOT / 'docs","ROOT/'scripts",'ROOT / "scripts'))


def test_host_return_tools_are_stdlib_only():
    """Login-node reconcile, submission and unfiltered collection must import with site packages disabled."""
    code="from kit import p4_contain, collect, v4_phase0_submission; assert callable(p4_contain.reconcile) and callable(collect.gather)"
    done=subprocess.run([sys.executable,'-S','-c',code],cwd=ROOT,capture_output=True,text=True,timeout=15)
    assert done.returncode==0,done.stderr


def test_submitted_cpu_success_exercises_live_report_pause_and_reconciliation(tmp_path,monkeypatch):
    """The exported rehearsal must catch live report dependencies on post-job-only fields."""
    for key in PRIVATE_INPUTS:monkeypatch.delenv(key,raising=False)
    from kit.simulate_v4_phase0 import rehearse
    result=rehearse(tmp_path/'success','success')
    assert result['runner_exit']==0
    assert result['live_report']['status']=='technical pass'
    assert result['pause']['state']=='PAUSE'
    assert result['reader']['status']=='incomplete'  # CPU stand-ins never certify hardware.
    assert result['reader']['allocation_gpu_hours']>0
    assert 'real GPU containment' in ' '.join(result['reader']['reasons'])
    ledger=json.loads((Path(result['archive']).parent/'work/k8b4/containment/allocation-ledger.json').read_text())
    assert ledger['allocations']['123']['terminal_observation']['state']=='COMPLETED'


@pytest.mark.parametrize('mutation,cause',[
    ('none',None),('work','WORK identity'),('job','job identity'),('plan','plan binding'),
    ('width','running job/start/width'),('unknown','exposure is unknown'),('extra','exactly one new'),
    ('limits','budget settings'),('hard_stop','hard stop'),('reservation','22 GPU-hours'),
    ('block','wrong block'),('qualification','100/560 budget'),('ceiling','100/560 budget')])
def test_live_submission_keeps_identity_and_budget_guards(tmp_path,monkeypatch,mutation,cause):
    """Skipping terminal-only fields must retain live identity, integrity and full-allocation budget refusals."""
    import copy
    import time
    from kit import p4_contain as pc
    from kit.v4_phase0_submission import record_submission,live_submission_accounting,receipt_path
    from kit.v4_archive import DirectoryEvidence
    work=tmp_path/'work';base=work/'v4/report-phase0';(base/'allocation').mkdir(parents=True)
    inputs=work/'v4/report-inputs';inputs.mkdir();prepare=inputs/'prepare-receipt-phase0.json';prepare.write_text('{}')
    presend=base/'presend/containment-presend.json';presend.parent.mkdir();presend.write_text('{}')
    plan=base/'allocation/phase0.json';doc={'phase':'phase0','block':'qualification','block_limit':100,'ceiling':560,'gpus':8,
        'sbatch_time':'01:50:00','phase0_allocation_cap_gpu_hours':22,'prior_allocations':{},
        'phase0_prepare_sha256':hashlib.sha256(prepare.read_bytes()).hexdigest(),'phase0_presend_sha256':hashlib.sha256(presend.read_bytes()).hexdigest()}
    plan.write_text(json.dumps(doc));wrapper=plan.with_suffix('.sbatch');wrapper.write_text('# CPU fixture\n')
    record_submission(work,'123',plan,wrapper);monkeypatch.setenv('SLURM_JOB_ID','123')
    start=pc.wd.precise_text(pc.wd.from_epoch(time.time()-120))
    ledger=pc.record_allocation(work,{'job_id':'123','state':'RUNNING','start':start,'end':None,'width':8,'time_limit_seconds':6600},'qualification',100,560)
    entry=ledger['allocations']['123'];entry['phase0']=True
    if mutation in ('work','job'):
        path=receipt_path(work);receipt=json.loads(path.read_text());receipt[mutation if mutation=='work' else 'job_id']='different' if mutation=='work' else '124';path.write_text(json.dumps(receipt))
    if mutation=='plan':plan.write_text('{}')
    if mutation=='width':entry['width']=1
    if mutation=='unknown':ledger['accounting_unknown']=True
    if mutation=='extra':ledger['allocations']['124']=copy.deepcopy(entry)
    if mutation=='limits':ledger['block_limits']['qualification']=101
    if mutation=='hard_stop':(work/'k8b4/containment/v4-stop.json').write_text('{}')
    if mutation=='reservation':entry['planned_end']=pc.wd.precise_text(pc.wd.from_epoch(time.time()+3*3600))
    if mutation=='block':entry['segments'][0]['block']='scientific'
    if mutation in ('qualification','ceiling'):
        seconds=(101 if mutation=='qualification' else 561)*3600/8
        end=pc.wd.precise_text(pc.wd.from_epoch(pc.epoch(start)-1));begin=pc.wd.precise_text(pc.wd.from_epoch(pc.epoch(end)-seconds))
        block='qualification' if mutation=='qualification' else 'scientific'
        prior={'start':begin,'end':end,'width':8,'segments':[{'start':begin,'end':end,'block':block}]}
        ledger['allocations']['122']=prior;doc['prior_allocations']={'122':prior};plan.write_text(json.dumps(doc))
        path=receipt_path(work);receipt=json.loads(path.read_text());receipt['plan_sha256']=hashlib.sha256(plan.read_bytes()).hexdigest();path.write_text(json.dumps(receipt))
    pc.wd.write_durably(work/pc.DIRECTORY/'allocation-ledger.json',ledger)
    evidence=DirectoryEvidence(work);report={'tables':{}}
    if cause:
        with pytest.raises(ValueError,match=cause):live_submission_accounting(evidence,report)
    else:
        assert live_submission_accounting(evidence,report)=='123'
        assert report['allocation_gpu_hours']>0 and not report['tables']['allocation_accounting']['terminal']
        assert 'submission_sha256' not in entry and 'terminal_observation' not in entry
