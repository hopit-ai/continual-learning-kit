"""Offline S8 general binding; synthetic raw files, real pinned metric/prompt code."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import types
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

import pytest

from kit import trace_general_build as b, trace_general_metrics as m, trace_general_score as s
from kit import trace_operator as o, trace_common as c
from kit.trace_score import metric_adapter


def ref(path):
    return dict(path=str(path), sha256=c.sha(path))


@pytest.fixture(scope='module')
def corpus(tmp_path_factory):
    root=tmp_path_factory.mktemp('general-raw')
    manifest=dict(schema='trace-general-inputs.v1',datasets={})
    for panel in m.PANELS:
        files={}
        for key in b.expected_files(panel):
            path=root/panel/key;path.parent.mkdir(parents=True,exist_ok=True)
            if panel=='MMLU':
                import csv
                with path.open('w',newline='') as stream:
                    writer=csv.writer(stream)
                    for i in range(5 if key.startswith('dev/') else 1):
                        writer.writerow([f'Question {i}?','one','two','three','four','ABCD'[i%4]])
            elif panel=='BBH': c.atomic(path,dict(examples=[dict(input='Question?',target='(B)')]))
            else:
                raw={'GSM8K':dict(question='How many?',answer='work #### 1,234'),
                     'TyDiQA':dict(passage_text='Passage.',question_text='Question?',answers=dict(text=['a gold','other'])),
                     'PIQA':dict(goal='Goal',sol1='first',sol2='second',label=1),
                     'BoolQ':dict(passage='Passage',question='Question',label='true')}[panel]
                path.write_text(json.dumps(raw)+'\n')
            files[key]=ref(path)
        manifest['datasets'][panel]=dict(revision='synthetic-offline-test',files=files)
    return manifest,b.build(manifest)


def test_builder_inventory_provenance_and_fixed_shots(corpus):
    manifest,panels=corpus
    assert b.validate(panels)
    assert {p:len(v['rows']) for p,v in panels.items()} == dict(MMLU=57,BBH=27,GSM8K=1,TyDiQA=11,PIQA=1,BoolQ=1)
    for panel,spec in panels.items():
        assert set(spec['source']['files'])==set(b.expected_files(panel))
        assert spec['manifest_sha256']==c.digest(manifest['datasets'][panel])
    row=panels['MMLU']['rows'][0]
    assert row['shot_ids']==[0,1,2,3,4]
    assert row['label_prompts'][0]['prompt'].count('Answer: A\n')==2
    assert '\n\nQuestion 1?' in row['label_prompts'][0]['prompt']
    assert panels['GSM8K']['rows'][0]['answer']=='1234'
    assert all(type(r['answer']) is list for r in panels['TyDiQA']['rows'])
    assert panels['BoolQ']['rows'][0]['answer']==1


@pytest.mark.parametrize('mutation', ['hash','inventory','revision'])
def test_builder_rejects_unbound_inputs(corpus,mutation):
    manifest=copy.deepcopy(corpus[0])
    source=manifest['datasets']['PIQA']
    if mutation=='hash': source['files']['validation.jsonl']['sha256']='0'*64
    if mutation=='inventory': source['files']['extra.jsonl']=source['files']['validation.jsonl']
    if mutation=='revision': source['revision']=''
    with pytest.raises(ValueError): b.build(manifest)


@pytest.mark.parametrize('mutation', ['cap','temperature','pin','subset','labels','gold','mode','shots','duplicate'])
def test_prepare_shape_rejects_changed_contract(corpus,mutation):
    panels=copy.deepcopy(corpus[1])
    if mutation=='cap': panels['TyDiQA']['response_cap']=512
    if mutation=='temperature': panels['GSM8K']['temperature']=.1
    if mutation=='pin': panels['BBH']['commit']='wrong'
    if mutation=='subset': panels['MMLU']['subsets'].pop('college_biology')
    if mutation=='labels': panels['PIQA']['rows'][0]['labels']=[1,0]
    if mutation=='gold': panels['TyDiQA']['rows'][0]['answer']='string gold'
    if mutation=='mode': panels['MMLU']['rows'][0]['label_prompts'][0]['render_mode']='chat'
    if mutation=='shots': panels['MMLU']['rows'][0]['shot_ids']=[1,2,3,4,5]
    if mutation=='duplicate': panels['MMLU']['rows'][1]['id']=panels['MMLU']['rows'][0]['id']
    with pytest.raises(ValueError): b.validate(panels)


@pytest.mark.parametrize('panel',m.PANELS)
def test_real_kit_scorers_pass_all_fixtures(panel):
    metric=dict(**ref(c.KIT/'trace_general_metrics.py'),function='score',
                fixtures=ref(o.RESOURCE/'fixtures'/f'{panel}.json'))
    fn=metric_adapter(metric)
    assert len(c.bound_json(metric['fixtures']))>=20
    for case in c.bound_json(metric['fixtures']):
        assert fn(case['row'],case['prediction'])==pytest.approx(case['score'],abs=1e-6)


def test_vendor_inventory_hashes_and_approval():
    p=m.verify_vendor()
    assert p['tag']=='0.1.5'
    assert p['owner_approval']=='13 Nov 2026: "Approve the metric download and v7i tag"'
    actual={x.relative_to(m.VENDOR).as_posix() for x in m.VENDOR.rglob('*')
            if x.is_file() and x.name!='PROVENANCE.json' and '__pycache__' not in x.parts and x.suffix!='.pyc'}
    assert actual==set(p['sha256'])
    assert 'Apache License' in (m.VENDOR/'LICENSE').read_text()


@pytest.mark.parametrize('failure',[False,True])
def test_bridge_restores_objects_and_owned_modules(monkeypatch,failure):
    monkeypatch.setattr(sys,'dont_write_bytecode',False)
    names=['datasets','evaluate','opencompass','opencompass.other']
    sentinels={name:types.ModuleType(name) for name in names}
    for name,obj in sentinels.items(): monkeypatch.setitem(sys.modules,name,obj)
    paths,modules=sys.path,sys.modules
    before=list(sys.path)
    if failure:
        original=m._load
        def bad(name,path):
            if name.endswith('tydiqa'): raise RuntimeError('injected module failure')
            return original(name,path)
        monkeypatch.setattr(m,'_load',bad)
        with pytest.raises(RuntimeError,match='injected'): m.score(dict(panel='PIQA',answer=0),0)
    else:
        assert m.score(dict(panel='TyDiQA',answer=['the cat']), 'CAT!')==100
    assert sys.path is paths and sys.path==before and sys.modules is modules
    assert all(sys.modules[name] is obj for name,obj in sentinels.items())
    assert not any(name.startswith('opencompass.') and name not in names for name in sys.modules)
    assert 'trace_oc_accuracy' not in sys.modules
    assert sys.dont_write_bytecode is False


def test_bridge_concurrency_has_no_cross_panel_state():
    cases=[dict(panel='TyDiQA',answer=['blue whale']),dict(panel='GSM8K',answer='1234')]*5
    predictions=['The blue whale.','1,234 kg']*5
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert list(pool.map(m.score,cases,predictions))==[100]*10


def test_general_authorities_cannot_be_overridden_before_asset_work(tmp_path):
    for panel in m.PANELS:
        with pytest.raises(ValueError,match='kit-supplied; remove: '+panel):
            o.bind(dict(operator=dict(metric_authorities={panel:{}})),tmp_path/'no-write')
    assert not (tmp_path/'no-write').exists()


@pytest.mark.parametrize('study',['s8','gold'])
def test_bind_uses_kit_general_metrics_and_keeps_task_authorities(tmp_path,monkeypatch,study):
    # Bind actual fixtures while isolating unrelated asset/admission/native setup.
    source=tmp_path/'task.py'
    source.write_text('def score(p,g): return {"accuracy": float(p==g)}\n')
    authority=dict(**ref(source),function='score',key='accuracy',scale_to_100=100)
    cfg=dict(studies=[study],operator=dict(metric_authorities=dict.fromkeys(c.TASKS,authority),
        admission_evidence={},native_probes={study:{name:None for name in o.NATIVE}}))
    command=tmp_path/'command.json';c.atomic(command,dict(argv=['offline-probe'],files=[ref(source)]))
    cfg['operator']['native_probes'][study]=dict.fromkeys(o.NATIVE,ref(command))
    monkeypatch.setattr(o,'stage_assets',lambda x:copy.deepcopy(x))
    monkeypatch.setattr(o,'admission',lambda *a,**kw:None)
    if study=='gold':
        cfg['operator']['metric_authorities'].update(dict.fromkeys(m.PANELS,{}))
        with pytest.warns(UserWarning,match='Ignoring unused.*BBH, BoolQ, GSM8K, MMLU, PIQA, TyDiQA'):
            bound=o.bind(cfg,tmp_path/'binding')
        assert not (set(m.PANELS) & set(bound['operator']['metric_authorities']))
        assert not (set(m.PANELS) & set(bound['metrics']))
        assert all(bound['metrics'][task]['authority']==authority for task in c.SUBSET)
        return
    bound=o.bind(cfg,tmp_path/'binding')
    for panel in m.PANELS:
        metric=bound['metrics'][panel]
        assert metric['path']==str(c.KIT/'trace_general_metrics.py') and 'authority' not in metric
        assert any(d['path'].endswith('opencompass-0.1.5/PROVENANCE.json') for d in metric['dependencies'])
        assert any(d['path'].endswith('opencompass/tasks/openicl_infer.py') for d in metric['dependencies'])
        metric_adapter(metric)
    assert all(bound['metrics'][task]['authority']==authority for task in c.TASKS)


def test_subset_bind_ignores_missing_general_authorities_before_staging(tmp_path,monkeypatch):
    source=tmp_path/'task.py'
    source.write_text('def score(p,g): return {"accuracy": float(p==g)}\n')
    authority=dict(**ref(source),function='score',key='accuracy',scale_to_100=100)
    missing=dict(path=str(tmp_path/'missing-general-authority.py'),sha256='0'*64)
    command=tmp_path/'command.json'
    c.atomic(command,dict(argv=['offline-probe'],files=[ref(source)]))
    cfg=dict(technical_synthetic=False,studies=['gold'],operator=dict(
        metric_authorities={**dict.fromkeys(c.SUBSET,authority),**dict.fromkeys(m.PANELS,missing)},
        admission_evidence={},native_probes=dict(gold=dict.fromkeys(o.NATIVE,ref(command)))))
    for role in ('base','teacher','judge','original_tokenizer'):
        directory=tmp_path/role
        directory.mkdir()
        (directory/'config.json').write_text('{}')
        cfg[role]=dict(model=str(directory),revision='synthetic-offline-test')
    original=copy.deepcopy(cfg)
    monkeypatch.setattr(o,'admission',lambda *a,**kw:None)
    with pytest.warns(UserWarning,match='Ignoring unused.*BBH, BoolQ, GSM8K, MMLU, PIQA, TyDiQA'):
        bound=o.bind(cfg,tmp_path/'binding')
    assert cfg==original
    assert not (set(m.PANELS) & set(bound['operator']['metric_authorities']))
    assert set(bound['metrics'])=={*c.SUBSET,'instruction','maths'}
    assert bound['base']['files']=={'config.json':c.sha(tmp_path/'base/config.json')}
    cfg['studies']=['s8']
    with pytest.raises(ValueError,match='kit-supplied; remove: BBH, BoolQ, GSM8K, MMLU, PIQA, TyDiQA'):
        o.bind(cfg,tmp_path/'refused')
    assert not (tmp_path/'refused').exists()


class Tokenizer:
    pad_token_id=0
    truncation_side='right'
    def __call__(self,texts,padding=False,truncation=False,max_length=None):
        arrays=[[1]+[2 if x!='#' else 0 for x in text] for text in texts]
        return dict(input_ids=[x[:max_length] if truncation else x for x in arrays])
    def apply_chat_template(self,messages,**kwargs):
        assert kwargs['enable_thinking'] is False
        return '|'.join(m['role']+':'+m['content'] for m in messages)+('GEN' if kwargs['add_generation_prompt'] else '')
    def encode(self,text,add_special_tokens=False): return list(range(len(text)))


def test_likelihood_whole_sequence_pad_denominator_ties_and_repeats():
    import torch
    tok=Tokenizer()
    calls=[]
    def logits(ids):
        calls.append(ids.clone())
        return torch.zeros((*ids.shape,4),dtype=torch.float64)
    row=dict(id='test',labels=['A','B'],label_prompts=[dict(render_mode='raw',prompt='a#b')]*2)
    first=s.likelihood(None,[row],None,runtime=(tok,logits))
    second=s.likelihood(None,[row],None,runtime=(tok,logits))
    # Two non-pad predicted targets; three non-pad inputs, including BOS.
    expected=2*torch.log(torch.tensor(4.)).item()/3
    assert first[0]['losses']==pytest.approx([expected]*2,abs=1e-6)
    assert first[0]['selected']=='A'
    assert first==second and max(abs(a-b) for a,b in zip(first[0]['losses'],second[0]['losses']))<=1e-6
    assert all(x.shape==(1,4) and x.device.type=='cpu' for x in calls)
    for bad in ([float('nan'),1], [float('inf'),1], []):
        with pytest.raises(ValueError): s.select_label(['A','B'],bad)


def test_raw_chat_generation_and_cap_audit_keep_legacy_encoding():
    from kit.trace_score import encode_tokens
    tok=Tokenizer()
    raw=dict(render_mode='raw',prompt='RAW')
    chat=dict(render_mode='chat',messages=[dict(role='user',content='USER'),dict(role='assistant',content='LABEL')])
    assert s.prompt_text(tok,raw,ppl=True)=='RAW'
    assert s.prompt_text(tok,chat,ppl=True)=='user:USER|assistant:LABEL'
    assert s.prompt_text(tok,chat,ppl=False).endswith('GEN')
    rows=[dict(prompt='OLD'),raw]
    actual=encode_tokens(tok,rows,3)
    legacy=tok.apply_chat_template([dict(role='user',content='OLD')],tokenize=False,
                                  add_generation_prompt=True,enable_thinking=False)
    assert actual['uncut_lengths']==[len(legacy),3]
    assert actual['token_ids']==[list(range(len(legacy)))[-3:],[0,1,2]]


def test_likelihood_records_and_warns_on_prompt_cap():
    import torch
    row=dict(id='long',labels=[0],label_prompts=[dict(render_mode='raw',prompt='x'*2100)])
    with pytest.warns(UserWarning,match='truncated'):
        record=s.likelihood(None,[row],None,runtime=(Tokenizer(),lambda ids:torch.zeros((*ids.shape,4))))[0]
    assert record['prompt_cap_hits']==1 and record['truncated_tokens']==53
    assert record['uncut_lengths']==[2101]


def test_aggregation_naive_pooled_and_config_weighted(corpus):
    rows=copy.deepcopy(corpus[1]['MMLU']['rows'])
    values=[100.]+[0.]*(len(rows)-1)
    rows.append(dict(rows[0],id='extra'));values.append(100.)
    result=s.aggregate('MMLU',rows,values)
    assert result['n']==57 and result['mean']==pytest.approx(100/57)
    assert result['pooled']['mean']==pytest.approx(200/58)
    assert result['mmlu-weighted']==pytest.approx(100*144/14042)
    assert result['subsets']['college_biology']['n']==2
    assert result['sum']/result['n']==result['mean']


@pytest.mark.parametrize('names',[[],['one'],['one','one'],['one','two','three'],'ab'])
def test_raters_require_two_distinct_ordered_names(names):
    from kit.trace_calibration import packets
    from kit.trace_judge import calibration_summary
    with pytest.raises(ValueError,match='exactly two'): packets([],names)
    with pytest.raises(ValueError,match='exactly two'): calibration_summary([],[],{},rater_names=names)


def test_declared_raters_order_and_identical_arithmetic():
    from kit.trace_judge import calibration_summary,DEFAULT_RATERS
    from kit.trace_calibration import packets
    pairs=[dict(id=str(i),panel=('Self-Instruct','LIMA','CoNa')[i//30],prompt='q',a='a',b='b',
                checkpoint_independent=True) for i in range(90)]
    labels=[dict(id=str(i),labels=[i%3-1,i%3-1]) for i in range(90)]
    raters={name:{str(i):i%3-1 for i in range(90)} for name in DEFAULT_RATERS}
    baseline=calibration_summary(pairs,labels,raters)
    names=['Second declared','First declared']
    renamed=calibration_summary(pairs,labels,dict(zip(names,raters.values())),rater_names=names)
    assert list(packets(pairs)['raters'])==list(DEFAULT_RATERS)==['gpt-6-astra','Claude Opus']
    assert list(packets(pairs,names)['raters'])==names
    for panel in baseline:
        assert [baseline[panel][k] for k in ('H','J','O','kappa')]==[renamed[panel][k] for k in ('H','J','O','kappa')]
        assert list(renamed[panel]['rater_prevalence'])==names
    with pytest.raises(ValueError,match='rater keys differ'): calibration_summary(pairs,labels,raters,rater_names=names)


def test_prompts_equal_complete_upstream_templates_and_retrievers(corpus,tmp_path):
    """Execute only copied vendored code, in a separate isolated interpreter."""
    import shutil
    stage=tmp_path/'upstream';shutil.copytree(m.VENDOR,stage,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    input_path=tmp_path/'manifest.json';c.atomic(input_path,corpus[0])
    script=tmp_path/'prompt_check.py';script.write_text(UPSTREAM_PROMPT_WORKER)
    out=tmp_path/'prompts.json'
    subprocess.run([sys.executable,'-I',str(script),str(stage),str(input_path),str(out)],check=True)
    expected=c.load(out)
    count=0
    for panel,spec in corpus[1].items():
        for row,item in zip(spec['rows'],expected[panel]):
            if 'label_prompts' in row: actual=row['label_prompts']
            elif row['render_mode']=='raw': actual=dict(render_mode='raw',prompt=row['prompt'])
            else: actual=dict(render_mode='chat',messages=row['messages'])
            assert actual==item
            count+=1
    assert count==98


UPSTREAM_PROMPT_WORKER=r'''
import importlib.util,json,sys,types,logging
from pathlib import Path
stage,inputs,out=map(Path,sys.argv[1:])
oc=stage/'opencompass'
def pkg(name,path=None):
 m=types.ModuleType(name);m.__path__=[str(path)] if path else [];sys.modules[name]=m;return m
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
class Registry:
 def register_module(self,*a,**k): return lambda obj:obj
for name,path in [('opencompass',oc),('opencompass.openicl',oc/'openicl'),('opencompass.utils',oc/'utils')]:pkg(name,path)
reg=pkg('opencompass.registry')
for name in ('ICL_PROMPT_TEMPLATES','ICL_RETRIEVERS'):setattr(reg,name,Registry())
pkg('mmengine');pkg('mmengine.config').ConfigDict=dict;pkg('mmengine.dist').is_main_process=lambda:False
pkg('opencompass.utils.types')._check_type_list=lambda v,allowed:v
for name in ('opencompass.utils.logging','opencompass.openicl.utils.logging'):pkg(name).get_logger=lambda *a:logging.getLogger('parity')
load('opencompass.utils.prompt',oc/'utils/prompt.py')
Template=load('opencompass.openicl.icl_prompt_template',oc/'openicl/icl_prompt_template.py').PromptTemplate
r=pkg('opencompass.openicl.icl_retriever',oc/'openicl/icl_retriever')
r.BaseRetriever=load('opencompass.openicl.icl_retriever.icl_base_retriever',oc/'openicl/icl_retriever/icl_base_retriever.py').BaseRetriever
r.FixKRetriever=load('opencompass.openicl.icl_retriever.icl_fix_k_retriever',oc/'openicl/icl_retriever/icl_fix_k_retriever.py').FixKRetriever
r.ZeroRetriever=load('opencompass.openicl.icl_retriever.icl_zero_retriever',oc/'openicl/icl_retriever/icl_zero_retriever.py').ZeroRetriever
for package,names in [('opencompass.openicl.icl_evaluator',['AccEvaluator']),('opencompass.openicl.icl_inferencer',['PPLInferencer','GenInferencer']),('opencompass.datasets',['MMLUDataset','HFDataset','BoolQDataset','BBHDataset','BBHEvaluator','bbh_mcq_postprocess','TydiQADataset','TydiQAEvaluator','gsm8k_postprocess','gsm8k_dataset_postprocess'])]:
 m=pkg(package)
 for name in names:setattr(m,name,type(name,(),{}))
configs={'MMLU':('mmlu/mmlu_ppl_ac766d.py','mmlu_datasets'),'BBH':('bbh/bbh_gen_5b92b0.py','bbh_datasets'),'GSM8K':('gsm8k/gsm8k_gen_1d7fe4.py','gsm8k_datasets'),'TyDiQA':('tydiqa/tydiqa_gen_978d2a.py','tydiqa_datasets'),'PIQA':('piqa/piqa_ppl_1cf9f0.py','piqa_datasets'),'BoolQ':('SuperGLUE_BoolQ/SuperGLUE_BoolQ_ppl_314b96.py','BoolQ_datasets')}
manifest=json.loads(inputs.read_text())
def portable(p):
 if isinstance(p,str):return dict(render_mode='raw',prompt=p)
 return dict(render_mode='chat',messages=[dict(role={'HUMAN':'user','BOT':'assistant'}[x['role']],content=x['prompt']) for x in p if isinstance(x,dict) and 'role' in x])
result={}
for panel,(rel,name) in configs.items():
 cfgs=getattr(load('config_'+panel,stage/'configs/datasets'/rel),name);result[panel]=[]
 files=manifest['datasets'][panel]['files']
 def read(key):
  path=Path(files[key]['path'])
  if path.suffix=='.csv':
   import csv
   return [dict(zip(('input','A','B','C','D','target'),r)) for r in csv.reader(path.open())]
  if path.suffix=='.json':return json.loads(path.read_text())['examples']
  return [json.loads(line) for line in path.read_text().splitlines()]
 for cfg in cfgs:
  subset=cfg.get('name');dev=[]
  if panel=='MMLU':dev=read(f'dev/{subset}_dev.csv');raw=read(f'test/{subset}_test.csv')
  elif panel=='BBH':raw=read(f'{subset}.json')
  elif panel=='TyDiQA':raw=read(f'{subset}/validation.jsonl')
  else:raw=read(next(iter(files)))
  dataset=types.SimpleNamespace(train=dev,test=raw,reader=types.SimpleNamespace(output_column=cfg['reader_cfg']['output_column']))
  retriever=r.FixKRetriever(dataset) if dev else r.ZeroRetriever(dataset)
  icfg=cfg['infer_cfg'];tp=Template(**{k:v for k,v in icfg['prompt_template'].items() if k!='type'})
  itp=Template(**{k:v for k,v in icfg['ice_template'].items() if k!='type'}) if dev else None
  ice=retriever.generate_ice(retriever.retrieve([0,1,2,3,4] if dev else None)[0],itp)
  if panel in ('MMLU','PIQA','BoolQ'):value=[portable(retriever.generate_label_prompt(0,ice,label,itp,tp)) for label in retriever.get_labels(itp,tp)]
  else:value=portable(retriever.generate_prompt_for_generate_task(0,ice,ice_template=itp,prompt_template=tp))
  result[panel].append(value)
out.write_text(json.dumps(result))
'''


def test_generation_general_raw_rows_record_caps_with_pinned_decoding(monkeypatch):
    from kit import trace_score as score, inference_lifecycle, m2_gpu
    observed={}
    class Engine:
        def generate(self,inputs,params,**kwargs):
            observed.update(inputs=inputs,params=params)
            return [types.SimpleNamespace(outputs=[types.SimpleNamespace(text='cat',
                finish_reason='length',token_ids=[1]*50)])]
    monkeypatch.setitem(sys.modules,'transformers',types.SimpleNamespace(AutoTokenizer=types.SimpleNamespace(
        from_pretrained=lambda *a,**kw:Tokenizer())))
    monkeypatch.setitem(sys.modules,'vllm',types.SimpleNamespace(SamplingParams=lambda **kw:kw))
    monkeypatch.setattr(inference_lifecycle,'create_llm',lambda **kw:Engine())
    monkeypatch.setattr(inference_lifecycle,'marker',lambda:0)
    monkeypatch.setattr(inference_lifecycle,'shutdown_since',lambda *a:None)
    monkeypatch.setattr(m2_gpu,'assert_current',lambda:None)
    monkeypatch.delenv('TRACE_TOKEN_CACHE',raising=False)
    row=dict(id='tydi',render_mode='raw',prompt='x'*2100)
    record=score.generation('fake','fake',[row],2048,50,0,101)[0]
    assert observed['inputs'][0]['prompt_token_ids']==list(range(2100))[-2048:]
    assert observed['params'][0]['temperature']==0 and observed['params'][0]['max_tokens']==50
    assert 'stop' not in observed['params'][0]
    assert record['prompt_cap_hit'] and record['prompt_truncated_tokens']==52 and record['response_cap_hit']


@pytest.mark.parametrize('drop_shots',[False,True])
def test_score_state_dispatches_ppl_and_generation_and_persists_subset_means(corpus,tmp_path,monkeypatch,drop_shots):
    from kit import trace_score as score
    protocol=dict(panels={'trace_general':corpus[1],
        'general-v1':[dict(id=p,panel=p,answer='A') for p in ('instruction','maths','knowledge')],
        'open':[dict(id='open',prompt='q')]},datasets={task:dict(test=[dict(id=task,answer='ok')]) for task in c.TASKS})
    c.atomic(tmp_path/'prepared.json',protocol)
    calls=[]
    def generation(model,tokenizer,rows,prompt_cap,response_cap,temperature,seed):
        calls.append((rows,prompt_cap,response_cap,temperature))
        return [dict(id=r['id'],text=(r['answer'][0] if type(r.get('answer')) is list else r.get('answer','ok')),
                     prompt_cap_hit=False,prompt_truncated_tokens=0,response_cap_hit=False) for r in rows]
    def likelihood(model,rows,tokenizer,runtime=None):
        assert runtime == ('shared-tokenizer', 'shared-logits')
        outputs=[]
        for index,row in enumerate(rows):
            used=dict.fromkeys(row['labels'],list(row.get('shot_ids',[])))
            if drop_shots and row['panel']=='MMLU':
                if index==0: used[row['labels'][-1]]=list(range(4))
                if index==1: used=dict.fromkeys(row['labels'],list(range(3)))
                if index==2: used=dict.fromkeys(row['labels'],[])
            outputs.append(dict(id=row['id'],selected=row['answer'],prompt_cap_hits=0,
                                truncated_tokens=0,shot_ids_by_label=used))
        return outputs
    monkeypatch.setattr(score,'generation',generation)
    monkeypatch.setattr(s,'likelihood',likelihood)
    runtime_entries=[]
    @contextmanager
    def runtime(*args):
        runtime_entries.append(args)
        yield 'shared-tokenizer', 'shared-logits'
    monkeypatch.setattr(s,'likelihood_runtime',runtime)
    monkeypatch.setattr(m.score,'score_many',m.score_many,raising=False)
    monkeypatch.setattr(score,'metric_adapter',lambda metric:m.score if metric in m.PANELS else lambda *a:50)
    monkeypatch.setitem(sys.modules,'kit.m3_score',types.SimpleNamespace(
        knowledge_reference=lambda *a:[dict(selected=0)]))
    cfg=dict(out=str(tmp_path),base=dict(model='fake'),metrics={p:p for p in (*c.TASKS,*m.PANELS,'instruction','maths')})
    score.score_state(cfg,'s8',101,dict(id='base'),'fake')
    assert len(runtime_entries)==1
    root=tmp_path/'scores/s8-101/base'
    result=c.load(root/'result.json')
    for panel in m.PANELS:
        assert result[panel]['mean']==100
        assert result[panel]['n']==len(b.subsets(panel))
        assert len(c.load(root/(panel+'.json'))['outputs'])==len(corpus[1][panel]['rows'])
    for panel in ('MMLU','PIQA','BoolQ'):
        assert c.load(root/(panel+'.json'))['audit']['shot_dropped_items']==(
            3 if drop_shots and panel=='MMLU' else 0)
    general_calls={rows[0]['panel']:(pc,rc,temp) for rows,pc,rc,temp in calls if rows[0].get('panel') in m.PANELS}
    assert general_calls==dict(GSM8K=(2048,512,0),BBH=(2048,512,0),TyDiQA=(2048,50,0))


@pytest.mark.parametrize('label', [True, False, 0, 1, None, 'True', 'false ', []])
def test_boolq_requires_exact_string_labels(label):
    with pytest.raises(ValueError, match='BoolQ label must be exactly the string'):
        b.build_row('BoolQ','BoolQ',0,dict(passage='p',question='q',label=label))


@pytest.mark.parametrize('panel', m.PANELS)
def test_score_many_identical_per_item_on_all_fixtures(panel):
    cases=c.load(o.RESOURCE/'fixtures'/f'{panel}.json')
    rows=[case['row'] for case in cases]
    predictions=[case['prediction'] for case in cases]
    assert m.score_many(rows,predictions)==[m.score(r,p) for r,p in zip(rows,predictions)]
    with pytest.raises(ValueError,match='predictions missing'):
        m.score_many(rows,predictions[:-1])


def test_bridge_verifies_once_loads_accuracy_once_and_restores_bytecode(monkeypatch):
    original_verify,original_load=m.verify_vendor,m._load
    calls=[]
    def verify():
        calls.append('verify')
        return original_verify()
    def load(name,path):
        assert sys.dont_write_bytecode is True
        calls.append(name)
        return original_load(name,path)
    m._verified_vendor.cache_clear()
    monkeypatch.setattr(m,'verify_vendor',verify)
    monkeypatch.setattr(m,'_load',load)
    monkeypatch.setattr(sys,'dont_write_bytecode',False)
    before=set(m.VENDOR.rglob('*.pyc'))
    for _ in range(2):
        assert m.score_many([dict(panel='PIQA',answer=0)]*3,[0]*3)==[100]*3
        assert sys.dont_write_bytecode is False
    assert calls.count('verify')==1 and calls.count('trace_oc_accuracy')==2
    assert set(m.VENDOR.rglob('*.pyc'))==before
    m._verified_vendor.cache_clear()


class _ExportTokenizer:
    """Test-local copy of the manager verifier's CPU tokenizer fixture.

    The verifier script is manager-only and is not exported, so the public
    suite must not import it."""
    pad_token_id=0
    truncation_side='right'
    def encode(self, text):
        extra = 1000 if text.endswith('Answer: B') else (3000 if text.endswith('Answer: D') else 0)
        return [1]+[ord(ch)%13+2 for ch in text]+([0,3] if text=='pad-target' else [])+[2]*extra
    def __call__(self, texts, padding=False, truncation=False, max_length=None):
        ids=[self.encode(text) for text in texts]
        return dict(input_ids=[x[:max_length] if truncation else x for x in ids])
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs['enable_thinking'] is False
        return ''.join(m['role']+':'+m['content']+'\n' for m in messages)


def _dropping_row():
    raw=dict(input='Query?', A='first', B='second', C='third', D='fourth', target='C')
    dev=[dict(raw,input=f'Shot {i} {{input}} '+('x'*400),target='ABCD'[i%4]) for i in range(5)]
    return b.build_row('MMLU','college_biology',0,raw,dev)


def test_export_tokenizer_fixture_matches_manager_verifier():
    verify=pytest.importorskip('scripts.trace_general_upstream_verify',
                               reason='upstream verifier is manager-only tooling')
    ours,theirs=_ExportTokenizer(),verify.DroppingTokenizer()
    for text in ('', 'abc', 'pad-target', 'x Answer: B', 'y Answer: D', 'Answer: C'):
        assert ours.encode(text)==theirs.encode(text)
    assert _dropping_row()==b.build_row('MMLU',verify.dropping_sample()['subset'],0,
        verify.dropping_sample()['raw'],verify.dropping_sample()['dev'])


def test_mmlu_census_counts_dropping_and_residual():
    census=pytest.importorskip('scripts.trace_general_mmlu_census',
                               reason='MMLU census is manager-only tooling').census
    result=census([_dropping_row()],_ExportTokenizer())
    assert (result['items'],result['need_shot_dropping'],result['remain_over_cap'])==(1,1,1)


def test_mmlu_drops_last_shots_carries_across_labels_and_records_residual():
    import torch
    row=_dropping_row()
    tok=_ExportTokenizer()
    with pytest.warns(UserWarning,match='truncated'):
        record=s.likelihood(None,[row],None,runtime=(tok,lambda ids:torch.zeros((*ids.shape,16))))[0]
    ids=record['shot_ids_by_label']
    assert 0 < len(ids['B']) < len(ids['A']) < 5
    assert ids['B']==ids['C'] and ids['D']==[]
    assert all(used==list(range(len(used))) for used in ids.values())
    assert record['truncated_tokens_by_label']['D']>0
    assert record['prompt_cap_hits']==1


def test_packet_cli_uses_slug_filenames_and_retains_rater_names(tmp_path,monkeypatch):
    from kit import trace_calibration
    pairs=[dict(id=str(i),panel=('Self-Instruct','LIMA','CoNa')[i//30],prompt='q',a='a',b='b',
                checkpoint_independent=True) for i in range(90)]
    path=tmp_path/'pairs.json';c.atomic(path,pairs)
    monkeypatch.setattr(sys,'argv',['packets','--pairs',str(path),'--out',str(tmp_path/'packets')])
    trace_calibration.main()
    assert {p.name for p in (tmp_path/'packets').iterdir()}=={
        'mapping.json','gpt-6-astra-high.json','claude-opus-high.json'}
    assert c.load(tmp_path/'packets/claude-opus-high.json')['model']=='Claude Opus'
