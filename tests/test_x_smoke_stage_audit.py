"""Owner's attempt-3 condition: every exact smoke stage through native D inputs.

CPU/offline only. The pinned RLHFDataset and collation execute unmodified. Only
Ray/verl package initializers (which construct the GPU stack) are bypassed; the
absent OmegaConf import is a mapping/type carrier, not a replacement validator.
"""
import ast,copy,hashlib,importlib.util,json,logging,os,sys
from pathlib import Path
from types import ModuleType,SimpleNamespace as NS
import numpy as np
import pytest,torch
from kit.s_common import ROOT,atomic,read
from kit import x_integration_smoke as smoke,x_smoke_inputs as inputs,sdft_objective as objective,v4_contract
from kit.p3b_loss import mixed_teacher_inputs
from kit.p_smoke import frozen_inputs
from test_v4_resolved_recipe import primary_yaml,overrides

REFERENCE=ROOT/'references/SDPO'
STAGES=('preparation','measure','control','p2','prefix','resume')


class Config(dict):
    def __getattr__(self,key):
        try:return self[key]
        except KeyError:raise AttributeError(key) from None


def config(value):
    if isinstance(value,dict):return Config({k:config(v) for k,v in value.items()})
    if isinstance(value,list):return [config(v) for v in value]
    return value


class Tokenizer:
    """Tiny deterministic tokenizer; no cached model or network is used."""
    pad_token_id=0;eos_token_id=2;eos_token='<eos>'
    def encode(self,text,**kwargs):
        return [3+ord(c)%251 for c in text.removesuffix(self.eos_token)[::8]]+([2] if text.endswith(self.eos_token) else [])
    def __call__(self,text,**kwargs):return {'input_ids':self.encode(text,**kwargs)}
    def apply_chat_template(self,messages,tokenize=True,add_generation_prompt=True,**kwargs):
        text='\n'.join(m['role']+': '+m['content'] for m in messages)+('\nassistant:' if add_generation_prompt else '')
        return self.encode(text) if tokenize else text
    def decode(self,ids,**kwargs):return 'tiny decoded teacher context'


@pytest.fixture
def pinned_loader(monkeypatch,tmp_path):
    import datasets
    monkeypatch.setattr(datasets.config,'HF_DATASETS_CACHE',tmp_path/'hf-cache')
    # Load the actual reader and its actual local-file helper without importing
    # verl.__init__ (Ray/tensordict/GPU libraries are absent on this CPU host).
    for name,relative in (('verl','verl'),('verl.utils','verl/utils'),('verl.utils.dataset','verl/utils/dataset')):
        package=ModuleType(name);package.__path__=[str(REFERENCE/relative)];monkeypatch.setitem(sys.modules,name,package)
    omega=ModuleType('omegaconf');omega.DictConfig=Config;omega.ListConfig=list
    omega.OmegaConf=NS(to_container=lambda value,**kwargs:value)
    monkeypatch.setitem(sys.modules,'omegaconf',omega)
    for name in ('verl.utils.fs','verl.utils.hdfs_io','verl.utils.import_utils','verl.utils.dataset.rl_dataset'):
        monkeypatch.delitem(sys.modules,name,raising=False)
    path=REFERENCE/'verl/utils/dataset/rl_dataset.py'
    spec=importlib.util.spec_from_file_location('verl.utils.dataset.rl_dataset',path)
    module=importlib.util.module_from_spec(spec);monkeypatch.setitem(sys.modules,spec.name,module);spec.loader.exec_module(module)
    yield module
    # Remove lazily loaded helpers too; no fake package may escape this fixture.
    for name in ('verl.utils.fs','verl.utils.hdfs_io','verl.utils.import_utils'):
        sys.modules.pop(name,None)


def resolved(cmd):
    base=primary_yaml(REFERENCE/'verl/trainer/config/sdpo.yaml')
    # The existing pinned-YAML override helper is permissive about new keys.
    # Check every override against its actual defaults BEFORE applying it.
    additions={'v4_context_source','actor_rollout_ref.model.override_config.attn_implementation'}
    for arg in cmd[cmd.index('--config-name')+2:]:
        key=arg.lstrip('+').split('=',1)[0]
        if arg.startswith('+'):
            assert key in additions or key.startswith('ray_kwargs.ray_init.runtime_env.env_vars.'),key
        else:
            node=base
            for part in key.split('.'):
                assert isinstance(node,dict) and part in node,('unknown pinned override',key,part)
                node=node[part]
    return config(overrides(base,cmd))


@pytest.fixture
def stage_inputs(tmp_path,monkeypatch):
    monkeypatch.setenv('V4_PROFILE','technical-smoke');monkeypatch.setenv('KIT_FINISH_GATE','1')
    tokenizer=Tokenizer();_,finqa,_=frozen_inputs(tokenizer,ROOT/'kit/resources/p/smoke3-settings.json')
    prep=inputs.write_data(tmp_path/'preparation-data',inputs.converted_rows(tokenizer,finqa,preparation=True))
    dev=inputs.write_data(tmp_path/'data',smoke.development_rows(tokenizer,finqa))
    batch=tmp_path/'p2/s101';atomic(batch.parent/'calibration.json',dict(variant='p2-l2-init',seeds={'101':dict(task_grad_norm=2.,incoming_distance=2.)}))
    stages={}
    for stage in STAGES:
        data=prep if stage=='preparation' else dev
        if stage=='resume':
            # Use the actual smoke's rewrite helper on the unconsumed replay,
            # keeping rows 0..63 identical to the prefix data.
            import pyarrow.parquet as pq
            rows=pq.read_table(dev/'train.parquet').to_pylist()
            smoke.resume_rows(rows,tokenizer)
            data=inputs.write_data(tmp_path/'resume-data',rows)
        run=tmp_path/('drift' if stage in ('prefix','resume') else stage)
        cmd,env=smoke.stage_command(stage,tmp_path/'base',data,run,batch=batch,anchor=tmp_path/'base',
            value=.1 if stage=='p2' else 0.,boundary=tmp_path/'drift/prefix.json')
        forwarded={arg.split('=',1)[0].rsplit('.',1)[1] for arg in cmd if arg.startswith('+ray_kwargs.ray_init.runtime_env.env_vars.')}
        stages[stage]=(data,cmd,{k:env[k] for k in forwarded})
    return tokenizer,finqa,stages


@pytest.mark.parametrize('stage',STAGES)
def test_every_stage_real_parquet_loader_collation_and_teacher_builder(stage,stage_inputs,pinned_loader,monkeypatch):
    tokenizer,finqa,stages=stage_inputs;data,cmd,env=stages[stage];cfg=resolved(cmd)
    for key,value in env.items():
        if key in ('P3B_DATA','P3B_REPLAY','V4_PROFILE','KIT_FINISH_GATE'):monkeypatch.setenv(key,value)
    # Same C runner allowances, then the unchanged objective and contract checks.
    from kit.c_sdft_entry import legacy_config
    objective.check_trainer_config(legacy_config(cfg))
    v4_contract.check_resolved(legacy_config(cfg),'D',profile_name='technical-smoke',
        resume_path=str(Path(cmd[-1].split('=',1)[1])) if stage=='resume' else None)
    from kit.p3b_train import check_data
    check_data(data)
    assert cfg.data.custom_cls.path is None and pinned_loader.get_dataset_class(cfg.data) is pinned_loader.RLHFDataset
    dataset=pinned_loader.RLHFDataset(str(data/'train.parquet'),tokenizer,cfg.data)
    assert len(dataset)==1280
    v4_contract.check_schedule(dataset.dataframe,4,tokenizer)
    from kit.c_data import encode
    # Execute the actual teacher entry function (including replay selection and
    # privileged-context handling); no trainer/worker is constructed here.
    source=ast.parse((ROOT/'kit/sdft_verl.py').read_text())
    fn=next(n for n in source.body if isinstance(n,ast.FunctionDef) and n.name=='build_sdft_teacher_batch')
    scope=dict(objective=objective,os=os,OmegaConf=NS(to_container=lambda value,**kw:value),_SHOWN=True)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),'kit/sdft_verl.py','exec'),scope)
    sys.modules['verl'].DataProto=NS(from_dict=lambda **kwargs:NS(**kwargs))
    from kit.c_reward import compute_score as legal_reward
    from kit.p3b_reward import compute_score as finqa_reward
    assert cfg.data.shuffle is False
    first=2 if stage=='resume' else 0  # continuation starts after the two yielded prefix batches
    for update in range(first,first+4):
        batch=pinned_loader.collate_fn([dataset[i] for i in range(update*32,(update+1)*32)])
        infos=batch['extra_info'];flags=[bool(e['p3b_replay']) for e in infos]
        assert [i+1 for i,f in enumerate(flags) if f]==[8,16,24,32]
        assert sum(flags)==4 and len(flags)-sum(flags)==28
        demonstrations=[None if flag else objective.privileged_context(info,'demonstration') for info,flag in zip(infos,flags)]
        responses=torch.tensor([[8,9,10,2]]*32);mask=torch.ones_like(responses)
        built=mixed_teacher_inputs(tokenizer,batch['raw_prompt'],demonstrations,responses,mask,replay=flags,
            chat_template_kwargs=cfg.data.apply_chat_template_kwargs,max_reprompt_len=14336)
        assert built['teacher_input_ids'].shape[0]==32 and built['self_distillation_mask'].tolist()==[1.]*32
        assert max(built['teacher_prompt_lengths'])<=14336
        tensors=dict(input_ids=torch.ones(32,8,dtype=torch.long),responses=responses,response_mask=mask)
        batch['uid']=np.array([str(i) for i in range(32)],dtype=object)
        actual,_=scope['build_sdft_teacher_batch'](NS(config=cfg,tokenizer=tokenizer),
            NS(non_tensor_batch=batch,batch=tensors),torch.zeros(32,4))
        assert torch.equal(actual.tensors['teacher_input_ids'],built['teacher_input_ids'])
        reward=finqa_reward if stage=='preparation' else legal_reward
        for prompt,info,src,target in zip(batch['raw_prompt'],infos,batch['data_source'],batch['reward_model']):
            encode(tokenizer,prompt,info['demonstration'])  # student/response cap and EOS-inclusive masking
            assert info['p3b_target_ids'][-1]==tokenizer.eos_token_id
            if not info['p3b_replay']:objective.teacher_messages(prompt,info['demonstration'])
            reward(src,'Answer: 2' if stage=='preparation' else 'Entailment',target['ground_truth'],dict(info,truncated=False))
    assert all(not dataset[i]['extra_info']['p3b_replay'] for i in (0,1,2,3))
    if os.environ.get('P2_CPU_STAGE_AUDIT_RECORDS'):
        destination=Path(os.environ['P2_CPU_STAGE_AUDIT_RECORDS'])
        assert destination.is_relative_to('/private'+'/tmp')
        record=dict(stage=stage,batches=4,first_batch=first+1,task_per_batch=28,replay_positions=[8,16,24,32],
            manifest_sha256=hashlib.sha256((data/'manifest.json').read_bytes()).hexdigest(),
            pinned_dataset_sha256=hashlib.sha256((REFERENCE/'verl/utils/dataset/rl_dataset.py').read_bytes()).hexdigest())
        with destination.open('a') as stream:stream.write(json.dumps(record,sort_keys=True)+'\n')


@pytest.mark.parametrize('stage',STAGES)
def test_every_stage_required_environment_is_forwarded(stage,stage_inputs):
    _,_,stages=stage_inputs;data,cmd,env=stages[stage]
    resolved(cmd)
    prefix='+ray_kwargs.ray_init.runtime_env.env_vars.'
    forwarded={arg[len(prefix):].split('=',1)[0]:json.loads(arg.split('=',1)[1]) for arg in cmd if arg.startswith(prefix)}
    required={'P3B_DATA','P3B_REPLAY','V4_PROFILE','V4_TELEMETRY','V4_CHECKPOINT_STATE','KIT_FINISH_GATE','KIT_V4_ARM','ARM',
        'SDPO_DIR','QWEN3_8B_TOKENIZER','PYTHONPATH','VERL_FILE_LOGGER_PATH','X_INTEGRATION_SMOKE','X_P2_DETERMINISTIC_SMOKE',
        'CUBLAS_WORKSPACE_CONFIG','PYTHONHASHSEED'}
    if stage in ('prefix','resume'):
        required.add('M2_ADDON_BOUNDARY')
        if stage=='prefix':required.add('M2_ADDON_PREFIX')
        assert 'X_P2_MODE' not in forwarded
    else:
        required.update({'X_P2_MODE','X_P2_BATCH','X_P2_RUN','X_P2_ANCHOR','X_P2_LAMBDA','X_P2_SEED'})
        if stage in ('control','p2'):required.update({'X_P2_CALIBRATION','X_P2_CALIBRATION_SHA'})
        if stage!='measure':required.add('X_P2_SMOKE_ROLE')
        if stage=='p2':required.add('X_P2_EARLY_VERDICT')
    assert required<=forwarded.keys()
    assert all(forwarded[k]==env[k] for k in required)
    assert forwarded['P3B_REPLAY']=='1' and forwarded['P3B_DATA']==str(data)
    assert forwarded.get('M2_ADDON_PREFIX')==('1' if stage=='prefix' else None)
    reward=next(a.split('=',1)[1] for a in cmd if a.startswith('custom_reward_function.path='))
    assert Path(reward).name==('p3b_reward.py' if stage=='preparation' else 'c_reward.py')


def test_schedule_matches_next_slots_compose_and_resume_changes_only_unconsumed_replay(stage_inputs,tmp_path):
    tokenizer,finqa,stages=stage_inputs
    from kit.next_slots import compose
    from kit.p3b_common import frozen_files
    import pyarrow.parquet as pq
    pools=tmp_path/'pools';atomic(pools/'manifest.json',dict(files={},technical_synthetic=True))
    for preparation in (True,False):
        task,replay=inputs.source_rows(finqa,preparation)
        expected,audit=compose(task,replay,pools,'generic',101,tokenizer,task_name='finqa' if preparation else 'contract_nli')
        rows=pq.read_table(stages['preparation' if preparation else 'control'][0]/'train.parquet').to_pylist()
        assert [r['id'] for r in rows]==[str(r['id']) for r in expected]
        assert audit['slot_map']=={'generic':[8,16,24,32]} and audit['exposures']['generic']==160
    prefix=pq.read_table(stages['prefix'][0]/'train.parquet').to_pylist()
    resumed=pq.read_table(stages['resume'][0]/'train.parquet').to_pylist()
    assert prefix[:64]==resumed[:64]
    for before,after in zip(prefix[64:],resumed[64:]):
        if before['extra_info']['p3b_replay']:
            assert after['extra_info']['demonstration']=='two'
            assert before['extra_info']['p3b_replay']==after['extra_info']['p3b_replay']
        else:assert before==after


def test_former_all_replay_preparation_fails_before_dispatch(stage_inputs,tmp_path):
    import pyarrow.parquet as pq
    _,_,stages=stage_inputs
    broken=pq.read_table(stages['preparation'][0]/'train.parquet').to_pylist()
    for row in broken:row['extra_info']['p3b_replay']=True
    with pytest.raises(ValueError,match='production generic slots'):inputs.write_data(tmp_path/'bad',broken)
    with pytest.raises(ValueError,match='needs task items'):
        mixed_teacher_inputs(Tokenizer(),[r['prompt'] for r in broken[:32]],[None]*32,
            torch.ones(32,4,dtype=torch.long),torch.ones(32,4,dtype=torch.long),replay=[True]*32,max_reprompt_len=14336)


def test_final_attempt_requires_current_complete_CPU_audit_before_claim(tmp_path,monkeypatch):
    from kit import x_smoke_settings as settings
    from kit.p2_smoke import code_hashes
    monkeypatch.setitem(settings.AUTHORISED_ATTEMPTS,'p2',3)  # Preserve this historical attempt-3 authority test.
    stem=settings.PART_STEMS['p2'];directory=tmp_path/'claims';proof=tmp_path/'audit.json'
    for n in (1,2):atomic(directory/(stem+f'.attempt{n}.json'),dict(attempt=n))
    original=settings.check_stage_audit
    monkeypatch.setattr(settings,'check_stage_audit',lambda:original(proof))
    with pytest.raises(ValueError,match='passing CPU stage audit'):
        smoke.reserve_attempt(directory,'p2',ledger=stem+' attempt 3')
    assert not (directory/(stem+'.attempt3.claim.json')).exists()
    value=dict(schema='p2-cpu-stage-audit.v1',passed=True,code_sha256=code_hashes(),
        test_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        stages={stage:dict(batches=4,task_per_batch=28,replay_positions=[8,16,24,32]) for stage in STAGES})
    atomic(proof,value);assert original(proof)==value
    assert smoke.reserve_attempt(directory,'p2',ledger=stem+' attempt 3')==(3,stem)
    with pytest.raises(ValueError,match='exhausted'):smoke.reserve_attempt(directory,'p2',ledger=stem+' attempt 4')
    value['code_sha256']={};atomic(proof,value)
    with pytest.raises(ValueError,match='code changed'):original(proof)


def test_attempt3_receipt_still_requires_full_control_P2_batch_pairing():
    from kit.x_p import validate_smoke
    from kit.p2_smoke import code_hashes,ASSERTIONS,SCHEMA
    child=dict(schema=SCHEMA,variant='p2-l2-init',placement='real-verl-fsdp2',world_size=2,
        model_revision=smoke.MODEL_REVISION,precision='fp32-masters/fp64-accumulation',technical_synthetic=False,
        passed=True,assertions={k:dict(status='PASS') for k in ASSERTIONS},code_sha256=code_hashes())
    value=dict(schema='x-integration-smoke.v3',part='p2',attempt=3,maximum_attempts=3,retries=0,gpu='H100:2',
        timeout_seconds=1800,total_ceiling_usd=9.77,evidence_stem='x-integration-p2-smoke',passed=True,p2_smoke=child)
    with pytest.raises(ValueError,match='paired batches'):validate_smoke(value,False)
    child['paired_batches']=dict(schema='x-smoke-paired-batches.v1',smoke_only=True,control_first=True,
        updates=[dict(update=i,identity=True,sha256='a'*64,input_hash='b'*64) for i in range(4)])
    assert validate_smoke(value,False)==child


def test_generic_replay_eligibility_preserves_full_pool_skip_order(tmp_path):
    from kit.next_slots import compose
    tokenizer=Tokenizer()
    pool=[dict(id='long',prompt=[dict(role='user',content='x'*20000)],response='target'),
        dict(id='short',prompt=[dict(role='user',content='tiny question')],response='frozen target')]
    task,replay=inputs.source_rows(pool)
    atomic(tmp_path/'manifest.json',dict(files={}))
    expected,audit=compose(task,replay,tmp_path,'generic',101,tokenizer,task_name='contract_nli')
    actual=inputs.converted_rows(tokenizer,pool)
    assert [r['id'] for r in actual]==[str(r['id']) for r in expected]
    assert audit['skipped_draw_count']>0 and audit['pool_selection']['generic']['skipped_ids']==['long']
    assert all(r['id']=='short' for r in actual if r['extra_info']['p3b_replay'])


def test_preparation_dispatch_uses_the_exact_audited_input_files(stage_inputs,tmp_path,monkeypatch):
    tokenizer,finqa,stages=stage_inputs
    class StopBeforeTrainer(Exception):pass
    def stop(cmd,env,run):
        # Actual prepare(), including its writer and stage command, has run.
        import pyarrow.parquet as pq
        for name in ('train.parquet','test.parquet','val.parquet'):
            assert pq.read_table(Path(env['P3B_DATA'])/name).to_pylist()==pq.read_table(stages['preparation'][0]/name).to_pylist()
        assert read(Path(env['P3B_DATA'])/'manifest.json')==read(stages['preparation'][0]/'manifest.json')
        assert resolved(cmd).trainer.total_training_steps==4 and env['X_P2_MODE']=='prepare'
        raise StopBeforeTrainer()
    monkeypatch.setattr(smoke,'execute',stop)
    with pytest.raises(StopBeforeTrainer):smoke.prepare(tmp_path/'actual',tmp_path/'base',tokenizer,finqa)


def test_resume_dispatch_uses_the_exact_audited_prefix_and_resume_files(stage_inputs,tmp_path,monkeypatch):
    import pyarrow.parquet as pq
    tokenizer,_,stages=stage_inputs
    rows=pq.read_table(stages['prefix'][0]/'train.parquet').to_pylist()
    root=tmp_path/'actual';data=inputs.write_data(root/'data',rows);base=root/'base';atomic(base/'config.json',dict(tiny=True))
    calls=[]
    def execute(cmd,env,run):
        stage='prefix' if not calls else 'resume';calls.append(stage)
        for name in ('train.parquet','test.parquet','val.parquet'):
            assert pq.read_table(data/name).to_pylist()==pq.read_table(stages[stage][0]/name).to_pylist()
        assert read(data/'manifest.json')==read(stages[stage][0]/'manifest.json')
        if stage=='prefix':
            atomic(run/'prefix.json',dict(update=2));atomic(run/'checkpoint-memory-step2.json',dict(status='PASS'))
        else:atomic(run/'restore-qualified.json',dict(status='PASS',before_update=3))
    monkeypatch.setattr(smoke,'execute',execute)
    assert smoke.run_resume(root,base,data,rows,tokenizer,base)['passed']
    assert calls==['prefix','resume']
