import ast
from copy import deepcopy
from types import SimpleNamespace as NS
from pathlib import Path
import pytest
from kit.nm_common import *
from kit import nm_prepare as p


def test_v2b_teacher_swap_keeps_student_and_target_prefixes():
    class Tok:
        eos_token='!';eos_token_id=0
        def encode(self,text,add_special_tokens=False):return [0 if x=='!' else ord(x) for x in text]
        def apply_chat_template(self,messages,**kwargs):return self.encode(' '.join(m['content'] for m in messages))
    rows=[dict(id='a',prompt='filing question',response='verified answer',teacher_context='retrieved checked case',gold='3')]
    caps=dict(prompt=10240,teacher=14336)
    control,ca=p.converted_rows(rows,Tok(),'v2b','C',caps,32)
    challenger,ta=p.converted_rows(rows,Tok(),'v2b','T',caps,32)
    assert control[0]['prompt']==challenger[0]['prompt']
    assert control[0]['extra_info']['m3_target_ids']==challenger[0]['extra_info']['m3_target_ids']
    assert control[0]['extra_info']['demonstration']=='verified answer'
    assert challenger[0]['extra_info']['demonstration']=='retrieved checked case'
    with pytest.raises(Invalid):p.converted_rows(rows,Tok(),'v2b','T',dict(prompt=2,teacher=100),32)


def check_function():
    # Import the real validator body on CPU without importing Ray/verl workers.
    from kit import nm_common,sdft_objective
    tree=ast.parse((KIT/'nm_native.py').read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='check_config')
    namespace=dict(vars(nm_common),deepcopy=deepcopy,objective=sdft_objective,OmegaConf=NS(to_container=lambda c,resolve:deepcopy(c)))
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(KIT/'nm_native.py'),'exec'),namespace)
    return namespace['check_config']


def native_config(tmp_path,family='freeze',rows=32,updates=40):
    b=tmp_path/'base.json';atomic(b,dict(base=True))
    caps=dict(prompt=2048 if family=='grpo' else 10240,teacher=6144 if family=='grpo' else 14336,response=2048)
    sd=dict(full_logit_distillation=True,alpha=0.,distillation_topk=None,distillation_add_tail=False,is_clip=None,
        teacher_regularization='ema',teacher_update_rate=.01,include_environment_feedback=False,max_reprompt_len=caps['teacher'])
    actor=dict(ppo_epochs=1,ppo_micro_batch_size_per_gpu=1,strategy='fsdp2',ppo_mini_batch_size=rows,self_distillation=sd,
        loss_agg_mode='seq-mean-token-mean',entropy_coeff=0.,use_kl_loss=False,policy_loss=dict(loss_mode='sdpo'),
        ulysses_sequence_parallel_size=1,optim=dict(lr=1e-5,weight_decay=.01,lr_warmup_steps=10,lr_scheduler_type='constant'),
        data_loader_seed=101,fsdp_config=dict(seed=101))
    return dict(nm=dict(family=family,arm='C',seed=101,base_revision=BASE,updates=updates,rows_per_update=rows,caps=caps,
            balance_batch=False,base_binding=str(b),base_binding_sha256=sha(b)),
        trainer=dict(n_gpus_per_node=8,nnodes=1,total_training_steps=updates,save_freq=4 if updates==8 else 10,balance_batch=False,
            test_freq=-1,resume_mode='disable',val_before_train=False),
        data=dict(train_batch_size=8 if family=='grpo' else rows,max_prompt_length=caps['prompt'],max_response_length=2048,
            filter_overlong_prompts=False,shuffle=False,truncation='error',seed=101),
        actor_rollout_ref=dict(actor=actor,rollout=dict(n=4 if family=='grpo' else 1,temperature=.7 if family=='grpo' else 1.,top_p=1.,calculate_log_probs=True),
            model=dict(use_remove_padding=False,use_fused_kernels=False)),
        algorithm=dict(rollout_correction=dict(rollout_is='token',rollout_is_threshold=2.)))


@pytest.mark.parametrize('family,rows,updates',[('freeze',32,40),('fisher',48,40),('grpo',32,40),('grpo',32,8)])
def test_real_native_validator_preserves_all_other_D_constants(tmp_path,monkeypatch,family,rows,updates):
    monkeypatch.setenv('KIT_FINISH_GATE','1');monkeypatch.setenv('V4_PROFILE','scientific')
    c=native_config(tmp_path,family,rows,updates);check=check_function()
    assert check(c)['family']==family
    c['actor_rollout_ref']['actor']['optim']['lr']=2e-5
    with pytest.raises(ValueError):check(c)


def test_issuer_component_excluded_even_without_direct_cik_overlap():
    from kit.nm_data import eligible_case
    def item(ident,cik,group):
        return dict(id=ident,sec=dict(issuer_ciks=[cik],issuer_group=group,accession=ident,filing_group=ident,
            required_accessions=[],required_filing_groups=[]))
    assert not eligible_case(item('a','1','same'),item('b','2','same'))
    assert eligible_case(item('a','1','one'),item('b','2','two'))


def test_cpu_lane_plan_counts_and_general_never_uses_sec_context(tmp_path,monkeypatch):
    from kit import nm_pipeline as pipeline
    model=tmp_path/'model';model.mkdir();(model/'config.json').write_text('{}')
    src=dict(model=str(model),sha256='a'*64,files={'config.json':'b'*64})
    monkeypatch.setattr(pipeline,'bound',lambda *args:dict(recipe={}))
    monkeypatch.setattr(pipeline,'exports_for',lambda *args:{str(s):src for s in STEPS})
    monkeypatch.setattr(pipeline.nm_prepare,'render_scoring',lambda items,*args:[r['prompt'] for r in items])
    bundle=dict(base=src,tokenizer=str(model),panels={'instruction':[dict(id='x',prompt='question')]})
    path=pipeline.prepare_lane(tmp_path,'v2b',101,'instruction',bundle)
    plan=load(path/'lane.json')
    assert len(plan['commands'])==6
    assert all(load(c['plan'])['panel']=='instruction' for c in plan['cells'])


def test_cpu_model_binding_includes_index_and_gpu_load_never_rehashes_weights(tmp_path,monkeypatch):
    from kit import nm_model,m3_actor
    model=tmp_path/'model';model.mkdir()
    atomic(model/'config.json',dict(model_type='qwen3',hidden_size=4096,num_hidden_layers=36))
    (model/'model-00001.safetensors').write_bytes(b'CPU fixture weights')
    atomic(model/'model.safetensors.index.json',dict(weight_map={'weight':'model-00001.safetensors'}))
    files={p.name:sha(p) for p in model.iterdir()}
    record=p.verify_model(dict(model=str(model),sha256=digest(files)))
    assert 'model.safetensors.index.json' in record['files']
    binding=tmp_path/'base.json';atomic(binding,dict(revision=BASE,identity=dict(model_file_hashes=files)))
    stamp=nm_model.metadata(model,files);original=m3_actor.verify_base_files
    def loader(directory,revision,mesh,*,binding,**kw):return m3_actor.verify_base_files(directory,binding,revision)
    monkeypatch.setattr(m3_actor,'load_frozen_base',loader)
    real_sha=nm_model.sha
    def small_only(path):
        assert Path(path)==binding
        return real_sha(path)
    monkeypatch.setattr(nm_model,'sha',small_only)
    assert nm_model.prepared_reference(model,BASE,None,binding=binding,binding_sha256=sha(binding),files_metadata=stamp)
    assert m3_actor.verify_base_files is original
    (model/'model-00001.safetensors').write_bytes(b'changed CPU fixture')
    with pytest.raises(Invalid):nm_model.prepared_reference(model,BASE,None,binding=binding,binding_sha256=sha(binding),files_metadata=stamp)
    assert m3_actor.verify_base_files is original


def test_gpu_lane_rejects_changed_cpu_cell_plan_before_starting_models(tmp_path,monkeypatch):
    from kit import nm_pipeline as pipeline,lane_pool
    cell=tmp_path/'cell.json';atomic(cell,dict(frozen=True))
    atomic(tmp_path/'lane.json',dict(cells=[dict(plan=str(cell),plan_sha256=sha(cell),output='score.json')],commands=[]))
    atomic(cell,dict(frozen=False))
    monkeypatch.setattr(lane_pool,'execute',lambda *a,**kw:(_ for _ in ()).throw(AssertionError('must not launch')))
    with pytest.raises(Invalid,match='CPU scoring plan changed'):pipeline.run_lane(tmp_path,tmp_path/'attempt')
