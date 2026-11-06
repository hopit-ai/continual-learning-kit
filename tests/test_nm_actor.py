from types import SimpleNamespace as NS
from copy import deepcopy
import torch
import pytest
from kit import nm_actor
from kit import sdft_actor

@pytest.fixture(autouse=True)
def one_thread():
    previous=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__();self.emb=torch.nn.Embedding(9,3);self.head=torch.nn.Linear(3,9,bias=False)
    def get_output_embeddings(self):return self.head
    def forward(self,input_ids,attention_mask,position_ids,use_cache=False):
        # Prefix dependency makes demonstration context affect task targets.
        hidden=self.emb(input_ids).cumsum(1)/(torch.arange(input_ids.shape[1],device=input_ids.device)[None,:,None]+1)
        return NS(logits=self.head(hidden))


class Rows:
    def __init__(self,rows):self.rows=rows;self.meta_info=dict(temperature=1.)
    def split(self,n):
        return [self] if n==32 else self.rows


class Row:
    def __init__(self,i,*,replay=False,adv=1.):
        ids=torch.tensor([[1,2,3,4,5,6,7,8]])
        self.batch=dict(input_ids=ids,responses=ids[:,3:],response_mask=torch.ones(1,5),attention_mask=torch.ones_like(ids),position_ids=torch.arange(8)[None],
            teacher_input_ids=ids.clone(),teacher_attention_mask=torch.ones_like(ids),teacher_position_ids=torch.arange(8)[None],
            self_distillation_mask=torch.ones(1),old_log_probs=torch.full((1,5),-2.),rollout_log_probs=torch.full((1,5),-2.),nm_advantage=torch.tensor([adv]))
        self.batch['teacher_input_ids'][0,1]=7
        self.non_tensor_batch={'extra_info':[dict(m3_replay=replay,m3_target_ids=[4,5,6,8],m3_eos_token_id=8,nm_update=1,nm_position=i+1)]}
    def to(self,device):return self


def owner(family='isdft',arm='C',recipe='I0',rows=32):
    torch.manual_seed(3);model=Tiny();teacher=deepcopy(model);teacher.requires_grad_(False)
    base=deepcopy(model);base.requires_grad_(False);base.eval()
    config=NS(ppo_epochs=1,ppo_mini_batch_size=32,self_distillation=NS(full_logit_distillation=True,alpha=0.,distillation_topk=None,distillation_add_tail=False,teacher_regularization='ema',teacher_update_rate=.01,include_environment_feedback=False,dont_reprompt_on_self_success=False,is_clip=None),loss_agg_mode='seq-mean-token-mean')
    # Use the repository's actual self-distillation validator contract.
    from kit.sdft_objective import check_self_distillation
    o=NS(actor_module=model,teacher_module=teacher,frozen_base_module=base if family in ('isdft','grpo') or recipe=='K' else None,
         actor_optimizer=torch.optim.AdamW(model.parameters(),lr=.001),config=config,scaler=None,ulysses_sequence_parallel_size=1,
         nm_config=dict(family=family,arm=arm,recipe=recipe,rows_per_update=rows,balance_batch=False),nm_world_size=1)
    o.ema_steps=0
    def ema():
        o.ema_steps+=1
        with torch.no_grad():
            for p,t in zip(model.parameters(),teacher.parameters()):t.lerp_(p,.01)
    def step():
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);o.actor_optimizer.step();return norm
    o._optimizer_step=step;o._update_teacher=ema
    return o


def test_native_nm_zero_component_matches_inherited_D():
    a=owner();b=owner();rows=Rows([Row(i) for i in range(32)])
    nm_actor.update_policy(a,deepcopy(rows));sdft_actor.update_policy(b,deepcopy(rows))
    for x,y in zip(a.actor_module.parameters(),b.actor_module.parameters()):assert torch.equal(x,y)
    for x,y in zip(a.teacher_module.parameters(),b.teacher_module.parameters()):assert torch.equal(x,y)
    assert a._v4_completed_updates==b._v4_completed_updates==1


@pytest.mark.parametrize('recipe,rows,replay_rows',[('K',32,4),('CR',48,16)])
def test_native_nm_replay_both_recipe_geometries(recipe,rows,replay_rows):
    a=owner('freeze','C',recipe,rows);batch=Rows([Row(i,replay=i>=rows-replay_rows) for i in range(rows)])
    r=nm_actor.update_policy(a,batch)
    assert r['nm/global_divisor']==32 and a.ema_steps==1
    assert all(p.grad is None for p in a.actor_module.parameters())


def test_native_isdft_anchor_on_task_only_and_reference_no_grad():
    a=owner('isdft','T',recipe='K');batch=Rows([Row(i,replay=i>=28) for i in range(32)])
    # K and task-anchor need distinct plain/supervised-prefix base projections.
    r=nm_actor.update_policy(a,batch)
    assert r['v4/completed_optimizer_updates']==1 and all(p.grad is None for p in a.frozen_base_module.parameters())


def test_native_grpo_has_no_ema_step_and_non_gating_telemetry():
    a=owner('grpo','T');batch=Rows([Row(i,adv=1 if i%4<2 else -1) for i in range(32)])
    batch.meta_info['nm_telemetry']={'nm/reward_mean':.5,'nm/mixed_groups':1.,'nm/zero_reward_groups':0.}
    r=nm_actor.update_policy(a,batch)
    assert a.ema_steps==0 and r['nm/reward_mean']==.5 and r['nm/base_forward_KL']>=-1e-5
    assert all(p.grad is None for p in a.frozen_base_module.parameters())


def test_isdft_shares_student_forward_for_aligned_rank_collectives():
    a=owner('isdft','T',recipe='K');calls=[]
    hook=a.actor_module.register_forward_pre_hook(lambda *args:calls.append(1))
    nm_actor.update_policy(a,Rows([Row(i,replay=i>=28) for i in range(32)]))
    hook.remove()
    assert len(calls)==32  # Replay and task ranks each execute one actor forward/backward.


@pytest.mark.parametrize('family,arm',[('isdft','C'),('isdft','T'),('grpo','C'),('freeze','T')])
def test_missing_task_demonstration_is_refused(family,arm):
    a=owner(family,arm);rows=Rows([Row(i) for i in range(32)])
    rows.rows[0].batch['self_distillation_mask'].zero_()
    with pytest.raises(ValueError,match='task demonstration target missing'):nm_actor.update_policy(a,rows)
    assert not a.actor_optimizer.state and a.ema_steps==0


def test_grpo_without_demonstration_uses_verifier_objective():
    a=owner('grpo','T');rows=Rows([Row(i) for i in range(32)])
    for row in rows.rows:row.batch.pop('self_distillation_mask')
    rows.meta_info['nm_telemetry']={}
    nm_actor.update_policy(a,rows)
    assert a.ema_steps==0
