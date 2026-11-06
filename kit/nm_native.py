"""Import only in the pinned trainer environment. NM subclasses, native state/EMA."""
from pathlib import Path
from copy import deepcopy
import ray
import torch
from omegaconf import OmegaConf,open_dict
import verl.trainer.main_ppo as main_ppo
from verl.trainer.ppo.ray_trainer import RayPPOTrainer,Role
from verl.single_controller.base.decorator import register,Dispatch
from kit.sdft_verl import SdftActorRolloutRefWorker,check_pinned
from kit import sdft_objective as objective
from kit.nm_common import *
from kit import nm_loss,nm_actor
from kit.nm_model import prepared_reference
from kit.m3_runner import teacher_batch


def check_config(config):
    c=OmegaConf.to_container(config,resolve=True);n=c['nm'];a=c['actor_rollout_ref']['actor'];r=c['actor_rollout_ref']['rollout']
    need(n['family'] in FAMILIES and n['arm'] in ('C','T') and n['seed'] in SEEDS,'unregistered native comparison')
    need(c['trainer']['n_gpus_per_node']==8 and c['trainer']['nnodes']==1,'native NM training needs eight ranks')
    need(n['base_revision']==BASE and n['updates'] in (8,40),'native dose/anchor differs')
    need(sha(n['base_binding'])==n['base_binding_sha256'],'CPU base binding changed')
    if 'incoming' in n:
        from kit.nm_model import metadata
        need(metadata(n['incoming']['model'],n['incoming']['files'])==n['incoming_metadata'],'prepared FinQA start changed')
    need(c['trainer']['total_training_steps']==n['updates'] and c['trainer']['save_freq']==(4 if n['updates']==8 else 10),'native checkpoint cadence differs')
    grouped=n['family']=='grpo';rows=n['rows_per_update'];caps=n['caps']
    need(rows in (32,48) and c['data']['train_batch_size']==(8 if grouped else rows),'native prompt geometry differs')
    need(r['n']==(4 if grouped else 1) and a['ppo_mini_batch_size']==rows,'grouped trajectory dose differs')
    need(r['temperature']==(.7 if grouped else 1.) and r['top_p']==1.,'sampling recipe differs')
    need(c['data']['max_prompt_length']==caps['prompt'] and c['data']['max_response_length']==2048 and a['self_distillation']['max_reprompt_len']==caps['teacher'],'admitted caps differ')
    need(c['data']['shuffle'] is False and c['trainer']['balance_batch']==n['balance_batch'],'bound balancing policy differs')
    need(a['ppo_micro_batch_size_per_gpu']==1 and a['strategy']=='fsdp2' and a['ppo_epochs']==1,'native reduction path differs')
    # All other D checks run on a detached copy, normalising only registered deviations.
    legacy=deepcopy(c);legacy['data']['train_batch_size']=32;legacy['data']['max_prompt_length']=2048
    legacy['actor_rollout_ref']['actor']['ppo_mini_batch_size']=32
    legacy['actor_rollout_ref']['actor']['self_distillation']['max_reprompt_len']=6144
    legacy['actor_rollout_ref']['rollout']['n']=1;legacy['actor_rollout_ref']['rollout']['temperature']=1.
    legacy['trainer']['total_training_steps']=40;legacy['trainer']['save_freq']=10
    objective.check_trainer_config(legacy)
    from kit.v4_contract import check_resolved
    check_resolved(legacy,'D',profile_name='scientific',resume_path=c['trainer'].get('resume_from_path') if c['trainer']['resume_mode']=='resume_path' else None)
    return n


class NMTrainer(RayPPOTrainer):
    _maybe_build_self_distillation_batch=teacher_batch
    def _update_actor(self,batch):
        n=self.config.nm;rows=n.rows_per_update;world=self.config.trainer.n_gpus_per_node
        need(len(batch)==rows,'complete NM update required')
        if n.family=='grpo':
            from copy import deepcopy
            for i in range(len(batch)):
                extra=deepcopy(batch.non_tensor_batch['extra_info'][i]);extra['nm_position']=i+1
                batch.non_tensor_batch['extra_info'][i]=extra
        if n.family=='grpo':
            rewards=batch.batch['token_level_scores'].sum(-1)
            need(bool(((rewards==0)|(rewards==1)).all()),'GRPO verifier must be binary')
            grouped=rewards.reshape(8,4);adv=nm_loss.advantages(grouped).reshape(-1)
            # Advantages are detached, shared by each response's tokens only in the NM loss.
            batch.batch['nm_advantage']=adv
            batch.meta_info['nm_telemetry']={'nm/reward_mean':float(rewards.mean()),
                'nm/mixed_groups':float(((grouped.sum(-1)>0)&(grouped.sum(-1)<4)).float().mean()),
                'nm/zero_reward_groups':float((grouped.sum(-1)==0).float().mean()),
                'nm/all_correct_groups':float((grouped.sum(-1)==4).float().mean()),
                'nm/zero_gradient_groups':float(((grouped.sum(-1)==0)|(grouped.sum(-1)==4)).float().mean())}
        # Freeze round-major schedule positions; transpose only to native rank dispatch.
        order=torch.tensor([i for rank in range(world) for i in range(rank,rows,world)],device=batch.batch['input_ids'].device)
        batch.reorder(order)
        try:return super()._update_actor(batch)
        finally:batch.reorder(torch.argsort(order))


class NMWorker(SdftActorRolloutRefWorker):
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        result=super().init_model()
        if not self._is_actor:return result
        n=OmegaConf.to_container(self.config.nm,resolve=True)
        self.actor.__class__=nm_actor.actor_class(type(self.actor));self.actor.nm_config=n;self.actor.nm_world_size=self.world_size
        self.actor.frozen_base_module=None
        if n['family']=='grpo' or (n['family']=='isdft' and n['arm']=='T') or n.get('recipe')=='K':
            if n['family']=='grpo' and n['arm']=='T':
                self.actor.frozen_base_module=self.actor.teacher_module
                self.actor.frozen_base_module.eval();self.actor.frozen_base_module.requires_grad_(False)
            else:
                self.actor.frozen_base_module=prepared_reference(n['base_model'],BASE,self.device_mesh,
                    binding=Path(n['base_binding']),binding_sha256=n['base_binding_sha256'],files_metadata=n['base_metadata'],attention_implementation=self.config.model.get('override_config',{}).get('attn_implementation','flash_attention_2'))
        self._nm_mask=None;self._nm_frozen=[];self._nm_frozen_teacher=[]
        if n.get('measurement'):
            torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
        if n['family']=='freeze' and n['arm']=='T':
            self._nm_mask=nm_loss.freeze_mask(self.actor.actor_module);nm_loss.filter_optimizer(self.actor_optimizer)
            self._nm_frozen=[(p,nm_loss.local(p).detach().clone()) for p in self.actor.actor_module.parameters() if not p.requires_grad]
            frozen_names={name for name,p in self.actor.actor_module.named_parameters() if not p.requires_grad}
            self._nm_frozen_teacher=[(p,nm_loss.local(p).detach().clone()) for name,p in self.actor.teacher_module.named_parameters() if name in frozen_names]
            need(len(self._nm_frozen_teacher)==len(self._nm_frozen),'EMA frozen mask geometry differs')
            need(all(torch.equal(before,student) for (_,before),(_,student) in zip(self._nm_frozen_teacher,self._nm_frozen)),'EMA frozen blocks must start at student')
        if n.get('penalty') is not None:
            from kit.p2_l2 import load_anchor
            flags=[(p,p.requires_grad) for p in self.actor.actor_module.parameters()]
            # Keep frozen coordinates in P2's original scalar definition.
            try:
                for p,_ in flags:p.requires_grad_(True)
                anchor=load_anchor(self.actor.actor_module,n['base_model'])
            finally:
                for p,flag in flags:p.requires_grad_(flag)
            weights=None
            if n['family']=='fisher':
                ready(Path(n['fisher_dir']).is_dir(),'Fisher artifact absent')
                need(sha(Path(n['fisher_dir'])/'manifest.json')==n['fisher_manifest_sha256'],'CPU Fisher manifest changed')
                manifest=load(Path(n['fisher_dir'])/'manifest.json');need(manifest['base_revision']==BASE,'wrong Fisher base')
                path=Path(n['fisher_dir'])/f'rank-{self.rank}.pt'
                need([path.stat().st_size,path.stat().st_mtime_ns]==manifest['metadata'][path.name],'Fisher shard changed after CPU hashing')
                if n['arm']=='T' or n.get('measurement'):weights=torch.load(path,map_location='cpu',weights_only=True)
            names={id(p):name for name,p in self.actor.actor_module.named_parameters()}
            segments=[]
            for p,start,base in anchor.segments:
                weight=torch.ones_like(base) if weights is None else weights[names[id(p)]].reshape(-1)[start:start+base.numel()]
                segments.append((p,start,base,weight))
            nm_actor.install_penalty(self.actor,nm_loss.WeightedAnchor(segments),n['penalty'],n.get('measurement',False))
        self._nm_steps=dict(optimizer=0,scheduler=0,ema=0)
        for owner,method,key in [(self.actor_optimizer,'step','optimizer'),(self.actor_lr_scheduler,'step','scheduler'),(self.actor,'_update_teacher','ema')]:
            original=getattr(owner,method)
            def counted(*args,_original=original,_key=key,**kwargs):
                value=_original(*args,**kwargs);self._nm_steps[_key]+=1;return value
            counted._wrapped_by_lr_sched=True;setattr(owner,method,counted)
        return result
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self,local_path,hdfs_path=None,global_step=0,max_ckpt_to_keep=None):
        for p,before in self._nm_frozen:
            need(p.grad is None and torch.equal(nm_loss.local(p),before),'frozen decoder block moved or received gradient')
            need(p not in self.actor_optimizer.state,'frozen tensor has optimizer state')
        for (p,_),(student,_) in zip(self._nm_frozen_teacher,self._nm_frozen):
            need(nm_loss.frozen_ema_equal(p,student,getattr(self.actor,'_v4_completed_updates',0)),'EMA bottom decoder block exceeds known rounding')
        result=super().save_checkpoint(local_path,hdfs_path,global_step,max_ckpt_to_keep)
        if self.rank==0:
            atomic(Path(local_path)/'nm-state.json',dict(steps=self._nm_steps,updates=getattr(self.actor,'_v4_completed_updates',0),
                config_sha256=digest(self.actor.nm_config),mask=self._nm_mask,history=getattr(self.actor,'_nm_history',[])))
        torch.distributed.barrier();return result
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self,local_path,hdfs_path=None,del_local_after_load=False):
        state=load(Path(local_path)/'nm-state.json') if local_path else None
        if state:
            need(state['config_sha256']==digest(self.actor.nm_config) and state['mask']==self._nm_mask,'native resume NM mask/coefficient/config differs')
        result=super().load_checkpoint(local_path,hdfs_path,del_local_after_load)
        if state:
            self._nm_steps=state['steps'];self.actor._v4_completed_updates=state['updates'];self.actor._nm_history=state['history']
        return result
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def nm_receipt(self):
        n=self.actor.nm_config;updates=getattr(self.actor,'_v4_completed_updates',0)
        need(updates==n['updates'],'native update count differs')
        need(self._nm_steps['optimizer']==updates==self._nm_steps['scheduler'],'native optimizer/scheduler counts differ')
        need(self._nm_steps['ema']==(0 if n['family']=='grpo' and n['arm']=='T' else updates),'native EMA count differs')
        return dict(status='COMPLETE',updates=updates,steps=self._nm_steps,history=getattr(self.actor,'_nm_history',[]),
            seed=n['seed'],arm=n['arm'],mask=self._nm_mask,config_sha256=digest(n),technical_synthetic=False)


class NMTaskRunner(main_ppo.TaskRunner):
    def add_actor_rollout_worker(self,config):
        actor,group=super().add_actor_rollout_worker(config)
        self.role_worker_mapping[Role.ActorRolloutRef]=ray.remote(NMWorker);return NMWorker,group
    def run(self,config):
        check_pinned();n=check_config(config)
        with open_dict(config.actor_rollout_ref):config.actor_rollout_ref.nm=config.nm
        previous=main_ppo.RayPPOTrainer
        class ReceiptTrainer(NMTrainer):
            def fit(self):
                result=super().fit();receipts=self.actor_rollout_wg.nm_receipt()
                need(all(r['config_sha256']==receipts[0]['config_sha256'] for r in receipts),'rank receipt mismatch')
                atomic(n['receipt'],dict(receipts[0],rank_receipts=receipts));return result
        main_ppo.RayPPOTrainer=ReceiptTrainer
        try:return super().run(config)
        finally:main_ppo.RayPPOTrainer=previous
