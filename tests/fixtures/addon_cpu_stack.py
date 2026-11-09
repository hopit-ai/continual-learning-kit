"""CPU transport stand-in around the REAL X runner, entry and boundary hooks.

Tiny torch weights/Adam/scheduler/EMA/RNG and loader state are really saved and
loaded. Ray and FSDP transport alone are replaced; this is not GPU qualification.
"""
from pathlib import Path
from types import SimpleNamespace as NS, ModuleType
from contextlib import nullcontext
from unittest.mock import patch
import copy
import importlib
import os
import random
import sys


def exercise(root, corrupt=None, boundary=10, chain=False):
    import torch
    from kit import x_resume, x_sdft_entry
    from kit.s_common import read
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    first_prefix=root/'boundaries/k10/prefix.json' if chain else root/'prefix.json'
    captured={};current={}
    class Attr(dict):
        __getattr__=dict.__getitem__
    class Loader:
        def __init__(self):self.position=0
        def state_dict(self):return dict(_num_yielded=self.position)
        def load_state_dict(self,state):self.position=state['_num_yielded']
    class Worker:
        def __init__(self):
            torch.manual_seed(101);random.seed(101)
            self.rank=0;self.model=torch.nn.Linear(2,1).float()
            self.opt=torch.optim.AdamW(self.model.parameters(),lr=1e-5)
            self.scheduler=torch.optim.lr_scheduler.LambdaLR(self.opt,lambda n:min(n/10,1))
            self.actor=NS(teacher_module=copy.deepcopy(self.model),_v4_completed_updates=0)
            self.checkpoint_manager=NS(model=self.model,optimizer=self.opt,lr_scheduler=self.scheduler,
                get_rng_state=lambda:dict(torch=torch.get_rng_state(),python=random.getstate()),
                _x_state_context=lambda model:nullcontext())
        def save_checkpoint(self,local_path):
            target=Path(local_path);target.mkdir(parents=True,exist_ok=True)
            torch.save(self.model.state_dict(),target/'model_world_size_1_rank_0.pt')
            torch.save(self.opt.state_dict(),target/'optim_world_size_1_rank_0.pt')
            torch.save(dict(rng=self.checkpoint_manager.get_rng_state(),scheduler=self.scheduler.state_dict()),target/'extra_state_world_size_1_rank_0.pt')
            (target/'v4-ema').mkdir(exist_ok=True)
            torch.save(self.actor.teacher_module.state_dict(),target/'v4-ema/model_world_size_1_rank_0.pt')
        def load_checkpoint(self,local_path):
            target=Path(local_path)
            self.model.load_state_dict(torch.load(target/'model_world_size_1_rank_0.pt',weights_only=False))
            self.opt.load_state_dict(torch.load(target/'optim_world_size_1_rank_0.pt',weights_only=False))
            extra=torch.load(target/'extra_state_world_size_1_rank_0.pt',weights_only=False)
            self.scheduler.load_state_dict(extra['scheduler'])
            self.actor.teacher_module.load_state_dict(torch.load(target/'v4-ema/model_world_size_1_rank_0.pt',weights_only=False))
            torch.set_rng_state(extra['rng']['torch']);random.setstate(extra['rng']['python'])
            restored_step=int(target.parent.name.removeprefix('global_step_'))
            self.actor._v4_completed_updates=restored_step
            damage=corrupt if not chain or restored_step==20 else None
            if damage == 'model':
                with torch.no_grad():next(self.model.parameters()).add_(1)
            if damage == 'EMA':
                with torch.no_grad():next(self.actor.teacher_module.parameters()).add_(1)
            if damage == 'rng':torch.rand(1)
            if damage == 'optimizer': self.opt.param_groups[0]['lr']+=1
            if damage == 'counter': self.actor._v4_completed_updates=restored_step-1
            if damage == 'scheduler': self.scheduler.last_epoch=restored_step-1
    class Trainer:
        def __init__(self,config,worker):
            self.config=config;self.worker=worker;self.global_steps=0;self.total_training_steps=40;self.train_dataloader=Loader()
        def _save_checkpoint(self):
            path=root/'sdft'/f'global_step_{self.global_steps}'
            self.worker.save_checkpoint(str(path/'actor'))
            torch.save(self.train_dataloader.state_dict(),path/'data.pt')
        def _load_checkpoint(self):
            if self.config.trainer.resume_mode=='disable':return
            path=Path(self.config.trainer.resume_from_path)
            self.worker.load_checkpoint(str(path/'actor'))
            self.train_dataloader.load_state_dict(torch.load(path/'data.pt',weights_only=False))
            self.global_steps=int(path.name.removeprefix('global_step_'))
            damage=corrupt if not chain or self.global_steps==20 else None
            if damage == 'loader':self.train_dataloader.position=self.global_steps-1
            if damage == 'step':self.global_steps-=1
        def fit(self):
            self._load_checkpoint()
            for step in range(self.global_steps+1,current['end']+1):
                self.global_steps=step;self.train_dataloader.position+=1
                batch=torch.randn(32,2)
                # Replay allocation changes after ten; task positions stay fixed.
                target=batch[:,:1]*current['slots'][step-1]+random.random()
                self.worker.opt.zero_grad();loss=(self.worker.model(batch)-target).square().mean()
                loss.backward();self.worker.opt.step();self.worker.scheduler.step()
                with torch.no_grad():
                    for teacher,student in zip(self.worker.actor.teacher_module.parameters(),self.worker.model.parameters()):
                        teacher.mul_(.99).add_(student,alpha=.01)
                self.worker.actor._v4_completed_updates=step
                captured.setdefault('updates',[]).append(step)
                if step==x_resume.boundary_step():self._save_checkpoint()
            captured['last']=self
    class ParentRunner:
        def __init__(self):self.role_worker_mapping={}
        def add_actor_rollout_worker(self,config):return Worker,object
        def run(self,config):
            actor,_=self.add_actor_rollout_worker(config)
            trainer=Trainer(config,actor());captured['trainer']=trainer
            trainer.fit()
    fake=ModuleType('kit.sdft_verl')
    fake.SdftTaskRunner=ParentRunner;fake.SdftActorRolloutRefWorker=Worker;fake.AsyncActorRolloutRefWorker=Worker
    fake.ray_trainer=NS(RayPPOTrainer=Trainer,Role=NS(ActorRolloutRef='actor'))
    fake.ray=NS(remote=lambda cls:cls);fake.check_pinned=lambda:captured.update(pin_checked=True)
    fake.main_ppo=NS(main=lambda:fake.main_ppo.TaskRunner().run(current['config']))
    # Import the actual C adapter and X runner; replace only the unavailable
    # external GPU transport. Module cleanup prevents contamination of other tests.
    names=('kit.sdft_verl','kit.c_sdft_runner','kit.x_sdft_runner','kit.x_worker')
    prior={name:sys.modules.get(name) for name in names}
    import kit
    attrs={name.rsplit('.',1)[-1]:getattr(kit,name.rsplit('.',1)[-1],None) for name in names}
    omega_prior=sys.modules.get('omegaconf')
    try:
        for name in names:sys.modules.pop(name,None)
        sys.modules['kit.sdft_verl']=fake
        sys.modules['omegaconf']=NS(OmegaConf=NS(to_container=lambda value,resolve:value))
        with patch.object(kit,'sdft_verl',fake,create=True), patch.dict(os.environ,
            M2_ADDON_BOUNDARY=str(first_prefix),M2_ADDON_PREFIX='1',
            X_TIER2_BOUNDARY_STEP=str(boundary),X_TIER2_RESUME_STEP=str(boundary)):
            runner=importlib.import_module('kit.x_sdft_runner')
            assert runner.AddonTaskRunner.__module__=='kit.x_sdft_runner'
            def config(mode,step=boundary):return Attr(trainer=Attr(resume_mode=mode,resume_from_path=str(root/'sdft'/f'global_step_{step}'),
                default_local_dir=str(root/'sdft'),n_gpus_per_node=1,nnodes=1))
            current.update(config=config('disable'),end=40,slots=[.5]*40)
            x_sdft_entry.main()
            prefix=read(first_prefix)
            assert captured['pin_checked'] and prefix['ranks']['0']['completed_updates']==boundary
            before=copy.deepcopy(prefix)
            restores=[]
            previous=first_prefix
            for start,end in (((10,20),(20,30),(30,40)) if chain else ((boundary,40),)):
                next_boundary=root/'boundaries'/f'k{end if end<40 else start}'/'prefix.json' if chain else previous
                os.environ.update(M2_ADDON_BOUNDARY=str(next_boundary),X_TIER2_BOUNDARY_STEP=str(end if end<40 else start),
                    X_TIER2_RESUME_BOUNDARY=str(previous),X_TIER2_RESUME_STEP=str(start))
                if end<40:os.environ['M2_ADDON_PREFIX']='1'
                else:os.environ.pop('M2_ADDON_PREFIX',None)
                current.update(config=config('resume_path',start),slots=([.5]*10+[2.]*10+[3.]*10+[4.]*10 if chain else [.5]*boundary+[2.]*(40-boundary)))
                assert x_resume.checked_resume(current['config'])==str(root/'sdft'/f'global_step_{start}')
                if corrupt=='file' and (not chain or start==20):
                    (root/'sdft'/f'global_step_{start}'/'data.pt').write_bytes(b'changed')
                x_sdft_entry.main()
                qualified=read(previous.parent/'restore-qualified.json')
                assert qualified['ranks']==before['ranks'] and qualified['dataloader']==before['dataloader']
                assert qualified['before_update']==start+1
                restores.append(start)
                if end<40:
                    previous=next_boundary;before=copy.deepcopy(read(previous))
            assert captured['last'].global_steps==40 and captured['last'].train_dataloader.position==40
            # Compare against an uninterrupted CPU run with the same new suffix.
            final=copy.deepcopy(captured['last'].worker.model.state_dict())
            final_ema=copy.deepcopy(captured['last'].worker.actor.teacher_module.state_dict())
            captured['chain_updates']=list(captured['updates'])
            os.environ.pop('M2_ADDON_BOUNDARY');current['config']=config('disable')
            reference=Trainer(current['config'],Worker())
            reference._save_checkpoint=lambda:None
            reference.fit()
            assert all(torch.equal(v,final[k]) for k,v in reference.worker.model.state_dict().items())
            assert all(torch.equal(v,final_ema[k]) for k,v in reference.worker.actor.teacher_module.state_dict().items())
            return dict(status='PASS',components=list(before['ranks']['0']['components'])+['dataloader'],
                before_update=qualified['before_update'],completed_updates=40,restore_steps=restores,real_cpu_tensor_save_restore=True)
    finally:
        if chain:
            from kit.s_common import atomic
            atomic(root/'executed-updates.json',captured.get('chain_updates',captured.get('updates',[])))
        if omega_prior is None:sys.modules.pop('omegaconf',None)
        else:sys.modules['omegaconf']=omega_prior
        for name,old in prior.items():
            if old is None:sys.modules.pop(name,None)
            else:sys.modules[name]=old
        for name,old in attrs.items():
            if old is None:
                if hasattr(kit,name):delattr(kit,name)
            else:setattr(kit,name,old)
