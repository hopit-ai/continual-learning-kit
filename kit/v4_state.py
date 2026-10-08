"""Save and restore v4 counters and EMA alongside the pinned full trainer state."""
from __future__ import annotations
import functools
import os
from pathlib import Path


def install_worker(cls):
    if os.environ.get('V4_CHECKPOINT_STATE')!='1' or getattr(cls,'_v4_state_wrapped',False):return
    from kit.v4_restore_check import install_checkpoint_probe
    install_checkpoint_probe()
    save,load=cls.save_checkpoint,cls.load_checkpoint
    def teacher_manager(worker):
        from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
        teacher=worker.actor.teacher_module
        if teacher is not worker.ref_module_fsdp:raise ValueError('v4 saved state requires the registered EMA teacher')
        return FSDPCheckpointManager(model=teacher,processing_class=worker.tokenizer,
            checkpoint_config={'save_contents':['model'],'load_contents':['model']})
    @functools.wraps(save)
    def saved(self,local_path,*args,**kwargs):
        result=save(self,local_path,*args,**kwargs)
        teacher_manager(self).save_checkpoint(str(Path(local_path)/'v4-ema'),global_step=kwargs.get('global_step',args[1] if len(args)>1 else 0))
        return result
    @functools.wraps(load)
    def loaded(self,local_path,*args,**kwargs):
        result=load(self,local_path,*args,**kwargs)
        if local_path is not None:
            teacher_manager(self).load_checkpoint(str(Path(local_path)/'v4-ema'))
            step=int(Path(local_path).parent.name.removeprefix('global_step_'))
            self.actor._v4_completed_updates=step
        return result
    cls.save_checkpoint=saved;cls.load_checkpoint=loaded;cls._v4_state_wrapped=True


def install_sft(cls):
    if os.environ.get('V4_CHECKPOINT_STATE')!='1' or getattr(cls,'_v4_state_wrapped',False):return
    from kit.v4_restore_check import install_checkpoint_probe,install_trainer_probe
    install_checkpoint_probe()
    install_trainer_probe(cls,save='save_checkpoint',load='load_checkpoint',sft=True)
    original=cls.load_checkpoint
    @functools.wraps(original)
    def loaded(self,*args,**kwargs):
        step=original(self,*args,**kwargs)
        self._v4_updates=int(step or 0)
        return step
    cls.load_checkpoint=loaded;cls._v4_state_wrapped=True
