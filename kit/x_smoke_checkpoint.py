"""Development-only checkpoint memory accounting and eager disposable cleanup."""
from pathlib import Path
import functools,json,os,shutil
from kit.s_common import atomic,read,require


def memory_current(cgroup=Path('/sys/fs/cgroup')):
    root=Path(cgroup)
    for name in ('memory.current','memory/memory.usage_in_bytes'):
        try:
            value=dict(bytes=int((root/name).read_text()),source=str(root/name),available=True)
            try:value['file_bytes']=dict(line.split() for line in (root/'memory.stat').read_text().splitlines()).get('file')
            except OSError:value['file_bytes']=None
            if value['file_bytes'] is not None:value['file_bytes']=int(value['file_bytes'])
            return value
        except (OSError,ValueError):pass
    return dict(bytes=None,file_bytes=None,source='cgroup memory.current unavailable',available=False)


def checkpoint_files(path,width,model_only=False):
    path=Path(path);names=['data.pt']
    for rank in range(width):
        names.extend(f'actor/{kind}_world_size_{width}_rank_{rank}.pt' for kind in (('model',) if model_only else ('model','optim','extra_state')))
        if not model_only:names.append(f'actor/v4-ema/model_world_size_{width}_rank_{rank}.pt')
    require(all((path/n).is_file() and (path/n).stat().st_size>0 for n in names),'development checkpoint lacks complete state')
    return {n:(path/n).stat().st_size for n in names}


def dispose(path):
    path=Path(path)
    if path.exists():shutil.rmtree(path)
    return not path.exists()


def install(cls):
    if not os.environ.get('X_INTEGRATION_SMOKE') or getattr(cls,'_x_integration_checkpoints',False):return
    require(os.environ['X_INTEGRATION_SMOKE']=='development-only' and os.environ.get('V4_PROFILE')=='technical-smoke',
        'checkpoint cleanup is development-only')
    save,load=cls._save_checkpoint,cls._load_checkpoint
    @functools.wraps(save)
    def saved(self,*args,**kwargs):
        from kit.x_smoke_memory import sample
        try:host_before=sample()
        except Exception:host_before=None  # telemetry must never prevent the save
        before=memory_current();step=int(self.global_steps)
        root=Path(self.config.trainer.default_local_dir);checkpoint=root/f'global_step_{step}'
        print('[record-partial] '+json.dumps(dict(stage='checkpoint-before-save',step=step,memory_current=before)),flush=True)
        try:return save(self,*args,**kwargs)
        finally:
            after=memory_current()
            try:host_after=sample()
            except Exception:host_after=None
            record=dict(schema='x-smoke-checkpoint-memory.v1',step=step,before=before,after=after,
                host_before=host_before,host_after=host_after)
            try:
                model_only=os.environ.get('X_SMOKE_PREPARATION_MODEL_ONLY')=='1'
                require(not model_only or os.environ.get('X_P2_SMOKE_ROLE')=='prepare','model-only checkpoint is preparation-only')
                files=checkpoint_files(checkpoint,int(self.config.trainer.n_gpus_per_node)*int(self.config.trainer.nnodes),model_only)
                size=sum(files.values())
                delta=after['bytes']-before['bytes'] if before['available'] and after['available'] else None
                file_delta=(after['file_bytes']-before['file_bytes'] if before.get('file_bytes') is not None and after.get('file_bytes') is not None else None)
                counted=(max(v for v in (delta,file_delta) if v is not None)>.25*size) if delta is not None or file_delta is not None else None
                # The drift prefix is indispensable until live restore qualifies.
                retain=bool(os.environ.get('M2_ADDON_BOUNDARY') and step==2) or bool(os.environ.get('X_P2_SMOKE_ROLE') and step==4)
                record.update(status='PASS',files=files,checkpoint_bytes=size,memory_delta_bytes=delta,
                    file_delta_bytes=file_delta,writes_count_towards_memory=counted,
                    inference='observed cgroup growth >25% of checkpoint bytes; conservative signal, not filesystem attribution',
                    deleted=dispose(checkpoint) if not retain else False,
                    retention_reason=('required until fp32 update-4 export' if step==4 else 'required until qualified update-2 restore') if retain else 'disposable checkpoint checked then deleted eagerly',
                    after_cleanup=memory_current())
            except (ValueError,OSError) as error:record.update(status='FAIL',error=str(error))
            atomic(root.parent/f'checkpoint-memory-step{step}.json',record)
            print('[record-partial] '+json.dumps(dict(stage='checkpoint-after-save',checkpoint=record)),flush=True)
    @functools.wraps(load)
    def loaded(self,*args,**kwargs):
        result=load(self,*args,**kwargs)
        if self.config.trainer.resume_mode=='resume_path':
            root=Path(self.config.trainer.default_local_dir)
            receipt=read(root.parent/'restore-qualified.json')
            require(receipt['status']=='PASS' and receipt['before_update']==3,'cannot remove unqualified drift prefix')
            record=dict(stage='checkpoint-after-qualified-load',step=2,
                deleted=dispose(root/'global_step_2'),memory_current=memory_current())
            atomic(root.parent/'checkpoint-loaded-cleanup.json',record)
            print('[record-partial] '+json.dumps(record),flush=True)
        return result
    cls._save_checkpoint,cls._load_checkpoint=saved,loaded;cls._x_integration_checkpoints=True
