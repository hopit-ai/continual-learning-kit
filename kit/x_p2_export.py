"""Actual HF export and fp64 checkpoint-distance checks, bounded CPU slices."""
from pathlib import Path
import json,math,subprocess,shutil
from contextlib import ExitStack
from kit.s_common import require


def merge(run,target,env):
    # Native verl merger explicitly casts to bf16. Preserve the actual saved
    # fp32 masters here; changing the reviewed trainer/exporter is unnecessary.
    import sys
    subprocess.run([sys.executable,'-m','kit.x_p2_export',str(run),str(target)],cwd='/root',env=env,check=True)
    return Path(target)


def assemble(parts,shape):
    import torch
    require(all(v.dtype==torch.float32 for v in parts),'checkpoint lacks fp32 master precision')
    value=torch.cat(parts,dim=0)
    require(list(value.shape)==list(shape),'checkpoint shard/padding geometry differs')
    return value.contiguous()


def export(run,target,source=None):
    import torch
    from torch.distributed.tensor import DTensor
    from safetensors.torch import save_file
    from kit.s_common import read,atomic
    run,target=Path(run),Path(target);source=Path(source) if source is not None else run/'sdft/global_step_4/actor'
    record=read(run/'smoke-rank0.json')
    require(record['world_size']==2,'registered export requires two ranks')
    require(not target.exists(),'export target already exists')
    target.mkdir()
    # mmap bounds resident CPU state; only one concatenated parameter at a time.
    states=[torch.load(source/f'model_world_size_2_rank_{r}.pt',map_location='cpu',weights_only=False,mmap=True) for r in range(2)]
    require(set(states[0])==set(states[1]),'checkpoint rank keys differ')
    parameters={n:row for row in record['geometry']['inventory'] for n in row['aliases']}
    require(set(parameters)<=set(states[0]),'checkpoint lacks registered parameter aliases')
    mapping={};total=0
    for i,name in enumerate(sorted(states[0])):
        values=[s.pop(name) for s in states]
        require(all(isinstance(v,DTensor) and len(v.placements)==1 and type(v.placements[0]).__name__=='Shard' and v.placements[0].dim==0 for v in values),'unsupported checkpoint layout; refuse precision amendment')
        shape=list(values[0].shape)
        require(all(list(v.shape)==shape for v in values),'checkpoint logical shapes differ')
        merged=assemble([v.to_local() for v in values],shape)
        if name in parameters:require(shape==parameters[name]['shape'],'export parameter shape changed')
        filename=f'model-{i:05d}.safetensors'
        save_file({name:merged},str(target/filename),metadata={'format':'pt'})
        mapping[name]=filename;total+=merged.numel()*merged.element_size()
        del merged,values
    for path in (source/'huggingface').iterdir():
        if path.is_file() and not path.name.endswith(('.safetensors','.bin')) and not path.name.endswith('.index.json'):
            shutil.copyfile(path,target/path.name)
    require((target/'config.json').exists(),'native checkpoint config missing')
    config=read(target/'config.json');config['torch_dtype']='float32';atomic(target/'config.json',config)
    atomic(target/'model.safetensors.index.json',dict(metadata=dict(total_size=total),weight_map=mapping))
    return target


def live_student(worker):
    """Export only the completed fp32 student, before native optimizer/EMA save.

    This uses the same real sharded state context as native restore qualification;
    the unchanged merger/distance check subsequently reads these saved DTensors.
    It never serializes or changes optimizer, scheduler, RNG or teacher state.
    """
    import os,torch
    from kit.p2_l2 import local
    from kit.s_common import atomic
    require(os.environ.get('X_INTEGRATION_SMOKE')=='development-only' and
        os.environ.get('X_P2_EARLY_VERDICT')=='1' and os.environ.get('X_P2_SMOKE_ROLE')=='p2','live export is smoke P2 only')
    require(worker.actor._v4_completed_updates==4,'live export requires all four completed updates')
    manager=worker.checkpoint_manager
    require(all(local(p).dtype==torch.float32 for p in manager.model.parameters()),'live export lacks fp32 masters')
    context=getattr(manager,'_x_state_context',None)
    if context is None:
        from verl.utils.checkpoint import fsdp_checkpoint_manager as module
        def context(model):
            return module.get_fsdp_state_ctx(model,module.StateDictType.SHARDED_STATE_DICT,
                module.ShardedStateDictConfig(offload_to_cpu=True),module.ShardedOptimStateDictConfig(offload_to_cpu=True))
    source=Path(os.environ['X_P2_RUN'])/'live-step4/actor';source.mkdir(parents=True,exist_ok=True)
    target=source/f'model_world_size_2_rank_{worker.rank}.pt'
    require(not target.exists(),'live final student already exported')
    with context(manager.model):
        state=manager.model.state_dict()
        require(all(local(v).dtype==torch.float32 for v in state.values()),'live exported state is not fp32')
        temporary=target.with_suffix('.tmp');torch.save(state,temporary);temporary.replace(target)
    if worker.rank==0:
        origin=Path(worker.config.model.path);metadata=source/'huggingface';metadata.mkdir()
        for path in origin.iterdir():
            if path.is_file() and path.suffix not in ('.safetensors','.bin') and not path.name.endswith('.index.json'):
                shutil.copyfile(path,metadata/path.name)
        require((metadata/'config.json').is_file(),'live export config missing')
    atomic(source/f'ready-r{worker.rank}.json',dict(rank=worker.rank,completed_updates=4,precision='fp32',student_only=True))
    return source


def distance(checkpoint,base,inventory):
    import torch
    from safetensors import safe_open
    with ExitStack() as stack:
        def handles(path):
            files=[stack.enter_context(safe_open(str(p),framework='pt',device='cpu')) for p in sorted(Path(path).glob('*.safetensors'))]
            return {n:f for f in files for n in f.keys()}
        left,right=handles(checkpoint),handles(base);total=0.
        for row in inventory:
            name=next((n for n in row['aliases'] if n in left and n in right),None)
            require(name is not None,'export missing tied/unique parameter')
            a,b=left[name].get_slice(name),right[name].get_slice(name);shape=list(a.get_shape())
            require(shape==row['shape']==list(b.get_shape()),'export parameter geometry changed')
            # Read bounded rows, then retain only the requested flat fragment.
            width=math.prod(shape[1:]) if len(shape)>1 else 1
            for start in range(0,math.prod(shape),65536):
                count=min(65536,math.prod(shape)-start);lo=start//width;hi=(start+count-1)//width+1
                aa=a[lo:hi].reshape(-1)[start-lo*width:start-lo*width+count]
                bb=b[lo:hi].reshape(-1)[start-lo*width:start-lo*width+count]
                require(aa.dtype==torch.float32,'export is not registered fp32 master precision; prospective amendment required')
                total+=float((aa.double()-bb.float().double()).square().sum())
    return dict(distance2=total,dtype='torch.float32',checkpoint=str(checkpoint),parameters=len(inventory))


if __name__=='__main__':
    import sys
    export(sys.argv[1],sys.argv[2])
