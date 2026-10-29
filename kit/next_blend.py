#!/usr/bin/env python3
"""One final × lambda: materialise on local NVMe, score serially, delete in finally."""
from pathlib import Path
import sys
if __package__ in (None, ''): sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kit.next_common import *
import hashlib
import shutil
import tempfile
import fcntl


def tensor_hash(value):
    import torch
    return hashlib.sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def interpolate(start, final, lam):
    import torch
    require(start.shape==final.shape and start.dtype==final.dtype, 'tensor shape/dtype mismatch')
    if not start.is_floating_point():
        require(torch.equal(start,final), 'nonfloating buffer differs')
        return start.clone()
    # bf16/fp16 operands are promoted BEFORE arithmetic. Endpoints avoid rounding drift.
    if lam==0: return start.clone()
    if lam==1: return final.clone()
    return ((1-lam)*start.float()+lam*final.float()).to(start.dtype)


def blend(start, final, dest, lam):
    from safetensors import safe_open
    from safetensors.torch import save_file
    def index(root):
        result={}
        for path in sorted(root.glob('*.safetensors')):
            with safe_open(path, framework='pt',device='cpu') as f:
                for key in f.keys():
                    require(key not in result, 'duplicate tensor key'); result[key]=path
        require(bool(result), 'no safetensors weights')
        return result
    left,right=index(start),index(final)
    require(left.keys()==right.keys(),'tensor keys differ')
    tokenizer_a,tokenizer_b=tokenizer_receipt(start,final)
    require(bool(tokenizer_a) and tokenizer_a==tokenizer_b,'tokenizer differs')
    dest.mkdir(exist_ok=False); receipt={}; weight_map={}; total=0
    # Keep at most one export shard in memory. Tensor keys may live in different parent shards.
    for number, source in enumerate(sorted(set(left.values()))):
        values={}
        with safe_open(source,framework='pt',device='cpu') as a:
            for key in sorted(k for k in left if left[k]==source):
                with safe_open(right[key],framework='pt',device='cpu') as b:
                    x,y=a.get_tensor(key),b.get_tensor(key)
                    value=interpolate(x,y,lam); values[key]=value
                    receipt[key]=dict(start=tensor_hash(x),final=tensor_hash(y),blend=tensor_hash(value),dtype=str(x.dtype),shape=list(x.shape))
                    total+=value.numel()*value.element_size()
        name=f'model-{number+1:05d}-of-{len(set(left.values())):05d}.safetensors'
        save_file(values,str(dest/name),metadata={'format':'pt'}); weight_map.update({k:name for k in values})
    for p in final.iterdir():
        if p.is_file() and p.suffix in ('.json','.jinja','.txt') and not p.name.endswith('.safetensors.index.json'):
            shutil.copyfile(p,dest/p.name)
    atomic(dest/'model.safetensors.index.json',dict(metadata=dict(total_size=total),weight_map=weight_map))
    return receipt


def tokenizer_receipt(start, final):
    # Only encoding assets define weight/token compatibility. Export metadata is disclosed.
    def core(root):
        return {p.name:s.file_hash(p) for p in root.iterdir() if p.is_file() and
                (p.name in ('tokenizer.json','vocab.json') or p.name.startswith('merges'))}
    return core(start),core(final)


def tokenizer_differences(start, final):
    def metadata(root):
        return {p.name:s.file_hash(p) for p in root.iterdir() if p.is_file() and
                (p.name.startswith(('tokenizer','special_tokens','added_tokens','chat_template')) or p.suffix=='.jinja')}
    left,right=metadata(start),metadata(final)
    return {name:dict(start=left.get(name),final=right.get(name)) for name in sorted(left.keys()|right.keys())
            if left.get(name)!=right.get(name)}


def recreate(a, folder, slot):
    require(not a.standin,'recreation requires real weights')
    require(a.lam not in (0,1),'endpoints use retained parent weights; only interior blends need recreation')
    frozen_files(folder)
    recorded=json.loads((folder/'tensor-hashes.json').read_text())
    require(recorded['parents']==slot and recorded['lam']==a.lam,'recreation inputs differ')
    require(not a.recreate.exists(),'recreation destination already exists')
    local_nvme(a.recreate.parent,False)
    try:
        tensors=blend(Path(slot['start']['path']),Path(slot['final']['path']),a.recreate,a.lam)
        require(tensors==recorded['tensors'],'recreated tensor hashes differ')
        require(tokenizer_differences(Path(slot['start']['path']),Path(slot['final']['path']))==recorded['tokenizer_metadata_differences'],
                'recreated tokenizer metadata differs')
    except BaseException:
        if a.recreate.exists():shutil.rmtree(a.recreate)
        raise
    print('RECREATED and verified',a.recreate)


def local_nvme(root, standin, *, sysfs=Path('/sys')):
    """Resolve the filesystem device through sysfs, including bind mounts and md RAID.

    st_dev describes the mounted filesystem even when findmnt prints /dev/md0[/tmp]
    or the container has no corresponding /dev node. All backing leaves must be NVMe.
    """
    require(root.is_dir(), 'NVMe root does not exist')
    if standin: return
    device = root.stat().st_dev
    node = sysfs/'dev/block'/f'{os.major(device)}:{os.minor(device)}'
    require(node.exists(), 'blend root must be local block storage (sysfs device missing)')
    seen = set();leaves=[]
    def nvme(path):
        path = path.resolve()
        require(path not in seen, 'cyclic sysfs block ancestry')
        seen.add(path)
        slaves = sorted((path/'slaves').iterdir()) if (path/'slaves').is_dir() else []
        try:
            if slaves:return all(nvme(child) for child in slaves)
            if (path/'partition').exists():return nvme(path.parent)
            transport=next(((parent/'transport').read_text().strip() for parent in (path/'device',path,*path.parents)
                if parent.is_relative_to(sysfs) and (parent/'transport').is_file()),None)
            transports=[transport] if transport is not None else []
            if not transports:
                for controller in (path/'device',path/'device/subsys',path/'device/subsystem'):
                    transports.extend(p.read_text().strip() for p in controller.glob('nvme*/transport'))
            leaves.append(dict(device=path.name,transports=sorted(transports)))
            return path.name.startswith('nvme') and (path/'dev').is_file() and bool(transports) and all(t=='pcie' for t in transports)
        finally:seen.remove(path)
    require(nvme(node), 'blend root must be node-local NVMe throughout its sysfs ancestry')
    return dict(filesystem_device=f'{os.major(device)}:{os.minor(device)}',leaves=leaves,all_transports_pcie=True)


def job(a):
    frozen_next(); local_nvme(a.nvme_root,a.standin)
    inputs=read_inputs(a.out,a.standin)
    slots=[r for r in inputs['finals'] if r['name']==a.slot]
    require(len(slots)==1,'unbound or missing screen slot'); slot=slots[0]
    for key in ('start','final'):
        parent=slot[key]
        observed=checkpoint(parent['path'],parent['receipt'],a.standin,base=key=='start' and slot['route']!='sequence')
        require(observed['model_file_hashes']==parent['identity']['model_file_hashes'],'parent identity changed')
    arm=f"v3-{slot['route']}-{slot['recipe']}-l{a.lam:g}"
    folder=a.out/'blends'/f'{a.slot}-l{a.lam:g}'
    bind(folder/'inputs.json',dict(slot=slot,lam=a.lam,technical_synthetic=a.standin))
    if getattr(a,'recreate',None) is not None:
        recreate(a,folder,slot);return
    machine(a.out,a.gpu,a.standin)
    if (folder/'manifest.json').exists(): frozen_files(folder); print('SKIP blend',folder.name); return
    if a.lam in (0,1):
        parent=slot['start' if a.lam==0 else 'final']
        panels=['finqa','general-v1']+(['math_screen'] if slot['route']!='finqa' else [])
        for panel in panels:
            shared_score(a,parent,arm,slot['seed'],40,panel,run)
        atomic(folder/'endpoint.json',dict(parent=parent,lam=a.lam,weights_materialised=False))
        freeze(folder,['inputs.json','endpoint.json'],a.standin)
        return
    key=t.sha(str(a.out.resolve())+'|'+a.slot+'|'+str(a.lam))[:24]
    scratch=a.nvme_root/('next-blend-'+key)
    lock=(a.nvme_root/('.next-blend-'+key+'.lock')).open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close();raise CorrectnessStop('duplicate blend job is already running')
    # A previous killed process may have left weights. The lock proves no live owner remains.
    if scratch.exists():shutil.rmtree(scratch)
    scratch.mkdir();model=scratch/'model'
    try:
        if a.standin:
            model.mkdir(); receipt={'fixture':dict(start='fixture',final='fixture',blend='fixture',dtype='bf16')}
            shutil.copyfile(Path(slot['final']['path'])/'config.json', model/'config.json')
            exported=dict(technical_synthetic=True, model_file_hashes={'config.json':s.file_hash(model/'config.json')})
        else:
            receipt=blend(Path(slot['start']['path']),Path(slot['final']['path']),model,a.lam)
            exported=identity(model)
        atomic(folder/'tensor-hashes.json',dict(parents=slot,lam=a.lam,tensors=receipt,
            tokenizer_metadata_differences={} if a.standin else tokenizer_differences(Path(slot['start']['path']),Path(slot['final']['path'])),
            recreate=[sys.executable,str(KIT/'next_blend.py'),'--out',str(a.out),'--slot',a.slot,'--lam',str(a.lam),'--nvme-root',str(a.nvme_root),'--gpu',a.gpu,'--recreate',str(a.nvme_root/(folder.name+'-recreated'))]+
                (['--package-a',str(a.package_a)] if a.package_a else [])+(['--finqa',str(a.finqa)] if a.finqa else [])+(['--standin'] if a.standin else []),technical_synthetic=a.standin))
        export_receipt=scratch/'export.json'; atomic(export_receipt,exported)
        panels=['finqa','general-v1']+(['math_screen'] if slot['route']!='finqa' else [])
        for panel in panels:
            command=[sys.executable,KIT/'p3b_score.py','--out',a.out,'--model',model,'--checkpoint-receipt',export_receipt,
                '--arm',arm,'--seed',slot['seed'],'--step',40,'--panel-set',panel,'--gpu',a.gpu]
            if a.package_a: command+=['--package-a',a.package_a]
            if a.finqa: command+=['--panel-root',a.finqa]
            if a.standin: command+=['--standin']
            run(command)
        freeze(folder,['inputs.json','tensor-hashes.json'],a.standin)
    finally:
        shutil.rmtree(scratch)
        lock.close()
        atomic(folder/'storage.json',dict(deleted=True,weights_retained=False,technical_synthetic=a.standin))


def configure(p):
    p.add_argument('--recreate',type=Path,help='rebuild into a new local-NVMe destination and verify the frozen tensor hashes; no scoring')
    p.add_argument('--slot',required=True);p.add_argument('--lam',type=float,choices=(0,.25,.5,.75,1),required=True)
    p.add_argument('--nvme-root',type=Path,required=True);p.add_argument('--gpu',default='0')
    p.add_argument('--package-a',type=Path);p.add_argument('--finqa',type=Path)
if __name__ == '__main__': raise SystemExit(entry(job,configure,'next-blend-{slot}-{lam}'))
