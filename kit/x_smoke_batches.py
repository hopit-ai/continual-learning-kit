"""Owner-authorised full actor-input pairing for the development P2 smoke only."""
import copy,os
from pathlib import Path
from kit.s_common import atomic,read,require,sha
from kit.v4_restore_check import state_hash


def batch_hash(batch):
    # Object arrays contain prompt IDs and other metadata; hashing their raw
    # bytes would hash process-local pointers rather than their contents.
    def semantic(value):
        if isinstance(value,dict):return {k:semantic(v) for k,v in value.items()}
        if isinstance(value,(list,tuple)):return type(value)(semantic(v) for v in value)
        if getattr(getattr(value,'dtype',None),'hasobject',False):return semantic(value.tolist())
        if value is None or isinstance(value,(str,bytes,int,float,bool)):return value
        if hasattr(value,'dtype') and hasattr(value,'shape'):return value
        # Any other object (e.g. one whose repr carries a memory address) is
        # identified by its type only, so pairing cannot crash falsely on it.
        return '<%s.%s>'%(type(value).__module__,type(value).__qualname__)
    return state_hash(dict(tensors=dict(batch.batch.items()),
        non_tensors=semantic(batch.non_tensor_batch),meta_info=semantic(batch.meta_info)))


def paired_input(batch,mode,update,root,run):
    if os.environ.get('X_INTEGRATION_SMOKE')!='development-only':return batch
    require(os.environ.get('V4_PROFILE')=='technical-smoke','batch pairing is development-only')
    if mode not in ('control','train'):return batch
    role='control' if mode=='control' else 'p2'
    require(os.environ.get('X_P2_SMOKE_ROLE')==role,'paired batch smoke role differs')
    require(update in range(4),'paired batch update outside registered smoke dose')
    import torch
    root=Path(root)/'paired-batches';root.mkdir(parents=True,exist_ok=True)
    path=root/f'update{update}.pt';manifest=root/f'update{update}.json'
    if mode=='control':
        require(not path.exists() and not manifest.exists(),'control batch already recorded')
        # DataProto.to mutates its receiver. Preserve the control's live input.
        cached=copy.deepcopy(batch).to('cpu')
        digest=batch_hash(cached)
        require(batch_hash(batch)==digest,'control batch changed during capture')
        with path.open('xb') as f:torch.save(cached,f)
        receipt=dict(update=update,sha256=sha(path),input_hash=digest,
            includes='full D actor DataProto: tensors, prompt IDs, non-tensor data and metadata')
        atomic(manifest,receipt)
    else:
        receipt=read(manifest)
        require(receipt['update']==update and sha(path)==receipt['sha256'],'paired control batch file changed')
        cached=torch.load(path,weights_only=False)
        require(batch_hash(cached)==receipt['input_hash'],'paired control batch input changed')
        if update==0:
            require(batch_hash(batch)==receipt['input_hash'],'X4 update-0 inputs differ between arms')
        batch=cached
    require(batch_hash(batch)==receipt['input_hash'],f'paired input identity failed at update {update}')
    atomic(Path(run)/f'paired-batch-update{update}.json',dict(receipt,role=role,identity=True))
    return batch


def paired_receipt(root,control,p2):
    """Fail closed unless all four actor calls used the same bound full inputs."""
    updates=[]
    for update in range(4):
        a=read(Path(control)/f'paired-batch-update{update}.json')
        b=read(Path(p2)/f'paired-batch-update{update}.json')
        expected=read(Path(root)/'paired-batches'/f'update{update}.json')
        require(a==dict(expected,role='control',identity=True) and b==dict(expected,role='p2',identity=True)
            and expected['update']==update and sha(Path(root)/'paired-batches'/f'update{update}.pt')==expected['sha256'],
            f'paired batch receipt identity failed at update {update}')
        updates.append(dict(expected,identity=True))
    receipt=dict(schema='x-smoke-paired-batches.v1',smoke_only=True,control_first=True,updates=updates)
    validate_pairing(receipt)
    return receipt


def validate_pairing(receipt):
    updates=receipt.get('updates',[])
    require(receipt.get('schema')=='x-smoke-paired-batches.v1' and receipt.get('smoke_only') is True and
        receipt.get('control_first') is True and len(updates)==4 and
        all(row.get('update')==i and row.get('identity') is True and
            all(isinstance(row.get(key),str) and len(row[key])==64 and
                all(c in '0123456789abcdef' for c in row[key]) for key in ('sha256','input_hash'))
            for i,row in enumerate(updates)),'P2 attempt 2 requires identity-bound paired batches at every update')
