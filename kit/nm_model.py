"""CPU-bound model files with bounded metadata checks in an allocation.

Reuse the unchanged inherited loader/sharding numerics. Its rank-zero admission
check consumes the CPU hash receipt, rather than rehashing a 32-GB base while
holding eight GPUs. The override exists only during this isolated NM load.
"""
from pathlib import Path
from kit.nm_common import need,sha,load,BASE
import hashlib
import json


def metadata(directory,names):
    root=Path(directory)
    return {name:[(root/name).stat().st_size,(root/name).stat().st_mtime_ns] for name in names}


def prepared_reference(directory,revision,mesh,*,binding,binding_sha256,files_metadata,**kwargs):
    from kit import m3_actor
    original=m3_actor.verify_base_files
    def verify(path,record_path,rev):
        need(Path(path).resolve()==Path(directory).resolve() and Path(record_path).resolve()==Path(binding).resolve() and rev==revision==BASE,'prepared reference identity differs')
        need(sha(binding)==binding_sha256,'CPU base binding changed')
        record=load(binding);files=record['identity']['model_file_hashes']
        need(record['revision']==BASE and set(files)==set(files_metadata),'prepared base file inventory differs')
        observed={p.name for p in Path(directory).iterdir() if p.name=='config.json' or p.name.endswith(('.safetensors','.safetensors.index.json'))}
        need(observed==set(files) and metadata(directory,files)==files_metadata,'prepared base files changed after CPU hashing')
        return hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()
    m3_actor.verify_base_files=verify
    try:return m3_actor.load_frozen_base(directory,revision,mesh,binding=binding,**kwargs)
    finally:m3_actor.verify_base_files=original
