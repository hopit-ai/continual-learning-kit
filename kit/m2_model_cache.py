"""Reuse CPU model hashes while every local file stat still matches.

The inherited identity and role/export checks stay unchanged. Core entry points
install the hook before inherited scorers check their models.
"""
import hashlib
import json
import os
from pathlib import Path
from kit import v4_teacher as t

uncached_identity=t.local_identity


def fingerprint(path):
    result={}
    for file in sorted(Path(path).resolve().iterdir()):
        if file.is_file() and file.suffix in ('.json','.jinja','.safetensors','.bin'):
            stat=file.stat()
            result[file.name]=dict(path=str(file.resolve()),size=stat.st_size,mtime_ns=stat.st_mtime_ns,
                ctime_ns=stat.st_ctime_ns,inode=stat.st_ino,device=stat.st_dev)
    return result


def local_identity(path,model_name='Qwen/Qwen3.6-27B'):
    receipt=os.environ.get('M2_MODEL_HASH_RECEIPTS_FILE')
    if not receipt:return uncached_identity(path,model_name)
    from kit.p3a_common import atomic,NotReady,CorrectnessStop
    file=Path(receipt);raw=file.read_bytes() if file.exists() else b'{}'
    seal=os.environ.get('M2_MODEL_HASH_RECEIPTS_SHA256')
    if os.environ.get('M2_EXECUTION_PHASE')=='gpu' and not seal:raise NotReady('CPU model hash receipt unsealed; re-run CPU staging')
    if seal and hashlib.sha256(raw).hexdigest()!=seal:raise CorrectnessStop('CPU model hash receipt changed')
    rows=json.loads(raw);key=json.dumps([str(Path(path).resolve()),model_name]);stats=fingerprint(path)
    if key in rows and rows[key]['files']==stats:return rows[key]['identity']
    if os.environ.get('M2_EXECUTION_PHASE')=='gpu':raise NotReady('model hash receipt stale/missing; re-run CPU staging')
    observed=uncached_identity(path,model_name)
    if fingerprint(path)!=stats:raise CorrectnessStop('model changed while hashing on CPU')
    rows[key]=dict(files=stats,identity=observed);atomic(file,rows)
    return observed


def install():
    t.local_identity=local_identity
