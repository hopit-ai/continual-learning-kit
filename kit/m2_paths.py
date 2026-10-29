"""Resolve immutable origin paths into verified node-local snapshot replicas."""
import json
import os
from pathlib import Path
from functools import lru_cache


@lru_cache(maxsize=8)
def _pairs(raw):
    return tuple(sorted(json.loads(raw).items(),key=lambda item:-len(item[0])))


@lru_cache(maxsize=8)
def _from_file(path,mtime,size):
    return _pairs(Path(path).read_text())


def mappings(name):
    path=os.environ.get(name+'_FILE')
    if path:
        stat=Path(path).stat()
        return _from_file(path,stat.st_mtime_ns,stat.st_size)
    return _pairs(os.environ.get(name,'{}'))


def local(path):
    if path is None:return None
    original=Path(path);value=str(original)
    replica_root=os.environ.get('M2_REPLICA_ROOT')
    if replica_root and (value==replica_root or value.startswith(replica_root+'/')):
        materialize(original)
        return original
    for origin,target in mappings('M2_PATH_MAP'):
        if value==origin or value.startswith(origin+'/'):
            result=Path(target+value[len(origin):]);materialize(result,origin,value)
            return result
    return original


def equivalent(left,right):
    """Only verified replica aliases may vary in an otherwise identical resume."""
    aliases=mappings('M2_CANONICAL_ROOTS')
    def normal(value):
        if isinstance(value,dict):return {k:normal(v) for k,v in value.items()}
        if isinstance(value,list):return [normal(v) for v in value]
        if isinstance(value,str) and '/' in value:
            for path,logical in aliases:value=value.replace(path,logical)
        return value
    return normal(left)==normal(right)


def materialize(path,origin=None,value=None,*,dependencies=None,replica_root=None,durable=None):
    """Resolve only the exact model/state prefix that a consumer actually opens."""
    dependency_file=os.environ.get('M2_DEPENDENCIES_FILE')
    if path.is_file():return
    if dependencies is None:
        if not dependency_file:return
        dependencies=json.loads(Path(dependency_file).read_text())
    replica_root=Path(replica_root or os.environ['M2_REPLICA_ROOT'])
    durable=durable or os.environ['M2_DURABLE']
    from kit import m2_snapshot as snapshots
    for key,ref in sorted(dependencies.items(),key=lambda pair:-len(pair[0])):
        target=replica_root/ref['sha256']/ref['path']
        if path==target or target in path.parents:
            stamp=replica_root/'.views'/ref['sha256']/__import__('hashlib').sha256(str(path).encode()).hexdigest()
            if stamp.exists():return
            relative=str(path.relative_to(replica_root/ref['sha256']))
            with snapshots.lock(replica_root/(ref['sha256']+'.lock')):
                snapshots.restore_view(durable,ref['task'],ref['sha256'],replica_root/ref['sha256'],
                    prefix=relative,expected_manifest=ref.get('manifest_sha256'))
                stamp.parent.mkdir(parents=True,exist_ok=True);stamp.write_text('verified\n')
            return
