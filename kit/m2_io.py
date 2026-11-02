"""Publish identical shared lane inputs with temp+rename, never partial copies."""
import errno
import os
from pathlib import Path
import shutil
import tempfile
from kit.m2_assignment import require,sha


def lease_fds():return tuple(__import__('json').loads(os.environ.get('M3_GPU_LOCK_FDS','[]')))


def copy_shared(source,target):
    source=Path(source);target=Path(target);target.parent.mkdir(parents=True,exist_ok=True)
    def verify():
        if source.is_dir():
            expected={str(p.relative_to(source)):sha(p) for p in source.rglob('*') if p.is_file()}
            actual={str(p.relative_to(target)):sha(p) for p in target.rglob('*') if p.is_file()}
            require(expected==actual,'shared lane input changed: '+str(target))
        else:require(sha(source)==sha(target),'shared lane input changed: '+str(target))
    if target.exists():verify();return
    with tempfile.TemporaryDirectory(prefix='.'+target.name+'-',dir=target.parent) as temporary:
        staged=Path(temporary)/target.name
        if source.is_dir():shutil.copytree(source,staged)
        else:shutil.copyfile(source,staged)
        try:os.replace(staged,target)
        except OSError as error:
            if error.errno not in (errno.EEXIST,errno.ENOTEMPTY):raise
        verify()
