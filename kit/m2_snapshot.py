"""Verified local tar -> durable tar handoffs. Only an uploaded, verified object gets a pointer."""
from __future__ import annotations
from contextlib import contextmanager
import fcntl
from contextvars import ContextVar
import io
import json
import os
from pathlib import Path
import shutil
import hashlib
import subprocess
import tarfile
import tempfile
import sys
import time
import uuid
from kit.m2_assignment import sha, require, digest


def safe_key(key):
    require(key and all(c.isalnum() or c in '-_.' for c in key) and key not in ('.','..'), 'unsafe snapshot key')
    return key


def atomic(path, doc):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    with temp.open('w') as stream:
        json.dump(doc,stream,indent=2,sort_keys=True);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    temp.replace(path)
    fd=os.open(path.parent,os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)


def immutable(path, raw):
    """Publish a write-once declaration atomically, with durable file and directory."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name('.declare-'+uuid.uuid4().hex)
    try:
        with temp.open('xb') as stream:
            stream.write(raw);stream.flush();os.fsync(stream.fileno())
        try:os.link(temp,path)
        except FileExistsError:require(path.read_bytes()==raw,'write-once immutable declaration changed: '+str(path))
        fd=os.open(path.parent,os.O_RDONLY)
        try:os.fsync(fd)
        finally:os.close(fd)
    finally:temp.unlink(missing_ok=True)


def flock_preflight(root,nodes=1):
    """Check filesystem locking policy and two-process flock exclusion/release."""
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    try:
        value=json.loads(subprocess.check_output(['findmnt','-J','-T',str(root),'-o','FSTYPE,OPTIONS'],text=True))['filesystems'][0]
        fstype=value['fstype'].lower();options=set(value.get('options','').split(','))
        shared={'lustre','nfs','nfs4','gpfs','spectrumscale','ceph','cephfs','beegfs','gfs2','ocfs2'}
        attestation=os.environ.get('M2_DURABLE_FLOCK_ATTEST')
        known_bad={'tmpfs','ramfs','overlay','overlayfs','ext2','ext3','ext4','xfs','btrfs','zfs','vfat','exfat'}
        require(nodes==1 or fstype in shared or (fstype not in known_bad and attestation==fstype),
                'durable filesystem lacks known cross-node flock support; unknown types require M2_DURABLE_FLOCK_ATTEST=<fstype>')
        if fstype in shared or nodes>1:
            require(not {'noflock','localflock','nolock'} & options and not any(o.startswith('local_lock=') and o!='local_lock=none' for o in options),
                    'durable filesystem disables cross-node flock')
        probe=root/('.flock-probe-'+uuid.uuid4().hex)
        code="import fcntl,sys; f=open(sys.argv[1],'a');\ntry: fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)\nexcept BlockingIOError: raise SystemExit(7)"
        try:
            with probe.open('a') as stream:
                fcntl.flock(stream,fcntl.LOCK_EX)
                require(subprocess.run([sys.executable,'-c',code,str(probe)],check=False).returncode==7,'functional flock exclusion failed')
            require(subprocess.run([sys.executable,'-c',code,str(probe)],check=False).returncode==0,'functional flock release failed')
        finally:probe.unlink(missing_ok=True)
        result=dict(filesystem=fstype,options=sorted(options),nodes=nodes,functional_flock='PASS',attestation=attestation if fstype not in shared else None)
        print('durable locking preflight:',json.dumps(result,sort_keys=True),flush=True)
        return result
    except (OSError,subprocess.CalledProcessError,KeyError,IndexError,json.JSONDecodeError) as error:
        require(False,'durable filesystem/flock preflight failed: '+str(error))


def compatible_parents(old, parents, parent_complete, partial):
    if old['parents']==parents:return True
    return bool(partial and not old['complete'] and 'parent_complete' in old and
                set(old['parents'])==set(parents)==set(old['parent_complete']) and all(not done or old['parents'][key]==parents.get(key)
                for key,done in old['parent_complete'].items()))


def header(path, expected_manifest=None):
    """Read only the first tar member; never rehash a durable multi-TB archive."""
    try:
        with tarfile.open(path,'r:') as archive:
            first=archive.next()
            require(first is not None and first.name=='snapshot.json' and first.isfile(),'snapshot manifest must be first')
            doc=json.load(archive.extractfile(first))
        if expected_manifest is not None:require(digest(doc)==expected_manifest,'snapshot manifest digest mismatch')
        return doc
    except (tarfile.TarError,EOFError,json.JSONDecodeError) as exc:
        require(False,'snapshot tar SHA256/manifest corruption: '+str(exc))


def copy_hashed(source,destination,expected):
    checksum=hashlib.sha256()
    with Path(source).open('rb') as src,Path(destination).open('wb') as dst:
        for block in iter(lambda:src.read(8*1024*1024),b''):
            checksum.update(block);dst.write(block)
        dst.flush();os.fsync(dst.fileno())
    require(checksum.hexdigest()==expected,'snapshot tar SHA256 mismatch during copy')


def quarantine(out):
    """Partial writes are retained outside the published state, never followed."""
    out=Path(out);discarded=out.parent/'discarded'/uuid.uuid4().hex
    for path in sorted(out.rglob('*'),key=lambda p:len(p.parts)):
        if path.is_symlink() or path.name.endswith('.tmp'):
            if not path.exists() and not path.is_symlink():continue
            dest=discarded/path.relative_to(out);dest.parent.mkdir(parents=True,exist_ok=True)
            path.rename(dest)


@contextmanager
def lock(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    with (root/'.lock').open('a') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX)
        yield


@contextmanager
def archive_lease(path,exclusive=False):
    """A prune never unlinks a tar while another node is opening/restoring it."""
    with Path(str(path)+'.lease').open('a') as stream:
        try:fcntl.flock(stream,(fcntl.LOCK_EX|fcntl.LOCK_NB) if exclusive else fcntl.LOCK_SH)
        except BlockingIOError:
            yield False
            return
        try:yield True
        finally:fcntl.flock(stream,fcntl.LOCK_UN)


def members(archive):
    values=archive.getmembers()
    require(len({m.name for m in values})==len(values), 'duplicate snapshot members')
    require(all(m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts for m in values), 'unsafe snapshot member')
    return values


def verify_tar(path, expected=None):
    if expected is not None: require(sha(path)==expected, 'snapshot tar SHA256 mismatch')
    with tarfile.open(path,'r:') as archive:
        values=members(archive)
        index={m.name:m for m in values}
        require('snapshot.json' in {m.name for m in values}, 'missing snapshot manifest')
        doc=json.load(archive.extractfile(index['snapshot.json']))
        require({m.name for m in values}=={'snapshot.json',*('tree/'+n for n in doc['files'])}, 'snapshot inventory mismatch')
        for name, record in doc['files'].items():
            stream=archive.extractfile(index['tree/'+name])
            import hashlib
            h=hashlib.sha256();size=0
            for block in iter(lambda:stream.read(1024*1024),b''):h.update(block);size+=len(block)
            require(size==record['bytes'] and h.hexdigest()==record['sha256'], 'snapshot member mismatch '+name)
    return doc


SPACE_RESERVE_BYTES=64*1024**3
_TECHNICAL_SPACE=ContextVar('m2_technical_space',default=False)


@contextmanager
def technical_space_budget(standin):
    """CPU synthetic fixtures need no 64 GiB real-model reserve; bytes still checked."""
    token=_TECHNICAL_SPACE.set(standin)
    try:yield
    finally:_TECHNICAL_SPACE.reset(token)



def space_guard(path,selected_bytes):
    """Pend before allocating snapshots/replicas, leaving a fixed node reserve."""
    path=Path(path)
    while not path.exists():path=path.parent
    required=2*selected_bytes+(0 if _TECHNICAL_SPACE.get() else SPACE_RESERVE_BYTES)
    available=shutil.disk_usage(path).free
    if available<required:raise Pending(f'insufficient local free space: {available} bytes available; {required} required (2x selected bytes + 64 GiB)')


def publish(out, durable, key, *, parents=None, binding=None, assignment=None, complete=True, copy=shutil.copyfile, origins=None, refresh_from=None, dependencies=None, parent_complete=None, partial_inputs=False):
    out=Path(out);root=Path(durable)/'snapshots'/safe_key(key)
    require(out.is_dir(), 'snapshot local state missing')
    if not complete:quarantine(out)
    files={}
    for path in sorted(out.rglob('*')):
        if path.name=='.lock': continue
        require(not path.is_symlink(), 'snapshot cannot contain an unverified symlink '+str(path))
        if path.is_file():
            require(not path.name.endswith('.tmp'), 'unfinished local atomic write '+str(path))
            files[str(path.relative_to(out))]=dict(sha256=sha(path),bytes=path.stat().st_size)
    doc=dict(schema='message2-stage-snapshot.v2',key=key,origin=str(out.resolve()),files=files,
        parents=parents or {},parent_complete=parent_complete if parent_complete is not None else {k:v is not None for k,v in (parents or {}).items()},binding=binding,assignment=assignment,complete=complete,origins=origins or {},dependencies=dependencies or {})
    space_guard(out.parent,sum(record['bytes'] for record in files.values()))
    # Build on local storage; the shared filesystem sees one tar and small receipts only.
    with tempfile.TemporaryDirectory(prefix='m2-snapshot-',dir=out.parent) as tmp:
        tarpath=Path(tmp)/'snapshot.tar'
        with tarfile.open(tarpath,'w:') as archive:
            raw=json.dumps(doc,sort_keys=True).encode();info=tarfile.TarInfo('snapshot.json');info.size=len(raw)
            archive.addfile(info,io.BytesIO(raw))
            for name in files:
                info=tarfile.TarInfo('tree/'+name);info.size=files[name]['bytes'];info.mode=(out/name).stat().st_mode & 0o777
                with (out/name).open('rb') as stream:archive.addfile(info,stream)
        verify_tar(tarpath);checksum=sha(tarpath)
        with lock(root):
            for stale in root.glob('.upload-*'):stale.unlink(missing_ok=True)
            current=root/'CURRENT.json';old=None
            if current.exists():
                old=read_pointer(durable,key)
                if old['sha256']==checksum:return old
                require(not old['complete'], 'completed snapshot is immutable: '+key)
                require(compatible_parents(old,doc['parents'],parent_complete,partial_inputs) and (old['binding']==binding or old['binding'] is None) and old['assignment']==assignment, 'resume input identity changed')
            for stale in root.glob('.upload-*'):stale.unlink(missing_ok=True)
            destination=root/(checksum+'.tar')
            upload=root/('.upload-'+uuid.uuid4().hex)
            try:
                if copy is shutil.copyfile:copy_hashed(tarpath,upload,checksum)
                else:
                    copy(tarpath,upload)
                    require(sha(upload)==checksum,'snapshot tar SHA256 mismatch during copy')
                upload.replace(destination)
                checksum_path=destination.with_suffix('.tar.sha256')
                immutable(checksum_path,(checksum+'\n').encode())
                immutable(destination.with_suffix('.tar.manifest.sha256'),(digest(doc)+'\n').encode())
                with destination.open('rb') as stream:os.fsync(stream.fileno())
                pointer=dict(tar=destination.name,sha256=checksum,manifest_sha256=digest(doc),
                    parents=doc['parents'],parent_complete=doc['parent_complete'],binding=binding,assignment=assignment,complete=complete)
                if old is not None:
                    # Grace starts when CURRENT retires this object, even if
                    # a long-running partial was created days ago. Bytes stay immutable.
                    try:os.utime(root/old['tar'],None)
                    except FileNotFoundError:pass
                atomic(current,pointer)
                try:prune_incomplete(durable,key)
                except Exception as error:print('snapshot prune deferred:',error,flush=True)
                return pointer
            finally:
                upload.unlink(missing_ok=True)


def read_pointer(durable,key, *, completed=False):
    root=Path(durable)/'snapshots'/safe_key(key);path=root/'CURRENT.json'
    if not path.exists(): raise Pending('snapshot pending: '+key)
    doc=json.loads(path.read_text())
    require(doc['tar']==doc['sha256']+'.tar','unsafe snapshot pointer')
    require((root/(doc['tar']+'.sha256')).read_text().strip()==doc['sha256'], 'snapshot companion checksum mismatch')
    if completed and not doc['complete']:raise Pending('snapshot unfinished: '+key)
    return doc


class Pending(RuntimeError):
    """Exit 3: the partner supplies sbatch dependencies; there is no internal wait loop."""


def restore(durable,key,destination, *, completed=True):
    pointer=read_pointer(durable,key,completed=completed)
    destination=Path(destination)
    require(not destination.exists(), 'restore destination must be fresh')
    destination.parent.mkdir(parents=True,exist_ok=True)
    space_guard(destination.parent,sum(r['bytes'] for r in header(Path(durable)/'snapshots'/key/pointer['tar'],pointer['manifest_sha256'])['files'].values()))
    with tempfile.TemporaryDirectory(prefix='m2-restore-',dir=destination.parent) as tmp:
        local=Path(tmp)/'snapshot.tar'
        source=Path(durable)/'snapshots'/key/pointer['tar']
        with archive_lease(source):copy_hashed(source,local,pointer['sha256'])
        doc=verify_tar(local)
        require(digest(doc)==pointer['manifest_sha256'] and doc['complete']==pointer['complete'], 'snapshot manifest/pointer mismatch')
        staging=Path(tmp)/'tree';staging.mkdir()
        with tarfile.open(local,'r:') as archive:
            index={m.name:m for m in archive.getmembers()}
            for name in doc['files']:
                path=staging/name;path.parent.mkdir(parents=True,exist_ok=True)
                with archive.extractfile(index['tree/'+name]) as source,path.open('wb') as target:shutil.copyfileobj(source,target)
                path.chmod(index['tree/'+name].mode)
        staging.replace(destination)
    return doc,pointer


def restore_at(durable,key,checksum,destination):
    """Restore a parent's exact historical object, never a newer CURRENT."""
    safe_key(key)
    require(len(checksum)==64 and all(c in '0123456789abcdef' for c in checksum),'unsafe snapshot checksum')
    destination=Path(destination)
    require(not destination.exists(),'restore destination must be fresh')
    destination.parent.mkdir(parents=True,exist_ok=True)
    space_guard(destination.parent,sum(r['bytes'] for r in header(Path(durable)/'snapshots'/key/(checksum+'.tar'))['files'].values()))
    with tempfile.TemporaryDirectory(prefix='m2-restore-',dir=destination.parent) as tmp:
        local=Path(tmp)/'snapshot.tar';source=Path(durable)/'snapshots'/key/(checksum+'.tar')
        with archive_lease(source):copy_hashed(source,local,checksum)
        doc=verify_tar(local);staging=Path(tmp)/'tree';staging.mkdir()
        with tarfile.open(local,'r:') as archive:
            index={m.name:m for m in archive.getmembers()}
            for name in doc['files']:
                path=staging/name;path.parent.mkdir(parents=True,exist_ok=True)
                with archive.extractfile(index['tree/'+name]) as source,path.open('wb') as target:shutil.copyfileobj(source,target)
                path.chmod(index['tree/'+name].mode)
        staging.replace(destination)
    return doc


def overlay(source,dest,namespace=None,compact=False,mode="full"):
    """Reducers assemble immutable shards; a conflicting path is a correctness stop."""
    source,dest=Path(source),Path(dest)
    skipped=set()
    for path in sorted(source.rglob('*')):
        if not path.is_file() or path.name=='.lock':continue
        relative=path.relative_to(source)
        if not selected_file(str(relative),mode):
            parts=relative.parts
            heavy=next((i for i,p in enumerate(parts) if p.startswith(('hf-','global_step_','checkpoint-'))),len(parts)-1)
            skipped.add(str(Path(*parts[:heavy+1])));continue
        if compact:
            parts=relative.parts
            if parts[0]=='own' and len(parts)>3 and parts[3] in ('raw','checks'):
                skipped.add(str(Path(*parts[:4])));continue
            if parts[0]=='scores' and len(parts)>2 and parts[2]=='raw':
                skipped.add(str(Path(*parts[:3])));continue
        target=dest/relative
        if namespace and relative.parts[0] in ('logs','shard-logs'):
            target=dest/'shard-logs'/namespace/relative
        if target.exists() and sha(target)!=sha(path) and relative.name in ('nomination.json','nomination-commit.json'):
            if (source/'nomination.json').exists() and (dest/'nomination.json').exists():
                old=json.loads((dest/'nomination.json').read_text());new=json.loads((source/'nomination.json').read_text())
                require(old['binding_sha256']==new['binding_sha256'] and old['nomination']['broad']==new['nomination']['broad'],'broad nomination changed during final resolution')
                if old.get('scope')=='broad' and new.get('scope')=='full':shutil.copyfile(path,target);continue
                if old.get('scope')=='full' and new.get('scope')=='broad':continue
        if target.exists() and relative.name=='assets.json' and sha(target)!=sha(path):
            old=json.loads(target.read_text());new=json.loads(path.read_text())
            require(old['evaluation']==new['evaluation'],'asset evaluation changed')
            if all(new['inputs'].get(k)==v for k,v in old['inputs'].items()):shutil.copyfile(path,target);continue
            if all(old['inputs'].get(k)==v for k,v in new['inputs'].items()):continue
        if target.exists():require(sha(target)==sha(path), 'conflicting shard '+str(path.relative_to(source)))
        else:target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
    return skipped


def selected_file(name,mode):
    parts=Path(name).parts
    heavy=any(p.startswith(('hf-','global_step_','checkpoint-')) for p in parts) or Path(name).suffix in ('.safetensors','.pt','.pth','.bin')
    if mode=='full':return True
    if mode=='score' and any(p.startswith('hf-') for p in parts):return True
    if parts and parts[0] in ('data','p2-data'):return False
    if parts and parts[0]=='runs':
        if len(parts)>3 and parts[2] in ('data','sdft','train'):return False
        if parts[-1].endswith('-schedule.jsonl'):return False
    return not heavy


def restore_view(durable,key,checksum,destination,mode='evidence',expected_manifest=None,prefix=None):
    source=Path(durable)/'snapshots'/safe_key(key)/(checksum+'.tar')
    with archive_lease(source):
        return _restore_view(durable,key,checksum,destination,mode,expected_manifest,prefix)


def _restore_view(durable,key,checksum,destination,mode='evidence',expected_manifest=None,prefix=None):
    """Copy only selected members, hashing their bytes; never copy a whole parent tar.

    The first member is bound by CURRENT's manifest digest. Every extracted file
    is independently checked against it. Unselected tensors are never read.
    """
    source=Path(durable)/'snapshots'/safe_key(key)/(checksum+'.tar')
    doc=header(source,expected_manifest)
    if expected_manifest is None:
        receipt=source.with_suffix('.tar.manifest.sha256')
        if receipt.exists():require(digest(doc)==receipt.read_text().strip(),'historical snapshot manifest changed')
        else:
            # Legacy objects need one full integrity pass before adopting a header receipt.
            doc=verify_tar(source,checksum)
            immutable(receipt,(digest(doc)+'\n').encode())
        require((source.with_suffix('.tar.sha256')).read_text().strip()==checksum,'snapshot checksum receipt changed')
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    def wanted(name):return (name==prefix or name.startswith(prefix.rstrip('/')+'/')) if prefix is not None else selected_file(name,mode)
    space_guard(destination,sum(r['bytes'] for name,r in doc['files'].items() if wanted(name) and not (destination/name).exists()))
    with tarfile.open(source,'r:') as archive:
        values=members(archive);index={m.name:m for m in values}
        require(set(index)=={'snapshot.json',*('tree/'+n for n in doc['files'])},'snapshot inventory mismatch')
        for name,record in doc['files'].items():
            if prefix is not None:
                if not (name==prefix or name.startswith(prefix.rstrip('/')+'/')):continue
            elif not selected_file(name,mode):continue
            target=destination/name
            if target.exists():
                require(target.is_file() and not target.is_symlink() and sha(target)==record['sha256'],'cached replica changed: '+name)
                continue
            target.parent.mkdir(parents=True,exist_ok=True)
            temp=target.with_name(target.name+'.'+uuid.uuid4().hex+'.tmp')
            try:
                checksum_file=hashlib.sha256();count=0
                with archive.extractfile(index['tree/'+name]) as src,temp.open('wb') as dst:
                    for block in iter(lambda:src.read(8*1024*1024),b''):
                        checksum_file.update(block);count+=len(block);dst.write(block)
                require(count==record['bytes'] and checksum_file.hexdigest()==record['sha256'],'snapshot member mismatch '+name)
                temp.chmod(index['tree/'+name].mode);temp.replace(target)
            finally:temp.unlink(missing_ok=True)
    return doc


def prune_incomplete(durable,key,grace_seconds=86400):
    """Keep CURRENT, ancestors and partials younger than a day; pruning is optional."""
    durable=Path(durable);protected=set()
    for current in (durable/'snapshots').glob('*/CURRENT.json'):
        pointer=json.loads(current.read_text());protected.add(pointer['sha256'])
        protected.update(v for v in pointer.get('parents',{}).values() if v)
        doc=header(current.parent/pointer['tar'],pointer['manifest_sha256'])
        protected.update(v['sha256'] for v in doc.get('dependencies',{}).values())
    root=durable/'snapshots'/key
    for path in root.glob('*.tar'):
        if path.stem not in protected and time.time()-path.stat().st_mtime>=grace_seconds and not header(path)['complete']:
            with archive_lease(path,exclusive=True) as acquired:
                if acquired:
                    path.unlink(missing_ok=True);path.with_suffix('.tar.sha256').unlink(missing_ok=True);path.with_suffix('.tar.manifest.sha256').unlink(missing_ok=True)
