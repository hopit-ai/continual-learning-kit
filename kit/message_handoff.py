#!/usr/bin/env python3
"""Verified durable handoff and scoring epochs for two partner-owned Slurm jobs."""
from pathlib import Path
import sys
if __package__ in (None, ''): sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kit.next_common import *
import argparse
import shutil
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import tempfile
import re


def atomic(path, value):
    """Use a private writer file; CURRENT is the only authoritative pointer."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp=path.with_name(path.name+f'.tmp-{os.getpid()}-{uuid.uuid4().hex}')
    try:
        with tmp.open('w') as handle:
            json.dump(value,handle,indent=2,sort_keys=True);handle.write('\n')
            handle.flush();os.fsync(handle.fileno())
        os.replace(tmp,path)
        sync_dir(path.parent)
    finally:
        tmp.unlink(missing_ok=True)


def sync_dir(path):
    fd=os.open(path,os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)


@contextmanager
def locked(durable):
    durable.mkdir(parents=True,exist_ok=True)
    with (durable/'.handoff.lock').open('a') as handle:
        try:
            fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            raise CorrectnessStop('handoff busy: another save/restore holds '+str(durable/'.handoff.lock'))
        except OSError:
            # Filesystems mounted without flock support (e.g. Lustre without -o flock): fall back to an atomic
            # mkdir lock that records host and pid; a dead holder on this host is detected and cleared.
            yield from _mkdir_lock(durable)
            return
        try: yield
        finally: fcntl.flock(handle,fcntl.LOCK_UN)


def _mkdir_lock(durable):
    lock=durable/'.handoff.lock.d'
    host=os.uname().nodename
    for _ in range(2):
        try:
            lock.mkdir()
            (lock/'owner.json').write_text(json.dumps(dict(host=host,pid=os.getpid())))
            break
        except FileExistsError:
            try: owner=json.loads((lock/'owner.json').read_text())
            except (OSError,ValueError): owner={}
            alive=True
            if owner.get('host')==host:
                try: os.kill(int(owner.get('pid',-1)),0)
                except (OSError,ValueError): alive=False
            if alive: raise CorrectnessStop('handoff busy: another save/restore holds '+str(lock))
            shutil.rmtree(lock,ignore_errors=True)
    else:
        raise CorrectnessStop('handoff lock could not be acquired: '+str(lock))
    try: yield
    finally: shutil.rmtree(lock,ignore_errors=True)


def disjoint(source, destination):
    source=source.resolve();destination=destination.resolve()
    require(source != destination and not destination.is_relative_to(source) and not source.is_relative_to(destination), 'handoff trees must be disjoint')


def inventory(root, previous=None, metadata_only=False):
    """Reuse a file digest only when its size and nanosecond mtime match."""
    require(root.is_dir(),'missing handoff tree: '+str(root))
    entries={};old=(previous or {}).get('entries',{})
    for p in sorted(root.rglob('*')):
        name=str(p.relative_to(root))
        if p.is_symlink(): entries[name]=dict(kind='symlink',target=os.readlink(p))
        elif p.is_file():
            st=p.stat();entry=dict(kind='file',bytes=st.st_size,mtime_ns=st.st_mtime_ns,mode=st.st_mode & 0o7777)
            if not metadata_only:
                cached=old.get(name,{})
                entry['sha256']=cached['sha256'] if all(cached.get(k)==v for k,v in entry.items()) and 'sha256' in cached else s.file_hash(p)
            entries[name]=entry
        elif p.is_dir(): entries[name]=dict(kind='directory')
        else: raise CorrectnessStop('unsupported handoff entry: '+name)
    return dict(entries=entries,files=sum(v['kind']=='file' for v in entries.values()),
                bytes=sum(v.get('bytes',0) for v in entries.values()))


def metadata(manifest):
    return {k:{a:b for a,b in v.items() if a!='sha256'} for k,v in manifest['entries'].items()}


def verify_tree(tree, manifest, changed=None):
    # rsync -a on older hosts preserves only whole-second mtimes. Source ns
    # remains in the manifest for detecting even same-second source changes.
    observed=inventory(tree,metadata_only=True)
    expected=metadata(manifest)
    for entries in (observed['entries'],expected):
        for entry in entries.values():
            if entry['kind']=='file' and 'mtime_ns' in entry: entry['mtime_ns']//=1_000_000_000
    for name,entry in observed['entries'].items():
        if name in expected:
            for key in ('mtime_ns','mode'):
                if key not in expected[name]: entry.pop(key,None)
    require(observed['entries']==expected and observed['files']==manifest['files'] and observed['bytes']==manifest['bytes'],
            'durable generation verification differs (size/mtime/type)')
    for name,entry in manifest['entries'].items():
        if entry['kind']=='file' and (changed is None or name in changed):
            require(s.file_hash(tree/name)==entry['sha256'],'durable generation verification differs (hash): '+name)


def committed(durable, verify=True):
    """CURRENT is the commit point. Partials and dead writer temps are ignored."""
    stale=list(durable.glob('CURRENT.json.tmp*'))
    if stale: print('WARNING: ignoring stale CURRENT writer temp(s): '+', '.join(p.name for p in stale),file=sys.stderr)
    pointer=durable/'CURRENT.json'
    if not pointer.exists():
        require(not durable.exists() or not any(p.name not in ('generations','.handoff.lock','.handoff.lock.d','job-starts') and not p.name.startswith('CURRENT.json.tmp') for p in durable.iterdir()),
                'durable tree has no committed generation; inspect/migrate old evidence first')
        return None
    current=json.loads(pointer.read_text());name=current['generation']
    require(Path(name).name==name and not name.startswith('.'),'invalid generation name')
    generation=durable/'generations'/name
    check(generation/'manifest.json',current['manifest_sha256'])
    require(json.loads((generation/'COMMITTED.json').read_text())==current,'uncommitted durable generation')
    if verify: verify_tree(generation/'tree',json.loads((generation/'manifest.json').read_text()))
    return generation


def cleanup(durable, keep):
    generations=durable/'generations'
    for path in generations.iterdir():
        if path.name not in keep:
            if path.is_dir() and not path.is_symlink(): shutil.rmtree(path)
            else: path.unlink()
    for path in durable.glob('CURRENT.json.tmp*'): path.unlink()


def save(source, durable):
    disjoint(source,durable)
    with locked(durable): return save_locked(source,durable)


def save_locked(source,durable):
    previous=committed(durable,verify=False)
    old=json.loads((previous/'manifest.json').read_text()) if previous else None
    manifest=inventory(source,old)
    if old==manifest:
        current=json.loads((durable/'CURRENT.json').read_text())
        atomic(durable/'last-job-end.json',dict(current,skipped=True,job=os.environ.get('SLURM_JOB_ID','standin'),time_ns=time.time_ns()))
        return dict(skipped=True,bytes_copied=0,changed_bytes=0,verified=True,**current)
    generations=durable/'generations';generations.mkdir(parents=True,exist_ok=True)
    name=str(time.time_ns())+'-'+uuid.uuid4().hex
    staging=generations/('.partial-'+name);tree=staging/'tree';tree.mkdir(parents=True)
    changed={k for k,v in manifest['entries'].items() if v!=(old or {}).get('entries',{}).get(k)}
    copied=0
    # Explicit lists avoid rsync's whole-second quick check hard-linking a
    # same-size file changed within that second. Never touch an old inode.
    for reuse in (True,False):
        names=[k for k,v in manifest['entries'].items() if (k not in changed)==reuse]
        if not names: continue
        with tempfile.NamedTemporaryFile(dir=staging,prefix='rsync-',delete=False) as listing:
            listing.write(b''.join(os.fsencode(k)+b'\0' for k in names));list_path=Path(listing.name)
        command=['rsync','-a','--no-r','--from0','--files-from='+str(list_path),'--stats']
        if reuse: command+=['--link-dest='+str((previous/'tree').resolve())]
        else: command+=['--ignore-times']
        command += [str(source.resolve())+'/',str(tree)+'/']
        result=subprocess.run(command,check=True,capture_output=True,text=True,env=dict(os.environ,LC_ALL='C'))
        match=re.search(r'Total transferred file size:\s*([\d,]+)',result.stdout)
        require(match is not None,'rsync did not report transferred bytes')
        copied+=int(match.group(1).replace(',',''));list_path.unlink()
    require(metadata(inventory(source,metadata_only=True))==metadata(manifest),'OUT changed during save; retry after stopping writers')
    verify_tree(tree,manifest,changed)
    # Cached files must actually share the previous inode, rather than silently
    # becoming full copies on a filesystem without hard-link support.
    if previous:
        for k,v in manifest['entries'].items():
            if k not in changed and v['kind']=='file':
                require(os.path.samefile(tree/k,previous/'tree'/k),'rsync failed to hard-link unchanged file: '+k)
    atomic(staging/'manifest.json',manifest)
    current=dict(generation=name,manifest_sha256=s.file_hash(staging/'manifest.json'))
    atomic(staging/'COMMITTED.json',current)
    staging.rename(generations/name);sync_dir(generations)
    atomic(durable/'CURRENT.json',current)
    cleanup(durable,{name,*([previous.name] if previous else [])})
    atomic(durable/'last-job-end.json',dict(current,job=os.environ.get('SLURM_JOB_ID','standin'),time_ns=time.time_ns()))
    return dict(destination=str(generations/name/'tree'),skipped=False,bytes_copied=copied,
                changed_bytes=sum(manifest['entries'][k].get('bytes',0) for k in changed),verified=True,**current)


def restore(durable, destination, move_aside=True):
    disjoint(durable,destination)
    with locked(durable):
        generation=committed(durable)
        if generation is None:
            destination.mkdir(parents=True,exist_ok=True)
            return 'FIRST JOB: no committed durable OUT yet'
        manifest=json.loads((generation/'manifest.json').read_text());expected=manifest['entries']
        moved=[];newer=[];backup=None
        if destination.exists():
            local=inventory(destination)['entries']
            # Compare content, not filesystem-dependent mtime precision.
            def content(v): return {k:x for k,x in v.items() if k!='mtime_ns'}
            differing=[k for k,v in local.items() if k not in expected or content(v)!=content(expected[k])]
            newer=[k for k,v in local.items() if v['kind']=='file' and (k not in expected or v.get('mtime_ns',0)>expected[k].get('mtime_ns',0))]
            differing=sorted(set(differing+newer))
            if newer: print('WARNING: NEWER/LOCAL-ONLY FILES WILL BE PRESERVED: '+', '.join(newer),file=sys.stderr)
            require(move_aside or not differing,'local OUT differs from CURRENT; rerun restore with --move-aside to preserve: '+', '.join(differing))
            if differing:
                backup=destination.with_name(destination.name+'.pre-restore-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')+'-'+uuid.uuid4().hex[:8])
                for name in sorted(differing,key=lambda k:(len(Path(k).parts),k)):
                    if any(Path(name).is_relative_to(Path(parent)) for parent in moved): continue
                    src=destination/name;dst=backup/name;dst.parent.mkdir(parents=True,exist_ok=True)
                    src.rename(dst);moved.append(name)
                print('PRESERVED local files in '+str(backup)+': '+', '.join(moved),file=sys.stderr)
        destination.mkdir(parents=True,exist_ok=True)
        subprocess.run(['rsync','-ac',str(generation/'tree')+'/',str(destination.resolve())+'/'],check=True)
        verify_tree(destination,manifest)
        # Restored files may carry whole-second mtimes (openrsync, Lustre). After content is verified, put back the
        # recorded nanosecond mtimes so the next save reuses cached digests and hard-links instead of re-copying OUT.
        for name,entry in expected.items():
            if entry.get('kind')=='file' and 'mtime_ns' in entry:
                target=destination/name
                st=target.stat()
                os.utime(target,ns=(st.st_atime_ns,entry['mtime_ns']),follow_symlinks=False)
        return dict(source=str(generation/'tree'),destination=str(destination),verified=True,
                    moved=moved,newer_local=newer,backup=str(backup) if backup else None)


def epoch(out, value):
    """Preserve incompatible old evidence outside active reader trees; re-score every control first."""
    path=out/'scoring-epoch.json'
    previous=json.loads(path.read_text()) if path.exists() else None
    changed=previous is not None and previous['identity']!=value
    if changed:
        archive=out/'scoring-history'/previous['epoch']
        archive.mkdir(parents=True,exist_ok=True)
        for tree in ('NEXT','C'):
            root=out/tree
            for name in ('scores','score-identities','c-score-identities','score-aliases','shared-identities','sec','blends',
                         'scoring-machine.json','next-scoring-machine.json','reading','reading-next.json','reading-probe.json','collect-complete.json','return'):
                src=root/name
                if src.exists():
                    dst=archive/tree/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.move(str(src),str(dst))
        # A new scoring epoch is never compared with old question rows.
    if previous is None or changed:
        atomic(path,dict(epoch=str(time.time_ns()),identity=value,controls_required=True,previous=previous))
    return changed


def record_job_start(durable, standin=False):
    """First job-script command, before restore; survive replacement of local OUT."""
    from kit.a4_common import _run_key, _restart_count, _slurm_times, _check_restart_count, _check_allocation_start
    now = time.time()
    job_id = os.environ.get('SLURM_JOB_ID')
    require(standin or job_id, 'SLURM_JOB_ID required for job start')
    job_id = job_id or 'standin'; restart = _restart_count()
    observed = {} if standin else _slurm_times(job_id)
    _check_restart_count(observed, restart)
    with locked(durable):
        path = durable/'job-starts'/(_run_key(job_id, restart)+'.json')
        if path.exists():
            value = json.loads(path.read_text())
            _check_allocation_start(value)
            return value
        # Environment first, then the scheduler's own StartTime, so later sacct Start matching works.
        started = os.environ.get('SLURM_JOB_START_TIME')
        if started is not None: started = float(started)
        elif observed.get('allocation_started_at') is not None: started = observed['allocation_started_at']
        else: started = now
        require(started <= now, 'allocation start is in the future')
        value = dict(slurm_job_id=job_id, slurm_restart_count=restart,
                     allocation_started_at=started, first_script_record_at=now)
        atomic(path, value); sync_dir(durable)
        return value


def job(a):
    require(a.action in ('save','restore','identity','path','job-start'),'unknown handoff action')
    if a.action=='job-start':
        print(record_job_start(a.durable, a.standin));return
    if a.action=='save':
        require(a.out.is_dir(),'missing OUT')
        print(save(a.out,a.durable));return
    if a.action=='restore':
        print(restore(a.durable,a.out,move_aside=getattr(a,"move_aside",True)));return
    if a.action=='path':
        # Save/restore verify data. Resolving the pointer must stay cheap.
        with locked(a.durable): generation=committed(a.durable,verify=False)
        require(generation is not None,'no committed durable generation')
        print(generation/'tree');return
    # Physical UUID plus host/environment, not the CUDA ordinal, defines a comparison epoch.
    import platform
    value=dict(uuid='CPU-STANDIN',host='CPU-STANDIN',environment='CPU-STANDIN') if a.standin else dict(
        uuid=subprocess.check_output(['nvidia-smi','-i',a.gpu,'--query-gpu=uuid','--format=csv,noheader'],text=True).strip(),
        host=platform.node(),environment=s.digest(dict(python=sys.executable,versions=t.runtime_versions())))
    require(bool(value['uuid']) and '\n' not in value['uuid'],'one physical scoring GPU required')
    changed=epoch(a.out,value)
    print('NEW SCORING EPOCH: rerun all base/reference controls and scoring rows before comparisons' if changed else 'SAME SCORING EPOCH / first job: run controls before comparisons')


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--durable',type=Path,required=True)
    p.add_argument('--action',choices=('save','restore','identity','path','job-start'),required=True);p.add_argument('--gpu',default='0');p.add_argument('--standin',action='store_true')
    p.add_argument("--move-aside",action="store_true",help="preserve differing local files beside OUT before restore")
    a=p.parse_args()
    try: job(a)
    except CorrectnessStop as exc:
        if a.action!='job-start': raise
        print('STOP: correctness:', exc); return 2
    return 0
if __name__=='__main__':raise SystemExit(main())
