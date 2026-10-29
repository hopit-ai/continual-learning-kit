"""Sample non-reclaimable job memory; retain raw cgroup counters and limits."""
from pathlib import Path
import os,threading


def number(path):
    try:
        value=Path(path).read_text().strip()
        return None if value=='max' else int(value)
    except (OSError,ValueError):return None


def pairs(path):
    try:return {k:int(v) for k,v in (line.split() for line in Path(path).read_text().splitlines())}
    except (OSError,ValueError):return {}


def hierarchy(root=Path('/sys/fs/cgroup'),membership=Path('/proc/self/cgroup'),mountinfo=Path('/proc/self/mountinfo')):
    """Map cgroup namespaces/mount roots, then walk visible ancestors."""
    root=Path(root);mount=root;leaf=root;version=2
    try:
        lines=membership.read_text().splitlines()
        unified=next((v.split(':',2)[2] for v in lines if v.startswith('0::')),None)
        if unified is None:
            version=1;relative=next(v.split(':',2)[2] for v in lines if 'memory' in v.split(':',2)[1].split(','));mount=root/'memory'
        else:relative=unified
        fsroot=Path('/')
        try:
            for line in mountinfo.read_text().splitlines():
                left,right=line.split(' - ',1);fields=left.split();kind=right.split()
                if kind[0]==('cgroup2' if version==2 else 'cgroup') and (version==2 or 'memory' in kind[-1].split(',')):
                    mount=Path(fields[4]);fsroot=Path(fields[3]);break
        except (OSError,ValueError,IndexError):pass
        try:leaf=mount/Path(relative).relative_to(fsroot)
        except ValueError:leaf=mount/relative.lstrip('/')
        if not leaf.is_dir():leaf=mount
    except (OSError,ValueError,StopIteration):pass
    ancestors=[]
    while leaf.is_relative_to(mount):
        ancestors.append(leaf)
        if leaf==mount:break
        leaf=leaf.parent
    return version,ancestors or [root]


def rss_sum(proc=Path('/proc')):
    total=0;count=0
    for path in Path(proc).glob('[0-9]*/status'):
        try:
            for line in path.read_text().splitlines():
                if line.startswith('VmRSS:'):total+=int(line.split()[1])*1024;count+=1
        except (OSError,ValueError):pass
    return (total if count else None),count


def sample(*,node=False,root=Path('/sys/fs/cgroup'),membership=Path('/proc/self/cgroup'),mountinfo=Path('/proc/self/mountinfo'),proc=Path('/proc')):
    version,paths=hierarchy(root,membership,mountinfo);leaf=paths[0]
    current_name,peak_name,limit_name=(('memory.current','memory.peak','memory.max') if version==2 else
        ('memory.usage_in_bytes','memory.max_usage_in_bytes','memory.limit_in_bytes'))
    limits=[dict(path=str(p/limit_name),bytes=number(p/limit_name)) for p in paths]
    # v1's huge sentinel also means unlimited; it must not hide a real ancestor.
    finite=[v['bytes'] for v in limits if v['bytes'] is not None and 0<v['bytes']<2**60]
    physical=os.sysconf('SC_PAGE_SIZE')*os.sysconf('SC_PHYS_PAGES')
    limit=min(finite) if finite else physical
    stat=pairs(leaf/'memory.stat');current=number(leaf/current_name);raw_peak=number(leaf/peak_name)
    events=[dict(path=str(p/('memory.events' if version==2 else 'memory.oom_control')),
        values=pairs(p/('memory.events' if version==2 else 'memory.oom_control'))) for p in paths]
    bounded=[i for i,v in enumerate(limits) if v['bytes']==limit] if finite else [0]
    scope_end=min(bounded) if bounded else 0
    kills=[v['values']['oom_kill'] for v in events[:scope_end+1] if 'oom_kill' in v['values']]
    file=stat.get('file',stat.get('cache'));without_file=max(0,current-file) if current is not None and file is not None else None
    if all(k in stat for k in ('anon','shmem','kernel')):
        used=sum(stat[k] for k in ('anon','shmem','kernel'));scope='cgroup-nonreclaimable';source=str(leaf/'memory.stat')+': anon+shmem+kernel'
    elif node and without_file is not None:
        used=without_file;scope='cgroup-current-minus-file';source=str(leaf)+': memory.current - file (old-kernel fallback)'
    else:
        used,count=rss_sum(proc);scope='proc-rss-sum';source=f'/proc/*/status RSS sum, {count} processes; shared pages conservatively counted per process'
    return dict(peak_bytes=used,current_nonreclaimable_bytes=used,scope=scope,source=source,
        raw_current_bytes=current,raw_peak_bytes=raw_peak,current_minus_file_bytes=without_file,stat=stat,
        limit_bytes=limit,limit_source='smallest finite cgroup/ancestor limit' if finite else 'labelled physical-memory fallback; no finite visible cgroup limit',
        limits=limits,oom_scope_paths=[e['path'] for e in events[:scope_end+1]],oom_kill=max(kills) if kills else None,oom_events=events)


def qualified(value,limit,headroom=.05):
    return (value.get('peak_bytes') is not None and value.get('scope') in ('cgroup-nonreclaimable','cgroup-current-minus-file','proc-rss-sum') and
        value.get('oom_kill')==0 and value.get('telemetry_complete',True) and value['peak_bytes']<=(1-headroom)*min(limit,value['limit_bytes']))


def merge_peak(previous,current):
    result=dict(current)
    for key in ('peak_bytes','raw_peak_bytes','oom_kill'):
        values=[v for v in (previous.get(key),current.get(key)) if v is not None]
        result[key]=max(values) if values else None
    raw=[v for v in (previous.get('raw_current_peak_bytes',previous.get('raw_current_bytes')),current.get('raw_current_bytes')) if v is not None]
    result['raw_current_peak_bytes']=max(raw) if raw else None
    result['limit_bytes']=min(previous['limit_bytes'],current['limit_bytes'])
    if previous.get('peak_bytes',-1) is not None and current.get('peak_bytes',-1) is not None and previous['peak_bytes']>current['peak_bytes']:
        result['peak_source']=previous.get('peak_source',previous['source'])
    else:result['peak_source']=current['source']
    # Missing telemetry after a qualified sample is a failure, not a held zero.
    result['telemetry_complete']=previous.get('telemetry_complete',True) and current.get('peak_bytes') is not None and current.get('oom_kill') is not None
    return result


class Monitor:
    def __init__(self,*,node=False):
        self.node=node;self.initial=sample(node=node);self.peak=dict(self.initial);self.stop=threading.Event()
        def watch():
            while not self.stop.wait(.1):self.sample()
        self.thread=threading.Thread(target=watch,daemon=True)
    def sample(self):
        self.peak=merge_peak(self.peak,sample(node=self.node));return dict(self.peak)
    def __enter__(self):self.thread.start();return self
    def __exit__(self,*args):self.sample();self.stop.set();self.thread.join()
