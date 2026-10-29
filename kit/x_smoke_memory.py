"""Modal aggregate-cgroup host gate; summed process RSS is diagnostic only.

The sampled peak of usage (including shared pages once) is bounded by 95% of
both the registered reservation and the effective visible cgroup limit. Missing
aggregate telemetry is UNMEASURED and never replaced with RSS or a kernel peak.
This module does not change the Slurm/node measure in x_memory.
"""
from pathlib import Path
import threading,time
from kit import x_memory
from kit.x_smoke_settings import MEMORY_GIB

SCOPE='modal-cgroup-aggregate-usage'
SAMPLE_SECONDS=.1


def sample(proc=Path('/proc'),*,root=Path('/sys/fs/cgroup'),membership=Path('/proc/self/cgroup'),mountinfo=Path('/proc/self/mountinfo')):
    root=Path(root)
    version,paths=x_memory.hierarchy(root,membership,mountinfo)
    # Some sandboxes expose controller files at the mount root, without usable
    # membership metadata. Prefer the resolved leaf, then visible mount roots.
    candidates=[(version,paths),(1,[root/'memory']),(2,[root]),(1,[root])]
    current=None
    for version,paths in candidates:
        current_name,limit_name,peak_name,event_name=(
            ('memory.current','memory.max','memory.peak','memory.events') if version==2 else
            ('memory.usage_in_bytes','memory.limit_in_bytes','memory.max_usage_in_bytes','memory.oom_control'))
        source=paths[0]/current_name
        current=x_memory.number(source)
        if current is not None and current>=0:break
    else:current=None
    limits=[]
    for p in paths:
        try:raw=(p/limit_name).read_text().strip()
        except OSError:raw=''
        limits.append(dict(path=str(p/limit_name),bytes=x_memory.number(p/limit_name),raw=raw))
    finite=[v['bytes'] for v in limits if v['bytes'] is not None and 0<v['bytes']<2**60]
    effective=min(finite) if finite else None
    # Unlimited cgroups are bounded by the reservation; absent limits cannot
    # establish an effective limit and leave the gate unmeasured.
    limit_available=bool(finite) or any(v['raw']=='max' or (v['bytes'] or 0)>=2**60 for v in limits)
    events=[dict(path=str(p/event_name),values=x_memory.pairs(p/event_name)) for p in paths]
    kills=[v['values']['oom_kill'] for v in events if 'oom_kill' in v['values']]
    processes=[]
    for path in Path(proc).glob('[0-9]*/status'):
        try:
            for line in path.read_text().splitlines():
                if line.startswith('VmRSS:'):
                    processes.append(dict(pid=int(path.parent.name),resident_bytes=int(line.split()[1])*1024))
        except (OSError,ValueError):pass  # a process may exit during the read
    rss=sum(p['resident_bytes'] for p in processes) if processes else None
    complete=current is not None and limit_available
    return dict(scope=SCOPE,source=str(source),cgroup_version=version,
        peak_bytes=current,current_bytes=current,raw_current_bytes=current,raw_current_peak_bytes=current,
        raw_peak_bytes=x_memory.number(paths[0]/peak_name),stat=x_memory.pairs(paths[0]/'memory.stat'),
        limit_bytes=min(MEMORY_GIB*1024**3,effective) if effective is not None else MEMORY_GIB*1024**3,
        effective_cgroup_limit_bytes=effective,limits=limits,
        limit_source='minimum of registered Modal 64 GiB reservation and finite cgroup/ancestor limits',
        oom_kill=max(kills) if kills else None,oom_events=events,oom_scope_paths=[e['path'] for e in events],
        oom_measure='available' if kills else 'unavailable; not inferred as zero',
        rss_sum_bytes=rss,rss_sum_peak_bytes=rss,processes=processes,rss_peak_processes=processes,
        rss_measure='diagnostic only; shared pages counted per process',
        telemetry_complete=complete,measurement_status='MEASURED' if complete else 'UNMEASURED',sampled=True)


def merge_peak(previous,current):
    result=dict(current)
    for key in ('peak_bytes','raw_peak_bytes','raw_current_peak_bytes','rss_sum_peak_bytes','oom_kill'):
        values=[v for v in (previous.get(key),current.get(key)) if v is not None]
        result[key]=max(values) if values else None
    result['limit_bytes']=min(previous['limit_bytes'],current['limit_bytes'])
    old=previous.get('peak_bytes')
    result['peak_source']=previous.get('peak_source',previous['source']) if old is not None and (current.get('peak_bytes') is None or old>current['peak_bytes']) else current['source']
    if (previous.get('rss_sum_peak_bytes') or 0)>(current.get('rss_sum_peak_bytes') or 0):
        result['rss_peak_processes']=previous.get('rss_peak_processes',previous.get('processes'))
    result['telemetry_complete']=previous.get('telemetry_complete',False) and current.get('telemetry_complete',False) and current.get('peak_bytes') is not None
    result['measurement_status']='MEASURED' if result['telemetry_complete'] else 'UNMEASURED'
    return result


def qualified(value,limit,headroom=.05):
    return (value.get('scope')==SCOPE and value.get('telemetry_complete',False) and
        value.get('sampled') is True and value.get('peak_bytes') is not None and
        value.get('oom_kill') in (None,0) and
        value['peak_bytes']<=(1-headroom)*min(limit,value['limit_bytes']))


def gate(value,limit):
    measured=value.get('scope')==SCOPE and value.get('telemetry_complete',False) and value.get('sampled') is True and value.get('peak_bytes') is not None
    passed=qualified(value,limit)
    return dict(status=('PASS' if passed else 'FAIL') if measured else 'UNMEASURED',passed=passed,
        samples=value.get('samples'),max_gap_seconds=value.get('max_gap_seconds'))


class Monitor:
    def __init__(self):
        self.last_sample=time.monotonic();self.initial=sample()
        self.peak=dict(self.initial,samples=1,max_gap_seconds=0.);self.stop=threading.Event();self.lock=threading.Lock()
        def watch():
            while not self.stop.wait(SAMPLE_SECONDS):self.sample()
        self.thread=threading.Thread(target=watch,daemon=True)
    def sample(self):
        value=sample()
        with self.lock:
            now=time.monotonic();gap=now-self.last_sample;self.last_sample=now
            self.peak=dict(merge_peak(self.peak,value),samples=self.peak['samples']+1,
                max_gap_seconds=max(self.peak['max_gap_seconds'],gap))
            return dict(self.peak)
    def include(self,value):
        with self.lock:
            if value is None:
                self.peak.update(telemetry_complete=False,measurement_status='UNMEASURED')
            else:
                self.peak=dict(merge_peak(self.peak,value),samples=self.peak['samples'],max_gap_seconds=self.peak['max_gap_seconds'])
            return dict(self.peak)
    def __enter__(self):self.thread.start();return self
    def __exit__(self,*args):self.stop.set();self.thread.join();self.sample()
