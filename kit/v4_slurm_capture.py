"""CPU-only phase-0 Slurm capture, selective masking and frozen admission proof."""
from __future__ import annotations
import json
import os
from pathlib import Path
import re
import tempfile
import sys
from kit.runner import bounded_command,write_durably
from kit.p4_frozen import seal
from kit.v4_prepare import digest,file_receipt,verify_files
BASE='v4/report-phase0/presend/'
NAMES=('version.txt','config.txt','partition.txt','qos.txt','cgroup.conf')
SCHEMA='v4-phase0-slurm-capture.v1'
KIT=Path(__file__).resolve().parent


def require(ok,reason):
    if not ok:raise ValueError(reason)


def commands(partition,qos,account):
    return [['scontrol','--version'],['scontrol','show','config'],['scontrol','show','partition',partition],
            ['sacctmgr','-n','-P','show','qos','format=Name,PreemptMode,GraceTime,Preempt']]


def controller_cgroup(config):
    """Prefer the controller's effective settings, including configless sites."""
    match=re.search(r'^\s*Cgroup Support Configuration\s*:?\s*\n(.*?)(?=^\s*[^=\n]+ Configuration\s*:?\s*$|\Z)',config,re.M|re.S)
    if not match:return ''
    return ''.join(line.strip()+'\n' for line in match[1].splitlines() if re.match(r'\s*\w+\s*=',line))


def read_cgroup(config,*,slurm_conf=None,etc_dir=Path('/etc/slurm')):
    """Read effective controller settings, then CgroupConf, SLURM_CONF, /etc/slurm."""
    observed={'file':'cgroup.conf','argv':['read-cgroup-config'],'returncode':0,
              'failure_type':None,'seconds':0,'timeout_seconds':120}
    text=controller_cgroup(config)
    if text:return text,{**observed,'source':'controller'}
    configured=re.search(r'^\s*CgroupConf\s*=\s*(.+?)\s*$',config,re.M)
    paths=[]
    if configured and configured[1] not in ('(null)','NONE'):paths.append(Path(configured[1]))
    conf=slurm_conf if slurm_conf is not None else os.environ.get('SLURM_CONF')
    if not conf:
        setting=re.search(r'^\s*(?:SLURM_CONF|SlurmConf)\s*=\s*(.+?)\s*$',config,re.M)
        if setting:conf=setting[1]
    if conf:paths.append(Path(conf).parent/'cgroup.conf')
    paths.append(Path(etc_dir)/'cgroup.conf')
    tried=[]
    for path in dict.fromkeys(paths):
        tried.append(str(path))
        try:
            if not path.is_file() or path.stat().st_size>1024*1024:continue
            text=path.read_text()
            if text.strip():return text,{**observed,'source':str(path),'tried':tried}
        except (OSError,UnicodeError):continue
    return '',{**observed,'source':'unavailable','tried':tried,'returncode':1,
               'failure_type':'cgroup_unavailable'}


def mask_observations(raw,*,partition,qos,account):
    """Replace only complete named identity VALUES; never match identity substrings."""
    userkeys=r'UserId|UserName|User|GroupId|SlurmUser|SlurmdUser|AllowUsers|DenyUsers'
    nodekeys=r'(?:AccountingStorageHost|AccountingStorageBackupHost|SlurmctldHost|SlurmctldAddr|SlurmctldPrimaryHost|SlurmctldBackupHost|ControlMachine|ControlAddr|BackupController|BackupAddr|NodeName|NodeList|Nodes|NodeAddr|NodeHostname|BatchHost|AllocNode|SrunHost)(?:\[\d+\])?(?::(?:Pid|Sid))?'
    special={'DEFAULT','NONE','None','N/A','ALL','(null)','<USER>','<NODE>','<IP>'}
    def identity(match):
        key,value=match[1],match[2]
        label='<USER>' if re.match(r'(?:'+userkeys+r')\s*=',key,re.I) else '<NODE>'
        if value.isdigit() or value in special:return match[0]
        if label=='<USER>':
            return key+','.join(part if part.isdigit() or part in special else '<USER>'+('('+part.split('(',1)[1] if '(' in part else '') for part in value.split(','))
        import ipaddress
        try:ipaddress.ip_address(value);return key+'<IP>'
        except ValueError:pass
        tail=''
        if '(' in value:
            inner=value.split('(',1)[1].rstrip(')')
            try:ipaddress.ip_address(inner);inner='<IP>'
            except ValueError:inner='<NODE>'
            tail='('+inner+')'
        elif value.count(':')==1 and value.rsplit(':',1)[1].isdigit():tail=':'+value.rsplit(':',1)[1]
        return key+label+tail
    pattern=r'(\b(?:'+userkeys+'|'+nodekeys+r')\s*=\s*)([^\s;]+)'
    def mask(text):
        text=re.sub(pattern,identity,text,flags=re.I)
        # This controller status sentence has a named host position, not an
        # arbitrary token replacement in other lines or paths.
        return re.sub(r'(\bSlurmctld\((?:primary|backup\d*)\)\s+at\s+)(\S+)(\s+is\s+)',r'\1<NODE>\3',text,flags=re.I)
    return {name:text if name=='qos.txt' else mask(text) for name,text in raw.items()}


def capture(work,partition,qos,account):
    """Capture before downloads/builds; no sbatch or GPU allocation is possible."""
    require(not os.environ.get('SLURM_JOB_ID'),'capture must run CPU-only on the login node outside any allocation')
    for key,value in (('PARTITION',partition),('QOS',qos),('ACCOUNT',account)):
        require(isinstance(value,str) and re.fullmatch(r'[A-Za-z0-9_.-]+',value),'missing/invalid '+key)
    work=Path(work)
    require(not work.is_symlink(),'capture WORK must be owned, not a symlink')
    require(not work.exists() or work.is_dir() and not any(work.iterdir()),'capture requires a fresh empty WORK; retain earlier evidence')
    work=work.resolve();folder=work/BASE;paste=folder/'paste';paste.mkdir(parents=True)
    argv=commands(partition,qos,account);observations=[];raw={}
    from kit.v4_phase0_site import storage_environment
    env={**os.environ,**storage_environment(work),'CUDA_VISIBLE_DEVICES':''}
    # Original stdout is private to this 0700 temporary directory, removed on
    # every exit. Only selectively masked bytes are published under WORK.
    with tempfile.TemporaryDirectory(prefix='v4-slurm-capture-',dir=env['TMPDIR']) as temp:
        for name,command in zip(NAMES,argv):
            log=Path(temp)/name
            observed=bounded_command(command,timeout=120,env=env,log=log)
            raw[name]=log.read_text();observations.append({'file':name,'argv':command,**observed})
        raw['cgroup.conf'],cgroup=read_cgroup(raw['config.txt'])
        observations.append(cgroup)
        masked=mask_observations(raw,partition=partition,qos=qos,account=account)
        for name,text in masked.items():
            target=paste/name
            with target.open('x') as handle:handle.write(text)
        checker=[sys.executable,str(KIT/'p4_contain.py'),'check','--work',str(work),'--out',temp,'--from-file',str(paste),'--qos',qos]
        checked=bounded_command(checker,timeout=120,env=env,log=Path(temp)/'check.log')
        log=(Path(temp)/'check.log').read_text()
        (folder/'check.log').write_text(mask_observations({'check.log':log,**raw},partition=partition,qos=qos,account=account)['check.log'])
        source=Path(temp)/'containment-presend.json'
        doc=json.loads(source.read_text()) if source.is_file() else {'schema':'kit-v4-containment-presend.v1','ok':False,'problems':['check --from-file did not produce a receipt'],'v4_qualified':False}
        problems=[f"capture {o['file']}: {o['failure_type'] or 'cpu_exit'} (exit {o['returncode']})" for o in observations if o['returncode']]
        if checked['returncode'] and doc.get('ok'):problems.append('check --from-file command failed: '+str(checked['failure_type']))
        doc=seal({**doc,'ok':bool(doc.get('ok') and not problems),'problems':[*doc.get('problems',[]),*problems]})
        write_durably(folder/'containment-presend.json',doc)
        write_durably(folder/'check-command.json',{'argv':checker,**checked,'log_sha256':digest(folder/'check.log')})
    from kit.p4_contain import values,duration_seconds
    try:site_minutes=duration_seconds(values(masked['partition.txt'])['MaxTime'])//60
    except (ValueError,KeyError):site_minutes=None
    binding=file_receipt(paste.iterdir());write_durably(folder/'paste-files.json',binding)
    receipt=seal({'schema':SCHEMA,'ok':doc['ok'],'before_allocation':True,'synthetic':False,
        'site':{'partition':partition,'qos':qos,'account':account},'site_minutes':site_minutes,
        'masking':'named user/node/host/IP values only; versions, paths, safety and assignment values retained',
        'observations':observations,'presend_sha256':digest(folder/'containment-presend.json'),
        'files':{name:digest(paste/name) for name in NAMES}})
    write_durably(folder/'capture.json',receipt)
    if not doc['ok']:
        from kit.collect import main as collect
        from kit.v4_phase0 import read_archive
        archive=Path(str(work)+'-return.tar.gz')
        require(collect(['--work',str(work),'--out',str(archive)])==0,'failed to collect capture refusal')
        write_durably(Path(str(work)+'-reading.json'),read_archive(archive))
    return receipt


def verify_capture(work):
    root=Path(work)/BASE;path=root/'capture.json'
    require(path.is_file(),'phase0 capture is absent (replaces paste; records partition/qos/account)')
    doc=json.loads(path.read_text());validate_capture(doc,{name:(root/'paste'/name).read_bytes() for name in NAMES},(root/'containment-presend.json').read_bytes())
    require(doc['ok'] and not doc.get('synthetic'),'Slurm capture failed or synthetic; stop before GPU planning')
    verify_files(json.loads((root/'paste-files.json').read_text()))
    return doc


def validate_capture(doc,files,presend):
    """Same immutable CPU proof for prepare/planning and archive-only readers."""
    require(doc==seal(doc) and doc.get('schema')==SCHEMA and doc.get('before_allocation') is True,'Slurm capture proof changed')
    site=doc['site'];expected=commands(site['partition'],site['qos'],site['account'])+[['read-cgroup-config']]
    require(len(doc['observations'])==5,'Slurm capture command inventory differs')
    for name,argv,observed in zip(NAMES,expected,doc['observations']):
        require(observed['argv']==argv and observed['file']==name and observed['timeout_seconds']==120,'Slurm capture argv/deadline differs')
        require(digest_bytes(files[name])==doc['files'][name],'Slurm capture bytes changed: '+name)
    require(digest_bytes(presend)==doc['presend_sha256'],'Slurm capture check binding differs')
    checked=json.loads(presend)
    require(checked==seal(checked),'Slurm capture check receipt changed')
    require(doc.get('ok')==checked.get('ok') and (not doc['ok'] or all(o['returncode']==0 for o in doc['observations'])),'Slurm capture command outcome differs')
    if doc['ok']:require(checked['preemption']['qos_name']==site['qos'],'Slurm capture checked QOS differs')
    from kit.p4_contain import values,duration_seconds
    try:site_minutes=duration_seconds(values(files['partition.txt'].decode())['MaxTime'])//60
    except (ValueError,KeyError):site_minutes=None
    require(doc['site_minutes']==site_minutes,'Slurm capture MaxTime differs')
    return doc


def digest_bytes(raw):
    import hashlib
    return hashlib.sha256(raw).hexdigest()
