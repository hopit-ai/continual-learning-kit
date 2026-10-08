#!/usr/bin/env python3
"""Confirmed native base admission and owned temporary/cache storage, stdlib only."""
import argparse
from contextlib import contextmanager
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import sysconfig
import hashlib
import re
from urllib.parse import urlparse, unquote
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
RAY_JOB_TMP='/tmp/phase0-ray' # Small sessions in the site's job-private tmpfs, not wheel caches.
VLLM_JOB_TMP='/tmp' # Existing job-private directory; vLLM names tiny sockets with UUIDs.
JOB_TMP='/tmp/phase0-tmp' # Allocation-only scratch and multiprocessing listeners.
JOB_TMP_MIN_GIB=1
BASE_VERSIONS={'torch':'2.9.0+cu128','vllm':'0.12.0','verl':'0.7.0.dev*','transformers':'4.57.1','flash_attn':'2.8.3*','ray':'2.53.0','numpy':'1.26.4'}
CACHE_KEYS={'TMPDIR':'tmp','PIP_CACHE_DIR':'pip','XDG_CACHE_HOME':'xdg','TRITON_CACHE_DIR':'triton','VLLM_CACHE_ROOT':'vllm',
    'TORCHINDUCTOR_CACHE_DIR':'torchinductor','TORCH_EXTENSIONS_DIR':'torch-extensions','CUDA_CACHE_PATH':'cuda','TORCH_HOME':'torch',
    'NUMBA_CACHE_DIR':'numba','FLASHINFER_WORKSPACE_BASE':'flashinfer'}


def validate_base_versions(versions):
    for name,pin in BASE_VERSIONS.items():
        value=versions.get(name)
        if not isinstance(value,str) or not (value.startswith(pin[:-1]) if pin.endswith('*') else value==pin):
            raise ValueError('host trainer dependency '+name+': expected '+pin+', found '+str(value)+'; keep the existing base unchanged and return this setup blocker')
    return versions


def installed_versions():
    result={}
    for name in BASE_VERSIONS:
        try:result[name]=metadata.version(name)
        except metadata.PackageNotFoundError:result[name]=None
    return result


def distribution_name(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def distribution_inventory(paths=None,*,skipped=None,allow_owned_verl=None):
    """Skip and record interrupted metadata; refuse ambiguous version inventories."""
    rows=[];seen={};skipped=[] if skipped is None else skipped
    for dist in metadata.distributions(**({'path':list(map(str,paths))} if paths is not None else {})):
        raw_name=dist.metadata.get('Name');info=str(getattr(dist,'_path',''))
        if not raw_name or raw_name.startswith('~') or Path(info).name.startswith('~'):
            skipped.append({'name':raw_name,'version':dist.version,'dist_info':info,
                'reason':'missing Name' if not raw_name else "interrupted-pip '~' prefix"})
            continue
        name=distribution_name(raw_name)
        row={'name':name,'version':dist.version,'location':str(Path(dist.locate_file('')).resolve())}
        direct=dist.read_text('direct_url.json')
        if direct:
            doc=json.loads(direct);url=urlparse(doc.get('url',''))
            if doc.get('dir_info',{}).get('editable') and url.scheme=='file' and url.netloc in ('','localhost'):
                row['editable_source']=str(Path(unquote(url.path)).resolve())
        for previous in seen.get(name,[]):
            if previous['version']!=row['version']:
                owned=str(Path(allow_owned_verl).resolve()) if allow_owned_verl is not None else None
                exception=name=='verl' and owned is not None and ((previous.get('editable_source')==owned) != (row.get('editable_source')==owned))
                if not exception:raise ValueError('duplicate distribution '+name+' has differing versions '+previous['version']+' and '+row['version']+'; retain this setup blocker and repair the base metadata separately')
        if row not in rows:rows.append(row)
        seen.setdefault(name,[]).append(row)
    return sorted(rows,key=lambda r:(r['name'],r['version'],r['location']))


def base_constraints(inventory):
    # verl is independently bound to the owned, pinned SDPO commit.
    return {distribution_name(r['name']):r['version'] for r in inventory if distribution_name(r['name'])!='verl'}


def validate_shadowing(base,own,owned_sdpo):
    versions={distribution_name(r['name']):r['version'] for r in base}
    for row in own:
        name=distribution_name(row['name'])
        if name=='verl':
            if Path(row.get('editable_source','')).resolve()!=Path(owned_sdpo).resolve():
                raise ValueError('trainer verl must come from the owned SDPO editable checkout')
            continue
        if name in versions and row['version']!=versions[name]:
            raise ValueError('trainer shadows native base distribution '+name+': '+versions[name]+' -> '+row['version'])


def write_trainer_inventory(work,visible,own,prefix,*,skipped=None):
    from kit.runner import write_durably
    doc={'schema':'v4-trainer-inventory.v1','prefix':str(prefix),'visible':visible,'own':own,'skipped_distributions':skipped or []}
    write_durably(Path(work)/'v4/report-phase0/environment-build/trainer-inventory.json',doc)
    return doc


def audit_trainer(work):
    work=Path(work).resolve();prefix=work/'phase0-envs/trainer'
    if Path(sys.prefix).resolve()!=prefix:raise ValueError('trainer inventory needs the prepared trainer interpreter')
    base=json.loads((work/'v4/report-phase0/environment-build/base-environment.json').read_text())
    skipped=[]
    own=distribution_inventory(sorted(set(sysconfig.get_path(k) for k in ('purelib','platlib'))),skipped=skipped)
    visible=distribution_inventory(skipped=skipped,allow_owned_verl=work/'phase0-source/SDPO')
    doc=write_trainer_inventory(work,visible,own,sys.prefix,skipped=skipped)
    validate_shadowing(base['inventory'],own,work/'phase0-source/SDPO')
    return doc


def planning_storage(work,launch):
    # Retain owned storage: job_container/tmpfs has no guaranteed cache capacity.
    temp=launch.get('TMPDIR','')
    if not temp:raise ValueError('prepared allocation TMPDIR is missing')
    sockets=socket_path_check(launch)
    from kit.v4_phase0_storage import phase0_disk_requirement
    disk=phase0_disk_requirement(launch)
    return {**disk,**sockets,'tmpdir':temp,
            'free_bytes':free_space(work,disk['required_gib'])}


def socket_base_check(path,label,key,suffix):
    """One pathname bound for all three allocation socket bases."""
    if not path or not os.path.isabs(path) or len(os.fsencode(path))+suffix>107:
        raise ValueError(label+' socket path exceeds the 107-byte limit or has no absolute base ('+key+' plus '+str(suffix)+' bytes); refuse before GPU rows')
    return {'socket_suffix_bytes':suffix,'maximum_socket_bytes':107}


def vllm_path_check(path):
    # get_open_zmq_ipc_path: ipc://<base>/<uuid4>. ipc:// is a protocol prefix,
    # not part of the Unix pathname; reserve '/' plus the 36-byte UUID.
    suffix=37
    return {'vllm_rpc_base_path':path,**socket_base_check(path,'vLLM','VLLM_RPC_BASE_PATH',suffix)}


def socket_path_check(launch):
    return {'multiprocessing':{'tmpdir':launch.get('TMPDIR',JOB_TMP),
                              **socket_base_check(launch.get('TMPDIR',JOB_TMP),'multiprocessing','TMPDIR',32)},
            'ray':ray_path_check(launch.get('RAY_TMPDIR',RAY_JOB_TMP)),
            'vllm':vllm_path_check(launch.get('VLLM_RPC_BASE_PATH',VLLM_JOB_TMP))}


def ray_path_check(path):
    return {'ray_tmpdir':path,**socket_base_check(path,'Ray','RAY_TMPDIR',68)}


def job_tmp_check(launch):
    """Check bounded small allocation scratch before Slurm steps or CUDA probes."""
    path=launch.get('TMPDIR')
    if not path or os.environ.get('TMPDIR')!=path:raise ValueError('allocation TMPDIR is missing or differs from the frozen prepared environment')
    sockets=socket_path_check(launch)
    if not Path(path).is_dir():raise ValueError('allocation TMPDIR directory is missing; the payload must create '+path)
    free=shutil.disk_usage(path).free
    if free<JOB_TMP_MIN_GIB*1024**3:
        raise ValueError('job-private TMPDIR '+path+' has '+str(round(free/1024**3,2))+' GiB free; at least '+str(JOB_TMP_MIN_GIB)+' GiB is required for multiprocessing metadata, torchrun scratch and temporary FinQA/tokenizer/scorer copies')
    return {'tmpdir':path,'minimum_gib':JOB_TMP_MIN_GIB,'free_bytes':free,'sockets':sockets}


def pip_check_output(raw):
    if isinstance(raw,bytes):raw=raw.decode(errors='replace')
    return sorted(line.strip() for line in raw.splitlines() if line.strip())


def pip_check_difference(base,trainer):
    return sorted(set(trainer)-set(base)-{'No broken requirements found.'})


def verify_pip_checks(base,trainer,base_raw,trainer_raw):
    """Re-derive new errors from both archived outputs, never trust the stored delta."""
    old=pip_check_output(base_raw);new=pip_check_output(trainer_raw)
    delta=pip_check_difference(old,new)
    if delta:raise ValueError('new pip check problems: '+'; '.join(delta))
    if base['output']!=old or trainer['output']!=new or trainer['new_lines']!=delta:
        raise ValueError('pip check output/difference binding differs')
    return {'base_output':old,'trainer_output':new,'new_lines':delta}


def free_space(path,gib):
    path=Path(path).absolute()
    while not path.exists():path=path.parent
    free=shutil.disk_usage(path).free
    if free<gib*1024**3:raise ValueError(f'owned storage free space {free/1024**3:.1f} GiB is below {gib} GiB required at {path}; choose a larger TASK_ROOT/WORK, never login /tmp')
    return free


def storage_environment(work,*,task_root=None,min_gib=0,allocation=False):
    work=Path(work).resolve();task=Path(task_root or os.environ.get('TASK_ROOT') or work.parent).resolve()
    if task==work or task.is_relative_to(work):raise ValueError('TASK_ROOT/HF cache must be outside WORK')
    free_space(work,min_gib)
    env={key:str(work/'phase0-cache'/folder) for key,folder in CACHE_KEYS.items()}
    env.update(RAY_TMPDIR=RAY_JOB_TMP,VLLM_RPC_BASE_PATH=VLLM_JOB_TMP,HF_HOME=str(task/'hf-cache'),HF_HUB_CACHE=str(task/'hf-cache/hub'),HF_ASSETS_CACHE=str(task/'hf-cache/assets'),
        HUGGINGFACE_HUB_CACHE=str(task/'hf-cache/hub'),TRANSFORMERS_CACHE=str(task/'hf-cache/hub'),
        PIP_DISABLE_PIP_VERSION_CHECK='1',PIP_DEFAULT_TIMEOUT='30',PIP_RETRIES='1',GIT_TERMINAL_PROMPT='0')
    for key in (*CACHE_KEYS,'HF_HOME','HF_HUB_CACHE','HF_ASSETS_CACHE'):Path(env[key]).mkdir(parents=True,exist_ok=True)
    # Freeze the allocation route on CPU, but create its short directory only in
    # the batch payload, under the site's job-private job_container/tmpfs /tmp.
    if allocation:env['TMPDIR']=JOB_TMP
    return env


@contextmanager
def storage(work,**kwargs):
    env=storage_environment(work,**kwargs);old={key:os.environ.get(key) for key in env};old_temp=tempfile.tempdir
    os.environ.update(env);tempfile.tempdir=env['TMPDIR']
    try:yield env
    finally:
        for key,value in old.items():
            if value is None:os.environ.pop(key,None)
            else:os.environ[key]=value
        tempfile.tempdir=old_temp


def base_environment(work):
    """Record native metadata and pre-existing pip conflicts before owned installs."""
    from kit.runner import write_durably,bounded_command
    folder=Path(work)/'v4/report-phase0/environment-build';folder.mkdir(parents=True,exist_ok=True)
    doc={'schema':'v4-host-base.v1','route':'host-venv','python':sys.executable,'prefix':sys.prefix,
        'base_prefix':sys.base_prefix,'is_venv':sys.prefix!=sys.base_prefix,'versions':installed_versions(),
        'inventory':[],'skipped_distributions':[],
        'site_packages':sorted(set(sysconfig.get_path(key) for key in ('purelib','platlib')))}
    try:
        doc['inventory']=distribution_inventory(skipped=doc['skipped_distributions'])
        log=folder/'base-pip-check.log';argv=[sys.executable,'-m','pip','check']
        with storage(work):
            result=bounded_command(argv,timeout=60,env={**os.environ,'CUDA_VISIBLE_DEVICES':'','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'},log=log)
        doc['pip_check']={'argv':argv,**result,'output':pip_check_output(log.read_bytes()),'log_sha256':hashlib.sha256(log.read_bytes()).hexdigest()}
        if result['returncode'] not in (0,1) or result.get('failure_type') not in (None,'cpu_exit'):
            raise ValueError('base pip check could not finish; retain this setup blocker')
        validate_base_versions(doc['versions'])
    except (ValueError,OSError) as exc:
        doc['setup_blocker']=str(exc)
        (folder.parent/'setup-blocker.txt').write_text(str(exc)+'\n')
        raise
    finally:write_durably(folder/'base-environment.json',doc)
    return doc


def base_bridge_text(paths):
    # Process the same trusted .pth files the existing native base already loads.
    return ''.join('import site; site.addsitedir('+repr(path)+')\n' for path in paths)


def bind_trainer_base(work,trainer,base):
    """Nested venvs inherit global sites, not parent-venv sites; bridge the admitted base explicitly."""
    from kit.runner import write_durably
    enabled=sys.prefix!=sys.base_prefix
    folder=Path(trainer)/'lib'/('python'+'.'.join(map(str,sys.version_info[:2])))/'site-packages'
    path=folder/'phase0-host-base.pth';text=base_bridge_text(base['site_packages'])
    if enabled:
        folder.mkdir(parents=True,exist_ok=True);path.write_text(text)
    doc={'enabled':enabled,'path':str(path) if enabled else None,'base_site_packages':base['site_packages'],
         'sha256':hashlib.sha256(text.encode()).hexdigest() if enabled else None}
    write_durably(Path(work)/'v4/report-phase0/environment-build/base-inheritance.json',doc)
    return doc


def check_network(urls):
    """Fail within ten seconds per allowed login-node endpoint, never probe HF."""
    import urllib.request
    for url in urls:
        try:
            with urllib.request.urlopen(urllib.request.Request(url,method='HEAD'),timeout=10) as response:
                if response.status>=400:raise ValueError('HTTP '+str(response.status))
        except Exception as exc:
            action=('Retain the setup blocker and return it; prepare cannot run under SLURM_JOB_ID.'
                    if urlparse(url).hostname in ('pypi.org','download.pytorch.org')
                    else 'Stop; use the same bounded zero-GPU compute job pattern for this step.')
            raise ValueError('CPU network unavailable at '+url+': '+str(exc)+'. '+action) from exc


def network_command(work,argv,*,seconds=600,urls=()):
    from kit.runner import bounded_command
    with storage(work,min_gib=30 if 'pip' in argv else 0):
        check_network(urls)
        folder=Path(work)/'v4/report-phase0/network';folder.mkdir(parents=True,exist_ok=True)
        log=folder/('command-'+str(len(list(folder.glob('command-*.log'))))+'.log')
        result=bounded_command(argv,timeout=seconds,env=os.environ.copy(),log=log)
        if result['returncode']:raise ValueError('CPU staging failed: '+str(result.get('failure_type'))+'; see '+str(log)+'. If blocked, retain the setup blocker and return it.')
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('operation',choices=('storage','network','audit-trainer','ray-check','socket-check'))
    p.add_argument('--work',type=Path,required=True);p.add_argument('--task-root',type=Path);p.add_argument('--min-gib',type=int,default=0)
    p.add_argument('--url',action='append',default=[]);p.add_argument('--seconds',type=int,default=600)
    p.add_argument('--allocation',action='store_true')
    argv=list(sys.argv[1:] if argv is None else argv);command=[]
    if '--' in argv:
        at=argv.index('--');argv,command=argv[:at],argv[at+1:]
    a=p.parse_args(argv)
    try:
        if a.operation=='storage':
            for key,value in storage_environment(a.work,task_root=a.task_root,min_gib=a.min_gib,allocation=a.allocation).items():print('export '+key+'='+shlex.quote(value))
        elif a.operation=='audit-trainer':audit_trainer(a.work)
        elif a.operation=='ray-check':
            from kit.runner import write_durably
            doc=ray_path_check(os.environ.get('RAY_TMPDIR',RAY_JOB_TMP))
            write_durably(a.work/'v4/report-phase0/ray-path-check.json',doc)
        elif a.operation=='socket-check':
            from kit.runner import write_durably
            write_durably(a.work/'v4/report-phase0/socket-path-check.json',socket_path_check(os.environ))
        else:
            if not command:raise ValueError('network operation needs a command after --')
            network_command(a.work,command,seconds=a.seconds,urls=a.url)
    except (ValueError,OSError) as exc:print('STOP: '+str(exc),file=sys.stderr);return 2
    return 0
if __name__=='__main__':raise SystemExit(main())
