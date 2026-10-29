"""Fail-closed P2-only node gate; controls/A never require its full-path pass."""
from pathlib import Path
from types import SimpleNamespace as NS
import hashlib,json,os,subprocess,time,shutil
from kit.s_common import ROOT,read,atomic,sha,require
from kit.x_p2_state import validate_geometry

SETTINGS=ROOT/'kit/resources/p2/node-limits.json'


def registered_tier2(arm):
    """Only reviewed tier-2 identities share the scored P2 memory profile."""
    from kit.x_tier2 import ARMS2,identity as tier2_identity
    return arm in ARMS2 and tier2_identity(arm)['p2']


def contains_p2(a,doc):
    """A future named/explicit stacking arm must not bypass the common gate."""
    arm=a.arm
    words=arm.lower().replace('+','-').replace('_','-').split('-')
    declared=doc.get('arm_variants',{}).get(arm)
    return (registered_tier2(arm) or arm=='p' and doc.get('p_variant')=='p2-l2-init' or 'p2' in words or
        arm in doc.get('p2_arms',[]) or declared=='p2-l2-init' or
        getattr(a,'p_variant',None)=='p2-l2-init' or getattr(a,'variant',None)=='p2-l2-init')


def identity(out,doc,seed,calibration_sha=None):
    from kit.x_common import incoming
    return dict(binding_sha256=sha(out/'binding.json'),incoming=incoming(doc,seed),anchor=doc['p2_anchor'],
        seed=seed,configuration=doc['configuration'],settings_sha256=sha(SETTINGS),calibration_sha256=calibration_sha,
        implementation=doc['implementation'],strategy='fsdp2',world_size=8,model_revision=doc['p2_anchor']['revision'])


def runtime_identity(env,cmd,path,value):
    atomic(path,value)
    for k,v in dict(X_P2_NODE_IDENTITY=str(path),X_P2_NODE_IDENTITY_SHA=sha(path)).items():
        env[k]=v;cmd.append('+ray_kwargs.ray_init.runtime_env.env_vars.'+k+'='+json.dumps(v))
    return cmd


def check_rank(value,expected,settings,*,measurement=False,completed_updates=None):
    require(value['schema']=='rung-p2-node-rank.v1' and value['identity']==expected and value['world_size']==8,'node feasibility identity/width differs')
    require(value['fp32_anchor'] and value['teacher_included'] and value['transients_included_asserted'] and value['finite_state'],'node feasibility missing state coverage')
    if measurement:return
    require(value['completed_updates']==(settings['disposable_updates'] if completed_updates is None else completed_updates) and value['optimizer_materialised'] and value['event']=='checkpoint-complete','full-path feasibility needs materialised optimizer and final checkpoint transients')
    gib=1024**3;g,h=value['gpu'],value['host']
    require(g['total_bytes']>=settings['gpu_min_total_gib']*gib and g['allocated_bytes']<=settings['gpu_max_allocated_gib']*gib and g['reserved_bytes']<=g['total_bytes']-settings['gpu_reserved_headroom_gib']*gib,'node GPU frozen limits exceeded')
    require(g['total_bytes']-max(g['reserved_bytes'],g['device_used_bytes'])>=max(settings['gpu_min_free_gib']*gib,settings['gpu_min_free_fraction']*g['total_bytes']),'node GPU headroom insufficient')
    require(h['scope'] in ('cgroup-nonreclaimable','cgroup-current-minus-file','proc-rss-sum') and h.get('oom_kill')==0 and h.get('telemetry_complete',True) and h['limit_bytes']>=settings['host_min_limit_gib']*gib and h['peak_bytes']<=settings['host_max_peak_gib']*gib,'node host scope/limit/peak failed')
    require(h['limit_bytes']-h['peak_bytes']>=max(settings['host_min_free_gib']*gib,settings['host_min_free_fraction']*h['limit_bytes']),'node host headroom insufficient')


def validate_measurement(out,doc,seed):
    rows=[read(out/'p2'/f's{seed}'/f'geometry-r{r}.json') for r in range(8)]
    expected=identity(out,doc,seed)
    for r,row in enumerate(rows):
        require(row['rank']==r,'X4 resource rank missing')
        check_rank(row,expected,read(SETTINGS),measurement=True)
    validate_geometry([r['geometry'] for r in rows],8)
    return rows


def validate_records(measured,full,expected,settings):
    require(len(measured)==len(full)==8,'eight measured and full-path ranks required')
    for rank,(a,b) in enumerate(zip(measured,full)):
        require(a['rank']==b['rank']==rank,'feasibility rank inventory differs')
        initial=dict(expected,calibration_sha256=None)
        check_rank(a,initial,settings,measurement=True);check_rank(b,expected,settings)
        require(a['geometry']==b['geometry'],'X4/X6 shard/tie/padding accounting differs')
    validate_geometry([v['geometry'] for v in full],8)
    require(len({v['gpu']['uuid'] for v in full})==8 and all(v['gpu']['uuid'] for v in full),'eight distinct physical GPU identities required')


def synthetic(out,doc,seed,calibration_sha):
    """Labelled CPU dry-run gate only; never accepted by the real reader."""
    root=out/'p2/feasibility'/f's{seed}'
    atomic(root/'receipt.json',dict(schema='rung-p2-feasibility.v1',status='PASS',identity=identity(out,doc,seed,calibration_sha),
        technical_synthetic=True,disposable=True,restarted_from_untouched=True,full_updates=40))


def seal(out,doc,seed,expected):
    """Finish a completed native pass without rerunning it after a requeue."""
    root=out/'p2/feasibility'/f's{seed}';run=root/'run'
    measured=[read(out/'p2'/f's{seed}'/f'geometry-r{r}.json') for r in range(8)]
    full=[read(run/f'resource-r{r}.json') for r in range(8)]
    validate_records(measured,full,expected,read(SETTINGS))
    atomic(root/'receipt.json',dict(schema='rung-p2-feasibility.v1',status='PASS',identity=expected,
        technical_synthetic=False,disposable=True,restarted_from_untouched=True,full_updates=40,
        rank_sha256={str(r):sha(run/f'resource-r{r}.json') for r in range(8)}))
    if (run/'sdft').exists():shutil.rmtree(run/'sdft')
    validate_local(out,doc,seed)
    (root/'pending.json').unlink(missing_ok=True)


def ensure(a,doc,calibration):
    """Called only for P-containing training. Failure returns False, preserving A."""
    if registered_tier2(getattr(a,'arm','p')):
        try:validate_local(a.out,doc,a.seed);return True
        except (ValueError,KeyError,OSError,TypeError):return False
    try:
        if getattr(a,'arm','p')!='p':
            atomic(a.out/'p2/feasibility'/f'unregistered-{a.arm}-s{a.seed}.json',dict(status='DEFERRED',reason='P2-containing stack requires its own registered incoming/schedule feasibility; P-only receipt cannot be reused'));return False
        if a.standin:
            if not (a.out/'p2/feasibility'/f's{a.seed}'/'receipt.json').exists():synthetic(a.out,doc,a.seed,sha(a.out/'p2/calibration.json'))
            validate_local(a.out,doc,a.seed);return True
        root=a.out/'p2/feasibility'/f's{a.seed}';receipt=root/'receipt.json'
        expected=identity(a.out,doc,a.seed,sha(a.out/'p2/calibration.json'))
        require(not (root/'terminal-failure.json').exists(),'completed disposable pass failed; no retry')
        if receipt.exists():
            validate_local(a.out,doc,a.seed);return True
        from kit.x_common import incoming,m
        require(m.checked_model(incoming(doc,a.seed),False)==incoming(doc,a.seed)['identity'],'disposable preflight incoming changed')
        from kit.x_p2_measure import runtime
        from kit import m2_train
        run=root/'run';data=a.out/'p2-data'/f's{a.seed}'
        m.frozen_files(data)
        require(sha(data/'manifest.json')==read(a.out/'p2'/f's{a.seed}'/'schedule.json')['data_manifest_sha256'],'disposable preflight data differ from X4')
        args=NS(**dict(vars(a),stage='legal'))
        cmd,_,env=m2_train.command(args,'D+replay',incoming(doc,a.seed)['model'],data,run)
        cmd=runtime(env,cmd,mode='train',batch=a.out/'p2'/f's{a.seed}',run=run,anchor=doc['p2_anchor']['model'],seed=a.seed,value=calibration['coefficient']['value'])
        cmd=runtime_identity(env,cmd,root/'identity.json',expected)
        if run.exists():
            if (run/'process-exit.json').exists():
                require(read(run/'process-exit.json')['returncode']==0,'completed disposable process failed; no retry')
                seal(a.out,doc,a.seed,expected);return True
            require(not (root/'run.interrupted-1').exists(),'disposable retry already consumed')
            run.rename(root/'run.interrupted-1')
        cmd=[v for v in cmd if not v.startswith('trainer.save_freq=')]+['trainer.save_freq=40']
        run.mkdir(parents=True);atomic(run/'command.json',cmd)
        measured=[read(a.out/'p2'/f's{a.seed}'/f'geometry-r{r}.json') for r in range(8)]
        for r,value in enumerate(measured):
            require(value['rank']==r,'X4 resource rank missing');check_rank(value,dict(expected,calibration_sha256=None),read(SETTINGS),measurement=True)
        validate_geometry([v['geometry'] for v in measured],8)
        start=time.monotonic()
        try:
            try:
                subprocess.run(cmd,cwd=a.sdpo,env=env,check=True)
                atomic(run/'process-exit.json',dict(returncode=0))
            except subprocess.CalledProcessError as error:
                if error.returncode in (-15,-9,143,137):
                    from kit.x_memory import sample
                    if sample(node=True).get('oom_kill')==0:
                        atomic(run/'interrupted.json',dict(returncode=error.returncode,oom_kill=0))
                        raise InterruptedError('disposable pass interrupted; one fresh requeue allowed') from error
                atomic(run/'process-exit.json',dict(returncode=error.returncode));raise
        finally:atomic(run/'cost.json',dict(gpu_hours=8*(time.monotonic()-start)/3600,seconds=time.monotonic()-start,disposable=True))
        seal(a.out,doc,a.seed,expected);return True
    except (ValueError,KeyError,OSError,TypeError,RuntimeError,subprocess.CalledProcessError) as error:
        failed_root=a.out/'p2/feasibility'/f's{a.seed}'
        exhausted=(failed_root/'run.interrupted-1').exists()
        if isinstance(error,InterruptedError) and not exhausted:
            atomic(failed_root/'pending.json',dict(status='PENDING',reason=str(error),fresh_retry_remaining=True))
            return False
        if (failed_root/'run/process-exit.json').exists() or exhausted:
            atomic(failed_root/'terminal-failure.json',dict(status='FAIL',reason=str(error)))
        atomic(a.out/'p2/feasibility'/f's{a.seed}'/'failure.json',dict(status='FAIL',reason=str(error),p2_only=True))
        (failed_root/'pending.json').unlink(missing_ok=True)
        return False


def retry_pending(out,seed):
    root=out/'p2/feasibility'/f's{seed}'
    return ((root/'pending.json').exists() and not (root/'receipt.json').exists() and
        not (root/'terminal-failure.json').exists() and not (root/'failure.json').exists())


def validate_local(out,doc,seed):
    root=out/'p2/feasibility'/f's{seed}';v=read(root/'receipt.json')
    expected=identity(out,doc,seed,sha(out/'p2/calibration.json'))
    require(v['schema']=='rung-p2-feasibility.v1' and v['status']=='PASS' and v['identity']==expected and v['disposable'] and v['restarted_from_untouched'] and v['full_updates']==40,'P2 feasibility missing/failed/stale')
    require(v['technical_synthetic'] is doc['technical_synthetic'],'P2 feasibility scope differs')
    if doc['technical_synthetic']:return v
    full=[]
    for r in range(8):
        path=root/'run'/f'resource-r{r}.json';require(sha(path)==v['rank_sha256'][str(r)],'node resource receipt changed');full.append(read(path))
    validate_records([read(out/'p2'/f's{seed}'/f'geometry-r{r}.json') for r in range(8)],full,expected,read(SETTINGS))
    return v


def validate_return(evidence,doc,standin):
    require(doc['p2_node_limits_sha256']==sha(SETTINGS),'prospective node limits changed')
    require(json.loads(evidence['p2/calibration.json'])['node_limits_sha256']==sha(SETTINGS),'X4 did not bind frozen node limits')
    calibration_sha=hashlib.sha256(evidence['p2/calibration.json']).hexdigest()
    for seed in range(101,104):
        # Calibration-only returns remain readable; scored/training P2 evidence
        # cannot evade the gate by omitting its completion marker.
        active=any(name.startswith(f'runs/x-p-s{seed}/') or any(name.startswith(f'scores/x-p-s{seed}-k{k}-') for k in (10,20,30,40)) for name in evidence)
        if not active:continue
        require(f'runs/x-p-s{seed}/complete.json' in evidence,'P2 completion missing; feasibility cannot be bypassed by removing its marker')
        v=validate_seed_return(evidence,doc,seed,standin)
        if standin:continue
        root=f'p2/feasibility/s{seed}/';expected=v['identity']
        full=[json.loads(evidence[root+f'run/resource-r{r}.json']) for r in range(8)]
        scored_identity=dict(expected,preflight_receipt_sha256=hashlib.sha256(evidence[root+'receipt.json']).hexdigest())
        scored=[json.loads(evidence[f'runs/x-p-s{seed}/resource-r{r}.json']) for r in range(8)]
        for r,v in enumerate(scored):
            require(v['rank']==r and v['geometry']==full[r]['geometry'],'scored P2 shard accounting differs')
            check_rank(v,scored_identity,read(SETTINGS))


def scored_runtime(env,cmd,a,doc,run):
    v=validate_local(a.out,doc,a.seed)
    expected=dict(v['identity'],preflight_receipt_sha256=sha(a.out/'p2/feasibility'/f's{a.seed}'/'receipt.json'))
    return runtime_identity(env,cmd,a.out/'p2'/f'scored-s{a.seed}-identity.json',expected)


def validate_seed_return(evidence,doc,seed,standin):
    """Validate the reusable full-path receipt independently of scored P2 files."""
    require(doc['p2_node_limits_sha256']==sha(SETTINGS),'prospective node limits changed')
    require(json.loads(evidence['p2/calibration.json'])['node_limits_sha256']==sha(SETTINGS),'X4 did not bind frozen node limits')
    calibration_sha=hashlib.sha256(evidence['p2/calibration.json']).hexdigest()
    root=f'p2/feasibility/s{seed}/';v=json.loads(evidence[root+'receipt.json'])
    expected=dict(binding_sha256=hashlib.sha256(evidence['binding.json']).hexdigest(),incoming=doc['spec']['nodes'][f'b-s{seed}']['checkpoint'],
        anchor=doc['p2_anchor'],seed=seed,configuration=doc['configuration'],settings_sha256=sha(SETTINGS),calibration_sha256=calibration_sha,
        implementation=doc['implementation'],strategy='fsdp2',world_size=8,model_revision=doc['p2_anchor']['revision'])
    require(v['schema']=='rung-p2-feasibility.v1' and v['status']=='PASS' and v['identity']==expected and v['technical_synthetic'] is standin and v['disposable'] and v['restarted_from_untouched'] and v['full_updates']==40,'returned node feasibility missing/failed/stale')
    if standin:return v
    full=[]
    for r in range(8):
        raw=evidence[root+f'run/resource-r{r}.json'];require(hashlib.sha256(raw).hexdigest()==v['rank_sha256'][str(r)],'returned node resource receipt changed');full.append(json.loads(raw))
    measured=[json.loads(evidence[f'p2/s{seed}/geometry-r{r}.json']) for r in range(8)]
    validate_records(measured,full,expected,read(SETTINGS))
    return v
