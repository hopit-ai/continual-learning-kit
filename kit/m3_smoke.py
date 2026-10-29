"""Prepared-only M3 Modal smoke: one container/attempt, two H100s, no retries.

Default CLI validates/plans only. Local text fixtures, code, partner_image spec,
public revision/hash policies and prices are frozen before launch. Public inputs
are fetched and verified inside the container only; no weights are uploaded.
No model/dataset download or remote action happens on import.
The small model never qualifies the eight-H100 8B resource preflight.

Runtime bundle: model_revision, partner runtime spec, quote_file, pricing (per-hour gpu/cpu/
ram_gib rates plus quote_sha256), timeout_seconds, startup_seconds, files
(relative name -> sha256), phase_forecast (each arm/phase binds a numeric field
in a sanitized hash-bound historical message-2 smoke extract and an explicit projection multiplier), commands (C0/K/G/CR -> argv for kit.m3_entry).
All commands run two updates, seed 101, stage 3, technical-smoke, local base.
Frozen JSONL schedules must carry every required domain and identical shared
prompts/targets across arms. Outputs are named C0.json, K.json, G.json, CR.json.
"""
from __future__ import annotations
import argparse
from collections import deque
from contextlib import nullcontext, contextmanager
import hashlib
import json
import math
import os
import re
import shutil
import signal
from pathlib import Path
import subprocess
import sys
import threading
import time
from kit.m3_contract import ARMS, validate_receipt, geometry, stage_number
from kit.m3_entry import parse_args, smoke_loopback_settings, SMOKE_INIT_TIMEOUT

MODEL_REVISION = '70d244cc86ccca08cf5af4e1e306ecf908b1ad5e'
DATASET_REVISION = '47c03c73621c4aab2b824b7818681117d662770e'
CPU_CORES, RAM_GIB, GPU_COUNT = 8, 64, 2
CEILING_USD = 10.
FORECAST_PHASES=('startup','rollout','updates','export','teardown')
CODE_FILES = ('m3_contract.py','m3_actor.py','m3_runner.py','m3_entry.py','m3_loss.py','m3_smoke.py',
              'sdft_actor.py','sdft_objective.py','sdft_verl.py','v4_contract.py')


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def budget(bundle):
    pricing = bundle['pricing']
    rates = [pricing[k] for k in ('gpu','cpu','ram_gib')]
    if any(not math.isfinite(v) or v <= 0 for v in rates) or len(pricing['quote_sha256']) != 64:
        raise ValueError('freeze complete positive resource prices and quote digest')
    timeout, startup = bundle['timeout_seconds'], bundle['startup_seconds']
    if not 1 <= timeout <= 1800 or not 1 <= startup <= 600:
        raise ValueError('hard runtime/startup bounds differ')
    hourly = GPU_COUNT*rates[0] + CPU_CORES*rates[1] + RAM_GIB*rates[2]
    ceiling = hourly*(timeout+startup)/3600
    if ceiling > CEILING_USD:
        raise ValueError('aggregate priced reservation exceeds ten dollars')
    return dict(hourly_usd=hourly, total_ceiling_usd=ceiling,
                timeout_seconds=timeout, startup_seconds=startup)


def receipt_forecast(bundle, root):
    """Price all four arms from hash-bound historical smoke phase measurements.

    Each prospective cell is {receipt: relative file, field: dotted numeric
    JSON selector, multiplier: positive scale}. Multipliers are explicit cost
    projections; neither observed nor forecast speed controls a pass decision.
    """
    forecasts=bundle.get('phase_forecast',{})
    if set(forecasts)!=set(ARMS):
        raise ValueError('missing four-arm receipt-based startup/rollout/export forecast; qualification INCOMPLETE')
    result={}
    for arm,cells in forecasts.items():
        if set(cells)!=set(FORECAST_PHASES): raise ValueError('smoke forecast omits a charged phase')
        result[arm]={}
        for phase,cell in cells.items():
            relative=cell['receipt']
            if relative not in bundle['files']: raise ValueError('forecast receipt must be frozen with inputs')
            location=(Path(root)/relative).resolve()
            if not location.is_relative_to(Path(root).resolve()) or sha_file(location)!=bundle['files'][relative]:
                raise ValueError('forecast receipt identity differs')
            value=json.loads(location.read_text())
            for part in cell['field'].split('.'):
                value=value[int(part)] if isinstance(value,list) else value[part]
            multiplier=cell['multiplier']
            if (type(value) not in (int,float) or not math.isfinite(value) or value<0 or
                    type(multiplier) not in (int,float) or not math.isfinite(multiplier) or multiplier<=0):
                raise ValueError('forecast requires a measured numeric phase and positive explicit multiplier')
            result[arm][phase]=value*multiplier
    total=sum(v for cells in result.values() for v in cells.values())
    if total>bundle['timeout_seconds']:
        raise ValueError('four-arm forecast cannot fit the one reservation; qualification INCOMPLETE')
    return dict(phase_forecast_seconds=result,forecast_seconds=total,
                forecast_cost_usd=total*budget(bundle)['hourly_usd']/3600)


def validate_bundle(bundle, root, *, check_rows=True):
    root = Path(root).resolve()
    if bundle['model_revision'] != MODEL_REVISION or bundle.get('runtime')!=partner_runtime_spec():
        raise ValueError('smoke model/partner runtime spec must be pinned')
    plan = budget(bundle)
    if 'quote_file' in bundle:
        quote_path=root/bundle['quote_file']
        quote=validate_quote(quote_path)
        if sha_file(quote_path)!=bundle['pricing']['quote_sha256']:
            raise ValueError('price quote identity differs')
        if any(bundle['pricing'][key]!=quote['hourly_rates_usd'][key] for key in ('gpu','cpu','ram_gib')):
            raise ValueError('bundle rates differ from the frozen quote')
        plan.update(quoted_total_usd=quote['total_usd'],egress_ceiling_usd=quote['receipt_egress_ceiling_usd'],
                    image_build_reserve_usd=quote['image_build_reserve_usd'])
        if any(bundle[key]!=quote[key] for key in ('timeout_seconds','startup_seconds')):
            raise ValueError('bundle reservation differs from the frozen quote')
    if set(bundle.get('code_sha256', {})) != set(CODE_FILES):
        raise ValueError('freeze all M3 code and inherited objective hashes')
    for name, expected in bundle['code_sha256'].items():
        if sha_file(Path(__file__).parent/name) != expected:
            raise ValueError('smoke code differs from frozen bundle')
    if 'kit_files_sha256' in bundle and bundle['kit_files_sha256']!=kit_file_hashes():
        raise ValueError('frozen kit tree differs')
    if set(bundle['commands']) != set(ARMS) or not bundle['files']:
        raise ValueError('exactly four frozen arm commands and input hashes required')
    for relative, digest in bundle['files'].items():
        candidate = (root/relative).resolve()
        if not candidate.is_relative_to(root) or sha_file(candidate) != digest:
            raise ValueError('frozen smoke input identity differs')
    plan.update(receipt_forecast(bundle,root))
    configurations = {}
    for arm, command in bundle['commands'].items():
        if command[:3] != ['python','-m','kit.m3_entry']:
            raise ValueError('smoke uses the M3 production entry point')
        config, overrides = parse_args(bundle_command(command, root)[3:])
        if (config['arm'] != arm or config['profile'] != 'technical-smoke' or config['seed'] != 101
                or stage_number(config['stage']) != 3 or config['base_revision'] != MODEL_REVISION):
            raise ValueError('smoke command identity differs')
        if Path(config['receipt']).name != arm+'.json':
            raise ValueError('smoke receipt filename differs')
        configurations[arm] = (config, overrides)
    if len({config['base_model'] for config,_ in configurations.values()}) != 1:
        raise ValueError('all arms must start from the same local base')
    if check_rows:
        base = (root/next(iter(configurations.values()))[0]['base_model']).resolve()
        if not base.is_relative_to(root):
            raise ValueError('smoke base must be staged inside the frozen bundle')
        model_files = [p for p in base.rglob('*') if p.is_file()]
        if not model_files or not (base/'config.json').is_file():
            raise ValueError('missing locally staged smoke base')
        if any(str(p.relative_to(root)) not in bundle['files'] for p in model_files):
            raise ValueError('all smoke base files must be hash-bound')
        from kit.m3_contract import verify_base_files
        verify_base_files(base,root/'inputs/base.json',MODEL_REVISION)
        check_schedules(bundle, root, configurations)
    return plan


def check_schedules(bundle, root, configurations):
    """Verify actual frozen input rows before any allocation or CUDA import."""
    common, task_signatures = {}, {}
    for arm, (config, overrides) in configurations.items():
        entries = dict(x.lstrip('+').split('=',1) for x in overrides if '=' in x)
        file_value = entries['data.train_files']
        if file_value.startswith('['):
            values = json.loads(file_value)
            if len(values) != 1: raise ValueError('one smoke schedule required')
            file_value = values[0]
        relative = str(Path(file_value).relative_to(root)) if Path(file_value).is_absolute() else file_value
        if relative not in bundle['files']:
            raise ValueError('training rows must be in the frozen input map')
        rows = read_jsonl(root/relative)
        shape = geometry(arm,3,2)
        if len(rows) != 2*shape['global_rows']:
            raise ValueError('smoke needs exactly two full updates')
        for step in range(2):
            batch = rows[step*shape['global_rows']:(step+1)*shape['global_rows']]
            replay = [r for r in batch if r['extra_info'].get('m3_replay')]
            task = [r for r in batch if not r['extra_info'].get('m3_replay')]
            if len(task) != shape['task_rows'] or len(replay) != shape['replay_rows']:
                raise ValueError('smoke row counts differ')
            domains = [r['extra_info']['m3_domain'] for r in replay]
            expected = ({'instruction':4} if arm in ('C0','K') else {'instruction':24} if arm=='G'
                        else {'instruction':4,'gsm8k':2,'arc':2,'finqa':8,'contract_nli':8})
            if {d:domains.count(d) for d in set(domains)} != expected:
                raise ValueError('smoke domain masks incomplete')
            for row in replay:
                ids, eos = row['extra_info']['m3_target_ids'], row['extra_info']['m3_eos_token_id']
                if not ids or ids[-1] != eos or eos in ids[:-1]:
                    raise ValueError('invalid EOS-inclusive smoke target')
            # Compare the same four instruction draws, including prefixes/labels.
            signature = [(r['prompt'],r['extra_info']['m3_target_ids']) for r in replay[:4]]
            if step in common and common[step] != signature:
                raise ValueError('shared generic smoke inputs differ')
            common[step] = signature
            task_signatures[arm,step] = [(r['prompt'],r['extra_info']['demonstration']) for r in task]
    for step in range(2):
        full = task_signatures['CR',step]
        reduced = [r for i,r in enumerate(full,1) if i not in (8,16,24,32)]
        if (task_signatures['G',step] != full or task_signatures['C0',step] != reduced
                or task_signatures['K',step] != reduced):
            raise ValueError('frozen shared task targets differ')



def cgroup_sample(reservation_bytes=RAM_GIB*1024**3, *, proc_cgroup='/proc/self/cgroup',
                  cgroup_root='/sys/fs/cgroup', proc_mountinfo=None):
    """Process cgroup working set: usage minus inactive file cache, v2 or v1.

    The explicit reservation caps an unlimited ('max') cgroup. Mount roots are
    honored for Slurm/container namespaces; no node-wide root usage fallback.
    Missing counters remain unmeasured rather than manufacturing a PASS.
    """
    if type(reservation_bytes) is not int or reservation_bytes<=0:raise ValueError('positive host reservation required')
    root=Path(cgroup_root)
    try:memberships=Path(proc_cgroup).read_text().splitlines()
    except OSError:return None
    groups=[]
    for line in memberships:
        parts=line.split(':',2)
        if len(parts)!=3:continue
        hierarchy,controllers,group=parts
        if hierarchy=='0' and not controllers:groups.insert(0,('v2',group))
        elif 'memory' in controllers.split(','):groups.append(('v1',group))
    mounts=[]
    mountinfo=proc_mountinfo or ('/proc/self/mountinfo' if str(root)=='/sys/fs/cgroup' else None)
    if mountinfo:
        try:
            for line in Path(mountinfo).read_text().splitlines():
                left,right=line.split(' - ',1);fields=left.split();tail=right.split()
                kind='v2' if tail[0]=='cgroup2' else 'v1' if tail[0]=='cgroup' and 'memory' in ','.join(tail[1:]).split(',') else None
                if kind:
                    def unescape(value):return re.sub(r'\\([0-7]{3})',lambda m:chr(int(m[1],8)),value)
                    mounts.append((kind,Path(unescape(fields[3])),Path(unescape(fields[4]))))
        except (OSError,ValueError,IndexError):pass
    for kind,group in groups:
        member=Path(group)
        candidates=[]
        for version,mount_root,mount_point in mounts:
            if version==kind and member.is_relative_to(mount_root):
                candidates.append(mount_point/member.relative_to(mount_root))
        relative=group.lstrip('/')
        if '..' not in Path(relative).parts:
            candidates.extend(([root/relative] if kind=='v2' else [root/'memory'/relative,root/relative]))
        for folder in dict.fromkeys(candidates):
            usage,limit=('memory.current','memory.max') if kind=='v2' else ('memory.usage_in_bytes','memory.limit_in_bytes')
            try:
                used=int((folder/usage).read_text());text=(folder/limit).read_text().strip()
                maximum=reservation_bytes if text=='max' else min(int(text),reservation_bytes)
                stats={parts[0]:int(parts[1]) for line in (folder/'memory.stat').read_text().splitlines()
                       if len(parts:=line.split())==2}
                inactive=stats.get('total_inactive_file',stats.get('inactive_file',0)) if kind=='v1' else stats.get('inactive_file',0)
                if used<0 or maximum<=0 or inactive<0:continue
                return max(0,used-inactive),maximum
            except (OSError,ValueError):continue
    return None


class MemoryPhaseTracker:
    """Separate training/startup peaks from diagnostic oracle overhead.

    Snapshot training before the oracle, reset counters, measure the oracle for
    telemetry only, release its cached CUDA blocks, reset again, then continue
    measuring optimizer/EMA and later training. A lock prevents sampler races.
    """
    def __init__(self, device_sample, reset_peak, release_cache, synchronize, host_sample):
        self.device_sample=device_sample;self.reset_peak=reset_peak
        self.release_cache=release_cache;self.synchronize=synchronize;self.host_sample=host_sample
        self.lock=threading.RLock();self.phase='training'
        self.peaks={name:dict(gpu_peak_fraction=0.,allocator_peak_reserved_bytes=0,cgroup_peak_fraction=None)
                    for name in ('training','oracle')}

    def sample(self):
        with self.lock:
            used,total,reserved=self.device_sample();value=self.peaks[self.phase]
            value['gpu_peak_fraction']=max(value['gpu_peak_fraction'],used/total,reserved/total)
            value['allocator_peak_reserved_bytes']=max(value['allocator_peak_reserved_bytes'],reserved)
            host=self.host_sample()
            if host is not None:value['cgroup_peak_fraction']=max(value['cgroup_peak_fraction'] or 0.,host[0]/host[1])

    @contextmanager
    def oracle_phase(self):
        with self.lock:
            self.synchronize();self.sample()  # Gate on the saved pre-oracle peak.
            self.reset_peak();self.phase='oracle'
        try:
            yield
        finally:
            with self.lock:
                self.synchronize();self.sample()
                self.release_cache()  # Freed oracle blocks must not pollute later reserved-byte peaks.
                self.reset_peak();self.phase='training'

    def snapshot(self):
        with self.lock:
            self.sample()
            return {name:dict(values) for name,values in self.peaks.items()}


def verdict(receipts, *, host_peak_fraction, actual_cost_usd, cpu_checks):
    """Fail-closed smoke verdict, with affected arms independently identified."""
    results = {}
    control = receipts.get('C0')
    for arm in ARMS:
        record = receipts.get(arm)
        if record is None:
            results[arm] = dict(status='INCOMPLETE', gates={})
            continue
        validate_receipt(record)
        if (record['arm'] != arm or record['profile'] != 'technical-smoke' or record['seed'] != 101
                or stage_number(record['stage']) != 3):
            raise ValueError('smoke receipt identity differs from frozen arm')
        updates = record.get('updates', [])
        oracle_updates=[u for u in updates if 'm3/gradient_relative_error' in u]
        gates = dict(record['gates'])
        gates.update(cpu_numerics=all(cpu_checks.get(k) is True for k in ('kl_value_gradient','accumulation','masks','frozen_hash')),
            two_updates=len(updates)==2 and record['completed_updates']==2,
            zero_nonzero_lr=len(updates)==2 and updates[0]['learning_rate']==0 and updates[1]['learning_rate']>0,
            gradient_agreement=len(oracle_updates)==1 and oracle_updates[0].get('learning_rate',0)>0
                and 0<=oracle_updates[0]['m3/gradient_relative_error']<=1e-4,
            frozen_reference=record['frozen_base_before']==record['frozen_base_after'] and not record['frozen_base_has_grad'],
            gpu_memory=record['telemetry']['gpu_peak_fraction'] <= .90,
            cgroup_ram=host_peak_fraction is not None and host_peak_fraction <= .95,
            aggregate_cost=actual_cost_usd is not None and 0 <= actual_cost_usd <= CEILING_USD)
        gates['cuda_full_kl']=record.get('cuda_full_kl',{}).get('passed') is True
        if arm=='K':
            gates['initial_replay_kl']=bool(updates) and -1e-5<=updates[0].get('m3/initial_replay_kl_token_mean',float('inf'))<=1e-3
            gates['real_frozen_base'] = bool(record['frozen_base_before'])
        # All timing comparisons are descriptive telemetry, never gate keys.
        total_seconds=sum(u.get('m3/update_seconds',0.) for u in updates)
        total_tokens=sum(u.get('m3/scheduled_tokens',0) for u in updates)
        timing=dict(update_seconds=total_seconds,scheduled_tokens=total_tokens,
                    seconds_per_scheduled_token=total_seconds/total_tokens if total_tokens else None)
        if control and control.get('updates'):
            ctime=sum(u.get('m3/update_seconds',0.) for u in control['updates'])
            ctokens=sum(u.get('m3/scheduled_tokens',0) for u in control['updates'])
            timing['time_ratio_to_c0']=total_seconds/ctime if ctime>0 else None
            timing['token_time_ratio_to_c0']=(total_seconds/total_tokens)/(ctime/ctokens) if total_tokens and ctokens and ctime>0 else None
        missing = host_peak_fraction is None or actual_cost_usd is None
        results[arm] = dict(status='INCOMPLETE' if missing else 'PASS' if all(v is True for v in gates.values()) else 'FAIL',
                            gates=gates,telemetry=timing)
    return dict(schema='m3-smoke.v1', arms=results, scientific_evidence=False,
                qualifies_8b_node=False, actual_cost_usd=actual_cost_usd,
                host_peak_fraction=host_peak_fraction)



def run_smoke_process(command, *, timeout, cwd, env):
    """Bound a local phase and retain sanitized tails, including init failures."""
    child=subprocess.Popen(command,cwd=cwd,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                           text=True,start_new_session=True)
    tails={key:deque(maxlen=160) for key in ('stdout','stderr')}
    def drain(key):
        for line in getattr(child,key):tails[key].append(safe_fetch_text(line))
    threads=[threading.Thread(target=drain,args=(key,),daemon=True) for key in tails]
    for thread in threads:thread.start()
    timed_out=False
    try:
        child.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out=True
        os.killpg(child.pid,signal.SIGKILL)
        child.wait(timeout=2)
    finally:
        for thread in threads:thread.join(timeout=1)
    return dict(returncode=child.returncode,timed_out=timed_out,
                **{key+'_tail':safe_fetch_text(''.join(lines)) for key,lines in tails.items()})


def execute(bundle_path):
    """Inside the prepared container: run exact production commands, no retries."""
    bundle_path = Path(bundle_path).resolve()
    root = bundle_path.parent
    bundle = json.loads(bundle_path.read_text())
    plan = validate_bundle(bundle,root)
    cpu_checks = cpu_numerics()
    settings=smoke_loopback_settings()
    os.environ.update(settings['offline_env'], **settings['env'])
    started = time.monotonic()
    stop, samples = threading.Event(), []
    def monitor():
        while not stop.wait(.1):
            sample = cgroup_sample()
            if sample is not None: samples.append(sample[0]/sample[1])
    monitor_thread = threading.Thread(target=monitor,daemon=True)
    monitor_thread.start()
    receipts = {}; processes={}
    try:
        for arm in ARMS:
            remaining = plan['timeout_seconds']-(time.monotonic()-started)
            if remaining <= 0: break
            try:
                command=bundle_command(bundle['commands'][arm],root)
                processes[arm]={}
                head=run_smoke_process(settings['ray_head'],cwd=root,
                    timeout=min(SMOKE_INIT_TIMEOUT,remaining),env=dict(os.environ))
                processes[arm]['ray_head']=head
                if head['returncode']!=0 or head['timed_out']:break
                remaining=plan['timeout_seconds']-(time.monotonic()-started)
                if remaining<=0:break
                result=run_smoke_process(command,cwd=root,timeout=remaining,
                    env=dict(os.environ,KIT_FINISH_GATE='1'))
                processes[arm]['training']=result
                if result['returncode']!=0 or result['timed_out']:break
                if result['returncode'] == 0:
                    config,_ = parse_args(command[3:])
                    receipt = Path(config['receipt'])
                    if not receipt.is_absolute(): receipt=root/receipt
                    if receipt.is_file(): receipts[arm]=json.loads(receipt.read_text())
            except subprocess.TimeoutExpired:
                break
            finally:
                remaining = plan['timeout_seconds']-(time.monotonic()-started)
                # Stop failed and successful arms alike, within the one timeout.
                if remaining > 0:
                    try:
                        subprocess.run(['ray','stop','--force','--grace-period=2'],timeout=min(settings['cleanup_timeout_seconds'],remaining),
                                       stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
                    except subprocess.TimeoutExpired:
                        pass
    finally:
        stop.set();monitor_thread.join(timeout=1)
    record = dict(receipts=receipts, cpu_checks=cpu_checks, processes=processes, loopback=settings,
                  host_peak_fraction=max(samples) if samples else None,
                  execution_seconds=time.monotonic()-started)
    print('[m3-smoke] '+json.dumps(record,sort_keys=True),flush=True)
    return record


def cpu_numerics():
    """Fresh independent tiny-tensor checks inside the smoke container."""
    import torch
    from kit.m3_loss import projected_forward_kl, response_prediction_mask
    from kit.m3_actor import freeze_reference, reference_hash
    torch.manual_seed(101)
    h=torch.randn(2,7,3,requires_grad=True); w=torch.randn(11,3,requires_grad=True)
    bh=torch.randn_like(h); bw=torch.randn_like(w)
    got=projected_forward_kl(h,w,bh,bw,3).mean()
    s=(h@w.T).log_softmax(-1); b=(bh@bw.T).log_softmax(-1)
    expected=(b.exp()*(b-s)).sum(-1).mean()
    gradients=torch.autograd.grad(got,(h,w),retain_graph=True)
    refs=torch.autograd.grad(expected,(h,w))
    kl_check=abs(float((got-expected).detach()))<=1e-5 and all(float((a-b).abs().max())<=1e-5 for a,b in zip(gradients,refs))
    accumulation=True
    # Closed-form global derivative; this CPU check makes no claim about DDP
    # reduction semantics. Actual smoke/preflight workers use the no-sync oracle.
    for count in (32,40,48,56):
        x=torch.arange(1,count+1,dtype=torch.float32)/100
        parameter=torch.tensor(.7,requires_grad=True)
        gradient,=torch.autograd.grad(((x*parameter-1).square()).sum()/32,parameter)
        analytic=(2*x*(x*.7-1)).sum()/32
        accumulation=accumulation and bool(torch.isclose(gradient,analytic,atol=1e-4))
    mask=response_prediction_mask(torch.tensor([[8,9,5,2,0]]),torch.tensor([[1,1,1,1,0]]),2,2)
    module=freeze_reference(torch.nn.Linear(3,3)); before=reference_hash(module)
    module(torch.ones(1,3))
    return dict(kl_value_gradient=kl_check,accumulation=accumulation,masks=mask.tolist()==[[1,1,0]],
                frozen_hash=before==reference_hash(module) and all(p.grad is None for p in module.parameters()))


MODEL_FIXED_HASHES = {
    'config.json':'1ddb5b89ebc90dcb417a45c213d818577e65976454d29385c8f6140771d95197',
    'generation_config.json':'2325da0f15bb848e018c5ae071b7943332e9f871d6b60e2ed22ca97d4cb993d2',
    'tokenizer.json':'aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4',
    'tokenizer_config.json':'d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101',
}
DATASET_SHA256 = '75f1b1f63039044034fe75d22f51f77f257a79291d0084262bb43695d4aac971'
LOCAL_SOURCE_JSONL='local-inputs/rlvr-ifeval.jsonl'
LOCAL_SOURCE_MANIFEST='local-inputs/manifest.json'
IMAGE_DEVIATION='image-deviation.json'
IMAGE_EVIDENCE='image-drift-evidence.md'
COMPATIBILITY_PINS={'pyarrow':'21.0.0','datasets':'4.8.5'}
PACKET_FILES=(LOCAL_SOURCE_JSONL,LOCAL_SOURCE_MANIFEST,IMAGE_DEVIATION,IMAGE_EVIDENCE,'request.json','quote.json','forecast-basis.json','gates.md','development-rows.json','public-input-pins.json','launch.py','README.md')



def read_jsonl(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path, rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w') as handle:
        for row in rows:
            handle.write(json.dumps(row,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n')


def validate_local_source(root):
    """JSON-only validation; origin is the locally verified pinned Parquet SHA."""
    root=Path(root);manifest=json.loads((root/LOCAL_SOURCE_MANIFEST).read_text())
    if (manifest.get('schema')!='m3-smoke-local-source.v1' or manifest.get('dataset_revision')!=DATASET_REVISION
            or manifest.get('parquet_sha256')!=DATASET_SHA256 or manifest.get('jsonl_file')!=LOCAL_SOURCE_JSONL
            or sha_file(root/LOCAL_SOURCE_JSONL)!=manifest.get('jsonl_sha256')):
        raise ValueError('local JSONL source identity differs from pinned Parquet conversion')
    rows=read_jsonl(root/LOCAL_SOURCE_JSONL)
    if not rows or len(rows)!=manifest.get('row_count') or any(not isinstance(row,dict) for row in rows):
        raise ValueError('local JSONL source row inventory differs')
    return manifest


def prepare_local(request_path, cache, *, parquet_reader=None):
    """Operator-only local Parquet conversion. No Modal/model/network access.

    Hash before importing Arrow; the GPU container never calls this function.
    The full converted public source is provenance, not scientific replay targets.
    """
    root=Path(request_path).resolve().parent;cache=validate_fetch_cache(cache)
    candidates=sorted((cache/'downloads'/'instruction').rglob('*.parquet'))
    if len(candidates)!=1 or sha_file(candidates[0])!=DATASET_SHA256:
        raise ValueError('local pinned instruction Parquet SHA256 mismatch or missing unique file')
    if parquet_reader is None:
        import pyarrow.parquet as pq
        parquet_reader=lambda path:pq.read_table(path).to_pylist()
    rows=parquet_reader(candidates[0])
    if not rows or any(not isinstance(row,dict) for row in rows):raise ValueError('invalid local source rows')
    write_jsonl(root/LOCAL_SOURCE_JSONL,rows)
    manifest=dict(schema='m3-smoke-local-source.v1',dataset='allenai/RLVR-IFeval',dataset_revision=DATASET_REVISION,
        parquet_sha256=DATASET_SHA256,jsonl_file=LOCAL_SOURCE_JSONL,jsonl_sha256=sha_file(root/LOCAL_SOURCE_JSONL),
        row_count=len(rows),conversion='all rows and fields; canonical UTF-8 JSONL; no filtering',
        source_use='Pinned public-source provenance only; engineering targets remain hand-authored.')
    (root/LOCAL_SOURCE_MANIFEST).write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    validate_local_source(root)
    return freeze_launch_packet(request_path)


def partner_runtime_spec():
    """Hash the exact shared builder, with no import-time Modal access."""
    path=Path(__file__).resolve().parents[1]/'scripts/smoke_common_modal.py'
    if not path.is_file():path=Path('/root/smoke_common_modal.py')
    return dict(builder='scripts/smoke_common_modal.py:partner_image',spec_sha256=sha_file(path),
                sdpo_commit='7c457fc1b1f636ae794eb0362ba37d4743b06fbc',compatibility_pins=dict(COMPATIBILITY_PINS))



def validate_image_deviation(root):
    """Freeze the manager-authorized pins and the CPU evidence without RPCs."""
    root=Path(root);value=json.loads((root/IMAGE_DEVIATION).read_text())
    if (value.get('schema')!='m3-image-deviation.v1' or value.get('declared_date')!='2026-10-29'
            or value.get('baseline_build_date')!='2026-10-26'
            or value.get('builder')!='scripts/smoke_common_modal.py:partner_image'
            or value.get('current_builder_spec_sha256')!=partner_runtime_spec()['spec_sha256']
            or value.get('compatibility_pins')!=COMPATIBILITY_PINS):
        raise ValueError('declared shared-image compatibility deviation differs')
    evidence=value.get('evidence',{})
    if (evidence.get('packet_file')!=IMAGE_EVIDENCE
            or evidence.get('source_path')!='../build/image-drift-20261029.md'
            or evidence.get('source_sha256')!=sha_file(root/IMAGE_EVIDENCE)):
        raise ValueError('declared image deviation evidence differs')
    return value


def validate_launch_packet(path):
    path=Path(path).resolve();root=path.parent;packet=json.loads(path.read_text())
    if packet.get('schema')!='m3-smoke-container-fetch.v1' or packet.get('runtime')!=partner_runtime_spec():
        raise ValueError('exact partner_image spec is not frozen')
    if packet['kit_files_sha256']!=kit_file_hashes():raise ValueError('frozen kit differs')
    if set(packet['files'])!=set(PACKET_FILES):raise ValueError('only frozen text packet files may be uploaded')
    for relative,digest in packet['files'].items():
        candidate=(root/relative).resolve()
        if not candidate.is_relative_to(root) or sha_file(candidate)!=digest:raise ValueError('frozen packet file differs')
    request=json.loads((root/'request.json').read_text());pins=json.loads((root/'public-input-pins.json').read_text())
    if (request.get('fetch_mode')!='container-pinned' or request.get('model_revision')!=MODEL_REVISION
            or request.get('dataset_revision')!=DATASET_REVISION or request.get('seed')!=101
            or request.get('arms')!=list(ARMS) or request.get('updates_per_arm')!=2
            or any(request.get('runtime_requires',{}).get(k)!=v for k,v in COMPATIBILITY_PINS.items())
            or request.get('image_deviation')!=IMAGE_DEVIATION
            or pins.get('model_revision')!=MODEL_REVISION or pins.get('dataset_revision')!=DATASET_REVISION
            or pins.get('token') is not False or pins.get('fetch_domain_allowlist')!=list(FETCH_DOMAINS)
            or pins['model_fixed_sha256']!=MODEL_FIXED_HASHES
            or pins['dataset_sha256']!=DATASET_SHA256 or pins.get('weight_hash_source')!='LFS SHA256 from metadata at exact pinned revision, resolved before download'):
        raise ValueError('pinned public input verification contract differs')
    image_deviation=validate_image_deviation(root)
    local_source=validate_local_source(root)
    rows=json.loads((root/'development-rows.json').read_text())
    if rows!={arm:engineering_rows(None,arm) for arm in ARMS}:raise ValueError('frozen short development schedules differ')
    quote=validate_quote(root/'quote.json')
    forecast=receipt_forecast(packet,root)
    return dict(quote_total_usd=quote['total_usd'],runtime_spec_sha256=packet['runtime']['spec_sha256'],image_deviation=image_deviation,
        model_revision=MODEL_REVISION,dataset_revision=DATASET_REVISION,local_source=local_source,
        fixture_rows={arm:len(rows[arm]) for arm in ARMS},input_fetch='in-container only; token=False',
        input_hash_policy='fixed SHA256 plus immutable-revision LFS/Git-blob metadata before use',
        sandbox_seconds=quote['reserved_seconds'],download_seconds=quote['download_seconds'],
        network_transition='worker network open; HF offline mode after verified fetch',
        loopback=smoke_loopback_settings(),**forecast)


def freeze_launch_packet(request_path):
    """Local text/hash preparation only; no public downloads and no Modal import."""
    request_path=Path(request_path).resolve();root=request_path.parent
    quote=validate_quote(root/'quote.json')
    (root/'development-rows.json').write_text(json.dumps({arm:engineering_rows(None,arm) for arm in ARMS},sort_keys=True,indent=2)+'\n')
    cells={phase:dict(receipt='forecast-basis.json',field='measured_seconds.'+field,multiplier=multiplier)
        for phase,field,multiplier in [('startup','model-stand-in',2.),('rollout','train-merge-D-chemistry-stand-in',.15),
        ('updates','train-merge-D-chemistry-stand-in',.65),('export','train-merge-D-chemistry-stand-in',.2),
        ('teardown','check-launcher-records-D-chemistry-stand-in',10.)]}
    from copy import deepcopy
    forecast={arm:deepcopy(cells) for arm in ARMS};forecast['K']['updates']['multiplier']*=1.3
    names=PACKET_FILES
    packet=dict(schema='m3-smoke-container-fetch.v1',runtime=partner_runtime_spec(),
        kit_files_sha256=kit_file_hashes(),files={name:sha_file(root/name) for name in names},
        timeout_seconds=quote['timeout_seconds'],startup_seconds=quote['startup_seconds'],
        pricing=dict(quote['hourly_rates_usd'],quote_sha256=sha_file(root/'quote.json')),phase_forecast=forecast)
    (root/'packet.json').write_text(json.dumps(packet,sort_keys=True,indent=2)+'\n')
    return preparation_plan(request_path)


FETCH_DOMAINS = (
    'huggingface.co','*.huggingface.co','hf.co','*.hf.co',
    'cdn-lfs.huggingface.co','cdn-lfs.hf.co','cdn-lfs-us-1.hf.co','cdn-lfs-eu-1.hf.co',
    'cdn-lfs-ap-1.hf.co','xethub.hf.co','*.xethub.hf.co',
    'cas-server.xethub.hf.co','cas-bridge.xethub.hf.co','transfer.xethub.hf.co',
    '*.amazonaws.com','*.cloudfront.net',
)
DIAGNOSTIC_LIMIT = 32768


class FetchVerificationError(ValueError):
    def __init__(self,message,category='verification',file=None):
        super().__init__(message);self.category=category;self.file=file


def safe_fetch_text(value, limit=DIAGNOSTIC_LIMIT):
    """Bound diagnostic text and remove auth material, signed URLs and identities."""
    value=str(value)
    value=re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]','',value)
    # A URL may contain userinfo, credentials in path segments or signed query
    # parameters. Keep only a known public service hostname, never its full URL.
    from urllib.parse import urlsplit
    def url(match):
        try:host=urlsplit(match.group()).hostname or ''
        except ValueError:host=''
        public=host in ('huggingface.co','hf.co') or host.endswith(('.huggingface.co','.hf.co','.amazonaws.com','.cloudfront.net'))
        return '[url:'+host+']' if public else '[url:redacted]'
    value=re.sub(r'(?i)https?://[^\s<>\"\']+',url,value)
    value=re.sub(r'(?i)\b(Bearer|Basic)\s+[A-Za-z0-9._~+/=-]+',r'\1 [REDACTED]',value)
    value=re.sub(r'\bhf_[A-Za-z0-9_-]+','[REDACTED]',value)
    value=re.sub(r'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b','[REDACTED]',value)
    value=re.sub(r'(?i)([\"\']?(?:authorization|proxy-authorization|cookie|set-cookie|password|secret|aws_secret_access_key|aws_session_token|aws_access_key_id|access[_-]?token|api[_-]?key|token|x-amz-[\w-]+)[\"\']?\s*[:=]\s*)(?:[\"\'][^\"\'\r\n]*[\"\']|[^\s,;}]+)',r'\1[REDACTED]',value)
    value=re.sub(r'(?i)\b(request|task|job|container|sandbox)[ _-]?id\s*[:=][^\r\n]*',r'\1 id=[REDACTED]',value)
    value=re.sub(r'\b[0-9a-fA-F]{8}-(?:[0-9a-fA-F]{4}-){3}[0-9a-fA-F]{12}\b','[REDACTED-ID]',value)
    value=re.sub(r'(?<![\w:])/(?:[^\s\"\'<>()[\]{}:,;]+/)*[^\s\"\'<>()[\]{}:,;]+','[path]',value)
    value=re.sub(r'\b[A-Za-z]:\\[^\r\n\"\']+','[path]',value)
    return value[-limit:]


def fetch_error(error, stage):
    raw=type(error).__name__+' '+str(error)
    lower=raw.lower();category=getattr(error,'category',None)
    status=getattr(getattr(error,'response',None),'status_code',None)
    if category is None:
        if any(term in lower for term in ('nameresolution','name resolution','failed to resolve','nodename','name or service','dns')):category='dns'
        elif 'timeout' in lower or 'timed out' in lower:category='timeout'
        elif status:category='http_'+str(status)
        elif any(term in lower for term in ('connection','network','proxy','ssl','certificate','denied')):category='network_or_access'
        else:category='unexpected'
    return dict(type=type(error).__name__,category=category,stage=stage,
                message=safe_fetch_text(str(error),4096),http_status=status)


def write_fetch_report(path, record):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(record,sort_keys=True,indent=2)+'\n');temporary.replace(path)


def lfs_sha256(entry):
    lfs=entry.lfs
    value=(lfs.get('sha256',lfs.get('oid')) if isinstance(lfs,dict) else getattr(lfs,'sha256',None)) if lfs else None
    return value.removeprefix('sha256:') if isinstance(value,str) else value


def verify_pinned_snapshot(directory, entries, fixed_hashes, *, results=None, progress=None):
    """Check content SHA256 for LFS, Git blob SHA1 only for ordinary Git files.

    A Git blob ID on an LFS entry identifies the pointer, not downloaded bytes;
    it must never be compared with the downloaded weight's content hash.
    Collect every file result before raising, including missing/failed files.
    """
    directory=Path(directory).resolve();found={};errors=[]
    for entry in entries:
        name=entry.rfilename;expected=lfs_sha256(entry)
        row=dict(file=name,status='PENDING',expected_fixed_sha256=fixed_hashes.get(name),
            expected_lfs_sha256=expected,expected_git_blob_sha1=entry.blob_id,
            git_blob_comparison='LFS pointer: not compared to content' if entry.lfs else 'ordinary Git content',
            actual_sha256=None,checks={})
        try:
            file=(directory/name).resolve()
            if not file.is_relative_to(directory) or not file.is_file():
                raise FetchVerificationError('missing pinned input file','missing_file',name)
            actual=sha_file(file);row.update(actual_sha256=actual,bytes=file.stat().st_size)
            if name in fixed_hashes:
                row['checks']['fixed_sha256']=actual==fixed_hashes[name]
                if actual!=fixed_hashes[name]:raise FetchVerificationError('pinned input SHA256 mismatch','fixed_hash_mismatch',name)
            if expected:
                row['checks']['lfs_content_sha256']=actual==expected
                if actual!=expected:raise FetchVerificationError('pinned LFS weight SHA256 mismatch','lfs_hash_mismatch',name)
            else:
                blob=hashlib.sha1(('blob '+str(file.stat().st_size)+'\0').encode()+file.read_bytes()).hexdigest()
                row['actual_git_blob_sha1']=blob;row['checks']['git_blob_sha1']=bool(entry.blob_id) and blob==entry.blob_id
                if not row['checks']['git_blob_sha1']:raise FetchVerificationError('pinned repository blob mismatch','git_blob_mismatch',name)
            if name.endswith('.safetensors') and not expected:
                raise FetchVerificationError('weight lacks pinned LFS SHA256','lfs_metadata_missing',name)
            found[name]=actual;row['status']='PASS'
        except Exception as error:
            row.update(status='FAIL',error=fetch_error(error,'verify'));errors.append(error)
        if results is not None:results[name]=row
        if progress:progress()
    if not set(fixed_hashes)<=set(found) and not errors:
        errors.append(FetchVerificationError('required pinned input omitted','missing_file'))
    if errors:raise errors[0]
    return found


def validate_fetch_cache(cache):
    cache=Path(cache).resolve()
    if any((p/'.git').exists() for p in (cache,*cache.parents)):
        raise ValueError('fetch cache must be outside the git tree')
    return cache


def fetch_public_inputs(packet_path, cache, *, api=None, snapshot=None):
    """The identical pin/download/verify path for local fetch-check and container."""
    packet_path=Path(packet_path).resolve();cache=validate_fetch_cache(cache)
    cache.mkdir(parents=True,exist_ok=True);report_path=cache/'fetch-report.json'
    report=dict(schema='m3-smoke-fetch.v2',status='FETCHING',inputs={},token=False,modal_calls=0,uploads=0,launches=0)
    def persist():write_fetch_report(report_path,report)
    persist()
    try:
        validate_launch_packet(packet_path)
        os.environ['HF_XET_CACHE']=str(cache/'xet-cache')
        from huggingface_hub import HfApi,snapshot_download,__version__
        report['client_versions']=dict(huggingface_hub=__version__,python=sys.version.split()[0])
        api=api or HfApi(endpoint='https://huggingface.co',token=False);snapshot=snapshot or snapshot_download
    except Exception as error:
        report.update(status='FAILED',error=fetch_error(error,'packet_or_client'));persist();return report
    for role,repo,revision,folder in (
        ('model','Qwen/Qwen3-1.7B',MODEL_REVISION,'model'),
        ('dataset','allenai/RLVR-IFeval',DATASET_REVISION,'instruction')):
        item=dict(repo=repo,requested_revision=revision,status='PENDING',stage='metadata',files={})
        report['inputs'][role]=item;persist()
        try:
            info=(api.model_info if role=='model' else api.dataset_info)(repo,revision=revision,files_metadata=True,token=False)
            item['resolved_revision']=info.sha
            if info.sha!=revision:raise FetchVerificationError('public input resolved revision differs','revision_mismatch')
            item['metadata_files']=[f.rfilename for f in info.siblings]
            if role=='model':
                assets=set(MODEL_FIXED_HASHES)|{'added_tokens.json','chat_template.jinja','special_tokens_map.json','vocab.json','merges.txt','tokenizer.model'}
                entries=[f for f in info.siblings if f.rfilename in assets or f.rfilename.endswith(('.safetensors','.safetensors.index.json'))]
                fixed=MODEL_FIXED_HASHES
                if not any(f.rfilename.endswith('.safetensors') for f in entries):
                    raise FetchVerificationError('pinned model has no weights','metadata_inventory')
            else:
                entries=[f for f in info.siblings if f.rfilename.endswith('.parquet')]
                if len(entries)!=1:raise FetchVerificationError('pinned instruction source file inventory differs','metadata_inventory')
                fixed={entries[0].rfilename:DATASET_SHA256}
            missing=sorted(set(fixed)-{f.rfilename for f in entries});item['missing_required_files']=missing
            if missing:raise FetchVerificationError('pinned metadata omits required files','metadata_inventory')
            for entry in entries:
                name=Path(entry.rfilename);expected=lfs_sha256(entry)
                if name.is_absolute() or '..' in name.parts:raise FetchVerificationError('invalid pinned file name','metadata_inventory')
                item['files'][entry.rfilename]=dict(file=entry.rfilename,status='PENDING',
                    expected_fixed_sha256=fixed.get(entry.rfilename),expected_lfs_sha256=expected,expected_git_blob_sha1=entry.blob_id)
                if entry.rfilename.endswith('.safetensors') and not (expected and re.fullmatch('[0-9a-f]{64}',expected)):
                    raise FetchVerificationError('weight lacks pinned LFS SHA256 before download','lfs_metadata_missing',entry.rfilename)
                if not entry.lfs and not re.fullmatch('[0-9a-f]{40}',entry.blob_id or ''):
                    raise FetchVerificationError('repository file lacks pinned Git identity before download','git_metadata_missing',entry.rfilename)
            item['stage']='download';persist();download_error=None
            try:
                snapshot(repo_id=repo,repo_type=role,revision=revision,local_dir=cache/'downloads'/folder,
                    cache_dir=cache/'hub-cache',endpoint='https://huggingface.co',allow_patterns=[f.rfilename for f in entries],token=False,max_workers=2)
            except Exception as error:
                download_error=error;item['download_error']=fetch_error(error,'download');persist()
            item['stage']='verify';persist();verification_error=None
            try:
                hashes=verify_pinned_snapshot(cache/'downloads'/folder,entries,fixed,results=item['files'],progress=persist)
                item.update(files_sha256=hashes,revision=revision,
                    expected_lfs_sha256={f.rfilename:lfs_sha256(f) for f in entries if f.lfs},
                    expected_git_blob_sha1={f.rfilename:f.blob_id for f in entries if not f.lfs})
            except Exception as error:verification_error=error;item['verification_error']=fetch_error(error,'verify')
            if download_error:
                item['stage']='download';raise download_error
            if verification_error:raise verification_error
            item.update(status='VERIFIED',stage='complete')
        except Exception as error:item.update(status='FAILED',error=fetch_error(error,item['stage']))
        persist()
    report['status']='VERIFIED' if all(v['status']=='VERIFIED' for v in report['inputs'].values()) else 'FAILED'
    if report['status']=='FAILED':
        role=next(k for k,v in report['inputs'].items() if v['status']=='FAILED')
        report['error']=dict(input=role,**report['inputs'][role]['error'])
    report['runtime_spec_sha256']=partner_runtime_spec()['spec_sha256'];persist()
    return report


def fetch_inside_container(packet_path, *, api=None, snapshot=None, work=None):
    """Verify all pins first, then prepare the production-worker data bundle."""
    packet_path=Path(packet_path).resolve();work=Path(work or '/work/m3-smoke')
    report=fetch_public_inputs(packet_path,work,api=api,snapshot=snapshot)
    if report['status']!='VERIFIED':return report
    try:
        root=work/'bundle';root.mkdir(parents=True,exist_ok=True)
        write_fetch_report(root/'input-verification.json',report['inputs'])
        def checked_cache(**options):
            item=report['inputs'][options['repo_type']]
            if options['revision']!=item['requested_revision']:raise ValueError('cached input revision differs')
            for name,expected in item['files_sha256'].items():
                if sha_file(Path(options['local_dir'])/name)!=expected:raise ValueError('verified cached input changed before use')
        result=prepare_public_bundle(packet_path.parent/'request.json',work,downloader=checked_cache,verified_inputs=report['inputs'])
        if result['status']!='READY':
            report['bundle_preparation']=result
            raise ValueError('pinned input preparation incomplete')
        bundle=work/'bundle/bundle.json';validate_bundle(json.loads(bundle.read_text()),bundle.parent)
    except Exception as error:
        report.update(status='FAILED',error=fetch_error(error,'bundle_prepare'))
        write_fetch_report(work/'fetch-report.json',report);return report
    write_fetch_report(work/'fetch-ready.json',report)
    return report


def run_fetch_child(packet_path, work, timeout, *, popen=None, action='fetch', cache=None):
    """Capture bounded sanitized tails in memory; never write raw child logs."""
    popen=popen or subprocess.Popen;work=validate_fetch_cache(work);work.mkdir(parents=True,exist_ok=True)
    tails={'stdout':deque(maxlen=160),'stderr':deque(maxlen=160)}
    def drain(stream,name):
        for line in stream:tails[name].append(safe_fetch_text(line))
    started=time.monotonic();timed_out=False
    command=[sys.executable,'-m','kit.m3_smoke',action,'--bundle',str(packet_path)]
    if cache is not None:command+=['--cache',str(cache)]
    initial=dict(schema='m3-smoke-fetch.v2',status='FETCHING',inputs={},stage='child_start')
    write_fetch_report(work/'fetch-report.json',initial)
    try:process=popen(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    except Exception as error:
        initial.update(status='FAILED',error=fetch_error(error,'child_start'),stdout_tail='',stderr_tail='')
        write_fetch_report(work/'fetch-report.json',initial);return initial
    threads=[threading.Thread(target=drain,args=(getattr(process,name),name),daemon=True) for name in tails]
    for thread in threads:thread.start()
    try:process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out=True;process.kill();process.wait()
    for thread in threads:thread.join(timeout=2)
    file=work/'fetch-report.json'
    try:record=json.loads(file.read_text())
    except (OSError,ValueError):record=dict(schema='m3-smoke-fetch.v2',status='FAILED',inputs={})
    record.update(child_returncode=process.returncode,timed_out=timed_out,fetch_seconds=time.monotonic()-started,
        stdout_tail=safe_fetch_text(''.join(tails['stdout'])),stderr_tail=safe_fetch_text(''.join(tails['stderr'])))
    if timed_out or process.returncode!=0 or record.get('status')!='VERIFIED':
        record['status']='FAILED'
        record.setdefault('error',dict(type='TimeoutExpired' if timed_out else 'FetchChildFailed',
            category='timeout' if timed_out else 'child_exit',stage='fetch_child',message='fetch child did not complete successfully'))
    write_fetch_report(file,record)
    return record


def bootstrap(packet_path):
    """One sandbox: verified public fetch, then open-network offline-mode train."""
    packet_path=Path(packet_path).resolve();validate_launch_packet(packet_path)
    quote=validate_quote(packet_path.parent/'quote.json');work=Path('/work/m3-smoke');work.mkdir(parents=True,exist_ok=True)
    ready=run_fetch_child(packet_path,work,quote['download_seconds'])
    print('[m3-fetch-result] '+json.dumps(ready,sort_keys=True),flush=True)
    if ready['status']!='VERIFIED':raise RuntimeError('pinned public fetch/hash verification failed; training refused')
    print('[m3-fetch-ready] '+json.dumps(ready,sort_keys=True),flush=True)
    os.environ.update(smoke_loopback_settings()['offline_env'])
    print('[m3-offline-ready] '+json.dumps(smoke_loopback_settings(),sort_keys=True),flush=True)
    execute(work/'bundle/bundle.json')



def check_attempt_budget(root, attempt, attempt_budget_usd=CEILING_USD):
    if attempt not in (1,2,3):raise ValueError('only an explicit first, second or third manual attempt is supported')
    if not math.isfinite(attempt_budget_usd) or not 0<attempt_budget_usd<=CEILING_USD:raise ValueError('invalid per-attempt budget; maximum ten dollars')
    root=Path(root);prior_cost=0.
    for previous in range(1,attempt):
        cost=json.loads((root/f'receipt.attempt{previous}.json').read_text())['actual_cost_usd']
        if not math.isfinite(cost) or cost<0:raise ValueError('invalid prior-attempt cost')
        prior_cost+=cost
    required=validate_quote(root/'quote.json')['total_usd']
    if required>attempt_budget_usd:
        raise ValueError(f'per-attempt reservation ${required:.8f} exceeds per-attempt budget ${attempt_budget_usd:.8f}')
    return prior_cost


def launch_modal(packet_path, attempt_file, *, image, app=None, attempt=1, attempt_budget_usd=CEILING_USD):
    """Operator-only call: exact partner image, local text packet, one sandbox."""
    import modal
    packet_path=Path(packet_path).resolve();root=packet_path.parent
    plan=validate_launch_packet(packet_path);quote=validate_quote(root/'quote.json')
    prior_cost=check_attempt_budget(root,attempt,attempt_budget_usd)
    with Path(attempt_file).open('x') as handle:json.dump(dict(attempt=attempt,retries=0,prior_attempt_cost_usd=prior_cost,attempt_budget_usd=attempt_budget_usd,budget_scope='per-attempt manager ruling',**plan),handle)
    owned_app=app is None;app=app or modal.App('m3-mechanics-smoke')
    started=time.monotonic();sandbox=None;raw=None;fetch=None;ready_seen=False;offline_ready=False;image_id=None;exitcode=None;error_type=None
    stdout_tail=deque(maxlen=160);stderr_tail=deque(maxlen=160);stderr=None
    with app.run() if owned_app else nullcontext():
        try:
            sandbox=modal.Sandbox.create('python','-m','kit.m3_smoke','bootstrap','--bundle','/inputs/packet.json',
                app=app,image=image,gpu='H100:2',cpu=(CPU_CORES,CPU_CORES),memory=(RAM_GIB*1024,RAM_GIB*1024),
                timeout=quote['reserved_seconds'],workdir='/inputs',
                block_network=False,
                env={'PYTHONPATH':'/root:/root/SDPO','WANDB_MODE':'disabled','OMP_NUM_THREADS':'8',
                    'TOKENIZERS_PARALLELISM':'false','KIT_V4_ARM':'D','ARM':'D','V4_CHECKPOINT_STATE':'1',
                    'HF_HUB_DISABLE_PROGRESS_BARS':'1','HF_HUB_ETAG_TIMEOUT':'20','HF_HUB_DOWNLOAD_TIMEOUT':'30'})
            image_id=image.object_id  # hydrated by Sandbox.create; never invent an ID
            def drain_stderr():
                for line in sandbox.stderr:stderr_tail.append(safe_fetch_text(line))
            stderr=threading.Thread(target=drain_stderr,daemon=True);stderr.start()
            for line in sandbox.stdout:
                if len(line.encode())>8*1024**2:raise ValueError('receipt exceeds egress allowance')
                if line.startswith('[m3-fetch-result] '):
                    fetch=json.loads(line[len('[m3-fetch-result] '):])
                elif line.startswith('[m3-fetch-ready] '):
                    if ready_seen:raise ValueError('duplicate fetch handshake')
                    ready_seen=True;fetch=json.loads(line[len('[m3-fetch-ready] '):])
                    if fetch.get('status')!='VERIFIED':raise ValueError('public input verification did not pass')
                elif line.startswith('[m3-offline-ready] '):
                    if not ready_seen:raise ValueError('offline mode before verified fetch')
                    if json.loads(line[len('[m3-offline-ready] '):])!=smoke_loopback_settings():
                        raise ValueError('offline/loopback settings differ')
                    offline_ready=True
                elif line.startswith('[m3-smoke] '):
                    if not offline_ready:raise ValueError('training started before verified-fetch offline mode')
                    raw=json.loads(line[len('[m3-smoke] '):])
                else:stdout_tail.append(safe_fetch_text(line))
            sandbox.wait();exitcode=sandbox.returncode;stderr.join(timeout=1)
        except Exception as error:
            error_type=type(error).__name__
        finally:
            if sandbox is not None:sandbox.terminate()
            if stderr is not None:stderr.join(timeout=2)
    elapsed=time.monotonic()-started
    hourly=GPU_COUNT*quote['hourly_rates_usd']['gpu']+CPU_CORES*quote['hourly_rates_usd']['cpu']+RAM_GIB*quote['hourly_rates_usd']['ram_gib']
    actual=elapsed*hourly/3600+quote['image_build_reserve_usd']+quote['receipt_egress_ceiling_usd']
    raw=raw or dict(receipts={},cpu_checks={},host_peak_fraction=None)
    fetch=fetch or dict(schema='m3-smoke-fetch.v2',status='FAILED',inputs={},error=dict(
        category='child_exit',stage='bootstrap',message='no structured fetch report received'))
    fetch.setdefault('stdout_tail',safe_fetch_text(''.join(stdout_tail)))
    fetch.setdefault('stderr_tail',safe_fetch_text(''.join(stderr_tail)))
    result=verdict(raw['receipts'],host_peak_fraction=raw['host_peak_fraction'],actual_cost_usd=actual,cpu_checks=raw['cpu_checks'])
    result.update(cost_basis='client wall time conservatively charged at full sandbox rate plus image-build and egress reserves',
        reservation_seconds=elapsed,image_build_reserve_usd=quote['image_build_reserve_usd'],
        resolved_modal_image_id=image_id,runtime_spec_sha256=partner_runtime_spec()['spec_sha256'],
        runtime_compatibility_pins=dict(COMPATIBILITY_PINS),image_deviation=plan['image_deviation'],
        fetch=fetch,training_network_blocked=False,training_network_open=True,training_offline_mode=offline_ready,
        training_offline_env=smoke_loopback_settings()['offline_env'],loopback=smoke_loopback_settings(),
        processes=raw.get('processes',{}),quote_sha256=sha_file(root/'quote.json'),
        sandbox_exitcode=exitcode,launch_error_type=error_type,receipts=raw['receipts'],cpu_checks=raw['cpu_checks'],
        attempt=attempt,attempts=attempt,reservation_attempts=1,retries=0,attempt_budget_usd=attempt_budget_usd,budget_scope='per-attempt manager ruling',actual_cost_usd=actual,prior_attempt_cost_usd=prior_cost,
        cumulative_cost_usd=prior_cost+actual,cumulative_quote_ceiling_usd=prior_cost+quote['total_usd'],
        container_stdout_tail=safe_fetch_text(''.join(stdout_tail)),container_stderr_tail=safe_fetch_text(''.join(stderr_tail)),
        total_ceiling_usd=quote['total_usd'],plan=plan)
    return result


def kit_file_hashes():
    root=Path(__file__).parent
    return {str(p.relative_to(root)):sha_file(p) for p in root.rglob('*')
            if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc' and p.name!='.DS_Store'}


def bundle_command(command, root):
    """Expand only the portable frozen bundle root, locally and in the sandbox."""
    return [part.replace('{bundle}',str(Path(root).resolve())) for part in command]


def validate_quote(path):
    """Validate priced maximum reservation, never assume credits or GPU-only cost."""
    quote=json.loads(Path(path).read_text())
    if (quote.get('schema')!='m3-smoke-quote.v1' or quote.get('source')!='https://modal.com/pricing'
            or quote.get('price_date')!='2026-10-29' or quote.get('billing_kind')!='Sandbox'
            or quote.get('resources')!={'gpu':2,'cpu':8,'ram_gib':64}
            or quote.get('attempts')!=1 or quote.get('retries')!=0):
        raise ValueError('quote source/date/resources/attempt authority differs')
    per_second={'gpu':.001097,'cpu':.00003942,'ram_gib':.00000667}
    if quote.get('per_second_rates_usd')!=per_second or quote.get('gpu_kind')!='H100 SXM5':
        raise ValueError('quote per-resource prices differ from the dated list')
    rates=quote['hourly_rates_usd']
    if any(not math.isclose(rates[k],v*3600,rel_tol=0,abs_tol=1e-10) for k,v in per_second.items()):
        raise ValueError('quote differs from the dated Modal Sandbox list prices')
    calculated=budget(dict(pricing=dict(rates,quote_sha256=sha_file(path)),
                           timeout_seconds=quote['timeout_seconds'],startup_seconds=quote['startup_seconds']))
    if quote.get('reserved_seconds')!=quote['timeout_seconds']+quote['startup_seconds']:
        raise ValueError('quote omits startup reservation')
    build=quote.get('image_build_reserve_usd',0.)
    total=calculated['total_ceiling_usd']+quote['receipt_egress_ceiling_usd']+build
    if (quote['receipt_egress_ceiling_usd']<.04*8/1024 or
            not math.isclose(quote['reservation_usd'],calculated['total_ceiling_usd'],abs_tol=1e-8)
            or not math.isclose(quote['total_usd'],total,abs_tol=1e-8) or total>CEILING_USD):
        raise ValueError('full reservation plus startup/receipt price differs or exceeds ten dollars')
    if (quote.get('image_builds')!=1 or quote.get('persistent_volumes')!=0 or build!=3.
            or quote.get('download_seconds')!=300 or quote['startup_seconds']!=600):
        raise ValueError('quote requires the partner image build reserve and priced in-container fetch')
    return quote


def preparation_plan(request_path, cache=None):
    """Read-only local plan: frozen text, code, fetch/hash policy and full quote."""
    request_path=Path(request_path).resolve()
    request=json.loads(request_path.read_text())
    quote=validate_quote(request_path.parent/request['quote_file'])
    blockers=[];bundle_plan=None
    try:
        bundle_plan=validate_launch_packet(request_path.parent/'packet.json')
    except (ValueError,KeyError,OSError) as error:
        blockers.append('Launch packet validation failed: '+type(error).__name__)
    result=dict(schema='m3-smoke-preparation-plan.v2',status='READY' if not blockers else 'BLOCKED',
        quote_valid=True,bundle_valid=bundle_plan is not None,quote_total_usd=quote['total_usd'],
        quote_sha256=sha_file(request_path.parent/request['quote_file']),bundle_plan=bundle_plan,blockers=blockers,
        input_bytes_verified=False,input_verification='Deferred to in-container pinned fetch, before tokenizer/model use.',
        training_runtime_validated=False,
        runtime_dependency_status='RESOLVED_BY_DECLARED_PINS_MANAGER_CPU_VERIFIED' if bundle_plan else 'UNVERIFIED_PACKET',
        runtime_import_verification=dict(source=IMAGE_EVIDENCE,verified_by='manager',scope='Modal CPU imports; GPU smoke unmeasured'),
        remote_calls=0,uploads=0,launches=0,scientific_evidence=False)

    archive=request_path.parent/'receipt.attempt1.json'
    if archive.is_file():
        prior=json.loads(archive.read_text())['actual_cost_usd']
        latest_attempt=3 if (request_path.parent/'receipt.attempt2.json').is_file() else 2
        if latest_attempt==3:prior+=json.loads((request_path.parent/'receipt.attempt2.json').read_text())['actual_cost_usd']
        result['manual_attempt'+str(latest_attempt)+'_budget']=dict(prior_cost_usd=prior,per_reservation_quote_usd=quote['total_usd'],
            cumulative_quote_usd=prior+quote['total_usd'],per_attempt_cap_usd=CEILING_USD,
            budget_scope='per-attempt manager ruling',within_per_attempt_cap=quote['total_usd']<=CEILING_USD)
    return result


def engineering_rows(tokenizer, arm):
    """Short explicit development fixtures; these certify mechanics, not pools.

    Domain labels exercise every production mask. Hand-authored demonstrations
    are not 27B targets, accepted replay pools or scientific eligibility evidence.
    All four arms share the exact task prefix and first four replay targets.
    """
    fixtures={
        'math_hard':('Compute the sum of one and one.',
            'Adding the first unit to the second gives two units in total. Answer: 2'),
        'instruction':('Give a short greeting.', 'Hello.'),
        'gsm8k':('A box has one red ball and one blue ball. How many balls?', 'There are two balls. Answer: 2'),
        'arc':('Which item is a mammal? A: cat. B: rock. Answer with the choice.', 'A'),
        'finqa':('Revenue was ten units and then twenty units. What is the increase?', 'The difference is ten units. Answer: 10'),
        'contract_nli':('Either party may terminate with notice. Claim: termination with notice is allowed.',
            'The clause explicitly permits either party to terminate when notice is given. Entailment')}
    shape=geometry(arm,3,2);rows=[]
    def row(domain, replay, update, position, identity):
        prompt,response=fixtures[domain]
        ids=tokenizer.encode(response,add_special_tokens=False)+[tokenizer.eos_token_id] if tokenizer is not None else None
        result=dict(id=identity,prompt=[dict(role='user',content=prompt)],
            data_source='p3b_replay' if replay else 'math_screen',
            reward_model=dict(style='rule',ground_truth='' if replay else '2'),
            extra_info=dict(index=identity,split='development',demonstration=response,m3_replay=replay,
                m3_domain=domain,
                m3_update=update,m3_position=position,technical_synthetic=True))
        if tokenizer is not None:result['extra_info'].update(m3_target_ids=ids,m3_eos_token_id=tokenizer.eos_token_id)
        return result
    domains=(['instruction']*shape['replay_rows'] if arm!='CR' else
             ['instruction']*4+['gsm8k']*2+['arc']*2+['finqa']*8+['contract_nli']*8)
    for update in (1,2):
        task_positions=[i for i in range(1,33) if arm in ('G','CR') or i not in (8,16,24,32)]
        batch=[row('math_hard',False,update,i,f'task-{update}-{i}') for i in task_positions]
        batch += [row(domain,True,update,len(batch)+i+1,f'{domain}-{update}-{i}') for i,domain in enumerate(domains)]
        # Metadata positions describe the frozen full accumulation order.
        for i,item in enumerate(batch,1):item['extra_info']['m3_position']=i
        rows.extend(batch)
    return rows


def stage_verified_file(source,target):
    # Keep one set of verified model pages in the container cgroup. Both paths
    # are in the same disposable work directory; never mutate either afterwards.
    if Path(target).exists():
        if sha_file(source)!=sha_file(target):raise ValueError('staged verified input changed')
        return
    try:os.link(source,target)
    except OSError:shutil.copyfile(source,target)


def prepare_public_bundle(request_path, cache, *, downloader=None, verified_inputs=None):
    """Container fetch/freeze helper. Never imports Modal, uploads or allocates.

    The optional downloader is an injectable CPU fixture boundary. Public fetches
    explicitly disable authentication. A complete bundle is published only once
    model/tokenizer, pinned source snapshot and shared runtime spec are available.
    """
    request_path=Path(request_path).resolve();cache=Path(cache).resolve()
    plan=preparation_plan(request_path,cache)  # quote/isolation first, before IO
    request=json.loads(request_path.read_text())
    if (cache/'bundle'/'bundle.json').exists():
        if plan['status']!='READY':raise ValueError('existing frozen bundle differs; never overwrite')
        return plan
    cache.mkdir(parents=True,exist_ok=True)
    if downloader is None:raise ValueError('a verified container downloader is required')
    sources=[dict(repo_id='Qwen/Qwen3-1.7B',revision=MODEL_REVISION,repo_type='model',
        local_dir=cache/'downloads'/'model',allow_patterns=['*.json','*.safetensors','*.model','vocab.*','merges.txt']),
        dict(repo_id='allenai/RLVR-IFeval',revision=DATASET_REVISION,repo_type='dataset',
        local_dir=cache/'downloads'/'instruction',allow_patterns=['*.parquet','*.jsonl','*.json'])]
    failures=[]
    for source in sources:
        try:downloader(**source,token=False,max_workers=2)
        except Exception as error:
            failures.append(dict(input=source['repo_type'],error=fetch_error(error,'verified_cache_recheck')))
    if failures:
        return dict(schema='m3-smoke-preparation.v1',status='BLOCKED',quote_valid=True,public_fetch_attempts=2,
            quote_total_usd=plan['quote_total_usd'],download_failures=failures,
            runtime_spec_frozen=True,
            bundle_frozen=False,modal_calls=0,uploads=0,launches=0)
    root=cache/'bundle';root.mkdir(exist_ok=True)
    model=root/'model';raw=cache/'downloads'/'model'
    if verified_inputs is None:raise ValueError('verified source inventory required')
    for name in verified_inputs['model']['files_sha256']:
        source=raw/name
        target=model/name;target.parent.mkdir(parents=True,exist_ok=True);stage_verified_file(source,target)
    if not (model/'config.json').is_file() or not list(model.glob('*.safetensors')):
        raise ValueError('pinned model snapshot is incomplete')
    dataset_files=[cache/'downloads'/'instruction'/name for name in verified_inputs['dataset']['files_sha256']]
    if not dataset_files:raise ValueError('pinned dataset snapshot is incomplete')
    source_manifest=validate_local_source(request_path.parent)
    for name in (LOCAL_SOURCE_JSONL,LOCAL_SOURCE_MANIFEST):
        target=root/name;target.parent.mkdir(parents=True,exist_ok=True)
        stage_verified_file(request_path.parent/name,target)
    # Downloaded Parquet bytes are hashed, never decoded in this container.
    if any(sha_file(source)!=source_manifest['parquet_sha256'] for source in dataset_files):
        raise ValueError('container source hash differs from local conversion binding')
    from transformers import AutoTokenizer
    from kit.m3_loss import audit_long_context_rows
    tokenizer=AutoTokenizer.from_pretrained(model,local_files_only=True)
    def write(name,value):
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value,sort_keys=True,indent=2)+'\n')
    write('m3-root.json',dict(schema='message3-root.v1',core_submitted=False,engineering_smoke=True))
    hashes={str(p.relative_to(model)):sha_file(p) for p in model.rglob('*') if p.is_file()
            and (p.name=='config.json' or p.suffix=='.safetensors' or p.name.endswith('.safetensors.index.json'))}
    write('inputs/base.json',dict(revision=MODEL_REVISION,identity=dict(model_file_hashes=hashes)))
    quote=json.loads((request_path.parent/request['quote_file']).read_text())
    shutil.copyfile(request_path.parent/request['quote_file'],root/'quote.json')
    shutil.copyfile(request_path.parent/'forecast-basis.json',root/'forecast-basis.json')
    commands={};audits={}
    recipe=json.loads((Path(__file__).parent/'resources/p3b/v4-recipes.json').read_text())['overrides']['D']
    for arm in ARMS:
        rows=json.loads((request_path.parent/'development-rows.json').read_text())[arm]
        for row in rows:
            extra=row['extra_info'];extra.update(m3_target_ids=tokenizer.encode(extra['demonstration'],add_special_tokens=False)+[tokenizer.eos_token_id],m3_eos_token_id=tokenizer.eos_token_id)
        audits[arm]=audit_long_context_rows(tokenizer,rows)
        data=root/'data'/arm;data.mkdir(parents=True,exist_ok=True)
        for name,subset in (('train',rows),('test',rows[:1])):
            write_jsonl(data/(name+'.jsonl'),subset)
        run='{bundle}/smoke';data_path='{bundle}/data/'+arm
        substitutions=dict(kit='/root/kit',sdpo='/root/SDPO',model='{bundle}/model',data=data_path,
                           run=run+'/'+arm,seed=101)
        overrides=[part.format(**substitutions) for part in recipe]
        changes={'trainer.n_gpus_per_node':2,'trainer.total_training_steps':2,'trainer.save_freq':-1,
            'trainer.default_local_dir':run+'/checkpoints/'+arm,
            'trainer.rollout_data_dir':run+'/rollouts/'+arm,'trainer.validation_data_dir':run+'/validation/'+arm,
            'data.train_files':[data_path+'/train.jsonl'],'data.val_files':[data_path+'/test.jsonl'],
            'data.train_batch_size':geometry(arm,3,2)['global_rows'],
            'actor_rollout_ref.actor.ppo_mini_batch_size':geometry(arm,3,2)['global_rows'],
            'actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu':1,
            'actor_rollout_ref.actor.use_dynamic_bsz':False,
            'actor_rollout_ref.nccl_timeout':SMOKE_INIT_TIMEOUT,
            'actor_rollout_ref.model.override_config.attn_implementation':'flash_attention_2'}
        overrides=[part for part in overrides if part.lstrip('+').split('=',1)[0] not in changes]
        overrides += ['++'+key+'='+json.dumps(value) for key,value in changes.items()]
        commands[arm]=['python','-m','kit.m3_entry','--arm',arm,'--stage','3','--seed','101',
            '--profile','technical-smoke','--base-model','{bundle}/model','--base-revision',MODEL_REVISION,
            '--receipt',run+'/'+arm+'.json','--config-name','sdpo',*overrides]
    write('development-inputs.json',dict(technical_synthetic=True,scientific_evidence=False,
        source_revision=DATASET_REVISION,source_use='Pinned source provenance only; hand-authored development fixtures exercise domain masks.',
        licences=dict(instruction='RLVR-IFeval ODC-BY; Allen Institute for AI',fixtures='Repository development fixtures'),
        teacher='Hand-authored development targets; never accepted as scientific 27B demonstrations.',length_audit=audits))
    cells={phase:dict(receipt='forecast-basis.json',field='measured_seconds.'+field,multiplier=multiplier)
           for phase,field,multiplier in [('startup','model-stand-in',2.),
            ('rollout','train-merge-D-chemistry-stand-in',.15),
            ('updates','train-merge-D-chemistry-stand-in',.65),
            ('export','train-merge-D-chemistry-stand-in',.2),
            ('teardown','check-launcher-records-D-chemistry-stand-in',10.)]}
    from copy import deepcopy
    forecast={arm:deepcopy(cells) for arm in ARMS};forecast['K']['updates']['multiplier']*=1.3
    bundle=dict(model_revision=MODEL_REVISION,runtime=partner_runtime_spec(),quote_file='quote.json',
        pricing=dict(quote['hourly_rates_usd'],quote_sha256=sha_file(root/'quote.json')),
        timeout_seconds=quote['timeout_seconds'],startup_seconds=quote['startup_seconds'],
        files={str(p.relative_to(root)):sha_file(p) for p in root.rglob('*') if p.is_file()},
        code_sha256={name:sha_file(Path(__file__).parent/name) for name in CODE_FILES},kit_files_sha256=kit_file_hashes(),
        commands=commands,phase_forecast=forecast)
    validate_bundle(bundle,root)
    write('bundle.json',bundle)
    return preparation_plan(request_path,cache)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('plan','execute','preflight','prepare','prepare-local','fetch','fetch-check','fetch-verify','bootstrap'))
    parser.add_argument('--bundle',type=Path)
    parser.add_argument('--request',type=Path)
    parser.add_argument('--cache',type=Path)
    parser.add_argument('--timeout',type=int,default=900,help='local fetch-check timeout in seconds')

    args=parser.parse_args(argv)
    if args.action=='prepare-local':
        if not args.request or not args.cache:parser.error('prepare-local needs --request and --cache')
        result=prepare_local(args.request,args.cache)
        print(json.dumps(result,sort_keys=True))
        if result['status']!='READY':raise SystemExit(2)
        return
    if args.action=='prepare':
        if not args.request:parser.error('prepare needs --request')
        result=freeze_launch_packet(args.request)
        print(json.dumps(result,sort_keys=True))
        if result['status']!='READY':raise SystemExit(2)
        return
    if args.action=='plan' and args.request:
        plan=preparation_plan(args.request,args.cache);print(json.dumps(plan,sort_keys=True))
        if plan['status']!='READY':raise SystemExit(2)
        return
    if not args.bundle:parser.error('this action needs --bundle')
    if args.action in ('fetch','fetch-check','fetch-verify'):
        if args.action in ('fetch-check','fetch-verify'):
            if not args.cache:parser.error('fetch-check requires --cache outside the git tree')
            if args.timeout<=0:parser.error('fetch-check timeout must be positive')
        try:
            if args.action=='fetch-check':
                result=run_fetch_child(args.bundle,args.cache,args.timeout,action='fetch-verify',cache=args.cache)
            elif args.action=='fetch-verify':result=fetch_public_inputs(args.bundle,args.cache)
            else:result=fetch_inside_container(args.bundle)
        except Exception as error:
            result=dict(schema='m3-smoke-fetch.v2',status='FAILED',inputs={},error=fetch_error(error,'fetch_entry'))
        print(json.dumps(result,sort_keys=True))
        if result['status']!='VERIFIED':raise SystemExit(2)
        return
    if args.action=='bootstrap':
        bootstrap(args.bundle);return
    if args.action=='preflight':
        print(json.dumps(run_preflight(args.bundle),sort_keys=True));return
    if args.action=='plan':
        print(json.dumps(validate_bundle(json.loads(args.bundle.read_text()),args.bundle.parent),sort_keys=True))
    else:
        execute(args.bundle)





def build_preflight_record(config, uninterrupted, prefix, resumed, input_manifest):
    """Combine measured four-update/fresh-process resume evidence; never science."""
    from kit.m3_contract import BASE_REVISION,code_digest,code_file_hashes,CODE_SHA256_SCOPE,PREFLIGHT_GATES
    for receipt in (uninterrupted,prefix,resumed):
        validate_receipt(receipt)
        if receipt['profile']!='preflight' or receipt['world_size']!=8 or receipt['arm']!=config['arm']:
            raise ValueError('preflight source receipt identity differs')
    updates=uninterrupted.get('updates',[])
    nonzero=[row for row in updates if row.get('learning_rate',0)>0]
    equality=bool(uninterrupted.get('state_hashes')) and uninterrupted['state_hashes']==resumed.get('state_hashes')
    required_states={'actor','ema','optimizer','scheduler','frozen_base'}
    equality=equality and set(uninterrupted.get('state_hashes',{}))==required_states
    legal=input_manifest.get('longest_legal_prompt_tokens',0)
    audits=[r.get('length_audit',{}).get('longest_by_domain',{}).get('contract_nli',{}).get('prompt_tokens') for r in (uninterrupted,prefix,resumed)]
    initial=updates[0].get('m3/initial_replay_kl_token_mean') if updates else None
    memory_gpu=max(r['telemetry']['gpu_peak_fraction'] for r in (uninterrupted,prefix,resumed))
    hosts=[r['telemetry'].get('cgroup_peak_fraction') for r in (uninterrupted,prefix,resumed)]
    host_peak=max(hosts) if all(v is not None for v in hosts) else None
    oracle_rows=[[v for v in r.get('updates',[]) if 'm3/gradient_relative_error' in v] for r in (uninterrupted,prefix,resumed)]
    oracle=[v['m3/gradient_relative_error'] for rows in oracle_rows for v in rows]
    oracle_valid=([len(rows) for rows in oracle_rows]==[1,1,0]
                  and all(v.get('learning_rate',0)>0 and 0<=v['m3/gradient_relative_error']<=1e-4 for rows in oracle_rows for v in rows))
    gates=dict(optimizer_materialized=all(r.get('optimizer_materialized') is True for r in (uninterrupted,prefix,resumed)),
        longest_legal_rows=legal>0 and audits==[legal]*3,
        checkpoint_resume_equal=equality and prefix['completed_updates']==2 and resumed['completed_updates']==4,
        finite_losses_gradients_ema=all(r['gates'].get('finite_losses_gradients_ema') is True for r in (uninterrupted,prefix,resumed)),
        masks=all(r['gates'].get('long_context') is True for r in (uninterrupted,prefix,resumed)),
        global_sum_over_32=len(updates)==4 and len(nonzero)==3 and oracle_valid,
        independent_gradient=oracle_valid,
        frozen_reference=all(r['frozen_base_before']==r['frozen_base_after'] and not r['frozen_base_has_grad'] for r in (uninterrupted,prefix,resumed)),
        gpu_memory=memory_gpu<=.90,cgroup_ram=host_peak is not None and host_peak<=.95,
        initial_replay_kl=config['arm']!='K' or initial is not None and -1e-5<=initial<=1e-3,
        cuda_full_kl=all(r.get('cuda_full_kl',{}).get('passed') is True for r in (uninterrupted,prefix,resumed)))
    assert set(gates)==set(PREFLIGHT_GATES)
    record=dict(schema='m3-node-preflight.v1',profile='preflight',scientific_evidence=False,
        arm=config['arm'],base_revision=BASE_REVISION,model_size='8B',world_size=8,
        status='PASS' if all(gates.values()) else 'FAIL',nonzero_lr_updates=len(nonzero),
        code_sha256=code_digest(),code_sha256_scope=CODE_SHA256_SCOPE,code_files_sha256=code_file_hashes(),
        input_manifest_sha256=sha_file(config['input_manifest']),source_schedule_audit_sha256=input_manifest['source_schedule_audit_sha256'],
        longest_legal_prompt_tokens=legal,longest_legal_row_id=input_manifest['longest_legal_row_id'],
        frozen_base_before=uninterrupted['frozen_base_before'],frozen_base_after=resumed['frozen_base_after'],
        frozen_base_has_grad=any(r['frozen_base_has_grad'] for r in (uninterrupted,prefix,resumed)),
        frozen_base_files_sha256=uninterrupted['frozen_base_files_sha256'],
        attention_implementation=uninterrupted['attention_implementation'],initial_replay_kl_token_mean=initial,
        gates=gates,state_hashes=uninterrupted.get('state_hashes'),telemetry=dict(gpu_peak_fraction=memory_gpu,
            cgroup_peak_fraction=host_peak,host_reservation_bytes=config['host_reservation_bytes'],
            nonzero_update_seconds=[v['m3/update_seconds'] for v in nonzero],
            oracle_gpu_peak_fraction=max(r['telemetry'].get('oracle_gpu_peak_fraction',0.) for r in (uninterrupted,prefix,resumed)),
            deterministic_algorithms_scope='training actor/EMA/base/oracle only; vLLM excluded; flags restored'))
    if (uninterrupted['frozen_base_files_sha256']!=resumed['frozen_base_files_sha256'] or
            uninterrupted['attention_implementation']!=resumed['attention_implementation']):
        record['status']='FAIL'
    if host_peak is None:record['status']='INCOMPLETE'
    return record


def archive_preflight_attempt(work, target):
    """Preserve failed/partial phase trees and their exact file hashes for retry."""
    work=Path(work);target=Path(target)
    present=[work/name for name in ('uninterrupted','prefix','resume','failure.json') if (work/name).exists()]
    if target.exists():present.append(target)
    if not present:return None
    attempts=work/'attempts';attempts.mkdir(parents=True,exist_ok=True)
    index=1
    while (attempts/f'attempt-{index}').exists():index+=1
    archive=attempts/f'attempt-{index}';archive.mkdir()
    hashes={}
    for source in present:
        destination=archive/('qualification.json' if source==target else source.name)
        for file in ([source] if source.is_file() else source.rglob('*')):
            if file.is_file():hashes[str(destination.relative_to(archive)/file.relative_to(source)) if source.is_dir() else destination.name]=sha_file(file)
        shutil.move(str(source),destination)
    (archive/'archive.json').write_text(json.dumps(dict(schema='m3-preflight-attempt.v1',files=hashes),indent=2,sort_keys=True)+'\n')
    return archive


def run_preflight(bundle_path, *, run_command=None):
    """Three isolated child processes, stage-2 legal preferred; no node-wide stop.

    m3a2 owns gpu_execution and the node-plan producer. The shared input schema
    lives in m3_contract. Legal length is derived only from its frozen schedule
    audit. Failed/partial phases are archived on retry; a passing qualification
    stays immutable. Each child's address=local Ray session shuts itself down.
    """
    from kit.m3_contract import (CONFIG_KEYS,find_m3_root,verify_base_files,validate_preflight,
                                 validate_preflight_inputs,PREFLIGHT_INPUT_STAGES)
    run_command=run_command or subprocess.run
    bundle_path=Path(bundle_path).resolve();bundle=json.loads(bundle_path.read_text())
    command=bundle['command']
    if command[1:3]!=['-m','kit.m3_entry']:raise ValueError('preflight requires the production M3 entry')
    config,overrides=parse_args(command[3:])
    if config['profile']!='preflight' or config['stage'] not in PREFLIGHT_INPUT_STAGES:
        raise ValueError('node preflight requires eight-rank legal stage 2 or stage 3 inputs')
    root=find_m3_root(config['receipt'],require=True)
    verify_base_files(config['base_model'],root/'inputs/base.json',config['base_revision'])
    manifest=validate_preflight_inputs(config['input_manifest'],arm=config['arm'],stage=config['stage'],root=root)
    work=root/'preflight'/config['arm'];target=root/'preflight'/(config['arm']+'.json')
    if target.is_file():
        prior=json.loads(target.read_text())
        if prior.get('status')=='PASS':
            validate_preflight(prior,config['arm'])
            if prior['input_manifest_sha256']!=sha_file(config['input_manifest']):raise ValueError('passing preflight input identity is immutable')
            return prior
    archive_preflight_attempt(work,target)
    records={};phase='uninterrupted'
    try:
        for phase in ('uninterrupted','prefix','resume'):
            receipt=work/phase/'mechanics.json';checkpoint=work/phase/'checkpoints'
            settings=dict(config,preflight_phase=phase,receipt=str(receipt))
            flags=[]
            for key in CONFIG_KEYS:
                value=settings[key]
                if value is not None:flags.extend(['--'+key.replace('_','-'),str(value)])
            steps=2 if phase=='prefix' else 4
            hydra=[*overrides,'++trainer.total_training_steps='+str(steps),'++trainer.save_freq=2',
                   '++trainer.default_local_dir='+json.dumps(str(checkpoint)),'++trainer.del_local_ckpt_after_load=false']
            if phase=='resume':hydra+=['++trainer.resume_mode=resume_path','++trainer.resume_from_path='+json.dumps(str(work/'prefix'/'checkpoints'/'global_step_2'))]
            else:hydra+=['++trainer.resume_mode=disable']
            run_command([command[0],'-m','kit.m3_entry',*flags,*hydra],check=True)
            records[phase]=json.loads(receipt.read_text())
        result=build_preflight_record(config,records['uninterrupted'],records['prefix'],records['resume'],manifest)
        if result['status']=='PASS':validate_preflight(result,config['arm'])
    except Exception as error:
        work.mkdir(parents=True,exist_ok=True)
        (work/'failure.json').write_text(json.dumps(dict(schema='m3-preflight-failure.v1',phase=phase,error_type=type(error).__name__))+'\n')
        raise
    target.parent.mkdir(parents=True,exist_ok=True)
    temporary=target.with_suffix('.json.tmp');temporary.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');temporary.replace(target)
    return result


if __name__=='__main__': main()
