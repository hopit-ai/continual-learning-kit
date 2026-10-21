#!/usr/bin/env python3
"""Bind C-selected W, audited slice, measured cost ceiling and physical scorer before output."""
from pathlib import Path
import sys
if __package__ in (None, ''): sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kit.a4_common import *
import math
import argparse
import subprocess


def ceiling(stage_gpu_hours): return 50*math.ceil((1.5*45*stage_gpu_hours+140)/50)


def selected_exports(files,recipe,standin):
    """Pin the original B finals that support W's selection, without using their weights as starts."""
    if standin:return {}
    result={}
    for route in ('standalone','sequence'):
        for seed in SEEDS:
            name=f'scores/{route}-{recipe}-s{seed}-k40-math_screen/identity.json'
            if name not in files:raise NotReady('original unreset B export identity pending: '+name)
            record=json.loads(files[name]);observed=record.get('model',{})
            origin=Path(observed.get('model_dir',''))
            require(origin.name=='hf-step40' and origin.parent.name==f'{route}-{recipe}-s{seed}',
                'B selection identity is not the original unreset final: '+name)
            require(bool(observed.get('model_file_hashes')) and not record.get('technical_synthetic'),
                'B final lacks real export file identities')
            result[f'{route}:{seed}']=dict(identity=observed,score_identity_sha256=t.sha(files[name]))
    return result


def job(a):
    from kit.next_w import job as select_w
    from types import SimpleNamespace as NS
    require(a.slice.is_dir(), 'missing audited slice')
    if not a.b_source.exists(): raise NotReady('completed unreset B required to bind W')
    select_w(NS(out=a.out/'selection', source=a.b_source, bootstrap=20, standin=a.standin))
    selected = json.loads((a.out/'selection/W.json').read_text())
    from kit.p3b_read import load as load_b
    b_rows,b_files=load_b(a.b_source)
    require(s.digest(b_rows)==selected['source_sha256'],'B evidence changed while binding W')
    b_exports=selected_exports(b_files,selected['recipe'],a.standin)
    expected = json.loads((RES/'slice.json').read_text())
    check(RES/'registration.md', expected['registration_sha256'])
    files = {p.name:s.file_hash(p) for p in sorted(a.slice.iterdir()) if p.is_file()}
    if not a.standin:
        require(files == expected['files'], 'audited primary slice differs; stop for prospective repair')
    manifest = json.loads((a.slice/'manifest.json').read_text())
    require(manifest['months'] == list(MONTHS), 'six registered months required')
    base = dict(model=str(a.base))
    observed = model_identity(base, a.standin)
    require(a.stage_gpu_hours > 0, 'measured stage GPU-hours required')
    cap = ceiling(a.stage_gpu_hours)
    approved = getattr(a, 'approved_ceiling', APPROVED_CEILINGS[selected['recipe']])
    approval_reference = getattr(a, 'approval_reference', APPROVAL_REFERENCE)
    approved_ceiling(dict(W=selected['recipe'], approved_ceiling_gpu_hours=approved,
                          approval_reference=approval_reference))
    timing=getattr(a,'timing_receipt',None)
    if not a.standin:require(timing is not None and timing.is_file(),'hash-bound measured W timing receipt required')
    if timing:
        measured=json.loads(timing.read_text())
        require(measured['recipe']==selected['recipe'] and measured['gpus']==8 and measured['seconds']>0,'wrong W timing receipt')
        require(abs(8*measured['seconds']/3600-a.stage_gpu_hours)<1e-6,'stage GPU-hours differ from receipt')
    from kit import score_forgetting as general
    if a.standin:
        retention = {}
    else:
        retention = dict(general_v1=general.load_panel()[1])
        # Re-run source/unit bindings and arithmetic before any output.
        for month in MONTHS:
            for item in json.loads((a.slice/f'batch-{month}.json').read_text()): sec.check_v4_item(item, finqa)
    replay = None
    if '+replay' in selected['recipe']:
        require(a.replay is not None, 'W requires the frozen checked B generic pool')
        m = frozen_files(a.replay)
        require(m['technical_synthetic'] == a.standin, 'replay synthetic mismatch')
        pool = t.read_jsonl(a.replay/'replay.jsonl')
        require(bool(pool), 'empty replay pool')
        evaluation = [dict(id=i['id'],prompt=finqa.render_prompt(i)) for mo in MONTHS
                      for i in json.loads((a.slice/f'batch-{mo}.json').read_text())]
        if not a.standin:
            evaluation += general.load_panel()[0]
        _, excluded = overlap(pool, evaluation)
        atomic(a.out/'replay-overlap.json', dict(excluded=excluded, checked=len(pool)))
        require(not excluded, 'replay overlaps 4A or retention panels')
        replay = dict(path=str(a.replay.resolve()), manifest=s.file_hash(a.replay/'manifest.json'))
    trainer_versions=t.runtime_versions() if a.standin else json.loads(subprocess.check_output([
        a.trainer_python,'-c','import json; from kit.v4_teacher import runtime_versions; print(json.dumps(runtime_versions()))'],
        cwd=KIT.parent,text=True))
    doc = dict(W=selected['recipe'], provisional=selected['provisional'], selection=selected,
        B_exports=b_exports,B_evidence_files={n:t.sha(b) for n,b in sorted(b_files.items())
            if n.endswith(('identity.json','export-40.json','scores.csv','scores.csv.gz'))},
        base=base, base_identity=observed, slice=str(a.slice.resolve()), slice_files=files,
        retention=retention, replay=replay, registration_sha256=expected['registration_sha256'],
        stage_gpu_hours=a.stage_gpu_hours, ceiling_gpu_hours=cap, approved_ceiling_gpu_hours=approved,
        approval_reference=approval_reference, approval_registration=APPROVAL_REGISTRATION,
        timing_sha256=s.file_hash(timing) if timing else None,
        allocation_gpus=getattr(a,'allocation_gpus',8),
        job_limits=dict(training=1.5*a.stage_gpu_hours*3600/8,
            scoring=getattr(a,'score_max_seconds',600),nll=getattr(a,'nll_max_seconds',600)),
        handoff_seconds=getattr(a,'handoff_seconds',300),
        software=dict(inference=t.runtime_versions(),trainer=trainer_versions),
        input_cap=10240, output_cap=2048, incumbent='static-k3', protocol=protocol(),
        technical_synthetic=a.standin, physical=machine(a.out,a.gpu,a.standin))
    require(doc['allocation_gpus']==8 and doc['handoff_seconds']>0 and
        all(v>0 and math.isfinite(v) for v in doc['job_limits'].values()),'invalid allocation/job limits')
    bind(a.out/'binding.json', doc)
    print('4A W =', doc['W'], '; approved ceiling =', approved, 'GPU-h; formula forecast =', cap, 'GPU-h; v1 absent')


def configure(p):
    p.add_argument('--b-source',type=Path,required=True); p.add_argument('--slice',type=Path,required=True)
    p.add_argument('--base',required=True); p.add_argument('--replay',type=Path)
    p.add_argument('--stage-gpu-hours',type=float,required=True); p.add_argument('--approved-ceiling',type=float,default=argparse.SUPPRESS)
    p.add_argument('--approval-reference',default=argparse.SUPPRESS)
    p.add_argument('--gpu',default='0')
    p.add_argument('--timing-receipt',type=Path)
    p.add_argument('--allocation-gpus',type=int,default=8)
    p.add_argument('--score-max-seconds',type=float,default=600)
    p.add_argument('--nll-max-seconds',type=float,default=600)
    p.add_argument('--handoff-seconds',type=float,default=300)
    p.add_argument('--trainer-python',default=sys.executable)
if __name__ == '__main__': raise SystemExit(entry(job, configure, 'a4-bind'))
