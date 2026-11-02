#!/usr/bin/env python3
"""CPU-only v6a rehearsal: real shell/controller/pool/watcher/readers, fake scores.

All fake likelihoods and generated outputs are explicitly technical_synthetic.
No CUDA, network, scheduler, training or external writes are used.
"""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import argparse
import hashlib
import json
import os
import shutil
import socket
import subprocess
from types import SimpleNamespace as NS
from unittest.mock import patch
from contextlib import ExitStack

from kit import combined_send as c,m3_manifest,m3_schedule as schedule
from kit.p3b_common import atomic,files_manifest


def fixture_uuid(directory):
    # Independent rehearsals must not contend on the same host-local fake lease.
    return 'GPU-v6a-'+hashlib.sha256(str(Path(directory).resolve()).encode()).hexdigest()[:16]


def pinned_base_config():
    return c.load(c.KIT/'resources/m2/arc-prompt-attempt2.json')['selection']['inputs']['base']['model_config']


def fixtures(directory):
    directory=Path(directory).resolve();directory.mkdir(parents=True,exist_ok=False)
    root=directory/'kcr';m3_manifest.init_root(root,core_submitted=False)
    model=directory/'model'
    atomic(model/'config.json',pinned_base_config())
    from kit.m3_train import bind_base
    bind_base(NS(out=root,model=model,receipt=None,standin=True))
    # bind_base's fixture only needs its original empty config; keep the measured
    # architecture in this labelled fake input for the real concurrency selector.
    atomic(model/'config.json',pinned_base_config())
    from kit.m2_assignment import freeze
    freeze(root/'m3-scoring-assignment.json',dict(schema='message2-scoring-assignment.v2',assignments={
        f'm3|{seed}':dict(node='CPU-STANDIN',uuid='CPU-STANDIN',environment='CPU-STANDIN') for seed in schedule.SEEDS}))
    for arm in schedule.ARMS:
        for seed in schedule.SEEDS:
            for stage in schedule.STAGES:
                run=root/'runs'/f'{arm}-s{seed}-{stage}'
                names=['complete.json'];atomic(run/'complete.json',dict(technical_synthetic=True))
                for step in (10,20,30,40):
                    checkpoint=run/f'hf-step{step}'
                    atomic(checkpoint/'config.json',c.load(model/'config.json'))
                    name=f'export-{step}.json';names.append(name)
                    atomic(run/name,dict(technical_synthetic=True,model_file_hashes={'config.json':c.sha(checkpoint/'config.json')}))
                atomic(run/'manifest.json',files_manifest(run,names))
    atomic(root/'hellaswag/manifest.json',dict(technical_synthetic=True))
    (root/'hellaswag/panel.jsonl').write_text(''.join(json.dumps(dict(id=f'hs-{i}',gold='A'))+'\n' for i in range(100)))
    atomic(root/'pools/availability.json',{a:dict(status='READY') for a in schedule.ARMS})
    # Deliberately failed/absent native validation must not block reference guards.
    atomic(root/'format-validation/m3-s101.json',dict(status='PENDING'))
    shim=directory/'python-standin'
    shim.write_text('#!'+sys.executable+'\nimport sys\nsys.path.insert(0,'+repr(str(ROOT))+')\n'
        'from scripts.v6a_dryrun import bridge\nraise SystemExit(bridge(sys.argv[1:]))\n')
    shim.chmod(0o755)
    binary=directory/'bin/nvidia-smi';binary.parent.mkdir()
    gpu=fixture_uuid(directory)
    binary.write_text('#!'+sys.executable+'\nimport sys\n'
        'a=" ".join(sys.argv[1:])\n'
        'if "query-compute-apps" in a:\n'
        ' from pathlib import Path\n'
        ' with Path('+repr(str(directory/'idle-checks.jsonl'))+').open("a") as f:f.write("idle\\n")\n'
        'elif "index,uuid,utilization" in a:print('+repr('5, '+gpu+', 88, 1024')+')\n'
        'elif "index,uuid" in a:print('+repr('5, '+gpu)+')\n'
        'else:print('+repr(gpu)+')\n')
    binary.chmod(0o755)
    tier=m3_manifest.build(dict(nodes=[socket.gethostname().split('.')[0]],trainer_python=str(shim),
        inference_python=str(shim),inputs=dict(contract_targets='/CPU-STANDIN/C',replay='/CPU-STANDIN/B',panels={},inventory='/CPU-STANDIN/inventory'),
        technical_synthetic=False))
    c.write(root/'m3-launch.json',tier)
    settings=dict(kcr_manifest=str(root/'m3-launch.json'),kcr_root=str(root),control_root=str(directory/'control'),
                  inference_python=str(shim))
    doc=c.build_v6a(settings);manifest=directory/'v6a.json';c.write(manifest,doc)
    return doc,manifest,shim


def fake_score(args):
    """Synthetic output writer; the actual native reader verifies these schemas."""
    from kit import m3_score,m2_assignment
    from kit.p3b_score import write_csv
    from kit.c_score import COLUMNS
    scenario=os.environ.get('V6A_CPU_SCENARIO')
    if args.arm=='base' and args.panel_set=='test' and scenario in ('pending','failure'):
        return 75 if scenario=='pending' else 7
    assert os.environ['UTIL_WATCH_ACTIVE']=='1'
    # The real inheritance validator must accept the parent's still-live fd.
    with m3_manifest.gpu_execution(args.out):pass
    root=m3_score.unit_root(args.out,args.arm,args.seed,args.stage,args.step,args.panel_set)
    proof=m3_score.proof_for(args.out,args.gpu,args.seed,True)
    _,observed=m3_score.checked(args.out,args.arm,args.seed,args.stage,args.step,True)
    metrics=('ifeval','math','knowledge') if args.panel_set=='general-v1' else (args.panel_set,)
    rows=[dict(arm=m3_score.unit_arm(args.arm,args.stage),seed=args.seed,step=args.step,panel=metric,task=metric,
               cap=2048,question_id=f'{metric}-{i}',correct=int(i<80),output_tokens=10,finished=1)
          for metric in metrics for i in range(100)]
    atomic(root/'m3-identity.json',dict(assignment=proof,gpu_identity=proof,model=observed,
        arm=args.arm,seed=args.seed,stage=args.stage,step=args.step,panel=args.panel_set,technical_synthetic=True))
    write_csv(root/'scores.csv',[{k:r.get(k,'') for k in COLUMNS} for r in rows],COLUMNS)
    atomic(root/'cost.json',dict(kind='CPU-STANDIN',gpu_hours=0,technical_synthetic=True))
    atomic(root/'manifest.json',files_manifest(root,['m3-identity.json','scores.csv','cost.json']))
    return 0


def bridge(argv):
    """Explicit interpreter shim used only by this harness, never production."""
    argv=list(argv)
    if argv[:1]==['-I']:argv=argv[1:]
    if argv[:1]==['-c']:
        sys.argv=['-c',*argv[2:]];exec(argv[1],{'__name__':'__main__'});return 0
    target=Path(argv[0]).name;args=argv[1:]
    from kit import lane_pool,m2_assignment,m3_score
    def measured(*a,**kw):
        return dict(node=socket.gethostname().split('.')[0],uuid=lane_pool.gpu_uuid(),environment='CPU-STANDIN')
    original=lane_pool.invocation
    def fast(*a,**kw):kw['stagger_seconds']=0;return original(*a,**kw)
    with ExitStack() as stack:
        stack.enter_context(patch.object(m2_assignment,'inventory',measured))
        stack.enter_context(patch.object(lane_pool,'invocation',fast))
        stack.enter_context(patch.object(sys,'argv',[argv[0],*args]))
        if target=='util_watch.py':
            from kit.util_watch import main
            return main(['--interval','.02','--window','.05',*args])
        if target=='combined_send.py':return c.main()
        if target=='m3_score.py':
            p=argparse.ArgumentParser();p.add_argument('--out',type=Path);p.add_argument('--standin',action='store_true')
            m3_score.configure(p);a=p.parse_args(args);a.standin=True
            if a.action=='prepare-score':m3_score.prepare_scoring(a);return 0
            return fake_score(a)
        if target=='ref_score_job.py':
            from kit import ref_score_job as ref
            # CPU preparation and the real reference writer run normally on the
            # fake checkpoints. Only likelihood arithmetic is replaced.
            if '--standin' not in sys.argv:sys.argv.append('--standin')
            action=args[args.index('--action')+1]
            if action=='score' and '--cell-index' not in args:
                a=NS(out=Path(args[args.index('--out')+1]),family='m3',seed=int(args[args.index('--seed')+1]),
                     checkpoints=Path(args[args.index('--checkpoints')+1]),panel='both',gpu='0',standin=True)
                ref.run_lane(a);return 0
            if action=='score' and os.environ.get('V6A_CPU_SCENARIO')=='failure' and args[args.index('--cell-index')+1]=='1':return 7
            try:return ref.main()
            except __import__('kit.p3b_common',fromlist=['NotReady']).NotReady:return 3
        if target=='report_text.py':
            from kit.report_text import main
            return main([*args,'--standin','--kit-tag','kit-message-v6a','--kit-commit','a'*40])
        raise ValueError('unexpected CPU shim target: '+target)


def run(directory, missing_intermediates=False):
    doc,manifest,shim=fixtures(directory);directory=Path(directory).resolve()
    if missing_intermediates:
        for run_root in (directory/'kcr/runs').iterdir():
            for step in (10,20,30):
                (run_root/f'export-{step}.json').unlink()
                shutil.rmtree(run_root/f'hf-step{step}')
    # The test reads an authority as fixture input; production receives this hash
    # only from the operator's message and never derives its own trust anchor.
    from kit.v6_infra import authority_sha256
    env={k:v for k,v in os.environ.items() if not k.startswith(('M2_','M3_','KIT_','SLURM_','CUDA_','UTIL_','REF_'))}
    env.update(KIT=str(c.KIT),COMBINED_MANIFEST=str(manifest),JOB_PY=str(shim),
        KIT_V6_AUTHORITY_SHA=authority_sha256(),PATH=str(directory/'bin')+os.pathsep+os.environ['PATH'],
        CUDA_VISIBLE_DEVICES='0',SLURM_JOB_GPUS='5',SLURM_GPUS_ON_NODE='1')
    calls=[]
    def row(ident,scenario='pass',expected=0):
        done=subprocess.run(['bash',str(c.KIT/'jobs/v6a/run.sh'),ident],env=dict(env,V6A_CPU_SCENARIO=scenario),
                            capture_output=True,text=True,timeout=120)
        calls.append(dict(row=ident,scenario=scenario,returncode=done.returncode))
        if done.returncode!=expected:
            state=Path(doc['settings']['control_root'])/'state'/(ident+'.json')
            raise AssertionError((calls[-1],done.stdout,done.stderr,state.read_text() if state.exists() else 'no state'))
    row('M3-rescore-101',expected=75)  # PENDING before readiness, no GPU admission
    row('REF-m3-assign')
    for seed in schedule.SEEDS:
        row(f'M3-rescore-{seed}-ready');row(f'REF-m3-{seed}-prepare')
    row('M3-rescore-101','pending',75)  # incomplete nonce retained, then full retry
    for seed in schedule.SEEDS:row(f'M3-rescore-{seed}')
    row('REF-m3-102','failure',1)
    row('REF-m3-101');row('REF-m3-103');row('REPORT-text')  # afterany, partial return
    from kit.text_return import parse_parts
    control=Path(doc['settings']['control_root'])
    partial=Path(c.load(control/'state/REPORT-text.json')['text_out'])
    partial_result=parse_parts([p.read_text() for p in sorted(partial.glob('RETURN-TEXT-*.md'))])
    assert all(d['verdict']=='PENDING' for d in partial_result['decisions'] if d['family']=='KCR')
    row('REF-m3-102');row('REPORT-text')
    control=Path(doc['settings']['control_root']);state=c.load(control/'state/REPORT-text.json')
    text_root=Path(state['text_out']);parts=sorted(text_root.glob('RETURN-TEXT-*.md'))
    parsed=parse_parts([p.read_text() for p in parts])
    assert len(parts)==3 and parsed['complete']
    decisions=[d for d in parsed['decisions'] if d['family']=='KCR']
    assert {d['comparison'] for d in decisions}=={'K-C0','CR-G'}
    assert all(d['verdict']=='INCONCLUSIVE' for d in decisions),decisions
    if missing_intermediates:
        reading=c.load(directory/'kcr/m3-read.json')
        assert reading['missing_intermediate_exports']=={str(s):36 for s in schedule.SEEDS}
        for comparison in ('K-C0','CR-G'):
            diagnostics=reading['contrasts'][comparison]['diagnostics']
            assert all(v['status']=='INCOMPLETE' and v['note']=='MISSING-EXPORT'
                       for k,v in diagnostics.items() if 'gain_auc' in k or 'step20' in k)
        first=parts[0].read_text()
        assert all(f'W|KCR|MISSING-EXPORT-seed-{s}|36' in first for s in schedule.SEEDS)
    telemetry=[c.load(p) for p in (control/'utilisation').rglob('*.summary.json')]
    assert all(d['gpu_uuids']==[fixture_uuid(directory)] for d in telemetry if d['gpus']==1 and d['telemetry_complete'])
    assert all(d['telemetry_complete'] for d in telemetry if d['gpus']==1 and d['returncode']==0)
    assert all(d['mean_util'] is None for d in telemetry if not d['telemetry_complete'])
    pools=[c.load(p) for p in (directory/'kcr').rglob('lane.json')]
    assert {r['K'] for r in pools}=={2,3}
    assert len(list((directory/'kcr/scoring-jobs/M3-rescore-101').glob('*')))==2
    assert len(list((directory/'kcr/reference-attempts/m3/102').glob('*')))==2
    assert len((directory/'idle-checks.jsonl').read_text().splitlines())==8
    result=dict(rows=len(doc['tasks']),wrapper_calls=len(calls),pending=sum(c['returncode']==75 for c in calls),
        failed=sum(c['returncode']==1 for c in calls),parts=len(parts),KCR=decisions,
        native_commands=sum(len(p['commands']) for p in pools if p['K']==3 and p['status']=='COMPLETE'),
        reference_commands=sum(len(p['commands']) for p in pools if p['K']==2 and p['status']=='COMPLETE'),
        unknown_telemetry=sum(not d['telemetry_complete'] for d in telemetry),
        technical_synthetic=True,text_out=str(text_root),calls=calls)
    c.write(directory/'dryrun-result.json',result);return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--missing-intermediates',action='store_true')
    a=parser.parse_args();result=run(a.out,a.missing_intermediates)
    print(json.dumps({k:v for k,v in result.items() if k not in ('KCR','calls')},sort_keys=True));return 0


if __name__=='__main__':raise SystemExit(main())
