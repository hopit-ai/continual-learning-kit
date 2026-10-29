#!/usr/bin/env python3
"""Every README-message row via real stand-in jobs, failure and durable resume included."""
from pathlib import Path
import sys
if __package__ in (None, ''): sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kit.c_common import *
from kit.m2_seed_policy import SEEDS as RETURN1_SEEDS
from kit.next_dryrun import invoke
from kit import next_dryrun, message_handoff, c_train
from types import SimpleNamespace as NS
from contextlib import redirect_stdout
from collections import Counter
import shutil
import subprocess
import tarfile
from unittest.mock import patch


def interrupt_save(out, durable):
    """A terminated copy must leave both the commit pointer and end receipt alone."""
    previous=(durable/'CURRENT.json').read_bytes()
    end=(durable/'last-job-end.json').read_bytes()
    (out/'C/interrupted-save-only').write_text('uncommitted local evidence')
    def terminated(command, **kwargs):
        target=Path(command[-1])
        (target/'half-copied-file').write_bytes(b'partial new generation')
        raise subprocess.CalledProcessError(143,command)
    with patch.object(message_handoff.subprocess,'run',side_effect=terminated):
        try: message_handoff.save(out,durable)
        except subprocess.CalledProcessError: pass
        else: raise AssertionError('interrupted save succeeded')
    require((durable/'CURRENT.json').read_bytes()==previous,'interrupted save published a generation')
    require((durable/'last-job-end.json').read_bytes()==end,'interrupted save changed the end marker')
    require(bool(list((durable/'generations').glob('.partial-*'))),'missing interrupted partial copy')


def storage_rehearsal(root):
    """Measure a large stable checkpoint plus a tiny changed receipt."""
    out=root/'out';out.mkdir(parents=True);durable=root/'durable'
    (out/'stable-checkpoint').write_bytes(b'x'*(16*1024*1024))
    (out/'receipt').write_bytes(b'first')
    first=message_handoff.save(out,durable)
    (out/'receipt').write_bytes(b'changed')
    second=message_handoff.save(out,durable)
    skipped=message_handoff.save(out,durable)
    require(second['bytes_copied']==7 and skipped['bytes_copied']==0,'incremental copy scaled with OUT')
    return dict(unchanged_checkpoint_bytes=16*1024*1024,
                bytes_per_save=[v['bytes_copied'] for v in (first,second,skipped)],unchanged_skipped=skipped['skipped'])


def job(a):
    require(a.standin,'rehearsal is stand-in only')
    require(not (a.out/'message-summary.json').exists(),'fresh rehearsal OUT required')
    a.out.mkdir(parents=True,exist_ok=True)
    durable=a.out.parent/(a.out.name+'-durable')
    require(not durable.exists(),'fresh durable rehearsal tree required')
    counts=Counter();failures=[];saves=[]
    def call(name,root,*args): return invoke(name,['--out',root,'--standin',*args],counts)
    def hand(action):
        if action=='save':
            receipt=message_handoff.save(a.out,durable);saves.append(receipt);print(receipt)
        else: message_handoff.job(NS(out=a.out,durable=durable,action=action,gpu='0',standin=True))
    def failure(label,expected,fn):
        try: fn()
        except expected: failures.append(label)
        else: raise AssertionError('failure path failed to stop: '+label)
    with (a.out/'message-dryrun.log').open('w') as log, redirect_stdout(log):
        hand('restore');hand('identity')
        next_dryrun.job(NS(out=a.out,standin=True,defer_pools=True))  # M2/M3: all 60 finals, N1–N6/N10–N12
        nxt=a.out/'NEXT';out=a.out/'C';fixtures=a.out/'fixtures'
        # M3b: the quick teacher job and its compact return travel with parent OUT.
        call('math_hard_teacher',a.out)
        from kit.math_hard_teacher_read import read as teacher_read
        require(teacher_read(a.out/'math-hard-teacher')['complete'],'incomplete hard-MATH teacher stand-in')
        with tarfile.open(a.out/'math-hard-teacher-return.tar.gz','w:gz') as archive:
            archive.add(a.out/'math-hard-teacher',arcname='math-hard-teacher')
        from kit.p3b_read import load,analyse
        rows,_=load(fixtures/'B-reading.csv')
        atomic(fixtures/'b-reading.json',dict(analyse(rows,20),technical_synthetic=True))
        inputs=json.loads((nxt/'next-inputs/inputs.json').read_text())
        starts={'base':dict(model=inputs['base']['path'],receipt=inputs['base']['receipt'])}
        for slot in inputs['finals']:
            if slot['route']=='finqa' and slot['recipe'] in ('D','F'):
                starts[f"{slot['recipe']}:{slot['seed']}"]=dict(model=slot['final']['path'],receipt=slot['final']['receipt'])
        atomic(fixtures/'c-starts.json',starts)
        call('c_bind',out,'--b-reading',fixtures/'b-reading.json','--b-source',fixtures/'B-reading.csv','--checkpoints',fixtures/'c-starts.json')
        doc=binding(out,True)
        require(doc['W']==inputs['w'],'C/next W mismatch')
        call('c_targets',out,'--prepare-only')
        call('c_targets',out)
        # M5 follows the two-GPU target gap, before any legal training.
        call('next_pools',nxt,'--package-a',fixtures/'A','--finqa-panel',fixtures/'finqa-panel.jsonl',
            '--finqa-targets',fixtures/'finqa-targets','--gsm-rows',fixtures/'gsm.jsonl','--arc-rows',fixtures/'arc.jsonl')
        shutil.copytree(nxt/'hellaswag',out/'hellaswag')
        for panel in ('panel','test','finqa','general-v1','hellaswag'):
            call('c_score',out,'--arm','base','--seed',101,'--step',0,'--panel-set',panel)
        def scores(arm,seed,steps,panels):
            for step in steps:
                for panel in panels:call('c_score',out,'--arm',arm,'--seed',seed,'--step',step,'--panel-set',panel)
        def train(recipe,route,seed,probe=None):
            args=['--recipe',recipe,'--route',route,'--seed',seed,'--replay',nxt/'replay']
            if probe:args+=['--probe-arm',probe,'--pool-root',nxt/'pools']
            call('c_train',out,*args)
        # M6: exercise a failed stage before completion, save, lose local OUT, restore, resume.
        first='standalone-'+arms(doc)[0]
        scores(first,101,(0,),('panel','test','finqa','general-v1'))
        original=c_train.training_rows
        def interrupted(*args): raise RuntimeError('stand-in interrupted trainer')
        c_train.training_rows=interrupted
        try:failure('interrupted training retained before completion',RuntimeError,lambda:train(arms(doc)[0],'standalone',101))
        finally:c_train.training_rows=original
        hand('save')
        snapshot={str(p.relative_to(out)):s.file_hash(p) for p in out.rglob('*') if p.is_file()}
        interrupt_save(a.out,durable)
        shutil.rmtree(out);shutil.rmtree(nxt)
        hand('restore');hand('identity')
        require(not (out/'interrupted-save-only').exists(),'restored uncommitted evidence')
        require(all(s.file_hash(out/name)==sha for name,sha in snapshot.items()),'restored C bytes differ')
        for recipe in arms(doc):
            for route in ('standalone','sequence'):
                for seed in RETURN1_SEEDS:
                    arm=route+'-'+recipe
                    scores(arm,seed,(0,),('panel','test','finqa','general-v1'))
                    train(recipe,route,seed)
                    scores(arm,seed,(10,20,30,40),('panel','test'))
                    scores(arm,seed,(20,40),('finqa','general-v1'))
        # M7: all four legal probes. Shared generic calls must SKIP an existing W run.
        for probe in ('generic','broad','earlier','both'):
            for seed in RETURN1_SEEDS:
                arm=probe_arm(doc,probe)
                scores(arm,seed,(0,),('panel','test','finqa','general-v1','hellaswag'))
                train('W','sequence',seed,probe)
                scores(arm,seed,(10,20,30,40),('panel','test'))
                scores(arm,seed,(20,40),('finqa','general-v1','hellaswag'))
        # M8/M9/M13: readers, verified durable copy, both compact returns and round trips.
        call('c_read',out/'reading','--source',out,'--bootstrap',20)
        call('c_probe_read',out/'reading','--source',out,'--bootstrap',20)
        require(json.loads((out/'reading/reading.json').read_text())['complete'],'incomplete core')
        hand('save')
        call('c_collect',out,'--archive',a.out/'package-c-return.tar.gz')
        call('next_collect',nxt,'--durable-root',message_handoff.committed(durable)/'tree/NEXT','--archive',a.out/'next-return-1.tar.gz')
        call('c_read',a.out/'returned-C','--source',a.out/'package-c-return.tar.gz','--bootstrap',20)
        call('c_probe_read',a.out/'returned-C','--source',a.out/'package-c-return.tar.gz','--bootstrap',20)
        for name in ('reading.json','reading-probe.json'):
            require(json.loads((out/'reading'/name).read_text())==json.loads((a.out/'returned-C'/name).read_text()),'archive reading differs: '+name)
        # Completed rows are idempotent across job boundaries.
        before={str(p.relative_to(out/'runs')):s.file_hash(p) for p in (out/'runs').rglob('*') if p.is_file()}
        hand('save');hand('restore')
        train('W','sequence',101,'both')
        scores(probe_arm(doc,'both'),101,(40,),('test',))
        require(before=={str(p.relative_to(out/'runs')):s.file_hash(p) for p in (out/'runs').rglob('*') if p.is_file()},'completed training changed on resume')
        # M10/M11: absent kit is an explicit pending gate; no 4A science is fabricated.
        def gate():
            if not (a.out/'absent-4A/README-4a.md').is_file():raise NotReady('4A kit absent')
        failure('4A requested without kit',NotReady,gate)
        # M1 changed host/GPU: archive entire scoring epoch, then rerun base and references.
        old_epoch=json.loads((a.out/'scoring-epoch.json').read_text())
        changed=dict(uuid='STANDIN-OTHER-GPU',host='STANDIN-OTHER-HOST',environment='CPU-STANDIN')
        require(message_handoff.epoch(a.out,changed),'changed GPU failed to start a new epoch')
        require(not (out/'scores').exists() and not (nxt/'scores').exists(),'mixed active scoring epochs')
        failure('comparison before new-GPU rescore',NotReady,lambda:call('c_probe_read',out/'reading','--source',out,'--bootstrap',20))
        history=a.out/'scoring-history'/old_epoch['epoch']
        require((history/'C/scores').is_dir() and (history/'NEXT/blends').is_dir(),'old GPU evidence lost')
        # Re-score every v0+/v3 base, endpoint reference and interior on the new epoch too.
        from kit import next_blend,next_score
        old_blend,old_score=next_blend.run,next_score.run
        def nested(command): return invoke(Path(command[1]).stem,command[2:],counts)
        next_blend.run=next_score.run=nested
        try:
            for seed,parent in [(0,inputs['base'])]+list(zip(SEEDS,inputs['sec'])):
                call('next_sec',nxt,'--seed',seed,'--checkpoint-path',parent['path'],'--checkpoint-receipt',parent['receipt'],
                    '--collector-checkpoint-path',parent['path'],'--collector-checkpoint-receipt',parent['receipt'])
            # All lambda=0/1 references precede interior comparisons.
            for lam in (0,1,.25,.5,.75):
                for slot in inputs['finals']:
                    call('next_blend',nxt,'--slot',slot['name'],'--lam',lam,'--nvme-root',a.out,'--package-a',fixtures/'A')
            call('next_read',nxt/'reading','--source',nxt,'--bootstrap',20)
        finally: next_blend.run,next_score.run=old_blend,old_score
        # Full new-GPU C scoring, preserving every training weight/receipt.
        for panel in ('panel','test','finqa','general-v1','hellaswag'):scores('base',101,(0,),(panel,))
        score_labels=[f'{route}-{recipe}' for recipe in arms(doc) for route in ('standalone','sequence')]
        score_labels+= [probe_arm(doc,p) for p in ('generic','broad','earlier','both')]
        for arm in dict.fromkeys(score_labels):
            for seed in RETURN1_SEEDS:
                scores(arm,seed,(0,10,20,30,40),('panel','test'))
                retention=('finqa','general-v1') + (('hellaswag',) if arm in [probe_arm(doc,p) for p in ('generic','broad','earlier','both')] else ())
                scores(arm,seed,(0,20,40),retention)
        call('c_read',out/'reading','--source',out,'--bootstrap',20)
        call('c_probe_read',out/'reading','--source',out,'--bootstrap',20)
        hand('save')
    storage=storage_rehearsal(a.out.parent/(a.out.name+"-storage"))
    result=dict(storage=storage,bytes_per_save=[v["bytes_copied"] for v in saves],standin_only=True,gpu_hours=0,all_message_rows=True,W=doc['W'],command_counts=dict(counts),
        core_and_probe_stages=len(list((out/'runs').glob('*/complete.json'))),active_C_score_jobs=len(list((out/'scores').glob('*/manifest.json'))),
        screen=json.loads((a.out/'dryrun-summary.json').read_text()),failure_paths=failures,
        resume_across_jobs=True,mid_save_interruption=True,restored_hashes=len(snapshot),changed_gpu_full_C_rescore=True,changed_gpu_full_screen_rescore=True,
        old_scoring_epoch_preserved=True,optional_4A='absent gate tested; no scientific payload',
        compact_archive_bytes={p.name:p.stat().st_size for p in (a.out/'package-c-return.tar.gz',a.out/'next-return-1.tar.gz')})
    atomic(a.out/'message-summary.json',result);print(json.dumps(result,sort_keys=True))


def configure(p):pass
if __name__=='__main__':raise SystemExit(entry(job,configure,'message-dryrun'))
