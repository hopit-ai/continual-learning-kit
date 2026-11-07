from pathlib import Path
import os
import subprocess
import sys
import json
import pytest
from kit.nm_common import *
from kit.nm_jobs import inventory,current


@pytest.mark.parametrize('gpus',[0,1,8])
@pytest.mark.parametrize('allocation,holder',[(dict(),False),(dict(UTIL_HOLDER_SLICE='1'),True),
    (dict(SLURM_GPUS_ON_NODE='8'),True),(dict(SLURM_JOB_GPUS='2-5'),True),
    (dict(SLURM_STEP_GPUS='GPU-a,GPU-b'),True),(dict(SLURM_JOB_GPUS='5'),False)])
def test_watcher_holder_scope_matches_row_and_child_exit_is_preserved(tmp_path,monkeypatch,gpus,allocation,holder):
    from types import SimpleNamespace
    from kit import nm_jobs,a4_common
    config=tmp_path/'config.json';atomic(config,dict(out=str(tmp_path/'out'),technical_synthetic=True))
    row=dict(id='NM-opr-fixture',family='opr',phase='generate',gpus=gpus,depends_on=[])
    monkeypatch.setattr(nm_jobs,'row_for',lambda *a:row)
    for key in ('UTIL_HOLDER_SLICE','SLURM_GPUS_ON_NODE','SLURM_JOB_GPUS','SLURM_STEP_GPUS'):monkeypatch.delenv(key,raising=False)
    for key,value in allocation.items():monkeypatch.setenv(key,value)
    monkeypatch.delenv('UTIL_WATCH_ACTIVE',raising=False)
    def child(command,*,env,pass_fds):
        scope=command[command.index('--expect-gpus'):command.index('--')]
        assert scope==['--expect-gpus',str(gpus),*(['--holder-slice'] if gpus==1 and holder else [])]
        assert ('UTIL_HOLDER_SLICE' in env)==(gpus==1 and 'UTIL_HOLDER_SLICE' in allocation)
        return SimpleNamespace(returncode=37)
    monkeypatch.setattr(a4_common,'gpu_run',child)
    assert nm_jobs.run(config,row['id'])==37
    assert current(tmp_path/'out',row['id'],complete=False)['exit_code']==37


def invoke(config,task):
    return subprocess.run(['bash',str(KIT/'jobs/nm/run-task.sh'),task],env=dict(os.environ,KIT=str(KIT),NM_CONFIG=str(config),JOB_PY=sys.executable),capture_output=True,text=True)


def test_manifest_is_explicit_topological_right_sized_and_frozen_file():
    inv=inventory();seen=set();counts={}
    for r in inv['rows']:
        assert r['id'] not in seen and all(d['id'] in seen for d in r['depends_on'])
        assert r['gpus'] in (0,1,2,8) and r['nodes']==1
        assert r['may_share_node']==(r['gpus']!=8)
        assert '--kill-on-invalid-dep=yes' in r['dependency_options']
        assert (r['phase'] in ('train','qualify','calibrate','estimate','estimator-preflight'))==(r['gpus']==8)
        seen.add(r['id']);counts[r['family']]=counts.get(r['family'],0)+1
    assert set(counts)==set(FAMILIES)
    assert load(KIT/'resources/nm/rows.json')==json.loads(json.dumps(inv))
    for seed in SEEDS:
        assert sum(r['phase']=='generate' and r.get('seed')==seed for r in inv['rows'])==1


def test_real_wrapper_every_row_pending_failure_resubmit_and_watcher(tmp_path):
    config=tmp_path/'config.json';out=tmp_path/'out';cfg=dict(out=str(out),technical_synthetic=True,transport_stress=True,fail_task='NM-grpo-C-s101-train')
    atomic(config,cfg)
    # Missing dependency exits promptly and retains a nonce before resubmission.
    missing=invoke(config,'NM-freeze-C-s101-export')
    assert missing.returncode==75,missing.stdout+missing.stderr
    failed_nonce=current(out,'NM-freeze-C-s101-export',complete=False)['nonce']
    receipts=[]
    for row in inventory()['rows']:
        result=invoke(config,row['id'])
        if row['id']==cfg['fail_task']:
            assert result.returncode==2,result.stdout+result.stderr
            bad=current(out,row['id'],complete=False);blocked=invoke(config,'NM-grpo-C-s101-export')
            assert blocked.returncode==75
            result=invoke(config,row['id'])
            assert current(out,row['id'])['nonce']!=bad['nonce']
        assert result.returncode==0,(row['id'],result.returncode,result.stdout,result.stderr)
        receipt=current(out,row['id']);receipts.append(receipt)
        summary=load(out/'attempts'/row['id']/receipt['nonce']/'util.summary.json')
        assert summary['expected_gpus']==row['gpus'] and summary['gpus']==row['gpus']
        assert summary['telemetry_complete'] and summary['returncode']==0
        if row['phase']=='score':
            cells=load(out/'attempts'/row['id']/receipt['nonce']/'scores.json')['cells']
            assert {c['uuid'] for c in cells}==set(summary['gpu_uuids'])
    assert current(out,'NM-freeze-C-s101-export')['nonce']!=failed_nonce
    assert len({r['nonce'] for r in receipts})==len(receipts)
    for family in FAMILIES:
        r=current(out,'NM-'+family+'-read');root=out/'attempts'/r['task']/r['nonce']
        reading=load(root/'reading.json')
        assert reading['status']=='TECHNICAL-ONLY'
        assert all(len(line)<=90 for line in (root/'summary.txt').read_text().splitlines())
        c=current(out,'NM-'+family+'-collect');returned=out/'attempts'/c['task']/c['nonce']
        from kit.text_return import parse_parts
        cycle=parse_parts([(returned/'return.txt').read_text()])
        imported=cycle['families'][family]
        assert imported['status']=='TECHNICAL-ONLY' and imported['endpoint_verdict_rederived']
        assert imported['simulated_status']==reading['simulated_status']
        assert imported['metadata']['account']['attempts']>=len([r for r in receipts if r['family']==family])
        assert imported['metadata']['account']['gpu_hours']==0
        assert all(len(line)<90 for line in (returned/'return.txt').read_text().splitlines())
        assert len(list((returned/'text').glob('RETURN-TEXT-*.md')))<=4
        assert not (returned/'return.tar.gz').exists()
    cycle=load(out/'return-cycle.json')
    assert cycle['families']==list(FAMILIES) and len(cycle['parts'])<=4
    assert cycle['characters']==sum(len((out/p).read_text()) for p in cycle['parts'])
    assert all(len((out/p).read_text())<=11400 for p in cycle['parts'])
    final=parse_parts([(out/p).read_text() for p in cycle['parts']])
    assert len(final['families']['opr']['metadata']['disagree'])==66
    atomic(tmp_path/'dry-run-summary.json',dict(rows=len(receipts),attempts=len(list((out/'attempts').glob('*/*/receipt.json'))),
        characters=cycle['characters'],parts=len(cycle['parts']),part_characters=[len((out/p).read_text()) for p in cycle['parts']],
        max_line=max(len(line) for p in cycle['parts'] for line in (out/p).read_text().splitlines())))


def test_archive_paths_no_challenger_read(tmp_path):
    config=tmp_path/'config.json';out=tmp_path/'out'
    atomic(config,dict(out=str(out),technical_synthetic=True,archive=['opr','isdft']))
    for family in ('opr','isdft'):
        for row in inventory()['rows']:
            if row['family']!=family:continue
            result=invoke(config,row['id']);assert result.returncode==0,result.stdout+result.stderr
            assert current(out,row['id'])['status']=='ARCHIVED UNREAD'
    assert not list(out.glob('**/reading.json'))


def test_explicit_pending_row_resubmits_fresh_attempt(tmp_path):
    config=tmp_path/'config.json';out=tmp_path/'out';cfg=dict(out=str(out),technical_synthetic=True,pending_task='NM-grpo-bind')
    atomic(config,cfg);first=invoke(config,'NM-grpo-bind');assert first.returncode==75
    old=current(out,'NM-grpo-bind',complete=False)['nonce'];cfg.pop('pending_task');atomic(config,cfg)
    assert invoke(config,'NM-grpo-bind').returncode==0
    assert current(out,'NM-grpo-bind')['nonce']!=old


def test_child_cannot_run_unwatched(tmp_path):
    p=tmp_path/'config.json';atomic(p,dict(out=str(tmp_path/'out'),technical_synthetic=True))
    result=subprocess.run([sys.executable,'-I',str(KIT/'nm_execute.py'),'--config',str(p),'--task','NM-grpo-bind','--attempt',str(tmp_path/'attempt')],env={k:v for k,v in os.environ.items() if k!='UTIL_WATCH_ACTIVE'},capture_output=True,text=True)
    assert result.returncode!=0 and 'requires watcher' in result.stderr


@pytest.mark.parametrize('settings,blocked',[({'opr_successes':3},'refresh'),({'opr_cohort_size':99},'cohort')])
def test_opr_scientific_shortfalls_through_real_wrapper(tmp_path,settings,blocked):
    config=tmp_path/'config.json';out=tmp_path/'out';atomic(config,dict(out=str(out),technical_synthetic=True,**settings))
    for task in ('bind','prepare','generation-qualify-s101','generate-s101','cohort-s101','refresh-s101'):
        result=invoke(config,'NM-opr-'+task)
        if task.startswith(blocked):
            assert result.returncode==75,result.stdout+result.stderr
            break
        assert result.returncode==0,result.stdout+result.stderr
    assert not list((out/'state').glob('*train.json'))


def test_invalid_hardware_receipt_through_real_reader_and_return(tmp_path):
    config=tmp_path/'config.json';out=tmp_path/'out'
    atomic(config,dict(out=str(out),technical_synthetic=True,invalid_identity_task='NM-freeze-s101-instruction',return_archive=True))
    for row in inventory()['rows']:
        if row['family']!='freeze':continue
        result=invoke(config,row['id']);assert result.returncode==0,result.stdout+result.stderr
    r=current(out,'NM-freeze-read');reading=load(out/'attempts'/r['task']/r['nonce']/'reading.json')
    assert reading['status']=='UNVERIFIED-HARDWARE'
    c=current(out,'NM-freeze-collect')
    from kit.nm_return import reproduce
    assert reproduce(out/'attempts'/c['task']/c['nonce']/'return.tar.gz')['status']=='UNVERIFIED-HARDWARE'


def test_score_prepare_depends_only_on_own_seed_exports():
    for row in inventory()['rows']:
        if row['phase']!='score-prepare':continue
        deps={d['id'] for d in row['depends_on']};prefix='NM-'+row['family']+'-'
        assert {d for d in deps if d.endswith('-export')}=={prefix+a+f'-s{row["seed"]}-export' for a in ('C','T')}
        assert all(not d.endswith('-export') or f'-s{row["seed"]}-' in d for d in deps)
        if row['family']=='v2b' and row['panel'] not in (*GENERAL,'hellaswag'):
            assert prefix+'september-contexts' in deps


def test_grpo_insensitive_D_completes_real_rows_and_text_only_cpu_return(tmp_path):
    config=tmp_path/'config.json';out=tmp_path/'out'
    atomic(config,dict(out=str(out),technical_synthetic=True,grpo_D_insensitive=True))
    for row in inventory()['rows']:
        if row['family']!='grpo':continue
        result=invoke(config,row['id']);assert result.returncode==0,(row['id'],result.stdout,result.stderr)
    assert current(out,'NM-grpo-read')['status']=='COMPLETE'
    receipt=current(out,'NM-grpo-collect');assert receipt['status']=='COMPLETE'
    cycle=load(out/'return-cycle.json');paths=[out/p for p in cycle['parts']]
    local=tmp_path/'local.json'
    proc=subprocess.run([sys.executable,'-I',str(KIT/'nm_return.py'),'--text',*map(str,paths),'--out',str(local)],capture_output=True,text=True)
    assert proc.returncode==0,proc.stderr
    imported=load(local)['families']['grpo']
    assert imported['status']=='PENDING/INCOMPLETE' and imported['endpoint_verdict_rederived']
    assert imported['endpoints']['D_sensitivity']['values']==[0.]*3
    reading=load(out/'attempts'/'NM-grpo-read'/current(out,'NM-grpo-read')['nonce']/'reading.json')
    assert imported['primary']==reading['primary'] and imported['guards']==reading['guards']


def test_inherited_watcher_marker_warns_and_runs_scientific_child(tmp_path,monkeypatch,capsys):
    from types import SimpleNamespace
    from kit import nm_jobs,a4_common
    config=tmp_path/'config.json';out=tmp_path/'out'
    atomic(config,dict(out=str(out),technical_synthetic=True))
    row=dict(id='NM-opr-fixture',family='opr',phase='generate',gpus=1,depends_on=[])
    monkeypatch.setattr(nm_jobs,'row_for',lambda *a:row)
    monkeypatch.setenv('UTIL_WATCH_ACTIVE','1')
    calls=[]
    def child(command,*,env,pass_fds):
        assert 'UTIL_WATCH_ACTIVE' not in env
        assert command.count(str(KIT/'util_watch.py'))==1
        calls.append(command)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(a4_common,'gpu_run',child)
    assert nm_jobs.run(config,row['id'])==0 and len(calls)==1
    record=current(out,row['id'])
    assert record['utilisation_warnings']==['INHERITED_UTIL_WATCH_ACTIVE_DROPPED']
    assert 'W|NM-opr|INHERITED_UTIL_WATCH_ACTIVE_DROPPED|1' in capsys.readouterr().out
