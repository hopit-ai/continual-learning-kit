"""Execution of the staged K/CR-only manifest. No scheduler calls or core gates."""
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import uuid
from contextlib import nullcontext

from kit.combined_send import KIT, load, sha, need, host, write
from kit.p3b_common import NotReady


def rescore_concurrency(ready, commands):
    """Generation K=3; any likelihood work caps the measured 8B lane at K=2."""
    known={'finqa','test','math_hard','general-v1','hellaswag','knowledge-ll'}
    for command in commands:
        if '--panel-set' not in command or command[command.index('--panel-set')+1] not in known:return 1
        for flag in ('--max-model-len','--max_model_len'):
            if flag in command and int(command[command.index(flag)+1])>12288:return 1
    checkpoints=load(ready)['checkpoints']
    if not checkpoints:return 1
    for item in checkpoints.values():
        try:config=load(Path(item['source']['model'])/'config.json')
        except (OSError,ValueError):return 1
        if not (config.get('model_type')=='qwen3' and config.get('hidden_size')==4096 and
                config.get('num_hidden_layers')==36 and config.get('intermediate_size')==12288):return 1
    from kit.lane_pool import likelihood_command
    return 2 if any(likelihood_command(dict(argv=c)) for c in commands) else 3


def rescore_plan(ready, commands):
    """Describe the CPU-verified engine profile to the shared capacity guard."""
    if rescore_concurrency(ready,commands)==1:return commands
    model=next(iter(load(ready)['checkpoints'].values()))['source']['model']
    return [dict(argv=c,engine=dict(backend='vllm',model=model,max_model_len=12288)) for c in commands]


def execute(doc, ident, *, runner=subprocess.run):
    from kit.m3_manifest import allocated_gpus, allocation_reservation, substitute, require_m3_root, Unbound
    from kit.m3_score import prepare_workspace, unit_root, prepared_commands
    from kit.lane_pool import execute as pool
    from kit.lane_evidence import complete_attempt
    s=doc['settings'];row=next(r for r in doc['tasks'] if r['id']==ident)
    root=Path(s['kcr_root']);control=Path(s['control_root'])
    nonce=uuid.uuid4().hex;started=time.time()
    env={k:v for k,v in os.environ.items() if not k.startswith('M2_') and
         k not in ('MANIFEST','DURABLE','LOCAL_ROOT','REF_POOL_CHILD','REF_ATTEMPT_ROOT','M3_GPU_LOCK_FDS')}
    env.update(KIT_V6A='1',KIT_LANE_JOB=nonce,M3_OUT=str(root),M3_MANIFEST=s['kcr_manifest'],
               INFERENCE_PY=s['inference_python'],M2_INFRA_AUTHORITY_SHA256='')
    state=control/'state'/(ident+'.json')
    attempt_state=control/'state'/'attempts'/ident/(nonce+'.json')
    def finish(code, **detail):
        attempt=dict(nonce=nonce,status='DONE' if code==0 else 'PENDING' if code in (3,75) else 'FAILED',
                     started_at=started,ended_at=time.time(),returncode=code,**detail)
        write(attempt_state,attempt)
        temporary=state.with_name(ident+'.'+nonce+'.tmp')
        write(temporary,dict(attempt,attempt_state=str(attempt_state)))
        temporary.replace(state)
        return code
    workspace=None
    try:
        need(not doc.get('technical_synthetic'), 'synthetic inventory is CPU rehearsal only')
        need(sha(s['kcr_manifest'])==doc['sources']['kcr_manifest'],'source manifest changed: kcr_manifest')
        need(host(socket.gethostname()) in doc['nodes']['kcr'],'v6a must run on a configured K/CR node')
        from kit.v6_infra import activate_v6a
        authority=activate_v6a()
        require_m3_root(root)
        env['M2_INFRA_AUTHORITY_SHA256']=authority
        blocked=[r for r in row['requires'] if not (control/'state'/(r+'.json')).is_file() or
                 load(control/'state'/(r+'.json')).get('status')!='DONE']
        if blocked and row['dependency']!='afterany':return finish(75,blocked_by=blocked)
        values=dict(kit=KIT,m3_out=root,score_gpu=os.environ.get('SCORE_GPU','0'))
        commands=[substitute(c,values) for c in row['commands']]
        # Use the specified inference environment, including for copied M3 commands.
        for command in commands:
            command[0]=s['inference_python']
        if ident=='REPORT-text':
            inputs=s.get('report_inputs',str(control/'report-inputs.local.json'))
            if not s.get('report_inputs'):
                write(inputs,dict(_mode='v6a',KCR=[dict(source=str(root))],
                    jobs=[dict(family='KCR',manifest=os.environ['COMBINED_MANIFEST'],state=str(control/'state'))],
                    utilisation=[dict(family='KCR',roots=[str(control/'utilisation'/'kcr')]),
                                 dict(family='REF',roots=[str(control/'utilisation'/'reference')])]))
            output=Path(s.get('text_out',str(control/'return-text')))/nonce
            command=[s['inference_python'],'-I',str(KIT/'report_text.py'),'--inputs',inputs,'--out',str(output)]
            code=runner(command,env=env).returncode
            return finish(code,text_out=str(output))
        if row['group']=='kcr' and row['gpus']:
            workspace=prepare_workspace(root,root/'scoring-jobs'/ident/nonce)
            for command in commands:command[command.index('--out')+1]=str(workspace)
            ready=row['scoring_ready'].replace('{m3_out}',str(root))
            if not Path(ready).is_file():return finish(75,reason='CPU scoring preparation missing')
            snapshot=workspace/'scoring-ready.local.json'
            snapshot.write_bytes(Path(ready).read_bytes());ready=str(snapshot)
            commands=prepared_commands(commands,ready)
            env['M3_SCORE_READY']=ready
        with allocation_reservation(allocated_gpus(1)) if row['gpus'] else nullcontext(()) as fds:
            if row['gpus']:
                env.update(M3_GPU_LOCK_FDS=json.dumps(fds),M3_NODE_POLICY=json.dumps(dict(nodes=doc['nodes']['kcr'])))
            if workspace:
                concurrency=rescore_concurrency(ready,commands)
                code=pool(rescore_plan(ready,commands),workspace/'pool',env=env,pass_fds=fds,runner=runner,concurrency=concurrency)
                if code==0:
                    identities=[]
                    for c in commands:
                        def value(flag):return c[c.index(flag)+1]
                        cell=unit_root(workspace,value('--arm'),int(value('--seed')),value('--stage'),int(value('--step')),value('--panel-set'))
                        identities.append(str((cell/'m3-identity.json').relative_to(workspace)))
                    complete_attempt(workspace,nonce,identities,extra_files=['scoring-ready.local.json'])
            else:
                code=0
                for command in commands:
                    code=runner(command,env=env,pass_fds=fds).returncode
                    if code:break
        return finish(code,**(dict(workspace=str(workspace)) if workspace else {}))
    except (NotReady,FileNotFoundError,Unbound) as error:return finish(75,reason=str(error))
    except Exception as error:return finish(1,error=type(error).__name__+': '+str(error))
