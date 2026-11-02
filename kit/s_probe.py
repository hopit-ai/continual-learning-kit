#!/usr/bin/env python3
"""Independent four-update incoming forks, with separate training and assigned NLL jobs."""
from pathlib import Path
import sys
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kit.s_common import *
from kit.s_generate import selected, domains_for
from kit.s_drift import nll
import shutil
import subprocess
import time


def job(a):
    from kit import m2_train
    if m2_train.run_prepared(a):return
    doc=context(a.out,a.standin)
    require(a.incoming in {c['incoming'] for c in doc['spec']['stages']},'unknown incoming')
    require(a.domain in domains_for(doc['spec'],a.incoming),'unregistered probe domain')
    if getattr(a,'phase','all')!='score':require(len(set(a.gpus.split(',')))==8,'probe requires eight distinct GPUs')
    checkpoint=doc['spec']['nodes'][a.incoming]['checkpoint'];identity=model(checkpoint,a.standin)
    require(identity==doc['identities'][a.incoming],'probe incoming changed')
    if not a.standin:
        from kit.m2_paths import local
        checkpoint=dict(checkpoint,model=str(local(checkpoint['model'])))
    own=selected(a.out,a.incoming,a.domain);root=a.out/'probes'/a.incoming/a.domain
    inputs=dict(incoming=identity,targets={k:digest(v) for k,v in own.items()},registration_sha256=sha(RESOURCE),technical_synthetic=a.standin)
    bind(root/'inputs.json',inputs)
    phase=getattr(a,'phase','all')
    from kit.m2_assignment import cache_identity
    family=None if phase=='train' else cache_identity(a.out)
    suffix='-'+family['family']+'-s'+str(family['seed']) if family else ''
    receipt_path=root/('receipt'+suffix+'.json');work=root/'work'
    if receipt_path.exists():
        require(read(receipt_path)['inputs']==inputs,'probe receipt identity changed')
        if family:
            from kit.m2_assignment import active
            require(read(receipt_path).get('assignment')==active(a.out,a.score_gpu,a.standin),'UNVERIFIED: stale probe assignment')
        if not family and work.exists():shutil.rmtree(work)
        print('SKIP probe:',a.incoming,a.domain);return
    if len(own['train'])!=128 or len(own['monitor'])!=64:
        bind(receipt_path,dict(inputs=inputs,assignment=__import__('kit.m2_assignment',fromlist=['active']).active(a.out,a.score_gpu,a.standin) if family else None,status='UNINFORMATIVE',reason='requires 128 training and 64 monitoring rows'));return
    start=time.monotonic()
    if phase in ('all','train') and not (work/'export.json').exists():
        # A partial fork restarts the identical four updates, never chains or extends.
        if work.exists() and not m2_train.published_training(a):shutil.rmtree(work)
        work.mkdir(parents=True,exist_ok=True)
        if a.standin:
            atomic(work/'hf-probe/config.json',dict(technical_synthetic=True))
            atomic(work/'training-receipt.json',dict(updates=4,batch=32,lrs=[0.,1e-6,2e-6,3e-6]))
            export={'config.json':sha(work/'hf-probe/config.json')}
        else:
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=a.gpus,PYTHONPATH=str(ROOT))
            command=[a.trainer_python,'-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node=8',
                str(ROOT/'kit/s_probe_train.py'),'--model',checkpoint['model'],'--schedule',str(a.out/'own'/a.incoming/a.domain/'train.jsonl'),
                '--output',str(work/'hf-probe'),'--seed','101']
            if m2_train.preparing():
                m2_train.prepare_training(a,command,env,work,[work/'native-probe.pt',work/'training-receipt.json'])
                return
            if not m2_train.published_training(a):subprocess.run(command,env=env,check=True)
            from kit.m2_train import defer_exports
            if not defer_exports():
                if (work/'native-probe.pt').exists():
                    import torch
                    from transformers import AutoConfig,AutoModelForCausalLM
                    with torch.device('meta'):net=AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(work/'hf-probe',local_files_only=True))
                    state=torch.load(work/'native-probe.pt',map_location='cpu',weights_only=True)
                    net.save_pretrained(work/'hf-probe',state_dict=state,safe_serialization=True)
                from kit.p3b_common import identity as export_identity
                export=export_identity(work/'hf-probe')['model_file_hashes']
        from kit.m2_train import defer_exports,queue_export
        if defer_exports():queue_export(a.out,'probe-export',work=str(work),standin=a.standin)
        else:atomic(work/'export.json',dict(model_file_hashes=export))
    train_seconds=(read(work/'trainer-exit.json')['seconds'] if m2_train.published_training(a) else time.monotonic()-start) if phase in ('all','train') else 0
    if phase in ('all','train') and not (root/'training-cost.json').exists():
        bind(root/'training-cost.json',dict(seconds=train_seconds,gpu_hours=0 if a.standin else 8*train_seconds/3600,technical_synthetic=a.standin,kind='independent_probe_training'))
    if phase=='train':return
    scoring_start=time.monotonic()
    require((work/'export.json').exists(),'probe training snapshot pending')
    from kit.m2_handoff import scoring_identity
    machine=scoring_identity(a.out,a.score_gpu,a.standin)
    os.environ['CUDA_VISIBLE_DEVICES']=a.score_gpu
    measurement=getattr(a,'measurement',None)
    before_path=work/('before'+suffix+'.json');after_path=work/('after'+suffix+'.json')
    if measurement=='reduce':
        before=read(before_path);after=read(after_path)
    else:
        if measurement in (None,'before'):
            before=nll(checkpoint.get('model','standin'),own['monitor'],a.standin,checkpoint.get('receipt'))
            if measurement:bind(before_path,before)
        if measurement in (None,'after'):
            if a.standin:
                base=nll(checkpoint.get('model','standin'),own['monitor'],True,checkpoint.get('receipt'))
                after=dict(base,token_mean=base['token_mean']-.1,sequence_mean=base['sequence_mean']-.1)
                if measurement:bind(after_path,after)
            else:
                schedule=a.out/'own'/a.incoming/a.domain/'monitor.jsonl'
                subprocess.run([a.inference_python,str(ROOT/'kit/a4_nll.py'),'--out',str(a.out),'--model',str(work/'hf-probe'),
                    '--receipt',str(work/'export.json'),'--schedule',str(schedule),'--output',str(after_path),'--gpu',a.score_gpu],check=True)
                after=read(after_path)
        if measurement:return
    bind(receipt_path,dict(inputs=inputs,assignment=machine.get('assignment'),status='COMPLETE',before=before,after=after,
        export_hashes=read(work/'export.json')['model_file_hashes'],training=read(work/'training-receipt.json'),
        drop=before['token_mean']-after['token_mean'],final_nll=after['token_mean'],seconds=time.monotonic()-start,
        gpu_hours=0 if a.standin or measurement=='reduce' else (time.monotonic()-scoring_start)/3600,output_deleted=not bool(family)))
    # Assigned shared probes stay until all assigned scoring snapshots are durable.
    if not family:shutil.rmtree(work)
    print('Probe scored:',a.incoming,a.domain)


def receipt_path(out,incoming,domain,*,family=None,seed=None):
    from kit.m2_assignment import cache_identity
    proof=cache_identity(out) if family is None else dict(family=family,seed=seed)
    suffix='-'+proof['family']+'-s'+str(proof['seed']) if proof else ''
    return out/'probes'/incoming/domain/('receipt'+suffix+'.json')


def configure(p):
    p.add_argument('--measurement',choices=('before','after','reduce'))
    p.add_argument('--incoming',required=True);p.add_argument('--domain',choices=DOMAINS,required=True)
    p.add_argument('--phase',choices=('all','train','score'),default='all')
    p.add_argument('--gpus',default='0,1,2,3,4,5,6,7');p.add_argument('--score-gpu',default='0')
    p.add_argument('--trainer-python',default=sys.executable);p.add_argument('--inference-python',default=sys.executable)
if __name__=='__main__':raise SystemExit(entry(job,configure))
