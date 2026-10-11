"""Explicit CPU checkpoint adapter for real v4 merge recovery; never model evidence."""
import argparse
from pathlib import Path
import sys
if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from kit import v4_teacher as t
from kit.v4_arms import checkpoint_path


def export(path):
    path=Path(path);path.mkdir(parents=True,exist_ok=True)
    for name in ('config.json','tokenizer.json','tokenizer_config.json'):
        t.atomic_json(path/name,{'technical_synthetic':True,'scientific_phase_allowed':False})
    (path/'model.safetensors').write_bytes(b'CPU STAND-IN, NOT MODEL WEIGHTS')


def prepare_run(work,phase,ident):
    from kit.v4_campaign_ops import job_for
    job=job_for(ident,phase);out=Path(work)/'runs'/ident
    (out/'env').mkdir(parents=True,exist_ok=True)
    steps=(20,40) if phase=='main' else (2,)
    for step in steps:
        root=checkpoint_path(out,job['arm'],step,actor=True);root.mkdir(parents=True,exist_ok=True)
        (root/'stand-in-checkpoint.txt').write_text('CPU STAND-IN; external merger adapter required')
    t.atomic_json(out/'env/optimizer-updates.json',{'completed_optimizer_updates':steps[-1]})
    t.atomic_json(out/'run-summary.json',{'arm':job['arm'],'merged':0,'technical_synthetic':True,'scientific_phase_allowed':False})


def recover(work,phase,ident):
    """Execute production recovery, timing, completeness and identity bodies.

    Only verl's model merger is replaced, at its subprocess boundary. Both
    exports are absent so every arm executes the real recovery path.
    """
    from unittest.mock import patch
    from kit import v4_campaign_ops as ops
    real_run=ops.run
    def merger(argv,env=None):
        argv=list(argv);index=argv.index('-m')
        if argv[index+1]!='verl.model_merger':raise ValueError('unexpected external merger')
        argv[index:index+3]=[str(Path(__file__).resolve())]
        return real_run(argv,env)
    from kit import v4_retention
    with patch.object(ops,'run',merger), patch.object(v4_retention,'reload_weights',return_value={'tensor_count':1,'dtype':'BF16'}):
        ops.merge(work,phase,ident)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend',choices=['fsdp'],required=True)
    parser.add_argument('--local_dir',type=Path,required=True)
    parser.add_argument('--target_dir',type=Path,required=True)
    args=parser.parse_args()
    if not (args.local_dir/'stand-in-checkpoint.txt').is_file():raise ValueError('stand-in checkpoint missing')
    export(args.target_dir)
    return 0


if __name__=='__main__':raise SystemExit(main())


def prepare_retained_run(work,phase,ident):
    """Run the actual main save/retire guards on complete CPU shard stand-ins."""
    from unittest.mock import patch
    from kit import v4_retention as policy, v4_campaign_ops as ops
    job=ops.job_for(ident,phase);out=Path(work)/'runs'/ident
    prepare_run(work,phase,ident)
    real_run=ops.run
    def merger(argv,env=None):
        argv=list(argv);index=argv.index('-m')
        if argv[index+1]!='verl.model_merger':raise ValueError('unexpected external merger')
        argv[index:index+3]=[str(Path(__file__).resolve())]
        return real_run(argv,env)
    # Begin with no future state. Save then retire in the actual serial order.
    for step in ((20,40) if phase=='main' else (2,)):
        root=checkpoint_path(out,job['arm'],step)
        policy.before_save(out,job['arm'],step)
        for name in policy.state_names(job['arm']):
            file=root/name;file.parent.mkdir(parents=True,exist_ok=True);file.write_bytes(('CPU STAND-IN '+name+str(step)).encode())
        with patch.object(ops,'run',merger),patch.object(policy,'reload_weights',return_value={'tensor_count':1,'dtype':'BF16'}):
            policy.after_save(out,job['arm'],step,40 if phase=='main' else 2)
    summary=__import__('json').loads((out/'run-summary.json').read_text())
    t.atomic_json(out/'run-summary.json',{**summary,'completed_optimizer_updates':40 if phase=='main' else 2,'dose_complete':True})


def evaluation_and_sweep(work,phase,row):
    """Explicit synthetic evidence; exercise retirement through real latest PASS records.

    Fake outputs cannot pass either scientific reader. Only model generation,
    tensor reload and source data are adapters; the production deletion bodies
    and real runner records are used unchanged.
    """
    from kit import v4_retention as policy,v4_campaign_ops as ops
    work=Path(work)
    if row.startswith('score-'):
        # general-v1 contains a hyphen; parse from registered slot IDs instead.
        from kit.v4_campaign import slots,ALL_ARMS
        candidates=[j['id'] for j in slots()]+['q-'+a+'-finqa' for a in ALL_ARMS]
        ident=next((i for i in candidates if row.startswith('score-'+i+'-')),None)
        if ident:
            taskcap=row[len('score-'+ident+'-'):];out=work/('v4/report-'+phase)/'eval'/(ident+'-'+taskcap)
            step=40 if phase=='main' else 2
        else:return
    elif row.startswith('probe-'):
        ident,task,step=row.removeprefix('probe-').rsplit('-',2);step=int(step)
        out=work/('v4/report-'+phase)/'probes'/f'{ident}-{task}-{step}'
    else:
        policy.sweep(work,starting_slot=row.removeprefix('train-') if row.startswith('train-') else None)
        return
    run=work/'runs'/ident;out.mkdir(parents=True,exist_ok=True)
    if step==0:
        initial=work/'v4/report-inputs/model-identities.json';initial.parent.mkdir(parents=True,exist_ok=True)
        identity={'model_file_hashes':{'model.safetensors':t.sha('CPU INITIAL STAND-IN')}}
        if not initial.exists():t.atomic_json(initial,{'initial':identity})
    else:identity=t.local_identity(run/f'hf-step{step}','Qwen/Qwen3-8B')
    t.atomic_json(out/'checkpoint-identity.json',identity)
    probe=row.startswith('probe-')
    task='finqa' if probe else taskcap.rsplit('-',1)[0]
    panel=work/'v4/report-inputs'/(task+('-probe.jsonl' if probe else '-panel.jsonl'))
    panel.parent.mkdir(parents=True,exist_ok=True)
    if not panel.exists():t.write_jsonl(panel,[{'id':str(i),'CPU_standin':True} for i in range(20 if probe else 300 if task=='general-v1' else 1147)])
    count=len(t.read_jsonl(panel))*(8 if probe else 1)
    if phase=='qualification':count=20
    t.write_jsonl(out/('raw_outputs.jsonl' if probe else 'responses.jsonl'),[{'id':str(i),'CPU_standin':True,'scientific_phase_allowed':False} for i in range(count)])
    for name in (('timing.json','probe-identity.json') if probe else ('forgetting.json' if task=='general-v1' else 'bed-score.json',)):
        t.atomic_json(out/name,{'CPU_standin':True,'scientific_phase_allowed':False})
    if not probe:t.write_jsonl(out/'tokens.jsonl',[{'id':str(i),'CPU_standin':True} for i in range(count)])
