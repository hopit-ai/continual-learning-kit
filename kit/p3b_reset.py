#!/usr/bin/env python3
"""One conditional 1% MLP reset to base; no optimiser state is copied."""
from pathlib import Path
import sys
if __package__ in (None, ''): sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kit.p3b_common import *
import re


def indices(layer, n):
    return sorted(sorted(range(n),key=lambda i:(t.sha(f'package-b-reset-v1|{layer}|{i}'),i))[:n//100])


def reset_tensors(state, base):
    """Mapping tensors may be a single layer/shard; every change recorded exactly."""
    records={}
    for name,value in state.items():
        match=re.fullmatch(r'model.layers.(\d+).mlp.(gate_proj|up_proj|down_proj).weight',name)
        if not match: continue
        layer, projection=match.groups(); axis=1 if projection=='down_proj' else 0
        require(name in base and value.shape==base[name].shape,'base reset geometry differs')
        chosen=indices(layer,value.shape[axis]); target=base[name].to(value.dtype)
        if axis: value[:,chosen]=target[:,chosen]
        else: value[chosen,:]=target[chosen,:]
        records[name]=dict(axis=axis,indices=chosen)
    return records


def job(a):
    require(a.trigger.is_file(),'missing written reset trigger')
    decision=json.loads(a.trigger.read_text())
    require(a.standin or decision.get('technical_synthetic') is False,'synthetic/unknown trigger cannot reset a real model')
    if decision.get('reset_recipe') is None: print('SKIP: no reset trigger'); return
    require(decision['reset_recipe']==a.recipe,'only registered priority recipe may reset')
    root=a.out / 'resets' / f'{a.recipe}-s{a.seed}'
    require(a.standin or a.incoming_receipt is not None,'original FinQA export receipt required')
    observed=dict(technical_synthetic=True) if a.standin else dict(incoming=identity(a.model,receipt=a.incoming_receipt),base=identity(a.base,'student'))
    if not a.standin:
        origin=Path(json.loads(a.incoming_receipt.read_text()).get('model_dir',''))
        slot=f'{a.recipe}-finqa-alone-s{a.seed}' if a.recipe in ('D','F') else f'finqa-{a.recipe}-s{a.seed}'
        require(slot in origin.parts and origin.name=='hf-step40','reset needs matched original FinQA final')
    bind(a.out / 'identities' / (root.name+'-reset.json'),dict(trigger=s.file_hash(a.trigger),**observed))
    if (root / 'manifest.json').exists():
        frozen_files(root)
        if not a.standin: identity(root / 'hf-reset',receipt=root / 'export.json')
        print('SKIP: reset export'); return
    root.mkdir(parents=True,exist_ok=True)
    if a.standin:
        import torch
        state={'model.layers.0.mlp.gate_proj.weight':torch.ones(200,2)}
        records=reset_tensors(state,{k:v*0 for k,v in state.items()})
    else:
        require(a.incoming_receipt is not None,'original FinQA export receipt required')
        from safetensors import safe_open
        from safetensors.torch import load_file,save_file
        import shutil
        base_index=json.loads((Path(a.base)/'model.safetensors.index.json').read_text())['weight_map']
        records={}; export=root / 'hf-reset'; export.mkdir(exist_ok=True)
        for source in Path(a.model).glob('*'):
            if source.is_file() and source.suffix in ('.json','.jinja'): shutil.copyfile(source,export / source.name)
        for shard in sorted(Path(a.model).glob('*.safetensors')):
            state=load_file(str(shard)); anchors={}
            for name in state:
                if re.fullmatch(r'model.layers.\d+.mlp.(gate_proj|up_proj|down_proj).weight',name):
                    with safe_open(str(Path(a.base)/base_index[name]),framework='pt',device='cpu') as f: anchors[name]=f.get_tensor(name)
            records.update(reset_tensors(state,anchors)); save_file(state,str(export/shard.name),metadata={'format':'pt'})
        require(len(records)==3*observed['base']['model_config']['num_hidden_layers'],'missing MLP projections')
        atomic(root / 'export.json',identity(export))
    atomic(root / 'changed.json',records)
    atomic(root / 'manifest.json',files_manifest(root,['changed.json']+([] if a.standin else ['export.json']),technical_synthetic=a.standin))


def configure(p):
    p.add_argument('--model');p.add_argument('--base');p.add_argument('--incoming-receipt',type=Path)
    p.add_argument('--trigger',type=Path,required=True);p.add_argument('--recipe',choices=('D+replay','D','F'),required=True)
    p.add_argument('--seed',type=int,choices=range(101,106),required=True)
if __name__ == '__main__': raise SystemExit(entry(job,configure,'p3b-reset-{recipe}-{seed}'))
