"""Independent sequential HF reference for the frozen HellaSwag likelihood rule.

No shared scoring/preprocessing code. The wrapper shares only a technical-error type. Character counts exclude the single
continuation separator. Exact score ties select the lowest original choice index.
"""
import math
import re


class ReferenceInvalid(ValueError):
    """A reference item cannot supply the declared conditional likelihood."""


def document(row):
    def clean(s):
        s=s.strip().replace(' [title] ','. ')
        s=re.sub(r'\[.*?\]','',s)
        return s.replace('  ',' ')
    context=clean(row['activity_label']+': '+row['ctx_a']+' '+row['ctx_b'].capitalize())
    return context,[clean(value) for value in row['endings']]


def choose(scores):
    assert len(scores)==4 and all(math.isfinite(s) for s in scores)
    return max(range(4),key=lambda index:(scores[index],-index))


def _score(model_path,items,tokenizer_path):
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from kit.m2_gpu import assert_current
    assert_current()
    tok=AutoTokenizer.from_pretrained(tokenizer_path,local_files_only=True)
    net=AutoModelForCausalLM.from_pretrained(model_path,local_files_only=True,torch_dtype=torch.bfloat16).to('cuda').eval()
    records=[]
    try:
        with torch.inference_mode():
            for row in items:
                context,endings=document(row);context_ids=tok.encode(context,add_special_tokens=False)
                if len(endings)!=4:raise ReferenceInvalid('four reference endings required')
                values=[]
                for ending in endings:
                    ids=tok.encode(context+' '+ending,add_special_tokens=False)
                    if not (context_ids and ids[:len(context_ids)]==context_ids and len(ids)>len(context_ids) and ending):raise ReferenceInvalid('reference conditional token boundary changed')
                    input_ids=torch.tensor([ids],device='cuda')
                    log_probs=net(input_ids=input_ids,use_cache=False).logits[0].float().log_softmax(-1)
                    # Explicit token-by-token reduction, independent of the vLLM path.
                    value=sum(float(log_probs[j-1,ids[j]].item()) for j in range(len(context_ids),len(ids)))
                    values.append(value/len(ending))
                records.append(dict(id=row['id'],scores=values,selected=choose(values) if all(math.isfinite(v) for v in values) else None))
    finally:
        del net
        torch.cuda.empty_cache()
    return records


def score(model_path,items,tokenizer_path):
    # HF initializes CUDA. Isolate it so later vLLM calls never inherit a CUDA
    # context from the independent reference in the controller process.
    import json,subprocess,sys,tempfile,os
    from pathlib import Path
    with tempfile.TemporaryDirectory(prefix='hs-reference-') as scratch:
        root=Path(scratch);source=root/'items.json';target=root/'reference.json'
        source.write_text(json.dumps(items))
        try:subprocess.run([sys.executable,'-m','kit.hellaswag_reference','--model',str(model_path),
            '--tokenizer',str(tokenizer_path),'--items',str(source),'--output',str(target)],check=True,
            env=dict(os.environ,PYTHONPATH=str(Path(__file__).resolve().parents[1])+os.pathsep+os.environ.get('PYTHONPATH','')))
        except subprocess.CalledProcessError as error:
            if error.returncode==2:
                from kit.v4_finqa_policy import CorrectnessStop
                raise CorrectnessStop('independent reference GPU/input identity refused') from error
            raise
        result=json.loads(target.read_text())
        if isinstance(result,dict) and 'technical_invalid' in result:
            from kit.m2_format import TechnicalInvalid
            raise TechnicalInvalid(result['technical_invalid'])
        return result


if __name__=='__main__':
    import argparse,json
    from pathlib import Path
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('model','tokenizer','items','output'):parser.add_argument('--'+name,required=True)
    args=parser.parse_args()
    from kit.v4_finqa_policy import CorrectnessStop
    try:Path(args.output).write_text(json.dumps(_score(args.model,json.loads(Path(args.items).read_text()),args.tokenizer)))
    except ReferenceInvalid as error:Path(args.output).write_text(json.dumps(dict(technical_invalid=str(error))))
    except CorrectnessStop as error:print(str(error));raise SystemExit(2)
