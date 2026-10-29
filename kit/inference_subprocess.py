"""One engine per child process; the controller never owns CUDA allocations."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def compute(kind,model,*,rows=None,tokenizer=None):
    from kit.m2_format import TechnicalInvalid
    from kit.v4_finqa_policy import CorrectnessStop
    with tempfile.TemporaryDirectory(prefix='m2-inference-') as scratch:
        request=Path(scratch)/'request.json';result=Path(scratch)/'result.json'
        request.write_text(json.dumps(dict(kind=kind,model=str(model),rows=rows,tokenizer=str(tokenizer) if tokenizer else None)))
        done=subprocess.run([sys.executable,'-m','kit.inference_subprocess','--request',str(request),'--result',str(result)],
            env=dict(os.environ,PYTHONPATH=str(Path(__file__).resolve().parents[1])+os.pathsep+os.environ.get('PYTHONPATH','')),check=False)
        if done.returncode==2:raise CorrectnessStop('inference child GPU/input identity refused')
        if done.returncode:raise subprocess.CalledProcessError(done.returncode,done.args)
        value=json.loads(result.read_text())
        if 'technical_invalid' in value:raise TechnicalInvalid(value['technical_invalid'])
        return value['value']


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--request',type=Path,required=True);p.add_argument('--result',type=Path,required=True);a=p.parse_args()
    from kit.m2_format import TechnicalInvalid,json_finite
    from kit.v4_finqa_policy import CorrectnessStop
    request=json.loads(a.request.read_text())
    try:
        if request['kind']=='likelihood':
            from kit.m2_format import _likelihoods
            value=_likelihoods(request['model'],request['rows'],request['tokenizer'])
        elif request['kind']=='knowledge':
            from kit.hellaswag_prompt_check import _knowledge_audit
            value=_knowledge_audit(request['model'])
        else:raise ValueError('unknown isolated engine task')
        result=dict(value=json_finite(value))
    except TechnicalInvalid as error:result=dict(technical_invalid=str(error))
    except CorrectnessStop as error:print(str(error));return 2
    a.result.write_text(json.dumps(result,allow_nan=False))
    return 0

if __name__=='__main__':raise SystemExit(main())
