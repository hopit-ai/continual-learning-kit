#!/usr/bin/env python3
"""Re-score inherited base/comparators on the assigned X family GPU; no training."""
from pathlib import Path
import sys
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kit.x_common import *
from types import SimpleNamespace as NS, FunctionType
import csv
import shutil


def job(a):
    doc=context(a.out,a.standin)
    from kit.m2_assignment import active
    proof=active(a.out,a.gpu,a.standin)
    require(proof is not None and proof['seed']==a.seed and proof['family'].startswith('x-'),'X reference family/seed required')
    root=a.out/'references'/f"{proof['family']}-s{a.seed}"
    bind(root/'binding.json',doc)
    source=a.out/'scoring-assignment.json'
    if not (root/source.name).exists():shutil.copyfile(source,root/source.name)
    if not (root/'hellaswag').exists():shutil.copytree(a.out/'hellaswag',root/'hellaswag')
    from kit.m2_format import validation_name,require_validation
    validation=validation_name(a.out,proof)
    generated_only = getattr(a, 'generated_only', False)
    if not a.standin and not generated_only:require_validation(a.out,proof)
    if validation.exists():
        target=validation_name(root,proof);target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():require(sha(target)==sha(validation),'reference technical validation changed')
        else:shutil.copyfile(validation,target)
    refs={'base':doc['base']}
    core=Path(doc['message2']);core_doc=m.binding(core,a.standin)
    comparator='standalone-legal-'+doc['configuration']
    if f'{comparator}:{a.seed}' in core_doc.get('comparators',{}):refs[comparator]=core_doc['comparators'][f'{comparator}:{a.seed}']
    from kit import m2_score
    values=[];files={}
    for arm,ref in refs.items():
        namespace=dict(m2_score.__dict__,binding=lambda out,standin:doc,
            record=lambda out,binding,arm,seed,step,ref=ref:ref)
        score=FunctionType(m2_score.job.__code__,namespace,'x_reference_score')
        for panel in (m.PANEL_SETS if generated_only else (*m.PANEL_SETS,'hellaswag')):
            args=NS(out=root,arm=arm,seed=a.seed,step=0 if arm=='base' else 40,panel_set=panel,
                gpu=a.gpu,standin=a.standin,panel_root=a.panel_root)
            score(args)
    for path in sorted((root/'scores').glob('*/scores.csv')):
        with path.open() as stream:values.extend(csv.DictReader(stream))
        files[str(path.relative_to(root))]=sha(path)
    bind(root/'rows.json',dict(assignment=proof,rows=values,files=files))


def configure(p):
    p.add_argument('--generated-only', action='store_true', help='Generated denominators only; likelihood guards are scored separately by ref_score_job.py')
    p.add_argument('--seed',type=int,choices=SEEDS,required=True);p.add_argument('--gpu',default='0');p.add_argument('--panel-root',type=Path)
if __name__=='__main__':raise SystemExit(entry(job,configure))


def gathered(files):
    """Hash-check the literal score rows used for every inherited denominator."""
    from kit.m2_assignment import verify, require as verified
    import io
    result={}
    for name,raw in files.items():
        if not name.startswith('references/') or not name.endswith('/rows.json'):continue
        root=name.rsplit('/',1)[0];doc=json.loads(raw)
        verify(doc,files['scoring-assignment.json'])
        actual=[]
        for path,expected in sorted(doc.get('files',{}).items()):
            value=files.get(root+'/'+path)
            verified(value is not None and hashlib.sha256(value).hexdigest()==expected,'inherited score rows changed')
            actual.extend(csv.DictReader(io.StringIO(value.decode())))
        verified(actual==doc['rows'],'inherited row alias differs from scored CSVs')
        result.setdefault(doc['assignment']['family'],[]).extend(actual)
    return result
