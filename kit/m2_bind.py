#!/usr/bin/env python3
"""Bind return-1 nomination and exact connected parents at send; no inference."""
from pathlib import Path
import sys
if __package__ in (None,''): sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kit.m2_common import *
from kit.p3b_read import load, analyse as b_analyse
from kit.c_bind import choose, check_start
from kit import c_common, c_read, c_probe_read
from kit.math_hard_teacher_read import read as teacher_read
from types import SimpleNamespace as NS
import tempfile


EXPECT_FIELDS=('W','provisional','nomination','hard_math','b_scores_sha256','c_scores_sha256','technical_synthetic')


def score_digest(rows):
    # C's compact collector pads general-panel rows with empty legal columns.
    # Canonicalize both that schema and order across archive/full-tree readers.
    from kit.c_score import COLUMNS
    projected=[{key:str(row.get(key,'')) for key in COLUMNS} for row in rows]
    projected.sort(key=lambda r:(r['arm'],int(r['seed']),int(r['step']),r['panel'],r['question_id']))
    return s.digest(projected)


def returned_teacher(source):
    if source.is_dir(): return teacher_read(source)
    # The message-1 teacher archive includes its entire journal. Write regular
    # members to scratch, without trusting archive paths or extracting links.
    _,files=load(source)
    with tempfile.TemporaryDirectory(prefix='m2-teacher-read-') as scratch:
        root=Path(scratch)
        for name,raw in files.items():
            path=Path(name)
            require(not path.is_absolute() and '..' not in path.parts,'unsafe teacher archive path')
            target=root/path; target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
        candidates=list(root.rglob('inputs.json'))
        require(len(candidates)==1,'exactly one returned teacher journal required')
        return teacher_read(candidates[0].parent)


def read_return1(a):
    """Same frozen readers on manager archives and the node's full trees."""
    if getattr(a,'decisions',None) is not None:
        return reviewed_return1(a)
    require(a.standin,'real return-1 binding requires --decisions and original archives')
    rows,_=load(a.b_source)
    supplied=json.loads(a.b_reading.read_text())
    require(supplied.get('technical_synthetic') is a.standin, 'synthetic B reading mismatch')
    computed=b_analyse(rows,20)
    decision=choose(computed)
    require(choose(supplied)==decision, 'B reading disagrees with unreset scores')
    crows,cfiles=load(a.c_source)
    require('binding.json' in cfiles,'returned C binding required')
    cdoc=json.loads(cfiles['binding.json'])
    require(cdoc['protocol']==c_common.protocol() and cdoc['technical_synthetic'] is a.standin,'C protocol/type changed')
    require(cdoc['start_policy']==c_common.START_POLICY,'C start policy changed')
    require(cdoc['W']==decision['W'], 'C and B W mismatch')
    require(decision['W']=='D', '21 October continuation specifies W=D; other W requires prospective clarification')
    core=c_read.analyse(crows,cdoc,1)
    if not core['complete']: raise NotReady('completed C return pending')
    # Recompute the frozen probe reader, including its completeness check. Never trust a supplied nomination.
    try:
        c_probe_read.job(NS(out=a.out/'return1',source=a.c_source,standin=a.standin,bootstrap=20))
        probe=json.loads((a.out/'return1/reading-probe.json').read_text())
        nomination=probe['v4_nomination']
    except NotReady as exc:
        # Incomplete nomination freezes pending; plain D is independent of replay admission.
        nomination=dict(broad=None,earlier=None,earlier_comparison=None,admission='Pending connected finals')
        atomic(a.out/'return1/reading-probe.json',dict(v4_nomination=nomination,reason=str(exc),
               technical_synthetic=a.standin,source_sha256=s.digest(crows)))
    require(nomination['admission']=='Pending connected finals', 'unexpected return-1 admission state')
    result=returned_teacher(a.teacher_source)
    require(result['technical_synthetic'] is a.standin, 'synthetic teacher evidence mismatch')
    expected=dict(W=decision['W'],provisional=decision['provisional'],nomination=nomination,
                  hard_math=result['status'],b_scores_sha256=score_digest(rows),c_scores_sha256=score_digest(crows),
                  technical_synthetic=a.standin)
    return expected,cdoc,result,decision


def tier_a_base(call):
    registered=json.loads((KIT/'resources/p3b/model-hashes.json').read_text())['student']['files']
    return (call['step']==0 and call['score_set'].startswith('scores/base-') and call['model_hashes']==registered)


def bind_rescore_ref(call,supplied):
    require(supplied.get('model') and (supplied.get('receipt') or tier_a_base(call)),
            'checkpoint model/receipt paths required')
    return dict(supplied,model_hashes=call['model_hashes'])


def reviewed_return1(a):
    from kit import m2_return1 as r
    ruling=r.decisions(a.decisions)
    from kit.arc_prompt_acceptance import validate_frozen, FROZEN
    arc_acceptance=validate_frozen(getattr(a,'arc_prompt_check',None) or FROZEN)
    from kit.hellaswag_prompt_check import validate_receipt
    format_acceptance=validate_receipt(getattr(a,'format_check',None))
    fingerprints=r.archives(a.return1_archives,ruling)
    cdoc,cfiles=r.binding(a.c_source,ruling)
    rows,_=load(a.b_source)
    computed=b_analyse(rows,20);decision=choose(computed)
    require(decision['W']=='D' and decision['provisional']==ruling['B_selection']['provisional'], 'B W rule differs from reviewed return')
    supplied=json.loads(a.b_reading.read_text())
    require(choose(supplied)==decision,'B reading disagrees with unreset B scores')
    # Old C efficacy/format evidence is not a nomination. It only supplies checked identities.
    nomination=dict(broad=None,earlier=None,earlier_comparison=None,admission='Pending connected finals',status='PENDING')
    mapping=json.loads(a.rescore_paths.read_text()) if getattr(a,'rescore_paths',None) else {}
    # Tier B is optional: only Tier A paths are mandatory at the combined send.
    needed={c['checkpoint'] for p in ruling['rescoring']['packages'] if p['tier']=='A' for c in p['calls']}
    require(needed<=set(mapping),'Tier A export paths required in --rescore-paths')
    refs={}
    for package in ruling['rescoring']['packages']:
        for call in package['calls']:
            key=call['checkpoint']
            if key in mapping:
                if key in refs:
                    # Optional Tier B can reuse the already checked Tier A base.
                    # Its different score-set name does not create an export receipt.
                    require(refs[key]['model_hashes']==call['model_hashes'],'shared checkpoint hashes differ')
                else:
                    refs[key]=bind_rescore_ref(call,mapping[key])
            if package['tier']=='A':
                require(call['identity_path'] in cfiles and t.sha(cfiles[call['identity_path']])==call['identity_sha256'],'original Tier A identity differs')
                require(call['companion_path'] in cfiles,'original Tier A companion missing')
    expected=dict(W='D',provisional=decision['provisional'],nomination=nomination,hard_math='QUALIFIES',
        b_scores_sha256=score_digest(rows),c_scores_sha256=fingerprints['package-c-return.tar.gz'],
        technical_synthetic=False,return1_archives=fingerprints,decisions_sha256=s.file_hash(a.decisions),
        rescore_refs=refs,arc_acceptance=arc_acceptance,format_acceptance=format_acceptance,nomination_policy='on-node-four-arm-reprobe')
    return expected,cdoc,ruling['teacher'],decision


def check_expect(path, observed):
    require(path.is_file(),'manager send-binding.json required')
    expected=json.loads(path.read_text())
    for key in (*EXPECT_FIELDS, *(['return1_archives','decisions_sha256','rescore_refs','arc_acceptance','format_acceptance','nomination_policy'] if not observed['technical_synthetic'] else [])):
        require(key in expected and expected[key]==observed[key], 'send binding mismatch: '+key)


def sibling_start(sibling,b_source,seed,standin):
    if standin: return dict(sibling,identity=checked_model(sibling,True))
    require(sibling.get('receipt'),'original B export receipt required')
    run=b_source/'runs'/f'finqa-D+replay-s{seed}'
    model=Path(sibling['model']).resolve(); receipt_path=Path(sibling['receipt'])
    require(model== (run/'hf-step40').resolve() and receipt_path.resolve()==(run/'export-40.json').resolve(),
            'sibling must use returned B prevention-trained FinQA final')
    receipt=json.loads(receipt_path.read_text())
    origin=Path(receipt.get('model_dir',''))
    require(f'finqa-D+replay-s{seed}' in origin.parts and origin.name=='hf-step40',
            'sibling must use B prevention-trained FinQA final')
    returned=json.loads((run/'export-40.json').read_text())
    observed=checked_model(sibling,False)
    require(observed['model_file_hashes']==returned['model_file_hashes']==receipt['model_file_hashes'],
            'sibling differs from returned B export')
    return dict(sibling,identity=observed,receipt_sha256=s.file_hash(receipt_path))


def job(a):
    require(not (getattr(a,'standin',False) and getattr(a,'decisions',None)), '--standin cannot be combined with --decisions')
    expected,cdoc,result,decision=read_return1(a)
    if not a.standin:
        from kit.m2_expect import EXPECTED, check_public
        from kit.m2_return1 import decisions
        # The binder reads live B exports from a tree; hash the original archive,
        # never a claimed hash from the local --expect document.
        archive=a.b_source if a.b_source.is_file() else getattr(a,'b_archive',None)
        require(archive is not None and archive.is_file(),'original B archive required for public expectation check')
        expected['b_archive_sha256']=s.file_hash(archive)
        check_public(EXPECTED,expected,cdoc,decisions(a.decisions))
    check_expect(a.expect,expected)
    nomination=expected['nomination']
    raw=json.loads(a.checkpoints.read_text()); starts={}; exports={}; comparators={}
    starts['base']=dict(cdoc['starts']['base'])
    checked_model(starts['base'],a.standin,base=True)
    for seed in SEEDS:
        parent=cdoc['starts'][f'D:{seed}']
        starts[f'finqa-plain:{seed}']=check_start(parent,'D',seed,a.standin)
        key=f'D+replay:{seed}'
        require(key in raw, 'missing B prevention-trained sibling final '+key)
        starts[f'finqa-sibling:{seed}']=sibling_start(raw[key],a.b_source,seed,a.standin)
        for step in STEPS[1:]:
            exported=c_common.checkpoint(cdoc,a.c_source,'sequence-D',seed,step)
            observed=checked_model(exported,a.standin)
            exports[f'{seed}:{step}']=dict(exported,identity=observed)
        standalone=c_common.checkpoint(cdoc,a.c_source,'standalone-D',seed,40)
        comparators[f'standalone-legal-plain:{seed}']=dict(standalone,identity=checked_model(standalone,a.standin))
        run=a.c_source/'runs'/f'sequence-D-s{seed}'
        summary=json.loads((run/'summary.json').read_text())
        require(summary['incoming']==checked_model(parent,a.standin), 'C plain-D parent is not the exact matched v4 final')
        starts[f'legal-plain:{seed}']=exports[f'{seed}:40']
    doc=dict(W=decision['W'],provisional=decision['provisional'],why=decision['why'],nomination=nomination,
             hard_math=result['status'],teacher_decision=result,starts=starts,c_exports=exports,
             c_binding_sha256=s.file_hash(a.c_source/'binding.json'),c_scores_sha256=expected['c_scores_sha256'],
             b_scores_sha256=expected['b_scores_sha256'],send_binding_sha256=s.file_hash(a.expect),
             c_source=str(a.c_source.resolve()),out=str(a.out.resolve()),
             comparators=comparators,protocol=protocol(),technical_synthetic=a.standin)
    if 'nomination_policy' in expected:
        doc.update({k:expected[k] for k in ('return1_archives','decisions_sha256','rescore_refs','arc_acceptance','format_acceptance','nomination_policy')})
        doc['reprobe_starts']={str(seed):dict(cdoc['starts'][f'D:{seed}']) for seed in SEEDS}
    bind(a.out/'binding.json',doc)
    save_status(a.out,doc)
    print('W =',doc['W'],'provisional =',doc['provisional'],'; nomination:',nomination,'; hard-MATH:',doc['hard_math'])


def return1_options(p):
    p.add_argument('--b-archive',type=Path,help='original B archive when --b-source is the live export tree')
    p.add_argument('--arc-prompt-check',type=Path)
    p.add_argument('--format-check',type=Path)
    p.add_argument('--decisions',type=Path)
    p.add_argument('--return1-archives',type=Path)
    p.add_argument('--rescore-paths',type=Path)


def configure(p):
    return1_options(p)
    for name in ('b-source','b-reading','c-source','checkpoints','teacher-source','expect'):
        p.add_argument('--'+name,type=Path,required=True)
if __name__=='__main__': raise SystemExit(entry(job,configure,'m2-bind'))
