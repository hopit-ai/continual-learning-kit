#!/usr/bin/env python3
"""Generate exact prospective transitions from current pins in a clean commit.

Only pin values still referenced by current target checkers are included. The
runtime-update map is an exact historical transition allowlist, not a set of
additional current input pins. Its original/v2 values are included only when a
current checker actually pins them. Rows record every live pin source.
The authority itself is excluded from pin discovery. The manager generates it LAST.

Release usage: --base 68100bf6 --target HEAD --out <authority-path>.
The release base is the pre-v6 kit commit (the scorer-replay base), never the
already-sent v6a commit: pins amended before v6a still need exact transitions.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from kit.v6_infra import AUTHORITY,SCHEMA,validate

# Every unlisted implementation is scheduling/handoff/identity code. These files
# also contain scientific scoring, generation, training or decision reductions.
# GPU-only portions require the sign-off listed in resources/v6/scorer-files.md.
SCORER_FILES=set('''kit/c_targets.py kit/eval_bed.py kit/hints.py
kit/likelihood_validation.py kit/m2_collect.py kit/m2_format.py kit/m2_read.py
kit/m2_rescore.py kit/m2_score.py kit/m2_targets.py kit/m2_train.py
kit/m3_pools.py kit/m3_read.py kit/m3_score.py kit/m3_train.py kit/next_pools.py
kit/next_read.py kit/p3b_read.py kit/pd_jobs.py kit/pd_probe.py kit/ref_guard.py
kit/ref_score_job.py kit/repair/make_targets.py kit/s_drift.py kit/s_generate.py
kit/s_probe.py kit/s_probe_train.py kit/s_score.py kit/score_forgetting.py
kit/sp_manifest.py kit/sp_prepare.py kit/sp_read.py kit/sp_score.py kit/v4_teacher.py
kit/x_collect.py kit/x_p.py kit/x_p2_feasibility.py kit/x_p2_measure.py
kit/x_read.py kit/x_references.py kit/x_run.py kit/x_signal.py kit/x_support.py
kit/x_tier2_run.py kit/x_tier2_train.py kit/x_train.py'''.split())

# Current checker inputs, not embedded code snapshots in measurement receipts,
# construction history, old send documents or replaced validation generations.
PIN_SECTIONS={
    'kit/resources/s/registration.json':'code',
    'kit/resources/sp/FROZEN.json':'code',
    'kit/resources/3c-screen/frozen-annex.json':'manifest',
    'kit/resources/3c-screen/public-inputs.json':'resources',
    'kit/resources/c/freeze.json':'files',
    'kit/resources/m2/likelihood-near-tie.json':'implementations',
}


def git(root,*args):return subprocess.check_output(['git',*args],cwd=root)


def blobs(root,ref,names):
    if not names:return {}
    raw=subprocess.run(['git','cat-file','--batch'],cwd=root,
        input=''.join(ref+':'+name+'\n' for name in names).encode(),stdout=subprocess.PIPE,check=True).stdout
    result={};offset=0
    for name in names:
        end=raw.index(b'\n',offset);header=raw[offset:end].split()
        if len(header)!=3 or header[1]!=b'blob':raise ValueError('missing committed file: '+name)
        size=int(header[2]);result[name]=raw[end+1:end+1+size];offset=end+size+2
    return result


def current_pins(root=ROOT,target='HEAD'):
    """Return live (path, pin) -> sources, read exclusively from target blobs."""
    names=git(root,'ls-tree','-r','--name-only',target,'kit','scripts','docs').decode().splitlines()
    tracked=set(names);pins={}
    def add(name,value,parent,source):
        if not isinstance(name,str) or not name.endswith(('.py','.sh')) or not isinstance(value,str) or not re.fullmatch('[0-9a-f]{64}',value):return
        candidates=(name,'kit/'+name,str(parent/name))
        if name=='scripts/second_task_screen_modal.py' and name not in tracked:
            candidates+=('kit/resources/3c-screen/second_task_screen_modal.py',)
        for candidate in candidates:
            if candidate in tracked:
                pins.setdefault((candidate,value),set()).add(source);return
        raise ValueError('current pin names a missing implementation: '+name)
    def walk(value,parent,source):
        if isinstance(value,dict):
            if isinstance(value.get('path'),str):add(value['path'],value.get('sha256'),parent,source+'/sha256')
            for name,item in value.items():
                add(name,item,parent,source+'/'+name)
                if name.endswith(('.py','.sh')) and isinstance(item,dict):
                    for key in ('original_sha256','v2_sha256','sha256'):add(name,item.get(key),parent,source+'/'+name+'/'+key)
                walk(item,parent,source+'/'+name)
        elif isinstance(value,list):
            for i,item in enumerate(value):walk(item,parent,source+'/'+str(i))
    pin_names=[name for name in names if name.startswith('kit/resources/') and name!=AUTHORITY and
        (name in PIN_SECTIONS or name=='kit/resources/m2/format-validity.json' or
         Path(name).name in ('SHA256SUMS','pins.json') or
         # Measurement receipts whose code_sha256 maps are checked live (x_p, x_smoke_settings).
         name.startswith('kit/resources/m2/receipts/') and name.endswith('.json'))]
    for name,raw in blobs(root,target,pin_names).items():
        if name.endswith('.json'):
            try:
                doc=json.loads(raw)
                if name=='kit/resources/m2/format-validity.json':
                    update=json.loads(git(root,'show',target+':kit/resources/m2/likelihood-near-tie.json'))
                    doc={k:v for k,v in doc['implementations'].items() if k not in update['replaces']}
                elif name in PIN_SECTIONS:doc=doc[PIN_SECTIONS[name]]
                walk(doc,Path(name).parent,name)
            except UnicodeDecodeError:continue
        else:
            for line in raw.decode().splitlines():
                fields=line.split()
                if len(fields)==2:add(fields[1].lstrip('*'),fields[0],Path(name).parent,name)
    return {key:sorted(sources) for key,sources in sorted(pins.items())}


def uncovered_pins(doc,root=ROOT,target='HEAD',working_tree=False):
    """Check all live pins; the historical exact runtime map remains valid."""
    pins=current_pins(root,target);names=sorted({name for name,_ in pins})
    sources={name:(root/name).read_bytes() for name in names} if working_tree else blobs(root,target,names)
    updates=json.loads(git(root,'show',target+':kit/resources/m2/runtime-updates.json'))['files'] if 'kit/m2_runtime.py' in names else {}
    pairs={(r['path'],r['pinned_sha256'],r['new_sha256']) for r in doc['files']}
    missing=[]
    for name,pinned in pins:
        actual=hashlib.sha256(sources[name]).hexdigest();legacy=updates.get(name,{})
        if actual==pinned or (legacy.get('original_sha256')==pinned and legacy.get('v2_sha256')==actual) or (name,pinned,actual) in pairs:continue
        missing.append((name,pinned,actual))
    return missing


def historical_pins(root,base):
    """v6a-breadth scan: every implementation hash recorded anywhere in the base's JSON or
    SHA256SUMS resources (receipts, legacy maps, runtime-update maps, send records).

    Kit checkers outside the registration sections (p3a_common inputs, next legacy maps,
    measurement receipts) compare code against these records, so changed files need exact
    pairs for them too. Retired helpers and non-JSON resources are skipped: no live checker
    can read a missing file. Pairs remain exact (pinned -> committed target bytes).
    """
    names=git(root,'ls-tree','-r','--name-only',base,'kit','scripts','docs').decode().splitlines()
    tracked=set(names);pins={}
    def add(name,value,parent,source):
        if not isinstance(name,str) or not name.endswith(('.py','.sh')) or not isinstance(value,str) or not re.fullmatch('[0-9a-f]{64}',value):return
        for candidate in (name,'kit/'+name,str(parent/name)):
            # Historical rows cover kit/ implementations only: the live non-registration
            # checkers read kit files, and scripts/ pins come from the current-pin pass.
            if candidate in tracked and candidate.startswith('kit/'):pins.setdefault((candidate,value),set()).add(source);return
    def walk(value,parent,source):
        if isinstance(value,dict):
            if isinstance(value.get('path'),str):
                for key in ('sha256','pinned_sha256','new_sha256'):add(value['path'],value.get(key),parent,source)
            for name,item in value.items():
                add(name,item,parent,source)
                if isinstance(name,str) and name.endswith(('.py','.sh')) and isinstance(item,dict):
                    for key in ('original_sha256','v2_sha256','sha256'):add(name,item.get(key),parent,source)
                walk(item,parent,source)
        elif isinstance(value,list):
            for item in value:walk(item,parent,source)
    pin_names=[n for n in names if n!=AUTHORITY and (n.endswith('.json') or Path(n).name=='SHA256SUMS')]
    for name,raw in blobs(root,base,pin_names).items():
        if name.endswith('.json'):
            try:walk(json.loads(raw),Path(name).parent,'historical:'+name)
            except (UnicodeDecodeError,ValueError):continue
        else:
            for line in raw.decode(errors='replace').splitlines():
                fields=line.split()
                if len(fields)==2:add(fields[1].lstrip('*'),fields[0],Path(name).parent,'historical:'+name)
    return pins


def generate(base,root=ROOT,target='HEAD'):
    # --untracked-files=no ignores local diagnostics, never uncommitted tracked
    # bytes. All generated hashes come from commits, not the working tree.
    if git(root,'status','--porcelain','--untracked-files=no').strip() or git(root,'status','--porcelain','--untracked-files=all','--','kit','scripts').strip():raise ValueError('authority generation requires a clean committed tree')
    base=git(root,'rev-parse','--verify',base+'^{commit}').decode().strip()
    target=git(root,'rev-parse','--verify',target+'^{commit}').decode().strip()
    pins={key:set(value) for key,value in current_pins(root,target).items()}
    for key,sources in historical_pins(root,base).items():pins.setdefault(key,set()).update(sources)
    pins={key:sorted(value) for key,value in sorted(pins.items())}
    changed=set(git(root,'diff','--name-only',base,target,'--','kit','scripts').decode().splitlines())
    names=sorted({name for name,_ in pins if name in changed});contents=blobs(root,target,names)
    files=[]
    for (name,pinned),sources in pins.items():
        if name not in changed:continue
        actual=hashlib.sha256(contents[name]).hexdigest()
        if actual==pinned:continue
        kind='scorer/reducer' if name in SCORER_FILES else 'scheduling'
        files.append(dict(path=name,pinned_sha256=pinned,new_sha256=actual,classification=kind,pin_sources=sources,
            reason=('engine backend/memory configuration; scoring definitions unchanged' if name=='kit/inference_lifecycle.py' else
                    'Scheduling/handoff only; scientific definitions unchanged' if kind=='scheduling' else
                    'Scorer/reducer orchestration; CPU golden replay and GPU-only sign-off scope: kit/resources/v6/scorer-files.md')))
    doc=validate(dict(schema=SCHEMA,statement='scientific definitions unchanged',base_commit=base,target_commit=target,files=files))
    if uncovered_pins(doc,root,target):raise ValueError('base omits current pin transitions; use the pre-v6 kit release base 68100bf6')
    return doc


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',required=True,help='pre-v6 kit release base 68100bf6; do not use v6a');p.add_argument('--target',default='HEAD');p.add_argument('--out',type=Path,required=True)
    a=p.parse_args(argv)
    try:doc=generate(a.base,target=a.target)
    except (ValueError,subprocess.CalledProcessError) as exc:p.error(str(exc))
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(doc,sort_keys=True,indent=2)+'\n')
    print(len(doc['files']),'exact prospective pin transitions generated for',doc['target_commit'])
if __name__=='__main__':main()
