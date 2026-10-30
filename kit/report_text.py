#!/usr/bin/env python3
"""Run native CPU readers and prepare privacy-filtered WhatsApp return parts.

Usage: python kit/report_text.py --inputs report-inputs.local.json --out TEXT_DIR
The local input map is never copied into text. See README-combined.md.
"""
from __future__ import annotations
from pathlib import Path
import sys
if __package__ in (None, ''): sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
from collections import defaultdict
import contextlib
import io
import json
import math
import subprocess
import tempfile
import traceback
from types import SimpleNamespace as NS
from kit.text_return import FAMILIES, SCHEMA, canonical, write_parts, job_inventory


def read(path): return json.loads(Path(path).read_text())


def native(family, spec, scratch, standin=False, bootstrap=10000):
    """Use archive-read entry points/validation, not cached reading JSON.

    Local S/X trees use their unchanged collectors first. Archives are temporary
    node-local intermediates; they are never the text-return payload.
    """
    out = scratch / family; out.mkdir(parents=True, exist_ok=True)
    source = Path(spec['source']) if spec.get('source') else None
    if family == 'M':
        from kit import m2_read
        m2_read.job(NS(source=source, out=out, standin=standin, bootstrap=bootstrap))
        result = read(out / 'reading.json')
        from kit.p3b_read import load
        from kit.m2_common import nomination_from_files, NotReady
        from kit.m2_admit import decision
        rows,files=load(source);bound=json.loads(files['binding.json'])
        try:nomination=nomination_from_files(bound,files)
        except NotReady:nomination=dict(broad=None,earlier=None)
        result['admission'],result['broad_confirmation']=decision(rows,nomination,bootstrap)
        return result
    if family == 'S':
        from kit.x_qualify import qualify
        if source.is_dir():
            from kit.s_collect import job
            archive = out / 'screen.tar.gz'
            job(NS(out=source, archive=archive, standin=standin)); source=archive
        result=qualify(source, standin)
        import tarfile
        with tarfile.open(source,'r:gz') as archive:
            costs=json.load(archive.extractfile('screen-s/costs.json')) if 'screen-s/costs.json' in archive.getnames() else []
        result['accounting']=dict(recorded_gpu_hours=sum(v['gpu_hours'] for v in costs) if costs else None,
                                  wall_seconds=sum(v.get('seconds') or 0 for v in costs) if costs else None)
        return result
    if family == 'X':
        from kit import x_read, x_collect
        if source.is_dir():
            x_read.job(NS(action='tables', out=source, standin=standin, bootstrap=bootstrap))
            archive=out/'x.tar.gz'
            x_collect.job(NS(out=source, archive=archive, standin=standin)); source=archive
        screen=Path(spec['screen']) if spec.get('screen') else None
        qualification=None
        if screen:
            if screen.is_dir():
                from kit.s_collect import job
                archive=out/'screen.tar.gz'; job(NS(out=screen,archive=archive,standin=standin)); screen=archive
            from kit.x_qualify import qualify
            qualification=out/'qualification.json'; qualification.write_text(canonical(qualify(screen,standin)))
        x_read.job(NS(action='read', source=source, out=out, standin=standin, bootstrap=bootstrap,
                      screen_qualification=qualification, screen_return=screen))
        result=read(out/'reading.json')
        # reading.json intentionally reports only eligible decisions. Carry the
        # native tables too, for every endpoint/guard and per-seed number.
        files=x_collect.load(source,standin)
        result['tables']=json.loads(files['comparison-tables.json'])
        return result
    if family == 'KCR':
        from kit.m3_read import read_job
        read_job(NS(out=source,standin=standin))
        return read(source/'m3-read.json')
    if family == 'PD':
        from kit.pd_read import read_evidence, collect_seed_evidence
        evidence=read(spec['evidence'])
        if 'seed_records' in evidence: evidence=collect_seed_evidence(evidence['seed_records'])
        result=read_evidence(evidence,read(spec['exports']),trusted=read(spec['trusted']))
        # Launcher timing includes failures, startup, all eight held ranks and
        # cleanup. Never add nested training/streamed-score timing a second time.
        timings=spec.get('launcher_timings',[])
        if len({Path(p).resolve() for p in timings})!=len(timings):raise ValueError('duplicate launcher timing')
        times=[read(p) for p in timings]
        result['accounting']=dict(recorded_gpu_hours=sum(t['gpu_hours'] for t in times) if times else None,
                                  wall_seconds=sum(t['wall_seconds'] for t in times) if times else None,
                                  attempts=len(times),timing_status='MEASURED' if times else 'UNMEASURED')
        return result
    if family == 'SP':
        from kit.sp_read import main
        args=['--sidecar',str(source),'--out',str(out)]
        if spec.get('inventory'): args+=['--inventory',spec['inventory']]
        for parent in spec.get('input',[]): args+=['--input',parent]
        main(args)
        return read(out/'sp-tables.json')
    if family == '4A':
        from kit.a4_read import job
        # Native live-tree 4A reading also accounts allocation tails. Prefer the
        # native archived view when supplied, exactly as the archive workflow.
        job(NS(out=source if source.is_dir() else out, source=None if source.is_dir() else source,
               standin=standin, bootstrap=bootstrap))
        return read((source/'reading/summary.json') if source.is_dir() else (out/'returned-reading/summary.json'))
    raise ValueError('unknown reader')


def ranges(values):
    out=[]
    for value in values:
        if out and value==out[-1][-1]+1:out[-1].append(value)
        else:out.append([value])
    return ','.join(str(v[0]) if len(v)==1 else f'{v[0]}-{v[-1]}' for v in out)


def job_summary(specs):
    """Public inventory indexes identify registered tasks without machine IDs.

    The local map retains original task names. Every registered task is present
    in exactly one completion range, even if it has never written a state.
    """
    lines=[]; private={}
    labels,_=job_inventory();public_index={name:i for i,name in enumerate(labels,1)}
    for index,spec in enumerate(specs,1):
        group=spec['family']
        if group not in FAMILIES and group not in ('core','preflight'):raise ValueError('unknown job family')
        key=group+str(index)
        try:tasks=read(spec['manifest'])['tasks']
        except (OSError,ValueError,KeyError):
            lines.append(f'J|{key}|unknown|PENDING');continue
        buckets=defaultdict(list);unknown=defaultdict(list)
        for number, task in enumerate(tasks,1):
            state={}
            if spec.get('durable'):
                from kit.m2_snapshot import read_pointer, Pending
                try:
                    pointer=read_pointer(spec['durable'],task['id'])
                    state={'status':'DONE' if pointer['complete'] else 'PENDING'}
                    # Read only the manifest-bound execution receipt in place;
                    # this neither restores weights nor consumes a disk reserve.
                    from kit.m2_snapshot import header, archive_lease
                    import tarfile,hashlib
                    archive=Path(spec['durable'])/'snapshots'/task['id']/pointer['tar']
                    with archive_lease(archive):
                        doc=header(archive,pointer['manifest_sha256'])
                        if doc['complete']!=pointer['complete']:raise ValueError('completion metadata mismatch')
                        name='execution/'+task['id']+'.json'
                        if name in doc['files']:
                            with tarfile.open(archive,'r:') as stream:raw=stream.extractfile('tree/'+name).read()
                            if hashlib.sha256(raw).hexdigest()!=doc['files'][name]['sha256']:raise ValueError('execution receipt digest')
                            result=json.loads(raw).get('result')
                            if result not in (None,0,3):state={'status':'FAILED'}
                except (Pending,OSError,ValueError,KeyError):state={'status':'PENDING'}
            else:
                path=Path(spec['state'])/(task['id']+'.json')
                if path.exists():
                    try:state=read(path)
                    except (OSError,ValueError):state={'status':'PENDING'}
            status=state.get('status','PENDING')
            if state.get('schema')=='pd-node-launch-receipt-v1' and status=='PENDING' and state.get('error_type'):
                status='FAILED'  # process failure; native scientific decision stays PENDING
            status='DONE' if status in ('DONE','COMPLETE','PROCESS_COMPLETED','PASS','SKIP') else 'FAILED' if status in ('FAILED','FAIL','CRASHED') else 'PENDING'
            if task['id'] in public_index:
                number_key=str(public_index[task['id']]);buckets[status].append(public_index[task['id']])
            else:
                number_key='u'+str(number);unknown[status].append(number)
            private[key+':'+number_key]=task['id']
        for status in sorted(set(buckets)|set(unknown)):
            groups=[]
            if buckets[status]:groups.append(ranges(sorted(buckets[status])))
            if unknown[status]:groups.append(','.join('u'+v for v in ranges(unknown[status]).split(',')))
            lines.append(f'J|{key}|'+','.join(groups)+f'|{status}')
    return lines,private


def accounting(readers):
    totals={}
    for family, results in readers.items():
        values=[]
        for r in results:
            try:
                v = (r.get('cost',{}).get('recorded_gpu_hours') if family in ('M','X') else
                     (r.get('costs',{}).get('actual',{}).get('total') if r.get('costs',{}).get('actual',{}).get('attempts') else None) if family=='KCR' else
                     r.get('accounting',{}).get('recorded_gpu_hours') if family=='PD' else
                     (r.get('wall_time',{}).get('gpu_hours') if r.get('wall_time',{}).get('status')=='MEASURED' else None) if family=='SP' else
                     r.get('shared_study_gpu_hours') if family=='4A' else
                     r.get('accounting',{}).get('recorded_gpu_hours'))
            except (AttributeError,TypeError,KeyError):v=None
            values.append(v if isinstance(v,(int,float)) and math.isfinite(v) and v>=0 else None)
        total=sum(values) if values and all(v is not None for v in values) else None
        totals[family]=total if total is not None and math.isfinite(total) else None
    known=sum(v for f,v in totals.items() if f in ('KCR','PD','SP') and v is not None)
    return dict(by_family=totals,tier_recorded_gpu_hours=known if math.isfinite(known) else None,tier_complete=all(totals.get(f) is not None for f in ('KCR','PD','SP')),
                allowance_gpu_hours=1800,tier_over_allowance_warning=known>1800,total_recorded_gpu_hours=(sum(v for v in totals.values() if v is not None) if math.isfinite(sum(v for v in totals.values() if v is not None)) else None),
                allocation_idle_reconciliation='PENDING',budget_gate=False)


def report(inputs, out, *, standin=False, tag='kit-message-v3', commit=None, max_chars=12000, bootstrap=10000, full_detail=False):
    readers={}; out=Path(out)
    with tempfile.TemporaryDirectory(prefix='text-readers-') as tmp:
        for family in FAMILIES:
            specs=inputs.get(family,[])
            results=[]
            for i,spec in enumerate(specs):
                # Family failures never prevent the remaining readers running.
                # Native output/tracebacks can contain private paths. Suppress
                # them here, including their Python warnings; never echo them.
                with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
                    try:
                        r=native(family,spec,Path(tmp)/str(i),standin,bootstrap)
                    except Exception as exc:
                        missing=isinstance(exc,FileNotFoundError)
                        refused=isinstance(exc,ValueError) or type(exc).__name__ in ('NotReady','Pending','Unbound')
                        r={'_reader_status':'PENDING' if missing or refused else 'UNVERIFIED','_warning':'MISSING_INPUT' if missing else 'READER_REFUSED' if refused else 'READER_ERROR'}
                        r['_local_exception']=type(exc).__name__+': '+str(exc)
                def warning_count(v):
                    if isinstance(v,dict):
                        return sum((len(x) if isinstance(x,list) else int(bool(x))) if k in ('warnings','flags','wall_time_warnings') else warning_count(x) for k,x in v.items())
                    if isinstance(v,list):return sum(warning_count(x) for x in v)
                    return 0
                try:
                    r['warning_count']=warning_count(r)
                    failures=r.get('identity_failures',[])
                    r['warning_summary']=dict(wall_time_risk=bool(r.get('wall_time',{}).get('flags') or r.get('wall_time_warnings')),
                        untimed=r.get('wall_time',{}).get('status')=='UNMEASURED' or r.get('timing_status')=='UNMEASURED',
                        unverified_constituents=len(failures),
                        by_seed={str(seed):sum(len(f.get('cell',[]))>1 and f['cell'][1]==seed for f in failures) for seed in (101,102,103)},
                        reader_unavailable=bool(r.get('_reader_status')))
                    r['warning_count']=warning_count(r)
                    canonical(r)
                except Exception as exc:
                    r={'_reader_status':'UNVERIFIED','_warning':'READER_ERROR','warning_count':1,'_local_exception':type(exc).__name__+': '+str(exc)}
                results.append(r)
            readers[family]=results or [{'_reader_status':'DEFERRED' if family=='4A' else 'PENDING','_warning':'NOT_CONFIGURED'}]
    jobs,private_jobs=job_summary(inputs.get('jobs',[]))
    if not jobs: jobs=['J|core|unknown|PENDING','J|KCR|unknown|PENDING','J|PD|unknown|PENDING','J|SP|unknown|PENDING']
    full=dict(schema=SCHEMA,technical_synthetic=standin,readers=readers,accounting=accounting(readers),jobs=jobs)
    # S collector costs are node-local input measurements, not bootstrap samples.
    if commit is None:
        identity=Path(__file__).with_name('KIT-IDENTITY.json')
        if identity.exists():
            ident=read(identity);commit=ident['commit'];tag=ident['tag']
        else:
            commit=subprocess.check_output(['git','-C',str(Path(__file__).resolve().parent),'rev-parse','HEAD'],text=True,stderr=subprocess.DEVNULL).strip()
    paths=write_parts(out,full,tag,commit,jobs,max_chars,full_detail=full_detail)
    (out/'job-map.local.json').write_text(canonical(private_jobs)+'\n')
    return paths


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--kit-tag',default='kit-message-v3');p.add_argument('--kit-commit')
    p.add_argument('--full',action='store_true',help='select numbered FULL parts (both sets are always written)')
    p.add_argument('--standin',action='store_true');p.add_argument('--max-chars',type=int,default=12000)
    a=p.parse_args(argv)
    try: paths=report(read(a.inputs),a.out,standin=a.standin,tag=a.kit_tag,commit=a.kit_commit,max_chars=a.max_chars,full_detail=a.full)
    except Exception:
        try:
            a.out.mkdir(parents=True,exist_ok=True)
            (a.out/'refusal.local.txt').write_text(traceback.format_exc())
        except OSError:pass
        print('Text report refused: invalid inputs, identity, output directory, or part capacity. Check local configuration.',file=sys.stderr)
        return 2
    print(f"Wrote {len(paths)} {'FULL' if a.full else 'ESSENTIAL'} ASCII parts; paste ESSENTIAL Parts 1-3 first (4 if present). FULL and *.local.json stay on the node.")
    return 0


if __name__=='__main__':raise SystemExit(main())
