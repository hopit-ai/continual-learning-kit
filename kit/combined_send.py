#!/usr/bin/env python3
"""Inventory and execute one explicit combined-send row; never submits or retries."""
from pathlib import Path
import sys
if __package__ in (None, ''): sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import hashlib
import json
import os
import socket
import subprocess
from kit import m2_node

KIT = Path(__file__).resolve().parent
SCHEMA = 'combined-send.v1'
REVIEW = KIT.parent/'docs/send/final-review-redacted.md'
WRAPPERS = {'prepare': 'prepare', 'train-stage': 'train-stage', 'score-family': 'score-family',
            'reduce-admission': 'reduce-admission', 'collect': 'collect'}


def load(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def host(value): return str(value['node'] if isinstance(value,dict) else value).split('.')[0]
def need(ok, message):
    if not ok: raise ValueError(message)
def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True)+'\n')


def price(task, group):
    """Per-job amounts are components; allowances below are not added twice."""
    if group == 'kcr':
        arm=task.get('arm'); stage=task.get('stage')
        if task['kind']=='train-stage':
            ranges={'C0': [[15.49,25.56]]*3, 'K': [[16.06,28.09]]*3,
                    'G': [[19.37,31.95],[23.24,38.34],[27.11,44.73]],
                    'CR': [[19.37,31.95],[23.24,38.34],[27.11,44.73]]}
            index={'finqa':0,'contract_nli':1,'math_hard':2}.get(stage)
            return dict(gpu_h=ranges[arm][index] if arm in ranges and index is not None else None,
                        basis='K/CR measured-smoke calibrated stage; setup/scoring/qualification are separate envelope components')
        if task['kind']=='preflight':
            return dict(gpu_h={'C0':[4.80,9.16],'K':[4.94,9.79],'G':[6.69,12.24],'CR':[6.69,12.24]}[arm],
                        basis='K/CR one qualification per arm; includes diagnostic/setup reserve')
        if task['id'].startswith('M3-rescore-') and task['kind']=='score-family':
            return dict(gpu_h=None, basis='K/CR complete pinned generated-panel seed lane; measured cost pending')
        if task['kind']=='score-family':
            return dict(gpu_h=[.8,1.6] if stage else None,
                        basis='K/CR 16 call-equivalent allowance per arm/seed/stage' if stage else 'K/CR shared base/validation total 2.10–4.20; per-job measurement pending')
        if task['id'] in ('M3-broad','M3-math-targets'):
            return dict(gpu_h=[2.,25.] if task['id']=='M3-broad' else [10.,30.],
                        basis='K/CR shared generation; compatible committed core targets remove new math generation charge')
        return dict(gpu_h=[0,0], basis='K/CR CPU/handoff')
    if group == 'pd': return dict(gpu_h=[73.864972, 87.873469], basis='PD measured reprice, per seed; within approved tier component allowance')
    if group == 'sp':
        n = len(task.get('entries', []))
        return dict(gpu_h=[n*.25, n] if task.get('uses_gpu') else [0, 0],
                    basis='SP(b) .25–1 GPU-h/entry; startup separate; unresolved coverage has no finite upper total')
    if task.get('id')=='M5':
        return dict(gpu_h=[10.,10.], basis='core 500-item bounded 27B target allowance; charge once')
    if task.get('row')=='S10' and task['kind']=='train-stage':
        return dict(gpu_h=[.75,1.4], basis='core Screen S four-update probe; included in S envelope')
    if task.get('row')=='X4':
        return dict(gpu_h=None, basis='core X4 all-three-seed feasibility/calibration envelope 5–7 GPU-h; per-job measurement pending')
    if task['kind']=='train-stage' and task['id'].startswith('X9-probes-'):
        return dict(gpu_h=[1.5,2.8], basis='two core probes x .75–1.4; included in X probe envelope')
    if task['kind'] == 'train-stage':
        # The core runbook prices legal/MATH full stages by the same measured D proxy.
        phase = next((c[c.index('--phase')+1] for c in task.get('commands', []) if '--phase' in c), None)
        return dict(gpu_h=[58*8/60, 58*8/60] if phase in (None,'full') else None,
                    basis='core 58 min x 8 GPU proxy/full stage' if phase in (None,'full') else 'split stage; included in full-stage envelope, measured split pending')
    gpu = task['kind'] == 'score-family' or task.get('uses_gpu') or task.get('row') in ('M3b', 'M5', 'S8', 'S10')
    return dict(gpu_h=None if gpu else [0, 0], basis='core measured scoring/preparation/qualification allocation pending' if gpu else 'CPU/handoff; no GPU compute')


def resources(task):
    request=task.get('resources',task)
    gpus=request.get('gpus')
    if gpus is None:
        gpus = (8 if task.get('kind') in ('train-stage', 'preflight') else
                2 if task.get('id') in ('M5', 'M3-math-targets') else
                1 if task.get('kind') == 'score-family' or task.get('uses_gpu') and not task['id'].startswith('PF-') else 0)
    need(gpus in (0, 1, 2, 8), 'unsupported GPU request')
    return dict(gpus=gpus, nodes=1, cpus=request.get('cpus',32 if gpus==8 else 8), share_node=gpus != 8)


def build(settings):
    from kit import m3_manifest, sp_manifest
    core = m2_node.load_manifest(settings['core_manifest'])
    tier = m3_manifest.load(settings['kcr_manifest'])
    side = load(settings['sp_manifest'])
    inv = load(settings['sp_inventory'])
    need(m3_manifest.isolation(tier, core)['passed'], 'K/CR isolation proof failed')
    sp_manifest.validate_sidecar(side, [core, tier])
    sp_manifest.verify_seal(inv)
    need(side['inventory_sha256'] == inv['identity_sha256'], 'SP inventory differs')
    nodes = dict(core=[host(n) for n in core['nodes']], kcr=[host(n) for n in tier['nodes']],
                 pd=[host(n) for n in settings['pd_nodes']], sp=[host(n) for n in settings['sp_nodes']])
    need(all(nodes.values()), 'all groups need explicit nodes')
    roots = [Path(settings[k]).resolve() for k in ('core_local_root', 'core_durable', 'kcr_root', 'pd_root', 'sp_root', 'control_root')]
    need(all(a != b and a not in b.parents and b not in a.parents for i,a in enumerate(roots) for b in roots[i+1:]), 'state roots must be separate, non-nested trees')
    rows=[]
    # Preflight is a first wave, regardless of the source manifests' ordering.
    for task in core['tasks']:
        if task['id'].startswith('PF-'):
            rows.append(dict(id=task['id'], group='core', source_task=task['id'], requires=task.get('requires', []),
                             phase='preflight', estimate=price(task,'core'), node_slot=task.get('node_slot'), kind=task['kind'], **resources(task)))
    for i,node in enumerate(sorted(set(sum(nodes.values(), [])))):
        rows.append(dict(id=f'PF-tier-{i}', group='preflight', node=node, requires=[], phase='preflight',
                         estimate=dict(gpu_h=[0,0], basis='CPU dependency/cgroup checks; UUID checked at GPU admission'), **resources({})))
    for group, source in (('core', core), ('kcr', tier), ('sp', side)):
        for task in source['tasks']:
            if task['id'].startswith('PF-'): continue
            rows.append(dict(id=task['id'], group=group, source_task=task['id'], kind=task['kind'],
                requires=task.get('requires', task.get('after', [])),
                partial_inputs=task.get('partial_inputs',False),
                dependency='afterany' if group=='sp' or task.get('partial_inputs') else 'afterok',
                phase='core' if group=='core' else 'tier', estimate=price(task,group), node_slot=task.get('node_slot'),command_count=task.get('command_count',task.get('expected_commands',len(task.get('commands',[])))), **resources(task)))
    from kit.pd_jobs import resource_inventory
    for task in resource_inventory()['rows']:
        if task.get('seed') is not None:need(str(task['seed']) in settings['pd_plans'],'PD needs all three seed plans')
        rows.append(dict(id=task['id'],group='pd',kind='collect' if task['phase']=='read' else
            'train-stage' if task['phase']=='train' else 'score-family' if task['gpus']==1 else 'prepare',
            seed=task.get('seed'),phase='tier',pd_phase=task['phase'],requires=task['requires'],dependency=task['dependency'],
            estimate=dict(gpu_h=[0,0] if not task['gpus'] else None,basis='PD split phase; all seeds 63–98 GPU-h; measured phase cost pending'),
            **resources(task)))
    # Reference guards are independent serial (family, seed, panel) lanes. Local
    # settings bind restored native roots/lists; no model paths enter the table.
    families = ['core','reprobe','m3', *sorted({t['family'] for t in core['tasks'] if t.get('family', '').startswith('x-tier')})]
    reference = settings.get('reference_jobs') or [dict(family=f, out=settings['kcr_root'] if f == 'm3' else
        str(Path(settings['core_local_root']) / ('reference-' + f)), checkpoints=None) for f in families]
    refs = []
    for item in reference:
        family = item['family']; root = item['out']; checkpoints = item.get('checkpoints') or str(Path(root) / 'checkpoints.local.json')
        prefix = 'REF-' + family
        command = [settings['inference_python'], '-I', str(KIT / 'ref_score_job.py'), '--out', root, '--family', family]
        assign = prefix + '-assign'
        rows.append(dict(id=assign,group='reference',kind='prepare',family=family, requires=[], phase='tier', dependency='afterok',
            commands=[[*command, '--action', 'assign', *(['--lane-assignment', item['lane_assignment']] if item.get('lane_assignment') else [])]],
            estimate=dict(gpu_h=[0,0], basis='CPU reference assignment freeze'), **resources({})))
        for seed in (101,102,103):
            ident=f'{prefix}-{seed}';prepared=ident+'-prepare'
            args=['--checkpoints',checkpoints,'--seed',str(seed)]
            rows.append(dict(id=prepared,group='reference',kind='prepare',family=family,requires=[assign,*item.get('requires',[])],
                phase='tier',dependency='afterok',commands=[[*command,'--action','prepare',*args,'--panel',panel] for panel in ('hellaswag','knowledge-ll')],
                estimate=dict(gpu_h=[0,0],basis='CPU reference checkpoint verification'),**resources({})))
            rows.append(dict(id=ident,group='reference',kind='score-family',requires=[prepared],phase='tier',dependency='afterok',
                commands=[[*command,'--action','score',*args,'--panel','both','--gpu','{score_gpu}']],family=family,seed=seed,
                command_count='2 × decision checkpoints (both panels)',
                estimate=dict(gpu_h=None,basis='both complete HF panel lanes on one allocated GPU; K=3'),**resources(dict(gpus=1))))
            refs.append(ident)
    rows.append(dict(id='REPORT-text', group='report', kind='collect', requires=refs, phase='tier', dependency='afterany',
        commands=[[settings['inference_python'], '-I', str(KIT / 'report_text.py'), '--inputs', settings.get('report_inputs',
            str(Path(settings['control_root']) / 'report-inputs.local.json')), '--out', settings.get('text_out',
            str(Path(settings['control_root']) / 'return-text'))]],
        estimate=dict(gpu_h=[0,0], basis='CPU reporter and reference guards'), **resources({})))
    for row in rows:
        if row['gpus']==1:row['lane_concurrency']=3
    ids={r['id'] for r in rows}
    need(len(ids)==len(rows), 'duplicate combined IDs')
    need(all(set(r['requires'])<=ids for r in rows), 'unknown combined prerequisite')
    coreids={r['id'] for r in rows if r['group']=='core'}
    need(all(set(r['requires'])<=coreids for r in rows if r['group']=='core'), 'core depends on tier')
    return dict(schema=SCHEMA, technical_synthetic=bool(settings.get('technical_synthetic')), nodes=nodes, settings=settings,
        sources={key:sha(settings[key]) for key in ('core_manifest','kcr_manifest','sp_manifest','sp_inventory')}, tasks=rows,
        budgets=[dict(component='Core M', gpu_h='109.6 floor includes 46.4 MATH +10 targets +30 scoring/export +23.2 legal; +23.2 per confirmation set; fresh 12-stage re-probe +92.8 proxy; independent 31-call rescore/overhead unpriced'),
          dict(component='Screen S', gpu_h='30–50 envelope; v2 scoring/transfer measurement pending'),
          dict(component='X tier 1', gpu_h='12 x 7.733333 = 92.8 training; probes/generation/scoring/X4/startup additional'),
          dict(component='X tier 2', gpu_h='36 x 7.733333 = 278.4 training; 295.2–316.8 historical envelope; v2 overheads remain measured separately'),
          dict(component='K/CR', gpu_h=[786.297517,1387.157525]),
          dict(component='PD', gpu_h=[62.211758,97.382143], rounded_request='63–98; approved component <=264 GPU-h'),
          dict(component='SP', gpu_h=inv['price']['sp_b_gpu_hours'], basis='entry sum 53.75–215 plus 1 GPU-h startup =54.75–216; SP(a)=0')],
        send_gate='Message-3 joins only on Opus SEND by 2026-11-02; otherwise send core alone. No launch authority.',
        total_gpu_h=None, total_reason='Incomplete measured components; component envelopes overlap per-job amounts; do not sum both.')


def build_v6a(settings):
    """Stage only completed K/CR training's scoring; no other send is an input."""
    from kit import m3_manifest
    tier=m3_manifest.load(settings['kcr_manifest'])
    root=Path(settings['kcr_root']).resolve();control=Path(settings['control_root']).resolve()
    need(root!=control and root not in control.parents and control not in root.parents,
         'K/CR and control roots must be separate, non-nested trees')
    rows=[];gpu_rows=[]
    for seed in (101,102,103):
        ident=f'M3-rescore-{seed}'
        for name in (ident+'-ready',ident):
            task=next(t for t in tier['tasks'] if t['id']==name)
            commands=[list(c) for c in task['commands']]
            for command in commands:
                if '--workspace' in command:
                    i=command.index('--workspace');del command[i:i+2]
            rows.append(dict(id=name,group='kcr',kind=task['kind'],seed=seed,
                requires=[] if name.endswith('-ready') else [ident+'-ready'],
                dependency='afterok',phase='v6a',commands=commands,
                scoring_ready=task.get('scoring_ready'),
                command_count=len(commands),lane_concurrency=3 if name==ident else 0,
                estimate=price(task,'kcr'),**resources(dict(gpus=int(name==ident)))))
        gpu_rows.append(ident)
    command=[settings['inference_python'],'-I',str(KIT/'ref_score_job.py'),
             '--out',str(root),'--family','m3']
    checkpoints=str(control/'reference-checkpoints.local.json')
    rows.append(dict(id='REF-m3-assign',group='reference',kind='prepare',family='m3',
        requires=[],dependency='afterok',phase='v6a',
        commands=[[*command,'--action','assign','--checkpoints',checkpoints,'--schedule-checkpoints']],
        estimate=dict(gpu_h=[0,0],basis='CPU assignment and schedule checkpoint list'),**resources(dict(gpus=0))))
    for seed in (101,102,103):
        ident=f'REF-m3-{seed}';args=['--checkpoints',checkpoints,'--seed',str(seed)]
        rows.append(dict(id=ident+'-prepare',group='reference',kind='prepare',family='m3',seed=seed,
            requires=['REF-m3-assign'],dependency='afterok',phase='v6a',
            commands=[[*command,'--action','prepare',*args,'--panel',p] for p in ('hellaswag','knowledge-ll')],
            estimate=dict(gpu_h=[0,0],basis='CPU complete checkpoint verification'),**resources(dict(gpus=0))))
        rows.append(dict(id=ident,group='reference',kind='score-family',family='m3',seed=seed,
            requires=[ident+'-prepare'],dependency='afterok',phase='v6a',lane_concurrency=2,
            commands=[[*command,'--action','score',*args,'--panel','both','--gpu','{score_gpu}']],
            command_count=13,estimate=dict(gpu_h=None,basis='13 decision checkpoints, both HF panels per model load; K=2'),
            **resources(dict(gpus=1))))
        gpu_rows.append(ident)
    rows.append(dict(id='REPORT-text',group='report',kind='collect',requires=gpu_rows,
        dependency='afterany',phase='v6a',commands=[],
        estimate=dict(gpu_h=[0,0],basis='CPU K/CR readers and Parts 1–3'),**resources(dict(gpus=0))))
    return dict(schema=SCHEMA,mode='v6a',technical_synthetic=bool(settings.get('technical_synthetic')),
        nodes={'kcr':[host(n) for n in tier['nodes']]},settings=dict(settings),
        sources={'kcr_manifest':sha(settings['kcr_manifest'])},tasks=rows,budgets=[],
        send_gate='v6a only: completed K/CR training reused; core M/X/S, PD and SP follow as v6b.',
        total_gpu_h=None)


def inventory(doc):
    lines=['# Combined launch inventory', '', doc['send_gate'], '',
           'Generated from every source task. Runtime locations stay in the local JSON; this table contains portable IDs only.', '',
           '| Job | Group | Prerequisites | Dependency | gpus | nodes | cpus | share node | commands | GPU-h estimate / basis |', '|---|---|---|---|---|---|---|---|---|---|']
    for row in doc['tasks']:
        p=row['estimate']; lines.append('| '+ ' | '.join((row['id'],row['group'],', '.join(row['requires']) or 'none', row.get('dependency','first wave'), str(row['gpus']), str(row['nodes']), str(row['cpus']), 'yes' if row['share_node'] else 'no',str(row.get('command_count',3 if row.get('pd_phase')=='pre' else 18 if row.get('pd_phase')=='post' else 'phase worker' if row['gpus'] else 'CPU')), str(p['gpu_h'])+'; '+p['basis']))+' |')
    lines += ['', 'Component envelopes (overlap per-job components; not additive):', '']
    for row in doc['budgets']: lines.append('- '+row['component']+': '+str(row['gpu_h'])+ ('; '+row['basis'] if row.get('basis') else ''))
    if doc.get('mode')!='v6a':lines += ['', 'Approved tier allowance: 1,800 GPU-h; component limits K/CR <=1,400, PD <=264, SP <=216. The revised upper case is about 1,700.54 GPU-h before unpriced rescoring; a PD whole-seed restart adds about 20.74–32.46 GPU-h plus CPU/queue cost. These are warnings monitored in partner receipts, never launch/scientific gates; no cells are dropped. Core is separate. Queue/restore/transfer time and failed attempts must be charged separately.', '']
    return '\n'.join(lines)


def environment(s):
    return dict(os.environ, KIT=str(KIT), MANIFEST=s['core_manifest'], M2_MANIFEST=s['core_manifest'],
        LOCAL_ROOT=s['core_local_root'], DURABLE=s['core_durable'], M3_OUT=s['kcr_root'], M3_MANIFEST=s['kcr_manifest'],
        SP_ROOT=s['sp_root'], SIDECAR=s['sp_manifest'], TRAINER_PY=s['trainer_python'], INFERENCE_PY=s['inference_python'])


def execute(doc, ident, *, runner=subprocess.run):
    need(doc.get('schema')==SCHEMA, 'combined schema differs')
    if doc.get('mode')=='v6a':
        from kit.v6a import execute as execute_v6a
        return execute_v6a(doc,ident,runner=runner)
    s=doc['settings']; row=next(r for r in doc['tasks'] if r['id']==ident); group=row['group']
    need(not doc['technical_synthetic'], 'synthetic inventory is CPU rehearsal only')
    need(sha(s['core_manifest'])==doc['sources']['core_manifest'], 'source manifest changed: core_manifest')
    node=host(socket.gethostname()); control=Path(s['control_root']); env=environment(s)
    def call(command, pass_fds=()):
        if row['gpus'] and group != 'core' and not env.get('UTIL_WATCH_ACTIVE'):
            command = [s['inference_python'], '-I', str(KIT / 'util_watch.py'), '--out', str(control/'util'/ident/(str(__import__('time').time_ns())+'.jsonl')), '--', *command]
            return runner(command, env=dict(env, UTIL_WATCH_ACTIVE='1'), **(dict(pass_fds=pass_fds) if pass_fds else {})).returncode
        return runner(command, env=env).returncode
    if group=='preflight':
        need(node==row['node'], 'wrong tier preflight node')
        return call([s['inference_python'],'-I',str(KIT/'combined_preflight.py'),'--settings-json',json.dumps(s),'--node',node])
    if group=='core':
        need(node in doc['nodes']['core'], 'wrong core node')
        task=next(t for t in load(s['core_manifest'])['tasks'] if t['id']==ident)
        slot=task.get('node_slot')
        need(slot is not None, 'freeze core manifest/assignment before combined build')
        need(host(load(s['core_manifest'])['nodes'][int(slot.split('/')[0])])==node, 'wrong frozen core slot')
        return call(['bash',str(KIT/'jobs/message2-v2'/f'{WRAPPERS[task["kind"]]}.sh'),'--task',ident,'--node-slot',slot])
    core_control=group=='report' or group=='reference' and row.get('family')!='m3'
    if not core_control:
        for key in ('kcr_manifest','sp_manifest','sp_inventory'):
            need(sha(s[key])==doc['sources'][key],'source manifest changed: '+key)
    env={key:value for key,value in env.items() if not key.startswith('M2_') and key not in ('MANIFEST','DURABLE','LOCAL_ROOT')}
    if not core_control:
        need(load(control/'core-submitted.json').get('core_manifest_sha256')==doc['sources']['core_manifest'], 'PENDING: record current core submissions first')
        decision=s.get('tier_send',{})
        need(decision.get('status')=='SEND' and '2026-01-01'<=decision.get('date','')<='2026-11-02' and decision.get('review_sha256')==sha(REVIEW), 'PENDING: tier needs Opus SEND by 2 November')
    allowed = set(sum(doc['nodes'].values(), [])) if group in ('reference','report') else set(doc['nodes'][group])
    need(node in allowed, 'tier must run on a configured node')
    if not core_control:
        proof=load(control/'preflight'/(node+'.json'))
        need(proof.get('status')=='PASS' and proof.get('technical_synthetic') is False and proof.get('settings_sha256')==hashlib.sha256(json.dumps(s,sort_keys=True).encode()).hexdigest(), 'tier preflight missing/stale')
        need(proof.get('hardware_sha256')==sha(s['tier_hardware']), 'tier hardware inventory changed')
    if group=='kcr': return call(['bash',str(KIT/'jobs/message3/run-task.sh'),ident])
    if group=='pd':
        root=Path(s['pd_root']);root.mkdir(parents=True,exist_ok=True)
        phase=row['pd_phase']
        if phase=='read':
            evidence=root/'seed-records.local.json'
            write(evidence,dict(seed_records=[load(root/('seed-'+str(seed))/'seed-evidence.verified.json') for seed in (101,102,103)]))
            return call([s['inference_python'],'-I',str(KIT/'pd_jobs_read.py'),'--evidence',str(evidence),
                '--exports',s.get('pd_exports',str(root/'exports.local.json')),'--trusted',s.get('pd_trusted',str(root/'trusted.local.json')),
                '--out',str(root/'reader.json')])
        seed=str(row['seed']);need(load(s['pd_plans'][seed]).get('seed')==row['seed'],'PD plan seed differs')
        return call([s['trainer_python'] if phase=='train' else s['inference_python'],'-I',str(KIT/'pd_node.py'),
            '--phase',phase,'--plan',s['pd_plans'][seed],'--out',str(root/('seed-'+seed))])
    if group in ('reference','report'):
        for command in row['commands']:
            code = call([v.replace('{score_gpu}', os.environ.get('SCORE_GPU','0')) for v in command])
            if code: return code
        return 0
    env['TASK']=ident
    if row.get('kind')=='sp-sidecar' and (ident.startswith('SP-prepare-') or ident=='SP-verify-pools'):
        # Author's VERIFY-ONLY gate: no scoring CLI starts with incomplete node inventories.
        gate=[s['inference_python'],'-I',str(KIT/'resources/sp/node_verification.py'),'run',
              '--frozen',str(KIT/'resources/sp'),'--pools',s['sp_node_pools'],
              '--receipt',str(Path(s['sp_root'])/'pool-verification.json'),'--']
    else: gate=[]
    code=call(gate+['bash',str(KIT/'jobs/message3-sp/run-task.sh')])
    if ident=='SP-read':
        code=max(code,call([s['inference_python'],'-I',str(KIT/'resources/sp/node_verification.py'),'report',
            '--frozen',str(KIT/'resources/sp'),'--receipt',str(Path(s['sp_root'])/'pool-verification.json'),
            '--sidecar',s['sp_root'],'--out',str(Path(s['sp_root'])/'verified-tables')]))
    return code

def main():
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='action',required=True)
    for action in ('build','build-v6a'):
        b=sub.add_parser(action); b.add_argument('--settings',required=True); b.add_argument('--out',required=True); b.add_argument('--inventory',required=True)
    r=sub.add_parser('run'); r.add_argument('--manifest',required=True); r.add_argument('--task',required=True)
    c=sub.add_parser('core-submitted'); c.add_argument('--manifest',required=True)
    a=p.parse_args()
    if a.action in ('build','build-v6a'):
        doc=(build_v6a if a.action=='build-v6a' else build)(load(a.settings)); need(not Path(a.out).exists(), 'use a fresh combined manifest')
        write(a.out,doc); Path(a.inventory).write_text(inventory(doc)); print(len(doc['tasks']),'combined rows; none submitted'); return 0
    doc=load(a.manifest)
    if a.action=='core-submitted':
        write(Path(doc['settings']['control_root'])/'core-submitted.json',dict(core_manifest_sha256=doc['sources']['core_manifest']))
        print('Operator attests all core jobs submitted; no scheduler call made'); return 0
    try: return execute(doc,a.task)
    except ValueError as error:
        print('PENDING:',str(error),file=sys.stderr); return 3
    except Exception as error:
        print('ERROR:',type(error).__name__,str(error),file=sys.stderr); return 1

if __name__=='__main__': raise SystemExit(main())
