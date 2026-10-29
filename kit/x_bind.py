#!/usr/bin/env python3
"""X1: bind resolved durable M9 and M10 disposition, three B finals and M3 targets."""
from pathlib import Path
import sys
if __package__ in (None, ''): sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kit.x_common import *


def durable_admission(core, message, durable):
    """Read CURRENT without opening M's lock for writing or creating lock dirs.

    The core may keep collecting independently. A changing CURRENT is a stop,
    rather than permission for the add-on to acquire a writer lock in M.
    """
    config = m.admission(core, message)
    require(config in m.CONFIGS, 'resolved admission must name one registered configuration')
    if (durable/'barriers/M9.json').exists():
        return m.durable_admission(core,message,durable)
    from kit.message_handoff import committed
    pointer = durable/'CURRENT.json'
    require(pointer.is_file(), 'M9 durable CURRENT pending')
    before = sha(pointer)
    generation = committed(durable, verify=False)
    require(generation is not None, 'M9 durable admission pending')
    manifest = read(generation/'manifest.json')
    require('admission.json' in manifest['entries'], 'M9 CURRENT has no admission')
    ref = generation/'tree/admission.json'
    require(sha(ref) == manifest['entries']['admission.json']['sha256'] == sha(core/'admission.json'),
            'M9 durable admission differs from local admission')
    require(sha(pointer) == before, 'M CURRENT changed during X binding; retry binding with stable evidence')
    return config


def m10(core, message):
    """Derive completion/registered skip/pending from existing frozen core evidence."""
    if message['hard_math'] != 'QUALIFIES':
        status = read(core/'stage3-status.json')
        expected = 'DROPPED' if message['hard_math'] == 'DOES_NOT_QUALIFY' else 'PENDING'
        require(status['stage3'] == expected, 'M10 disposition mismatch')
        log = core/'logs/m2-run-math-sibling.log'
        require(log.is_file() and 'SKIP: stage 3 '+message['hard_math'] in log.read_text(),
                'M10 has not recorded its skip/pending disposition')
        return dict(status=expected, evidence=sha(core/'stage3-status.json'), m10_log_sha256=sha(log))
    log = core/'logs/m2-run-math-sibling.log'
    if log.is_file() and 'PENDING: sibling MATH waits; return evidence' in log.read_text():
        return dict(status='PENDING',m10_log_sha256=sha(log),reason='registered M10 pending disposition')
    required = [core/'runs'/f'math-{arm}-s{seed}'/'complete.json'
                for arm in ('sibling',) for seed in SEEDS]
    require(all(p.exists() for p in required), 'M10 completion or registered disposition pending')
    for p in required: m.frozen_files(p.parent)
    scores = [core/'scores'/f'math-sibling-s{seed}-k{step}-{panel}'/'manifest.json'
              for seed in SEEDS for step in STEPS for panel in m.PANEL_SETS]
    require(all(p.exists() for p in scores), 'M10 boundary/curve scoring unfinished')
    for p in scores: m.frozen_files(p.parent)
    required += scores
    return dict(status='COMPLETE', evidence={str(p.relative_to(core)): sha(p) for p in required})


def job(a):
    if (a.message2_durable/'barriers/M9.json').exists():independent(a.out,a.message2,a.screen_s)
    else:independent(a.out,a.durable,a.message2,a.message2_durable,a.screen_s,a.screen_s_durable)
    message = m.binding(a.message2, a.standin)
    config = durable_admission(a.message2, message, a.message2_durable)
    status = m10(a.message2, message)
    screen = __import__('kit.s_common', fromlist=['context']).context(a.screen_s, a.standin)
    assets = read(a.message2/'assets.json')
    for ref in assets['inputs'].values():
        require(sha(Path(ref['path'])/'manifest.json') == ref['manifest'], 'M3 input changed')
        m.frozen_files(Path(ref['path']))
    spec = read(a.spec)
    from kit.p2_l2 import VARIANT,REVISION
    variant=spec.get('p_variant','disabled')
    require(variant in ('disabled',VARIANT),'closed or unknown P variant')
    anchor=dict(message['starts']['base'])
    if variant==VARIANT:
        anchor['identity']=m.checked_model(anchor,a.standin,base=True)
        anchor['revision']=REVISION
    require(set(spec['incomings']) == {str(s) for s in SEEDS}, 'all three incoming exposure proofs required')
    nodes, identities = {}, {}
    for seed in SEEDS:
        ref = message['starts'][f'finqa-sibling:{seed}']
        observed = m.checked_model(ref, a.standin)
        require(observed == ref['identity'], 'M2 bound B final changed')
        ident = f'b-s{seed}'
        checkpoint_ref = dict(ref)
        if a.standin:
            checkpoint_ref.update(technical_synthetic=True, identity=observed.get('identity', str(seed)))
        identities[ident] = model(checkpoint_ref, a.standin)
        nodes[ident] = dict(checkpoint=checkpoint_ref, exposure=spec['incomings'][str(seed)], root=False)
    panels = {name: sha(checked(ref)) for name, ref in screen['spec']['panels'].items()}
    # Freeze the actual M3 target manifest, never a regenerated ContractNLI pool.
    target = Path(assets['inputs']['c_targets']['path'])
    for seed in SEEDS:
        require(len(rows(target/f'schedule-{seed}.jsonl')) == 1280, 'M3 requires fixed 40-update target schedule')
    audit, partition = a.screen_s/'audit.json', a.screen_s/'frozen/manifest.json'
    require(read(partition)['audit_sha256'] == sha(audit), 'Screen S partition/audit mismatch')
    # P2 has no utility batch/target prerequisite. FinQA exposure membership
    # remains in each closed ancestry proof used by X2, without a second file.
    for name in ('ifeval', 'gsm8k', 'arc', 'finqa'):
        from kit.s_freeze import partitions
        partitions(a.screen_s, name)
    doc = dict(message2=str(a.message2.resolve()), message2_sha256=sha(a.message2/'binding.json'),
        admission_sha256=sha(a.message2/'admission.json'), admission=read(a.message2/'admission.json'),
        core_scoring_machine=dict(identity=read(a.message2/'m2-scoring-machine.json'),
            sha256=sha(a.message2/'m2-scoring-machine.json')),
        p_variant=variant,p2_anchor=anchor,p2_node_limits_sha256=sha(ROOT/'kit/resources/p2/node-limits.json'),
        m10=status, configuration=config, domains=domains(config), protected=[PANELS[d] for d in domains(config)],
        assets=assets, base=message['starts']['base'], screen_s=str(a.screen_s.resolve()), screen_binding_sha256=sha(a.screen_s/'binding.json'),
        screen_audit_sha256=sha(audit), screen_partition_sha256=sha(partition), panels=panels,
        sources=screen['sources'], identities=identities,
        spec=dict(nodes=nodes, stages=[dict(incoming=f'b-s{s}') for s in SEEDS], panels=screen['spec']['panels']),
        registration_sha256=sha(RESOURCE), implementation=implementation(),
        scoring_protocol=message['protocol'],
        technical_synthetic=a.standin)
    bind(a.out/'binding.json', doc)
    # Preserve compatible original comparators/base evidence for inherited bars
    # and acquired-answer survival. Missing comparators remain pending.
    import csv
    inherited = []
    arms = {'base', 'finqa-sibling', 'standalone-legal-'+config, 'standalone-legal-plain'}
    for path in sorted((a.message2/'scores').glob('*/scores.csv')):
        with path.open() as f: values = list(csv.DictReader(f))
        if values and values[0]['arm'] in arms:
            m.frozen_files(path.parent)
            if not (a.message2/'scoring-assignment.json').exists():inherited.extend(values)
    bind(a.out/'inherited.json', dict(rows=inherited, sha256=digest(inherited)))
    # Local scoring assets only; M trees are never modified.
    import shutil
    source = a.message2/'hellaswag'
    if not (a.out/'hellaswag').exists(): shutil.copytree(source, a.out/'hellaswag')
    require(sha(a.out/'hellaswag/manifest.json') == sha(source/'manifest.json'), 'HellaSwag changed')
    print('X1 bound:', config, '; M10:', status['status'])


def configure(p):
    for name in ('durable', 'message2', 'message2-durable', 'screen-s', 'screen-s-durable', 'spec'):
        p.add_argument('--'+name, type=Path, required=True)
if __name__ == '__main__': raise SystemExit(entry(job, configure))
