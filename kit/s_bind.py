#!/usr/bin/env python3
"""Bind the prospective Screen S cohort to message 2 and original export identities."""
from pathlib import Path
import sys
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kit.s_common import *
from kit.s_inputs import check_sources


def validate_spec(spec):
    nodes, stages = spec['nodes'], spec['stages']
    import re
    expected_slots = {r['id'] for r in inventory_slots()}
    require(all(re.fullmatch(r'[A-Za-z0-9_-]+', name) for name in [*nodes, *[c['id'] for c in stages]]), 'unsafe checkpoint/stage ID')
    for name in ('v4_step20', 'v4_step10'):
        require(len(spec[name]) == len(expected_slots) and {r['slot'] for r in spec[name]} == expected_slots, 'require exact three-seed v4 slots')
    require(len(stages) == PRIMARY_STAGES, 'primary cohort requires 12 D/F stages')
    expected = {(t, r, s) for t in (1, 2) for r in ('D', 'F') for s in SEEDS}
    require({(c['tier'], c['recipe'], c['seed']) for c in stages} == expected, 'wrong primary cohort')
    require(len({c['id'] for c in stages}) == PRIMARY_STAGES, 'duplicate stage')
    for c in stages:
        require(c['incoming'] in nodes and c['tier'] in (1, 2), 'unknown incoming checkpoint')
        require(c['stage'] == ('base-to-finqa' if c['tier'] == 1 else 'finqa-to-second-task'), 'wrong stage')
        require(set(c['checkpoints']) <= {'10', '20', '40'} and {'20', '40'} <= set(c['checkpoints']), '20/40 checkpoints required')
        for step, ident in c['checkpoints'].items():
            require(ident in nodes and nodes[ident]['parent'] == c['incoming'], 'checkpoint must retain exact stage-start parent')
            require(nodes[ident]['step'] == int(step), 'checkpoint step mismatch')
        if c['tier'] == 1:
            require(nodes[c['incoming']]['root'] is True, 'tier 1 must start from base')
        else:
            start = next(x for x in stages if (x['tier'], x['recipe'], x['seed']) == (1, c['recipe'], c['seed']))
            require(c['incoming'] == start['checkpoints']['40'], 'tier 2 must start at matching v4 final')
    roots = {c['incoming'] for c in stages if c['tier'] == 1}
    require(len(roots) == 1, 'shared base checkpoint required')
    require(len({c['incoming'] for c in stages}) == INCOMING_CHECKPOINTS, 'expected 7 incoming checkpoints')
    for ident, node in nodes.items():
        visited = set()
        cursor = ident
        while cursor is not None:
            require(cursor in nodes and cursor not in visited, 'broken/cyclic ancestry')
            visited.add(cursor)
            cursor = nodes[cursor]['parent']
        require(node['root'] == (node['parent'] is None), 'invalid root')


def job(a):
    out, message2 = a.out.resolve(), a.message2.resolve()
    require(out != message2 and message2 not in out.parents, 'S_OUT must be separate from and outside M2_OUT')
    reg = registration()
    from kit.m2_common import binding
    message = binding(a.message2, a.standin)
    spec = read(a.spec)
    validate_spec(spec)
    sources = check_sources(a.sources, a.standin)
    if not a.standin:
        from kit.p3b_common import identity as original_identity
        from kit.c_bind import check_start
        base = next(c['incoming'] for c in spec['stages'] if c['tier'] == 1)
        observed_base = original_identity(spec['nodes'][base]['checkpoint']['model'], 'student')
        require(observed_base['model_file_hashes'] == original_identity(message['starts']['base']['model'], 'student')['model_file_hashes'], 'base differs from message 2')
        for stage in spec['stages']:
            incoming = spec['nodes'][stage['incoming']]['checkpoint']
            if stage['tier'] == 1:
                final = spec['nodes'][stage['checkpoints']['40']]['checkpoint']
                observed_final = check_start(final, stage['recipe'], stage['seed'], False)
                if stage['recipe'] == 'D':
                    key=f"finqa-plain:{stage['seed']}"
                    if stage['seed'] in SEEDS:require(key in message['starts'],'message-2 D incoming missing')
                    if key in message['starts']:
                        expected = model(message['starts'][key], False)
                        require(observed_final['identity']['model_file_hashes'] == expected['model_file_hashes'], 'v4 D final differs from message 2')
                for step, ident in stage['checkpoints'].items():
                    origin = Path(read(spec['nodes'][ident]['checkpoint']['receipt'])['model_dir'])
                    require(origin.name == 'hf-step'+step and f"{stage['recipe']}-finqa-alone-s{stage['seed']}" in origin.parts, 'wrong original v4 checkpoint')
            else:
                summary = read(checked(stage['summary']))
                require(summary['route'] == 'sequence' and summary['recipe'] == stage['recipe'] and summary['seed'] == stage['seed'], 'not an unreset D/F sequence stage')
                require(summary['incoming']['model_file_hashes'] == model(incoming, False)['model_file_hashes'], 'B sequence summary incoming mismatch')
                for step, ident in stage['checkpoints'].items():
                    origin = Path(read(spec['nodes'][ident]['checkpoint']['receipt'])['model_dir'])
                    require(origin.name == 'hf-step'+step and f"sequence-{stage['recipe']}-s{stage['seed']}" in origin.parts, 'wrong original B sequence checkpoint')
        for slot in spec['v4_step20']:
            origin = Path(read(slot['checkpoint']['receipt'])['model_dir'])
            require(origin.name == 'hf-step20' and slot['slot'] in origin.parts, 'wrong v4 step-20 export')
            model(slot['checkpoint'], False)
        # The immutable general panel is always the kit's registered bytes.
        require(spec['panels']['general-v1']['sha256'] == sha(ROOT/'kit/panels/general-v1.jsonl'), 'general-v1 panel changed')
    identities = {key: model(node['checkpoint'], a.standin) for key, node in spec['nodes'].items()}
    slots20 = {r['slot']: r['checkpoint'] for r in spec['v4_step20']}
    slots10 = {r['slot']: r.get('checkpoint') for r in spec['v4_step10']}
    for stage in spec['stages']:
        if stage['tier'] != 1:
            continue
        slot = f"{stage['recipe']}-finqa-alone-s{stage['seed']}"
        require(model(slots20[slot], a.standin) == identities[stage['checkpoints']['20']], 'v4 step-20 inventory/cohort mismatch')
        ref10 = slots10[slot]
        available = ref10 is not None and (a.standin or (Path(ref10['model']).is_dir() and Path(ref10['receipt']).is_file()))
        require(('10' in stage['checkpoints']) == available, 'v4 step-10 inventory/cohort availability mismatch')
        if available:
            require(model(ref10, a.standin) == identities[stage['checkpoints']['10']], 'v4 step-10 inventory/cohort mismatch')
    # The panel inventory is closed at binding, before generation or measurement.
    panels = {name: sha(checked(ref)) for name, ref in spec['panels'].items()}
    for name, ref in spec['panels'].items():
        rr = rows(checked(ref))
        require(rr and len({r['id'] for r in rr}) == len(rr), 'empty/duplicate evaluation panel: '+name)
        if not a.standin and name in reg['panel_counts']:
            require(len(rr) == reg['panel_counts'][name], 'incomplete evaluation panel: '+name)
    require(set(reg['panel_inventory']) <= set(panels), 'missing evaluation panel inventory')
    bind(a.out/'binding.json', dict(spec=spec, sources=read(a.sources), source_checks=sources,
        identities=identities, panels=panels, message2_sha256=sha(a.message2/'binding.json'),
        registration_sha256=sha(RESOURCE), technical_synthetic=a.standin))
    print('Screen S bound: 12 stages, 7 incoming checkpoints; code frozen')


def configure(p):
    p.add_argument('--message2', type=Path, required=True)
    p.add_argument('--spec', type=Path, required=True)
    p.add_argument('--sources', type=Path, required=True)

if __name__ == '__main__':
    raise SystemExit(entry(job, configure))
