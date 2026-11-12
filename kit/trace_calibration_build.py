#!/usr/bin/env python3
"""Offline manager-side selection, manual authoring validation and pair sealing."""
from pathlib import Path
import sys
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
from collections import Counter
import json
import re
from kit.trace_common import canonical, digest, load, need, seal, sha

SOURCES = {
    'Self-Instruct': dict(repository='yizhongw/self-instruct', revision='0b26ccaa',
                         file='seed_tasks.jsonl', count=175),
    'LIMA': dict(repository='BeyonderXX/TRACE',
                 revision='462e39f616134f4f819efeb3baea8638c03c7db4',
                 file='inference/HHH/helpful/LIMA/eval.json', count=300),
    'CoNa': dict(repository='BeyonderXX/TRACE',
                 revision='462e39f616134f4f819efeb3baea8638c03c7db4',
                 file='inference/HHH/harmless/CoNa/eval.json', count=178),
}
QUALITY = ('clearly_better', 'near_tie', 'clearly_worse')
SEED = 'trace-calibration-20261112-v1'
FIELDS = {'id', 'panel', 'prompt', 'a', 'b', 'checkpoint_independent'}


def text(value):
    need(isinstance(value, str) and bool(value.strip()), 'nonempty text required')
    return value


def read_source(panel, path):
    """One prompt per source task; Self-Instruct uses its first human instance."""
    path = Path(path)
    if panel != 'Self-Instruct':
        rows = load(path)
        need(isinstance(rows, list), 'TRACE eval must be a JSON array')
    else:
        with path.open(encoding='utf-8') as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
    need(len(rows) == SOURCES[panel]['count'], 'pinned source population differs')
    result = []
    for index, row in enumerate(rows):
        if panel == 'Self-Instruct':
            instruction = text(row['instruction'])
            instance = row['instances'][0]
            source_input = instance['input']
            need(isinstance(source_input, str), 'Self-Instruct input must be text')
            prompt = instruction + ('\n\n'+source_input if source_input.strip() else '')
            answer = text(instance['output'])
        else:
            need(row['answer'] == '', 'TRACE eval must be unanswered')
            prompt, answer = text(row['prompt']), None
        result.append(dict(source_index=index, prompt=prompt, existing_answer=answer))
    need(len({r['prompt'] for r in result}) == len(result), 'duplicate source prompts')
    return result


def plan(paths, synthetic=False, approvals=None):
    provenance, selected = {}, []
    for panel, pin in SOURCES.items():
        rows = read_source(panel, paths[panel])
        provenance[panel] = dict(pin, sha256=sha(paths[panel]))
        if not synthetic:
            receipt = (approvals or {}).get(panel, {})
            need(receipt.get('approved') is True and
                 all(receipt.get(k) == v for k, v in provenance[panel].items()),
                 'approved pinned-file provenance required')
            need(provenance[panel]['sha256'] == SOURCE_SHA256[panel],
                 'approved pinned-file provenance bytes differ')
        ranked = sorted(rows, key=lambda r: digest([SEED, panel, r['source_index'], r['prompt']]))[:30]
        for index, row in enumerate(ranked):
            pair_id = digest([SEED, panel, row['source_index'], row['prompt']])[:32]
            selected.append(dict(id=pair_id, panel=panel, **row,
                                 quality=QUALITY[index % 3]))
    return dict(schema='trace-calibration-authoring.v1', technical_synthetic=synthetic,
                seed=SEED, sources=provenance, selected=selected)


def authoring_template(selection):
    return [dict(id=r['id'], a=r['existing_answer'] or '', b='',
                 quality=r['quality'], checkpoint_independent=True,
                 identifiers=[], reviewed=False) for r in selection['selected']]


def clean_answer(answer, identifiers):
    answer = text(answer)
    # Authors list exact attribution strings, never broad words such as "I".
    for identifier in identifiers:
        text(identifier)
        answer = re.sub(re.escape(identifier), '', answer, flags=re.IGNORECASE)
    need(not re.search(r'(?im)^\s*(author|model|arm)\s*:', answer),
         'unstripped author/model/arm attribution')
    return text(answer.strip())


def build(selection, authored):
    need(selection.get('seed') == SEED, 'selection algorithm differs')
    selected = selection['selected']
    need(type(selection.get('technical_synthetic')) is bool, 'explicit source mode required')
    need(len(selected) == len(authored) == 90, 'ninety authored pairs required')
    lookup = {r['id']: r for r in authored}
    need(len(lookup) == 90 and set(lookup) == {r['id'] for r in selected},
         'authored ID inventory differs')
    need(Counter(r['panel'] for r in selected) == Counter(dict.fromkeys(SOURCES, 30)),
         'calibration panels differ')
    pairs, key = [], []
    for row in selected:
        need(row['id'] == digest([SEED, row['panel'], row['source_index'], row['prompt']])[:32],
             'selection ID/prompt changed')
        draft = lookup[row['id']]
        need(set(draft) == {'id', 'a', 'b', 'quality', 'checkpoint_independent',
                            'identifiers', 'reviewed'}, 'authoring schema differs')
        need(draft['reviewed'] is True and draft['checkpoint_independent'] is True,
             'manual review/checkpoint independence incomplete')
        need(draft['quality'] == row['quality'] and row['quality'] in QUALITY,
             'planned quality mix changed')
        need(isinstance(draft['identifiers'], list), 'identifier list required')
        if row['panel'] == 'Self-Instruct':
            need(draft['a'] == row['existing_answer'], 'human instance output changed')
        a, b = [clean_answer(draft[k], draft['identifiers']) for k in ('a', 'b')]
        need(a != b, 'identical answers do not form a calibration pair')
        reverse = int(digest([SEED, row['id'], 'order'])[:8], 16) % 2 == 1
        pairs.append(dict(id=row['id'], panel=row['panel'], prompt=row['prompt'],
                          a=b if reverse else a, b=a if reverse else b,
                          checkpoint_independent=True))
        key.append(dict(id=row['id'], quality=row['quality'], reverse=reverse,
                        source_index=row['source_index']))
    for panel in SOURCES:
        need(Counter(r['quality'] for r in selected if r['panel'] == panel) ==
             Counter(dict.fromkeys(QUALITY, 10)), 'quality mix differs')
    return pairs, key


def publish(selection, authored, public, local):
    public, local = Path(public).resolve(), Path(local).resolve()
    need(public != local and public not in local.parents and local not in public.parents,
         'LOCAL key and public outputs need separate trees')
    pairs, key = build(selection, authored)
    public.mkdir(parents=True, exist_ok=True)
    local.mkdir(parents=True, exist_ok=True)
    payload = b''.join(canonical(r)+b'\n' for r in pairs)
    path = public/'calibration.jsonl'
    if path.exists():
        need(path.read_bytes() == payload, 'sealed calibration changed')
    else:
        with path.open('xb') as stream:
            stream.write(payload)
    seal(public/'pairs.json', pairs)
    seal(public/'calibration-digest.json', dict(sha256=sha(path), rows=90,
         technical_synthetic=selection['technical_synthetic']))
    seal(local/'LOCAL-key.json', dict(selection=selection, authored=authored, mapping=key,
                                     calibration_sha256=sha(path)))
    return sha(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    select = commands.add_parser('select')
    for flag in ('self-instruct', 'lima', 'cona'):
        select.add_argument('--'+flag, type=Path, required=True)
    select.add_argument('--out', type=Path, required=True)
    select.add_argument('--synthetic', action='store_true')
    select.add_argument('--approvals', type=Path)
    finish = commands.add_parser('seal')
    for flag in ('selection', 'authored', 'public', 'local'):
        finish.add_argument('--'+flag, type=Path, required=True)
    node = commands.add_parser('node',help='assemble sealed-90 from bound node inputs')
    node.add_argument('--config',type=Path,required=True)
    node.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    if args.action == 'select':
        selection = plan(dict(zip(SOURCES, (args.self_instruct, args.lima, args.cona))),
                         args.synthetic, load(args.approvals) if args.approvals else None)
        seal(args.out/'selection.json', selection)
        seal(args.out/'authored-template.json', authoring_template(selection))
    elif args.action=='node':
        print(assemble_node_config(args.config,args.out))
    else:
        selection = load(args.selection)
        # Re-selection requires the licensed bytes; provenance hashes stay LOCAL.
        print(publish(selection, load(args.authored), args.public, args.local))



# Public immutable source bindings; no prompts, relations or rater answers.
SOURCE_SHA256 = {
    'Self-Instruct': '7779004fa198fdf27cf70a159363879d8a26c53329e11b436af17b3941875f48',
    'LIMA': '32d358c81ea71b87aed46b17fa43e1fdfe8c3946c1e345873c22ea900650e6aa',
    'CoNa': 'a0d4c70cb26b990d7bd6a0e7c42cab8e5bdfe530e14e7941fcbdd51c4e3d4ec6',
}


def public_selection(sources):
    import hashlib
    rows = []
    for panel in SOURCES:
        need(len(sources[panel]) == SOURCES[panel]['count'], 'source count differs')
        ranked = sorted(sources[panel], key=lambda r: digest(
            [SEED, panel, r['source_index'], r['prompt']]))[:30]
        for row in ranked:
            rows.append(dict(id=digest([SEED, panel, row['source_index'], row['prompt']])[:32],
                panel=panel, source_index=row['source_index'],
                prompt_sha256=hashlib.sha256(row['prompt'].encode()).hexdigest()))
    return dict(schema='trace-calibration-selection.v1', seed=SEED,
                selected=sorted(rows, key=lambda r: r['id']))


def assemble(selection, authored, sources):
    """Public assembly only. Planned relations are never needed on the node."""
    need(selection == public_selection(sources), 'selection/source/prompt hash differs')
    selected = selection['selected']
    need(set(authored) == {r['id'] for r in selected}, 'authored ID inventory differs')
    pairs = []
    for row in selected:
        source = sources[row['panel']][row['source_index']]
        draft = authored[row['id']]
        need(set(draft) == ({'b'} if row['panel']=='Self-Instruct' else {'a','b'}),
             'authored answer schema differs')
        a = source['existing_answer'] if row['panel']=='Self-Instruct' else draft['a']
        a, b = text(a), text(draft['b'])
        need(a != b, 'identical calibration answers')
        reverse = int(digest([SEED,row['id'],'order'])[:8],16)%2 == 1
        pairs.append(dict(id=row['id'],panel=row['panel'],prompt=source['prompt'],
                          a=b if reverse else a,b=a if reverse else b,
                          checkpoint_independent=True))
    return pairs


def validate_relations(selection, authored_path, selection_path, key):
    """Manager-only planned-relation validation, separate from public assembly."""
    need(key['seed']==SEED and key['source_sha256']==SOURCE_SHA256, 'local source key differs')
    need(sha(authored_path)==key['authored_sha256'] and
         sha(selection_path)==key['selection_sha256'], 'local artifact key differs')
    rows = selection['selected']
    need(set(key['pairs'])=={r['id'] for r in rows}, 'local relation inventory differs')
    for panel in SOURCES:
        need(Counter(key['pairs'][r['id']]['quality'] for r in rows if r['panel']==panel)
             == Counter(dict.fromkeys(QUALITY,10)), 'planned quality mix differs')
    for row in rows:
        k=key['pairs'][row['id']]
        need(k['panel']==row['panel'] and k['source_index']==row['source_index'] and
             k['reverse']==(int(digest([SEED,row['id'],'order'])[:8],16)%2==1),
             'local relation mapping differs')


def load_authored(path, sidecar):
    need(Path(sidecar).read_text() == sha(path)+'  authored-v1.json\n',
         'authored sidecar differs')
    return load(path)


def seal_pairs(pairs, destination):
    destination=Path(destination)
    payload=b''.join(canonical(r)+b'\n' for r in pairs)
    destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists():
        need(destination.read_bytes()==payload, 'sealed calibration changed')
    else:
        with destination.open('xb') as stream:
            stream.write(payload)
    return sha(destination)


def node_assemble(spec, panels, destination):
    from kit.trace_common import bound_file, bound_json
    seed_path=bound_file(spec['self_instruct'])
    need(sha(seed_path)==SOURCE_SHA256['Self-Instruct'], 'Self-Instruct seed binding differs')
    sources={'Self-Instruct':read_source('Self-Instruct',seed_path)}
    need(digest([r['prompt'] for r in panels if r['panel']=='Self-Instruct']) ==
         PANEL_PROMPTS['Self-Instruct'], 'bound TRACE open panel differs')
    for panel in ('LIMA','CoNa'):
        prompts=[r['prompt'] for r in panels if r['panel']==panel]
        need(digest(prompts)==PANEL_PROMPTS[panel], 'bound TRACE open panel differs')
        sources[panel]=[dict(source_index=i,prompt=p,existing_answer=None)
                        for i,p in enumerate(prompts)]
    selection=bound_json(spec['selection'])
    authored_path=bound_file(spec['authored'])
    sidecar=bound_file(spec['sidecar'])
    pairs=assemble(selection,load_authored(authored_path,sidecar),sources)
    seal_pairs(pairs,destination)
    return pairs

PANEL_PROMPTS = {'Self-Instruct': 'fc57f55affb6b50360a3eb02260a7cc42e44cebe1ce37741431813731b5390c8', 'LIMA': '65db54536391e3b1365bbd70990945a59da60522872aa8051a5af683a89574b2', 'CoNa': 'ae033b4c14b3ad9557a440eec9afbda6d3522a28adce418a06c4923c58af1bac'}


EXPECTED_SHA256 = 'b4713433a390203a266b7b9c4e1b6a34fc5a7df5f2c0ea12086886426c054de5'


def assemble_node_config(config, out):
    from kit.trace_common import bound_json
    cfg=load(config)
    out=Path(out)
    need(cfg['calibration']['schema']=='trace-calibration-assembly.v1', 'assembly schema differs')
    pairs=node_assemble(cfg['calibration'],bound_json(cfg['panels'])['open'],
                        out/'sealed/calibration.jsonl')
    checksum=sha(out/'sealed/calibration.jsonl')
    need(checksum==EXPECTED_SHA256, 'sealed-90 calibration digest differs')
    seal(out/'sealed/pairs.json',pairs)
    seal(out/'receipt.json',dict(rows=90,sha256=checksum,paid_calls=0,gpu_hours=0))
    return checksum


if __name__ == '__main__':
    main()
