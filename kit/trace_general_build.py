#!/usr/bin/env python3
"""Offline, hash-bound raw datasets to OpenCompass 0.1.5 TRACE general rows."""
from pathlib import Path
import sys
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import ast
import csv
import json

from kit.trace_common import bound_file, digest, load, need, seal
from kit.trace_general_metrics import VENDOR, PIN, PANELS, gsm8k_gold, verify_vendor

CONFIGS = {
    'MMLU': 'mmlu/mmlu_ppl_ac766d.py', 'PIQA': 'piqa/piqa_ppl_1cf9f0.py',
    'BoolQ': 'SuperGLUE_BoolQ/SuperGLUE_BoolQ_ppl_314b96.py',
    'GSM8K': 'gsm8k/gsm8k_gen_1d7fe4.py', 'BBH': 'bbh/bbh_gen_5b92b0.py',
    'TyDiQA': 'tydiqa/tydiqa_gen_978d2a.py'}
SCHEMA = 'trace-general-opencompass-0.1.5.v1'


def _data(node, env):
    """Small data-expression reader, never import/exec dataset configs/mmengine."""
    if isinstance(node, ast.Constant): return node.value
    if isinstance(node, ast.Name):
        need(node.id in env, 'unknown config symbol: '+node.id)
        return env[node.id]
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_data(x, env) for x in node.elts]
    if isinstance(node, ast.Dict):
        return {_data(k, env): _data(v, env) for k, v in zip(node.keys, node.values)}
    if isinstance(node, ast.Subscript):
        return _data(node.value, env)[_data(node.slice, env)]
    if isinstance(node, ast.JoinedStr):
        return ''.join(str(_data(x.value, env)) if isinstance(x, ast.FormattedValue)
                       else x.value for x in node.values)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _data(node.left, env) + _data(node.right, env)
    if isinstance(node, (ast.DictComp, ast.ListComp)):
        need(len(node.generators) == 1, 'unsupported config comprehension')
        gen = node.generators[0]
        need(isinstance(gen.target, ast.Name) and not gen.ifs and not gen.is_async,
             'unsupported config iterator')
        entries = []
        for value in _data(gen.iter, env):
            scope = dict(env, **{gen.target.id: value})
            entries.append((_data(node.key, scope), _data(node.value, scope))
                           if isinstance(node, ast.DictComp) else _data(node.elt, scope))
        return dict(entries) if isinstance(node, ast.DictComp) else entries
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id == 'dict':
            need(not node.args and all(k.arg for k in node.keywords), 'unsupported dict call')
            return {k.arg: _data(k.value, env) for k in node.keywords}
        if isinstance(node.func, ast.Attribute) and node.func.attr == 'replace':
            obj = _data(node.func.value, env)
            need(isinstance(obj, str) and not node.keywords, 'unsupported config replace')
            return obj.replace(*[_data(x, env) for x in node.args])
    raise ValueError('unsupported config data expression: '+type(node).__name__)


def config_value(panel, name, **env):
    path = VENDOR/'configs/datasets'/CONFIGS[panel]
    tree = ast.parse(path.read_text(encoding='utf-8'))
    # Types are inert identifiers, never imported or called.
    symbols = {alias.asname or alias.name: alias.name for statement in tree.body
               if isinstance(statement, ast.ImportFrom) for alias in statement.names}
    symbols.update(env)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name
                                               for t in node.targets):
            return _data(node.value, symbols)
    raise ValueError('missing pinned config assignment: '+name)


def subsets(panel):
    if panel == 'MMLU': return config_value(panel, 'mmlu_all_sets')
    if panel == 'BBH':
        return config_value(panel, 'bbh_multiple_choice_sets') + config_value(panel, 'bbh_free_form_sets')
    if panel == 'TyDiQA': return config_value(panel, 'langs')
    return [panel]


def expected_files(panel):
    if panel == 'MMLU':
        return [f'{split}/{subject}_{split}.csv' for subject in subsets(panel) for split in ('dev', 'test')]
    if panel == 'BBH': return [f'{task}.json' for task in subsets(panel)]
    if panel == 'TyDiQA': return [f'{lang}/validation.jsonl' for lang in subsets(panel)]
    return [{'GSM8K': 'test.jsonl', 'PIQA': 'validation.jsonl', 'BoolQ': 'val.jsonl'}[panel]]


def safe_format(text, entry):
    # OpenCompass utils/prompt.py:14-28 uses successive literal replacements.
    for key, value in entry.items():
        text = text.replace('{'+key+'}', str(value))
    return text


def render(template, entry, *, ice='', output_field=None):
    entry = dict(entry)
    if output_field:
        entry[output_field] = ''
    def text(value):
        return safe_format(value.replace('</E>', ice), entry)
    if isinstance(template, str):
        return dict(render_mode='raw', prompt=text(template))
    need(set(template) == {'round'}, 'unsupported pinned meta template')
    return dict(render_mode='chat', messages=[dict(
        role={'HUMAN': 'user', 'BOT': 'assistant'}[turn['role']], content=text(turn['prompt']))
        for turn in template['round']])


def build_row(panel, subset, index, raw, dev=()):
    row = dict(id=f'trace-general/{panel}/{subset}/{index}', panel=panel, subset=subset)
    if panel == 'MMLU':
        hint = f'The following are multiple choice questions (with answers) about  {subset.replace("_", " ")}.\n\n'
        cfg = config_value(panel, 'mmlu_infer_cfg', _hint=hint)
        need(len(dev) >= 5, 'MMLU requires five fixed dev examples')
        shots = [safe_format(cfg['ice_template']['template'][r['target']], r) for r in dev[:5]]
        ice = '\n'.join(shots) + '\n'
        templates = cfg['prompt_template']['template']
        row.update(answer=raw['target'], labels=list(templates),
                   label_prompts=[render(t, raw, ice=ice) for t in templates.values()],
                   shot_ids=[0, 1, 2, 3, 4],
                   shot_prompts=[safe_format(shot, raw) for shot in shots], ice_prefix=hint)
    elif panel in ('PIQA', 'BoolQ'):
        cfg = config_value(panel, 'piqa_infer_cfg' if panel == 'PIQA' else 'BoolQ_infer_cfg')
        templates = cfg['prompt_template']['template']
        if panel == 'BoolQ':
            need(type(raw.get('label')) is str and raw['label'] in ('true', 'false'),
                 'BoolQ label must be exactly the string "true" or "false"')
        gold = raw['label'] if panel == 'PIQA' else int(raw['label'] == 'true')
        row.update(answer=gold, labels=list(templates),
                   label_prompts=[render(t, raw) for t in templates.values()], shot_ids=[])
    else:
        if panel == 'GSM8K':
            cfg = config_value(panel, 'gsm8k_infer_cfg')
            # Builder uses the byte-identical upstream dataset postprocessor.
            row['answer'] = gsm8k_gold(raw['answer'])
            output = 'answer'
        elif panel == 'BBH':
            hint = (VENDOR/'configs/datasets/bbh/lib_prompt'/f'{subset}.txt').read_text(encoding='utf-8')
            cfg = config_value(panel, 'bbh_infer_cfg', _hint=hint)
            row['answer'], output = raw['target'], 'target'
        else:
            hint = config_value(panel, 'prefixs_prompt')[subset]
            cfg = config_value(panel, 'tydiqa_infer_cfg', _hint=hint)
            gold = raw['answers']['text']
            need(type(gold) is list and gold and all(isinstance(x, str) for x in gold),
                 'TyDiQA requires list answers.text')
            row['answer'], output = gold, 'answer'
        row.update(render(cfg['prompt_template']['template'], raw, output_field=output))
    if 'label_prompts' in row:
        row['prompt'] = raw.get('input', raw.get('goal', raw.get('question', '')))
    else:
        row.setdefault('prompt', row['messages'][-1]['content'] if 'messages' in row else '')
    return row


def _read(path):
    if path.suffix == '.csv':
        with path.open(encoding='utf-8', newline='') as stream:
            rows = list(csv.reader(stream))
        need(all(len(r) == 6 for r in rows), 'MMLU CSV must have six columns, no header')
        return [dict(zip(('input', 'A', 'B', 'C', 'D', 'target'), r)) for r in rows]
    if path.suffix == '.json': return load(path)['examples']
    with path.open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def build(manifest):
    provenance = verify_vendor()
    need(manifest.get('schema') == 'trace-general-inputs.v1', 'general dataset manifest schema differs')
    need(set(manifest['datasets']) == set(PANELS), 'general dataset inventory differs')
    result = {}
    for panel in PANELS:
        source = manifest['datasets'][panel]
        need(isinstance(source.get('revision'), str) and source['revision'].strip(),
             'explicit dataset revision required')
        need(set(source['files']) == set(expected_files(panel)), 'dataset file inventory differs: '+panel)
        files, rows, counts = {}, [], {}
        # Hash every staged file before reading any of this panel's data.
        paths = {key: bound_file(ref) for key, ref in source['files'].items()}
        data = {key: _read(path) for key, path in paths.items()}
        for key, raw in data.items():
            files[key] = dict(sha256=source['files'][key]['sha256'], n=len(raw))
        for subset in subsets(panel):
            dev = ()
            if panel == 'MMLU':
                dev = data[f'dev/{subset}_dev.csv']
                raw_rows = data[f'test/{subset}_test.csv']
            elif panel == 'BBH': raw_rows = data[f'{subset}.json']
            elif panel == 'TyDiQA': raw_rows = data[f'{subset}/validation.jsonl']
            else: raw_rows = data[expected_files(panel)[0]]
            need(raw_rows, 'empty general subset: '+subset)
            rows.extend(build_row(panel, subset, i, raw, dev) for i, raw in enumerate(raw_rows))
            counts[subset] = len(raw_rows)
        result[panel] = dict(schema=SCHEMA, commit=PIN, config=CONFIGS[panel],
            config_sha256=provenance['sha256']['configs/datasets/'+CONFIGS[panel]],
            mode='ppl' if panel in ('MMLU', 'PIQA', 'BoolQ') else 'generation',
            prompt_cap=2048, response_cap=50 if panel == 'TyDiQA' else 512,
            temperature=0, rows=rows, subsets=counts,
            source=dict(revision=source['revision'], files=files),
            manifest_sha256=digest(source))
    validate(result)
    return result


def validate(panels):
    """Reject legacy/operator-invented general layouts before scoring."""
    from collections import Counter
    need(set(panels) == set(PANELS), 'TRACE general panel inventory differs')
    provenance = verify_vendor()
    all_ids = set()
    for panel, spec in panels.items():
        ppl = panel in ('MMLU', 'PIQA', 'BoolQ')
        need(spec['schema'] == SCHEMA and spec['commit'] == PIN and spec['config'] == CONFIGS[panel] and
             spec['config_sha256'] == provenance['sha256']['configs/datasets/'+CONFIGS[panel]],
             'general config pin differs')
        need(spec['mode'] == ('ppl' if ppl else 'generation') and spec['prompt_cap'] == 2048 and
             spec['response_cap'] == (50 if panel == 'TyDiQA' else 512) and spec['temperature'] == 0,
             'general inference settings differ')
        need(set(spec['subsets']) == set(subsets(panel)) and
             Counter(r['subset'] for r in spec['rows']) == Counter(spec['subsets']),
             'general subset counts differ')
        need(all(type(n) is int and n > 0 for n in spec['subsets'].values()), 'empty general subset')
        source = spec['source']
        need(isinstance(source['revision'], str) and source['revision'].strip() and
             set(source['files']) == set(expected_files(panel)), 'general dataset provenance differs')
        def hash_value(value):
            return isinstance(value, str) and len(value) == 64 and all(x in '0123456789abcdef' for x in value)
        need(hash_value(spec['manifest_sha256']) and
             all(hash_value(f['sha256']) and type(f['n']) is int and f['n'] > 0
                 for f in source['files'].values()), 'general dataset hashes/counts absent')
        for subset, count in spec['subsets'].items():
            key = (f'test/{subset}_test.csv' if panel == 'MMLU' else
                   f'{subset}.json' if panel == 'BBH' else
                   f'{subset}/validation.jsonl' if panel == 'TyDiQA' else expected_files(panel)[0])
            need(source['files'][key]['n'] == count, 'general raw/subset counts differ')
            if panel == 'MMLU':
                need(source['files'][f'dev/{subset}_dev.csv']['n'] >= 5, 'MMLU dev count differs')
        for row in spec['rows']:
            need(row['id'] not in all_ids and row['panel'] == panel, 'general row identity differs')
            all_ids.add(row['id'])
            if ppl:
                labels = list('ABCD') if panel == 'MMLU' else [0, 1]
                need(row['labels'] == labels and type(row['answer']) is (str if panel == 'MMLU' else int) and
                     row['answer'] in labels and
                     len(row['label_prompts']) == len(labels), 'general PPL label schema differs')
                need(row['shot_ids'] == ([0,1,2,3,4] if panel == 'MMLU' else []), 'general shots differ')
                if panel == 'MMLU':
                    need(len(row.get('shot_prompts', [])) == 5 and
                         all(isinstance(s, str) for s in row['shot_prompts']) and
                         isinstance(row.get('ice_prefix'), str),
                         'MMLU shot rendering metadata missing; rebuild staged general rows')
                    ice = row['ice_prefix'] + '\n'.join(row['shot_prompts']) + '\n'
                    need(all(p.get('prompt', '').startswith(ice) for p in row['label_prompts']),
                         'MMLU shot rendering differs')
                prompts = row['label_prompts']
            else:
                prompts = [row]
                if panel == 'TyDiQA':
                    need(type(row['answer']) is list and row['answer'] and
                         all(isinstance(x, str) for x in row['answer']), 'TyDiQA golds must be lists')
            for prompt in prompts:
                mode = 'chat' if panel in ('BoolQ', 'GSM8K', 'BBH') else 'raw'
                need(prompt['render_mode'] == mode, 'general rendering mode differs')
                if mode == 'raw': need(isinstance(prompt['prompt'], str), 'raw general prompt absent')
                else:
                    need(prompt['messages'] and all(m['role'] in ('user','assistant') and
                         isinstance(m['content'], str) for m in prompt['messages']), 'general chat rounds absent')
                    need(prompt['messages'][-1]['role'] == ('assistant' if ppl else 'user'),
                         'general final round differs')
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    seal(args.out, build(load(args.manifest)))


if __name__ == '__main__': main()
