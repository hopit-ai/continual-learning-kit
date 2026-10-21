#!/usr/bin/env python3
"""Offline, deterministic legal headroom preparation and matched-panel scoring."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT.parent / 'legal-raw'
LEX_REVISION = 'c23fdff1a6bf74e0e1a71cb86f1e781d37da888c'
ZIP_SHA256 = 'e03fc77bbf8b53e2976a250e81d8a294bc3d5e5fb014521e477dee9340d6287b'
STUDENT_REVISION = 'b968826d9c46dd6066d109eabc6255188de91218'
TEACHER_REVISION = '6a9e13bd6fc8f0983b9b99948120bc37f49c13e9'
TASKS = ('case_hold', 'contract_nli')
LABELS = {'case_hold': tuple('ABCDE'), 'contract_nli': ('entailment', 'contradiction', 'not_mentioned')}
INPUT_CAP, OUTPUT_CAP, CONTEXT = 10240, 2048, 12288
DEFAULT_TOKENIZER = ROOT / 'references/r100-tokenizer-cache/models--Qwen--Qwen3-8B/snapshots' / STUDENT_REVISION
PINNED_TOKENIZER_HASHES = {
    'tokenizer.json': 'aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4',
    'tokenizer_config.json': 'd5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101',
}
PREFIX = 'C-20261014|'
SCHEMA = 'legal-headroom.v1'


class PreparationError(ValueError):
    pass


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def digest(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def normalized(text):
    return ' '.join(unicodedata.normalize('NFKC', text).casefold().split())


def prompt_for(task, content):
    if task == 'case_hold':
        choices = '\n'.join(f'{label}. {ending}' for label, ending in zip('ABCDE', content['endings']))
        return ('Select the holding that best completes the citation marked <HOLDING> in the legal context.\n\n'
                f"Context:\n{content['context']}\n\nHoldings:\n{choices}\n\nFinish with Answer: <A-E>")
    return ('Determine whether the hypothesis is entailed by the agreement, contradicted by it, or not mentioned. '
            'Use only the full agreement below.\n\n'
            f"Agreement:\n{content['document']}\n\nHypothesis:\n{content['hypothesis']}\n\n"
            'Finish with Answer: entailment|contradiction|not_mentioned (choose exactly one label).')


def make_item(task, split, source_id, content, group):
    prompt = prompt_for(task, content)
    return {'id': f'{task}:{split}:{source_id}', 'source_split': split, 'source_id': str(source_id),
            'group': group, 'content': content, 'gold': content['label'],
            'rank_sha256': digest(PREFIX + canonical(content)),
            'prompt': prompt, 'prompt_sha256': digest(prompt)}


def rank(item):
    return item['rank_sha256'], canonical(item['content']), item['id']


def remove_cross_split_groups(splits, group_fields=('group',)):
    """Remove every member of a group shared across source splits, without favouring a split."""
    owners = {field: defaultdict(set) for field in group_fields}
    for split, items in splits.items():
        for item in items:
            for field in group_fields:
                owners[field][item[field]].add(split)
    clean, removed = {}, {}
    for split, items in splits.items():
        clean[split] = [i for i in items if all(len(owners[f][i[f]]) == 1 for f in group_fields)]
        removed[split] = len(items) - len(clean[split])
    return clean, removed


def select(items, quotas, max_per_group):
    """Single global hash order; take eligible rows while each label and document quota has room."""
    counts, groups, chosen = Counter(), Counter(), []
    for item in sorted(items, key=rank):
        label, group = item['gold'], item['group']
        if counts[label] < quotas[label] and groups[group] < max_per_group:
            chosen.append(item)
            counts[label] += 1
            groups[group] += 1
    if any(counts[k] != n for k, n in quotas.items()):
        raise PreparationError(f'cannot meet quotas {quotas}; selected {dict(counts)} with group cap {max_per_group}')
    return chosen


def render(tokenizer, prompt):
    return tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)


def count_inputs(tokenizer, items):
    # Count the actual chat-wrapped input, including the thinking-off generation prefix.
    for offset in range(0, len(items), 64):
        batch = items[offset:offset + 64]
        encoded = tokenizer([render(tokenizer, i['prompt']) for i in batch], add_special_tokens=False,
                            truncation=False)['input_ids']
        for item, ids in zip(batch, encoded):
            item['input_tokens'] = len(ids)


def freeze_casehold(source, tokenizer):
    import pyarrow.parquet as pq
    splits, sources = {}, {}
    for split in ('train', 'validation', 'test'):
        path = Path(source) / f'{split}-00000-of-00001.parquet'
        sources[path.name] = file_hash(path)
        rows = pq.read_table(path).to_pylist()
        items = []
        for index, row in enumerate(rows):
            if len(row['endings']) != 5 or row['label'] not in range(5):
                raise PreparationError('invalid CaseHOLD source row')
            content = {'context': row['context'], 'endings': row['endings'], 'label': 'ABCDE'[row['label']]}
            items.append(make_item('case_hold', split, index, content, digest(normalized(row['context']))))
        splits[split] = items
    clean, removed = remove_cross_split_groups(splits)
    result = {}
    for name, split, each in [('panel', 'validation', 20), ('train', 'train', 40), ('test', 'test', 20)]:
        result[name] = select(clean[split], dict.fromkeys('ABCDE', each), 1)
        count_inputs(tokenizer, result[name])
        if any(i['input_tokens'] > INPUT_CAP for i in result[name]):
            raise PreparationError('CaseHOLD selected input exceeds context budget; no truncation permitted')
    return result, {'dataset': 'coastalcph/lex_glue', 'revision': LEX_REVISION, 'licence': 'CC-BY-4.0',
                    'source_files_sha256': sources, 'source_counts': {k: len(v) for k, v in splits.items()},
                    'cross_split_normalized_context_excluded': removed,
                    'length_excluded': dict.fromkeys(splits, 0),
                    'deduplication': 'NFKC, casefold, whitespace normalization, exact match; all cross-split members excluded; one context per selected split'}


def freeze_contract(source, tokenizer):
    splits, sources, hypotheses = {}, {}, None
    choices = {'Entailment': 'entailment', 'Contradiction': 'contradiction', 'NotMentioned': 'not_mentioned'}
    for split in ('train', 'dev', 'test'):
        path = Path(source) / f'{split}.json'
        sources[path.name] = file_hash(path)
        data = json.loads(path.read_text())
        if len(data['labels']) != 17:
            raise PreparationError('expected the original 17 ContractNLI hypotheses')
        if hypotheses is not None and hypotheses != data['labels']:
            raise PreparationError('hypotheses differ across source splits')
        hypotheses = data['labels']
        items = []
        for doc in data['documents']:
            annotations = doc['annotation_sets']
            if len(annotations) != 1 or set(annotations[0]['annotations']) != set(hypotheses):
                raise PreparationError('unexpected ContractNLI annotation structure')
            for hid, hypothesis in hypotheses.items():
                label = choices[annotations[0]['annotations'][hid]['choice']]
                content = {'document': doc['text'], 'hypothesis': hypothesis['hypothesis'], 'label': label}
                item = make_item('contract_nli', split, f"{doc['id']}:{hid}", content, digest(normalized(doc['text'])))
                item.update(document_id=str(doc['id']), hypothesis_id=hid, source_url=doc.get('url'),
                            source_file_name=doc['file_name'])
                items.append(item)
        splits[split] = items
    clean, removed = remove_cross_split_groups(splits, ('group', 'document_id'))
    excluded = {}
    for split, items in clean.items():
        count_inputs(tokenizer, items)
        eligible = [i for i in items if i['input_tokens'] <= INPUT_CAP]
        excluded[split] = len(items) - len(eligible)
        clean[split] = eligible
    labels = LABELS['contract_nli']
    result = {name: select(clean[split], dict(zip(labels, quotas)), 3)
              for name, split, quotas in [('panel', 'dev', (34, 33, 33)), ('train', 'train', (67, 67, 66)),
                                          ('test', 'test', (34, 33, 33))]}
    return result, {'dataset': 'ContractNLI (Hitachi America, Ltd.)', 'licence': 'CC-BY-4.0 per TERMS',
                    'archive_sha256': ZIP_SHA256, 'source_files_sha256': sources,
                    'source_counts': {k: len(v) for k, v in splits.items()},
                    'cross_split_document_excluded': removed, 'length_excluded': excluded,
                    'hypotheses': hypotheses, 'max_pairs_per_document': 3,
                    'deduplication': 'document ID and NFKC/casefold/whitespace-normalized document content; all cross-split members excluded'}


def jsonl_text(rows):
    return ''.join(canonical(row) + '\n' for row in rows)


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().split('\n') if line.strip()]


def prepare(args):
    from transformers import AutoTokenizer
    token_path = Path(args.tokenizer).resolve()
    if not token_path.is_dir():
        raise PreparationError(f'local pinned Qwen3 tokenizer missing: {token_path}; pass --tokenizer (offline only)')
    tokenizer_hashes = {p.name: file_hash(p) for p in sorted(token_path.iterdir())
                        if p.is_file() and (p.name.startswith('tokenizer') or p.suffix in ('.jinja', '.txt'))}
    if any(tokenizer_hashes.get(name) != sha for name, sha in PINNED_TOKENIZER_HASHES.items()):
        raise PreparationError('tokenizer bytes differ from pinned Qwen3-8B revision')
    tokenizer = AutoTokenizer.from_pretrained(str(token_path), local_files_only=True)
    archive = Path(args.contract_source).parent / 'contract-nli.zip'
    if file_hash(archive) != ZIP_SHA256:
        raise PreparationError('ContractNLI archive hash differs from approved source')
    # Compute both tasks before writing anything: an unmet quota exits 2 without a partial freeze.
    built = {'case_hold': freeze_casehold(args.casehold_source, tokenizer),
             'contract_nli': freeze_contract(args.contract_source, tokenizer)}
    files = {}
    for task, (parts, provenance) in built.items():
        entries = {}
        for part, items in parts.items():
            name = f'{part}.jsonl'
            payload = jsonl_text(items)
            files[f'{task}/{name}'] = payload
            entries[part] = {'file': name, 'sha256': digest(payload), 'items': len(items),
                             'labels': dict(sorted(Counter(i['gold'] for i in items).items())),
                             'documents_or_contexts': len({i['group'] for i in items}),
                             'max_input_tokens': max(i['input_tokens'] for i in items),
                             'membership': [{'id': i['id'], 'rank_sha256': i['rank_sha256'],
                                             'prompt_sha256': i['prompt_sha256']} for i in items]}
        manifest = {'schema': SCHEMA, 'task': task, 'source': provenance,
                    'ranking': 'sha256(UTF-8("C-20261014|" + canonical JSON content)); global ascending order; skip full label/group quotas',
                    'canonical_json': 'ensure_ascii=False, sort_keys=True, separators=(comma, colon); original content preserved; includes gold label',
                    'tokenizer': {'model': 'Qwen/Qwen3-8B', 'revision': STUDENT_REVISION, 'files_sha256': tokenizer_hashes},
                    'input_cap': INPUT_CAP, 'context': CONTEXT, 'output_cap': OUTPUT_CAP,
                    'token_counting': 'full user chat template, add_generation_prompt=True, enable_thinking=False; no truncation',
                    'splits': entries}
        files[f'{task}/manifest.json'] = json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + '\n'
    out = Path(args.out)
    if out.exists():
        existing = {str(p.relative_to(out)): p.read_text() for p in out.rglob('*') if p.is_file()}
        if existing != files:
            raise PreparationError('refusing to change existing freeze; choose a fresh --out')
    else:
        out.mkdir(parents=True)
        for name, payload in files.items():
            target = out / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(payload, encoding='utf-8')
    for task, (_, provenance) in built.items():
        print(task, canonical({'splits': {k: {'items': len(v), 'labels': dict(Counter(i['gold'] for i in v))}
                                         for k, v in built[task][0].items()},
                               'length_excluded': provenance['length_excluded']}))
    return 0


def load_panel(data, task, split='panel'):
    folder = Path(data) / task
    manifest = json.loads((folder / 'manifest.json').read_text())
    if manifest['task'] != task or manifest['schema'] != SCHEMA:
        raise ValueError('unexpected freeze manifest')
    for entry in manifest['splits'].values():
        if file_hash(folder / entry['file']) != entry['sha256']:
            raise ValueError(f'{task}: frozen file hash mismatch')
    entry = manifest['splits'][split]
    panel = read_jsonl(folder / entry['file'])
    if split != 'train' and (len(panel) != 100 or len({i['id'] for i in panel}) != 100):
        raise ValueError('expected 100 distinct frozen %s items' % split)
    for item in panel:
        if item['prompt'] != prompt_for(task, item['content']) or digest(item['prompt']) != item['prompt_sha256']:
            raise ValueError('prompt differs from frozen content')
    return panel, entry['sha256']


def parse_answer(task, response):
    # The final Answer: line controls, even if it is malformed. Never fall back to an earlier valid answer.
    if not isinstance(response, str):
        return None
    candidates = [line.strip() for line in response.splitlines() if re.match(r'^\s*Answer\b', line, re.IGNORECASE)]
    if not candidates:
        return None
    match = re.fullmatch(r'Answer: (' + '|'.join(LABELS[task]) + r')', candidates[-1])
    return match.group(1) if match else None


def parse_answer_tolerant(task, response):
    """Format-tolerant reading, declared 17 Oct after seeing that the frozen 8B writes markdown answers
    ("**Answer: B**", or a "Final Answer:" heading followed by "**<E>**"). Only presentation is relaxed:
    emphasis, backticks, angle brackets and brackets around the label, and a label on the line right after an
    answer heading. The last answer line still controls; a missing or invalid label is still wrong."""
    if not isinstance(response, str):
        return None
    lines = [re.sub(r'[*`<>()\[\]]', '', line).strip() for line in response.splitlines()]
    labels = '|'.join(LABELS[task])
    last = None
    for i, line in enumerate(lines):
        if re.search(r'\banswer\b', line, re.IGNORECASE):
            m = re.search(r'answer\s*[:：-]?\s*(' + labels + r')\b', line, re.IGNORECASE)
            if m:
                last = m.group(1)
            else:
                nxt = next((l for l in lines[i + 1:] if l), '')
                m = re.fullmatch(r'#*\s*(' + labels + r')\.?', nxt, re.IGNORECASE)
                last = m.group(1) if m else None
    if last is None:
        # A reply that is only a valid label (e.g. "not_mentioned") is an unambiguous answer.
        whole = ' '.join(l for l in lines if l)
        m = re.fullmatch(r'(' + labels + r')\.?', whole, re.IGNORECASE)
        last = m.group(1) if m else None
    if last is None:
        return None
    canon = {l.lower(): l for l in LABELS[task]}
    return canon.get(last.lower())


def grade(task, panel, rows, panel_sha256=None):
    expected = {i['id'] for i in panel}
    by_id = {}
    for row in rows:
        if row['id'] not in expected or row['id'] in by_id:
            raise ValueError('unknown or duplicate response ID')
        if panel_sha256 is not None and row.get('panel_sha256') != panel_sha256:
            raise ValueError('response is not bound to this frozen panel')
        by_id[row['id']] = row
    correct, per_label, verdicts = 0, {}, {}
    for label in LABELS[task]:
        per_label[label] = {'correct': 0, 'total': 0}
    for item in panel:
        prediction = parse_answer(task, by_id.get(item['id'], {}).get('response', ''))
        ok = prediction == item['gold']
        verdicts[item['id']] = {'gold': item['gold'], 'prediction': prediction, 'correct': ok}
        correct += ok
        per_label[item['gold']]['total'] += 1
        per_label[item['gold']]['correct'] += ok
    token_totals = {kind: (sum(r[kind] for r in rows) if len(rows) == len(panel) and
                           all(isinstance(r.get(kind), int) and r[kind] >= 0 for r in rows) else None)
                    for kind in ('input_tokens', 'output_tokens')}
    all_tokens = sum(token_totals.values()) if None not in token_totals.values() else None
    tolerant = sum(parse_answer_tolerant(task, by_id.get(i['id'], {}).get('response', '')) == i['gold'] for i in panel)
    return {'correct': correct, 'total': len(panel), 'accuracy': correct / len(panel),
            'tolerant_correct': tolerant, 'tolerant_accuracy': tolerant / len(panel),
            'complete': len(rows) == len(panel), 'missing': len(panel) - len(rows),
            'malformed_or_missing': sum(v['prediction'] is None for v in verdicts.values()),
            'per_label': per_label, **token_totals,
            'output_tokens_per_correct': token_totals['output_tokens'] / correct if correct and token_totals['output_tokens'] is not None else None,
            'total_tokens_per_correct': all_tokens / correct if correct and all_tokens is not None else None,
            'per_item': verdicts}


def score(args):
    panel, sha = load_panel(args.data, args.task)
    paths = {role: Path(getattr(args, role)) for role in ('student', 'teacher', 'frontier')}
    scores = {role: grade(args.task, panel, read_jsonl(path), sha) for role, path in paths.items()}
    s, t, f = (scores[role] for role in ('student', 'teacher', 'frontier'))
    result = {'schema': SCHEMA, 'task': args.task, 'panel_sha256': sha, 'models': scores,
              'response_files_sha256': {role: file_hash(path) for role, path in paths.items()},
              'complete': all(m['complete'] for m in scores.values()),
              'teacher_correct_student_wrong': sum(t['per_item'][i['id']]['correct'] and not s['per_item'][i['id']]['correct'] for i in panel),
              'student_frontier_ratio': s['accuracy'] / f['accuracy'] if f['accuracy'] else None,
              'student_in_30_70_guide_band': .3 <= s['accuracy'] <= .7,
              'readout_rule': 'Report frozen student accuracy and student/frontier ratio; 30–70% is a guide, never a veto.'}
    lines = [f"# {args.task} legal headroom", '', f'Frozen panel SHA-256: `{sha}`', '',
             '| Model | Correct / 100 | Output tokens / correct | Total tokens / correct |',
             '|---|---:|---:|---:|']
    for role, m in scores.items():
        fmt = lambda x: 'unknown / undefined' if x is None else f'{x:.2f}'
        lines.append(f"| {role} | {m['correct']}/100 | {fmt(m['output_tokens_per_correct'])} | {fmt(m['total_tokens_per_correct'])} |")
    lines += ['', f"Teacher correct, student wrong: {result['teacher_correct_student_wrong']}/100.",
              f"Student/frontier accuracy ratio: {fmt(result['student_frontier_ratio'])}.",
              f"Complete response sets: {result['complete']}. Missing/malformed answers count wrong.", '',
              result['readout_rule'], '', '| Label / choice | Student | Teacher | Frontier |', '|---|---:|---:|---:|']
    for label in LABELS[args.task]:
        cells = [f"{scores[r]['per_label'][label]['correct']}/{scores[r]['per_label'][label]['total']}" for r in scores]
        lines.append('| ' + ' | '.join([label, *cells]) + ' |')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    (out / 'readout.json').write_text(json.dumps(result, indent=2) + '\n')
    (out / 'readout.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--casehold-source', default=str(RAW / 'hf/datasets--coastalcph--lex_glue/snapshots' / LEX_REVISION / 'case_hold'))
    p.add_argument('--contract-source', default=str(RAW / 'contract-nli'))
    p.add_argument('--tokenizer', default=str(DEFAULT_TOKENIZER))
    p.add_argument('--out', default=str(ROOT / 'data/legal'))
    p = sub.add_parser('score')
    p.add_argument('--task', choices=TASKS, required=True)
    p.add_argument('--data', default=str(ROOT / 'data/legal'))
    for role in ('student', 'teacher', 'frontier'):
        p.add_argument('--' + role, required=True)
    p.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    try:
        return prepare(args) if args.command == 'prepare' else score(args)
    except (ValueError, OSError, KeyError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
