"""3A CPU-only panel construction. No model responses are inputs."""
from __future__ import annotations
import argparse
import collections
import hashlib
import json
import random
import re
import shutil
from decimal import Decimal
from pathlib import Path

from kit.v4_pool import CHEM_DIR, REMOVED_FAMILIES, TAXONOMY, freeze_chemistry, read_json, sha
from kit.v4_contract import INITIAL_8B_REVISION

ROOT = Path(__file__).resolve().parents[1]
PANELS = ('original', 'letters', 'balanced', 'rank_preserved')
SEPARATION = Decimal('0.10')
OPTION = re.compile(r'^([A-D]): (.*)$', re.M)
PREREG = ROOT / 'docs/phase3/3a-shortcut-preregistration.md'
DEFAULT = ROOT / 'kit/resources/3a-shortcut'


def options(prompt):
    pairs = OPTION.findall(prompt)
    if len(pairs) != 4 or [x[0] for x in pairs] != list('ABCD'):
        raise ValueError('expected four A: ... D: option lines')
    return [x[1] for x in pairs]


def replace_options(prompt, texts):
    texts = iter(texts)
    return OPTION.sub(lambda m: m[1] + ': ' + next(texts), prompt)


def rng_for(key):
    return random.Random(int(sha(('3a-v1|' + key).encode()), 16))


def draw(pool, gold, rank, key):
    """Training-option draws, not molecular predictions. No extrapolation fallback."""
    rng = rng_for(key)
    chosen = []
    for below, count in ((True, rank - 1), (False, 4 - rank)):
        if not count:
            continue
        candidates = sorted(t for t in pool if (Decimal(t) < gold) == below
                            and abs(Decimal(t) - gold) >= SEPARATION)
        rng.shuffle(candidates)
        selected = []
        for text in candidates:
            value = Decimal(text)
            if all(abs(value - Decimal(t)) >= SEPARATION for t in selected):
                selected.append(text)
            if len(selected) == count:
                break
        if len(selected) != count:
            raise ValueError(f'insufficient separated training values: {key}, rank {rank}')
        chosen.extend(selected)
    rng.shuffle(chosen)
    return chosen


def make_row(item, family, panel, variant, texts=None, gold=None, rank=None):
    prompt = item['prompt'] if texts is None else replace_options(item['prompt'], texts)
    index = str(item['idx']) if panel == 'original' else f"{item['idx']}:{panel}:{variant}"
    return {'data_source': 'sciknoweval',
            'prompt': [{'role': 'system', 'content': item['system']}, {'role': 'user', 'content': prompt}],
            'ability': 'mcq', 'reward_model': {'style': 'mcq', 'ground_truth': gold or item['answer']},
            'extra_info': {'index': index, 'question_id': f"sciknoweval-test-{item['idx']}",
                           'family': family, 'affected': family in REMOVED_FAMILIES,
                           'panel': panel, 'variant': variant, 'gold_rank': rank}}


def rows(path):
    return [json.loads(line) for line in path.read_text().split('\n') if line.strip()]


def item_id(row):
    return 'sciknoweval-test-' + str(row['extra_info']['index'])


def build(sdpo, out):
    sdpo, out = Path(sdpo), Path(out)
    frozen = freeze_chemistry(sdpo)  # reuse programme source/taxonomy identity checks
    tax = json.loads(TAXONOMY.read_text())
    mapping = {(q['split'], q['position']): q['family'] for q in tax['questions']}
    source = {s: read_json(sdpo / CHEM_DIR / (s + '.json')) for s in ('train', 'test')}
    pools = {f: set() for f in REMOVED_FAMILIES}
    for i, item in enumerate(source['train']):
        f = mapping['train', i]
        if f in pools:
            pools[f].update(options(item['prompt']))
    panels = {p: [] for p in PANELS}
    for i, item in enumerate(source['test']):
        f = mapping['test', i]
        original = options(item['prompt'])
        gold_pos = 'ABCD'.index(item['answer'])
        panels['original'].append(make_row(item, f, 'original', 0))
        for rotation in (1, 2, 3):
            texts = original[rotation:] + original[:rotation]
            letter = 'ABCD'[(gold_pos - rotation) % 4]
            panels['letters'].append(make_row(item, f, 'letters', rotation, texts, letter))
        if f not in pools:
            continue
        gold_text, gold_value = original[gold_pos], Decimal(original[gold_pos])
        rank_order = list(range(1, 5))
        rng_for(str(item['idx']) + '|rank-order').shuffle(rank_order)
        for variant, rank in enumerate(rank_order):
            letter = 'ABCD'[(gold_pos - variant) % 4]
            for panel, target in (('balanced', rank), ('rank_preserved', 3)):
                distractors = draw(pools[f], gold_value, target, f"{item['idx']}|{panel}|{variant}")
                distractors.insert('ABCD'.index(letter), gold_text)
                panels[panel].append(make_row(item, f, panel, variant, distractors, letter, target))
    if out.exists():
        raise ValueError(f'choose a new panel directory: {out}')
    out.mkdir(parents=True)
    for p, items in panels.items():
        (out / p).mkdir()
        (out / p / 'test.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False, sort_keys=True) + '\n' for r in items))
    (out / 'sources').mkdir()
    for split in ('train', 'test'):
        shutil.copyfile(sdpo / CHEM_DIR / (split + '.json'), out / 'sources' / (split + '.json'))
    shutil.copyfile(TAXONOMY, out / 'sources/taxonomy.json')
    shutil.copyfile(PREREG, out / 'preregistration.md')
    scorer = Path('verl/utils/reward_score/feedback/mcq.py')
    (out / 'scorer' / scorer).parent.mkdir(parents=True)
    shutil.copyfile(sdpo / scorer, out / 'scorer' / scorer)
    from kit.chemistry_shortcut_validate import validate
    validation = validate(out)
    (out / 'validation.json').write_text(json.dumps(validation, indent=2, sort_keys=True) + '\n')
    if validation['errors']:
        raise ValueError('panel construction errors: ' + '; '.join(validation['errors']))
    manifest = {'schema': 'chemistry-shortcut-panels.v1', 'base_revision': INITIAL_8B_REVISION,
                'source_commit': frozen['source_commit'], 'sources': frozen['sources'],
                'panels': {p: len(items) for p, items in panels.items()},
                'minimum_separation': str(SEPARATION), 'seed_rule': 'sha256(3a-v1|key), Python Random',
                'files': {str(p.relative_to(out)): sha(p.read_bytes()) for p in sorted(out.rglob('*')) if p.is_file()},
                'code': {f'kit/{name}.py': sha((ROOT / 'kit' / (name + '.py')).read_bytes())
                         for name in ('chemistry_shortcut_panels', 'chemistry_shortcut_validate', 'eval_bed')}}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sdpo', type=Path, required=True)
    parser.add_argument('--out', type=Path, default=DEFAULT)
    args = parser.parse_args()
    print(json.dumps(build(args.sdpo, args.out)['panels']))


if __name__ == '__main__':
    main()
