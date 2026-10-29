#!/usr/bin/env python3
"""Validate staged, pinned Screen S sources without a network boundary."""
from pathlib import Path
import sys
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kit.s_common import *


def pins():
    from kit.next_pools import GSM, ARC
    from kit.p3b_replay_data import SOURCE
    from kit.beds.finqa import DATASET_COMMIT
    return dict(
        ifeval=dict(repo='allenai/RLVR-IFeval', revision=SOURCE, split='train', count=14973),
        gsm8k=dict(repo='openai/gsm8k', revision=GSM, split='train', count=7473),
        arc=dict(repo='allenai/ai2_arc', revision=ARC, split='validation', count=299),
        finqa=dict(repo='czyssrs/FinQA', revision=DATASET_COMMIT, split='dev', count=883))


def load(domain, ref, standin=False):
    from kit import next_pools as broad
    path = checked(ref)
    pin = pins()[domain]
    require(all(ref[k] == pin[k] for k in ('repo', 'revision', 'split')), 'wrong source pin/split: '+domain)
    if not standin:
        require(ref['sha256'] == registration()['sources'][domain]['sha256'], 'unregistered source bytes: '+domain)
    if standin:
        result = rows(path)
    elif domain == 'finqa':
        from kit.beds import finqa
        raw = read(path)
        result = [dict(id=str(r['id']), prompt=finqa.render_prompt(r), gold=finqa.gold_of(r), source=domain) for r in raw]
    else:
        import pyarrow.parquet as pq
        raw = pq.read_table(path).to_pylist()
        if domain == 'gsm8k':
            result = broad.candidates(raw, [])
        elif domain == 'arc':
            result = broad.candidates([], raw)
            result = [dict(r, id=r['id'].replace('arc-c-train-', 'arc-c-validation-', 1)) for r in result]
        else:
            from kit.p3b_replay_data import checker
            result = []
            for i, r in enumerate(raw):
                prompt = r.get('prompt', r.get('messages'))
                if isinstance(prompt, list):
                    require(len(prompt) == 1 and prompt[0]['role'] == 'user', 'unexpected RLVR chat')
                    prompt = prompt[0]['content']
                require(isinstance(prompt, str) and bool(prompt.strip()), 'missing RLVR prompt')
                gt = r.get('ground_truth', (r.get('reward_model') or {}).get('ground_truth'))
                checker(None, gt, validate_only=True)
                result.append(dict(id=str(r.get('id', r.get('key', i))), prompt=prompt, ground_truth=gt, source=domain))
        if pin['count'] is not None:
            require(len(result) == pin['count'], 'source population changed: '+domain)
    if not standin and domain == 'finqa':
        require(len(result) == pin['count'], 'FinQA dev population changed')
    require(len(result) == len({r['id'] for r in result}) and bool(result), 'empty/duplicate source IDs')
    require(all(r.get('prompt') and ('ground_truth' in r if domain == 'ifeval' else 'gold' in r) for r in result), 'missing prompt/gold')
    return [dict(r, domain=domain) for r in result]


def check_sources(source, standin=False):
    doc = read(source)
    require(set(doc) == set(DOMAINS), 'all four staged source pins required')
    return {d: dict(pin=pins()[d], sha256=sha(checked(doc[d])), rows=len(load(d, doc[d], standin))) for d in DOMAINS}


def job(a):
    atomic(a.out/'staged-inputs.json', check_sources(a.sources, a.standin))

if __name__ == '__main__':
    raise SystemExit(entry(job, lambda p: p.add_argument('--sources', type=Path, required=True)))
