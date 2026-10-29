#!/usr/bin/env python3
"""Manager-only pre-tag check. Network is opt-in by running this CLI, never at import.

Requires huggingface_hub and pyarrow. Downloads only the dataset parquet files
read by the kit; model checks use API metadata only. All network access is in
network(), which offline tests replace. Local loaders never access the network.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def network(operation, **kwargs):
    """The sole network boundary: HF revision/file metadata or one dataset file."""
    from huggingface_hub import HfApi, hf_hub_download
    if operation == 'info':
        info = HfApi().repo_info(**kwargs, files_metadata=True)
        return {'revision': info.sha, 'files': [s.rfilename for s in info.siblings]}
    if operation == 'download':
        if kwargs.get('repo_type') != 'dataset' or not kwargs['filename'].endswith('.parquet'):
            raise ValueError('only dataset parquet downloads are allowed')
        return hf_hub_download(**kwargs)
    raise ValueError('unknown network operation: ' + operation)


def pins():
    from kit import next_hellaswag as hs, next_pools as pools, second_task_screen as screen
    math = json.loads((ROOT/'kit/resources/math-hard-teacher/data-manifest.json').read_text())['source']
    models = json.loads((ROOT/'kit/resources/p3b/model-hashes.json').read_text())
    result = [
        dict(name='hellaswag', repo='Rowan/hellaswag', revision=hs.REVISION,
             repo_type='dataset', fragment='validation', hashes=hs.VALIDATION_FILES,
             panel_sha256='2b0fc85c6e80854bb47982d35a18d15bc83a5655de6202634547071eccbe487d'),
        dict(name='gsm8k', repo='openai/gsm8k', revision=pools.GSM,
             repo_type='dataset', fragment='main/train-', hashes={}),
        dict(name='arc', repo='allenai/ai2_arc', revision=pools.ARC,
             repo_type='dataset', fragment='ARC-Challenge/train-', hashes={}),
        dict(name='math', repo=math['dataset'], revision=math['revision'],
             repo_type='dataset', hashes=math['parquet_sha256'],
             derived_files={'train_pool.jsonl': 'a35cecee96920a6a657b526477fbffc75cc09ba221f108919ed10bbca6ae830d',
                            'eval.jsonl': '0bee96d6576a5144a0150d26170b08b6a04644bffcdc31db7e1f7e58f2c21b98',
                            'manifest.json': 'ee2417d5b1b772e994b80b70ea8866b5f49e79e70d241559e96145c55e1a0b8f'}),
    ]
    for role in ('teacher', 'student'):
        repo, revision = screen.MODELS[role]
        if revision != models[role]['revision']:
            raise ValueError('model pins disagree: ' + role)
        result.append(dict(name=role, repo=repo, revision=revision, repo_type='model',
                           hashes=models[role]['files']))
    return result


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def unique(rows):
    require(bool(rows), 'empty loaded rows')
    require(len({r['id'] for r in rows}) == len(rows), 'duplicate loaded IDs')


def load_rows(pin, root):
    """Run the production row parsers and their checks on downloaded bytes."""
    from kit import next_hellaswag as hs, next_pools as pools
    name = pin['name']
    if name in ('hellaswag', 'gsm8k', 'arc'):
        raw, _ = hs.download(pin['repo'], pin['revision'], pin['fragment'],
                             pin['hashes'] or None, root=root)
        if name == 'hellaswag':
            activity = [r for r in raw if r['source_id'].startswith('activitynet~')]
            require(len(raw) == 10042 and len(activity) == 3243, 'HellaSwag population count changed')
            rows = hs.panel_rows(raw)
            unique(rows)
            require(len(rows) == 100, 'HellaSwag panel count changed')
            # Exactly the bytes emitted by the kit's write_jsonl.
            from kit.next_common import write_rows
            panel = root/'panel.jsonl'
            write_rows(panel, rows)
            digest = hashlib.sha256(panel.read_bytes()).hexdigest()
            require(digest == pin['panel_sha256'],
                    'frozen HellaSwag panel SHA256 changed')
            return dict(rows=len(raw), activitynet=len(activity), panel=len(rows), panel_sha256=digest)
        rows = pools.candidates(raw if name == 'gsm8k' else [], raw if name == 'arc' else [])
        unique(rows)
        require(len(rows) == {'gsm8k': 7473, 'arc': 1119}[name], name + ' training count changed')
        return dict(rows=len(rows), unique_ids=len(rows))
    if name == 'math':
        import pyarrow.parquet as pq
        from kit import second_task_screen as screen, math_hard_teacher as teacher
        tree = root/'MATH'
        counts = Counter()
        for filename in sorted(pin['hashes']):
            subject, file = filename.split('/')
            split = file.split('-')[0]
            for i, row in enumerate(pq.read_table(root/filename).to_pylist()):
                target = tree/split/subject/f'{i}.json'
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(json.dumps({k: row[k] for k in ('problem', 'level', 'type', 'solution')}, ensure_ascii=False))
                counts[split] += 1
        require(counts == {'train': 7500, 'test': 5000}, 'MATH source counts changed')
        pool, panel, _, _ = screen.math_inputs(tree)
        unique(pool); unique(panel)
        frozen = teacher.frozen()
        require(len(pool) == 500 and len(panel) == 100, 'MATH selected counts changed')
        return dict(source_counts=dict(counts), pool=len(pool), panel=len(panel), hard_panel=len(frozen['panel']))
    raise ValueError('no kit loader: ' + name)


def check_hf(pin, scratch, receipt):
    info = network('info', repo_id=pin['repo'], repo_type=pin['repo_type'], revision=pin['revision'])
    receipt['resolved_revision'] = info['revision']
    receipt['listed_files'] = sorted(info['files'])
    require(info['revision'] == pin['revision'], 'revision resolved to a different commit')
    listed = set(info['files'])
    if pin['repo_type'] == 'model':
        require(set(pin['hashes']) <= listed, 'model expected files absent: ' + ', '.join(sorted(set(pin['hashes'])-listed)))
        receipt['expected_hashes'] = pin['hashes']
        receipt['validation'] = 'revision and file listing only; model bytes not downloaded or hash-verified'
        return
    if pin['hashes']:
        files = sorted(pin['hashes'])
        require(set(files) <= listed, 'required parquet files absent at revision')
    else:
        files = sorted(f for f in listed if f.endswith('.parquet') and pin['fragment'] in f)
    require(bool(files), 'no required parquet files at revision (loader-only revisions cannot be staged)')
    root = scratch/pin['name']
    root.mkdir()
    for filename in files:
        # Avoid redirects to unexpected local paths even in a malformed API listing.
        require(not Path(filename).is_absolute() and '..' not in Path(filename).parts, 'invalid dataset path')
        source = Path(network('download', repo_id=pin['repo'], repo_type='dataset',
                              revision=pin['revision'], filename=filename))
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        receipt['hashes'][filename] = digest
        require(filename not in pin['hashes'] or digest == pin['hashes'][filename], 'pinned SHA256 mismatch: ' + filename)
        target = root/filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(source.resolve())
    receipt['counts'] = load_rows(pin, root)
    if pin['name'] == 'math':
        check_math_hard(pin, receipt)


def check_math_hard(pin, receipt):
    """New derived dataset inputs, from the same immutable MATH source; no extra download."""
    from kit.m2_common import hard_data, M
    rows = hard_data()
    manifest = json.loads((M/'manifest.json').read_text())
    require(manifest['source']['dataset'] == pin['repo'] and manifest['source']['revision'] == pin['revision'],
            'hard-MATH source pin differs')
    observed = {}
    for filename, expected in pin.get('derived_files', {}).items():
        digest = hashlib.sha256((M/filename).read_bytes()).hexdigest()
        require(digest == expected, 'hard-MATH derived SHA256 mismatch: ' + filename)
        # The repository data and public kit copy must stay byte-identical.
        require(hashlib.sha256((ROOT/'data/math_hard'/filename).read_bytes()).hexdigest() == digest,
                'hard-MATH repository/kit copy differs: ' + filename)
        observed[filename] = digest
    receipt['derived_dataset_inputs'] = dict(hashes=observed, train_pool=len(rows['train_pool.jsonl']), evaluation=len(rows['eval.jsonl']))


def check_contract(source, receipt):
    from kit import c_targets
    require(source is not None, 'supply --contract-source with approved original train.json/dev.json/test.json')
    expected = json.loads((ROOT/'kit/resources/c/contract_nli/manifest.json').read_text())['source']
    for filename, digest in expected['source_files_sha256'].items():
        actual = hashlib.sha256((source/filename).read_bytes()).hexdigest()
        receipt['hashes'][filename] = actual
        require(actual == digest, 'pinned SHA256 mismatch: ' + filename)
    splits = c_targets.source_rows(source, False)
    for rows in splits.values():
        unique(rows)
    counts = {k: len(v) for k, v in splits.items()}
    require(counts == expected['source_counts'], 'ContractNLI source counts changed')
    receipt['counts'] = counts


def run(contract_source=None):
    receipts = []
    with tempfile.TemporaryDirectory(prefix='check-pinned-inputs-') as scratch:
        for pin in pins() + [dict(name='contract_nli', repo_type='local')]:
            receipt = {k: pin[k] for k in ('name', 'repo', 'revision', 'repo_type') if k in pin}
            receipt.update(ok=False, reason='', hashes={})
            try:
                if pin['repo_type'] == 'local':
                    check_contract(contract_source, receipt)
                else:
                    check_hf(pin, Path(scratch), receipt)
                receipt.update(ok=True, reason='passed')
            except Exception as exc:
                receipt['reason'] = type(exc).__name__ + ': ' + str(exc)
            receipts.append(receipt)
    return receipts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract-source', type=Path)
    parser.add_argument('--receipt', type=Path, help='also save the JSON receipt here')
    parser.add_argument('--screen-s-sources', type=Path,
                        help='offline Screen S staged-source manifest; checks four pins/splits and local bytes only')
    args = parser.parse_args(argv)
    if args.screen_s_sources:
        from kit.s_inputs import check_sources
        try:
            receipts = [dict(name=k, ok=True, reason='passed offline source loader', **v)
                        for k, v in check_sources(args.screen_s_sources).items()]
        except Exception as exc:
            receipts = [dict(name='screen-s', ok=False, reason=str(exc), hashes={})]
    else:
        receipts = run(args.contract_source)
    text = json.dumps(receipts, indent=2, sort_keys=True) + '\n'
    if args.receipt:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(text)
    print(text, end='')
    return 0 if all(r['ok'] for r in receipts) else 1


if __name__ == '__main__':
    raise SystemExit(main())
