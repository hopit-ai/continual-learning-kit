"""CR-only task exclusions before the unchanged schedulers (owner, 23 Nov)."""
from pathlib import Path
import json
import subprocess
import sys
from kit.nm_common import KIT, atomic, load, need, ready, sha, seal
from kit.p3b_common import t, files_manifest

PROTOCOL = 'cr-task-filter-20261123.v1'


def exclude(rows, panels):
    from kit.m3_pools import Matcher, text_of
    from rapidfuzz.distance import Levenshtein
    matcher = Matcher(panels)
    references = dict(matcher.reference)
    kept, exclusions = [], []
    for row in rows:
        value = matcher.prepared(text_of(row['prompt']))
        hit, rule = matcher.find(row, value)
        if hit is None:
            kept.append(row)
            continue
        other = references[hit]
        a, x = value; b, y = other
        jaccard = len(x & y)/max(1, len(x | y))
        edit = 1-Levenshtein.distance(a, b)/max(1, len(a), len(b))
        branch = ('normalised_equality' if rule == 'normalised_equality' else
                  'jaccard' if x and y and min(len(x),len(y)) >= .8*max(len(x),len(y))
                  and jaccard >= .8 else 'edit_similarity')
        exclusions.append(dict(id=str(row['id']), panel_item=hit, rule=rule,
                               rule_branch=branch, jaccard=jaccard, edit_similarity=edit))
    ready(bool(kept), 'CR task exclusions PENDING: empty training corpus')
    return kept, exclusions


def write_stage(root, schedules, exclusions, metadata):
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    names = []
    for name, rows in dict(schedules, **{'exclusions.jsonl': exclusions}).items():
        path = root/name
        if path.exists():
            need(t.read_jsonl(path) == rows, 'CR filtered task bytes changed')
        else:
            t.write_jsonl(path, rows)
        names.append(name)
    seal(root/'receipt.json', metadata); names.append('receipt.json')
    seal(root/'manifest.json', files_manifest(root, names, schedule_protocol=PROTOCOL,
                                             filter_code_sha256=sha(KIT/'cr_filter.py')))
    print('CR exclusions', metadata['stage'], json.dumps(metadata['counts'], sort_keys=True))


def finqa(cfg, source, panels, check, root, metadata):
    """Use bind_python for the parquet dependencies and native B scheduler."""
    request = check/'filter-request.json'
    atomic(request, dict(cfg=cfg, source=source, panels=panels, root=str(root),
                         work=str(check/'filtered-finqa'), metadata=metadata))
    bootstrap = ('import sys;sys.path[:0]='+repr([str(KIT.parent)])+
                 ';from kit.cr_filter import native_filtered;native_filtered(sys.argv[1])')
    subprocess.run([cfg['bind_python'], '-I', '-c', bootstrap, str(request)], check=True)


def native_filtered(request):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from kit import v4_campaign_ops as b, cr_finqa
    cfg = load(request); source = cfg['source']; work = Path(cfg['work'])
    raw = pq.read_table(source['corpus']).to_pylist()
    projected = [dict(r, id=str(r['extra_info']['index'])) for r in raw]
    kept, exclusions = exclude(projected, cfg['panels'])
    ids = {r['id'] for r in kept}
    filtered = [r for r in raw if str(r['extra_info']['index']) in ids]
    original = work/'source'; common = b.paths(original, 'main')/'common'
    common.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(filtered), common/'finqa.parquet')
    manifest = load(Path(source['corpus']).parent/'manifest.json')
    manifest['files']['finqa']['sha256'] = sha(common/'finqa.parquet')
    atomic(common/'manifest.json', manifest)
    test = original/'v4/datasets/datasets/v4_finqa/test.parquet'
    test.parent.mkdir(parents=True)
    test.symlink_to(Path(source['b_work'])/'v4/datasets/datasets/v4_finqa/test.parquet')
    native = work/'native'; req = work/'native.json'
    atomic(req, dict(source=dict(source,b_work=str(original)), work=str(native),
                     standin=bool(cfg['cfg'].get('technical_synthetic')),
                     tokenizer=cfg['cfg'].get('tokenizer')))
    cr_finqa.native_worker(req)  # unchanged B shuffle/cycle and target projection
    schedules = {f'schedule-{seed}.jsonl': t.read_jsonl(native/f'schedule-{seed}.jsonl')
                 for seed in cr_finqa.FRESH}
    metadata = dict(cfg['metadata'], counts=dict(before=len(raw), excluded=len(exclusions), after=len(kept)))
    write_stage(cfg['root'], schedules, exclusions, metadata)


def verify(root, stage, binding):
    from kit.p3b_common import frozen_files
    root = Path(root)
    try:
        manifest = frozen_files(root)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
        ready(False, 'CR filtered task seal changed; owner escalation: '+type(error).__name__)
    ready(manifest.get('schedule_protocol') == PROTOCOL and
          manifest.get('filter_code_sha256') == sha(KIT/'cr_filter.py'),
          'CR filtered task protocol/code differs; owner escalation')
    receipt = load(root/'receipt.json')
    ready(receipt.get('stage') == stage and receipt.get('confirmation_binding') == binding,
          'CR filtered task binding differs; owner escalation')
    ready('exclusions.jsonl' in manifest['files'], 'CR exclusions seal missing; owner escalation')
    freeze = root.parent/'pre-data-freeze.json'
    if freeze.exists():
        ready(load(freeze).get('task_manifests',{}).get(stage)==sha(root/'manifest.json'),
              'CR filtered task manifest changed; owner escalation')
    return receipt
