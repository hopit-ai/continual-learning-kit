"""Offline, owner-declared gold/27b overlap repair. Never repairs S8."""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import json
from pathlib import Path
import re
import tempfile

from kit.trace_common import (KIT, SUBSET, bound_file, digest, load, need,
                              read_rows, sha)
from kit.trace_overlap import audit_rows
from kit.p3b_replay_data import prepared

RULE = KIT/'resources/trace-repair/rule-20261122.md'
RULE_VERSION = 'trace-repair-20261122.v1'


def content(row):
    return prepared(row['prompt'])[0], prepared(row['answer'])[0]


def pair_rule(train, held, reasons, trace_heldout):
    if 'source_group' in reasons:
        return 'R1'
    if not trace_heldout:
        return 'R2'
    if train['task'] in ('FOMC', 'MeetingBank'):
        return 'R3'
    need(train['task'] in ('NumGLUE-ds', 'NumGLUE-cm'), 'unsupported repair task')
    left, right = prepared(train['prompt'])[0], prepared(held['prompt'])[0]
    numbers = r'\d+(?:\.\d+)?'
    if left == right or re.findall(numbers, left) == re.findall(numbers, right):
        return 'R4'
    return None  # Template sibling with different quantities.


def rule_review_problems(datasets, exclusions, flags, reviews):
    """Re-adjudicate rule reviews against current bound content.

    Called only for shipped-rule evidence. Manual evidence retains its existing
    audit semantics. Return operator-facing problems rather than raising.
    """
    evidence = sha(RULE)
    ruled = [r for r in reviews if r.get('evidence_sha256') == evidence]
    problems = []
    train = {(task, row['id']): row for task, splits in datasets.items()
             for row in splits['train']}
    held = {(task, row['id']): (row, True) for task, splits in datasets.items()
            for split in ('eval', 'test') for row in splits[split]}
    for row in exclusions:
        key = (row.get('task', 'exclusion'), row['id'])
        if key in held:
            problems.append('stale rule review: ambiguous heldout/exclusion identity')
        held[key] = (row, False)
    pairs = {p['id']: p for p in flags}
    for review in ruled:
        pair = pairs.get(review['id'])
        if pair is None:
            problems.append('stale rule review: pair absent from current audit: '+review['id'])
            continue
        other, is_trace = held[tuple(pair['heldout'])]
        current = train[tuple(pair['train'])]
        if (review.get('disposition') != 'cleared'
                or current['task'] not in ('NumGLUE-ds', 'NumGLUE-cm')
                or pair_rule(current, other, pair['reasons'], is_trace) is not None):
            problems.append('stale rule review: current pair is not R4 cleared: '+pair['id'])

    return problems


def repair_binding_problems(cfg, studies):
    """Validate rule repair bindings before preparation reads datasets."""
    evidence = sha(RULE)
    problems = []

    def check(condition, binding):
        if not condition:
            problems.append('stale repair binding: '+binding)

    def check_file(ref, binding):
        try:
            check(sha(ref['path']) == ref.get('sha256'), binding+' sha256')
        except (KeyError, TypeError, OSError):
            problems.append('stale repair binding: '+binding+' file or sha256')

    check_file(cfg['audit_review'], 'audit_review')
    if not set(studies) & {'gold', '27b'}:
        return problems
    for task in SUBSET:
        for split in ('train', 'eval', 'test'):
            check_file(cfg['datasets'].get(task, {}).get(split, {}), task+'/'+split)
    check_file(cfg['exclusions'], 'exclusions')
    ref = cfg.get('repair_manifest')
    if not isinstance(ref, dict):
        problems.append('stale repair binding: repair_manifest required for rule reviews')
        return problems
    try:
        path = Path(ref['path'])
        if not path.is_file() or sha(path) != ref.get('sha256'):
            problems.append('stale repair binding: repair_manifest sha256 or file')
            return problems
        manifest = load(path)
        check(manifest['rule_sha256'] == evidence, 'rule_sha256 (shipped rule)')
        for task in SUBSET:
            check(cfg['datasets'][task]['train']['sha256'] ==
                  manifest['outputs']['datasets'][task]['train']['sha256'],
                  task+'/train output sha256')
            for split in ('eval', 'test'):
                check(cfg['datasets'][task][split]['sha256'] ==
                      manifest['input_sha256']['datasets'][task][split],
                      task+'/'+split+' input sha256')
        check(cfg['exclusions']['sha256'] == manifest['input_sha256']['exclusions'],
              'exclusions input sha256')
        check(cfg['audit_review']['sha256'] == manifest['outputs']['audit_review']['sha256'],
              'audit_review manifest sha256')
    except (KeyError, TypeError, ValueError, OSError):
        problems.append('stale repair binding: repair_manifest incomplete or unreadable')
    return problems


def empty_transcript(row):
    if row['task'] != 'MeetingBank':
        return False
    # This is the released instruction, not an arbitrary prompt with markers.
    instruction = ('Write a summary of the following meeting transcripts. '
                   'Meeting transcripts: Summary:')
    return ''.join(row['prompt'].split()) == ''.join(instruction.split())


def counts(rows):
    return dict(rows=len(rows), distinct=len({content(row) for row in rows}))


def self_check(datasets, exclusions, reviews):
    audit = audit_rows(datasets, exclusions, reviews)
    per_task = {task: counts(splits['train']) for task, splits in datasets.items()}
    problems = []
    if audit['unresolved']:
        problems.append(dict(rule='R1-R4', reason='unresolved repaired overlap',
                             unresolved=audit['unresolved']))
    for task, value in per_task.items():
        if value['distinct'] < 500:
            problems.append(dict(task=task, rule='500 distinct',
                                 reason='fewer than 500 distinct training contents'))
    return audit, per_task, problems


def repair(cfg, out, manual_review=None):
    need(cfg.get('studies') == ['gold', '27b'], 'repair requires studies [gold, 27b]; S8 is out of scope')
    need(set(SUBSET) <= set(cfg['datasets']), 'subset task inventory incomplete')
    datasets = {task: {split: read_rows(ref) for split, ref in cfg['datasets'][task].items()}
                for task in SUBSET}
    need(all(set(splits) == {'train', 'eval', 'test'} for splits in datasets.values()),
         'released splits incomplete')
    exclusions = read_rows(cfg['exclusions'])
    audit = audit_rows(datasets, exclusions)
    if manual_review is not None:
        given = load(manual_review)['pairs']
        need(sorted(given, key=lambda p: p['id']) == sorted(audit['flagged'], key=lambda p: p['id']),
             'manual-review pairs differ from recomputed audit')
    train = {(task, row['id']): row for task, splits in datasets.items() for row in splits['train']}
    held = {(task, row['id']): (row, True) for task, splits in datasets.items()
            for split in ('eval', 'test') for row in splits[split]}
    for row in exclusions:
        key = (row.get('task', 'exclusion'), row['id'])
        need(key not in held, 'ambiguous held-out/exclusion identity')
        held[key] = (row, False)
    drops, cleared = defaultdict(lambda: defaultdict(list)), []
    evidence = sha(RULE)
    for pair in audit['flagged']:
        key = tuple(pair['train'])
        other, is_trace = held[tuple(pair['heldout'])]
        rule = pair_rule(train[key], other, pair['reasons'], is_trace)
        if rule:
            drops[key][rule].append(pair['id'])
        else:
            cleared.append(dict(id=pair['id'], disposition='cleared', evidence_sha256=evidence))
    for key, row in train.items():
        if empty_transcript(row):
            drops[key]['R5']  # Independent of flags; an empty list records no pair.
    verdicts = {}
    for key, row in train.items():
        value = key in drops
        identity = content(row)
        need(identity not in verdicts or verdicts[identity] == value,
             'identical normalised training contents have inconsistent verdicts')
        verdicts[identity] = value
    repaired = {task: dict(splits, train=[r for r in splits['train'] if (task, r['id']) not in drops])
                for task, splits in datasets.items()}
    # Reviews for removed train IDs are not valid review-array members.
    kept_pair_ids = {p['id'] for p in audit['flagged'] if tuple(p['train']) not in drops}
    reviews = [r for r in cleared if r['id'] in kept_pair_ids]
    after_audit, after, problems = self_check(repaired, exclusions, reviews)
    dropped = [dict(task=key[0], id=key[1], rules=[dict(rule=rule, pair_ids=ids)
               for rule, ids in sorted(rules.items())]) for key, rules in sorted(drops.items())]
    rule_counts = {task: dict(Counter(rule for (t, _), rules in drops.items() if t == task
                                    for rule in rules)) for task in SUBSET}
    manifest = dict(rule_version=RULE_VERSION, rule_sha256=evidence,
        input_config_sha256=digest(cfg),
        input_sha256=dict(datasets={task: {split: sha(bound_file(ref)) for split, ref in cfg['datasets'][task].items()}
                                   for task in SUBSET}, exclusions=sha(bound_file(cfg['exclusions'])),
                          manual_review=sha(manual_review) if manual_review else None),
        pairs_before=dict(Counter(p['train'][0] for p in audit['flagged'])),
        dropped=dropped, drops_per_rule=rule_counts,
        counts={task: dict(before=counts(datasets[task]['train']), after=after[task]) for task in SUBSET},
        cleared_pair_count=len(cleared), remaining_cleared_pair_count=len(reviews),
        cleared_reason='template sibling with different quantities',
        unresolved=after_audit['unresolved'])
    if problems:
        return dict(status='PENDING', problems=problems, manifest=manifest)
    out = Path(out)
    need(not out.exists(), 'choose a fresh repair directory')
    out.parent.mkdir(parents=True, exist_ok=True)
    # All checks precede publication. Rename a complete sibling directory once.
    with tempfile.TemporaryDirectory(prefix='.trace-repair-', dir=out.parent) as temporary:
        stage = Path(temporary)/'complete'
        stage.mkdir()
        edits = dict(datasets={}, audit_review=None,
                     out='CHOOSE_A_FRESH_PREPARATION_STAGE_DIRECTORY')
        for task in SUBSET:
            name = task+'-train.jsonl'
            path = stage/name
            path.write_text(''.join(json.dumps(row, ensure_ascii=False, sort_keys=True)+'\n'
                                   for row in repaired[task]['train']), encoding='utf-8')
            edits['datasets'][task] = dict(train=dict(path=str(out/name), sha256=sha(path)))
        review_path = stage/'audit_review.json'
        review_path.write_text(json.dumps(reviews, indent=2, sort_keys=True)+'\n', encoding='utf-8')
        edits['audit_review'] = dict(path=str(out/'audit_review.json'), sha256=sha(review_path))
        # The manifest binds outputs; its own reference belongs only in config
        # edits, avoiding a self-referential hash.
        manifest['outputs'] = deepcopy(edits)
        (stage/'repair-manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True)+'\n', encoding='utf-8')
        edits['repair_manifest'] = dict(path=str(out/'repair-manifest.json'),
                                       sha256=sha(stage/'repair-manifest.json'))
        stage.rename(out)
    return dict(status='COMPLETE', manifest=manifest, config_edits=edits,
                next='Merge these train/audit_review/repair_manifest bindings into the original config, choose a fresh stage directory, rerun preparation.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--manual-review', type=Path)
    args = parser.parse_args()
    result = repair(load(args.config), args.out.resolve(), args.manual_review)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result['status'] == 'COMPLETE' else 3


if __name__ == '__main__':
    raise SystemExit(main())
