#!/usr/bin/env python3
"""SP scoring: template-reliance panel verdicts, frozen correct cohorts and length metrics.

    python kit/sp_score.py panel  --entry ENTRY.json --model DIR --gpu N --out DIR   # one GPU
    python kit/sp_score.py cohort --cell CELL.json --out FILE                       # CPU
    python kit/sp_score.py length --cell CELL.json --cohort FILE --out FILE         # CPU

Scoring-only and descriptive (registration v2, SP). Nothing here trains, admits or vetoes an arm,
and nothing writes into a message-2 output root: every output path is under the SP sidecar root.
The frozen panel, checkers and fixtures belong to kit/sp_prepare.py and kit/resources/sp/
(m3sp-author); this file consumes them read-only through sp_prepare.verify_frozen/load_checkers.

Length convention (bound by LENGTH_CONVENTION; refused if it differs between checkpoints):
generated tokens are the engine's output token IDs for the response, counted with EOS when the
engine returned it; prompt/input tokens and padding are never counted. A response is truncated
iff finish_reason is "length". Tokens per correct answer is the sum of ALL generated tokens in the
scope, including incorrect and truncated answers, divided by the number correct. A graded
question without raw response/finish metadata (including c_score's finish_reason="missing"
zero-token placeholder) is MISSING: its scope is INCOMPLETE and reports no length value, so a
missing answer is never counted as a short one and never silently leaves the denominator.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kit.pd_contract import digest, is_sha, require, seal, sha256_file, verify_seal

CAP = 2048
DECODING = dict(temperature=0.0, top_p=1.0, top_k=-1, repetition_penalty=1.0, max_tokens=CAP,
                seed=0, thinking=False, eager=True, batch_invariant=True)
LENGTH_CONVENTION = dict(
    schema='sp-length-v2', cap=CAP,
    count='engine output token IDs of the response; EOS counted when returned; input and padding excluded',
    truncated="finish_reason == 'length'", tokens_per_correct='sum of all generated tokens / number correct',
    missing='no raw row, or finish_reason missing: scope INCOMPLETE, no value')
FINISHES = ('stop', 'eos', 'length')
MISSING_FINISHES = (None, '', 'missing')
COHORT_SCHEMA = 'sp-correct-cohort-v1'
COHORT_KINDS = ('incoming', 'base')
# Registered correctness column per generative panel (registration v2, SP(a)).
CORRECT_COLUMNS = {'finqa': 'correct', 'test': 'tolerant_correct', 'panel': 'tolerant_correct',
                   'general-v1': 'correct', 'math_hard': 'correct', 'sp': 'strict_all_constraints_pass'}
NOT_APPLICABLE = ('hellaswag', 'knowledge-ll')


def read_jsonl(path):
    with open(path, encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with open(tmp, 'w', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + '\n')
    tmp.replace(path)


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, sort_keys=True, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    tmp.replace(path)


# ---------------------------------------------------------------------------------------------
# Template-reliance panel verdicts (frozen panel owned by sp_prepare / resources/sp).

def load_panel(root=None):
    """(FROZEN doc, 400 panel rows, checker module) from the frozen SP panel, verified."""
    from kit import sp_prepare as prep
    root = prep.RESOURCES if root is None else Path(root)
    doc = prep.verify_frozen(root)
    ck = prep.load_checkers(expected_sha256=doc['checker_sha256'])
    panel = read_jsonl(Path(root) / 'panel.jsonl')
    require(len({r['id'] for r in panel}) == len(panel), 'duplicate SP panel IDs')
    require(len(panel) == doc.get('n_prompts', len(panel)), 'SP panel incomplete')
    return doc, panel, ck


def verify_fixtures(root, ck):
    """Checkers must still accept every witness and reject every mutation before scoring."""
    panel = read_jsonl(Path(root) / 'panel.jsonl')
    constraints = {r['request_id']: r['constraints'] for r in panel}
    bad = [f for f in read_jsonl(Path(root) / 'fixtures.jsonl')
           if ck.check_all(f['text'], constraints[f['request_id']]) != f['expected']]
    require(not bad, 'SP checker fixtures fail: %d' % len(bad))


def score_prompt(item, response, ck):
    verdicts = ck.check_all(response, item['constraints'])
    strict = int(all(verdicts))
    # Fence-related JSON failures are reported separately; strict accuracy always uses raw JSON.
    fence = None
    if any(c['family'] == 'json_only' for c in item['constraints']) and hasattr(ck, 'json_fence_failure'):
        fence = int(bool(ck.json_fence_failure(response)))
    return dict(id=item['id'], request_id=item['request_id'], wording=item['wording'],
                primary_family=item['primary_family'], strict=strict, strict_all_constraints_pass=strict,
                json_fence_failure=fence,
                instructions=[dict(family=c['family'], passed=int(v)) for c, v in zip(item['constraints'], verdicts)])


def score_panel(panel, raw, ck):
    """Strict all-constraints verdicts on raw text; complete coverage of every panel prompt required."""
    responses = {r['id']: r for r in raw}
    require(len(responses) == len(raw), 'duplicate response IDs')
    require(set(responses) == {r['id'] for r in panel}, 'incomplete or foreign SP responses')
    return [score_prompt(item, responses[item['id']]['response'], ck) for item in panel]


# ---------------------------------------------------------------------------------------------
# Frozen correct cohorts.

def _score_map(score_rows, checkpoint_id=None, column='correct'):
    out = {}
    for row in score_rows:
        if checkpoint_id is not None and row.get('checkpoint_id') not in (None, checkpoint_id):
            raise ValueError('cohort rows must all come from the incoming checkpoint')
        qid = str(row['question_id'])
        require(qid not in out, 'duplicate question_id in score rows: ' + qid)
        require(str(row.get(column, '')).strip() != '', 'UNVERIFIED: no %s grade for %s' % (column, qid))
        value = int(row[column])
        require(value in (0, 1), column + ' must be 0/1')
        out[qid] = value
    return out


def freeze_correct_cohort(path, *, family, seed, panel, incoming, score_rows, scores_sha256, raw_sha256=None,
                          kind='incoming', panel_sha256=None, column='correct'):
    """Freeze C = {question IDs graded correct at the untouched incoming checkpoint}.

    `incoming` is the incoming checkpoint identity (checkpoint_id and weights_sha256). Only that
    checkpoint's rows are accepted, so C can never be selected on a descendant's outcome. The file
    is write-once: refreezing different content raises. kind='base' freezes the separate
    descriptive base-correct cohort; it is never pooled with stage cohorts.
    """
    require(kind in COHORT_KINDS, 'unknown cohort kind')
    require(seed in (101, 102, 103) or (kind == 'base' and seed is None), 'cohort seed must be 101-103')
    require(isinstance(incoming, dict) and incoming.get('checkpoint_id') and is_sha(incoming.get('weights_sha256')),
            'incoming checkpoint identity required')
    require(is_sha(scores_sha256), 'incoming score hash required')
    scores = _score_map(score_rows, incoming['checkpoint_id'], column)
    require(bool(scores), 'no incoming score rows')
    doc = seal(dict(schema=COHORT_SCHEMA, kind=kind, family=family, seed=seed, panel=panel, panel_sha256=panel_sha256,
                    correct_column=column,
                    incoming=dict(checkpoint_id=incoming['checkpoint_id'], weights_sha256=incoming['weights_sha256']),
                    scores_sha256=scores_sha256, raw_sha256=raw_sha256, population=len(scores),
                    population_sha256=digest(sorted(scores)), population_ids=sorted(scores),
                    question_ids=sorted(q for q, v in scores.items() if v == 1),
                    length_convention_sha256=digest(LENGTH_CONVENTION)))
    path = Path(path)
    if path.exists():
        existing = json.loads(path.read_text())
        verify_seal(existing)
        require(existing['identity_sha256'] == doc['identity_sha256'], 'cohort already frozen with different content: ' + path.name)
        return existing
    write_json(path, doc)
    return doc


def load_cohort(path):
    doc = json.loads(Path(path).read_text())
    verify_seal(doc)
    require(doc.get('schema') == COHORT_SCHEMA and doc.get('kind') in COHORT_KINDS, 'not an SP cohort')
    return doc


# ---------------------------------------------------------------------------------------------
# Length metrics.

def token_count(row, cap=CAP):
    """(generated tokens, source) of one raw row, or None when the row is MISSING."""
    finish = row.get('finish_reason')
    if finish in MISSING_FINISHES or not isinstance(row.get('response'), str):
        return None
    require(finish in FINISHES, 'UNVERIFIED: unknown finish_reason %r for %s' % (finish, row.get('id')))
    if isinstance(row.get('token_ids'), list):
        n = len(row['token_ids'])
        if row.get('output_tokens') is not None:
            require(int(row['output_tokens']) == n, 'output_tokens disagrees with token_ids: ' + str(row.get('id')))
        source = 'token_ids'
    elif row.get('output_tokens') is not None:
        n, source = int(row['output_tokens']), 'output_tokens'
    else:
        return None
    require(0 <= n <= cap, 'generated tokens exceed the cap: ' + str(row.get('id')))
    return n, source


def _scope(ids, raw, counts, correct):
    n = len(ids)
    missing = sorted(i for i in ids if i not in counts or i not in correct)
    right = sum(correct[i] for i in ids) if all(i in correct for i in ids) else None
    base = dict(n=n, n_correct=right, accuracy=100.0 * right / n if n and right is not None else None, missing=len(missing))
    if missing:
        return dict(base, status='INCOMPLETE', total_tokens=None, mean_tokens=None, tokens_per_correct=None,
                    truncation_rate=None, truncated=None, undefined='missing raw/finish metadata', missing_ids=missing[:20])
    if n == 0:
        return dict(base, status='COMPLETE', total_tokens=0, mean_tokens=None, tokens_per_correct=None,
                    truncation_rate=None, truncated=0, undefined='empty scope')
    total = sum(counts[i] for i in ids)
    truncated = sum(raw[i]['finish_reason'] == 'length' for i in ids)
    return dict(base, status='COMPLETE', total_tokens=total, mean_tokens=total / n,
                tokens_per_correct=total / right if right else None, truncation_rate=truncated / n,
                truncated=truncated, undefined=None if right else 'zero correct')


def length_metrics(raw, correct, cohort=None, *, cap=CAP, eos_token_ids=None):
    """Full-panel and frozen-cohort length metrics for one checkpoint/panel cell.

    raw: generated rows (id, finish_reason, token_ids or output_tokens). correct: {id: 0/1} for
    every graded question in the cell. Graded questions with no usable raw row are MISSING and make
    their scopes INCOMPLETE. Raw rows for ungraded questions are foreign and refused.
    """
    rows = {}
    for r in raw:
        key = str(r['id'])
        require(key not in rows, 'duplicate response IDs')
        rows[key] = r
    correct = {str(k): int(v) for k, v in correct.items()}
    ids = sorted(correct)
    if cohort is not None:
        verify_seal(cohort)
        ids = cohort.get('population_ids')
        require(ids is not None and len(ids) == cohort['population'] and len(set(ids)) == len(ids) and
                digest(sorted(ids)) == cohort['population_sha256'], 'UNVERIFIED: frozen population identity missing or changed')
        require(set(correct) <= set(ids), 'UNVERIFIED: foreign grades outside frozen population')
    require(set(rows) <= set(ids), 'foreign raw outputs: questions outside registered population')
    counts, sources = {}, set()
    for key, r in rows.items():
        value = token_count(r, cap)
        if value is not None:
            counts[key], source = value
            sources.add(source)
    require(len(sources) <= 1, 'UNVERIFIED: mixed token-count sources in one cell')
    convention = dict(LENGTH_CONVENTION, cap=cap, source=sources.pop() if sources else None)
    eos = None
    if eos_token_ids is not None and convention['source'] == 'token_ids':
        ends = set(eos_token_ids)
        eos = sum(bool(rows[k]['token_ids']) and rows[k]['token_ids'][-1] in ends for k in counts)
    out = dict(convention=convention, convention_sha256=digest(convention), eos_terminated=eos,
               context_truncations=sum(rows[k]['finish_reason'] == 'length' and counts[k] < cap for k in counts),
               full=_scope(ids, rows, counts, correct), cohort=None)
    if cohort is not None:
        verify_seal(cohort)
        members = set(cohort['question_ids'])
        require(members <= set(ids), 'UNVERIFIED: cohort questions outside frozen population')
        scope = _scope(sorted(members), rows, counts, correct)
        scope.update(cohort_size=len(members), cohort_identity_sha256=cohort['identity_sha256'], cohort_kind=cohort['kind'])
        if not members:
            scope['undefined'] = 'empty cohort'
        out['cohort'] = scope
    out['status'] = 'INCOMPLETE' if out['full']['status'] == 'INCOMPLETE' else 'COMPLETE'
    return out


def same_convention(*metrics):
    """Length comparisons across checkpoints require one bound convention (missing-only cells excepted)."""
    hashes = {m['convention_sha256'] for m in metrics if m['convention']['source'] is not None}
    require(len(hashes) <= 1, 'UNVERIFIED: length conventions differ across checkpoints')
    return hashes.pop() if hashes else None


# ---------------------------------------------------------------------------------------------
# Cell IO (existing scorer outputs reused; nothing regenerated).

def read_scores_csv(path):
    with open(path, newline='') as stream:
        return list(csv.DictReader(stream))


def correct_column(panel, sub=None):
    require(panel not in NOT_APPLICABLE, 'N/A: %s has no generated length' % panel)
    require(panel in CORRECT_COLUMNS, 'no registered correctness column for panel %r' % panel)
    return CORRECT_COLUMNS[panel]


def cell_inputs(cell):
    """A cell names one checkpoint/panel's raw outputs and grades.

    {"raw": file, "scores": file, "panel": panel set, "panel_filter": optional sub-panel value}
    Returns (raw rows, score rows, {question_id: correct under the registered column}).
    """
    raw = read_jsonl(cell['raw']) if cell.get('raw') else []
    scores = read_scores_csv(cell['scores'])
    column = cell.get('correct_column') or correct_column(cell.get('panel', 'finqa'))
    if cell.get('panel_filter'):
        scores = [r for r in scores if r.get('panel') == cell['panel_filter']]
        keep = {str(r['question_id']) for r in scores}
        raw = [r for r in raw if str(r['id']) in keep]
    graded = _score_map(scores, None, column)
    return raw, scores, graded


# ---------------------------------------------------------------------------------------------
# GPU job: one checkpoint, every SP prompt, greedy thinking-off, cap 2,048.

def generate(model, panel, gpu, standin=False):
    """(raw rows, timing seconds {load, score}). Stand-in mode is CPU-only and synthetic."""
    if standin:
        from kit.sp_read import standin_responses
        return standin_responses(panel), dict(load=0.0, score=0.0)
    import os
    os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu), KIT_DETERMINISTIC='1', VLLM_BATCH_INVARIANT='1',
                      VLLM_ATTENTION_BACKEND='FLASH_ATTN')
    from kit import score_forgetting as general
    from transformers import AutoTokenizer
    from vllm import SamplingParams
    from kit.inference_lifecycle import create_llm, marker, shutdown_since
    started = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(str(model), local_files_only=True)
    prompts = [general.render(tokenizer, r['prompt']) for r in panel]
    mark = marker()
    try:
        engine = create_llm(model=str(model), enable_lora=False, **general.ENGINE, enforce_eager=True, seed=0)
        loaded = time.monotonic()
        outputs = engine.generate(prompts, SamplingParams(n=1, **general.DECODING))  # as m2_score's legal path
        scored = time.monotonic()
    finally:
        shutdown_since(mark)
    require(len(outputs) == len(panel), 'incomplete SP generation batch')
    rows = [dict(id=r['id'], response=o.outputs[0].text, token_ids=list(o.outputs[0].token_ids),
                 output_tokens=len(o.outputs[0].token_ids), prompt_tokens=len(o.prompt_token_ids),
                 finish_reason=o.outputs[0].finish_reason) for r, o in zip(panel, outputs)]
    return rows, dict(load=loaded - started, score=scored - loaded)


def verify_model(entry, model):
    """Every bound model file must hash to its inventory value; nothing is copied or pinned."""
    files = entry.get('model_file_hashes') or {}
    require(bool(files), 'UNVERIFIED: inventory entry carries no model file hashes')
    for name, expected in files.items():
        require(Path(name).name == name, 'non-portable model file name')
        require((Path(model) / name).is_file(), 'MISSING: durable checkpoint file absent: ' + name)
        require(sha256_file(Path(model) / name) == expected, 'UNVERIFIED: live model file differs from inventory: ' + name)
    require(digest(files) == entry['weights_sha256'], 'UNVERIFIED: inventory weights digest mismatch')


def verify_outputs(root, required=('identity.json', 'raw.jsonl', 'scores.jsonl', 'length.json')):
    """Verify every returned byte before using a cached score or aggregating a return."""
    root = Path(root)
    manifest = json.loads((root / 'manifest.json').read_text())
    require(set(required) <= set(manifest['files']), 'UNVERIFIED: incomplete SP output manifest')
    for name, expected in manifest['files'].items():
        relative = Path(name)
        require(not relative.is_absolute() and '..' not in relative.parts, 'unsafe output member')
        require((root / name).is_file() and sha256_file(root / name) == expected, 'UNVERIFIED: output changed: ' + name)
    return manifest


def observed_hardware(entry, expected):
    """Resolve UUID before CUDA initialization, then verify visible CUDA UUID and environment."""
    require(expected and expected.get('uuid') and expected.get('environment'), 'UNVERIFIED: assigned hardware missing')
    verify_seal(expected)
    require(expected.get('assignment') == entry['assignment'] and expected.get('hardware_sha256'), 'UNVERIFIED: hardware authority differs')
    torch = sys.modules.get('torch')
    require(torch is None or not torch.cuda.is_initialized(), 'UNVERIFIED: CUDA initialized before UUID resolution')
    from kit import m2_gpu, m2_assignment
    index = m2_gpu.prepare(expected['uuid'], expected_uuid=expected['uuid'])['index']
    observed = m2_assignment.inventory(index)
    require(all(observed[k] == expected[k] for k in ('uuid', 'environment')), 'UNVERIFIED: observed GPU/environment differs')
    m2_gpu.assert_current()
    return index, {k: observed[k] for k in ('uuid', 'environment')}


def panel_cohort_path(out, use):
    """Cohort keyed by the incoming edge, independent of aliases of identical weights."""
    incoming = use.get('incoming_key')
    source, family, arm, seed = (incoming[0], incoming[1], incoming[2], incoming[3]) if incoming else (
        use.get('source'), use.get('family'), use.get('arm', 'default'), use.get('seed'))
    return Path(out) / 'cohorts' / ('sp.%s.%s.s%s.%s.sp.json' % (source, family, seed, arm))


def panel_cohorts(out, entry, raw, scored, frozen_sha256):
    """Write incoming strict cohorts first; final cells can only consume that frozen population."""
    metrics = {}
    grades = {r['id']: r['strict_all_constraints_pass'] for r in scored}
    for use in entry.get('uses', []):
        name = use.get('comparison') or use.get('arm') or 'default'
        path = panel_cohort_path(out, use)
        co = None
        if use.get('role') == 'incoming':
            co = freeze_correct_cohort(path, family=str(use['family']) + '.' + name, seed=use['seed'], panel='sp',
                incoming=dict(checkpoint_id=entry['entry_id'], weights_sha256=entry['weights_sha256']),
                score_rows=[dict(question_id=r['id'], strict_all_constraints_pass=r['strict_all_constraints_pass']) for r in scored],
                scores_sha256=digest(scored), raw_sha256=digest(raw), panel_sha256=frozen_sha256,
                column='strict_all_constraints_pass')
        elif use.get('role') == 'final' and path.is_file():
            co = load_cohort(path)
        if co is not None:
            metrics[name] = length_metrics(raw, grades, co)
        elif use.get('role') == 'final':
            metrics[name] = dict(status='MISSING-INCOMING')
    return metrics


def panel_job(entry, model, gpu, out, frozen_root=None, standin=False, gpu_identity=None, local_export=None):
    """Score one inventory entry on the SP panel; idempotent per entry under {sp}/panel/<entry>."""
    out = Path(out)
    verified_started = time.monotonic()
    root = out / 'panel' / entry['entry_id']
    if frozen_root is None:
        from kit.sp_prepare import RESOURCES as frozen_root
    doc, panel, ck = load_panel(frozen_root)
    verify_fixtures(frozen_root, ck)
    observed = gpu_identity or dict(uuid=None, environment=None)
    if not standin:
        from kit.sp_manifest import durable_binding
        proof = durable_binding(local_export, verified=entry.get('durable_provenance'))
        require(proof == entry.get('durable_provenance') and Path(model).resolve() == Path(local_export['model']).resolve(),
                'UNVERIFIED: owner durable binding differs')
        verify_model(entry, model)
        gpu, observed = observed_hardware(entry, gpu_identity)
    request = dict(entry=entry, frozen_sha256=doc.get('identity_sha256') or digest(doc),
                   checker_sha256=doc.get('checker_sha256'), decoding=DECODING, decoding_sha256=digest(DECODING),
                   gpu_identity=observed, hardware_sha256=(gpu_identity or {}).get('hardware_sha256'), technical_synthetic=standin)
    if (root / 'manifest.json').exists():
        manifest = verify_outputs(root)
        previous = json.loads((root / 'identity.json').read_text())
        require(all(previous.get(k) == v for k, v in request.items()), 'UNVERIFIED: cached SP scoring identity differs')
        return manifest
    verification_seconds = time.monotonic() - verified_started
    reused = None
    for candidate in sorted((out / 'panel').glob('*/identity.json')):
        if not (candidate.parent / 'manifest.json').is_file():
            continue
        previous = json.loads(candidate.read_text())
        if (previous.get('entry', {}).get('weights_sha256') == entry['weights_sha256'] and
            all(previous.get(k) == request[k] for k in ('frozen_sha256', 'checker_sha256', 'decoding_sha256',
                                                       'gpu_identity', 'hardware_sha256', 'technical_synthetic'))):
            verify_outputs(candidate.parent)
            reused = candidate.parent
            break
    if reused is None:
        raw, timing = generate(model, panel, gpu, standin)
    else:
        raw, timing = read_jsonl(reused / 'raw.jsonl'), dict(load=0.0, score=0.0)
    timing['load'] += verification_seconds
    receipt_started = time.monotonic()
    scored = score_panel(panel, raw, ck)
    metrics = length_metrics(raw, {r['id']: r['strict'] for r in scored})
    metrics['comparisons'] = panel_cohorts(out, entry, raw, scored, request['frozen_sha256'])
    write_jsonl(root / 'raw.jsonl', raw)
    write_jsonl(root / 'scores.jsonl', scored)
    write_json(root / 'length.json', metrics)
    timing['receipt'] = time.monotonic() - receipt_started
    timing['serial_wall_seconds'] = sum(timing.values())
    write_json(root / 'identity.json', dict(request,
        timing=timing, gpu_hours=0.0 if standin else timing['serial_wall_seconds'] / 3600))
    manifest = dict(files={n: sha256_file(root / n) for n in ('raw.jsonl', 'scores.jsonl', 'length.json', 'identity.json')})
    write_json(root / 'manifest.json', manifest)
    return manifest


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('action', choices=('panel', 'cohort', 'length'))
    p.add_argument('--entry', type=Path); p.add_argument('--model', type=Path); p.add_argument('--gpu', default='0')
    p.add_argument('--gpu-identity', type=Path, help='JSON with the assigned physical GPU uuid and environment')
    p.add_argument('--models', type=Path, help='local owner export map; never returned')
    p.add_argument('--cell', type=Path); p.add_argument('--cohort', type=Path)
    p.add_argument('--frozen', type=Path)
    p.add_argument('--out', type=Path, required=True); p.add_argument('--standin', action='store_true')
    a = p.parse_args(argv)
    if a.action == 'panel':
        entry = json.loads(a.entry.read_text())
        identity = json.loads(a.gpu_identity.read_text()) if a.gpu_identity else None
        local = json.loads(a.models.read_text()).get(entry['entry_id']) if a.models else None
        panel_job(entry, a.model, a.gpu, a.out, a.frozen, a.standin, identity, local)
        return 0
    cell = json.loads(a.cell.read_text())
    raw, scores, correct = cell_inputs(cell)
    column = cell.get('correct_column') or correct_column(cell.get('panel', 'finqa'))
    if a.action == 'cohort':
        freeze_correct_cohort(a.out, family=cell['family'], seed=cell['seed'], panel=cell['panel'],
                              incoming=dict(checkpoint_id=cell['checkpoint_id'], weights_sha256=cell['weights_sha256']),
                              score_rows=scores, scores_sha256=sha256_file(cell['scores']),
                              raw_sha256=sha256_file(cell['raw']) if cell.get('raw') else None,
                              kind=cell.get('kind', 'incoming'), column=column)
        return 0
    cohort = load_cohort(a.cohort) if a.cohort else None
    if cohort is not None:
        require(cohort['family'] == cell['family'] and cohort['panel'] == cell['panel'] and
                (cohort['kind'] == 'base' or cohort['seed'] == cell['seed']), 'cohort belongs to another family/seed/panel')
    write_json(a.out, dict(cell={k: v for k, v in cell.items() if k not in ('raw', 'scores')},
                           metrics=length_metrics(raw, correct, cohort)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
