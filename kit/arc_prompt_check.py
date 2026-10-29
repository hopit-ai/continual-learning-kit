"""Container-safe ARC diagnostic; reuse the broad builder's generation and checks."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

from kit import next_pools as pools
from kit.next_hellaswag import strict_label
from kit.smoke_steps import Steps
from kit.v4_contract import INITIAL_8B_REVISION

MODEL = 'Qwen/Qwen3-8B'
COUNT = 128
GENERATION=dict(seed=0,temperature=0,top_p=1,max_tokens=2048,thinking=False,generator='kit.v4_teacher.VLLMGenerator',tensor_parallel_size=1,allow_batch_fallback=False)
SCHEMA = 'arc-prompt-check.v1'
METRICS = dict(
    parse_rate='strict_label returned a label with stop/eos completion / 128; missing/truncated count against 128',
    correct_rate='unchanged next_pools.correct returned True / 128 (including length-stopped outputs)',
    accepted_rate='correct, finish_reason stop/eos, and unchanged encode_example accepted / 128',
    passed='all 128 requests completed and shutdown, >=127 strict parseable completed answers and >=80 accepted rows')
# Include the full kit Python closure, including lazy verifier/tokenizer imports.
KIT = Path(__file__).resolve().parent


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()


def code_hashes():
    return {str(p.relative_to(KIT)): sha(p.read_bytes()) for p in sorted(KIT.rglob('*.py'))}


def order_key(row):
    return (sha(('next-v4-broad-v1|arc|' + row['order_id']).encode()), row['order_id'])


def check_rows(rows):
    if len(rows) < COUNT or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('need at least 128 unique frozen ARC candidates')
    if rows != sorted(rows, key=order_key):
        raise ValueError('ARC candidates changed the builder frozen order')
    for row in rows:
        labels = row['labels']
        if (row['source'] != 'arc' or not row['id'].startswith('arc-c-train-') or
                row['id'] != 'arc-c-train-' + row['order_id'] or
                not labels or len(set(labels)) != len(labels) or
                row['gold'] not in labels or len(row['gold']) != 1 or
                not row['prompt'].endswith(pools.ARC_PROMPT) or 'fake_response' in row):
            raise ValueError('not real fixed-prompt ARC candidates')


def build_plan(candidates):
    """Freeze the actual post-exclusion candidate list, never reconstruct its pool."""
    candidates = Path(candidates)
    raw = candidates.read_bytes()
    inputs_raw = candidates.with_name('inputs.json').read_bytes()
    inputs = json.loads(inputs_raw)
    if (inputs.get('pool_protocol') != pools.POOL_PROTOCOL or
            inputs.get('arc_prompt') != pools.ARC_PROMPT or
            inputs.get('technical_synthetic') is not False or
            inputs.get('revisions', {}).get('arc') != pools.ARC):
        raise ValueError('require real fixed-prompt pools/candidates.jsonl and inputs.json')
    manifest = candidates.with_name('manifest.json')
    if manifest.exists():
        doc = json.loads(manifest.read_bytes())
        if doc['files'][candidates.name] != sha(raw):
            raise ValueError('frozen candidates hash mismatch')
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    arc = [row for row in rows if row['source'] == 'arc']
    check_rows(arc)
    selected = arc[:COUNT]
    return dict(schema=SCHEMA, model=MODEL, revision=INITIAL_8B_REVISION,
                pool_protocol=pools.POOL_PROTOCOL, arc_prompt=pools.ARC_PROMPT,
                selection='first 128 ARC rows of the frozen post-exclusion candidate list',
                candidates_path=str(candidates.resolve()), candidates_sha256=sha(raw),
                inputs_sha256=sha(inputs_raw), inputs=inputs, arc_candidates=len(arc),
                rows=selected, sample_sha256=sha(canonical(selected)), kit_code_sha256=code_hashes())


def validate_plan(plan):
    if (plan['schema'] != SCHEMA or plan['model'] != MODEL or
            plan['revision'] != INITIAL_8B_REVISION or
            plan['pool_protocol'] != pools.POOL_PROTOCOL or plan['arc_prompt'] != pools.ARC_PROMPT or
            len(plan['rows']) != COUNT or sha(canonical(plan['rows'])) != plan['sample_sha256'] or
            plan['kit_code_sha256'] != code_hashes()):
        raise ValueError('selection/model/prompt/code changed after freezing')
    check_rows(plan['rows'])


def run(plan, work):
    validate_plan(plan)
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    from kit.v4_teacher import VLLMGenerator
    from kit.v4_sft import encode_example
    from kit.inference_lifecycle import marker, shutdown_since

    work = Path(work)
    work.mkdir(parents=True, exist_ok=False)
    os.environ.update(CUDA_VISIBLE_DEVICES='0', V4_PHASE0_INFERENCE='0')
    os.environ.pop('V4_BATCH_FALLBACK_CAUSE', None)
    from kit.m2_gpu import prepare, assert_current
    gpu_mapping=prepare('0')
    assert_current()
    model = snapshot_download(repo_id=MODEL, revision=INITIAL_8B_REVISION, local_dir=str(work/'model'))
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    mark = marker()
    outcomes, rejected, parse_rejected = [], [], []
    try:
        # Exactly next_pools.job: serial requests, same class and constructor.
        generator = VLLMGenerator(model, 2048, tensor_parallel_size=1, allow_batch_fallback=False)
        for index, row in enumerate(plan['rows']):
            messages = [dict(role='user', content=row['prompt'])]
            output = generator(messages, 1, seed=0, temperature=0, top_p=1)
            text = output['text']
            parsed = strict_label(text, row['labels'])
            correct = pools.correct(row, text)
            accepted = output['finish_reason'] in ('stop', 'eos') and correct
            reasons = []
            if parsed is None:
                reasons.append('strict_parser')
            elif not correct:
                reasons.append('wrong_answer')
            if output['finish_reason'] not in ('stop', 'eos'):
                reasons.append('finish_reason:' + output['finish_reason'])
            encoding_valid=False
            if accepted:
                try:
                    encode_example(tokenizer, messages, text)
                    encoding_valid=True
                except ValueError:
                    accepted = False
                    reasons.append('encode_example')
            outcome = dict(id=row['id'], gold=row['gold'], parsed=parsed, correct=correct,response=text,encoding_valid=encoding_valid,
                           accepted=accepted, finish_reason=output['finish_reason'],
                           response_sha256=sha(text.encode()), reasons=reasons,
                           input_tokens=output.get('input_tokens'), output_tokens=output.get('teacher_tokens'),
                           rendered_prompt_sha256=output.get('rendered_prompt_sha256'))
            outcomes.append(outcome)
            if not accepted:
                example = dict(outcome, prompt=row['prompt'], response=text)
                if len(rejected) < 8:
                    rejected.append(example)
                if parsed is None and len(parse_rejected) < 8:
                    parse_rejected.append(example)
            if (index + 1) % 16 == 0:
                print(f'[progress] ARC {index + 1}/{COUNT}', flush=True)
        counts = dict(parse_count=sum(r['parsed'] is not None and r['finish_reason'] in ('stop','eos') for r in outcomes),
                      correct_count=sum(r['correct'] for r in outcomes),
                      accepted_count=sum(r['accepted'] for r in outcomes))
        from kit.m2_format import policy
        thresholds=policy()['arc']
        passed=counts['parse_count']>=thresholds['min_strict_parseable'] and counts['accepted_count']>=thresholds['min_accepted']
        return dict(completed=True,passed=passed,error=None if passed else 'frozen ARC acceptance thresholds failed',n=COUNT,**counts,
                    parse_rate=counts['parse_count']/COUNT, correct_rate=counts['correct_count']/COUNT,
                    accepted_rate=counts['accepted_count']/COUNT, outcomes=outcomes,
                    rejected_examples=rejected, parse_rejected_examples=parse_rejected,
                    batch_policy=generator.batch_policy,gpu_mapping=gpu_mapping,gpu_uuid_asserted=True)
    finally:
        # Close vLLM while progress/record stdout is still open.
        shutdown_since(mark)


def main(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--selection', type=Path)
    parser.add_argument('--work', type=Path, default=Path('/tmp/arc-prompt-check'))
    parser.add_argument('--import-check', action='store_true')
    args = parser.parse_args(argv)
    if args.import_check:
        from kit.v4_teacher import VLLMGenerator
        from kit.v4_sft import encode_example
        assert callable(VLLMGenerator) and callable(encode_example)
        print('[import-check] ARC OK', flush=True)
        return 0
    if args.selection is None:
        parser.error('--selection required')
    steps = Steps({})
    steps.plan(['arc-base-128'])
    started = time.monotonic()
    plan, result = {}, {}
    try:
        raw = args.selection.read_bytes()
        plan = json.loads(raw)
        result = run(plan, args.work)
        result['selection_sha256'] = sha(raw)
    except Exception as error:
        result = dict(passed=False, completed=False, error=type(error).__name__ + ': ' + str(error),
                      n=None, parse_rate=None, correct_rate=None, accepted_rate=None)
    steps.record.append(dict(name='arc-base-128', returncode=0 if result['passed'] else 1,
                             seconds=round(time.monotonic()-started, 1), tail=[],
                             errors=[] if result['passed'] else [result['error']]))
    from kit.m2_format import POLICY
    steps.done(schema=SCHEMA, diagnostic_only=True, scientific_evidence=False,
               generation=GENERATION,format_validity_sha256=sha(POLICY.read_bytes()),
               metric_definitions=METRICS,
               model=MODEL, model_revision=INITIAL_8B_REVISION,
               candidates_sha256=plan.get('candidates_sha256'), sample_sha256=plan.get('sample_sha256'),
               candidate_ids=[r['id'] for r in plan.get('rows', [])],
               inputs_sha256=plan.get('inputs_sha256'), kit_code_sha256=plan.get('kit_code_sha256'), **result)
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
