#!/usr/bin/env python3
"""Freeze the full-source C pool, then journal up to four gold-blind 27B attempts."""
from pathlib import Path
import sys
if __package__ in (None, ''): sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kit.c_common import *
from collections import Counter, defaultdict
import time
import re

PREFIX = 'package-c-train-v1|'
INSTRUCTION = 'Give complete worked reasoning using only the agreement. Finish with exactly one strict Answer: <label> line.'


def messages(item):
    return [dict(role='system', content=INSTRUCTION), dict(role='user', content=item['prompt'])]


def select_pool(splits, evaluation, tokenizer, limit=1024):
    owners = {field: defaultdict(set) for field in ('group', 'document_id')}
    for split, rows in splits.items():
        for row in rows:
            for field in owners: owners[field][row[field]].add(split)
    held = {f: {r[f] for r in evaluation if f in r} for f in owners}
    groups = defaultdict(list); excluded = []
    for row in splits['train']:
        reason = next(('cross_split_' + f for f in owners if len(owners[f][row[f]]) > 1), None)
        reason = reason or next(('heldout_' + f for f in owners if row[f] in held[f]), None)
        if not reason:
            pids, _ = t.tokenize_student(tokenizer, [dict(role='user', content=row['prompt'])], 'Answer: entailment')
            if len(pids) > 10240: reason = 'input_over_10240'
        if reason: excluded.append(dict(id=row['id'], reason=reason)); continue
        groups[row['group']].append(dict(row, input_tokens=len(pids)))
    for group in groups:
        groups[group].sort(key=lambda r: (legal.digest(PREFIX + legal.canonical(r['content'])), r['id']))
    ordered = sorted(groups, key=lambda g: (legal.digest(PREFIX + g), g))
    candidates = [groups[g][turn] for turn in range(4) for g in ordered if len(groups[g]) > turn]
    # Apply B's prompt guard before counting a selected pair; never use model outcomes.
    eligible, overlaps = overlap(candidates, evaluation)
    selected = eligible[:limit]
    excluded += overlaps
    return selected, excluded


def source_rows(source, standin):
    provenance = json.loads((C / 'contract_nli/manifest.json').read_text())['source']
    splits = {}; hypotheses = None
    for split in ('train', 'dev', 'test'):
        path = source / (split + '.json')
        if not standin: check(path, provenance['source_files_sha256'][path.name])
        doc = json.loads(path.read_text())
        require(hypotheses is None or hypotheses == doc['labels'], 'original hypotheses changed across splits')
        hypotheses = doc['labels']; rows = []
        require(standin or len(hypotheses) == 17, 'need original 17 hypotheses')
        for item in doc['documents']:
            require(len(item['annotation_sets']) == 1, 'unexpected annotation sets')
            annotations = item['annotation_sets'][0]['annotations']
            require(set(annotations) == set(hypotheses), 'missing original annotations')
            for hid, h in hypotheses.items():
                label = {'Entailment':'entailment','Contradiction':'contradiction','NotMentioned':'not_mentioned'}[annotations[hid]['choice']]
                content = dict(document=item['text'], hypothesis=h['hypothesis'], label=label)
                row = legal.make_item('contract_nli', split, f"{item['id']}:{hid}", content, legal.digest(legal.normalized(item['text'])))
                rows.append(dict(row, document_id=str(item['id']), hypothesis_id=hid))
        splits[split] = rows
    return splits


def verify(item, raw, tokenizer):
    text = raw.get('text', ''); reasons = []
    prediction = legal.parse_answer('contract_nli', text)
    if prediction != item['gold']: reasons.append('label_mismatch_or_invalid')
    lines = text.rstrip().splitlines()
    if not lines or lines[-1] != 'Answer: ' + item['gold']: reasons.append('strict_terminal_line_required')
    reasoning = '\n'.join(lines[:-1])
    if len(reasoning.split()) < 8 or re.search(r'</?think\b', text, re.I): reasons.append('incomplete_worked_reasoning')
    if raw.get('finish_reason') not in ('stop', 'eos'): reasons.append('unfinished')
    pids, tids = t.tokenize_student(tokenizer, [dict(role='user', content=item['prompt'])], text)
    if len(tids) > 2048: reasons.append('response_over_2048_including_eos')
    if len(pids) > 10240: reasons.append('input_over_10240')
    return dict(id=item['id'], attempt=raw['attempt'], verified=not reasons, reasons=reasons,
                text=text, response_tokens_including_eos=len(tids), prediction=prediction)


class Teacher(t.VLLMGenerator):
    """B's request generation, with room for full legal agreements in the 27B template."""
    def __init__(self, model):
        os.environ.update(VLLM_ENABLE_V1_MULTIPROCESSING='0', VLLM_BATCH_INVARIANT='1', VLLM_ATTENTION_BACKEND='FLASH_ATTN')
        os.environ.setdefault('VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS', '3000')
        import vllm
        require(vllm.__version__ == '0.18.0', 'teacher requires inference vLLM 0.18.0')
        from vllm import LLM, SamplingParams
        self.params = SamplingParams; self.input_cap = 13824; self.max_new_tokens = 2560
        started = time.monotonic()
        self.engine = LLM(model=str(model), tokenizer=str(model), tensor_parallel_size=2,
            distributed_executor_backend='mp', dtype='bfloat16', max_model_len=16384,
            max_num_seqs=32, gpu_memory_utilization=.90, seed=27101, enforce_eager=True,
            attention_config={'backend':'FLASH_ATTN'}, gdn_prefill_backend='triton')
        self.tokenizer = self.engine.get_tokenizer()
        self.reload_timing = dict(seconds=time.monotonic()-started)


def synthetic_rows():
    splits = dict(train=[], dev=[], test=[])
    for i in range(24):
        label = legal.LABELS['contract_nli'][i % 3]
        content = dict(document=f'Synthetic training agreement {i}.', hypothesis=f'Test synthetic term {i}.', label=label)
        row = legal.make_item('contract_nli', 'train', str(i), content, f'synthetic-document-{i}')
        splits['train'].append(dict(row, document_id='synthetic-'+str(i), hypothesis_id=str(i % 3)))
    return splits


def evaluation_items(a):
    from kit.score_forgetting import load_panel
    general, _ = load_panel()
    evaluation = [r for values in panels().values() for r in values] + general
    finqa_root = getattr(a, 'finqa_panel', None)
    require(a.standin or finqa_root is not None, 'frozen FinQA evaluation root required for overlap checks')
    if finqa_root is not None:
        from kit import eval_bed as bed
        from kit.v4_contract import check_eval_items
        from types import SimpleNamespace
        finqa = bed.items_of(bed.load_bed('finqa'), SimpleNamespace(root=str(finqa_root),split='test',limit=None,heldout_n=None,allow_subset=False))
        check_eval_items(finqa,'finqa','test')
        require(len(finqa)==1147, 'FinQA overlap panel must contain all 1147 questions')
        evaluation += finqa
    return evaluation


def job(a):
    protocol(); root = a.out / 'targets'
    with lock(root):
        require(a.standin or (a.source and a.tokenizer), 'offline source directory and pinned tokenizer required')
        if a.standin:
            from kit.c_sim import Tokenizer
            tokenizer = Tokenizer(); identities = dict(technical_synthetic=True)
        else:
            from transformers import AutoTokenizer
            identities = dict(tokenizer=identity(a.tokenizer, 'student'))
            tokenizer = AutoTokenizer.from_pretrained(a.tokenizer, local_files_only=True)
        evaluation = evaluation_items(a)
        inputs = dict(evaluation_prompts=s.digest([dict(id=r['id'],prompt=r['prompt']) for r in evaluation]), protocol=protocol(), technical_synthetic=a.standin,
                      source={p.name:s.file_hash(p) for p in sorted(a.source.glob('*.json'))} if a.source else 'synthetic',
                      tokenizer=identities, general=s.file_hash(KIT / 'panels/general-v1.jsonl'))
        bind(root / 'inputs.json', inputs)
        if not (root / 'prepared/manifest.json').exists():
            splits = source_rows(a.source, a.standin) if a.source else synthetic_rows()
            selected, excluded = select_pool(splits, evaluation, tokenizer)
            prep = root / 'prepared'; prep.mkdir(exist_ok=True)
            t.write_jsonl(prep / 'train.jsonl', selected); t.write_jsonl(prep / 'exclusions.jsonl', excluded)
            atomic(prep / 'manifest.json', files_manifest(prep, ['train.jsonl', 'exclusions.jsonl'],
                source_counts={k:len(v) for k,v in splits.items()}, selected=len(selected), requested=1024,
                shortfall=max(0,1024-len(selected)), labels=dict(Counter(r['gold'] for r in selected)),
                hypotheses=dict(Counter(r['hypothesis_id'] for r in selected)), documents=dict(Counter(r['group'] for r in selected)),
                technical_synthetic=a.standin))
        frozen_files(root / 'prepared')
        if a.prepare_only: print('Frozen training membership; generation not started'); return
        require(a.standin or len(set(a.gpu.split(','))) == 2, 'teacher uses two distinct GPUs')
        observed = dict(technical_synthetic=True) if a.standin else identity(a.model, 'teacher')
        bind(root / 'teacher-identity.json', observed)
        if (root / 'manifest.json').exists(): frozen_files(root); print('SKIP: common C targets'); return
        items = t.read_jsonl(root / 'prepared/train.jsonl'); require(items, 'zero eligible training items')
        accepted = []; rejected = []; raw_dir = root / 'teacher/raw'; raw_dir.mkdir(parents=True, exist_ok=True)
        generator = None; started = time.monotonic(); attempt_count = 0
        try:
            for attempt, (seed, temp) in enumerate(t.SCHEDULE, 1):
                remaining = [r for r in items if r['id'] not in {x['id'] for x in accepted}]
                for offset in range(0, len(remaining), 32):
                    batch = remaining[offset:offset+32]; pending = []; raws = {}
                    for item in batch:
                        path = raw_dir / f"{t.sha(item['id'])}-{attempt}.json"
                        request = dict(id=item['id'], attempt=attempt, seed=seed, temperature=temp, top_p=1, messages=messages(item))
                        if path.exists():
                            raw = json.loads(path.read_text()); require(raw['request'] == request, 'teacher journal request changed'); raws[item['id']] = raw
                        else: pending.append((item, path, request))
                    if pending:
                        if a.standin:
                            outputs = []
                            for item, _, _ in pending:
                                # Exercise rejection, later acceptance, and exhaustion in CPU rehearsal.
                                good = int(item['source_id']) % 4 != 0 or attempt == 2
                                if a.standin_zero: good = False
                                text = 'The agreement and hypothesis have been compared using the stated terms.\nAnswer: ' + item['gold'] if good else 'unfinished'
                                outputs.append(dict(text=text, finish_reason='stop', teacher_tokens=len(text.split())))
                        else:
                            if generator is None:
                                os.environ['CUDA_VISIBLE_DEVICES'] = a.gpu; generator = Teacher(a.model)
                            outputs = generator.generate_many([r for _,_,r in pending])
                        require(len(outputs) == len(pending), 'incomplete teacher batch')
                        # Entire returned batch durable before verifying any member.
                        for (_, path, request), output in zip(pending, outputs):
                            raw = dict(output, attempt=attempt, request=request, technical_synthetic=a.standin)
                            atomic(path, raw); raws[request['id']] = raw; attempt_count += 1
                    for item in batch:
                        checked = verify(item, raws[item['id']], tokenizer)
                        (accepted if checked['verified'] else rejected).append(checked)
        finally:
            costs = root / 'teacher/costs'; costs.mkdir(exist_ok=True)
            atomic(costs / f'{time.time_ns()}.json', dict(kind='teacher', new_attempts=attempt_count,
                gpu_hours=0 if a.standin else 2*(time.monotonic()-started)/3600, technical_synthetic=a.standin))
            t.write_jsonl(root / 'teacher/rejections.jsonl', rejected)
            t.write_jsonl(root / 'teacher/accepted.jsonl', accepted)
        journal = [json.loads(p.read_text()) for p in sorted(raw_dir.glob('*.json'))]
        atomic(root / 'coverage.json', dict(population=len(items), accepted=len(accepted), coverage=len(accepted)/len(items),
            teacher_attempts=len(journal),teacher_output_tokens=sum(r.get('teacher_tokens',0) for r in journal),
            teacher_input_tokens=sum(r.get('input_tokens',0) for r in journal),
            accepted_student_response_tokens_including_eos=sum(r['response_tokens_including_eos'] for r in accepted),
            per_label={label:dict(total=sum(r['gold']==label for r in items), accepted=sum(next(x for x in items if x['id']==r['id'])['gold']==label for r in accepted)) for label in legal.LABELS['contract_nli']}))
        if not accepted: raise NotReady('zero valid teacher targets: training input undefined; no replacement subset')
        lookup = {r['id']:r for r in items}
        rows = [dict(id=r['id'], prompt=[dict(role='user', content=lookup[r['id']]['prompt'])], response=r['text'],
                     gold=lookup[r['id']]['gold'], group=lookup[r['id']]['group'], task='contract_nli') for r in accepted]
        t.write_jsonl(root / 'pool.jsonl', rows)
        names = ['inputs.json', 'teacher-identity.json', 'prepared/manifest.json', 'pool.jsonl', 'coverage.json', 'teacher/accepted.jsonl', 'teacher/rejections.jsonl']
        coverage = {}
        for seed in SEEDS:
            order = cycle(rows, seed, 1280); name = f'schedule-{seed}.jsonl'; t.write_jsonl(root / name, order); names.append(name)
            coverage[str(seed)] = dict(unique=len({r['id'] for r in order}), repetitions=1280-len({r['id'] for r in order}),
                                       labels=dict(Counter(r['gold'] for r in order)))
        names += [str(p.relative_to(root)) for p in sorted(raw_dir.glob('*.json'))]
        atomic(root / 'manifest.json', files_manifest(root, names, technical_synthetic=a.standin, valid=len(rows), schedules=coverage,
               protocol=protocol(), response_cap_including_eos=2048, attempts=t.SCHEDULE))
        print('Accepted', len(rows), '/', len(items), '; five shared 40x32 schedules frozen')


def configure(p):
    device(p, '0,1'); p.add_argument('--tokenizer'); p.add_argument('--source', type=Path); p.add_argument('--finqa-panel', type=Path)
    p.add_argument('--prepare-only', action='store_true'); p.add_argument('--standin-zero', action='store_true')
if __name__ == '__main__': raise SystemExit(entry(job, configure, 'c-targets'))
