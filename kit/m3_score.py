#!/usr/bin/env python3
"""Message-3 scoring on one dedicated physical GPU per seed, shared by C0/K/G/CR, base and every incoming checkpoint.

Actions (isolated M3 root only; never message-2's scoring-assignment.json or output tree):
  assign              freeze m3-scoring-assignment.json from the M3 inventory, refusing core-held nodes/GPUs
  validate-hellaswag  format-free HellaSwag likelihood validation on this seed's assignment
  validate-knowledge  format-free knowledge likelihood validation on this seed's assignment
                      (both exactly as the HellaSwag rule: frozen base + one incoming checkpoint, C0 stage-1 final of
                      seed 101; unanimous choices or the 1 Nov strict near-tie rule, else PENDING)
  score               one checkpoint x panel, through the inherited scorers, on the assigned GPU
  alias               step-0 (incoming) unit = the preceding final / base on the same assignment; no rescoring
The HellaSwag-100 panel is the tier's own freeze (kit/next_hellaswag.py --action prepare in the M3 root).
Likelihood units are scored as soon as their checkpoint exists; the reader counts them only after the seed's
validation for that panel has PASSED.
Panels: finqa, test (legal), math_hard, general-v1 (instruction/maths; generated-letter knowledge diagnostic only),
hellaswag (format-free likelihood) and knowledge-ll (format-free length-normalised likelihood, registration v2).
The assignment uses the message-2 v2 schema so kit/m2_assignment.active/verify enforce it unchanged.
"""
from pathlib import Path
import sys
if __package__ in (None, ''): sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kit.p3b_common import (KIT, NotReady, atomic, bind, files_manifest, frozen_files, require, s, t)
from kit import m3_schedule as plan
from types import SimpleNamespace as NS
import json
import math
import os
import re
import time
from contextlib import contextmanager
from functools import wraps

FAMILY = 'm3'
ASSIGNMENT = 'm3-scoring-assignment.json'
PANELS = ('finqa', 'test', 'math_hard', 'general-v1', 'hellaswag', 'knowledge-ll')
LIKELIHOOD_PANELS = ('hellaswag', 'knowledge-ll')
KNOWLEDGE_POLICY = KIT / 'resources/m3/knowledge-likelihood.json'
GENERAL = KIT / 'panels/general-v1.jsonl'
DECODING = dict(seed=0, temperature=0, thinking=False, max_new_tokens=2048, eager=True, batch_invariant=True)


def unit_arm(arm, stage):
    return 'base' if arm == 'base' else f'{arm}.{stage}'


def unit_root(out, arm, seed, stage, step, panel):
    """Inherited scorer layout: scores/<arm>-s<seed>-k<step>-<panel>, with arm '<arm>.<stage>' (or 'base')."""
    return Path(out) / 'scores' / f'{unit_arm(arm, stage)}-s{seed}-k{step}-{panel}'


# ----------------------------------------------------------------------------- assignment

def build_assignment(nodes, message2_nodes=(), core_uuids=(), *, standin=False):
    """One dedicated GPU UUID/environment per seed; every arm and constituent of that seed scores there.
    Nodes or GPUs held by message-2 core assignments are refused."""
    from kit.m3_manifest import host_name
    nodes = [dict(n, node=host_name(n['node'])) for n in nodes]
    message2_nodes = [host_name(n) for n in message2_nodes]
    require(nodes and all(n.get('node') and n.get('gpus') for n in nodes), 'M3 hardware inventory incomplete')
    require(len({n['node'] for n in nodes}) == len(nodes), 'duplicate node')
    require(not ({n['node'] for n in nodes} & set(message2_nodes)), 'M3 scoring must not use message-2 core nodes')
    gpus = [(n['node'], n['gpus'][i]) for i in range(max(len(n['gpus']) for n in nodes)) for n in nodes if i < len(n['gpus'])]
    require(len({g['uuid'] for _, g in gpus}) == len(gpus), 'physical UUID appears twice')
    require(not ({g['uuid'] for _, g in gpus} & set(core_uuids)), 'M3 inventory contains a GPU held by a core assignment')
    require(standin or len(gpus) >= len(plan.SEEDS), 'scientific assignments require three distinct physical GPUs')
    values = {}
    for index, seed in enumerate(plan.SEEDS):
        node, g = gpus[index % len(gpus)]
        values[f'{FAMILY}|{seed}'] = dict(node=node, uuid=g['uuid'], environment=g['environment'])
    return dict(schema='message2-scoring-assignment.v2', assignments=values, purpose='message3-k-cr.v2')


def assignment_path(out):
    return Path(out) / ASSIGNMENT


def bind_environment(out, seed):
    """Scorers read the assignment through M2_ASSIGNMENT/M2_FAMILY/M2_FAMILY_SEED (kit/m2_assignment.active)."""
    path = assignment_path(out)
    if not path.exists(): raise NotReady('freeze the M3 scoring assignment first (m3_score.py --action assign)')
    os.environ.update(M2_ASSIGNMENT=str(path), M2_FAMILY=FAMILY, M2_FAMILY_SEED=str(seed))
    return path


def assign_job(a):
    from kit import m2_assignment
    path = assignment_path(a.out)
    if path.exists():
        doc = m2_assignment.load(path)
        require(a.standin or len({v['uuid'] for v in doc['assignments'].values()}) >= len(plan.SEEDS), 'scientific assignments require three distinct physical GPUs')
        print('SKIP: frozen M3 assignment'); return
    inventory = json.loads(a.inventory.read_text())
    if not isinstance(inventory, dict): inventory = dict(nodes=inventory)
    core_nodes = set(inventory.get('message2_nodes', [])); core_uuids = set()
    if inventory.get('message2_assignment'):
        core = json.loads(Path(inventory['message2_assignment']).read_text())
        core_nodes |= {v['node'] for v in core['assignments'].values()}; core_uuids |= {v['uuid'] for v in core['assignments'].values()}
    print(m2_assignment.freeze(path, build_assignment(inventory['nodes'], core_nodes, core_uuids, standin=a.standin)))


@contextmanager
def normalized_assignment_inventory():
    """Normalize the inherited platform.node() inventory through an M3-only process-local wrapper."""
    from kit import m2_assignment
    from kit.m3_manifest import host_name
    original = m2_assignment.inventory
    def inventory(*args, **kwargs):
        value = dict(original(*args, **kwargs)); value['node'] = host_name(value['node']); return value
    m2_assignment.inventory = inventory
    try: yield
    finally: m2_assignment.inventory = original


def proof_for(out, gpu, seed, standin):
    from kit import m2_assignment
    bind_environment(out, seed)
    with normalized_assignment_inventory():
        proof = m2_assignment.active(out, gpu, standin, family=FAMILY, seed=seed)
    require(proof is not None and proof['family'] == FAMILY and proof['seed'] == seed, 'UNVERIFIED: off-assignment M3 scoring')
    return proof


def resolved_gpu(a):
    """active() already resolved and prepared the physical assignment; never reset it to the CLI default."""
    if not a.standin:
        from kit.m2_gpu import assert_current
        a.gpu = os.environ['CUDA_VISIBLE_DEVICES']
        assert_current()
    return a.gpu


@contextmanager
def gpu_phase(a, kind, **identity):
    start = time.time(); complete = False
    try:
        yield
        complete = True
    finally:
        end = time.time()
        atomic(Path(a.out) / 'gpu-costs' / (str(time.time_ns()) + '.json'),
               dict(kind=kind, seed=a.seed, started_at=start, ended_at=end, wall_seconds=end-start,
                    gpu_hours=0 if a.standin else (end-start)/3600, completed=complete,
                    technical_synthetic=a.standin, **identity))


def costed_score(fn):
    @wraps(fn)
    def wrapped(a):
        with normalized_assignment_inventory(), gpu_phase(a, 'm3_scoring_attempt', arm=a.arm, stage=a.stage, step=a.step, panel=a.panel_set):
            return fn(a)
    return wrapped


# ----------------------------------------------------------------------------- HellaSwag

VALIDATION_CHECKPOINTS = (('base', ('base', 101, 'finqa', 0)), ('C0-s101-finqa-k40', ('C0', 101, 'finqa', 40)))
VALIDATION_FILE = {'hellaswag': '{family}-s{seed}.json', 'knowledge-ll': 'knowledge-{family}-s{seed}.json'}


def validation_path(out, panel, family, seed):
    return Path(out) / 'format-validation' / VALIDATION_FILE[panel].format(family=family, seed=seed)


def validation_checkpoints(out):
    """The HellaSwag rule's set: frozen base plus one incoming checkpoint (C0 stage-1 final, seed 101)."""
    from kit.m3_train import record
    return {name: record(out, *key) for name, key in VALIDATION_CHECKPOINTS}


def validate(a, panel, rows, production_fn, reference_fn, policy_sha256, declaration):
    """One validation record per seed's assignment; PASS under the 1 Nov near-tie amendment; original scorer choices are unchanged."""
    from kit.m2_format import validation
    from kit.m3_train import base_record
    if panel=='hellaswag':
        from kit.likelihood_validation import lengths
        require({r['id'] for r in rows}<=set(lengths()),'configuration error: HellaSwag panel IDs missing from validation length table')
    proof = proof_for(a.out, a.gpu, a.seed, a.standin)
    resolved_gpu(a)
    checkpoints = validation_checkpoints(a.out); base = base_record(a.out)
    path = validation_path(a.out, panel, proof['family'], proof['seed'])
    identity = dict(policy_sha256=policy_sha256, assignment=proof, items_sha256=s.digest(rows),
                    checkpoints={name: dict(model=c['model'], model_file_hashes=c['identity'].get('model_file_hashes')) for name, c in checkpoints.items()})
    from kit.likelihood_validation import matches_check,RULE
    reused={}
    if path.exists():
        old=json.loads(path.read_text())
        require(all(old[k]==v for k,v in identity.items()),panel+' validation declaration changed')
        require(old.get('status') in ('PASS','FAIL','PENDING'),'unknown validation status')
        for check in old.get('checks',[]):
            if 'production' not in check or 'reference' not in check:continue
            result=validation(check['production'],check['reference'],[r['id'] for r in rows])
            if result['complete_item_choice_coverage'] and result['finite_likelihoods']:
                require(check['checkpoint'] in checkpoints and matches_check(check,result,legacy=True),'stored validation diagnostics differ')
                require(check['checkpoint'] not in reused,'duplicate stored validation checkpoint')
                reused[check['checkpoint']]=dict(check,**result)
        if old.get('validation_rule')==RULE and old['status']=='PENDING' and any(c['status']!='PASS' for c in reused.values()):
            require(False,panel+' reference disagreement PENDING; rerun forbidden')
        if old.get('validation_rule')==RULE and old['status']=='PASS':
            require(set(reused)==set(checkpoints) and all(c['status']=='PASS' for c in reused.values()),'stored validation PASS differs')
            return
        # Derive new diagnostics on CPU, preserving the exact original receipt.
        archive=path.parent/'attempts'/path.stem/('legacy-'+str(time.time_ns())+'.json')
        archive.parent.mkdir(parents=True,exist_ok=True);path.rename(archive)
        if any(c['status']!='PASS' for c in reused.values()):
            bind(path,dict(old,status='PENDING',checks=list(reused.values()),validation_rule=RULE,previous_receipt_sha256=s.file_hash(archive)))
            require(False,panel+' legacy reference disagreement PENDING; rerun forbidden')
    checks = []
    for name, checkpoint in checkpoints.items():
        if name in reused:
            checks.append(reused[name]);continue
        try:
            if a.standin: production = reference = standin_likelihoods(rows, name)
            else:
                checked(a.out, *dict(VALIDATION_CHECKPOINTS)[name], a.standin)
                with gpu_phase(a, 'm3_likelihood_validation', panel=panel, checkpoint=name, phase='production'):
                    production = production_fn(checkpoint['model'], rows, base['model'])
                with gpu_phase(a, 'm3_likelihood_validation', panel=panel, checkpoint=name, phase='reference'):
                    reference = reference_fn(str(checkpoint['model']), rows, str(base['model']))
            result = validation(production, reference, [r['id'] for r in rows])
            require(result['complete_item_choice_coverage'] and result['finite_likelihoods'], 'incomplete/nonfinite likelihood results; retry after technical repair')
        except Exception as error:
            atomic(path.parent / 'attempts' / path.stem / (str(time.time_ns()) + '.json'),
                   dict(identity, status='ERROR', checkpoint=name, checks=checks, error_type=type(error).__name__, technical_synthetic=a.standin))
            raise
        check = dict(checkpoint=name, **result, production=production, reference=reference)
        if result['status']!='PASS':
            checks.append(check)
            bind(path,dict(identity,status='PENDING',checks=checks,declaration=declaration,technical_synthetic=a.standin,validation_rule=RULE))
            require(False,panel+' independent reference disagreement PENDING; rerun forbidden')
        checks.append(check)
    bind(path,dict(identity,status='PASS',checks=checks,declaration=declaration,technical_synthetic=a.standin,validation_rule=RULE))


def validate_hellaswag(a):
    from kit import m2_format as formatting
    from kit import hellaswag_reference
    rows = formatting.items(a.out, a.standin)
    validate(a, 'hellaswag', rows, formatting.likelihoods, hellaswag_reference.score, s.file_hash(formatting.POLICY), formatting.policy())


def validation_status(out, panel, seed, standin=False, notes=None):
    """Recompute agreement and every frozen identity; a bare or synthetic PASS cannot authorize a scientific read."""
    from kit import m2_assignment, m2_format
    path = validation_path(out, panel, FAMILY, seed)
    if not path.exists(): return None
    try:
        value = json.loads(path.read_text())
        from kit.likelihood_validation import matches_check,RULE
        legacy=all('validation_rule' not in c for c in value.get('checks',[]))
        require((value.get('status')=='PASS' or legacy and value.get('status') in ('FAIL','PENDING')) and bool(value.get('technical_synthetic'))==standin,'validation provenance/status differs')
        raw = assignment_path(out).read_bytes()
        m2_assignment.verify(dict(assignment=value['assignment']), raw)
        require((value['assignment']['family'], value['assignment']['seed']) == (FAMILY, seed), 'validation family/seed differs')
        policy = KNOWLEDGE_POLICY if panel == 'knowledge-ll' else m2_format.POLICY
        require(value['policy_sha256'] == s.file_hash(policy), 'validation policy differs')
        rows = knowledge_items() if panel == 'knowledge-ll' else m2_format.items(Path(out), standin)
        require(len(rows) == 100 and value['items_sha256'] == s.digest(rows), 'validation items differ')
        checkpoints = validation_checkpoints(Path(out))
        require(set(value['checkpoints']) == set(checkpoints), 'validation checkpoint coverage differs')
        for name, c in checkpoints.items():
            require(value['checkpoints'][name]['model_file_hashes'] == c['identity']['model_file_hashes'], 'validation checkpoint weights differ')
        checks = value.get('checks', [])
        require(len(checks) == len(checkpoints) and {c['checkpoint'] for c in checks} == set(checkpoints), 'reference checkpoint coverage differs')
        for check in checks:
            result = m2_format.validation(check['production'], check['reference'], [r['id'] for r in rows])
            require(result['status']=='PASS' and matches_check(check,result,legacy=legacy),'reference agreement evidence differs')
        if legacy and value.get('status') in ('FAIL','PENDING') and notes is not None:
            notes.append('Legacy '+value['status']+' receipt re-evaluated PASS under '+RULE+'; stored receipt unchanged; gaps recomputed on CPU')
        return s.file_hash(path)
    except (KeyError, ValueError, TypeError, RuntimeError, OSError): return None


# ----------------------------------------------------------------------------- knowledge likelihood (v2)

def knowledge_document(row):
    """Frozen parsing (kit/resources/m3/knowledge-likelihood.json): question context and option texts."""
    head, sep, rest = row['prompt'].partition('\n\n(A) ')
    require(sep and head.strip(), 'knowledge item lacks a question/options block: ' + str(row['id']))
    block = ('(A) ' + rest).split('\n\n')[0].split('\n')
    labels, options = [], []
    for line in block:
        match = re.fullmatch(r'\(([A-Z])\) (.+)', line)
        require(match is not None, 'malformed knowledge option line: ' + str(row['id']))
        labels.append(match.group(1)); options.append(match.group(2).strip())
    require(labels == list('ABCD') and row['answer'] in labels and all(options), 'knowledge item needs options A-D and a gold letter: ' + str(row['id']))
    return 'Question: ' + head.strip() + '\nAnswer:', options, labels.index(row['answer'])


def knowledge_items():
    rows = [r for r in t.read_jsonl(GENERAL) if r['panel'] == 'knowledge']
    require(len(rows) == 100 and len({r['id'] for r in rows}) == 100, 'knowledge-ll requires the 100 frozen knowledge items')
    items = []
    for r in rows:
        context, options, label = knowledge_document(r)
        items.append(dict(id=r['id'], context=context, endings=options, label=label))
    return items


def selected(scores):
    require(len(scores) == 4 and all(math.isfinite(x) for x in scores), 'finite likelihoods required for every option')
    return sorted(range(4), key=lambda i: (-scores[i], i))[0]


def knowledge_production(model, rows, tokenizer_path):
    """vLLM prompt_logprobs over context + ' ' + option; option tokens summed and divided by option characters."""
    from kit.m2_gpu import assert_current
    assert_current()
    from transformers import AutoTokenizer
    from vllm import SamplingParams
    from kit.inference_lifecycle import marker, create_llm, shutdown_since
    from kit.m2_format import tokenizer_identity
    tokenizer_identity(tokenizer_path)
    tok = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    requests = []; lengths = []; starts = []
    for row in rows:
        context_ids = tok.encode(row['context'], add_special_tokens=False)
        for ending in row['endings']:
            ids = tok.encode(row['context'] + ' ' + ending, add_special_tokens=False)
            require(bool(context_ids) and ids[:len(context_ids)] == context_ids and len(ids) > len(context_ids), 'conditional token boundary changed')
            requests.append(dict(prompt_token_ids=ids)); lengths.append(len(ending)); starts.append(len(context_ids))
    mark = marker()
    try:
        engine = create_llm(model=str(model), tokenizer=str(tokenizer_path), tensor_parallel_size=1, distributed_executor_backend='uni',
                            dtype='bfloat16', max_model_len=12288, max_num_seqs=1, gpu_memory_utilization=.9, enforce_eager=True, seed=0)
        outputs = engine.generate(requests, SamplingParams(temperature=0, max_tokens=1, prompt_logprobs=1), use_tqdm=False)
        require(len(outputs) == len(requests), 'complete item/option likelihood coverage required')
        values = []
        for request, output, start, denominator in zip(requests, outputs, starts, lengths):
            ids = request['prompt_token_ids']; logs = output.prompt_logprobs
            require(len(logs) == len(ids), 'complete prompt likelihood coverage required')
            values.append(sum(logs[j][ids[j]].logprob for j in range(start, len(ids))) / denominator)
        return [dict(id=row['id'], scores=values[i * 4:i * 4 + 4], selected=selected(values[i * 4:i * 4 + 4])) for i, row in enumerate(rows)]
    finally: shutdown_since(mark)


def knowledge_reference(model, rows, tokenizer_path):
    """Independent sequential HF reference: own parse of the frozen panel rows, explicit token log-softmax reduction."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from kit.m2_gpu import assert_current
    assert_current()
    raw = {r['id']: r for r in t.read_jsonl(GENERAL) if r['panel'] == 'knowledge'}
    tok = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    net = AutoModelForCausalLM.from_pretrained(model, local_files_only=True, torch_dtype=torch.bfloat16).to('cuda').eval()
    records = []
    try:
        with torch.inference_mode():
            for row in rows:
                prompt = raw[row['id']]['prompt']
                question = prompt[:prompt.index('\n\n(A) ')].strip()
                options = re.findall(r'^\([A-D]\) (.+)$', prompt[prompt.index('\n\n(A) ') + 2:].split('\n\n')[0], flags=re.M)
                context = f'Question: {question}\nAnswer:'; context_ids = tok.encode(context, add_special_tokens=False)
                values = []
                for option in (o.strip() for o in options):
                    ids = tok.encode(context + ' ' + option, add_special_tokens=False)
                    assert ids[:len(context_ids)] == context_ids and len(ids) > len(context_ids)
                    logp = net(input_ids=torch.tensor([ids], device='cuda'), use_cache=False).logits[0].float().log_softmax(-1)
                    values.append(sum(float(logp[j - 1, ids[j]].item()) for j in range(len(context_ids), len(ids))) / len(option))
                records.append(dict(id=row['id'], scores=values, selected=max(range(4), key=lambda i: (values[i], -i))))
    finally:
        del net
        torch.cuda.empty_cache()
    return records


def standin_likelihoods(rows, salt):
    """CPU fixture: deterministic finite scores; the selected option varies by item and checkpoint."""
    out = []
    for r in rows:
        pick = int(t.sha(salt + '|' + r['id']), 16) % 4
        scores = [-1.0 - i for i in range(4)]; scores[pick], scores[0] = scores[0], scores[pick]
        out.append(dict(id=r['id'], scores=scores, selected=selected(scores)))
    return out


def validate_knowledge(a):
    validate(a, 'knowledge-ll', knowledge_items(), knowledge_production, knowledge_reference, s.file_hash(KNOWLEDGE_POLICY),
             dict(json.loads(KNOWLEDGE_POLICY.read_text()), panel_sha256=s.file_hash(GENERAL)))


def score_knowledge(a, root, source, proof, unit, base_model):
    """Scored as soon as the checkpoint exists; counted by the reader only after the seed's validation PASSES."""
    from kit.p3b_score import write_csv
    from kit.c_score import COLUMNS
    validation = None
    rows = knowledge_items(); started = time.monotonic()
    likelihood = standin_likelihoods(rows, root.name) if a.standin else knowledge_production(source['model'], rows, base_model)
    require(len(likelihood) == 100 and {r['id'] for r in likelihood} == {r['id'] for r in rows}, 'complete knowledge likelihood coverage required')
    gold = {r['id']: r['label'] for r in rows}
    scored = [dict(arm=unit, seed=a.seed, step=a.step, task='knowledge', panel='knowledge', cap='', question_id=r['id'],
                   correct=int(selected(r['scores']) == gold[r['id']]), format_free_correct=int(selected(r['scores']) == gold[r['id']]),
                   output_tokens=0, finished=1) for r in likelihood]
    root.mkdir(parents=True, exist_ok=True)
    write_csv(root / 'scores.csv', [{k: r.get(k, '') for k in COLUMNS} for r in scored], COLUMNS)
    atomic(root / 'likelihoods.json', dict(policy_sha256=s.file_hash(KNOWLEDGE_POLICY), validation_sha256=validation, items_sha256=s.digest(rows), rows=likelihood))
    atomic(root / 'cost.json', dict(kind='m3_scoring', panel='knowledge-ll', gpu_hours=0 if a.standin else (time.monotonic() - started) / 3600, technical_synthetic=a.standin))
    atomic(root / 'manifest.json', files_manifest(root, ['scores.csv', 'likelihoods.json', 'cost.json']))
    return s.digest(rows), dict(policy_sha256=s.file_hash(KNOWLEDGE_POLICY), validation_sha256=validation)


# ----------------------------------------------------------------------------- scoring

def checked(out, arm, seed, stage, step, standin):
    """Verify the (possibly moved) export against its training receipt before scoring it."""
    from kit.m3_train import record
    source = record(out, arm, seed, stage, step)
    if standin: return source, dict(technical_synthetic=True, model=source['model'], model_file_hashes=source['identity'].get('model_file_hashes'))
    from kit.p3b_common import identity
    observed = identity(source['model'], 'student' if source['base'] else None, source.get('receipt'))
    require(observed['model_file_hashes'] == source['identity']['model_file_hashes'], 'UNVERIFIED: checkpoint identity changed')
    return source, observed


def generate_scores(a, panel, source, root, unit):
    """Legal test / hard-MATH generation and scoring, as kit/m2_score.py does for the same panels."""
    from kit.m2_score import items_for
    from kit.c_score import legal_scores, COLUMNS
    from kit.p3b_score import write_csv
    from kit.p3b_common import retain_partial
    items = items_for(panel, a.standin, a.panel_root)
    retain_partial(root); root.mkdir(parents=True, exist_ok=True); started = time.monotonic()
    os.environ.update(CUDA_VISIBLE_DEVICES=a.gpu, KIT_DETERMINISTIC='1', VLLM_BATCH_INVARIANT='1', VLLM_ATTENTION_BACKEND='FLASH_ATTN')
    if not a.standin:
        from kit.m2_gpu import assert_current
        assert_current()
    if a.standin:
        from kit.m2_sim import responses
        raw = responses(items, unit, a.step, panel)
    elif panel == 'test':
        from kit import score_forgetting as general
        from kit.c_common import legal
        from transformers import AutoTokenizer
        from vllm import SamplingParams
        from kit.inference_lifecycle import create_llm as LLM
        tokenizer = AutoTokenizer.from_pretrained(source['model'], local_files_only=True)
        prompts = [legal.render(tokenizer, r['prompt']) for r in items]
        require(all(len(tokenizer.encode(p, add_special_tokens=False)) <= 10240 for p in prompts), 'legal input exceeds frozen cap')
        engine = LLM(model=source['model'], enable_lora=False, **dict(general.ENGINE, max_model_len=12288), enforce_eager=True, seed=0)
        outputs = engine.generate(prompts, SamplingParams(n=1, **general.DECODING))
        require(len(outputs) == len(items), 'incomplete legal scoring batch')
        raw = [dict(id=r['id'], response=o.outputs[0].text, output_tokens=len(o.outputs[0].token_ids), prompt_tokens=len(o.prompt_token_ids),
                    finish_reason=o.outputs[0].finish_reason) for r, o in zip(items, outputs)]
    else:
        from kit import eval_bed
        args = NS(bed='math_screen', root=None, split='test', model=source['model'], max_new_tokens=2048, max_model_len=12288,
                  out=str(root / 'raw'), limit=None, heldout_n=None, allow_subset=False)
        eval_bed.cmd_generate(args, frozen_items=[dict(r, ground_truth=r['gold']) for r in items])
        raw = t.read_jsonl(root / 'raw/responses.jsonl')
    require(len(raw) == len(items) and {r['id'] for r in raw} == {r['id'] for r in items}, 'incomplete/foreign score responses')
    if panel == 'test': rows = legal_scores(items, raw, unit, a.seed, a.step, 'test')
    else:
        from kit.beds import math_screen
        lookup = {r['id']: r for r in items}; rows = []
        for r in raw:
            correct = r['standin_correct'] if a.standin else math_screen.compute_score(math_screen.DATA_SOURCE, r['response'], lookup[r['id']]['gold'])['score']
            rows.append(dict(arm=unit, seed=a.seed, step=a.step, task=panel, panel=panel, cap=2048, question_id=r['id'], correct=int(correct),
                             output_tokens=r['output_tokens'], finished=int(r['finish_reason'] in ('stop', 'eos'))))
    (root / 'raw').mkdir(exist_ok=True); t.write_jsonl(root / 'raw/responses.jsonl', raw)
    write_csv(root / 'scores.csv', [{k: r.get(k, '') for k in COLUMNS} for r in rows], COLUMNS)
    atomic(root / 'cost.json', dict(kind='m3_scoring', panel=panel, gpu_hours=0 if a.standin else (time.monotonic() - started) / 3600, technical_synthetic=a.standin))
    atomic(root / 'manifest.json', files_manifest(root, ['scores.csv', 'cost.json', 'raw/responses.jsonl']))
    return s.digest(items)


def format_free(a, root, source, proof):
    """Replace HellaSwag correctness with the frozen format-free likelihood choice (message-2 amendment)."""
    import csv
    from kit import m2_format as formatting
    from kit.p3b_score import write_csv
    from kit.c_score import COLUMNS
    from kit.next_hellaswag import strict_label
    from kit.m3_train import base_record
    validation = None  # the seed's validation (base + C0 s101 stage-1 final) gates the reader, not scoring
    population = formatting.items(a.out, a.standin)
    if a.standin: likelihood = [dict(id=r['id'], scores=[0., -1., -2., -3.], selected=0) for r in population]
    else: likelihood = formatting.likelihoods(source['model'], population, base_record(a.out)['model'])
    require(len(likelihood) == 100 and {r['id'] for r in likelihood} == {r['id'] for r in population}, 'complete HellaSwag likelihood coverage required')
    lookup = {r['id']: r for r in likelihood}; golds = {r['id']: int(r['label']) for r in population}
    responses = {r['id']: r for r in t.read_jsonl(root / 'responses.jsonl')}
    manifest = frozen_files(root)
    with (root / 'scores.csv').open(newline='') as stream: scored = list(csv.DictReader(stream))
    for r in scored:
        r['strict_diagnostic_correct'] = r['correct']; r['incorrect_format'] = int(strict_label(responses[r['question_id']]['response']) is None)
        r['format_free_correct'] = int(formatting.selected(lookup[r['question_id']]['scores']) == golds[r['question_id']]); r['correct'] = r['format_free_correct']
    write_csv(root / 'scores.csv', scored, COLUMNS)
    atomic(root / 'likelihoods.json', dict(policy_sha256=s.file_hash(formatting.POLICY), validation_sha256=validation, items_sha256=s.digest(population), rows=likelihood))
    manifest['files'].update({'scores.csv': s.file_hash(root / 'scores.csv'), 'likelihoods.json': s.file_hash(root / 'likelihoods.json')})
    atomic(root / 'manifest.json', manifest)
    return dict(policy_sha256=s.file_hash(formatting.POLICY), validation_sha256=validation)


@costed_score
def score_job(a):
    require(a.panel_set in PANELS and a.seed in plan.SEEDS, 'unregistered panel/seed')
    proof = proof_for(a.out, a.gpu, a.seed, a.standin)
    resolved_gpu(a)
    root = unit_root(a.out, a.arm, a.seed, a.stage, a.step, a.panel_set); unit = unit_arm(a.arm, a.stage)
    if (root / 'manifest.json').exists() and 'm3-identity.json' in frozen_files(root)['files']: print('SKIP:', root.name); return
    source, observed = checked(a.out, a.arm, a.seed, a.stage, a.step, a.standin)
    method = None
    receipt = Path(source['receipt']) if source.get('receipt') and not source['base'] else None
    model = source['model']
    if a.standin:
        # Inherited stand-in scorers check a fixture config hash, exactly as kit/m2_score.py arranges.
        fixture = a.out / 'standin-models' / root.name; atomic(fixture / 'config.json', dict(fixture=True))
        receipt = fixture / 'receipt.json'; atomic(receipt, dict(technical_synthetic=True, model_file_hashes={'config.json': s.file_hash(fixture / 'config.json')}))
        model = str(fixture)
    if a.panel_set == 'knowledge-ll':
        from kit.m3_train import base_record
        panel_sha, method = score_knowledge(a, root, source, proof, unit, base_record(a.out)['model'])
    elif (root / 'manifest.json').exists():
        panel_sha = json.loads((root / 'identity.json').read_text()).get('panel_sha256') if (root / 'identity.json').exists() else None
    elif a.panel_set in ('finqa', 'general-v1'):
        from kit.p3b_score import job as score
        score(NS(out=a.out, arm=unit, seed=a.seed, step=a.step, panel_set=a.panel_set, model=model, gpu=a.gpu, standin=a.standin,
                 checkpoint_receipt=receipt, panel_root=a.panel_root, package_a=None))
        panel_sha = json.loads((root / 'identity.json').read_text())['panel_sha256']
    elif a.panel_set == 'hellaswag':
        from kit.next_hellaswag import score
        score(NS(out=a.out, arm=unit, seed=a.seed, step=a.step, model=model, gpu=a.gpu, standin=a.standin, checkpoint_receipt=receipt))
        panel_sha = frozen_files(a.out / 'hellaswag').get('panel_sha256')
    else: panel_sha = generate_scores(a, a.panel_set, source, root, unit)
    if a.panel_set == 'hellaswag':
        if 'likelihoods.json' in frozen_files(root)['files']:  # resumed tail: never re-apply the format-free pass
            done = json.loads((root / 'likelihoods.json').read_text()); method = dict(policy_sha256=done['policy_sha256'], validation_sha256=done['validation_sha256'])
        else:
            with gpu_phase(a, 'm3_hellaswag_likelihood', included_in='m3_scoring_attempt'):
                method = format_free(a, root, source, proof)
    meta = dict(schema='message3-score.v2', model=observed, assignment=proof, gpu_identity=proof, arm=a.arm, seed=a.seed, stage=a.stage,
                step=a.step, panel=a.panel_set, panel_sha256=panel_sha, decoding=DECODING, format_free=method, technical_synthetic=a.standin)
    atomic(root / 'm3-identity.json', meta)
    manifest = frozen_files(root); manifest['files']['m3-identity.json'] = s.file_hash(root / 'm3-identity.json'); atomic(root / 'manifest.json', manifest)


def alias_job(a):
    """Incoming (step-0) units point at the identical checkpoint's scores on the same assignment."""
    require(a.step == 0, 'only step-0 units are aliases')
    if a.stage == plan.STAGES[0]: source = unit_root(a.out, 'base', a.seed, a.stage, 0, a.panel_set)
    else: source = unit_root(a.out, a.arm, a.seed, plan.STAGES[plan.STAGES.index(a.stage) - 1], 40, a.panel_set)
    if not (source / 'manifest.json').exists(): raise NotReady('alias source not scored: ' + source.name)
    manifest = frozen_files(source); require('m3-identity.json' in manifest['files'], 'alias source lacks M3 identity')
    root = unit_root(a.out, a.arm, a.seed, a.stage, 0, a.panel_set)
    if a.arm == 'base' or root == source: return
    value = dict(source=source.name, scores_sha256=manifest['files']['scores.csv'], identity_sha256=manifest['files']['m3-identity.json'],
                 arm=a.arm, seed=a.seed, stage=a.stage, step=0, panel=a.panel_set)
    bind(root / 'alias.json', value)
    atomic(root / 'manifest.json', files_manifest(root, ['alias.json']))


def resolve(out, arm, seed, stage, step, panel):
    """Unit directory holding scores for (arm, seed, stage, step, panel), following one alias hop with hash checks."""
    root = unit_root(out, arm, seed, stage, step, panel)
    if not (root / 'manifest.json').exists(): return None
    manifest = frozen_files(root)
    if 'alias.json' in manifest['files']:
        alias = json.loads((root / 'alias.json').read_text()); target = Path(out) / 'scores' / alias['source']
        target_manifest = frozen_files(target)
        require(target_manifest['files'].get('scores.csv') == alias['scores_sha256'] and
                target_manifest['files'].get('m3-identity.json') == alias['identity_sha256'], 'UNVERIFIED: stale alias ' + root.name)
        return target
    return root


def job(a):
    from kit.m3_manifest import require_m3_root
    require_m3_root(a.out)
    from contextlib import nullcontext
    from kit.m3_manifest import gpu_execution
    with gpu_execution(a.out) if not a.standin and a.action in ('score', 'validate-hellaswag', 'validate-knowledge') else nullcontext():
        return {'assign': assign_job, 'validate-hellaswag': validate_hellaswag,
                'validate-knowledge': validate_knowledge, 'score': score_job, 'alias': alias_job}[a.action](a)

def configure(p):
    p.add_argument('--action', choices=('assign', 'validate-hellaswag', 'validate-knowledge', 'score', 'alias'), required=True)
    p.add_argument('--inventory', type=Path)
    p.add_argument('--arm', choices=('base', *plan.ARMS)); p.add_argument('--seed', type=int, choices=plan.SEEDS)
    p.add_argument('--stage', choices=plan.STAGES, default=plan.STAGES[0]); p.add_argument('--step', type=int, choices=(0, 10, 20, 30, 40), default=0)
    p.add_argument('--panel-set', choices=PANELS); p.add_argument('--gpu', default='0'); p.add_argument('--panel-root', type=Path)


if __name__ == '__main__':
    from kit.p3b_common import entry
    raise SystemExit(entry(job, configure, 'm3-score-{action}'))
