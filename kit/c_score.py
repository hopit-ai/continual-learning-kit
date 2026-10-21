#!/usr/bin/env python3
"""Score one C checkpoint/panel on the single physical scoring GPU, both legal parsers."""
from pathlib import Path
import sys
if __package__ in (None, ''): sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kit.c_common import *
from kit.p3b_score import write_csv
from types import SimpleNamespace
import time

COLUMNS = 'arm seed step task panel cap question_id correct strict_correct tolerant_correct strict_format tolerant_parseable output_tokens finished group gold'.split()


def legal_scores(items, raw, arm, seed, step, panel):
    by_id = {}
    for r in raw:
        require(r['id'] not in by_id and r['id'] in {i['id'] for i in items}, 'unknown/duplicate legal response')
        require(isinstance(r.get('output_tokens'), int) and r['output_tokens'] >= 0, 'missing token accounting')
        by_id[r['id']] = r
    rows = []
    for item in items:
        r = by_id.get(item['id'], dict(response='', output_tokens=0, finish_reason='missing'))
        strict = legal.parse_answer('contract_nli', r['response']); tolerant = legal.parse_answer_tolerant('contract_nli', r['response'])
        rows.append(dict(arm=arm, seed=seed, step=step, task='contract_nli', panel=panel, cap=2048,
            question_id=item['id'], correct=int(tolerant == item['gold']), strict_correct=int(strict == item['gold']),
            tolerant_correct=int(tolerant == item['gold']), strict_format=int(strict is not None), tolerant_parseable=int(tolerant is not None),
            output_tokens=r['output_tokens'], finished=int(r['finish_reason'] in ('stop','eos')), group=item['group'], gold=item['gold']))
    return rows


def job(a):
    doc = binding(a.out, a.standin)
    from kit.next_common import machine as scoring_identity
    gpu_identity = scoring_identity(a.out, a.gpu, a.standin)
    require(',' not in a.gpu and bool(a.gpu), 'use one physical scoring GPU')
    require(a.panel_set in ('panel','test') or a.step in RETENTION_STEPS, 'retention scored at 0/20/40 only')
    record = checkpoint(doc, a.out, a.arm, a.seed, a.step)
    observed = checked_model(record, a.standin, base=a.arm == 'base' or (a.step == 0 and a.arm.startswith('standalone-')))
    root = a.out / 'scores' / f'{a.arm}-s{a.seed}-k{a.step}-{a.panel_set}'
    meta = dict(model=observed, panel=a.panel_set, binding_sha256=s.file_hash(a.out / 'binding.json'), protocol=protocol(), gpu=a.gpu,
                gpu_identity=gpu_identity, arm=a.arm, seed=a.seed, step=a.step, technical_synthetic=a.standin)
    bind(a.out / 'c-score-identities' / (root.name+'.json'), meta)
    if (root / 'manifest.json').exists(): frozen_files(root); print('SKIP:', root.name); return
    # B's deterministic FinQA/general scorers retain their prompts, graders, and raw evidence.
    if a.panel_set == 'hellaswag':
        from kit.next_hellaswag import score as hs_score
        args = SimpleNamespace(**vars(a)); args.model = record['model']; args.checkpoint_receipt = Path(record['receipt']) if record.get('receipt') else None
        if a.standin:
            # HellaSwag stand-in checks a fixture config; legal stages have receipt-only exports.
            fixture = a.out / 'standin-hellaswag-models' / root.name
            model = fixture / 'model'; atomic(model / 'config.json', dict(fixture=True))
            receipt = fixture / 'receipt.json'; atomic(receipt, dict(technical_synthetic=True, model_file_hashes={'config.json':s.file_hash(model / 'config.json')}))
            args.model = str(model); args.checkpoint_receipt = receipt
        hs_score(args)
        return
    if a.panel_set in ('finqa','general-v1'):
        from kit.p3b_score import job as b_score
        args = SimpleNamespace(**vars(a)); args.model = record['model']; args.checkpoint_receipt = Path(record['receipt']) if record.get('receipt') else None
        args.package_a = None
        b_score(args)
        atomic(root / 'c-protocol.json', meta)
        manifest = frozen_files(root); manifest['files']['c-protocol.json'] = s.file_hash(root / 'c-protocol.json')
        atomic(root / 'manifest.json', manifest)
        return
    items = panels()[a.panel_set]
    retain_partial(root); root.mkdir(parents=True, exist_ok=True)
    os.environ.update(CUDA_VISIBLE_DEVICES=a.gpu, KIT_DETERMINISTIC='1', VLLM_BATCH_INVARIANT='1', VLLM_ATTENTION_BACKEND='FLASH_ATTN')
    started = time.monotonic()
    if a.standin:
        raw = []
        for i, r in enumerate(items):
            correct = i < (42 if a.step == 0 else 42 + a.step)
            label = r['gold'] if correct else next(x for x in legal.LABELS['contract_nli'] if x != r['gold'])
            text = 'Reasoned result.\n' + ('**Answer: '+label+'**' if i % 4 == 0 else 'Answer: '+label)
            raw.append(dict(id=r['id'], response=text, output_tokens=10, finish_reason='stop', token_ids=list(range(10))))
        machine = 'CPU-STANDIN'
    else:
        from kit import score_forgetting as general
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
        tokenizer = AutoTokenizer.from_pretrained(record['model'], local_files_only=True)
        prompts = [legal.render(tokenizer, r['prompt']) for r in items]
        require(all(len(tokenizer.encode(p, add_special_tokens=False)) <= 10240 for p in prompts), 'frozen legal input over cap')
        engine = LLM(model=record['model'], enable_lora=False, **dict(general.ENGINE, max_model_len=12288), enforce_eager=True, seed=0)
        outputs = engine.generate(prompts, SamplingParams(n=1, **general.DECODING))
        require(len(outputs) == len(items), 'incomplete scoring engine batch')
        raw = []
        for item, output in zip(items, outputs):
            completion = output.outputs[0]
            raw.append(dict(id=item['id'], response=completion.text, output_tokens=len(completion.token_ids),
                finish_reason=completion.finish_reason, token_ids=list(completion.token_ids),
                prompt_tokens=len(output.prompt_token_ids), panel_sha256=legal.load_panel(C, 'contract_nli', a.panel_set)[1]))
        machine = general.machine_fingerprint(True)['id']
    bind(a.out / 'scoring-machine.json', dict(id=machine, gpu=a.gpu, technical_synthetic=a.standin))
    rows = legal_scores(items, raw, a.arm, a.seed, a.step, a.panel_set)
    raw_root = root / 'raw'; raw_root.mkdir()
    t.write_jsonl(raw_root / 'responses.jsonl', raw)
    write_csv(root / 'scores.csv', rows, COLUMNS)
    atomic(root / 'identity.json', dict(**meta, machine=machine, panel_sha256=legal.load_panel(C, 'contract_nli', a.panel_set)[1],
        deterministic=True, eager=True, batch_invariant=True, max_model_len=12288, max_new_tokens=2048, temperature=0, thinking=False))
    atomic(root / 'cost.json', dict(kind='scoring', gpu_hours=0 if a.standin else (time.monotonic()-started)/3600, technical_synthetic=a.standin))
    atomic(root / 'manifest.json', files_manifest(root, ['scores.csv','identity.json','cost.json','raw/responses.jsonl']))


def configure(p):
    p.add_argument('--gpu', default='0'); p.add_argument('--arm', required=True)
    p.add_argument('--seed', type=int, choices=SEEDS, required=True); p.add_argument('--step', type=int, choices=STEPS, required=True)
    p.add_argument('--panel-set', choices=('panel','test','finqa','general-v1','hellaswag'), required=True); p.add_argument('--panel-root', type=Path)
if __name__ == '__main__': raise SystemExit(entry(job, configure, 'c-score-{arm}-{seed}-{step}-{panel_set}'))
