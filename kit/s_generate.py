#!/usr/bin/env python3
"""One greedy verified own response per frozen row and incoming checkpoint."""
from pathlib import Path
import sys
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kit.s_common import *
from kit.s_freeze import partitions
import time


def correct(domain, row, output, standin=False):
    if output['finish_reason'] not in ('stop', 'eos'):
        return False
    if domain == 'ifeval':
        from kit.p3b_replay_data import checker
        if standin:
            try:
                import langdetect  # noqa: F401
            except ModuleNotFoundError:
                from unittest.mock import patch
                from types import SimpleNamespace
                def forbidden(*args):
                    raise ValueError('language detection cannot be simulated')
                with patch.dict(sys.modules, langdetect=SimpleNamespace(detect=forbidden)):
                    return all(c['passed'] for c in checker(output['text'], row['ground_truth']))
        return all(c['passed'] for c in checker(output['text'], row['ground_truth']))
    if domain == 'finqa':
        from kit.beds.finqa import is_correct, extract_answer
        return is_correct(extract_answer(output['text']), row['gold'])
    from kit.next_pools import correct as broad_correct
    return broad_correct(dict(row, source=domain), output['text'])


def selected(out, incoming, domain):
    root = out/'own'/incoming/domain
    doc = read(root/'manifest.json')
    require(doc['incoming'] == read(out/'binding.json')['identities'][incoming], 'own response incoming changed')
    for part in ('train', 'monitor'):
        require(sha(root/(part+'.jsonl')) == doc['files'][part+'.jsonl'], 'own responses changed')
    require(doc['audit_sha256'] == sha(out/'audit.json') and doc['partition_sha256'] == sha(out/'frozen/manifest.json'), 'own-response freeze changed')
    result = {p: rows(root/(p+'.jsonl')) for p in ('train', 'monitor')}
    require(all(r['verified'] is True and r['incoming'] == incoming for rr in result.values() for r in rr), 'extra credit requires incoming-verified behaviour')
    require(not {r['id'] for r in result['train']} & {r['id'] for r in result['monitor']}, 'monitoring rows cannot train')
    return result


def domains_for(spec, incoming):
    return DOMAINS[:3] if spec['nodes'][incoming]['root'] else DOMAINS


def job(a):
    doc = context(a.out, a.standin)
    spec = doc['spec']
    require(a.incoming in {c['incoming'] for c in spec['stages']}, 'unregistered incoming')
    checkpoint = spec['nodes'][a.incoming]['checkpoint']
    identity = model(checkpoint, a.standin)
    require(identity == doc['identities'][a.incoming], 'incoming checkpoint changed')
    from kit.m2_handoff import scoring_identity
    scoring_identity(a.out, a.gpu, a.standin)
    os.environ.update(CUDA_VISIBLE_DEVICES=a.gpu, V4_PHASE0_INFERENCE='0', KIT_DETERMINISTIC='1')
    generator = tok = None
    if not a.standin:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(__import__('kit.m2_paths',fromlist=['local']).local(checkpoint['model']), local_files_only=True)
    for domain in domains_for(spec, a.incoming):
        root = a.out/'own'/a.incoming/domain
        inputs = dict(incoming=identity, audit_sha256=sha(a.out/'audit.json'), partition_sha256=sha(a.out/'frozen/manifest.json'),
                      registration_sha256=sha(RESOURCE), technical_synthetic=a.standin)
        bind(root/'inputs.json', inputs)
        if (root/'manifest.json').exists():
            selected(a.out, a.incoming, domain)
            print('SKIP own:', a.incoming, domain)
            continue
        start = time.monotonic()
        frozen = partitions(a.out, domain)
        files = {}
        for part, count in (('train', 128), ('monitor', 64)):
            accepted = []
            for row in frozen[part]:
                path = root/'raw'/part/(digest(row['id'])+'.json')
                messages = row['prompt'] if isinstance(row['prompt'], list) else [dict(role='user', content=row['prompt'])]
                if path.exists():
                    raw = read(path)
                elif a.standin:
                    raw = dict(text=row['fake_response'], finish_reason='stop', token_ids=[1, 2], prompt_tokens=4)
                    atomic(path, raw)
                else:
                    if generator is None:
                        from kit.v4_teacher import VLLMGenerator
                        generator = VLLMGenerator(checkpoint['model'], 2048, input_cap=10240, tensor_parallel_size=1, allow_batch_fallback=False)
                    raw = generator(messages, 1, seed=0, temperature=0, top_p=1)
                    atomic(path, raw)
                verified = correct(domain, row, raw, a.standin)
                if verified and tok is not None:
                    from kit.v4_teacher import tokenize_student
                    pids, tids = tokenize_student(tok, messages, raw['text'])
                    verified = 0 < len(pids) <= 10240 and 0 < len(tids) <= 2048
                bind(root/'checks'/part/(digest(row['id'])+'.json'), dict(id=row['id'], verified=verified, raw_sha256=sha(path)))
                if verified:
                    accepted.append(dict(id=row['id'], prompt=row['prompt'], response=raw['text'], incoming=a.incoming,
                                         verified=True, domain=domain, raw_sha256=sha(path)))
                if len(accepted) == count:
                    break
            target = root/(part+'.jsonl')
            write_rows(target, accepted)
            files[target.name] = sha(target)
        atomic(root/'cost.json', dict(seconds=time.monotonic()-start, gpu_hours=0 if a.standin else (time.monotonic()-start)/3600,
                                      technical_synthetic=a.standin))
        bind(root/'manifest.json', dict(inputs, files=files, train_n=len(rows(root/'train.jsonl')), monitor_n=len(rows(root/'monitor.jsonl')),
             selected_digests={part:digest(rows(root/(part+'.jsonl'))) for part in ('train','monitor')},
             no_replenishment=True, decoding=dict(seed=0, temperature=0, top_p=1, thinking=False, max_tokens=2048)))
        print('Own responses:', a.incoming, domain, len(rows(root/'train.jsonl')), len(rows(root/'monitor.jsonl')))


def configure(p):
    p.add_argument('--incoming', required=True)
    p.add_argument('--gpu', default='0')
if __name__ == '__main__':
    raise SystemExit(entry(job, configure))
