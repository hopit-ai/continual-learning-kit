"""Direct calls to the frozen screen/evaluator, shared by three GPU job scripts."""
import os
from types import SimpleNamespace
from kit.p3a_common import *


def identity(a, role):
    model = model_path(a.model)
    observed = t.local_identity(model, s.MODELS[role][0])
    if observed['model_revision'] != s.MODELS[role][1]:
        print('WARNING: model revision differs/unknown; flag provenance:', observed['model_revision'])
    path = a.out / INPUTS / (role + '-model.json')
    if path.exists() and json.loads(path.read_text())['model_file_hashes'] != observed['model_file_hashes']:
        raise CorrectnessStop('model input hashes changed: ' + role)
    atomic(path, observed)
    return model


def phase(a, name):
    """Teacher TP2 or frozen-8B rewrite TP1; one fresh process per invocation."""
    freeze = frozen(a.out, a.standin)
    if not a.standin: model_path(a.model); model_path(a.tokenizer)
    if not allowed(a.out, a.candidate, freeze, a.standin): return
    base = a.out / INPUTS / 'prepared' / a.candidate
    out = a.out / OUTPUTS / a.candidate
    if name == 'rewrite' and not (out / 'teacher-complete.json').exists():
        raise NotReady('rewrite needs the teacher phase to complete first')
    rewriting = name == 'rewrite'
    if not a.standin:
        # Check the supplied model against the recorded one before reusing any finished work.
        model = identity(a, 'student' if rewriting else 'teacher')
        identity(SimpleNamespace(out=a.out, model=a.tokenizer), 'student')
    if (out / (name + '-complete.json')).exists(): print('SKIP:', name, a.candidate); return
    demos = t.read_jsonl(out / 'teacher/demonstrations.jsonl') if rewriting else None
    os.environ.update(CUDA_VISIBLE_DEVICES=a.gpu, KIT_DETERMINISTIC='1', V4_PHASE0_INFERENCE='0' if rewriting else '1')
    if a.standin:
        from kit.sim.p3a_row import Tokenizer, Generator
        tokenizer, generator = Tokenizer(), Generator(a.candidate, a.scenario)
        observed = {'technical_synthetic': True, 'gpu_count': 0}
    else:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(a.tokenizer, local_files_only=True)
        generator = t.VLLMGenerator(model, t.REWRITE_NEW_TOKENS if rewriting else t.TEACHER_NEW_TOKENS,
            input_cap=t.REWRITE_INPUT_CAP if rewriting else t.INPUT_CAP,
            tensor_parallel_size=1 if rewriting else 2, allow_batch_fallback=False)
        observed = json.loads((a.out / INPUTS / (('student' if rewriting else 'teacher') + '-model.json')).read_text())
    atomic(out / (name + '-identity.json'), observed)
    s.run_phase(t.read_jsonl(base / 'sample.jsonl'), out / name, generator, tokenizer,
        demonstrations=demos, synthetic=a.standin, tests=str(base / 'tests.jsonl') if a.candidate == 'code' else None)
    atomic(out / (name + '-complete.json'), {'complete': True, 'technical_synthetic': a.standin})


def score_complete(result, panel):
    """The evaluator writes bed-score.json non-atomically: complete only if it parses and covers the panel."""
    try: scores = json.loads(result.read_text())['per_item']
    except (OSError, ValueError, KeyError, TypeError): return False
    return set(scores) == {r['id'] for r in panel}


def baseline(a):
    """Conditional MATH headroom, B=2048/context=12288/seed=0, eager."""
    from kit import eval_bed
    freeze = frozen(a.out, a.standin)
    if not a.standin: model_path(a.model)
    if not allowed(a.out, 'math_screen', freeze, a.standin, baseline=True): return
    base = a.out / INPUTS / 'prepared/math_screen'
    out = a.out / OUTPUTS / 'math_screen/step1/baseline'
    panel = t.read_jsonl(base / 'heldout.jsonl')
    items = [dict(id=r['id'], prompt=r['prompt'], ground_truth=r['gold']) for r in panel]
    result = out / 'bed-score.json'
    model = '' if a.standin else identity(a, 'student')
    if not score_complete(result, panel):
        retain_partial(out)
        os.environ.update(CUDA_VISIBLE_DEVICES=a.gpu, KIT_DETERMINISTIC='1', V4_PHASE0_INFERENCE='1')
        args = SimpleNamespace(bed='math_screen', root=str(base), split='heldout', model=model,
            max_new_tokens=2048, max_model_len=12288, out=str(out), limit=None, heldout_n=100, allow_subset=False)
        if a.standin:
            answers = {r['id']: '\\boxed{' + ('1/2' if a.scenario == 'math-stop' else '2/3') + '}' for r in panel}
            graded = eval_bed.grade(eval_bed.load_bed('math_screen'), args, items, answers)
            out.mkdir(parents=True)
            t.write_jsonl(out / 'responses.jsonl', [dict(id=k, response=v, technical_synthetic=True) for k,v in answers.items()])
            eval_bed.write_result(out, bed='math_screen', split='heldout', graded=graded, extra={'technical_synthetic': True})
        else: eval_bed.cmd_generate(args, frozen_items=items)
    else: print('SKIP: complete MATH baseline')
    scores = json.loads(result.read_text())['per_item']
    row = s.headroom(panel, [dict(id=k, correct=bool(v)) for k,v in scores.items()])
    row.update(candidate='math_screen', step=1, freeze_sha256=freeze, paid=not a.standin, technical_synthetic=a.standin)
    atomic(a.out / INPUTS / 'math_screen-step1.json', row)
    atomic(out.parent / 'decision.json', row)
    print('MATH headroom:', row['status'])


def configure(p, name):
    p.add_argument('--model', help='local pinned HF model directory')
    p.add_argument('--gpu', default=os.environ.get('CUDA_VISIBLE_DEVICES', '3,4' if name == 'teacher' else '3'))
    p.add_argument('--scenario', choices=('lcb-pass', 'math-runs', 'math-stop'), default='lcb-pass')
    if name != 'baseline':
        p.add_argument('--candidate', choices=s.ORDER, default='code')
        p.add_argument('--tokenizer', help='frozen Qwen3-8B tokenizer directory')
