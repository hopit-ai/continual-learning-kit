#!/usr/bin/env python3
"""3C launcher. Default prints a free plan; --execute is a paid manager action.

Environment and subprocess isolation adapted from build/v4-27b-check and
build/v4-spider-v2-check. No campaign-specific gates or TP/precision fallbacks.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kit import second_task_screen as s, v4_teacher as t

VOLUME = 'sdft-outputs'
# Attempt-4 runtime/rounds on build/v4-27b-check; downloads from ledger rows 57–58.
# Round each measured proxy upward. Assume no retry attrition or 8B speedup.
STEP2_PROJECTION = {
    'two_cold_downloads': 2 * 218,
    'two_model_loads': 2 * 709,
    'teacher_four_rounds': 610 + 3 * 801,  # 100*(244/40), then ceil(100*176.14/22)
    'rewrite_four_rounds': 610 + 3 * 801,  # conservative 27B proxy, not measured 8B
    'sequential_verification': 800 * 40 * .05,  # 32,000 tests at assumed 50 ms/test
    'hashing_and_journals': 120,
}
STEP2_PROJECTED_SECONDS = int(sum(STEP2_PROJECTION.values()))  # 9,600; assumptions, not a bound
FINALIZE_SECONDS = 60
CAPS = {1: 1000, 2: 2 * STEP2_PROJECTED_SECONDS + FINALIZE_SECONDS}
STARTUP = 600  # larger image with the all-test file; model downloads/load are inside CAPS
IDLE = 2
ANNEX = 'docs/phase3/evidence/3c-second-task-screen/frozen-annex.json'


def price(step):
    gpus = 1 if step == 1 else 2
    rate = gpus * .001097 + 8 * .0000131 + 64 * .00000222
    seconds = CAPS[step] + STARTUP + IDLE
    expected = (300, 600) if step == 1 else (3000, STEP2_PROJECTED_SECONDS)
    total = 2 * sum(
        (CAPS[k] + STARTUP + IDLE) * ((1 if k == 1 else 2) * .001097 + 8 * .0000131 + 64 * .00000222) * 1.25 + .10
        for k in (1, 2))
    return {'step': step, 'gpus_allocated': gpus, 'function_seconds': CAPS[step],
        'effective_deadline_seconds': CAPS[step] - FINALIZE_SECONDS,
        'startup_seconds': STARTUP, 'idle_seconds': IDLE, 'rate_usd_s': rate,
        'reserved_usd': round(seconds * rate * 1.25 + .10, 4),
        'expected_usd_range': [round((n + STARTUP + IDLE) * rate + .10, 2) for n in expected],
        'both_candidates_reserved_usd': round(total, 4), 'ceiling_usd': 20,
        'exceeds_original_ceiling': total > 20,
        'additional_reservation_usd': round(max(0, total - 20), 4),
        'step2_projection_seconds': STEP2_PROJECTION,
        'projection_basis': 'cold starts; measured 27B proxy for both models; verification/IO assumptions; no guaranteed runtime',
        'price_source': 'https://modal.com/pricing', 'approval_pending': True}


def worker(prepared, candidate, phase, out):
    """A fresh subprocess per model releases all teacher VRAM before rewrites."""
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    from kit import eval_bed
    base, out = Path(prepared) / candidate, Path(out)
    out.mkdir(parents=True, exist_ok=True)
    role = 'teacher' if phase == 'teacher' else 'student'
    model, revision = s.MODELS[role]
    snapshot = snapshot_download(model, revision=revision, cache_dir='/tmp/3c-model-cache',
        allow_patterns=['*.json', '*.jinja', '*.safetensors', '*.model', '*.txt'])
    identity = t.local_identity(snapshot, model)
    if identity['model_revision'] != revision:
        raise ValueError('resolved model revision changed')
    t.atomic_json(out / (phase + '-identity.json'), identity)
    os.environ['V4_PHASE0_INFERENCE'] = '1'
    os.environ['KIT_DETERMINISTIC'] = '1'
    os.environ['CODE_TESTS'] = str(base / 'tests.jsonl')
    if phase == 'baseline':
        from types import SimpleNamespace
        panel = t.read_jsonl(base / 'heldout.jsonl')
        items = [{'id': r['id'], 'prompt': r['prompt'], 'ground_truth': r['gold']} for r in panel]
        args = SimpleNamespace(bed=candidate, root=str(base), split='heldout',
            model=snapshot, max_new_tokens=2048, max_model_len=12288,
            out=str(out / 'baseline'), limit=None, heldout_n=100, allow_subset=False)
        eval_bed.cmd_generate(args, frozen_items=items)
    else:
        student_model, student_revision = s.MODELS['student']
        token_path = snapshot_download(student_model, revision=student_revision,
            cache_dir='/tmp/3c-model-cache', allow_patterns=['*.json', '*.jinja', '*.model', '*.txt'])
        tokenizer = AutoTokenizer.from_pretrained(token_path)
        sample = t.read_jsonl(base / 'sample.jsonl')
        rewriting = phase == 'rewrite'
        generator = t.VLLMGenerator(snapshot, t.REWRITE_NEW_TOKENS if rewriting else t.TEACHER_NEW_TOKENS,
            input_cap=t.REWRITE_INPUT_CAP if rewriting else t.INPUT_CAP,
            tensor_parallel_size=1 if rewriting else 2, allow_batch_fallback=False)
        demos = t.read_jsonl(out / 'teacher/demonstrations.jsonl') if rewriting else None
        s.run_phase(sample, out / phase, generator, tokenizer, demonstrations=demos,
                    tests=str(base / 'tests.jsonl') if candidate == 'code' else None)
    t.atomic_json(out / (phase + '-complete.json'), {'complete': True})


def report(prepared, candidate, step, out, freeze_hash, error=None):
    base, out = Path(prepared) / candidate, Path(out)
    if step == 1:
        path = out / 'baseline/bed-score.json'
        records = []
        if path.exists():
            scores = json.loads(path.read_text())['per_item']
            records = [{'id': k, 'correct': bool(v)} for k, v in scores.items()]
        result = s.headroom(t.read_jsonl(base / 'heldout.jsonl'), records)
    else:
        def rows(path):
            return t.read_jsonl(out / path) if (out / path).exists() else []
        complete = all((out / (phase + '-complete.json')).exists() for phase in ['teacher', 'rewrite'])
        result = s.admission(t.read_jsonl(base / 'sample.jsonl'),
            rows('teacher/demonstrations.jsonl'), rows('rewrite/rewrites.jsonl'), complete=complete,
            pool=t.read_jsonl(base / 'train.jsonl') if (base / 'train.jsonl').exists() else None)
    if error:
        result.update(status='INCOMPLETE', complete=False, selected_for_partner_qualification=False, error=error)
    result.update(candidate=candidate, step=step, freeze_sha256=freeze_hash,
                  pricing=price(step), paid=True, technical_synthetic=False)
    t.atomic_json(out / 'decision.json', result)
    return result


def container(candidate, step, run_id, freeze_hash):
    """Bound total model/scoring time and preserve journals before container timeout."""
    import modal
    volume = modal.Volume.from_name(VOLUME)
    out = Path('/root/out') / run_id
    out.mkdir(parents=True, exist_ok=False)
    deadline = time.monotonic() + CAPS[step] - FINALIZE_SECONDS
    error = None
    t.atomic_json(out / 'start.json', {'candidate': candidate, 'step': step,
        'freeze_sha256': freeze_hash, 'pricing': price(step), 'started_unix': time.time()})
    volume.commit()
    try:
        for phase in (['baseline'] if step == 1 else ['teacher', 'rewrite']):
            command = [sys.executable, '/root/screen/scripts/second_task_screen_modal.py',
                '--worker', phase, '--prepared', '/root/prepared', '--candidate', candidate, '--out', str(out)]
            with (out / (phase + '.log')).open('w') as log:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    while process.poll() is None:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0: raise TimeoutError('registered time/cost cap exhausted')
                        try: process.wait(timeout=min(30, remaining))
                        except subprocess.TimeoutExpired: volume.commit()
                    if process.returncode: raise RuntimeError(f'{phase} exited {process.returncode}')
                finally:
                    if process.poll() is None:
                        os.killpg(process.pid, signal.SIGTERM)
                        try: process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL); process.wait()
            volume.commit()
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
    finally:
        result = report('/root/prepared', candidate, step, out, freeze_hash, error)
        volume.commit()
    return result


def check_dependencies(args, freeze_hash):
    """The registered scientific order, not an extra operational admission rule."""
    def read(path, candidate, step):
        if not path: raise ValueError('required earlier scientific decision is missing')
        row = json.loads(Path(path).read_text())
        if (row.get('candidate') != candidate or row.get('step') != step
            or row.get('freeze_sha256') != freeze_hash or not row.get('complete')
            or row.get('technical_synthetic') or not row.get('paid')):
            raise ValueError('earlier decision does not belong to this complete real screen')
        return row
    if args.candidate == 'math_screen':
        prior = json.loads(Path(args.previous).read_text()) if args.previous else {}
        if prior.get('step') not in (1, 2): raise ValueError('LCB decision required before MATH')
        if read(args.previous, 'code', prior['step'])['status'] != 'FAIL':
            raise ValueError('MATH follows a completed LCB failure only')
    if args.step == 2 and read(args.baseline, args.candidate, 1)['status'] != 'PASS':
        raise ValueError('teacher follows a complete baseline pass only')


def execute(args):
    manifest = s.load_frozen(args.prepared, args.annex)
    freeze_hash = s.digest(manifest)
    check_dependencies(args, freeze_hash)
    out = Path(args.out)
    if out.exists(): raise ValueError('use a new local evidence directory')
    # This import and all Modal resource declarations occur only after --execute.
    import modal
    app = modal.App('p3-3c-second-task-screen')
    if args.step == 1:
        # The 8B headroom scoring uses the image the 3A panel smoke ran kit/eval_bed.py's deterministic mode in
        # (12 Oct). vLLM 0.18 refuses batch-invariant mode unless the attention backend is passed as
        # attention_config, which eval_bed.py does not do (first step-1 attempt, 12 Oct: engine start refused).
        base = (modal.Image.from_registry('nvcr.io/nvidia/vllm:25.12.post1-py3', add_python=None)
            .pip_install('transformers==4.57.1', 'typing_extensions>=4.14'))
    else:
        base = (modal.Image.from_registry('nvidia/cuda:12.9.1-devel-ubuntu24.04', add_python='3.12')
            .pip_install('vllm==0.18.0', 'torch==2.10.0+cu129',
                         'torchaudio==2.10.0+cu129', 'torchvision==0.25.0+cu129',
                         extra_index_url='https://download.pytorch.org/whl/cu129'))
    image = (base
        .env({'PYTHONPATH': '/root/screen', 'OMP_NUM_THREADS': '8'})
        .add_local_dir(str(ROOT / 'kit'), '/root/screen/kit', copy=True)
        .add_local_file(__file__, '/root/screen/scripts/second_task_screen_modal.py', copy=True)
        .add_local_dir(str(Path(args.prepared).resolve()), '/root/prepared', copy=True))
    volume = modal.Volume.from_name(VOLUME, create_if_missing=True)
    fn = app.function(image=image, gpu='H100' if args.step == 1 else 'H100:2',
        cpu=(8, 8), memory=(65536, 65536), timeout=CAPS[args.step], startup_timeout=STARTUP,
        retries=0, max_containers=1, scaledown_window=IDLE, single_use_containers=True,
        volumes={'/root/out': volume})(container)
    run_id = 'p3-3c-' + args.candidate + '-step' + str(args.step) + '-' + str(time.time_ns())
    out.mkdir(parents=True)
    t.atomic_json(out / 'dispatch.json', {'run_id': run_id, 'freeze_sha256': freeze_hash, 'pricing': price(args.step)})
    try:
        with app.run():
            result = fn.remote(args.candidate, args.step, run_id, freeze_hash)
            print(json.dumps(result, indent=2))
    finally:
        collect(run_id, out)


def collect(run_id, out):
    import modal
    volume = modal.Volume.from_name(VOLUME)
    for entry in volume.iterdir(run_id, recursive=True):
        if entry.type != modal.volume.FileEntryType.FILE: continue
        relative = Path(entry.path.lstrip('/')).relative_to(run_id)
        if '..' in relative.parts: raise ValueError('invalid evidence path')
        path = Path(out) / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('wb') as stream:
            for chunk in volume.read_file(entry.path): stream.write(chunk)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate', choices=s.ORDER, default='code')
    p.add_argument('--step', type=int, choices=[1, 2], default=1)
    p.add_argument('--prepared', default='work/outputs/3c-screen-prepared')
    p.add_argument('--annex', default=ANNEX)
    p.add_argument('--out', default='work/outputs/3c-screen-code-step1')
    p.add_argument('--baseline'); p.add_argument('--previous')
    p.add_argument('--execute', action='store_true')
    p.add_argument('--collect', metavar='RUN_ID')
    p.add_argument('--worker', choices=['baseline', 'teacher', 'rewrite'], help=argparse.SUPPRESS)
    a = p.parse_args(argv)
    if a.worker: worker(a.prepared, a.candidate, a.worker, a.out)
    elif a.collect: collect(a.collect, a.out)
    elif a.execute: execute(a)
    else: print(json.dumps({'mode': 'DRY; no Modal imports or network', 'candidate': a.candidate, **price(a.step)}, indent=2))


if __name__ == '__main__':
    main()
