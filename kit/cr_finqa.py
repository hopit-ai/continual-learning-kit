"""Confirmation-only adapter to B's unchanged native FinQA schedule builder.

B's slot registry ends at 105. The isolated worker routes the same FinQA job
with only its seed substituted; the ordering and SFT writer run unchanged.
"""
from pathlib import Path
import sys
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import os
import subprocess
import tempfile
from kit.nm_common import KIT, atomic, load, sha, ready, Pending

PROTOCOL = 'b-native-finqa-regeneration.v1'
B_CODE = ('v4_campaign_ops.py', 'v4_campaign.py', 'v4_sft.py', 'v4_teacher.py',
          'v4_contract.py', 'v4_finqa_policy.py', 'next_pools.py', 'beds/finqa.py')
OLD = (101, 102, 103)
FRESH = (104, 105, 106)


def native_paths(source):
    """Discover optional stored B orders on the bound node, never trust omission."""
    base = Path(source['b_work']) / 'v4/report-main'
    paths = [base/'common/manifest.json', base/'common/finqa.parquet',
             Path(source['b_work'])/'v4/datasets/datasets/v4_finqa/test.parquet']
    ready(Path(source['corpus']).resolve() == paths[1].resolve(), 'FinQA corpus must be B original COMMON parquet; owner escalation')
    for seed in (104, 105):
        order = base/f'finqa-s{seed}-schedule.json'
        folder = base/f'scheduled/finqa-s{seed}'
        if order.exists(): paths.append(order)
        if folder.exists(): paths.extend(folder/name for name in ('manifest.json', 'train.parquet', 'val.parquet'))
    return paths


class StandinTokenizer:
    """CPU fixture only; synthetic rows cannot qualify scientific schedules."""
    eos_token, eos_token_id, pad_token_id = '<eos>', 2, 0
    def encode(self, text, **kwargs):
        return [8]*len(text[:-5])+[2] if text.endswith(self.eos_token) else [8]*len(text)
    def apply_chat_template(self, messages, tokenize=False, **kwargs):
        text = '[user]'+''.join(m['content'] for m in messages)+'[assistant]'
        return self.encode(text) if tokenize else text


def native_worker(request):
    """Run in a disposable process: no B module/global changes reach M3/v7."""
    from kit import v4_campaign_ops as b, v4_sft
    from kit.m3_pools import finqa_rows
    import pyarrow.parquet as pq
    cfg = load(request); source = cfg['source']; work = Path(cfg['work'])
    original = Path(source['b_work']); base = b.paths(work, 'main')
    base.mkdir(parents=True)
    (base/'common').symlink_to(b.paths(original, 'main')/'common', target_is_directory=True)
    test = work/'v4/datasets/datasets/v4_finqa/test.parquet'
    test.parent.mkdir(parents=True)
    test.symlink_to(original/'v4/datasets/datasets/v4_finqa/test.parquet')
    if cfg['standin']:
        tokenizer = StandinTokenizer()
        v4_sft.load_student_tokenizer = lambda *args: tokenizer
    else:
        from kit.m3_train import tokenizer_for
        from types import SimpleNamespace
        tokenizer = tokenizer_for(SimpleNamespace(standin=False, tokenizer=cfg['tokenizer']))
    os.environ['QWEN3_8B_TOKENIZER'] = cfg.get('tokenizer') or 'CPU-STANDIN'
    template = b.job_for('D-finqa-alone-s101', 'main')
    original_job_for = b.job_for
    for seed in (*OLD, *FRESH):
        # Only dispatch metadata changes. B's schedule function and sampler are
        # called directly, with phase='main' and the original COMMON bytes.
        b.job_for = lambda ident, phase, seed=seed: dict(template, seed=seed)
        try: b.schedule(work, 'main', template['id'])
        finally: b.job_for = original_job_for
        folder = base/f'scheduled/finqa-s{seed}'
        verified, _ = finqa_rows([folder], False, tokenizer)
        lookup = {r['id']: r for r in verified}
        ordered = [lookup[str(r['extra_info']['index'])] for r in pq.read_table(folder/'train.parquet').to_pylist()]
        # Same projection/writer as m3_pools.finqa_task_schedules/finqa_targets.
        b.t.write_jsonl(work/f'schedule-{seed}.jsonl', ordered)


def reproduce(source, cfg, attempt):
    """Publish proof even on failure; release fresh JSONL only after every gate."""
    attempt = Path(attempt); attempt.mkdir(parents=True, exist_ok=True)
    proof = dict(protocol=PROTOCOL, status='PENDING', escalation='owner', checks=[],
                 regenerated={}, errors=[], code={name:sha(KIT/name) for name in B_CODE})
    def compare(label, expected, observed):
        expected = Path(expected); observed = Path(observed)
        same = expected.is_file() and observed.is_file() and expected.read_bytes() == observed.read_bytes()
        proof['checks'].append(dict(name=label, expected=sha(expected) if expected.is_file() else None,
                                    observed=sha(observed) if observed.is_file() else None, byte_equal=same))
        if not same: proof['errors'].append('FinQA '+label+' mismatch; owner escalation')
    try:
        ready(source.get('algorithm') == PROTOCOL, 'FinQA native algorithm missing; owner escalation')
        ready(set(source.get('v6a_schedules', {})) == set(map(str, OLD)), 'FinQA v6a schedules missing; owner escalation')
        for path in native_paths(source):
            ready(path.is_file(), 'FinQA native input missing: '+str(path)+'; owner escalation')
        with tempfile.TemporaryDirectory(prefix='finqa-native-', dir=attempt) as temporary:
            scratch = Path(temporary); work = scratch/'work'
            request = scratch/'request.json'
            atomic(request, dict(source=source, work=str(work), tokenizer=cfg.get('tokenizer'),
                                 standin=bool(cfg.get('technical_synthetic'))))
            done = subprocess.run([cfg.get('job_python', sys.executable), '-I', str(KIT/'cr_finqa.py'), str(request)],
                                  capture_output=True, text=True, timeout=600)
            ready(done.returncode == 0, 'FinQA B native procedure failed: '+done.stderr[-2000:]+'; owner escalation')
            base = Path(source['b_work'])/'v4/report-main'
            generated = work/'v4/report-main'
            for seed in OLD:
                compare('v6a-'+str(seed), source['v6a_schedules'][str(seed)], work/f'schedule-{seed}.jsonl')
            for seed in (104, 105):
                order = base/f'finqa-s{seed}-schedule.json'
                folder = base/f'scheduled/finqa-s{seed}'
                if order.exists(): compare('B-order-'+str(seed), order, generated/order.name)
                if folder.exists():
                    # B's stored IDs and targets are compared after exactly the
                    # same M3 projection, rather than parquet encoding metadata.
                    from kit.m3_pools import finqa_rows
                    from kit.p3b_common import t
                    import pyarrow.parquet as pq
                    tokenizer = StandinTokenizer() if cfg.get('technical_synthetic') else None
                    if tokenizer is None:
                        from kit.m3_train import tokenizer_for
                        from types import SimpleNamespace
                        tokenizer = tokenizer_for(SimpleNamespace(standin=False, tokenizer=cfg.get('tokenizer')))
                    manifest = load(folder/'manifest.json')
                    ready(manifest.get('seed') == seed and manifest.get('schema') == 'kit-v4-sft-data.v1'
                          and manifest.get('arm') == 'COMMON', 'FinQA B stored manifest differs; owner escalation')
                    ready(sha(folder/'val.parquet') == manifest.get('val_sha256'), 'FinQA B stored val hash differs; owner escalation')
                    rows = pq.read_table(folder/'train.parquet').to_pylist()
                    ready([r['extra_info']['index'] for r in rows] == manifest.get('ids'), 'FinQA B stored order differs; owner escalation')
                    verified, _ = finqa_rows([folder], False, tokenizer)
                    lookup = {r['id']:r for r in verified}
                    projected = scratch/f'B-{seed}.jsonl'
                    t.write_jsonl(projected, [lookup[str(r['extra_info']['index'])] for r in rows])
                    compare('B-targets-'+str(seed), projected, work/f'schedule-{seed}.jsonl')
            proof['regenerated'] = {str(seed):dict(schedule_sha256=sha(work/f'schedule-{seed}.jsonl'),
                order_sha256=sha(generated/f'finqa-s{seed}-schedule.json')) for seed in FRESH}
            ready(not proof['errors'], '; '.join(proof['errors']))
            # No consumer-visible fresh schedules exist before the proof passes.
            for seed in FRESH:
                destination = attempt/'finqa-generated'/f'schedule-{seed}.jsonl'
                destination.parent.mkdir(exist_ok=True)
                destination.write_bytes((work/destination.name).read_bytes())
            proof.update(status='COMPLETE', escalation=None)
    except (OSError, ValueError, KeyError, Pending, subprocess.TimeoutExpired) as error:
        if str(error) not in proof['errors']: proof['errors'].append('FinQA '+str(error)+'; owner escalation')
    atomic(attempt/'finqa-reproduction.json', proof)
    return proof


if __name__ == '__main__':
    native_worker(Path(sys.argv[1]))
