"""Confirmation-only adapter to B's unchanged native FinQA schedule builder.

B's slot registry ends at 105. The isolated worker routes the same FinQA job
with only its seed substituted; the ordering and SFT writer run unchanged.
"""
from pathlib import Path
import sys
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import os
import json
import math
import re
import subprocess
import tempfile
from kit.nm_common import KIT, atomic, load, sha, raw_hash, ready, Pending

PROTOCOL = 'b-native-finqa-regeneration.v1'
B_CODE = ('v4_campaign_ops.py', 'v4_campaign.py', 'v4_sft.py', 'v4_teacher.py',
          'v4_contract.py', 'v4_finqa_policy.py', 'next_pools.py', 'beds/finqa.py')
OLD = (101, 102, 103)
FRESH = (104, 105, 106)


def receipt_value(value, errors, label='input', depth=0):
    """Bound receipt depth and replace values canonical JSON cannot represent."""
    if depth > 64:
        errors.append(label+' exceeds maximum nesting depth')
        return '<excessive nesting>'
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if not isinstance(key, str):
                errors.append(label+' has a non-string key')
                continue
            result[key] = receipt_value(item, errors, label+'.'+key, depth+1)
        return result
    if isinstance(value, list):
        return [receipt_value(item, errors, label+'[]', depth+1) for item in value]
    errors.append(label+' contains a non-finite or unsupported value')
    return None


class InputChecks:
    """One boundary for operator-controlled paths, reads, hashes and resolution."""
    def __init__(self, errors):
        self.errors = errors

    def access(self, operation, value, label='input'):
        try:
            if not isinstance(value, (str, Path)) or not str(value):
                raise ValueError('nonempty path required')
            path = Path(value)
            if operation == 'path': return path
            if operation == 'resolve': return path.resolve()
            if operation == 'sha': return sha(path)
            if operation == 'read': return path.read_bytes()
            if operation == 'exists': return path.exists()
            if operation == 'members': return [p for p in path.rglob('*') if p.is_file()]
            raise ValueError('unknown input operation '+operation)
        except (OSError, RuntimeError, ValueError, RecursionError) as error:
            self.errors.append(label+' '+operation+' failed: '+type(error).__name__+' '+str(error))
            return None

    @staticmethod
    def valid_digest(value):
        return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None

    def document(self, path, expected, label):
        # Parse only the exact bytes whose digest was successfully verified.
        if not self.valid_digest(expected):
            self.errors.append(label+' digest invalid')
            return None
        raw = self.access('read', path, label)
        if raw is None: return None
        if raw_hash(raw) != expected:
            self.errors.append(label+' hash mismatch')
            return None
        try:
            doc = receipt_value(json.loads(raw), self.errors, label)
            if not isinstance(doc, dict):
                raise ValueError('JSON object required')
            return doc
        except (ValueError, RecursionError) as error:
            self.errors.append(label+' JSON invalid: '+type(error).__name__+' '+str(error))
            return None


def native_paths(source):
    """Discover optional stored B orders on the bound node, never trust omission."""
    errors = []
    checked = InputChecks(errors)
    ready(isinstance(source, dict), 'FinQA task source must be an object; owner escalation')
    work = checked.access('path', source.get('b_work'), 'FinQA b_work')
    corpus = checked.access('resolve', source.get('corpus'), 'FinQA corpus')
    ready(not errors, '; '.join(errors)+'; owner escalation')
    base = work / 'v4/report-main'
    paths = [base/'common/manifest.json', base/'common/finqa.parquet',
             work/'v4/datasets/datasets/v4_finqa/test.parquet']
    common = checked.access('resolve', paths[1], 'FinQA COMMON')
    ready(not errors, '; '.join(errors)+'; owner escalation')
    ready(corpus == common, 'FinQA corpus must be B original COMMON parquet; owner escalation')
    for seed in (104, 105):
        order = base/f'finqa-s{seed}-schedule.json'
        folder = base/f'scheduled/finqa-s{seed}'
        if checked.access('exists', order, 'FinQA order'): paths.append(order)
        if checked.access('exists', folder, 'FinQA folder'):
            paths.extend(folder/name for name in ('manifest.json', 'train.parquet', 'val.parquet'))
    ready(not errors, '; '.join(errors)+'; owner escalation')
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


def reproduce(source, cfg, attempt, *, verified_inventory=None):
    """Publish proof even on failure; release fresh JSONL only after every gate."""
    attempt = Path(attempt)
    proof = dict(protocol=PROTOCOL, status='PENDING', escalation='owner', checks=[],
                 regenerated={}, errors=[])
    checked = InputChecks(proof['errors'])

    def bound_bytes(path):
        resolved = checked.access('resolve', path, 'FinQA bound input')
        expected = inventory.get(str(resolved)) if resolved is not None else None
        ready(checked.valid_digest(expected), 'FinQA unbound native input: '+str(path)+'; owner escalation')
        raw = checked.access('read', path, 'FinQA bound input')
        ready(raw is not None and raw_hash(raw) == expected,
              'FinQA native input changed/unreadable: '+str(path)+'; owner escalation')
        return raw

    def compare(label, expected, observed, *, bound=True):
        before = bound_bytes(expected) if bound else checked.access('read', expected, 'FinQA '+label)
        after = checked.access('read', observed, 'FinQA regenerated '+label)
        same = before is not None and after is not None and before == after
        proof['checks'].append(dict(name=label, expected=raw_hash(before) if before is not None else None,
                                    observed=raw_hash(after) if after is not None else None, byte_equal=same))
        if not same: proof['errors'].append('FinQA '+label+' mismatch; owner escalation')

    try:
        cfg = receipt_value(cfg, proof['errors'], 'FinQA configuration')
        source = receipt_value(source, proof['errors'], 'FinQA source')
        ready(isinstance(cfg, dict) and isinstance(source, dict), 'FinQA configuration/source must be objects; owner escalation')
        ready(source.get('algorithm') == PROTOCOL, 'FinQA native algorithm missing; owner escalation')
        schedules = source.get('v6a_schedules')
        ready(isinstance(schedules, dict) and set(schedules) == set(map(str, OLD)),
              'FinQA v6a schedules missing; owner escalation')
        python = cfg.get('job_python', sys.executable)
        ready(isinstance(python, str) and bool(python), 'FinQA job_python must be a nonempty string; owner escalation')
        if not cfg.get('technical_synthetic'):
            ready(checked.access('path', cfg.get('tokenizer'), 'FinQA tokenizer') is not None,
                  'FinQA tokenizer path invalid; owner escalation')
        proof['code'] = {name:sha(KIT/name) for name in B_CODE}
        paths = native_paths(source)
        inventory = verified_inventory
        if inventory is None:
            # Standalone callers pin their initial discovery. bind_inputs passes
            # only the digest-verified operator inventory.
            inventory = {}
            for path in (*paths, *schedules.values()):
                resolved = checked.access('resolve', path, 'FinQA initial input')
                expected = checked.access('sha', path, 'FinQA initial input')
                if resolved is not None: inventory[str(resolved)] = expected
        ready(isinstance(inventory, dict), 'FinQA verified inventory must be an object; owner escalation')
        for path in (*paths, *schedules.values()): bound_bytes(path)
        ready(not proof['errors'], '; '.join(proof['errors']))
        attempt.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='finqa-native-', dir=attempt) as temporary:
            scratch = Path(temporary); work = scratch/'work'
            request = scratch/'request.json'
            atomic(request, dict(source=source, work=str(work), tokenizer=cfg.get('tokenizer'),
                                 standin=bool(cfg.get('technical_synthetic'))))
            done = subprocess.run([python, '-I', str(KIT/'cr_finqa.py'), str(request)],
                                  capture_output=True, text=True, timeout=600)
            ready(done.returncode == 0, 'FinQA B native procedure failed: '+done.stderr[-2000:]+'; owner escalation')
            # Refuse optional files that appeared while the worker ran. Every
            # parent read below checks the digest of the bytes actually consumed.
            for path in native_paths(source): bound_bytes(path)
            # The unchanged B worker reads symlinked COMMON/tokenizer inputs.
            # Pre/post checks cannot exclude a change-and-restore during that
            # subprocess. A stable snapshot of those large inputs is still needed
            # to eliminate that remaining TOCTOU window.
            base = Path(source['b_work'])/'v4/report-main'
            generated = work/'v4/report-main'
            for seed in OLD:
                compare('v6a-'+str(seed), schedules[str(seed)], work/f'schedule-{seed}.jsonl')
            for seed in (104, 105):
                order = base/f'finqa-s{seed}-schedule.json'
                folder = base/f'scheduled/finqa-s{seed}'
                if checked.access('exists', order, 'FinQA stored order'):
                    compare('B-order-'+str(seed), order, generated/order.name)
                if checked.access('exists', folder, 'FinQA stored folder'):
                    from kit.m3_pools import finqa_rows
                    from kit.p3b_common import t
                    import pyarrow as pa
                    import pyarrow.parquet as pq
                    # Use one verified snapshot for both schema checks and M3's
                    # projection, so its own reads cannot consume changed bytes.
                    snapshot = scratch/f'B-{seed}'
                    snapshot.mkdir()
                    for name in ('manifest.json', 'train.parquet', 'val.parquet'):
                        (snapshot/name).write_bytes(bound_bytes(folder/name))
                    manifest = checked.document(snapshot/'manifest.json',
                        raw_hash((snapshot/'manifest.json').read_bytes()), 'FinQA B stored manifest')
                    ready(manifest is not None, 'FinQA B stored manifest must be an object; owner escalation')
                    ready(manifest.get('seed') == seed and manifest.get('schema') == 'kit-v4-sft-data.v1'
                          and manifest.get('arm') == 'COMMON', 'FinQA B stored manifest differs; owner escalation')
                    ready(sha(snapshot/'val.parquet') == manifest.get('val_sha256'), 'FinQA B stored val hash differs; owner escalation')
                    rows = pq.read_table(pa.BufferReader((snapshot/'train.parquet').read_bytes())).to_pylist()
                    ready(isinstance(rows, list) and all(isinstance(r, dict) and isinstance(r.get('extra_info'), dict)
                          and type(r['extra_info'].get('index')) in (str, int) for r in rows),
                          'FinQA B stored training rows/extra_info invalid; owner escalation')
                    indices = [r['extra_info']['index'] for r in rows]
                    ready(indices == manifest.get('ids'), 'FinQA B stored order differs; owner escalation')
                    tokenizer = StandinTokenizer() if cfg.get('technical_synthetic') else None
                    if tokenizer is None:
                        from kit.m3_train import tokenizer_for
                        from types import SimpleNamespace
                        tokenizer = tokenizer_for(SimpleNamespace(standin=False, tokenizer=cfg.get('tokenizer')))
                    verified, _ = finqa_rows([snapshot], False, tokenizer)
                    lookup = {r['id']:r for r in verified}
                    projected = scratch/f'B-{seed}.jsonl'
                    t.write_jsonl(projected, [lookup[str(index)] for index in indices])
                    compare('B-targets-'+str(seed), projected, work/f'schedule-{seed}.jsonl', bound=False)
            proof['regenerated'] = {str(seed):dict(schedule_sha256=checked.access('sha', work/f'schedule-{seed}.jsonl', 'FinQA fresh schedule'),
                order_sha256=checked.access('sha', generated/f'finqa-s{seed}-schedule.json', 'FinQA fresh order')) for seed in FRESH}
            ready(not proof['errors'], '; '.join(proof['errors']))
            for seed in FRESH:
                destination = attempt/'finqa-generated'/f'schedule-{seed}.jsonl'
                destination.parent.mkdir(exist_ok=True)
                destination.write_bytes((work/destination.name).read_bytes())
            proof.update(status='COMPLETE', escalation=None)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, RecursionError, Pending, subprocess.TimeoutExpired) as error:
        proof['errors'].append('FinQA '+type(error).__name__+' '+str(error)+'; owner escalation')
    except Exception as error:
        proof['errors'].append('FinQA input checking failed: '+type(error).__name__+' '+str(error)+'; owner escalation')
    sanitization_errors = []
    proof = receipt_value(proof, sanitization_errors, 'FinQA proof')
    proof['errors'].extend(sanitization_errors)
    # Receipt publication failures remain loud, outside the input boundary.
    atomic(attempt/'finqa-reproduction.json', proof)
    return proof


if __name__ == '__main__':
    native_worker(Path(sys.argv[1]))
