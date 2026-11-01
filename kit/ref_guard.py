"""Reference guard provenance and per-(seed, panel) contrast checks (v5)."""
from pathlib import Path
import csv
import io
import json
import hashlib

METHOD = 'REFERENCE-SCORED'
PANELS = ('hellaswag', 'knowledge-ll')
ASSIGNMENT = 'ref-scoring-assignment.json'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def require(ok, reason):
    if not ok: raise ValueError(reason)


def local_files(out):
    out = Path(out)
    return {str(p.relative_to(out)): p.read_bytes() for p in out.rglob('*')
            if p.is_file() and (p.suffix in ('.json', '.csv') or p == out / 'hellaswag/panel.jsonl') and
            (p.is_relative_to(out / 'reference-scores') or p.name in (ASSIGNMENT, 'binding.json', 'm2-identity.json') or
             p == out / 'hellaswag/panel.jsonl' or p == out / 'inputs/base.json' or p.name.startswith('export-'))}


def records(files, standin=None):
    """Verify hash-bound reference units before allowing them to replace guard values."""
    from kit.m2_assignment import verify
    result = []
    for name, raw in sorted(files.items()):
        if not name.startswith('reference-scores/') or not name.endswith('/ref-identity.json'): continue
        prefix = name.rsplit('/', 1)[0] + '/'
        meta = json.loads(raw)
        require(meta['method'] == METHOD and meta['panel'] in PANELS, 'unknown reference method/panel')
        from kit.p3b_common import s, KIT
        implementation = KIT / ('hellaswag_reference.py' if meta['panel'] == 'hellaswag' else 'm3_score.py')
        require(meta['implementation_sha256'] == s.file_hash(implementation), 'reference implementation changed')
        if standin is not None:
            require(meta['technical_synthetic'] is standin, 'synthetic reference provenance differs')
        manifest = json.loads(files[prefix + 'manifest.json'])['files']
        for child in ('ref-identity.json', 'scores.csv', 'likelihoods.json', 'cost.json'):
            require(child in manifest and sha(files[prefix + child]) == manifest[child], 'reference constituent hash changed')
        try: verify(meta, files[ASSIGNMENT])
        except Exception as error: meta['_hardware_error'] = str(error)
        proof = meta['assignment']
        require(proof['family'] == meta['family'] + '.' + meta['panel'] and proof['seed'] == meta['seed'], 'foreign reference family/seed')
        unit = meta['arm'] if meta['family'] != 'm3' or meta['arm'] == 'base' else meta['arm'] + '.' + meta['stage']
        require(prefix == f"reference-scores/{meta['family']}/{unit}-s{meta['seed']}-k{meta['step']}-{meta['panel']}/", 'reference directory identity differs')
        require(meta['step'] in (0, 40), 'reference unit is not a decision checkpoint')
        validate_model(meta, files)
        rows = list(csv.DictReader(io.StringIO(files[prefix + 'scores.csv'].decode())))
        panel = 'knowledge' if meta['panel'] == 'knowledge-ll' else 'hellaswag'
        require(len(rows) == len({r['question_id'] for r in rows}) == 100, 'reference panel must contain 100 unique items')
        require(all(r['arm'] == unit and int(r['seed']) == meta['seed'] and int(r['step']) == meta['step'] and
                    r['panel'] == panel and r['correct'] in ('0', '1') for r in rows), 'reference CSV identity differs')
        from kit.m3_score import knowledge_items, selected
        likelihood = json.loads(files[prefix + 'likelihoods.json'])
        population = likelihood['items']
        require(len(population) == len({r['id'] for r in population}) == 100, 'reference population incomplete')
        if panel == 'knowledge':
            require(population == knowledge_items(), 'reference knowledge panel changed')
        else:
            from kit.p3b_common import s
            require(meta['panel_sha256'] == s.digest(population), 'reference HellaSwag population changed')
            # Compare to this root's frozen HellaSwag membership when available.
            frozen = files.get('hellaswag/panel.jsonl')
            require(meta['technical_synthetic'] or frozen is not None, 'reference HellaSwag freeze missing')
            if frozen:
                labels = {r['id']: 'ABCD'.index(r['gold']) for r in (json.loads(line) for line in io.StringIO(frozen.decode()) if line.strip())}
                require(labels == {r['id']: int(r['label']) for r in population}, 'foreign HellaSwag freeze')
        choices = likelihood['rows']
        require(len(choices) == len({r['id'] for r in choices}) == 100 and
                {r['id'] for r in choices} == {r['id'] for r in population} == {r['question_id'] for r in rows}, 'reference likelihood coverage differs')
        gold = {r['id']: int(r['label']) for r in population}
        correct = {r['id']: int(selected(r['scores']) == gold[r['id']]) for r in choices}
        require(all(int(r['correct']) == correct[r['question_id']] for r in rows), 'reference correctness differs from likelihoods')
        result.append((meta, rows))
    return result


def native_files(files):
    """View for the receipt-bound legacy validator, whose exact bytes stay intact.

    Verify reference evidence separately. Superseded native HellaSwag units may
    be partial because failed validation stopped their identity tail. Keep the
    original files in the archive; only exclude them from this validation view.
    """
    if not any(name.startswith('reference-scores/') for name in files): return files
    superseded = set()
    for meta, _ in records(files):
        if meta['panel'] != 'hellaswag': continue
        arm = meta['arm'] if meta['family'] != 'm3' or meta['arm'] == 'base' else meta['arm'] + '.' + meta['stage']
        unit = f"{arm}-s{meta['seed']}-k{meta['step']}-hellaswag/"
        superseded.add('scores/' + unit)
        superseded.add(f"references/{meta['family']}-s{meta['seed']}/scores/" + unit)
    return {name: value for name, value in files.items() if name != ASSIGNMENT and
            not name.startswith(('reference-scores/', *superseded))}


def overlay(rows, files, standin=None, family=None):
    """Keep generation diagnostics; replace only likelihood guard correctness, with in-memory provenance."""
    rows = [dict(r) for r in rows]
    if standin is None and 'binding.json' in files:
        standin = json.loads(files['binding.json']).get('technical_synthetic')
    identities = {}
    validations = {}
    for r in rows:
        if r['panel'] not in ('knowledge', 'hellaswag'): continue
        if standin is False and r['panel'] in ('knowledge', 'hellaswag'):
            r['_guard_validation'] = 'PENDING'
        panel = 'general-v1' if r['panel'] == 'knowledge' else r['panel']
        unit = f"scores/{r['arm']}-s{r['seed']}-k{r['step']}-{panel}/m2-identity.json"
        if unit in files:
            if unit not in identities:
                meta = json.loads(files[unit])
                identities[unit] = meta.get('assignment') or (meta.get('gpu_identity') if standin is not True else None) or {}
            proof = identities[unit]
            r.update(_guard_uuid=proof.get('uuid'), _guard_environment=proof.get('environment'))
            name = ('knowledge-' if r['panel'] == 'knowledge' else '') + f"{proof.get('family')}-s{r['seed']}.json"
            validation = files.get('format-validation/' + name)
            if validation:
                if name not in validations: validations[name] = json.loads(validation).get('status', 'PENDING')
                r['_guard_validation'] = validations[name]
    seen = set()
    for meta, scored in records(files, standin):
        if family is not None and meta['family'] != family: continue
        if family is None and meta['family'].startswith('x-') and meta['arm'] == 'base': continue
        for r in scored:
            key = (r['arm'], int(r['seed']), int(r['step']), r['panel'], r['question_id'])
            require(key not in seen, 'duplicate reference score unit'); seen.add(key)
        lookup = {r['question_id']: r for r in scored}
        selected_rows = [r for r in rows if (r['arm'], int(r['seed']), int(r['step']), r['panel']) ==
                         (scored[0]['arm'], meta['seed'], meta['step'], scored[0]['panel'])]
        require(not selected_rows or {r['question_id'] for r in selected_rows} == set(lookup), 'reference/generated panel membership differs')
        if not selected_rows:
            selected_rows = [dict(r) for r in scored]; rows.extend(selected_rows)
        for r in selected_rows:
            proof = meta['assignment']
            r.update(correct=lookup[r['question_id']]['correct'], format_free_correct=lookup[r['question_id']]['correct'],
                     _guard_method=METHOD, _guard_uuid=proof['uuid'], _guard_environment=proof['environment'],
                     _guard_family=meta['family'], _guard_hardware_error=meta.get('_hardware_error'))
    return rows


def validate_model(meta, files):
    """Bind reference scores to the same native checkpoint authority as generated scores."""
    if meta['technical_synthetic']: return
    arm, seed, step = meta['arm'], meta['seed'], meta['step']
    expected = None
    if meta['family'] == 'm3':
        from kit.m3_schedule import STAGES
        stage = meta['stage']
        if arm == 'base' or step == 0 and stage == STAGES[0]:
            expected = json.loads(files['inputs/base.json'])['identity']
        else:
            if step == 0: stage, step = STAGES[STAGES.index(stage) - 1], 40
            expected = json.loads(files[f'runs/{arm}-s{seed}-{stage}/export-{step}.json'])
    elif 'binding.json' in files:
        doc = json.loads(files['binding.json'])
        if meta['family'] == 'core':
            if arm == 'base': expected = doc['starts']['base']['identity']
            elif arm.startswith('finqa-'): expected = doc['starts'][f'{arm}:{seed}']['identity']
            elif arm.startswith('standalone-'): expected = doc['comparators'][f'{arm}:{seed}']['identity']
            elif arm == 'legal-plain': expected = doc['starts'][('legal-plain:' if step else 'finqa-plain:') + str(seed)]['identity']
            elif step == 0 and arm.startswith('legal-'): expected = doc['starts'][f'finqa-sibling:{seed}']['identity']
            elif step == 0 and arm == 'math-plain': expected = doc['starts'][f'legal-plain:{seed}']['identity']
            else:
                if step == 0:
                    arm = 'legal-' + json.loads(files['admission.json'])['configuration']; step = 40
                expected = json.loads(files[f'runs/{arm}-s{seed}/export-{step}.json'])
        else:
            require(meta['family'].startswith('x-tier'), 'unknown reference family')
            if arm == 'base':
                # The X inherited base is hash-bound by x_references; compare it there.
                identities = [json.loads(raw) for name, raw in files.items() if name.startswith('references/') and name.endswith('/m2-identity.json')]
                matches = [m['model'] for m in identities if m['arm'] == 'base' and m['seed'] == seed]
                require(matches, 'reference inherited base authority missing')
                expected = matches[0]
            elif step == 0: expected = doc['spec']['nodes'][f'b-s{seed}']['checkpoint']['identity']
            else: expected = json.loads(files[f'runs/{arm}-s{seed}/export-{step}.json'])
    require(expected is not None and meta['model']['model_file_hashes'] == expected['model_file_hashes'], 'reference checkpoint model differs from native export')


def check(rows):
    """One method for the likelihood guards; one physical GPU/environment per seed and panel."""
    rows = [r for r in rows if r['panel'] in ('knowledge', 'hellaswag') and int(r.get('step', 40)) in (0, 40)]
    methods = {}
    hardware = {}
    if any(r.get('_guard_hardware_error') for r in rows):
        return dict(status='UNVERIFIED-HARDWARE', reason='reference assignment UUID/environment differs')
    for r in rows:
        panel = r['panel']; method = r.get('_guard_method', 'KIT-SCORED')
        methods.setdefault(panel, set()).add(method)
        hardware.setdefault((int(r['seed']), panel), set()).add((r.get('_guard_uuid'), r.get('_guard_environment')))
    if len(set().union(*methods.values())) > 1:
        return dict(status='PENDING', reason='MIXED-SCORERS')
    for key, values in hardware.items():
        reference = methods[key[1]] == {METHOD}
        if len(values) > 1 or reference and any(not uuid or not env for uuid, env in values):
            return dict(status='UNVERIFIED-HARDWARE', reason='one GPU/environment required per (seed, panel)')
    if any(r.get('_guard_method') != METHOD for r in rows):
        return dict(status='PENDING', reason='REFERENCE-SCORED required for decision checkpoint guards')
    if any(METHOD in v for v in methods.values()):
        return dict(status='COMPLETE', method=METHOD)
    return dict(status='COMPLETE')


def apply(result, rows):
    proof = check(rows)
    if proof['status'] != 'COMPLETE': result.update(proof)
    elif proof.get('method'): result['method'] = proof['method']
    return result
