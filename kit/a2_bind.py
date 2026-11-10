"""Bind supplied node artifacts only. Never open a 4A reader or discover outcomes."""
from collections import Counter
from kit.a2_common import *
from kit import a4_ledger


def receipt(record):
    path = checked(record)
    require(path.name == 'manifest.json', 'input directory must be bound by its manifest hash')
    frozen_files(path.parent)
    return path.parent


def bind_inputs(cfg):
    synthetic = cfg.get('technical_synthetic', False)
    data = load(checked(cfg.get('inputs')))
    require(data.get('schema') == 'a2-node-input.v1', 'wrong A2 input schema')
    require(data.get('technical_synthetic') is synthetic, 'input synthetic identity differs')
    require(set(data['slice']) == set(data['ledger']) == set(data['contexts']) == set(MONTHS),
            'only complete January-June inputs permitted')
    audit = load(checked(data.get('audit')))
    require(audit.get('audited') is True and audit.get('months') == list(MONTHS),
            'audited 4A January-June slice required')
    require(audit.get('families') == list(FAMILIES), 'registered families changed')
    total = holdouts = 0
    arrival_context_bindings = {}
    ids = set()
    general = checked(data.get('general'))
    if not synthetic:
        require(s.file_hash(general) == s.file_hash(KIT/'panels/general-v1.jsonl'),
                'general-v1 panel changed')
    base = data.get('base')
    if not base:
        raise NotReady('base identity absent')
    base_identity = dict(model=base, technical_synthetic=True) if synthetic else identity(base, 'student')
    if synthetic:
        from kit.c_sim import Tokenizer
        tok = Tokenizer()
    else:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(base, local_files_only=True)
    for month in MONTHS:
        batch_path = checked(data['slice'][month])
        if not synthetic:
            expected = load(KIT/'resources/a4/slice.json')['files'][f'batch-{month}.json']
            require(s.file_hash(batch_path) == expected, 'slice differs from frozen 4A audit')
        batch = load(batch_path)
        require(audit.get('slice_hashes', {}).get(month) == s.file_hash(batch_path), 'slice not covered by audit')
        lookup = {i['id']: i for i in batch}
        require(len(lookup) == len(batch) and not ids.intersection(lookup), 'duplicate stream IDs')
        ids.update(lookup)
        total += len(batch)
        holdouts += sum(i['sec']['issuer_holdout'] for i in batch)
        require(all(i['sec']['template'] in FAMILIES and
                    i['sec']['accepted'].startswith(month) for i in batch), 'slice chronology/family')
        root = receipt(data['ledger'][month])
        require({'traces.jsonl', 'admission.jsonl', 'sealed.json'} <= set(load(root/'manifest.json')['files']),
                'raw ledger lacks admission or whole-month seal receipt')
        traces = t.read_jsonl(root/'traces.jsonl')
        require(len({r['id'] for r in traces}) == len(traces), 'duplicate ledger trace')
        for trace in traces:
            require(synthetic or (isinstance(trace.get('collector_checkpoint'), dict) and
                    trace['collector_checkpoint'].get('model_file_hashes') == base_identity['model_file_hashes']),
                    'raw ledger collector is not the frozen original base')
            require(trace['id'] in lookup and trace['item'] == lookup[trace['id']] and
                    trace['month'] == month, 'ledger/slice identity differs')
            raw = dict(response=trace['response'], finish_reason='stop',
                       output_tokens=len(t.tokenize_student(tok, [dict(role='user', content=a4_ledger.finqa.render_prompt(trace['item']))], trace['response'])[1]))
            require(not a4_ledger.admission(trace['item'], raw, tok), 'invalid raw ledger admission')
            require(trace['response_sha256'] == digest(trace['response'].encode()), 'raw response changed')
            require(trace.get('provenance', {}).get('unedited') is True, 'raw trace editing')
        # Bind the verified archive once, independently of later rewrite inputs.
        context = receipt(data['contexts'][month])
        rows = t.read_jsonl(context/'prompts.jsonl')
        require(len({r['id'] for r in rows}) == len(rows), 'duplicate archived contexts')
        arrival_context_bindings[month] = dict(
            manifest=dict(data['contexts'][month]), prompts_sha256=s.file_hash(context/'prompts.jsonl'),
            off_prompt_sha256={r['id']: digest(r['off']['prompt'].encode()) for r in rows
                               if isinstance(r.get('off', {}).get('prompt'), str)})
    require(synthetic or (total, holdouts) == (395, 92), 'audited slice denominators differ')
    result = dict(schema='a2-binding.v1', arm=cfg['arm'], inputs=data,
                  inputs_sha256=cfg['inputs']['sha256'], protocol=protocol(),
                  base_identity=base_identity, total=total, holdouts=holdouts,
                  general_sha256=s.file_hash(general), technical_synthetic=synthetic,
                  arrival_context_bindings=arrival_context_bindings,
                  registration_sha256=REGISTRATION, source='PARTNER-COMPUTED',
                  independent_reverification='UNVERIFIED')
    result['software'] = dict(trainer=t.runtime_versions(), inference=t.runtime_versions())
    if not synthetic:
        result['software']['inference'] = load(checked(data.get('inference_environment')))
    if cfg['arm'] == 'J':
        result['connected'] = connected(cfg.get('connected'), synthetic)
    if cfg['arm'] in ('P', 'RP'):
        result['i0'] = i0(cfg.get('i0'), synthetic)
    return result


def connected(record, synthetic):
    value = load(checked(record))
    if value.get('status') != 'VERIFIED':
        raise NotReady('connected trajectory not admitted')
    require(value.get('seeds') == list(SEEDS) and value.get('continuous') is True,
            'connected trajectory seeds/continuity')
    # Core reader dispositions and leakage receipts are inputs, not reader invocations.
    for key in ('recipe_dispositions', 'exclusions'):
        checked(value.get(key))
    exclusions = load(checked(value['exclusions']))
    require(exclusions.get('jan_jun_overlap') == [] and
            exclusions.get('oct_nov_overlap') == [] and
            exclusions.get('sealed_confirmation_manifest_sha256') and
            'jul_sep_exposure' in exclusions, 'connected SEC exclusion evidence invalid')
    for seed in SEEDS:
        lineage = value.get('lineages', {}).get(str(seed))
        if not lineage or not lineage.get('stages'):
            raise NotReady('continuous checkpoint lineage missing')
        parent = lineage['base']
        require(lineage.get('complete_registered_sequence') is True, 'missing connected stages')
        for stage in lineage['stages']:
            require(stage['parent'] == parent and not stage.get('reset') and
                    stage.get('seed') == seed, 'reset/stitched connected ancestry')
            for key in ('acquisition', 'effective_config', 'data_manifest', 'schedule_manifest', 'export'):
                checked(stage.get(key))
            if not synthetic:
                identity(stage['final'], receipt=stage['export']['path'])
            parent = stage['final']
        require(parent == lineage['terminal'], 'connected terminal mismatch')
    return value


def i0(record, synthetic):
    value = load(checked(record))
    if value.get('verified') is not True:
        raise NotReady('I0 pool/target/map verification pending')
    if value.get('source') != 'I0' or value.get('stage') != 'terminal-contractnli':
        raise NotReady('I0 terminal ContractNLI/legal source ambiguous')
    require(value.get('seeds') == list(SEEDS), 'I0 seed mapping changed')
    for key in ('pool', 'stored_targets', 'draw_rules', 'overlap_receipt'):
        checked(value.get(key))
    for seed in SEEDS:
        source = value.get('maps', {}).get(str(seed))
        if not source:
            raise NotReady('I0 realised seed map missing')
        checked(source)
        spec = load(checked(source))
        if any(k not in spec for k in ('seed', 'slots', 'accepted_order', 'skipped_draws')):
            raise NotReady('I0 realised map/ordering/draw receipt incomplete')
        require(spec.get('seed') == seed and len(spec['slots']) == 40, 'I0 realised map length/seed')
        require(all(len(set(slots)) == len(slots) and all(type(p) is int and 1 <= p <= 32
                    for p in slots) for slots in spec['slots']), 'I0 generic positions invalid')
        require(spec.get('accepted_order') is not None and spec.get('skipped_draws') is not None,
                'I0 realised ordering/draw evidence absent')
    return value


def binding(out):
    path = Path(out)/'binding.json'
    if not path.exists():
        raise NotReady('A2 binding absent')
    result = load(path)
    require(result['protocol'] == protocol(), 'bound implementation changed')
    return result


def batch(doc, month):
    return sorted(load(checked(doc['inputs']['slice'][month])),
                  key=lambda i: (i['sec']['accepted'], i['sec']['accession'], i['id']))


def ledger(doc, through):
    return [r for month in MONTHS[:through]
            for r in t.read_jsonl(receipt(doc['inputs']['ledger'][month])/'traces.jsonl')]


def contexts(doc, month):
    root = receipt(doc['inputs']['contexts'][month])
    return {r['id']: r for r in t.read_jsonl(root/'prompts.jsonl')}
