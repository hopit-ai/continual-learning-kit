#!/usr/bin/env python3
"""Independent A2 archives and pasted text parts; CPU imports recompute all bars."""
from pathlib import Path
import sys
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import base64
import copy
import gzip
import io
import re
import tarfile
import tempfile
import uuid
from kit.a2_common import *
from kit.a2_read import analyse
from kit.text_return import frame_text


def reproduce(payload, capture=None):
    require(payload['arm'] in ('J', 'P', 'R', 'RP') and payload['registration_sha256'] == REGISTRATION,
            'return registration/arm differs')
    if payload.get('status') in ('ARCHIVED UNREAD', 'NOT APPLICABLE', 'PENDING'):
        return dict(arm=payload['arm'], status=payload['status'], provenance='PARTNER-COMPUTED',
                    independent_reverification='UNVERIFIED',
                    **{k: payload[k] for k in ('accounting', 'qualification', 'identity') if k in payload})
    result = analyse(payload['arm'], payload.get('cells', []), synthetic=payload['technical_synthetic'],
                     confirmation=payload.get('confirmation', False), receipt=payload.get('confirmation_receipt'),
                     p_applicable=payload.get('p_applicable', True), assignment=payload.get('assignment'),
                     counts=payload.get('keyed_counts'), capture=capture)
    if 'identity' in payload:
        result['identity'] = payload['identity']
    if 'p_applicable' in payload:
        result['p_applicable'] = payload['p_applicable']
    result['accounting'] = payload['accounting']
    if 'qualification' in payload:
        result['qualification'] = payload['qualification']
    if 'diagnostics' in payload:
        result['diagnostics'] = diagnostic_summary(payload['diagnostics'])
    return result


def write_parts(directory, payload, commit):
    require(re.fullmatch('[0-9a-f]{40}', commit), 'kit commit must be a full git identity')
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    # Both transports carry one fresh collect identity. It is not a checksum.
    transport = dict(collect=uuid.uuid4().hex, kit_commit=commit,
                     phase='confirmation' if payload.get('confirmation') else 'exploratory')
    full = dict(payload, return_set=transport)
    essential = dict(compact_counts(payload), return_set=transport)
    reading = reproduce(payload)
    rows = ['T|PARTNER-COMPUTED|independent-reverification=UNVERIFIED',
            'T|TRANSCRIPTION-UNVERIFIED', 'T|CHECKSUM-SCOPE|advisory-only']
    if payload.get('qualification'):
        price = payload['qualification']
        rows.extend(['T|A2-QUALIFICATION-SPEND|'+str(price['qualification_gpu_hours']),
                     'T|A2-FULL-PRICE|'+str(price.get('total_gpu_hours', 'NOT-PRICED')),
                     'T|A2-PRICE-DECISION|'+price['decision'].replace(' ', ':')])
    for label, contrast in reading.get('contrasts', {}).items():
        primary = contrast['primary']
        verdict = reading['status'] if reading['status'] != 'COMPLETE' else contrast['verdict']
        rows.append('|'.join(['D', 'A2-'+payload['arm'], label, verdict.replace(' ', ':'),
                              *[canonical(primary[k]).decode() for k in ('mean', 'lower', 'upper', 'positive')], '-']))
        for name, kind, result, bar in [('primary', 'P', primary, 'mean>=2;LB>0;pos=3'),
                *[(k, 'G', v, 'LB>=-1') for k, v in contrast['guards'].items()]]:
            rows.append('|'.join(['E', 'A2-'+payload['arm'], label+'.'+name, kind,
                *[canonical(result[k]).decode() for k in ('values', 'mean', 'sd', 'lower', 'upper', 'positive', 'n')], bar, 'two']))

    def chunks(value):
        packed = base64.b64encode(gzip.compress(canonical(value), mtime=0)).decode('ascii')
        pieces = [packed[i:i+3000] for i in range(0, len(packed), 3000)]
        records = rows + ['T|A2|packed|'+canonical([i, len(pieces), piece]).decode()
                          for i, piece in enumerate(pieces)]
        result, size = [[]], 0
        for line in records:
            require(len(line) < 10000, 'logical text record too large')
            if size+len(line)+len(line)//89+1 > 10500:
                result.append([])
                size = 0
            result[-1].append(line)
            size += len(line)+len(line)//89+1
        return result

    full_chunks = chunks(full)
    reference = f'T|FULL-SET|{len(full_chunks)}|{digest(full_chunks)}'
    rendered = []
    for mode, values in [('ESSENTIAL', chunks(essential)), ('FULL', full_chunks)]:
        for index, chunk in enumerate(values, 1):
            value = frame_text('a2-'+payload['arm'].lower(), commit, index, len(values), mode,
                               reference, ['T|A2-SET|'+transport['collect'], *chunk], width=89)
            require(value.isascii() and len(value) <= 11400 and all(len(line) < 90 for line in value.splitlines()),
                    'text part overflow')
            rendered.append((f'RETURN-TEXT-{mode}-{index}-of-{len(values)}.md', value))
    paths = [directory/name for name, _ in rendered]
    with tempfile.TemporaryDirectory(prefix='.return-parts-', dir=directory) as scratch:
        staged = Path(scratch)
        for name, value in rendered:
            (staged/name).write_text(value)
        with lock(directory/'.return-lock'):
            older = sorted(directory.glob('RETURN-TEXT-*.md'))
            superseded = directory/'superseded'/uuid.uuid4().hex
            moved = []
            published = []
            try:
                if older:
                    superseded.mkdir(parents=True)
                for path in older:
                    path.replace(superseded/path.name)
                    moved.append(path)
                for path in paths:
                    (staged/path.name).replace(path)
                    published.append(path)
            except BaseException:
                for path in published:
                    path.unlink()
                for path in moved:
                    (superseded/path.name).replace(path)
                raise
    essential_count = sum('-ESSENTIAL-' in path.name for path in paths)
    print(f'A2 {payload["arm"]} {transport["phase"]} ESSENTIAL parts: {essential_count}', flush=True)
    if essential_count > 4:
        print('WARNING: ESSENTIAL exceeds 4 parts', flush=True)
    return paths


def _import_parts(texts, payload_only=False):
    parts = {}
    original_texts = {}
    identity = None
    checksum_status = []
    n = None
    for text in texts:
        lines = text.splitlines()
        flat = re.sub(r'\s+', '', text)
        match = re.match(r'KIT(a2-[a-z]+)([0-9a-f]{40})PART([0-9]+)/([0-9]+)', flat)
        require(match is not None, 'invalid A2 part header')
        own = (match[1], match[2])
        require(identity in (None, own), 'mixed A2 kit identities')
        identity = own
        index, count = int(match[3]), int(match[4])
        require(1 <= index <= count and n in (None, count), 'duplicate/mixed part numbering')
        if index in parts:
            require(original_texts[index] == text, 'duplicate/mixed part numbering')
            continue
        n = count
        original_texts[index] = text
        body, _, checksum = flat[match.end():].rpartition('SHA256')
        require(bool(checksum), 'part checksum field absent')
        # Digests are advisory, even when the content was transcribed differently.
        records = re.findall(r'T\|A2\|(meta|cell|item|diagnostic|fragment|packed)\|(.+?)(?=T\|A2\||$)', body)
        parsed = []
        for kind, raw in records:
            parsed.append((kind, json.loads(raw)))
        parts[index] = parsed
        checksum_status.append('TRANSCRIPTION-UNVERIFIED')
    require(n is not None and set(parts) == set(range(1, n+1)), 'INCOMPLETE: missing A2 text parts')
    packed_rows = [v for index in sorted(parts) for kind, v in parts[index] if kind == 'packed']
    if packed_rows:
        require(all(kind == 'packed' for rows in parts.values() for kind, _ in rows), 'mixed packed/legacy text')
        count = packed_rows[0][1]
        require(type(count) is int and count > 0 and all(v[1] == count for v in packed_rows), 'packed chunk counts differ')
        chunks = {v[0]: v[2] for v in packed_rows}
        require(len(chunks) == len(packed_rows) and set(chunks) == set(range(count)), 'duplicate/missing packed chunks')
        packed = base64.b64decode(''.join(chunks[i] for i in range(count)), validate=True)
        with gzip.GzipFile(fileobj=io.BytesIO(packed)) as stream:
            raw = stream.read(28*1024*1024+1)
        require(len(raw) <= 28*1024*1024, 'packed return exceeds CPU import bound')
        payload = json.loads(raw)
        require(identity[0] == 'a2-'+payload['arm'].lower(), 'cross-arm pasted return')
        if 'return_set' in payload:
            require(payload['return_set']['kit_commit'] == identity[1], 'mixed A2 kit identities')
        if payload_only:
            return payload
        result = reproduce(payload)
        result.update(provenance='PARTNER-COMPUTED', independent_reverification='UNVERIFIED',
                      transcription='TRANSCRIPTION-UNVERIFIED', checksums='advisory')
        return result
    fragments = {}
    decoded = []
    for index in sorted(parts):
        for kind, value in parts[index]:
            if kind == 'fragment':
                number, piece, count, raw = value
                require(type(number) is int and type(piece) is int and type(count) is int and
                        0 <= piece < count and isinstance(raw, str), 'invalid text fragment')
                values = fragments.setdefault(number, dict(count=count, pieces={}))
                require(values['count'] == count and piece not in values['pieces'], 'duplicate/mixed fragment')
                values['pieces'][piece] = raw
            else:
                decoded.append((kind, value))
    for values in fragments.values():
        require(set(values['pieces']) == set(range(values['count'])), 'missing text fragments')
        raw = ''.join(values['pieces'][i] for i in range(values['count']))
        match = re.fullmatch(r'T\|A2\|(meta|cell|item|diagnostic)\|(.+)', raw)
        require(match is not None, 'invalid reconstructed text record')
        decoded.append((match[1], json.loads(match[2])))
    metadata, cells, items, diagnostics = [], {}, [], {}
    for kind, value in decoded:
        if kind == 'meta':
            metadata.append(value)
        elif kind == 'cell':
            key = tuple(value['key'])
            require(key not in cells, 'duplicate A2 cell')
            cells[key] = dict(value, rows=[])
        elif kind == 'diagnostic':
            require(value[0] not in diagnostics, 'duplicate A2 diagnostic')
            diagnostics[value[0]] = value[1]
        else:
            items.append(value)
    require(len(metadata) == 1, 'missing/duplicate A2 metadata')
    payload = metadata[0]
    require(identity[0] == 'a2-'+payload['arm'].lower(), 'cross-arm pasted return')
    for key, item in items:
        require(tuple(key) in cells, 'item without scientific cell')
        cells[tuple(key)]['rows'].append(item)
    payload['cells'] = list(cells.values())
    if diagnostics:
        payload['diagnostics'] = diagnostics
    if payload_only:
        return payload
    result = reproduce(payload)
    result.update(provenance='PARTNER-COMPUTED', independent_reverification='UNVERIFIED',
                  transcription='TRANSCRIPTION-UNVERIFIED', checksums='advisory')
    return result




def diagnostic_summary(diagnostics):
    """Keep exact length distributions and loss sums/counts; bulk stays in FULL."""
    from collections import Counter
    result = copy.deepcopy(diagnostics)
    for key, value in result.items():
        if not isinstance(value, dict):
            continue
        if key.startswith('rewrite/example/') and 'raw' in value:
            result[key] = dict(accepted=value['accepted'], detail='FULL',
                raw_sha256=digest(value['raw'].encode()), rewritten_sha256=digest(value['rewritten'].encode()),
                raw_characters=len(value['raw']), rewritten_characters=len(value['rewritten']))
            continue
        if 'response_characters' in value:
            value['response_character_histogram'] = dict(sorted(Counter(map(str, value.pop('response_characters'))).items()))
        runtime = value.get('runtime')
        if runtime and 'loss_metrics' in runtime:
            logs = runtime.pop('loss_metrics')
            metrics = {}
            for field in sorted({k for row in logs for k in row}):
                numbers = [r[field] for r in logs if type(r.get(field)) in (int, float) and math.isfinite(r[field])]
                if numbers:
                    metrics[field] = dict(n=len(numbers), total=sum(numbers), mean=sum(numbers)/len(numbers))
            runtime['loss_metric_statistics'] = metrics
    return result

def compact_counts(payload):
    """Exact sufficient statistics, computed from units, never from pasted bars."""
    value = {k: copy.deepcopy(v) for k, v in payload.items()
             if k not in ('cells', 'assignment', 'return_set', 'keyed_counts')}
    if 'diagnostics' in value:
        value['diagnostics'] = diagnostic_summary(value['diagnostics'])
    if 'cells' in payload:
        capture = {}
        reproduce(payload, capture=capture)
        value['keyed_counts'] = capture
        assignment = payload.get('assignment')
        if assignment is not None:
            value['assignment'] = {k: assignment[k] for k in ('status', 'nonce', 'allocated_uuids') if k in assignment}
    elif 'keyed_counts' in payload:
        value['keyed_counts'] = copy.deepcopy(payload['keyed_counts'])
        if 'assignment' in payload:
            value['assignment'] = copy.deepcopy(payload['assignment'])
    return value


def import_parts(texts, full_archive=None):
    """Import complete count or legacy sets; FULL is an exact evidence cross-check.

    Physical and set SHA256s are advisory. Collect IDs and scientific counts are
    structural identities/evidence, so mixtures and disagreements are refused.
    """
    sets, identity = {}, None
    for text in texts:
        flat = re.sub(r'\s+', '', text)
        mode = re.search(r'T\|MODE\|(ESSENTIAL|FULL)', flat)
        own = re.search(r'T\|A2-SET\|([0-9a-f]{32})(?=T\|)', flat)
        if own:
            require(mode is not None, 'missing A2 return mode')
            require(identity in (None, own[1]), 'mixed A2 collect identities')
            identity = own[1]
        sets.setdefault(mode[1] if mode else 'FULL', []).append(text)
    require(sets, 'INCOMPLETE: missing A2 text parts')
    payloads = {mode: _import_parts(parts, payload_only=True) for mode, parts in sets.items()}
    for mode, payload in payloads.items():
        transport = payload.get('return_set')
        if transport is not None:
            require(identity is not None, 'missing A2 collect identity')
        if 'keyed_counts' in payload:
            require(transport is not None and mode == 'ESSENTIAL', 'missing A2 count transport identity')
        if identity is not None:
            require(transport and transport['collect'] == identity, 'mixed A2 collect identities')
            for text in sets[mode]:
                flat = re.sub(r'\s+', '', text)
                require('T|A2-SET|'+identity in flat and transport['kit_commit'] in flat,
                        'missing/mixed A2 collect identity')
            if 'cells' in payload or 'keyed_counts' in payload:
                require((mode == 'ESSENTIAL') == ('keyed_counts' in payload),
                        'A2 transport mode differs from payload')
    essential = payloads.get('ESSENTIAL')
    full = payloads.get('FULL')
    if full_archive is not None:
        archive = archive_payload(full_archive)
        if full is not None:
            require(compact_counts(full) == compact_counts(archive), 'FULL text/archive disagreement')
        full = archive
    if essential is not None and full is not None:
        require(compact_counts(essential) == compact_counts(full), 'ESSENTIAL/FULL disagreement')
    payload = essential if essential is not None else full
    require(payload is not None, 'INCOMPLETE: no A2 return')
    result = reproduce(payload)
    result.update(provenance='PARTNER-COMPUTED', independent_reverification='UNVERIFIED',
                  transcription='TRANSCRIPTION-UNVERIFIED', checksums='advisory')
    return result


def archive_payload(path):
    with tarfile.open(path, 'r:gz') as tar:
        members = tar.getmembers()
        require([m.name for m in members] == ['a2-evidence.json'] and
                members[0].isfile() and members[0].size <= 28*1024*1024, 'invalid independent A2 archive')
        return json.load(tar.extractfile(members[0]))


def read_archive(path):
    return reproduce(archive_payload(path))


def return_identity(cfg):
    """Identify the canonical sealed config and, separately, its file bytes."""
    from kit.a2_budget import sealed_config
    sealed = sealed_config(cfg)
    path = Path(cfg['out'])/'config.json'
    bound = Path(cfg['out'])/'binding.json'
    pins = load(bound)['protocol'] if cfg.get('disposition') != 'archived' and bound.is_file() else protocol()
    commit = cfg.get('kit_commit')
    if not commit:
        import subprocess
        commit = subprocess.check_output(['git', '-C', str(KIT), 'rev-parse', 'HEAD'], text=True).strip()
    result = dict(config_sha256=digest(sealed), protocol_sha256=digest(pins), kit_commit=commit)
    if cfg.get('disposition') != 'archived' and path.is_file():
        result['config_file_sha256'] = digest(path.read_bytes())
    return result

def disposition_return(cfg, status, confirmation=False):
    """An unselected or zero-share arm archives without opening any node input."""
    import subprocess
    payload = dict(arm=cfg['arm'], registration_sha256=REGISTRATION, status=status,
                   identity=return_identity(cfg), confirmation=confirmation)
    archive = Path(cfg['archive'])
    if confirmation:
        archive = archive.with_name('confirmation-'+archive.name)
    archive.parent.mkdir(parents=True, exist_ok=True)
    raw = canonical(payload)
    with tarfile.open(archive, 'w:gz') as tar:
        info = tarfile.TarInfo('a2-evidence.json')
        info.size = len(raw)
        tar.addfile(info, io.BytesIO(raw))
    commit = cfg.get('kit_commit') or subprocess.check_output(['git', '-C', str(KIT), 'rev-parse', 'HEAD'], text=True).strip()
    write_parts(Path(cfg['text_out'])/('confirmation' if confirmation else 'exploratory'), payload, commit)
    return reproduce(payload)


def qualification_return(cfg, out):
    """CPU-only return even when repricing stopped every scientific dependent."""
    from kit.a2_budget import reconcile, price_evidence, sealed_config
    try:
        price = price_evidence(out, cfg['arm'])
    except (NotReady, FileNotFoundError):
        price = dict(decision='NOT PRICED', status='NOT PRICED',
                     config_sha256=digest(sealed_config(cfg)),
                     qualification_gpu_hours=reconcile(out, cfg.get('technical_synthetic', False))['gpu_hours'])
    payload = dict(arm=cfg['arm'], registration_sha256=REGISTRATION, status='PENDING',
                   identity=return_identity(cfg),
                   qualification=price, accounting=reconcile(out, cfg.get('technical_synthetic', False)))
    raw = canonical(payload)
    archive = Path(cfg['archive'])
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, 'w:gz') as tar:
        info = tarfile.TarInfo('a2-evidence.json')
        info.size = len(raw)
        tar.addfile(info, io.BytesIO(raw))
    commit = cfg.get('kit_commit')
    if not commit:
        import subprocess
        commit = subprocess.check_output(['git', '-C', str(KIT), 'rev-parse', 'HEAD'], text=True).strip()
    paths = write_parts(Path(cfg['text_out'])/'exploratory', payload, commit)
    target = Path(out)/'qualification-return.json'
    atomic(target, reproduce(payload))
    if price['decision'] != 'OWNER ESCALATION':
        raise NotReady('qualification return written; collect remains unsealed until pricing and later rows complete')
    return [target]


def collect(cfg, out, confirmation=False):
    from kit.a2_jobs import completed
    from kit.a2_read import read
    from kit.a2_budget import reconcile
    root = completed(out, 'A2-'+cfg['arm']+('-confirmation-score' if confirmation else '-comparison'))
    payload = dict(arm=cfg['arm'], registration_sha256=REGISTRATION,
                   identity=return_identity(cfg),
                   technical_synthetic=cfg.get('technical_synthetic', False), cells=load(root/'cells.json'),
                   assignment=load(root/'receipt.json'),
                   p_applicable=load(out/'binding.json').get('P_applicable', True),
                   accounting=reconcile(out, cfg.get('technical_synthetic', False)))
    payload['qualification'] = load(checked(load(Path(out)/'qualification/price-ref.json')))
    from kit.a2_pipeline import diagnostics
    payload['diagnostics'] = diagnostics(out, load(out/'binding.json'))
    if confirmation:
        payload.update(confirmation=True, confirmation_receipt=load(out/'confirmation-binding.json'))
    # The archive is self-contained for its CPU reader and contains no weights.
    raw = canonical(payload)
    archive = Path(cfg['archive'])
    if confirmation:
        archive = archive.with_name('confirmation-'+archive.name)
    require(archive.is_absolute() and not archive.is_relative_to(Path(out)/'runs'), 'external independent return path required')
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, 'w:gz') as tar:
        info = tarfile.TarInfo('a2-evidence.json')
        info.size = len(raw)
        tar.addfile(info, io.BytesIO(raw))
    require(archive.stat().st_size <= 28*1024*1024, 'A2 archive exceeds return cap')
    result = reproduce(payload)
    require(result == read_archive(archive), 'A2 archive reader does not reproduce')
    prefix = 'confirmation-' if confirmation else ''
    atomic(out/(prefix+'returned-reading.json'), result)
    commit = cfg.get('kit_commit')
    if not commit:
        import subprocess
        commit = subprocess.check_output(['git', '-C', str(KIT), 'rev-parse', 'HEAD'], text=True).strip()
    text_out = Path(cfg['text_out'])/('confirmation' if confirmation else 'exploratory')
    paths = write_parts(text_out, payload, commit)
    essential = [p.read_text() for p in paths if '-ESSENTIAL-' in p.name]
    imported = import_parts(essential)
    require({k: v for k, v in imported.items() if k != 'transcription'} == result,
            'ESSENTIAL reading differs from archive')
    require(import_parts([p.read_text() for p in paths]) == imported, 'FULL reading differs')
    atomic(out/(prefix+'text-imported.json'), imported)
    # All bulk checkpoints, contexts, schedules, raw answers and attempt receipts
    # stay on the durable node. Their manifest is returned without moving models.
    bulk = {str(p.relative_to(out)): s.file_hash(p) for p in Path(out).rglob('*')
            if p.is_file() and not p.name.endswith('durable-manifest.json')}
    manifest = out/(prefix+'durable-manifest.json')
    atomic(manifest, dict(files=bulk, preserve=True))
    return [out/(prefix+'returned-reading.json'), out/(prefix+'text-imported.json'), manifest]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive', type=Path, help='FULL archive, optionally cross-check --text')
    p.add_argument('--text', type=Path, nargs='+')
    p.add_argument('--arm', choices=('J', 'P', 'R', 'RP'), required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    require(a.archive or a.text, '--archive or --text required')
    try:
        result = import_parts([p.read_text() for p in a.text], a.archive) if a.text else read_archive(a.archive)
    except CorrectnessStop as error:
        if not str(error).startswith('INCOMPLETE:'):
            raise
        atomic(a.out, dict(arm=a.arm, status='INCOMPLETE', reason=str(error),
                           provenance='PARTNER-COMPUTED', transcription='TRANSCRIPTION-UNVERIFIED'))
        print('INCOMPLETE')
        return PENDING
    require(result['arm'] == a.arm, 'wrong standalone arm return')
    atomic(a.out, result)
    print(result['status'])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
