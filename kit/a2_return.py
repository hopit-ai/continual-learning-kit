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


def reproduce(payload):
    require(payload['arm'] in ('J', 'P', 'R', 'RP') and payload['registration_sha256'] == REGISTRATION,
            'return registration/arm differs')
    if payload.get('status') in ('ARCHIVED UNREAD', 'NOT APPLICABLE', 'PENDING'):
        return dict(arm=payload['arm'], status=payload['status'], provenance='PARTNER-COMPUTED',
                    independent_reverification='UNVERIFIED',
                    **{k: payload[k] for k in ('accounting', 'qualification') if k in payload})
    result = analyse(payload['arm'], payload['cells'], synthetic=payload['technical_synthetic'],
                     confirmation=payload.get('confirmation', False), receipt=payload.get('confirmation_receipt'),
                     p_applicable=payload.get('p_applicable', True), assignment=payload.get('assignment'))
    result['accounting'] = payload['accounting']
    if 'qualification' in payload:
        result['qualification'] = payload['qualification']
    if 'diagnostics' in payload:
        result['diagnostics'] = payload['diagnostics']
    return result


def write_parts(directory, payload, commit):
    require(re.fullmatch('[0-9a-f]{40}', commit), 'kit commit must be a full git identity')
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    # Exact numeric endpoints remain visible; no pasted verdict or interval is trusted.
    def record(kind, value):
        text = canonical(value).decode().replace(' ', r'\u0020').replace('|', r'\u007c')
        return 'T|A2|'+kind+'|'+text
    compact = copy.deepcopy(payload)
    fields = ('question_id', 'correct', 'historical_correct', 'holdout', 'family',
              'input_tokens', 'output_tokens', 'finished')
    for cell in compact.get('cells', []):
        cell['rows'] = [{k: r[k] for k in fields if k in r} for r in cell['rows']]
        cell['model_sha256'] = digest(cell.pop('model', {}))
    for value in compact.get('diagnostics', {}).values():
        runtime = value.get('runtime') if isinstance(value, dict) else None
        if runtime and 'loss_metrics' in runtime:
            logs = runtime.pop('loss_metrics')
            keys = {k for row in logs for k in row}
            means = {}
            for key in keys:
                values = [r[key] for r in logs if type(r.get(key)) in (int, float) and math.isfinite(r[key])]
                if values:
                    means[key] = sum(values)/len(values)
            runtime['loss_metric_means'] = means
    packed = base64.b64encode(gzip.compress(canonical(compact), mtime=0)).decode('ascii')
    pieces = [packed[i:i+3000] for i in range(0, len(packed), 3000)]
    rows = ['T|PARTNER-COMPUTED|independent-reverification=UNVERIFIED',
            'T|CHECKSUM-SCOPE|advisory-only']
    reading = reproduce(payload)
    if payload.get('qualification'):
        price = payload['qualification']
        rows.extend(['T|A2-QUALIFICATION-SPEND|'+str(price['qualification_gpu_hours']),
                     'T|A2-FULL-PRICE|'+str(price.get('total_gpu_hours', 'NOT-PRICED')),
                     'T|A2-PRICE-DECISION|'+price['decision'].replace(' ', ':')])
    for label, contrast in reading.get('contrasts', {}).items():
        p = contrast['primary']
        verdict = reading['status'] if reading['status'] != 'COMPLETE' else contrast['verdict']
        rows.append('|'.join(['D', 'A2-'+payload['arm'], label, verdict.replace(' ', ':'),
                              *[canonical(p[k]).decode() for k in ('mean', 'lower', 'upper', 'positive')], '-']))
        for name, kind, result, bar in [('primary', 'P', p, 'mean>=2;LB>0;pos=3'),
                *[(k, 'G', v, 'LB>=-1') for k, v in contrast['guards'].items()]]:
            rows.append('|'.join(['E', 'A2-'+payload['arm'], label+'.'+name, kind,
                *[canonical(result[k]).decode() for k in ('values', 'mean', 'sd', 'lower', 'upper', 'positive', 'n')], bar, 'two']))
    rows.extend(record('packed', [i, len(pieces), piece]) for i, piece in enumerate(pieces))
    chunks = [[]]
    size = 0
    for line in rows:
        require(len(line) < 10000, 'logical text record too large')
        if size+len(line)+len(line)//89+1 > 10500:
            chunks.append([])
            size = 0
        chunks[-1].append(line)
        size += len(line)+len(line)//89+1
    full_set = digest(chunks)
    rendered = []
    for index, chunk in enumerate(chunks, 1):
        value = frame_text('a2-'+payload['arm'].lower(), commit, index, len(chunks), 'FULL',
                           f'T|FULL-SET|{len(chunks)}|{full_set}', chunk, width=89)
        require(value.isascii() and len(value) <= 11400 and all(len(line) < 90 for line in value.splitlines()), 'text part overflow')
        rendered.append((f'RETURN-TEXT-{index}-of-{len(chunks)}.md', value))
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
    return paths


def import_parts(texts):
    parts = {}
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
        require(1 <= index <= count and index not in parts and n in (None, count), 'duplicate/mixed part numbering')
        n = count
        body, _, checksum = flat[match.end():].rpartition('SHA256')
        require(bool(checksum), 'part checksum field absent')
        # Digests are advisory, even when the content was transcribed differently.
        records = re.findall(r'T\|A2\|(meta|cell|item|diagnostic|fragment|packed)\|(.+?)(?=T\|A2\||$)', body)
        parsed = []
        for kind, raw in records:
            parsed.append((kind, json.loads(raw)))
        parts[index] = parsed
        checksum_status.append('TRANSCRIPTION-UNVERIFIED')
    require(n is not None and set(parts) == set(range(1, n+1)), 'missing A2 text parts')
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
    result = reproduce(payload)
    result.update(provenance='PARTNER-COMPUTED', independent_reverification='UNVERIFIED',
                  transcription='TRANSCRIPTION-UNVERIFIED', checksums='advisory')
    return result


def read_archive(path):
    with tarfile.open(path, 'r:gz') as tar:
        members = tar.getmembers()
        require([m.name for m in members] == ['a2-evidence.json'] and
                members[0].isfile() and members[0].size <= 28*1024*1024, 'invalid independent A2 archive')
        payload = json.load(tar.extractfile(members[0]))
    return reproduce(payload)


def disposition_return(cfg, status, confirmation=False):
    """An unselected or zero-share arm archives without opening any node input."""
    import subprocess
    payload = dict(arm=cfg['arm'], registration_sha256=REGISTRATION, status=status)
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
    from kit.a2_budget import reconcile
    from kit.a2_jobs import completed
    pointer = Path(out)/'qualification/price-ref.json'
    try:
        completed(out, 'A2-'+cfg['arm']+'-price')
        price = load(checked(load(pointer)))
    except (NotReady, FileNotFoundError):
        price = dict(decision='NOT PRICED', status='NOT PRICED',
                     qualification_gpu_hours=reconcile(out, cfg.get('technical_synthetic', False))['gpu_hours'])
    payload = dict(arm=cfg['arm'], registration_sha256=REGISTRATION, status='PENDING',
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
    imported = import_parts([p.read_text() for p in paths])
    require(imported.get('contrasts') == result.get('contrasts') and imported['status'] == result['status'], 'text import bars differ')
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
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--archive', type=Path)
    group.add_argument('--text', type=Path, nargs='+')
    p.add_argument('--arm', choices=('J', 'P', 'R', 'RP'), required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    result = read_archive(a.archive) if a.archive else import_parts([p.read_text() for p in a.text])
    require(result['arm'] == a.arm, 'wrong standalone arm return')
    atomic(a.out, result)
    print(result['status'])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
