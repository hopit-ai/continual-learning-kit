#!/usr/bin/env python3
"""Node-only, verify-only exclusion gate and pair-level return adapter.

Run with python -I. No inference unless an explicitly supplied command follows
`run -- ...`. Input paths stay in the node-local manifest; returned identities
contain only portable inventory IDs, byte hashes and row counts.
"""
from pathlib import Path
import argparse
import json
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from kit import sp_prepare as sp


def _row(raw, index):
    request = raw.get('request', {})
    extra = raw.get('extra_info', {})
    row = {k: raw[k] for k in sp.TEXT_FIELDS if k in raw}
    row['id'] = str(raw.get('id', request.get('id', extra.get('index', index))))
    prompt = raw.get('prompt', raw.get('messages', request.get('messages')))
    if isinstance(prompt, str) and prompt.lstrip().startswith('['):
        try:
            decoded = json.loads(prompt)
            if isinstance(decoded, list): prompt = decoded
        except ValueError: pass
    if prompt is not None: row['prompt'] = prompt
    response = raw.get('response', raw.get('text', extra.get('demonstration')))
    if response is not None: row['response'] = response
    if not sp.row_texts(row): raise ValueError('inventory row contains no auditable text')
    return row


def _load(entry, base, required):
    name = required['inventory_id']
    path = base / entry['path']
    expected = required.get('sha256')
    if entry.get('format', 'jsonl') != required['format']:
        raise ValueError('inventory format mismatch: ' + name)
    if required['format'] == 'journal':
        # Bind a complete file list in the frozen producer manifest. Read only
        # the declared raw journal JSON, never arbitrary code or pickle files.
        manifest = json.loads(path.read_text())
        sha = sp.sha256_file(path)
        if sha != entry.get('sha256'): raise ValueError('journal manifest bytes changed: ' + name)
        files = manifest.get('files', {})
        names = sorted(n for n in files if n.startswith('teacher/raw/') and n.endswith('.json'))
        if not names: raise ValueError('journal manifest has no raw response files: ' + name)
        actual = {p.relative_to(path.parent).as_posix() for p in (path.parent/'teacher/raw').glob('*.json')}
        if actual != set(names): raise ValueError('journal file inventory differs: ' + name)
        raw = []
        for member in names:
            relative = Path(member)
            if relative.is_absolute() or '..' in relative.parts: raise ValueError('unsafe journal member')
            file = path.parent / relative
            if sp.sha256_file(file) != files[member]: raise ValueError('journal member bytes changed: ' + name)
            raw.append(json.loads(file.read_text()))
    else:
        sha = sp.sha256_file(path)
        if sha != entry.get('sha256') or (expected and sha != expected):
            raise ValueError('inventory bytes differ from binding: ' + name)
        if required['format'] == 'parquet':
            import pyarrow.parquet as pq
            raw = pq.read_table(path).to_pylist()
        else:
            raw = sp.read_jsonl(path)
    if not raw: raise ValueError('empty node inventory: ' + name)
    rows = [_row(r, i) for i, r in enumerate(raw)]
    return dict(name=name, role=required['role'], replay=required['role'] in sp.REPLAY_ROLES,
                sha256=sha, rows=rows, scope='node-held-reference')


def verify(doc, spec_path, requests, ck):
    policy = doc['node_verification']
    spec_path = Path(spec_path)
    spec = json.loads(spec_path.read_text())
    problems = []
    required = {r['inventory_id']: r for r in policy['unavailable_inventories']}
    entries = spec.get('inventories', [])
    provided = {r.get('inventory_id'): r for r in entries}
    if (spec.get('schema') != 'sp-node-inventories-v1' or spec.get('complete_inventory') is not True
            or spec.get('ancestry_closed') is not True or len(provided) != len(entries)
            or spec.get('policy_sha256') != sp.digest(policy)):
        problems.append(dict(kind='invalid-node-inventory-attestation'))
    pools = []
    for name, requirement in required.items():
        if name not in provided:
            problems.append(dict(kind='missing-node-inventory', inventory_id=name)); continue
        try:
            pools.append(_load(provided[name], spec_path.resolve().parent, requirement))
        except (ValueError, KeyError, OSError, ImportError, TypeError):
            problems.append(dict(kind='invalid-node-inventory-binding', inventory_id=name))
    for name in sorted(set(provided)-set(required)):
        problems.append(dict(kind='undeclared-node-inventory', inventory_id=str(name)))
    # Even with missing inventories, check the available valid files and report
    # localizable hits. Unknown coverage is separately represented for all pairs.
    checked = sp.audit(requests, pools, ck)
    problems += checked['hits']
    if spec.get('returned_pools_spec'):
        try:
            returned = sp.load_pools(spec_path.resolve().parent/spec['returned_pools_spec'], returned=True)
            problems += sp.verify_returned_pools(doc, returned, requests, ck)['hits']
        except (ValueError, KeyError, OSError, TypeError):
            problems.append(dict(kind='invalid-returned-pools-binding'))
    result = sp.pair_verification_result(requests, problems, sp.pool_identities(pools))
    result.update(policy_sha256=sp.digest(policy), node_inventory_sha256=sp.sha256_file(spec_path),
                  checked_before_scoring=True, rules=policy['rules'])
    return result


def annotate(scored, verification):
    """Add verification labels without changing scores, order or membership."""
    statuses = verification.get('pair_status', {})
    return [dict(row, verification_status=statuses.get(row.get('request_id', str(row.get('id','')).rsplit(':',1)[0]), 'UNVERIFIED'))
            for row in scored]


def preflight(frozen, pools, receipt):
    result = sp.recheck(frozen, pools, receipt)
    print('SP node VERIFY-ONLY:', result['status'], '; affected pairs:', len(result.get('affected_request_ids', [])), flush=True)
    return result


def report(frozen, receipt, sidecar, out, inventory_path=None, inputs=()):
    """Read existing scores, retaining every pair and every full-panel statistic."""
    from kit import sp_read as rd
    from kit.sp_manifest import discover
    doc = sp.verify_frozen(frozen)
    verification = json.loads(Path(receipt).read_text())
    sp.require(verification.get('frozen_sha256') == sp.sha256_file(Path(frozen)/'FROZEN.json')
               and verification.get('policy_sha256') == sp.digest(doc['node_verification']), 'unbound node verification receipt')
    inventory = json.loads(Path(inventory_path).read_text()) if inventory_path else None
    if inventory is not None: rd.verify_seal(inventory)
    cells = rd.read_sidecar(Path(sidecar), inventory)
    tables = rd.family_tables(cells, inventory, Path(sidecar))
    tables['bootstrap'] = {c['entry']['entry_id']: rd.bootstrap_pairs(c['scored']) for c in cells}
    execution = [json.loads(p.read_text()) for p in (Path(sidecar)/'state').glob('SP-panel-*.json')]
    tables['wall_time'] = rd.wall_time(cells, inventory, [c for root in inputs for c in discover(root)], execution)
    tables.update(pool_verification=verification, pair_verification=verification.get('pair_status', {}),
                  affected_request_ids=verification.get('affected_request_ids', []),
                  verification_scope='PAIR_LOCAL', statistics_include_unverified_pairs=bool(verification.get('unverified_request_ids')),
                  panel_changed=False, deleted_pairs=0, rescoring_allowed=False)
    # Keep overall completeness and all raw statistics. No family-wide
    # UNVERIFIED flag is inferred from a hit on one request.
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    sp.write_json(out/'sp-tables.json', tables)
    sp.write_jsonl(out/'pair-verification.jsonl', [dict(request_id=rid,verification_status=status)
                                                  for rid,status in verification.get('pair_status',{}).items()])
    return tables


def run_command(frozen, pools, receipt, command, runner=subprocess.run):
    result = preflight(frozen, pools, receipt)
    if not result.get('scoring_allowed'): return 2
    if not command: return 0
    return runner(command, check=False).returncode


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    command = []
    if '--' in argv:
        index = argv.index('--'); command = argv[index+1:]; argv = argv[:index]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('verify','run','report'))
    p.add_argument('--frozen', type=Path, default=sp.RESOURCES)
    p.add_argument('--pools', type=Path)
    p.add_argument('--receipt', type=Path, required=True)
    p.add_argument('--sidecar', type=Path)
    p.add_argument('--out', type=Path)
    p.add_argument('--inventory', type=Path)
    p.add_argument('--input', type=Path, action='append', default=[])
    a = p.parse_args(argv)
    if a.action == 'report':
        if not a.sidecar or not a.out: p.error('report needs --sidecar and --out')
        report(a.frozen,a.receipt,a.sidecar,a.out,a.inventory,a.input); return 0
    if a.pools is None: p.error('verify/run needs --pools')
    if a.action == 'run' and not command: p.error('run requires a scoring command after --')
    return run_command(a.frozen,a.pools,a.receipt,command if a.action=='run' else [])


if __name__ == '__main__': raise SystemExit(main())
