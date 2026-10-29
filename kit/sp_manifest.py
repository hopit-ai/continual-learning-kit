#!/usr/bin/env python3
"""SP checkpoint inventory and isolated sidecar jobs (combined message 2+3, registration v2).

    python kit/sp_manifest.py inventory --launch LAUNCH.json [--launch M3-LAUNCH.json] [--pd PD.json]
                                        [--tier TIER.json ...] [--identities ROOT ...] --out INVENTORY.json
    python kit/sp_manifest.py sidecar   --launch LAUNCH.json [--launch ...] --inventory INVENTORY.json --out SIDECAR.json
    python kit/sp_manifest.py validate  --launch LAUNCH.json [--launch ...] --sidecar SIDECAR.json
    python kit/sp_manifest.py price     --inventory INVENTORY.json
    python kit/sp_manifest.py verify-pools [--superset FILE] --pools-dir DIR --out FILE
                                        (default superset: kit/resources/m3/sp-candidate-superset-v1.json)
    python kit/sp_manifest.py run       --sidecar SIDECAR.json --task ID --kit DIR --sp DIR [...]
    python kit/sp_manifest.py length    --inventory INVENTORY.json --sp DIR --input ROOT ...

Coverage (SP(b), owner ruling 29 Oct): durable chain and arm checkpoints of M, repaired chain
rescoring, X tiers 1/2, K, CR and any included GRPO, at base, incoming and FINAL only (no
intermediates). Screen S, PD probe forks and separate 4A are excluded. PD start checkpoints are
chain checkpoints already covered by M; the inventory only cross-references them. SP(a) length
reuses existing raw outputs at every scored step, CPU only, with zero new core GPU work.

Scorer layouts read (never edited), manager ruling on m3b2 Q2/Q3:
- kit/m2_score.py (core waves via kit/m2_run.py, re-probe via kit/m2_reprobe.py, X via
  kit/x_score.py): <out>/scores/<arm>-s<seed>-k<step>-<panel>/m2-identity.json; re-probe writes
  arm 'legal-<config>' (or 'base'); X writes 'x-<arm>', 'x-pre' or 'x-t2-<comparison>-pre', and
  its step-0 arm units are aliases (alias.json + copied identity, raw outputs only at the source).
- kit/m2_rescore.py: <out>/packages/<id>/scores/<arm>-s<seed>-k<step>-<panel>/, assignment family
  'return1.<id>'; bound at runtime by that family/seed.
- kit/m3_score.py (K/CR, message3-launch.v1): <out>/scores/<arm.stage|base>-s<seed>-k<step>-<panel>/
  m3-identity.json; step-0 incoming units are alias.json folders pointing at the earlier stage final.
- kit/m2_run.py declares no arm/step on its command line; expand_m2_run mirrors its fixed rules.

Isolation rules (validate_sidecar enforces them; run never fails a parent):
- every SP task ID starts with "SP-"; no parent task requires, imports, overlays or barriers on one;
- SP tasks wait on parents with afterany semantics; a failed or missing parent leaves SP cells
  MISSING, never retried by SP and never blocking anything;
- SP commands write only under {sp}, the separate sidecar root; {out} and {durable} are refused;
- SP reads durable checkpoints in place and never copies, pins or extends their retention: a
  deleted checkpoint is MISSING;
- each connected comparison family/seed has its own SP GPU assignment, and every checkpoint of
  that family/seed is scored there;
- run records DONE / FAILED / MISSING-INPUT in {sp}/state/<task>.json and always exits 0.

Partner-local model locations and GPU identities are supplied to run as local map files and are
never written into the inventory, the sidecar manifest or any returned SP file.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kit import pd_contract
from kit.pd_contract import digest, require, seal, verify_seal

SEEDS = (101, 102, 103)
INVENTORY_SCHEMA = 'sp-checkpoint-inventory-v3'
TIER_SCHEMA = 'sp-tier-checkpoints-v1'
SIDECAR_SCHEMA = 'message3-sp-sidecar.v3'
SUPERSET_SCHEMA = 'sp-candidate-superset-v1'
# m3a2 publishes the frozen candidate superset here (manager ruling, round 3).
SUPERSET_PATH = Path('resources') / 'm3' / 'sp-candidate-superset-v1.json'
TIERS = ('K', 'CR', 'GRPO')
FINAL_STEP = 40
SP_B_ROLES = ('base', 'incoming', 'final')
# Separate 4A work and Screen S are never SP checkpoints.
EXCLUDED = (
    ('4A is separate', re.compile(r'^(?:4a|a4)(?:$|[-_.:])', re.I)),
    ('Screen S is out of this tier', re.compile(r'^(?:screen-s|s-inventory|s-generation|s\d|s-)', re.I)),
)
GENERAL_SUBPANELS = ('ifeval', 'knowledge', 'math')
NOT_APPLICABLE = ('hellaswag', 'knowledge-ll')   # likelihood: no generated length
# Registration v2 budget: SP(a) 0 GPU-h; SP(b) 0.25-1.0 GPU-h per entry plus 1 startup.
PRICE = dict(sp_a_gpu_h=0.0, per_entry_gpu_h=(0.25, 1.0), startup_gpu_h=1.0, wall_time_risk_hours=24.0)
PARENT_LINK_KEYS = ('requires', 'barriers', 'evidence_from', 'overlay')
IDENTITY_FILES = {'m2-identity.json': 'message2', 'm3-identity.json': 'message3'}
CELL_NAME = re.compile(r'^(?P<arm>.+)-s(?P<seed>\d+)-k(?P<step>\d+)-(?P<panel>[^/]+)$')

# Mirrors of the core scorers' fixed constants (read, never edited): kit/c_common STEPS and
# RETENTION_STEPS, kit/m2_common PANEL_SETS; tests/test_sp.py checks they still agree.
M2_STEPS = (0, 10, 20, 30, 40)
M2_RETENTION_STEPS = (0, 20, 40)
M2_PANEL_SETS = ('finqa', 'test', 'math_hard', 'general-v1')
# Replay waves that run only after a nomination/admission; absent cells are NOT-RUN, not MISSING.
CONDITIONAL_ARMS = ('legal-broad', 'legal-earlier', 'legal-both', 'math-sibling')


def _int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def options(command):
    out, i = {}, 0
    while i < len(command):
        token = command[i]
        if isinstance(token, str) and token.startswith('--'):
            nxt = command[i + 1] if i + 1 < len(command) else None
            if nxt is None or (isinstance(nxt, str) and nxt.startswith('--')):
                out[token[2:]] = True; i += 1
            else:
                out[token[2:]] = nxt; i += 2
        else:
            i += 1
    return out


def as_list(launches):
    return [launches] if isinstance(launches, dict) else list(launches)


def exclusion(entry):
    for reason, pattern in EXCLUDED:
        if any(pattern.match(str(entry.get(k) or '')) for k in ('family', 'source', 'tier')):
            return reason
    if str(entry.get('row') or '').upper().startswith(('A4', 'S')) and entry.get('source') == 'message2':
        return 'Screen S is out of this tier' if str(entry['row']).upper().startswith('S') else '4A is separate'
    return None


def role_of(entry):
    """base / incoming / final / intermediate for a planned entry."""
    if entry.get('role'):
        return entry['role']
    step = entry.get('step')
    if step == 0:
        return 'base' if entry.get('arm') == 'base' else 'incoming'
    if step == FINAL_STEP:
        return 'final'
    return 'intermediate' if step is not None else 'runtime'


# ---------------------------------------------------------------------------------------------
# Planned inventory.

def expand_m2_run(o):
    """(arm, step, panel) cells one kit/m2_run.py command scores, exactly as m2_run.job does."""
    action, mode, phase = o.get('action'), o.get('mode', 'all'), o.get('score-phase', 'all')
    cells = []
    def scores(arm, step, panels):
        if mode == 'train' or phase == 'incoming' and step != 0 or phase == 'final' and step == 0:
            return
        cells.extend((arm, step, p) for p in panels)
    if action == 'boundaries':
        scores('base', 0, (*M2_PANEL_SETS, 'hellaswag'))
        for trajectory in ('plain', 'sibling'):
            scores('finqa-' + trajectory, 40, M2_PANEL_SETS)
        for step in M2_STEPS:
            scores('legal-plain', step, M2_PANEL_SETS if step in M2_RETENTION_STEPS else ('test',))
        scores('standalone-legal-plain', 40, ('test',))
    elif isinstance(action, str) and action.startswith('legal-'):
        config = action.removeprefix('legal-')
        for arm in (('legal-earlier', 'legal-both') if config == 'next' else ('legal-' + config,)):
            scores(arm, 0, (*M2_PANEL_SETS, 'hellaswag'))
            for step in M2_STEPS[1:]:
                scores(arm, step, (*M2_PANEL_SETS, 'hellaswag') if step in M2_RETENTION_STEPS else ('test',))
    elif isinstance(action, str) and action.startswith('math-'):
        arm = 'math-' + action.removeprefix('math-')
        for step in M2_STEPS:
            scores(arm, step, M2_PANEL_SETS)
    return cells


def expand_x_p(o):
    """kit/x_run.py --action p --mode score: the P arm's scored steps (written as arm x-p)."""
    if o.get('action') != 'p' or o.get('mode') != 'score':
        return []
    full = ('finqa', 'test', 'math_hard', 'general-v1', 'hellaswag')
    return [('x-p', k, p) for k in (10, 20, 30, 40) for p in (('test',) if k in (10, 30) else full)]


def written_arm(script, arm, family, stage=None):
    """The arm name the scorer writes into its score folder and identity."""
    if script == 'm2_reprobe':
        return arm if arm == 'base' else 'legal-' + arm
    if script == 'x_score':
        if arm == 'pre':
            return 'x-t2-%s-pre' % str(family).removeprefix('x-tier2.') if str(family).startswith('x-tier2.') else 'x-pre'
        return 'x-' + arm
    if script == 'm3_score':
        return 'base' if arm == 'base' else '%s.%s' % (arm, stage)
    return arm


def planned_from_launch(launch):
    """Every checkpoint-scoring command of a launch manifest, as (entries, skipped).

    message-2 (message2-launch.v2) and K/CR (message3-launch.v1) manifests are read, never edited.
    Static cells: m2_score, m2_reprobe --action score, x_score, m3_score --action score|alias, and the
    expanded m2_run and x_run P-score commands. Runtime cells: m2_rescore --action score (bound by
    the return-1 package family/seed). Every other command (format validation, references, probes,
    measurements, Screen S, support generation) is not a checkpoint score and is listed as skipped.
    """
    source = 'message3' if str(launch.get('schema', '')).startswith('message3') else 'message2'
    prefix = 'm3' if source == 'message3' else 'm2'
    entries, skipped = [], []
    for task in launch['tasks']:
        for index, command in enumerate(task.get('commands', [])):
            o = options(command)
            script = Path(str(command[1])).stem if len(command) > 1 else str(command[0])
            action = o.get('action') if isinstance(o.get('action'), str) else None
            seed = _int(o.get('seed')) or _int(task.get('seed'))
            base = dict(source=source, task=task['id'], row=task.get('row'),
                        family=task.get('family') or ('m3' if source == 'message3' else None), seed=seed,
                        script=script, action=action)
            pid = '%s:%s:%d' % (prefix, task['id'], index)
            cells = None
            if script == 'm2_run':
                cells = expand_m2_run(o)
            elif script == 'x_run':
                cells = expand_x_p(o)
            elif ((script in ('m2_score', 'x_score') or script == 'm2_reprobe' and action == 'score'
                   or script == 'm3_score' and action in ('score', 'alias'))
                  and o.get('arm') is not None and o.get('step') is not None):
                cells = [(written_arm(script, o['arm'], task.get('family'), o.get('stage')), _int(o['step']), o.get('panel-set'))]
            elif script == 'm2_rescore' and action == 'score':
                entries.append(dict(base, planned_id=pid, arm=None, step=None, panel=None, resolution='runtime',
                                    conditional=False, role=None))
                continue
            if not cells:
                skipped.append(dict(base, planned_id=pid, scoring_task=task.get('kind') == 'score-family'))
                continue
            for arm, step, panel in cells:
                entry = dict(base, planned_id=pid if len(cells) == 1 else '%s:%s:k%d:%s' % (pid, arm, step, panel),
                             arm=arm, step=step, panel=panel, resolution='static', conditional=arm in CONDITIONAL_ARMS,
                             comparison=arm)
                entry['role'] = role_of(entry)
                if script == 'm3_score' and action == 'alias':
                    stages = ('finqa', 'contract_nli', 'math_hard')
                    stage = stages.index(o['stage'])
                    target_arm = 'base' if stage == 0 else written_arm(script, o['arm'], base['family'], stages[stage - 1])
                    entry['alias_target'] = [source, base['family'], seed, target_arm, 0 if stage == 0 else 40]
                elif script == 'x_score' and step == 0 and (o['arm'] in ('control', 'drift', 'plasticity') or str(o['arm']).startswith('t2-')):
                    entry['alias_target'] = [source, base['family'], seed, written_arm(script, 'pre', base['family']), 0]
                entries.append(entry)
    return entries, skipped


def pd_starts(doc):
    """PD start checkpoints (cross-reference only; PD probe forks are excluded from SP)."""
    pd_contract.validate_manifest(doc)
    return [dict(checkpoint_id=c['checkpoint_id'], start=c['start'], seed=c['seed'], weights_sha256=c['weights_sha256'])
            for c in doc['checkpoints']]


def planned_from_tier(doc):
    """Optional K / CR / GRPO checkpoint lists with bound file hashes (sp-tier-checkpoints-v1)."""
    require(doc.get('schema') == TIER_SCHEMA and doc.get('tier') in TIERS, 'unknown tier checkpoint list')
    entries = []
    for c in doc['checkpoints']:
        require(all(k in c for k in ('checkpoint_id', 'family', 'seed', 'arm', 'step')), 'incomplete tier checkpoint')
        entry = dict(planned_id='%s:%s' % (doc['tier'].lower(), c['checkpoint_id']), source=doc['tier'].lower(),
                     tier=doc['tier'], task=c.get('task'), family=c['family'], seed=c['seed'], arm=c['arm'], step=c['step'],
                     panel=c.get('panel'), resolution='bound' if c.get('model_file_hashes') else 'runtime',
                     conditional=False, checkpoint_id=c['checkpoint_id'])
        entry['role'] = role_of(dict(entry, role=c.get('role')))
        if c.get('model_file_hashes'):
            require(c.get('weights_sha256') == digest(c['model_file_hashes']), 'tier weights digest mismatch')
            entry.update(weights_sha256=c['weights_sha256'], model_file_hashes=c['model_file_hashes'], weights_bound=True)
        entries.append(entry)
    return entries


def screen(entries):
    """Drop 4A, Screen S and seeds outside 101-103; record every drop."""
    kept, dropped = [], []
    for e in entries:
        reason = exclusion(e)
        if reason is None and e.get('seed') not in (None, 0, *SEEDS):
            reason = 'seed outside 101-103'
        if reason:
            dropped.append(dict(planned_id=e['planned_id'], reason=reason))
        else:
            kept.append(e)
    return kept, dropped


# ---------------------------------------------------------------------------------------------
# Existing scorer cells.

def weights_key(model):
    """Weights identity from an observed model identity: the full file-hash inventory."""
    if isinstance(model, dict) and isinstance(model.get('model_file_hashes'), dict) and model['model_file_hashes']:
        return digest(model['model_file_hashes']), dict(model['model_file_hashes'])
    require(isinstance(model, dict) and model.get('technical_synthetic'), 'UNVERIFIED: model identity has no file hashes')
    return digest(dict(technical_synthetic=True, model=model.get('model'))), None


def gpu_key(gpu):
    gpu = gpu or {}
    return (gpu.get('uuid'), gpu.get('environment')) if gpu.get('uuid') else None


def _raw_of(cell):
    return next((p for p in (cell / 'raw' / 'responses.jsonl', cell / 'responses.jsonl') if p.exists()), None)


def _alias_source(cell, alias):
    """X aliases name 'scores/<cell>' relative to the run root; M3 aliases name the sibling folder."""
    name = str(alias.get('source', ''))
    return (cell.parent.parent / name) if '/' in name else (cell.parent / name)


def _identity_in(cell):
    for name, source in IDENTITY_FILES.items():
        if (cell / name).exists():
            return json.loads((cell / name).read_text()), source
    return None, None


def discover(root):
    """Existing scorer cells under one parent output root, following one alias hop with hash checks.

    Each cell: source (message2|message3), family (assignment family), arm (as written in the folder
    name), seed, step, panel, model, GPU identity, raw/scores paths (the alias source's raw outputs
    for alias units) and status (OK or UNVERIFIED for a stale alias).
    """
    root = Path(root)
    folders = sorted({p.parent for name in (*IDENTITY_FILES, 'alias.json') for p in root.glob('**/' + name)})
    cells = []
    for cell in folders:
        match = CELL_NAME.match(cell.name)
        if not match:
            continue
        meta, source = _identity_in(cell)
        raw, scores, status = _raw_of(cell), cell / 'scores.csv', 'OK'
        if (cell / 'alias.json').exists():
            alias = json.loads((cell / 'alias.json').read_text())
            target = _alias_source(cell, alias)
            target_meta, target_source = _identity_in(target)
            identity_file = next((target / n for n in IDENTITY_FILES if (target / n).exists()), None)
            if target_meta is None or not (target / 'scores.csv').exists() or \
                    pd_contract.sha256_file(target / 'scores.csv') != alias.get('scores_sha256') or \
                    identity_file is None or pd_contract.sha256_file(identity_file) != alias.get('identity_sha256'):
                status = 'UNVERIFIED'
            if meta is None:
                meta, source = target_meta, target_source
            raw = _raw_of(target) if target_meta is not None else None
            scores = target / 'scores.csv'
        if meta is None:
            continue
        for folder in {cell, target} if (cell / 'alias.json').exists() else {cell}:
            if (folder / 'manifest.json').exists():
                try:
                    from kit.sp_score import verify_outputs
                    verify_outputs(folder, required=())
                except (ValueError, OSError, KeyError):
                    status = 'UNVERIFIED'
        assignment = meta.get('assignment') or {}
        cells.append(dict(source=source, cell=str(cell.relative_to(root)), arm=match['arm'], seed=int(match['seed']),
                          step=int(match['step']), panel=match['panel'], family=assignment.get('family'),
                          model=meta.get('model'), panel_sha256=meta.get('panel_sha256'),
                          prompt_sha256=meta.get('prompt_sha256'), decoding=meta.get('decoding'),
                          timing=meta.get('timing'), gpu_hours=meta.get('gpu_hours'),
                          gpu_identity={k: (meta.get('gpu_identity') or {}).get(k) for k in ('uuid', 'environment')},
                          raw=str(raw) if raw else None, scores=str(scores) if scores.exists() else None, status=status))
    return cells


def _match_key(e):
    return (e.get('source'), e.get('family'), e.get('arm'), e.get('seed'), e.get('step'), e.get('panel'))


class Cells:
    """Lookup by full key; a cell without an assignment family (CPU stand-ins) matches only when the
    rest of its key is unique among the planned cells, otherwise it stays unmatched (fail-closed)."""

    def __init__(self, identities, planned=()):
        self.full, self.loose = {}, {}
        unique = {}
        for e in planned:
            if e['resolution'] == 'static':
                unique.setdefault(_match_key(e)[:1] + _match_key(e)[2:], set()).add(e.get('family'))
        for ident in identities:
            if ident.get('status', 'OK') != 'OK':
                continue
            if ident.get('family') is not None:
                self.full.setdefault(_match_key(ident), []).append(ident)
            else:
                loose = _match_key(ident)[:1] + _match_key(ident)[2:]
                if len(unique.get(loose, ())) == 1:
                    self.loose.setdefault(loose, []).append(ident)

    def get(self, key):
        return self.full.get(tuple(key)) or self.loose.get(tuple(key)[:1] + tuple(key)[2:], [])


def bind(planned, identities):
    """Distinct checkpoints and scoring cells; MISSING and NOT-RUN (conditional) entries are kept."""
    identities = list(identities)
    lookup = Cells(identities, planned)
    static_keys = {_match_key(e) for e in planned if e['resolution'] == 'static'}
    checkpoints, cells, missing, not_run = {}, {}, [], []
    for e in planned:
        if e['resolution'] == 'bound':
            found = [dict(weights=(e['weights_sha256'], e.get('model_file_hashes')))]
        elif e['resolution'] == 'static':
            found = lookup.get(_match_key(e))
        else:
            # Runtime rescoring binds the cells its family/seed assignment scored.
            found = [i for i in identities if i.get('planned_id') == e['planned_id']] or \
                [i for i in identities if i.get('status', 'OK') == 'OK' and i.get('family') is not None and
                 (i.get('source'), i.get('family'), i.get('seed')) == (e['source'], e.get('family'), e.get('seed'))
                 and _match_key(i) not in static_keys]
        if not found:
            (not_run if e.get('conditional') else missing).append(
                dict(planned_id=e['planned_id'], status='NOT-RUN' if e.get('conditional') else 'MISSING'))
            continue
        for ident in found:
            key, files = ident.get('weights') or weights_key(ident['model'])
            c = checkpoints.setdefault(key, dict(weights_sha256=key, model_file_hashes=files, planned_ids=[], uses=[],
                                                 weights_bound=files is not None))
            if e['planned_id'] not in c['planned_ids']:
                c['planned_ids'].append(e['planned_id'])
            arm = e.get('arm') if e.get('arm') is not None else ident.get('arm')
            step = e.get('step') if e.get('step') is not None else ident.get('step')
            use = dict(family=e.get('family'), seed=e.get('seed'), role=e.get('role') or role_of(dict(arm=arm, step=step)),
                       arm=arm, step=step, source=e['source'], comparison=e.get('comparison', arm),
                       incoming_key=e['incoming_key'][:5] if e.get('incoming_key') else None)
            if use not in c['uses']:
                c['uses'].append(use)
            cell = dict(weights_sha256=key, panel=e.get('panel') or ident.get('panel'), panel_sha256=ident.get('panel_sha256'),
                        prompt_sha256=ident.get('prompt_sha256'),
                        decoding_sha256=digest(ident['decoding']) if ident.get('decoding') else None,
                        gpu=list(gpu_key(ident.get('gpu_identity')) or ()))
            record = cells.setdefault(digest(cell), dict(cell, planned_ids=[]))
            if e['planned_id'] not in record['planned_ids']:
                record['planned_ids'].append(e['planned_id'])
    return checkpoints, cells, missing, not_run


def assignment_of(use):
    """SP's own GPU assignment for a connected comparison family/seed."""
    return 'sp:%s-s%s' % (use['family'], use['seed'])


def sp_entries(checkpoints):
    """One SP(b) scoring per distinct (weights, SP assignment), base/incoming/final uses only."""
    entries = {}
    for c in checkpoints.values():
        for use in c['uses']:
            if use['role'] not in SP_B_ROLES:
                continue
            assignment = assignment_of(use)
            key = digest(dict(weights=c['weights_sha256'], assignment=assignment))
            e = entries.setdefault(key, dict(entry_id='sp-' + key[:16], weights_sha256=c['weights_sha256'],
                                             model_file_hashes=c['model_file_hashes'], weights_bound=c['weights_bound'],
                                             assignment=assignment, uses=[], planned_ids=list(c['planned_ids'])))
            if use not in e['uses']:
                e['uses'].append(use)
    for e in entries.values():
        first = e['uses'][0]
        e.update(family=first['family'], seed=first['seed'], role=first['role'], checkpoint_id=e['weights_sha256'][:16])
    return sorted(entries.values(), key=lambda e: e['entry_id'])


def expand_return1(planned, binding):
    """Expand the owner's exact package calls before freezing, never invent a panel cross-product."""
    if not binding:
        return planned
    expanded = []
    for e in planned:
        if e['script'] != 'm2_rescore' or e['resolution'] != 'runtime':
            expanded.append(e); continue
        package = next((p for p in binding['packages'] if 'return1.' + p['id'] == e['family']), None)
        require(package is not None and package['seed'] == e['seed'], 'return-1 package not bound')
        calls = [(False, c) for c in package['calls']] + [(True, c) for c in package.get('retained_calls', [])]
        for index, (retained, call) in enumerate(calls):
            match = CELL_NAME.match(Path(call['score_set']).name)
            require(match is not None and (retained or int(match['seed']) == e['seed']), 'invalid return-1 call')
            item = dict(e, planned_id=e['planned_id'] + ':call%d' % index, resolution='static',
                        arm=match['arm'], step=call['step'], panel=call['panel_set'],
                        comparison=match['arm'], return1_call_sha256=digest(call), retained=retained)
            item['role'] = role_of(item)
            expanded.append(item)
    return expanded


def retained_return1(root, planned):
    """Core retained calls keep their original identities and may lack raw text (then MISSING)."""
    root = Path(root)
    binding = root / 'binding.json'
    if not binding.is_file():
        return []
    doc = json.loads(binding.read_text())
    out = []
    for package in doc.get('packages', []):
        for call in package.get('retained_calls', []):
            e = next((p for p in planned if p.get('retained') and p.get('family') == 'return1.' + package['id'] and
                      p.get('return1_call_sha256') == digest(call)), None)
            if e is None:
                continue
            base = root / 'retained' / package['id']
            cell = base / call['score_set']
            identity, companion = base / call['identity_path'], base / call['companion_path']
            if not all(p.is_file() for p in (identity, companion, cell / 'scores.csv')):
                continue
            meta = json.loads(companion.read_text())
            status = 'OK' if pd_contract.sha256_file(identity) == call['identity_sha256'] else 'UNVERIFIED'
            gpu = meta.get('gpu_identity') or {}
            if (gpu.get('uuid'), gpu.get('environment')) != (package['assigned_gpu_uuid'], package['compatible_environment']):
                status = 'UNVERIFIED'
            out.append(dict(source='message2', family=e['family'], seed=e['seed'], arm=e['arm'], step=e['step'], panel=e['panel'],
                            model=dict(model_file_hashes=call['model_hashes']), status=status,
                            cell=str(cell.relative_to(root)), scores=str(cell / 'scores.csv'),
                            raw=str(_raw_of(cell)) if _raw_of(cell) else None, gpu_identity={k: gpu.get(k) for k in ('uuid', 'environment')},
                            panel_sha256=None, prompt_sha256=None, decoding=None))
    return out


def comparison_edges(planned):
    """Each descendant names its registered step-zero cell, independently of GPU grouping."""
    for e in planned:
        if e.get('resolution') != 'static':
            continue
        candidates = [p for p in planned if p.get('resolution') == 'static' and p.get('step') == 0 and
                      (p['source'], p['family'], p['seed'], p['panel']) ==
                      (e['source'], e['family'], e['seed'], e['panel'])]
        same = [p for p in candidates if p['arm'] == e['arm']]
        pre = [p for p in candidates if _is_pre(p['arm'])]
        incoming = same or pre
        keys = {_match_key(p) for p in incoming}
        e['incoming_key'] = list(next(iter(keys))) if len(keys) == 1 else None
        e['comparison'] = e.get('comparison') or e.get('arm')
    return planned


def unresolved_entries(planned, bound_entries):
    """Keep future checkpoints in the immutable coverage plan and its price allowance."""
    def checkpoint(p):
        return tuple(p.get('alias_target') or (p['source'], p['family'], p['seed'], p['arm'], p['step']))
    by_id = {p['planned_id']: p for p in planned}
    bound_targets = {checkpoint(by_id[pid]): e for e in bound_entries for pid in e['planned_ids'] if pid in by_id}
    covered = {p for e in bound_entries for p in e['planned_ids']}
    groups = {}
    for p in planned:
        if p['planned_id'] in covered or p.get('role') not in (*SP_B_ROLES, None):
            continue
        key = checkpoint(p)
        groups.setdefault(key, []).append(p)
    out = []
    for key, members in sorted(groups.items(), key=lambda kv: str(kv[0])):
        p = members[0]
        uses = []
        for member in members:
            use = {k: member.get(k) for k in ('source', 'family', 'seed', 'arm', 'step', 'role', 'comparison', 'incoming_key')}
            if use.get('incoming_key'):
                use['incoming_key'] = use['incoming_key'][:5]
            if use not in uses:
                uses.append(use)
        if key in bound_targets:
            entry = bound_targets[key]
            entry['uses'] += [u for u in uses if u not in entry['uses']]
            entry['planned_ids'] += [m['planned_id'] for m in members if m['planned_id'] not in entry['planned_ids']]
            continue
        out.append(dict(entry_id='sp-plan-' + digest(key)[:16], checkpoint_id=None,
                        assignment=assignment_of(p), family=p['family'], seed=p['seed'], role=p.get('role'),
                        weights_sha256=None, model_file_hashes=None, weights_bound=False,
                        status='UNRESOLVED', coverage_unknown=p['resolution'] == 'runtime',
                        uses=uses, planned_ids=[m['planned_id'] for m in members], alias_target=list(key)))
    return out


def inventory(launches, pd=None, tiers=(), identities=(), return1=None):
    identities = list(identities)
    launches = as_list(launches)
    planned, skipped = [], []
    for launch in launches:
        entries, skip = planned_from_launch(launch)
        planned += entries; skipped += skip
    for doc in tiers:
        planned += planned_from_tier(doc)
    require(len({e['planned_id'] for e in planned}) == len(planned), 'duplicate planned inventory ID')
    planned = comparison_edges(expand_return1(planned, return1))
    kept, dropped = screen(planned)
    checkpoints, cells, missing, not_run = bind(kept, list(identities))
    entries = sp_entries(checkpoints)
    entries += unresolved_entries(kept, entries)
    starts = pd_starts(pd) if pd is not None else []
    for s in starts:
        s['in_chain_inventory'] = s['weights_sha256'] in checkpoints
    unverified = sorted(i['cell'] for i in identities if i.get('status', 'OK') != 'OK' and i.get('cell'))
    return seal(dict(schema=INVENTORY_SCHEMA, launch_sha256=[digest(l) for l in launches],
                     pd_sha256=pd['identity_sha256'] if pd else None, tiers=sorted(d['tier'] for d in tiers),
                     planned=kept, dropped=dropped, skipped=skipped,
                     other_scoring=[s for s in skipped if s.get('scoring_task') and not exclusion(s)],
                     checkpoints=sorted(checkpoints.values(), key=lambda c: c['weights_sha256']),
                     cells=sorted(cells.values(), key=lambda c: digest(c)), missing=missing, not_run=not_run,
                     return1_binding_sha256=digest(return1) if return1 else None,
                     parent_gpu_uuids=sorted({i['gpu_identity']['uuid'] for i in identities if gpu_key(i.get('gpu_identity'))}),
                     unverified_cells=unverified, pd_starts=starts, sp_entries=entries, price=price_entries(entries)))


def price_entries(entries):
    """All entries priced; nothing is silently dropped. Timing is a receipt, never a pass gate."""
    lo, hi = PRICE['per_entry_gpu_h']
    per = {}
    for e in entries:
        per[e['assignment']] = per.get(e['assignment'], 0) + 1
    risk = sorted(a for a, n in per.items() if n * hi > PRICE['wall_time_risk_hours'])
    unknown = any(e.get('coverage_unknown') for e in entries)
    return dict(entries=len(entries), coverage_unknown=unknown,
                sp_a_gpu_hours=PRICE['sp_a_gpu_h'],
                sp_b_gpu_hours=(PRICE['startup_gpu_h'] + len(entries) * lo, None if unknown else PRICE['startup_gpu_h'] + len(entries) * hi),
                entries_per_assignment=per, wall_time_risk_at_upper_allowance=risk, gate=False, basis=dict(PRICE),
                measured=False)


def freeze_inventory(path, doc):
    """Write-once: an inventory frozen before send cannot be silently replaced."""
    verify_seal(doc)
    path = Path(path)
    if path.exists():
        existing = json.loads(path.read_text())
        verify_seal(existing)
        require(existing['identity_sha256'] == doc['identity_sha256'], 'SP inventory already frozen with different content')
        return existing
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, sort_keys=True, indent=1) + '\n')
    return doc


# ---------------------------------------------------------------------------------------------
# Candidate-superset verification (returned broad/K/CR pools only verify; nothing is deleted).

def prompt_hash(row):
    import hashlib
    import unicodedata
    prompt = row.get('prompt', row.get('content', ''))
    if isinstance(prompt, list):
        prompt = '\n'.join(m.get('content', '') for m in prompt if isinstance(m, dict))
    text = ' '.join(unicodedata.normalize('NFKC', str(prompt)).casefold().split())
    return hashlib.sha256(text.encode()).hexdigest()


def verify_pools(superset, pools):
    """Each returned pool must be a subset of its frozen candidate superset.

    superset: {"schema", "pools": {name: [normalised-prompt sha256, ...]}} frozen before send.
    pools: {name: rows}. A mismatch is UNVERIFIED; no SP row, request or result is deleted.
    """
    require(superset.get('schema') == SUPERSET_SCHEMA, 'unknown candidate superset schema')
    result = {}
    for name, rows in sorted(pools.items()):
        allowed = set(superset['pools'].get(name, ()))
        if not allowed:
            result[name] = dict(status='UNVERIFIED', reason='no frozen superset for this pool', rows=len(rows))
            continue
        outside = sorted({prompt_hash(r) for r in rows} - allowed)
        result[name] = dict(status='VERIFIED' if not outside else 'UNVERIFIED', rows=len(rows), outside=len(outside),
                            outside_sha256=outside[:20])
    return dict(schema='sp-prompt-containment-only-v1', superset_sha256=digest(superset), pools=result, deletes=0,
                containment_status='VERIFIED' if result and all(r['status'] == 'VERIFIED' for r in result.values()) else 'UNVERIFIED',
                status='UNVERIFIED', reason='prompt containment alone cannot verify exclusions; use sp_prepare.recheck')


def verify_pools_file(superset_path, pools):
    """verify_pools from m3a2's published file; an unpublished superset is UNVERIFIED, never an error."""
    path = Path(superset_path)
    if not path.is_file():
        return dict(schema='sp-pool-verification-v1', superset_sha256=None, pools={}, deletes=0, status='UNVERIFIED',
                    reason='candidate superset not published at ' + SUPERSET_PATH.as_posix())
    return verify_pools(json.loads(path.read_text()), pools)


# ---------------------------------------------------------------------------------------------
# Sidecar manifest.

def build_sidecar(launches, inv, settings=None):
    settings = settings or {}
    launches = as_list(launches)
    verify_seal(inv)
    py = settings.get('inference_python', 'python3')
    parent = {t['id'] for l in launches for t in l['tasks']}
    planned = {e['planned_id']: e for e in inv['planned']}
    tasks = []

    def add(ident, after, commands, uses_gpu=False, **extra):
        tasks.append(dict(id=ident, kind='sp-sidecar', after=sorted(set(after)), dependency='afterany',
                          commands=commands, writes='{sp}', uses_gpu=uses_gpu, blocks=[], **extra))
        return ident

    pool_parents = [t for t in settings.get('pool_tasks', ('M-pools', 'R-broad', 'R-nominate', 'M3-pools')) if t in parent]
    verify = add('SP-verify-pools', pool_parents, [[py, '{kit}/sp_manifest.py', 'verify-pools',
                                                    '--pools-spec', '{sp}/returned-pools.json',
                                                    '--out', '{sp}/pool-verification.json']])
    by_assignment = {}
    for e in inv['sp_entries']:
        by_assignment.setdefault(e['assignment'], []).append(e)
    panels = []
    for assignment, entries in sorted(by_assignment.items()):
        after, commands = [], []
        for e in sorted(entries, key=lambda e: (min(({'base': 0, 'incoming': 1, 'final': 2}.get(u.get('role'), 3)
                                                    for u in e['uses']), default=3), e['entry_id'])):
            after += [planned[p]['task'] for p in e['planned_ids'] if planned.get(p, {}).get('task') in parent]
            commands.append([py, '{kit}/jobs/message3-sp/durable_adapter.py', '--sources', '{owner_sources_file}',
                             '--entry-id', e['entry_id'], '--out', '{sp}'])
            commands.append([py, '{kit}/sp_manifest.py', 'entry', '--inventory', '{sp}/inventory.json',
                             '--entry-id', e['entry_id'], '--models', '{models_file}', '--input', '{inputs}',
                             '--out', '{sp}/entries/%s.json' % e['entry_id']])
            commands.append([py, '{kit}/sp_score.py', 'panel', '--entry', '{sp}/entries/%s.json' % e['entry_id'],
                             '--model', '{model:%s}' % e['entry_id'], '--gpu', '{score_gpu}',
                             '--models', '{models_file}',
                             '--gpu-identity', '{gpu_identity:%s}' % assignment, '--out', '{sp}'])
        name = 'SP-panel-' + re.sub(r'[^A-Za-z0-9_.-]+', '-', assignment.split(':', 1)[-1])
        panels.append(add(name, after, commands, uses_gpu=True, assignment=assignment,
                          entries=[e['entry_id'] for e in entries]))
    scorers = sorted({p['task'] for p in inv['planned'] if p.get('task') in parent})
    length = add('SP-length', scorers, [[py, '{kit}/sp_manifest.py', 'length', '--inventory', '{sp}/inventory.json',
                                         '--sp', '{sp}', '--input', '{inputs}']])
    add('SP-read', [*panels, length, verify], [[py, '{kit}/sp_read.py', '--sidecar', '{sp}', '--inventory',
                                                '{sp}/inventory.json', '--input', '{inputs}', '--out', '{sp}/tables']])
    return dict(schema=SIDECAR_SCHEMA, parent_launch_sha256=[digest(l) for l in launches],
                inventory_sha256=inv['identity_sha256'], tasks=tasks)


def validate_sidecar(sidecar, launches):
    """Raise unless the sidecar can never block, alter or write into any parent job."""
    launches = as_list(launches)
    require(sidecar.get('schema') == SIDECAR_SCHEMA, 'wrong sidecar schema')
    require(sidecar.get('parent_launch_sha256') == [digest(l) for l in launches], 'sidecar built for different launch manifests')
    parent = {}
    for launch in launches:
        for task in launch['tasks']:
            require(task['id'] not in parent, 'parent task ID repeated across launch manifests: ' + task['id'])
            parent[task['id']] = task
    ids = [t['id'] for t in sidecar['tasks']]
    require(len(ids) == len(set(ids)), 'duplicate sidecar task')
    require(all(i.startswith('SP-') for i in ids), 'sidecar task IDs must start with SP-')
    require(not set(ids) & set(parent), 'sidecar task ID collides with a parent task')
    require(not any(i.startswith('SP-') for i in parent), 'a parent manifest contains an SP task')
    for task in parent.values():
        links = [x for k in PARENT_LINK_KEYS for x in task.get(k, [])]
        links += [i.get('task') for i in task.get('imports', []) if isinstance(i, dict)]
        require(not any(isinstance(x, str) and x.startswith('SP-') for x in links),
                'parent task %s depends on the SP sidecar' % task['id'])
        require(not re.search(r'sp_(prepare|score|read|manifest)\.py', json.dumps(task.get('commands', []))),
                'parent task %s runs SP code' % task['id'])
    known = set(parent) | set(ids)
    gpus = []
    for task in sidecar['tasks']:
        require(task.get('dependency') == 'afterany', task['id'] + ': SP waits with afterany only')
        require(task.get('blocks') == [], task['id'] + ': SP tasks block nothing')
        require(all(a in known for a in task['after']), task['id'] + ': unknown dependency')
        require(not any(k in task for k in ('commit_barrier', 'required_values', 'barriers', 'required_outputs')),
                task['id'] + ': SP tasks carry no barriers or required outputs')
        for command in task['commands']:
            for arg in command:
                require('{out}' not in str(arg) and '{durable}' not in str(arg),
                        task['id'] + ': SP never writes into parent output roots')
        if task.get('uses_gpu'):
            gpus.append(task.get('assignment'))
    require(len(gpus) == len(set(gpus)) and None not in gpus, 'one SP GPU task per family/seed assignment')
    order, done, waiting = [], set(), list(sidecar['tasks'])
    while waiting:
        ready = [t for t in waiting if all(a in done or a in parent for a in t['after'])]
        require(bool(ready), 'sidecar dependency cycle')
        order += [t['id'] for t in ready]; done.update(t['id'] for t in ready)
        waiting = [t for t in waiting if t['id'] not in done]
    return order


# ---------------------------------------------------------------------------------------------
# Runner: always exits 0, records status.

class MissingInput(Exception):
    pass


def parent_hardware(parent_inventories):
    """Complete owner occupancy receipts; an omitted hold state conservatively means held."""
    def records(value):
        if isinstance(value, dict):
            return ([value] if value.get('uuid') else []) + [r for v in value.values() for r in records(v)]
        if isinstance(value, list):
            return [r for v in value for r in records(v)]
        return []
    require(bool(parent_inventories) and all(p.get('complete_inventory') is True for p in parent_inventories),
            'complete parent hardware inventories required')
    rows = [r for p in parent_inventories for r in records(p)]
    all_uuids = {r['uuid'] for r in rows}
    require(bool(all_uuids), 'parent hardware inventory contains no UUIDs')
    held = set()
    for row in rows:
        if row.get('held') is False:
            require(row.get('released') is True and pd_contract.is_sha(row.get('release_receipt_sha256')),
                    'released parent GPU needs an owner release receipt')
        else:
            held.add(row['uuid'])
    return all_uuids, held


def freeze_hardware(path, assignments, parent_inventories):
    """Separate immutable assignment. Historical parent devices require explicit release evidence."""
    historical, parent = parent_hardware(parent_inventories)
    seen = set()
    for key, value in assignments.items():
        require(key.startswith('sp:') and value.get('uuid') and value.get('environment'), 'incomplete SP GPU assignment')
        require(value['uuid'] not in parent | seen, 'SP GPU shared with parent or another assignment')
        seen.add(value['uuid'])
    doc = seal(dict(schema='sp-hardware-v1', assignments={k: {f: v[f] for f in ('uuid', 'environment')}
                                                         for k, v in assignments.items()},
                    parent_gpu_uuids=sorted(parent), historical_parent_gpu_uuids=sorted(historical),
                    parent_inventory_sha256=[digest(p) for p in parent_inventories]))
    return freeze_inventory(path, doc)


def archive_fingerprint(path):
    stat = Path(path).stat()
    return dict(device=stat.st_dev, inode=stat.st_ino, bytes=stat.st_size,
                mtime_ns=stat.st_mtime_ns, ctime_ns=stat.st_ctime_ns)


def verify_archive_once(path, expected, verified):
    """One full read at entry binding; later consumers require the same immutable file proof."""
    before = archive_fingerprint(path)
    if verified is None:
        require(pd_contract.sha256_file(path) == expected, 'UNVERIFIED: durable snapshot bytes changed')
    else:
        require(verified.get('snapshot_sha256') == expected and verified.get('archive_fingerprint') == before,
                'UNVERIFIED: durable snapshot changed after entry binding')
    require(archive_fingerprint(path) == before, 'UNVERIFIED: snapshot changed during verification')
    return before


def native_owner_document(local):
    """Read native owner artifacts only. Produces metadata, never checkpoints or retention state."""
    import hashlib
    import tarfile
    root, model = Path(local['durable_root']).resolve(), Path(local['model'])
    require(local.get('durable') is True, 'MISSING: owner durable location must be attested')
    require(model.is_dir() and model.resolve().is_relative_to(root) and not model.is_symlink(),
            'MISSING: surviving durable owner export required')
    kind = local['kind']
    extras = {}
    if kind == 'm2-snapshot':
        archive = Path(local['archive'])
        require(archive.is_file() and archive.resolve().is_relative_to(root), 'MISSING: owner snapshot absent')
        checksum = archive.with_suffix('.tar.sha256').read_text().strip()
        header_hash = archive.with_suffix('.tar.manifest.sha256').read_text().strip()
        with tarfile.open(archive, 'r:') as stream:
            first = stream.next()
            require(first is not None and first.name == 'snapshot.json', 'UNVERIFIED: native snapshot header missing')
            header = json.load(stream.extractfile(first))
            require(header.get('schema') == 'message2-stage-snapshot.v2' and header.get('complete') is True,
                    'MISSING: native snapshot is not complete')
            require(digest(header) == header_hash, 'UNVERIFIED: native snapshot header changed')
            member_name = 'tree/' + local['owner_receipt_member']
            members = [m for m in stream.getmembers() if m.name == member_name]
            require(len(members) == 1 and members[0].isfile(), 'UNVERIFIED: owner export receipt missing')
            owner_raw = stream.extractfile(members[0]).read()
        owner_hash = hashlib.sha256(owner_raw).hexdigest()
        require(header['files'].get(local['owner_receipt_member'], {}).get('sha256') == owner_hash,
                'UNVERIFIED: native owner receipt changed')
        owner = json.loads(owner_raw)
        files = owner.get('model_file_hashes', owner.get('identity', {}).get('model_file_hashes'))
        require(bool(files), 'UNVERIFIED: owner export receipt lacks file hashes')
        prefix = local['export_prefix'].rstrip('/') + '/'
        require(all(header['files'].get(prefix + name, {}).get('sha256') == sha for name, sha in files.items()),
                'UNVERIFIED: export differs from native snapshot inventory')
        completion = header
        extras = dict(snapshot_sha256=checksum, snapshot_manifest_sha256=header_hash)
        complete_hash = header_hash
    elif kind == 'm3-export':
        owner_path = Path(local['owner_receipt'])
        require(owner_path.resolve().is_relative_to(root), 'UNVERIFIED: owner receipt outside durable root')
        run = owner_path.parent
        manifest_raw, complete_raw = (run / 'manifest.json').read_bytes(), (run / 'complete.json').read_bytes()
        manifest, completion = json.loads(manifest_raw), json.loads(complete_raw)
        require(completion.get('complete') is True and completion.get('receipt_status') == 'PASS',
                'MISSING: owner M3 export not complete and qualified')
        owner_raw = owner_path.read_bytes(); owner_hash = hashlib.sha256(owner_raw).hexdigest()
        require(manifest['files'].get(owner_path.name) == owner_hash, 'UNVERIFIED: native M3 export receipt changed')
        owner = json.loads(owner_raw); files = owner.get('model_file_hashes')
        require(bool(files), 'UNVERIFIED: native M3 export hashes missing')
        require(owner_path.name.startswith('export-') and owner_path.suffix == '.json', 'UNVERIFIED: native export receipt name')
        step = owner_path.stem.removeprefix('export-')
        require(model.resolve() == (run / ('hf-step' + step)).resolve(), 'UNVERIFIED: M3 export location differs')
        extras = dict(owner_manifest_sha256=hashlib.sha256(manifest_raw).hexdigest())
        complete_hash = hashlib.sha256(complete_raw).hexdigest()
    else:
        raise ValueError('unknown native owner export kind')
    require(all(Path(name).name == name and (model / name).is_file() for name in files), 'MISSING: durable export files absent')
    deadlines = [value['expires_at'] for value in (owner, completion, local) if value.get('expires_at') is not None]
    expiry = min(deadlines) if deadlines else None
    if expiry is not None and time.time() >= expiry:
        raise MissingInput('MISSING: native owner export expired')
    policy = dict(basis=kind, completion_sha256=complete_hash, expires_at=expiry,
                  read_surviving_exports_only=True, extends_retention=False)
    return seal(dict(schema='sp-durable-export-v1', adapter='sp-native-owner-v1', durable=True, kind=kind,
                     export_locator_sha256=digest(model.resolve().relative_to(root).as_posix()),
                     model_file_hashes=files, owner_receipt_sha256=owner_hash, owner_complete_sha256=complete_hash,
                     retention_policy=policy, retention_policy_sha256=digest(policy), expires_at=expiry, **extras))


def adapt_owner_exports(spec_path, out, entry_id=None):
    """CPU-only metadata adapter. Native owner bytes are never written, copied, restored or pinned."""
    import fcntl
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    # Independent SP assignment jobs can refresh their entries concurrently. This SP-owned
    # metadata lock prevents lost map updates; it never locks or leases an owner checkpoint.
    with (out / '.models-local.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            return _adapt_owner_exports(spec_path, out, entry_id)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _adapt_owner_exports(spec_path, out, entry_id=None):
    from kit.sp_score import write_json
    out = Path(out)
    if not spec_path or not Path(spec_path).is_file():
        return dict(status='MISSING', entries={})
    sources = json.loads(Path(spec_path).read_text())
    sources = sources.get('entries', sources)
    map_path = out / 'models.local.json'
    models = json.loads(map_path.read_text()) if map_path.is_file() else {}
    states = {}
    if entry_id is not None and entry_id not in sources:
        models.pop(entry_id, None)
        states[entry_id] = dict(status='MISSING', reason='unbound-owner-source')
    for ident, source in sorted(sources.items()):
        if entry_id is not None and ident != entry_id:
            continue
        require(re.fullmatch(r'[A-Za-z0-9_.-]+', ident) is not None, 'unsafe SP entry ID')
        try:
            doc = native_owner_document(source)
            receipt = out / 'adapters' / (ident + '.json')
            freeze_inventory(receipt, doc)
            models[ident] = dict(source, receipt=str(receipt.resolve()), receipt_sha256=pd_contract.sha256_file(receipt))
            states[ident] = dict(status='READY')
        except (OSError, ValueError, KeyError, MissingInput) as exc:
            models.pop(ident, None)
            states[ident] = dict(status='MISSING', reason=type(exc).__name__)
    write_json(map_path, models)  # local map: never part of the returned scientific receipts
    write_json(out / 'state' / ('adapter-' + (entry_id or 'plan') + '.json'), dict(entries=states))
    return dict(status='READY' if states and all(s['status'] == 'READY' for s in states.values()) else 'MISSING', entries=states)


def durable_binding(local, verified=None):
    """Consume an owner-issued surviving durable export; never restore or create one.

    The local map supplies receipt, immutable receipt hash, model and archive locations. The
    portable owner receipt names the existing export relative to itself and binds the owner's
    immutable snapshot manifest, export file inventory and unchanged retention policy.
    """
    require(isinstance(local, dict), 'MISSING: owner durable export receipt required')
    receipt = Path(local['receipt'])
    if local.get('kind') in ('m2-snapshot', 'm3-export'):
        require(receipt.is_file() and pd_contract.sha256_file(receipt) == local.get('receipt_sha256'),
                'UNVERIFIED: SP adapter receipt changed')
        doc = json.loads(receipt.read_text()); verify_seal(doc)
        require(doc == native_owner_document(local), 'UNVERIFIED: native owner artifact binding changed')
        if doc.get('expires_at') is not None and time.time() >= doc['expires_at']:
            raise MissingInput('MISSING: native owner export expired')
        proof = {k: doc[k] for k in ('owner_receipt_sha256', 'retention_policy_sha256', 'expires_at', 'model_file_hashes')}
        proof.update(receipt_sha256=local['receipt_sha256'], owner_complete_sha256=doc['owner_complete_sha256'])
        if doc['kind'] == 'm2-snapshot':
            proof.update(snapshot_sha256=doc['snapshot_sha256'], snapshot_manifest_sha256=doc['snapshot_manifest_sha256'])
            proof['archive_fingerprint'] = verify_archive_once(local['archive'], doc['snapshot_sha256'], verified)
        else:
            proof['owner_manifest_sha256'] = doc['owner_manifest_sha256']
        require(verified is None or proof == verified, 'UNVERIFIED: frozen durable proof changed')
        return proof
    model, archive = Path(local['model']), Path(local['archive'])
    if not receipt.is_file() or not model.is_dir() or not archive.is_file():
        raise MissingInput('MISSING: owner durable export or snapshot no longer survives')
    require(pd_contract.sha256_file(receipt) == local.get('receipt_sha256'), 'UNVERIFIED: owner export receipt changed')
    doc = json.loads(receipt.read_text())
    require(doc.get('schema') == 'sp-durable-export-v1' and doc.get('durable') is True,
            'UNVERIFIED: owner durable export receipt required')
    require(doc.get('retention_policy_sha256') and doc.get('owner_receipt_sha256'), 'UNVERIFIED: retention/owner provenance missing')
    if doc.get('expires_at') is not None and time.time() >= doc['expires_at']:
        raise MissingInput('MISSING: owner durable export expired')
    relative = Path(doc['export'])
    require(not relative.is_absolute() and '..' not in relative.parts, 'unsafe durable export reference')
    require(model.resolve() == (receipt.parent / relative).resolve(), 'UNVERIFIED: transient export location')
    require(not model.is_symlink(), 'UNVERIFIED: durable export is a symlink')
    durable_root = Path(local['durable_root']).resolve()
    require(all(p.resolve().is_relative_to(durable_root) for p in (model, receipt, archive)),
            'UNVERIFIED: export/receipt/snapshot outside owner durable root')
    # Read the immutable snapshot header in place; no extraction, copying or retention mutation.
    import tarfile
    with tarfile.open(archive, 'r:') as stream:
        first = stream.next()
        require(first is not None and first.name == 'snapshot.json', 'UNVERIFIED: owner snapshot header missing')
        snapshot = json.load(stream.extractfile(first))
        def member(name):
            value = next((m for m in stream.getmembers() if m.name == 'tree/' + name), None)
            require(value is not None and value.isfile(), 'UNVERIFIED: owner policy/receipt member absent')
            return stream.extractfile(value).read()
        policy_raw = member(doc['retention_policy_member'])
        owner_raw = member(doc['owner_receipt_member'])
    require(digest(snapshot) == doc['snapshot_manifest_sha256'], 'UNVERIFIED: durable snapshot identity changed')
    require(snapshot.get('complete') is True, 'MISSING: owner snapshot incomplete')
    fingerprint = verify_archive_once(archive, doc['snapshot_sha256'], verified)
    prefix = doc['snapshot_export_prefix'].rstrip('/') + '/'
    files = doc['model_file_hashes']
    for name, sha in files.items():
        require(snapshot['files'].get(prefix + name, {}).get('sha256') == sha, 'UNVERIFIED: export not in owner snapshot')
    # Owner receipt is itself a member of that immutable snapshot, independently of SP's adapter.
    require(snapshot['files'].get(doc['owner_receipt_member'], {}).get('sha256') == doc['owner_receipt_sha256'],
            'UNVERIFIED: owner receipt not in snapshot')
    require(snapshot['files'].get(doc['retention_policy_member'], {}).get('sha256') == doc['retention_policy_sha256'],
            'UNVERIFIED: retention policy not bound by owner snapshot')
    import hashlib
    require(hashlib.sha256(policy_raw).hexdigest() == doc['retention_policy_sha256'] and
            hashlib.sha256(owner_raw).hexdigest() == doc['owner_receipt_sha256'], 'UNVERIFIED: owner provenance bytes differ')
    policy, owner = json.loads(policy_raw), json.loads(owner_raw)
    require('expires_at' in policy and policy['expires_at'] == doc.get('expires_at'),
            'UNVERIFIED: adapter changes owner retention deadline')
    require(owner.get('model_file_hashes', owner.get('identity', {}).get('model_file_hashes')) == files,
            'UNVERIFIED: owner export receipt weights differ')
    return dict(receipt_sha256=local['receipt_sha256'], snapshot_sha256=doc['snapshot_sha256'],
                archive_fingerprint=fingerprint,
                snapshot_manifest_sha256=doc['snapshot_manifest_sha256'], owner_receipt_sha256=doc['owner_receipt_sha256'],
                retention_policy_sha256=doc['retention_policy_sha256'], expires_at=doc.get('expires_at'),
                model_file_hashes=files)


def bind_runtime_entry(entry, inv, inputs, local):
    """Bind missing future identities after parents finish; coverage/entry IDs never change."""
    if entry.get('coverage_unknown'):
        raise MissingInput('MISSING: exact return-1 package calls must be bound before inventory freeze')
    plan = [p for p in inv['planned'] if p['planned_id'] in entry['planned_ids']]
    found = [i for root in inputs for i in [*discover(root), *retained_return1(root, inv['planned'])]]
    candidates = [i for i in found if i.get('status') == 'OK' and any(_match_key(i) == _match_key(p) for p in plan)]
    require(bool(candidates) or entry.get('weights_bound'), 'MISSING: parent checkpoint not produced')
    weights = {weights_key(i['model'])[0] for i in candidates}
    require(len(weights) <= 1, 'UNVERIFIED: panels disagree on checkpoint weights')
    proof = durable_binding(local)
    expected = digest(proof['model_file_hashes'])
    require(not weights or weights == {expected}, 'UNVERIFIED: owner export differs from parent scores')
    require(not entry.get('weights_sha256') or entry['weights_sha256'] == expected, 'UNVERIFIED: bound export changed')
    return dict(entry, weights_sha256=expected, model_file_hashes=proof['model_file_hashes'], weights_bound=True,
                checkpoint_id=expected[:16], status='BOUND', durable_provenance=proof)


def substitute(arg, values, maps):
    def model(m):
        models_path = values.get('{models_file}')
        models = json.loads(Path(models_path).read_text()) if models_path and Path(models_path).is_file() else maps.get('models', {})
        found = models.get(m.group(1))
        if not found:
            raise MissingInput('no local model location for ' + m.group(1))
        bound_path = Path(values['{sp}']) / 'entries' / (m.group(1) + '.json')
        if not bound_path.is_file():
            raise MissingInput('MISSING: immutable runtime entry proof absent')
        bound = json.loads(bound_path.read_text()); verify_seal(bound)
        proof = durable_binding(found, verified=bound.get('durable_provenance'))
        require(proof == bound.get('durable_provenance'), 'UNVERIFIED: frozen durable proof differs')
        return str(found['model'])
    def gpu(m):
        hardware = maps.get('hardware')
        if not hardware:
            raise MissingInput('no frozen SP hardware authority')
        verify_seal(hardware)
        require(hardware.get('schema') == 'sp-hardware-v1', 'UNVERIFIED: wrong hardware authority schema')
        assigned = [v.get('uuid') for v in hardware['assignments'].values()]
        observed_parents = set(maps.get('observed_parent_gpu_uuids', []))
        historical = set(hardware.get('historical_parent_gpu_uuids', hardware['parent_gpu_uuids']))
        require(observed_parents <= historical, 'UNVERIFIED: parent hardware inventory omitted observed UUIDs')
        if set(assigned) & historical:
            if not maps.get('parent_inventories'):
                raise MissingInput('fresh parent occupancy receipts required for released GPU reuse')
            current_scope, current_held = parent_hardware(maps['parent_inventories'])
            require(historical <= current_scope, 'UNVERIFIED: current parent hardware inventory incomplete')
            require(not set(assigned) & current_held, 'UNVERIFIED: SP GPU currently held by a parent')
        require(len(assigned) == len(set(assigned)) and not set(assigned) & set(hardware['parent_gpu_uuids']),
                'UNVERIFIED: shared SP/parent physical GPU')
        found = hardware['assignments'].get(m.group(1))
        if not found or not found.get('uuid') or not found.get('environment'):
            raise MissingInput('no physical GPU identity for assignment ' + m.group(1))
        path = Path(values['{sp}']) / 'state' / 'gpu' / (re.sub(r'[^A-Za-z0-9_.-]+', '-', m.group(1)) + '.json')
        path.parent.mkdir(parents=True, exist_ok=True)
        proof = dict(found, assignment=m.group(1), hardware_sha256=hardware['identity_sha256'])
        freeze_inventory(path, seal(proof))
        return str(path)
    out = re.sub(r'\{model:([^}]+)\}', model, str(arg))
    out = re.sub(r'\{gpu_identity:([^}]+)\}', gpu, out)
    for key, value in values.items():
        out = out.replace(key, str(value))
    if re.search(r'\{[a-z_]+(?::[^}]*)?\}', out):
        raise MissingInput('unresolved SP placeholder')
    return out


def expand_inputs(command, inputs):
    """'--input {inputs}' becomes one '--input ROOT' per parent output root supplied."""
    out, i = [], 0
    while i < len(command):
        if command[i] == '--input' and i + 1 < len(command) and command[i + 1] == '{inputs}':
            if not inputs:
                raise MissingInput('no parent output roots supplied')
            for root in inputs:
                out += ['--input', str(root)]
            i += 2
        else:
            out.append(command[i]); i += 1
    return out


def run(sidecar, task_id, values, maps, inputs=(), runner=subprocess.run):
    """Run one SP task. Never raises and never returns nonzero: status goes to {sp}/state/.

    Commands are independent units (one entry's preparation and scoring per pair): a failed or
    missing entry is recorded and the remaining entries of the assignment still run.
    """
    state = Path(values['{sp}']) / 'state'
    state.mkdir(parents=True, exist_ok=True)
    record = dict(task=task_id, started=time.time(), commands=[])
    try:
        task = next(t for t in sidecar['tasks'] if t['id'] == task_id)
        record['assignment'] = task.get('assignment')
        if task.get('uses_gpu'):
            maps = dict(maps, observed_parent_gpu_uuids=sorted({c['gpu_identity']['uuid'] for root in inputs for c in discover(root)
                                                              if gpu_key(c.get('gpu_identity'))}))
        prepared = True
        for raw in task['commands']:
            item = dict(script=Path(str(raw[1])).name if len(raw) > 1 else None)
            if '--entry-id' in raw:
                item['entry_id'] = raw[raw.index('--entry-id') + 1]
            elif '--entry' in raw:
                item['entry_id'] = Path(raw[raw.index('--entry') + 1]).stem
            started = time.monotonic()
            try:
                if len(raw) > 1 and str(raw[1]).endswith('/sp_score.py') and not prepared:
                    raise MissingInput('MISSING: this entry failed runtime binding')
                command = [substitute(a, values, maps) for a in expand_inputs(raw, inputs)]
                result = runner(command, check=False)
                item.update(status='DONE' if result.returncode == 0 else 'FAILED', returncode=result.returncode)
            except MissingInput as exc:
                item.update(status='MISSING-INPUT', error=str(exc))
            except Exception as exc:  # noqa: BLE001 - the sidecar must never propagate a failure
                item.update(status='FAILED', error=type(exc).__name__)
            item['wall_seconds'] = time.monotonic() - started
            record['commands'].append(item)
            if len(raw) > 2 and str(raw[1]).endswith('/sp_manifest.py') and raw[2] == 'entry':
                prepared = item['status'] == 'DONE'
        statuses = {c['status'] for c in record['commands']}
        record['status'] = 'FAILED' if 'FAILED' in statuses else 'MISSING-INPUT' if 'MISSING-INPUT' in statuses else 'DONE'
    except StopIteration:
        record['status'] = 'FAILED'; record['error'] = 'unknown SP task'
    except Exception as exc:  # noqa: BLE001
        record['status'] = 'FAILED'; record['error'] = type(exc).__name__
    record['ended'] = time.time()
    try:
        (state / (task_id + '.json')).write_text(json.dumps(record, sort_keys=True) + '\n')
    except OSError:
        pass
    return 0


# ---------------------------------------------------------------------------------------------
# SP(a): frozen-cohort length over existing scorer cells (CPU only).

def _is_pre(arm):
    return arm == 'pre' or re.fullmatch(r'x-(?:t2-.+-)?pre', str(arm)) is not None


def length_plan(planned):
    """Comparison families derivable from static entries; the rest stay PENDING, never guessed.

    Within (source, family, seed, panel): an X pre-replacement arm at step 0 is the incoming
    checkpoint for every arm; otherwise each arm's step-0 cell is its own incoming (for M3 that is
    the alias of the earlier stage final). Runtime rescoring has no fixed comparison plan here.
    """
    groups, pending = {}, []
    for e in planned:
        if e['resolution'] != 'static':
            if e['source'] in ('message2', 'message3'):
                pending.append(e['planned_id'])
            continue
        if e.get('panel') in NOT_APPLICABLE:
            continue
        groups.setdefault((e['source'], e.get('family'), e.get('seed'), e.get('panel')), []).append(e)
    families = []
    for (source, family, seed, panel), members in sorted(groups.items(), key=lambda kv: json.dumps(kv[0], default=str)):
        pre = [m for m in members if _is_pre(m.get('arm')) and m.get('step') == 0]
        for arm in sorted({m['arm'] for m in members if not _is_pre(m['arm'])}):
            incoming = pre or [m for m in members if m['arm'] == arm and m.get('step') == 0]
            incoming = list({_match_key(m): m for m in incoming}.values())
            descendants = sorted([m for m in members if m['arm'] == arm and m.get('step') not in (0, None)], key=lambda m: m['step'])
            if not descendants:
                continue
            if len(incoming) != 1:
                if str(family).startswith('return1.'):
                    families.append(dict(family='%s.%s' % (family, arm), source=source, seed=seed, panel=panel,
                                         conditional=False, incoming=(source, family, arm, seed, 0, panel),
                                         descendants=[_match_key(m) for m in descendants]))
                continue
            families.append(dict(family='%s.%s' % (family, arm), source=source, seed=seed, panel=panel,
                                 conditional=any(m.get('conditional') for m in descendants),
                                 incoming=_match_key(incoming[0]), descendants=[_match_key(m) for m in descendants]))
    return families, sorted(set(pending))


def _cell(c, panel, sub):
    from kit import sp_score
    return sp_score.cell_inputs(dict(raw=c.get('raw'), scores=c['scores'], panel=panel, panel_filter=sub))


def _cell_name(c):
    return Path(c['cell']).name


def base_cohorts(cells, sp_root):
    """Descriptive base-correct cohorts per (source, family, seed, panel), separate from stage cohorts."""
    from kit import sp_score
    out = []
    bases = [c for c in cells if c.get('arm') == 'base' and c.get('step') == 0 and c.get('scores')
             and c.get('panel') not in NOT_APPLICABLE]
    for base in sorted(bases, key=lambda c: (str(c['source']), str(c['family']), str(c['seed']), str(c['panel']))):
        for sub in (GENERAL_SUBPANELS if base['panel'] == 'general-v1' else (None,)):
            name = 'base.%s.%s.%s' % (base['source'], base.get('family'), sub or base['panel'])
            _, scores, _ = _cell(base, base['panel'], sub)
            if not scores:
                out.append(dict(family=name, seed=base['seed'], kind='base', label='DESCRIPTIVE',
                                status='MISSING-INCOMING', cells=[]))
                continue
            weights, _ = weights_key(base['model'])
            cohort = sp_score.freeze_correct_cohort(
                Path(sp_root) / 'cohorts' / ('%s-s%s.json' % (name, base['seed'])), family=name, seed=base['seed'],
                panel=sub or base['panel'], incoming=dict(checkpoint_id=_cell_name(base), weights_sha256=weights),
                score_rows=scores, scores_sha256=pd_contract.sha256_file(base['scores']),
                raw_sha256=pd_contract.sha256_file(base['raw']) if base.get('raw') else None, kind='base',
                panel_sha256=base.get('panel_sha256'), column=sp_score.correct_column(base['panel']))
            members = []
            for c in sorted(cells, key=lambda c: c['cell']):
                if (c['source'], c.get('family'), c['seed'], c['panel']) != \
                        (base['source'], base.get('family'), base['seed'], base['panel']) or not c.get('scores'):
                    continue
                raw, _, correct = _cell(c, c['panel'], sub)
                members.append(dict(cell=c['cell'], arm=c['arm'], step=c['step'],
                                    same_gpu_as_base=gpu_key(c['gpu_identity']) is not None and gpu_key(c['gpu_identity']) == gpu_key(base['gpu_identity']),
                                    metrics=sp_score.length_metrics(raw, correct, cohort)))
            out.append(dict(family=name, seed=base['seed'], kind='base', label='DESCRIPTIVE', cells=members))
    return out


def run_length(inv, sp_root, inputs):
    """Freeze incoming cohorts, then full-panel and cohort length for every family cell."""
    from kit import sp_score
    verify_seal(inv)
    discovered = [c for root in inputs for c in [*discover(root), *retained_return1(root, inv['planned'])]]
    lookup = Cells(discovered, inv['planned'])
    families, pending = length_plan(inv['planned'])
    out = dict(label='DESCRIPTIVE', gpu_hours=0.0, pending_plans=pending, families=[],
               not_applicable=[dict(key=list(_match_key(p)), status='N/A', reason='likelihood has no generated response')
                               for p in inv['planned'] if p.get('panel') in NOT_APPLICABLE],
               unverified_cells=sorted(c['cell'] for c in discovered if c['status'] != 'OK'))
    first = lambda key: (lookup.get(key) or [None])[0]
    for fam in families:
        for sub in (GENERAL_SUBPANELS if fam['panel'] == 'general-v1' else (None,)):
            name = fam['family'] + ('.' + sub if sub else '')
            cohort_name = '%s.%s.s%s.%s' % (fam['source'], fam['family'], fam['seed'], sub or fam['panel'])
            record = dict(family=name, source=fam['source'], seed=fam['seed'], panel=sub or fam['panel'], cells=[], status='COMPLETE')
            inc = first(fam['incoming'])
            if inc is None or not inc.get('scores'):
                record['status'] = 'NOT-RUN' if fam['conditional'] else 'MISSING-INCOMING'
                record['cells'] = [dict(key=list(key), status=record['status']) for key in [fam['incoming'], *fam['descendants']]]
                out['families'].append(record); continue
            weights, _ = weights_key(inc['model'])
            _, scores, _ = _cell(inc, fam['panel'], sub)
            if not scores:
                record['status'] = 'MISSING-INCOMING'
                record['cells'] = [dict(key=list(key), status=record['status']) for key in [fam['incoming'], *fam['descendants']]]
                out['families'].append(record)
                continue
            cohort = sp_score.freeze_correct_cohort(
                Path(sp_root) / 'cohorts' / (cohort_name + '.json'), family=name, seed=fam['seed'],
                panel=sub or fam['panel'], incoming=dict(checkpoint_id=_cell_name(inc), weights_sha256=weights),
                score_rows=scores, scores_sha256=pd_contract.sha256_file(inc['scores']),
                raw_sha256=pd_contract.sha256_file(inc['raw']) if inc.get('raw') else None,
                panel_sha256=inc.get('panel_sha256'), column=sp_score.correct_column(fam['panel']))
            for key in [tuple(fam['incoming'])] + [tuple(d) for d in fam['descendants']]:
                c = first(key)
                if c is None or not c.get('scores'):
                    record['cells'].append(dict(key=list(key), status='MISSING')); record['status'] = 'INCOMPLETE'; continue
                raw, _, correct = _cell(c, fam['panel'], sub)
                metrics = sp_score.length_metrics(raw, correct, cohort)
                if metrics['status'] == 'INCOMPLETE':
                    record['status'] = 'INCOMPLETE'
                same_gpu = gpu_key(c['gpu_identity']) is not None and gpu_key(c['gpu_identity']) == gpu_key(inc['gpu_identity'])
                record['cells'].append(dict(key=list(key), cell=c['cell'], status='SCORED', same_gpu_as_incoming=same_gpu,
                                            metrics=metrics))
            out['families'].append(record)
    out['base_cohorts'] = base_cohorts(discovered, sp_root)
    target = Path(sp_root) / 'length' / 'length.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(out, sort_keys=True, indent=1) + '\n')
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('action', choices=('inventory', 'sidecar', 'validate', 'price', 'run', 'entry', 'length', 'verify-pools', 'hardware'))
    p.add_argument('--launch', type=Path, action='append', default=[]); p.add_argument('--pd', type=Path)
    p.add_argument('--tier', type=Path, action='append', default=[])
    p.add_argument('--identities', type=Path, action='append', default=[])
    p.add_argument('--inventory', type=Path); p.add_argument('--sidecar', type=Path)
    p.add_argument('--entry-id'); p.add_argument('--task')
    p.add_argument('--kit', type=Path); p.add_argument('--sp', type=Path)
    p.add_argument('--input', type=Path, action='append', default=[])
    p.add_argument('--superset', type=Path); p.add_argument('--pools-dir', type=Path)
    p.add_argument('--pools-spec', type=Path); p.add_argument('--frozen', type=Path)
    p.add_argument('--return1', type=Path, help='owner return-1 binding or reviewed decisions')
    p.add_argument('--hardware', type=Path)
    p.add_argument('--parent-hardware', type=Path, action='append', default=[])
    p.add_argument('--score-gpu', default='0')
    p.add_argument('--models', type=Path, help='local JSON {entry_id: model dir}; never returned')
    p.add_argument('--owner-sources', type=Path, help='local native owner export source map; never returned')
    p.add_argument('--gpu-identities', type=Path, help='local JSON {assignment: {uuid, environment}}')
    p.add_argument('--out', type=Path)
    a = p.parse_args(argv)
    load = lambda path: json.loads(Path(path).read_text())
    if a.action == 'run':
        maps = dict(models=load(a.models) if a.models and a.models.exists() else {},
                    hardware=load(a.hardware) if a.hardware and a.hardware.exists() else None,
                    parent_inventories=[load(p) for p in a.parent_hardware])
        values = {k: v for k, v in {'{kit}': a.kit, '{sp}': a.sp, '{score_gpu}': a.score_gpu,
                                  '{models_file}': a.models,
                                  '{owner_sources_file}': a.owner_sources or a.sp / 'owner-exports.local.json'}.items() if v is not None}
        return run(load(a.sidecar), a.task, values, maps, a.input)
    if a.action == 'hardware':
        freeze_hardware(a.out, load(a.gpu_identities), [load(p) for p in a.parent_hardware])
        return 0
    launches = [load(path) for path in a.launch]
    if a.action == 'inventory':
        if a.return1:
            binding = load(a.return1)
            binding = binding.get('rescoring', binding)
        else:
            # The core producer uses these pinned decisions when constructing its launch.
            from kit.m2_return1 import decisions
            binding = decisions()['rescoring'] if any('return1.' in str(t.get('family')) for l in launches for t in l['tasks']) else None
        doc = inventory(launches, load(a.pd) if a.pd else None, [load(t) for t in a.tier],
                        [c for root in a.identities for c in discover(root)], return1=binding)
        freeze_inventory(a.out, doc)
        print('SP inventory:', len(doc['checkpoints']), 'checkpoints,', len(doc['sp_entries']), 'SP(b) entries,',
              len(doc['missing']), 'missing,', len(doc['not_run']), 'not run; SP(b) GPU-h range', doc['price']['sp_b_gpu_hours'])
        return 0
    if a.action == 'sidecar':
        doc = build_sidecar(launches, load(a.inventory))
        validate_sidecar(doc, launches)
        a.out.write_text(json.dumps(doc, sort_keys=True, indent=1) + '\n')
        print(len(doc['tasks']), 'SP sidecar tasks; none submitted')
        return 0
    if a.action == 'validate':
        print(len(validate_sidecar(load(a.sidecar), launches)), 'SP sidecar tasks valid and isolated')
        return 0
    if a.action == 'price':
        print(json.dumps(load(a.inventory)['price'], indent=1))
        return 0
    if a.action == 'entry':
        inv = load(a.inventory); verify_seal(inv)
        entry = next(e for e in inv['sp_entries'] if e['entry_id'] == a.entry_id)
        entry = bind_runtime_entry(entry, inv, a.input, load(a.models).get(a.entry_id))
        freeze_inventory(a.out, seal(entry))
        return 0
    if a.action == 'verify-pools':
        from kit import sp_prepare
        result = sp_prepare.recheck(a.frozen or sp_prepare.RESOURCES, a.pools_spec)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(result, sort_keys=True, indent=1) + '\n')
        print('SP pool verification:', result['status'])
        return 0
    run_length(load(a.inventory), a.sp, a.input)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
