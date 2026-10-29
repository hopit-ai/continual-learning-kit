#!/usr/bin/env python3
"""SP unseen-wording panel: validate authored pairs, build fixtures, audit leakage, freeze.

    python kit/sp_prepare.py audit  --pools POOLS.json --out DIR [--source DIR]
    python kit/sp_prepare.py freeze --pools POOLS.json --out DIR [--source DIR] [--inspection FILE]
    python kit/sp_prepare.py recheck --frozen DIR --pools POOLS.json --out FILE

CPU only, standard library (rapidfuzz is used when installed, for speed only). The 200 request
pairs are authored under kit/resources/sp/source/ (read-only here; schema in
kit/resources/sp/SCHEMA.md). No model is run. Supplied response fields are used only for exclusion checks.

POOLS.json lists the comparison pools as {"pools": [{"name", "role", "path", "replay"}]}. Paths are
local to the machine running the audit and are never written to any output; outputs carry each
pool's name, role, row count and SHA256 only.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
from heapq import nsmallest
import importlib.util
import gzip
import json
import re
import sys
import unicodedata
from pathlib import Path

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kit.pd_contract import digest, sha256_file, require

KIT = Path(__file__).resolve().parent
RESOURCES = KIT / 'resources' / 'sp'
SOURCE = RESOURCES / 'source'
CHECKERS = RESOURCES / 'checkers.py'
SCHEMA = 'sp-panel-freeze-v2'
AUDIT_SCHEMA = 'sp-leakage-audit-v2'

N_REQUESTS = 200
PER_FAMILY = 20          # requests per primary family
PAIRED_PER_FAMILY = 10   # of which carry a compatible second constraint
WORDINGS = ('canonical', 'paraphrase')
# PD's registered duplicate thresholds (draft PD section): token-5-gram Jaccard >= 0.8 or
# normalised edit similarity >= 0.9, after Unicode/whitespace/case normalisation.
SHINGLE = 5
JACCARD = 0.8
EDIT = 0.9
NEAREST = 5
# Every role the registration names; a missing role leaves the panel PENDING, never frozen.
REQUIRED_ROLES = ('generic-candidates', 'generic-accepted', 'broad', 'earlier', 'k', 'cr', 'ancestry', 'panels')
REPLAY_ROLES = ('generic-candidates', 'generic-accepted', 'broad', 'earlier', 'k', 'cr', 'ancestry')
TEXT_FIELDS = ('content', 'prompt', 'response', 'completion', 'target', 'answer', 'output', 'chosen', 'question')

NUMBER_WORDS = {
    0: 'zero', 1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five', 6: 'six', 7: 'seven', 8: 'eight',
    9: 'nine', 10: 'ten', 11: 'eleven', 12: 'twelve', 13: 'thirteen', 14: 'fourteen', 15: 'fifteen',
    16: 'sixteen', 17: 'seventeen', 18: 'eighteen', 19: 'nineteen', 20: 'twenty', 30: 'thirty',
    40: 'forty', 50: 'fifty', 60: 'sixty', 70: 'seventy', 80: 'eighty', 90: 'ninety', 100: 'hundred',
}
_NUMBER = re.compile(r'\d+(?:[.,]\d+)*|\b(?:%s)\b' % '|'.join(sorted(NUMBER_WORDS.values(), key=len, reverse=True)), re.I)
_QUOTED = re.compile(r'"[^"\n]*"|“[^”\n]*”|\'[^\'\n]+\'|‘[^’\n]+’')
FILLER = ('also', 'then', 'still', 'again', 'here', 'quite')


# ---------------------------------------------------------------------------------------------
# Checkers, normalisation and similarity.

def load_checkers(path=CHECKERS, expected_sha256=None):
    if expected_sha256 is not None:
        require(sha256_file(path) == expected_sha256, 'SP checker bytes differ from the frozen hash')
    spec = importlib.util.spec_from_file_location('sp_checkers_' + sha256_file(path)[:12], path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_templates():
    spec = importlib.util.spec_from_file_location('sp_canonical_templates', RESOURCES / 'templates.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    require(module.inventory() == json.loads((RESOURCES / 'templates.json').read_text()), 'template provenance inventory changed')
    return module


def normal(text):
    """Unicode NFKC, case-fold, collapse whitespace (the PD/B duplicate normalisation)."""
    if isinstance(text, list):
        text = '\n'.join(m['content'] for m in text if isinstance(m, dict) and isinstance(m.get('content'), str))
    return ' '.join(unicodedata.normalize('NFKC', text).casefold().split())


def prepared(text):
    n = normal(text)
    w = n.split()
    return n, frozenset(tuple(w[i:i + SHINGLE]) for i in range(len(w) - SHINGLE + 1))


try:  # Speed only; the pure-Python fallback computes the same distance.
    from rapidfuzz.distance import Levenshtein as _RAPID
except ImportError:  # pragma: no cover - depends on the environment
    _RAPID = None


def distance(a, b, cutoff=None):
    """Levenshtein distance; with cutoff, any value > cutoff may be returned as cutoff + 1."""
    if _RAPID is not None:
        return _RAPID.distance(a, b, score_cutoff=cutoff)
    if len(a) < len(b):
        a, b = b, a
    n, m = len(a), len(b)
    if cutoff is None:
        previous = list(range(m + 1))
        for i, ca in enumerate(a, 1):
            current = [i]
            for j, cb in enumerate(b, 1):
                current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
            previous = current
        return previous[-1]
    # Ukkonen band: cells with |i - j| > cutoff already cost more than cutoff.
    big = cutoff + 1
    if n - m > cutoff:
        return big
    previous = [j if j <= cutoff else big for j in range(m + 1)]
    for i in range(1, n + 1):
        current = [big] * (m + 1)
        if i <= cutoff:
            current[0] = i
        best = current[0]
        ca = a[i - 1]
        for j in range(max(1, i - cutoff), min(m, i + cutoff) + 1):
            v = previous[j - 1] + (ca != b[j - 1])
            if previous[j] + 1 < v:
                v = previous[j] + 1
            if current[j - 1] + 1 < v:
                v = current[j - 1] + 1
            current[j] = v if v < big else big
            if current[j] < best:
                best = current[j]
        if best > cutoff:
            return big
        previous = current
    return previous[m] if previous[m] <= cutoff else big


def edit_similarity(a, b):
    longest = max(len(a), len(b))
    return 1.0 if longest == 0 else 1.0 - distance(a, b) / longest


def near(pa, pb):
    """Which registered rule makes two prepared texts duplicates, or None."""
    (a, x), (b, y) = pa, pb
    if a == b:
        return 'exact'
    if x and y and min(len(x), len(y)) >= JACCARD * max(len(x), len(y)) and len(x & y) / len(x | y) >= JACCARD:
        return 'jaccard'
    longest = max(len(a), len(b))
    limit = int(longest * (1 - EDIT) + 1e-9)   # similarity >= 0.9  <=>  distance <= 0.1 * longest
    if abs(len(a) - len(b)) > limit:
        return None
    return 'edit' if distance(a, b, cutoff=limit) <= limit else None


def mask(text, values=()):
    """Strip parameter values from an instruction clause: known string values, quotes, numbers.

    Known values, quoted strings and numeric values share the same <p> marker,
    including numeric words. Replay-family patterns also mask unquoted values.
    """
    out = unicodedata.normalize('NFKC', text)
    for value in sorted({str(v) for v in values if str(v).strip()}, key=len, reverse=True):
        out = re.sub(r'(?<!\w)' + re.escape(value) + r'(?!\w)', ' <p> ', out, flags=re.I)
    for pattern in (
        r'(?<=Include keywords ).+?(?= in your response)',
        r'(?<=Do not include keywords ).+?(?= in the response)',
        r'(?<=the word ).+?(?= should appear)',
        r'(?<=this exact phrase ).+?(?= No other words should follow this phrase)',
    ):
        out = re.sub(pattern, ' <p> ', out, flags=re.I)
    out = _QUOTED.sub(' <p> ', out)
    out = _NUMBER.sub(' <p> ', out)
    return normal(out)


def constraint_values(constraints):
    values = []
    for c in constraints:
        values += [v for v in c['args'].values()]
    return values


# ---------------------------------------------------------------------------------------------
# Source validation.

def read_jsonl(path):
    rows = []
    with open(path, encoding='utf-8') as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except ValueError as exc:
                    raise ValueError('%s line %d: %s' % (Path(path).name, number, exc)) from exc
    return rows


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with open(tmp, 'w', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + '\n')
    tmp.replace(path)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, sort_keys=True, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    tmp.replace(path)


# The authored source (source/README.md) names this family exact_word_frequency.
FAMILY_ALIASES = {'exact_word_frequency': 'word_frequency'}
REVIEWS = RESOURCES / 'reviews.jsonl'


def adapt(row, reviews=()):
    """Map one authored row (source/README.md schema) to the internal request form.

    Authored keys: id, content, constraints[{family, parameters}], canonical_instruction,
    paraphrase_instruction, witness. Prompts render as content + blank line + instruction, exactly
    as the README specifies; content is never reworded. Rows already in internal form pass through.
    Reviews are never part of the source: they come from the independent reviewers' file.
    """
    if 'canonical_instruction' not in row:
        return dict(row, reviews=list(row.get('reviews', [])) + list(reviews))
    constraints = [dict(family=FAMILY_ALIASES.get(c.get('family'), c.get('family')), args=dict(c.get('parameters') or {}))
                   for c in row.get('constraints') or []]
    wording = {w: dict(clause=row[w + '_instruction'], prompt=row['content'] + '\n\n' + row[w + '_instruction'])
               for w in WORDINGS}
    return dict(request_id=row.get('id'), primary_family=constraints[0]['family'] if constraints else None,
                content=row.get('content'), constraints=constraints, witness=row.get('witness'),
                mutations=[dict(m, target=FAMILY_ALIASES.get(m.get('target'), m.get('target'))) for m in row.get('mutations', [])],
                reviews=list(reviews), **wording)


def load_reviews(path=REVIEWS):
    """Independent pair reviews: JSONL {request_id, reviewer, semantic_equivalence, clarity, satisfiable}."""
    out = {}
    if path is not None and Path(path).exists():
        for r in read_jsonl(path):
            out.setdefault(r.get('request_id', r.get('id')), []).append(r)
    return out


def load_source(source=SOURCE, reviews=REVIEWS):
    source = Path(source)
    path = source / 'requests.jsonl'
    if not path.exists():
        raise FileNotFoundError('PENDING: authored SP request pairs not present (%s)' % path.name)
    attestations = load_reviews(reviews)
    rows = [adapt(r, attestations.get(r.get('id', r.get('request_id')), ())) for r in read_jsonl(path)]
    files = {p.name: sha256_file(p) for p in sorted(source.iterdir()) if p.is_file()}
    if reviews is not None and Path(reviews).exists():
        files['../' + Path(reviews).name] = sha256_file(reviews)
    return rows, files


def _number_in(clause, value):
    text = normal(clause)
    if re.search(r'(?<![\d.,])%d(?![\d]|[.,]\d)' % value, text):
        return True
    word = NUMBER_WORDS.get(value)
    return word is not None and re.search(r'\b%s\b' % word, text) is not None


def validate_request(row, ck, require_reviews=False):
    """All problems with one authored request, as strings (empty list = valid)."""
    errors = []
    rid = row.get('request_id')
    def bad(message):
        errors.append('%s: %s' % (rid, message))
    if not isinstance(rid, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,63}', rid):
        return ['%r: request_id must be a short portable identifier' % (rid,)]
    constraints = row.get('constraints')
    try:
        ck.validate_constraints(constraints)
    except (ValueError, TypeError) as exc:
        bad(str(exc))
        return errors
    if row.get('primary_family') != constraints[0]['family']:
        bad('primary_family must equal the first constraint family')
    content = row.get('content')
    if not isinstance(content, str) or not content.strip():
        bad('missing content')
        content = ''
    clauses = {}
    for wording in WORDINGS:
        part = row.get(wording)
        if not isinstance(part, dict) or not all(isinstance(part.get(k), str) and part[k].strip() for k in ('clause', 'prompt')):
            bad(wording + ' needs clause and prompt text')
            continue
        clauses[wording] = part['clause']
        if content and content not in part['prompt']:
            bad(wording + ' prompt must contain the content verbatim (content preserved across wordings)')
        if part['clause'] not in part['prompt']:
            bad(wording + ' prompt must contain its clause verbatim')
        for c in constraints:
            for key, value in c['args'].items():
                if isinstance(value, str) and value not in part['clause']:
                    bad('%s clause omits argument %s=%r' % (wording, key, value))
                if isinstance(value, int) and not _number_in(part['clause'], value):
                    bad('%s clause omits argument %s=%r' % (wording, key, value))
    if len(clauses) == 2:
        if normal(clauses['canonical']) == normal(clauses['paraphrase']):
            bad('paraphrase clause must reword the canonical clause')
        if normal(row['canonical']['prompt']) == normal(row['paraphrase']['prompt']):
            bad('paraphrase prompt must differ from the canonical prompt')
    if row.get('schema') != 'sp-source-v2':
        bad('source schema must be sp-source-v2; use the documented adapter')
    for wording in WORDINGS:
        part = row.get(wording, {})
        if part.get('prompt') != content + '\n\n' + part.get('clause', ''):
            bad(wording + ' prompt must equal content + two newlines + clause exactly')
        if not isinstance(part.get('clauses'), list) or not all(isinstance(c, str) and c for c in part['clauses']):
            bad(wording + ' needs atomic clauses')
        elif part.get('clause') != ('\n' if wording == 'canonical' else ' ').join(part['clauses']):
            bad(wording + ' atomic clauses do not reconstruct the full clause')
    expected, bindings = load_templates().render(constraints)
    if row.get('canonical', {}).get('clause') != expected or row.get('canonical', {}).get('template_bindings') != bindings:
        bad('canonical is not an exact provenance-bound replay template instantiation')
    if require_reviews:
        bad('external hash-bound two-reviewer receipts required; inline approvals are not accepted')
    if not isinstance(row.get('witness'), str) or not row['witness'].strip():
        bad('missing passing witness')
    for m in row.get('mutations', []):
        if not (isinstance(m, dict) and m.get('target') in [c['family'] for c in constraints] and isinstance(m.get('text'), str)):
            bad('authored mutation needs a target family of this request and text')
    return errors


def validate_source(rows, ck):
    errors = []
    for row in rows:
        errors += validate_request(row, ck)
    ids = [r.get('request_id') for r in rows]
    if len(ids) != len(set(ids)):
        errors.append('duplicate request_id')
    if len(rows) != N_REQUESTS:
        errors.append('expected %d requests, found %d' % (N_REQUESTS, len(rows)))
    for family in ck.FAMILIES:
        mine = [r for r in rows if r.get('primary_family') == family]
        paired = [r for r in mine if isinstance(r.get('constraints'), list) and len(r['constraints']) == 2]
        if len(mine) != PER_FAMILY or len(paired) != PAIRED_PER_FAMILY:
            errors.append('%s: need %d requests (%d with a second constraint), found %d (%d)'
                          % (family, PER_FAMILY, PAIRED_PER_FAMILY, len(mine), len(paired)))
    return errors


# ---------------------------------------------------------------------------------------------
# Witness and failing-mutation fixtures.

def _cased(word, like):
    return word[:1].upper() + word[1:] if like[:1].isupper() else word


def mutation_candidates(text, constraint, constraints, ck):
    """Deterministic candidate edits aimed at failing exactly `constraint`; caller filters."""
    w = text
    family, args = constraint['family'], constraint['args']
    reserved = {ck._fold(c['args']['word']) for c in constraints if 'word' in c['args']}
    fillers = [f for f in FILLER if ck._fold(f) not in reserved]
    spans = [(s, e) for s, e in ck.word_spans(w) if w[s:e].isalpha() and ck._fold(w[s:e]) not in reserved]
    if family == 'required_word':
        target = ck._fold(args['word'])
        hits = [(s, e) for s, e in ck.word_spans(w) if ck._fold(w[s:e]) == target]
        for filler in fillers:
            out = w
            for s, e in reversed(hits):
                out = out[:s] + _cased(filler, out[s:e]) + out[e:]
            yield out
    elif family in ('forbidden_word', 'word_frequency'):
        for s, e in spans:
            yield w[:s] + _cased(args['word'], w[s:e]) + w[e:]
    elif family == 'word_count_interval':
        n = len(w.split())
        extra = args['max'] - n + 1
        if extra > 0:
            for s, e in spans:
                for filler in fillers[:2]:
                    yield w[:e] + (' ' + filler) * extra + w[e:]
        need = n - args['min'] + 1
        if 0 < need <= len(spans):
            for offset in range(0, len(spans) - need + 1):
                out = w
                for s, e in reversed(spans[offset:offset + need]):
                    tail = 1 if out[e:e + 1] == ' ' else 0
                    out = out[:s] + out[e + tail:]
                yield out
    elif family == 'sentence_count':
        for s, e in spans:
            if w[e:e + 1] == ' ':
                yield w[:e] + '.' + w[e:]
        for m in re.finditer(r'[.!?]+(?=[ \t])', w):
            yield w[:m.start()] + ',' + w[m.end():]
    elif family == 'paragraph_count':
        for m in re.finditer(re.escape('* * *'), w):
            yield w[:m.start()] + ' ' + w[m.end():]
        yield w + '\n* * *\nadditional paragraph'
        for m in re.finditer(r'\n[ \t\r\f\v]*\n(?:[ \t\r\f\v]*\n)*', w):
            yield w[:m.start()] + '\n' + w[m.end():]
            yield w[:m.start()] + ' ' + w[m.end():]
        for m in re.finditer(r'[.!?](?= (?=\S))', w):
            yield w[:m.end()] + '\n\n' + w[m.end() + 1:]
        for m in re.finditer(r'(?<!\n)\n(?![ \t\r\f\v]*\n)', w.strip()):
            body = w.strip()
            yield body[:m.start()] + '\n\n' + body[m.end():]
    elif family == 'bullet_count':
        for m in re.finditer(r'^([ \t]*)[-*•][ \t]+(?=\S)', w, re.M):
            yield w[:m.start()] + m.group(1) + w[m.end():]
        for m in re.finditer(r'^[ \t]*[-*•][ \t]+\S.*$', w, re.M):
            yield w[:m.end()] + '\n' + m.group(0) + w[m.end():]
    elif family == 'json_only':
        body = w.rstrip()
        if body:
            yield body[:-1] + w[len(body):]
            yield body[:-1] + ',' + body[-1:] + w[len(body):]
    elif family == 'lowercase_only':
        guard = len(w.rstrip()) - max([len(c['args'].get('suffix', '')) for c in constraints] + [0])
        positions = [i for i, ch in enumerate(w[:max(guard, 0)]) if ch.isalpha() and ch.upper() != ch][:80]
        for i in positions:
            yield w[:i] + w[i].upper() + w[i + 1:]
    elif family == 'required_suffix':
        body = w.rstrip()
        if body:
            yield body[:-1] + w[len(body):]
            replacement = '!' if body[-1] != '!' else '?'
            yield body[:-1] + replacement + w[len(body):]
            yield body + ' ' + (fillers[0] if fillers else 'x')


def fixtures_for(row, ck):
    """Witness plus at least one single-constraint failing mutation per constraint."""
    constraints = row['constraints']
    witness = row['witness']
    verdicts = ck.check_all(witness, constraints)
    out = [dict(request_id=row['request_id'], kind='witness', target=None, origin='authored',
                text=witness, expected=[True] * len(constraints), observed=verdicts)]
    errors = [] if all(verdicts) else ['%s: witness fails %s' % (row['request_id'],
              [c['family'] for c, v in zip(constraints, verdicts) if not v])]
    for index, constraint in enumerate(constraints):
        expected = [i != index for i in range(len(constraints))]
        authored = [m['text'] for m in row.get('mutations', []) if m['target'] == constraint['family']]
        chosen = []
        for text in authored:
            observed = ck.check_all(text, constraints)
            if observed != expected:
                errors.append('%s: authored %s mutation is not single-constraint failing (%s)'
                              % (row['request_id'], constraint['family'], observed))
            chosen.append(dict(text=text, origin='authored', observed=observed))
        if not authored:
            for text in mutation_candidates(witness, constraint, constraints, ck):
                if text != witness and ck.check_all(text, constraints) == expected:
                    chosen.append(dict(text=text, origin='generated', observed=expected))
                    break
        if not chosen:
            errors.append('%s: no single-constraint failing mutation for %s; author one'
                          % (row['request_id'], constraint['family']))
        for item in chosen:
            out.append(dict(request_id=row['request_id'], kind='mutation', target=constraint['family'],
                            origin=item['origin'], text=item['text'], expected=expected, observed=item['observed']))
    return out, errors


def check_fixtures(fixtures, requests, ck):
    """Re-run every fixture through the checkers; any disagreement is a checker failure."""
    lookup = {r['request_id']: r for r in requests}
    failures = []
    for f in fixtures:
        observed = ck.check_all(f['text'], lookup[f['request_id']]['constraints'])
        if observed != f['expected']:
            failures.append(dict(request_id=f['request_id'], kind=f['kind'], target=f['target'], observed=observed))
    return failures


# ---------------------------------------------------------------------------------------------
# Duplicate and leakage audit.

def row_texts(row):
    """Every text a pool row could leak: prompt (string or chat), responses and message contents."""
    texts = []
    for field in TEXT_FIELDS:
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            texts.append((field, value))
        elif isinstance(value, list):
            joined = '\n'.join(m['content'] for m in value if isinstance(m, dict) and isinstance(m.get('content'), str))
            if joined.strip():
                texts.append((field, joined))
    for message in row.get('messages', []) if isinstance(row.get('messages'), list) else []:
        if isinstance(message, dict) and isinstance(message.get('content'), str) and message['content'].strip():
            texts.append(('messages.' + str(message.get('role')), message['content']))
    return texts


def clauses_of(text, ck):
    """Candidate instruction clauses of a pool prompt: its lines and their sentences."""
    out = []
    for line in text.splitlines():
        if not line.strip():
            continue
        out.append(line.strip())
        parts = ck.sentences(line)
        if len(parts) > 1:
            out += [p.strip() for p in parts]
    return list(dict.fromkeys(out))


def prompt_text(row):
    value = row.get('prompt', row.get('question'))
    if isinstance(value, str) and value.strip(): return value
    if isinstance(value, list):
        return '\n'.join(m['content'] for m in value if isinstance(m, dict) and m.get('role') == 'user' and isinstance(m.get('content'), str))
    messages = row.get('messages', [])
    return '\n'.join(m['content'] for m in messages if isinstance(m, dict) and m.get('role') == 'user' and isinstance(m.get('content'), str))


def candidate_members(pool):
    return sorted({digest(normal(prompt_text(row))) for row in pool['rows'] if prompt_text(row)})


def load_pools(spec_path, returned=False, available_scope=False):
    spec = json.loads(Path(spec_path).read_text())
    require(spec.get('schema') == 'sp-pools-v2', 'pool specification must be sp-pools-v2')
    require(spec.get('complete_inventory') is True or (available_scope and not returned),
            'PENDING: complete candidate/reference inventory not attested')
    if not returned:
        require(spec.get('frozen_before_scoring') is True, 'PENDING: candidate supersets must be frozen before scoring')
    pools = []
    base = Path(spec_path).resolve().parent
    for entry in spec['pools']:
        role = entry.get('role')
        require(role in REQUIRED_ROLES, 'unknown pool role')
        require(isinstance(entry.get('name'), str) and re.fullmatch(r'[A-Za-z0-9_.-]+', entry['name']), 'pool needs a portable name')
        path = Path(entry['path'])
        path = path if path.is_absolute() else base / path
        actual = sha256_file(path)
        require(actual == entry.get('sha256'), 'pool bytes do not match declared hash: ' + entry['name'])
        rows = read_jsonl(path)
        require(bool(rows), 'PENDING: empty exclusion pool ' + entry['name'])
        scope = entry.get('scope')
        expected_scope = 'candidate-superset' if role in ('generic-candidates', 'broad', 'earlier', 'k', 'cr', 'ancestry') else 'frozen-reference'
        if not returned:
            require(scope == expected_scope, 'PENDING: wrong exclusion scope for ' + entry['name'])
        if scope == 'candidate-superset' or returned:
            require(all(prompt_text(row) for row in rows), 'candidate/return rows require user prompts')
        pools.append(dict(name=entry['name'], role=role, replay=role in REPLAY_ROLES,
                          sha256=actual, rows=rows, scope=scope,
                          candidate_superset_name=entry.get('candidate_superset_name'),
                          candidate_superset_sha256=entry.get('candidate_superset_sha256')))
    require(len({p['name'] for p in pools}) == len(pools), 'duplicate pool name')
    require(_RAPID is not None or sum(len(p['rows']) for p in pools) <= 5000,
            'rapidfuzz is required to audit full-size pools')
    return pools


def pool_identities(pools):
    return [dict(name=p['name'], role=p['role'], replay=p['replay'], sha256=p['sha256'], rows=len(p['rows']),
                 scope=p.get('scope'), candidate_members=candidate_members(p)) for p in pools]


def _prompts(requests):
    for r in requests:
        for wording in WORDINGS:
            yield r, wording, r[wording]['prompt']


def audit(requests, pools, ck):
    """Frozen input exclusions only; canonical template reuse is intentional.

    Response fields, when supplied by the exclusion custodian, are compared by
    code only and never used to construct/adapt the panel. No model inference.
    """
    hits = []
    content = [(r['request_id'], prepared(r['content'])) for r in requests]
    witnesses = [(r['request_id'], prepared(r['witness'])) for r in requests]
    replay_clauses = []

    @lru_cache(maxsize=65536)
    def content_matches(text):
        target = prepared(text)
        return tuple((rid, rule) for rid, query in content if (rule := near(query, target)))

    @lru_cache(maxsize=4096)
    def witness_matches(text):
        target = prepared(text)
        return tuple((rid, rule) for rid, query in witnesses if (rule := near(query, target)))

    # Repeated legal paragraphs, replay templates and source copies are common.
    # Cache comparisons without dropping any source identity or changing which
    # first matching segment supplies the reported rule.
    for pool in pools:
        for row in pool['rows']:
            rid = str(row.get('id', row.get('request_id', '')))
            for field, text in row_texts(row):
                segments = list(dict.fromkeys([text] + text.split('\n') + text.split('\n\n') + ck.sentences(text)))
                matches = {}
                for segment in segments:
                    if segment.strip():
                        for request_id, rule in content_matches(segment):
                            matches.setdefault(request_id, rule)
                for request_id, _ in content:
                    if request_id in matches:
                        hits.append(dict(kind='content-vs-pool', request_id=request_id, wording='both', pool=pool['name'],
                                         role=pool['role'], row_id=rid, field=field, rule=matches[request_id]))
                for request_id, rule in witness_matches(text):
                    hits.append(dict(kind='witness-vs-pool', request_id=request_id, pool=pool['name'],
                                     role=pool['role'], row_id=rid, field=field, rule=rule))
            if pool['replay']:
                for clause in clauses_of(prompt_text(row), ck):
                    masked = mask(clause)
                    if masked:
                        replay_clauses.append(dict(pool=pool['name'], row_id=rid, clause=clause, masked=masked))
    for i, (ra, pa) in enumerate(content):
        for rb, pb in content[i + 1:]:
            rule = near(pa, pb)
            if rule: hits.append(dict(kind='content-vs-request', request_id=ra, other=rb, rule=rule))
    # Only paraphrase clauses are subject to template leakage checks. Compare
    # each atomic clause separately so a clean secondary cannot hide a copy.
    nearest = {}
    unique_masks = {rc['masked']: prepared(rc['masked']) for rc in replay_clauses}
    for r in requests:
        values = constraint_values(r['constraints'])
        masked_parts = [mask(c, values) for c in r['paraphrase']['clauses']]
        scores = {}
        for index, masked in enumerate(masked_parts):
            pm = prepared(masked)
            comparisons = {text: (near(pm, target), round(edit_similarity(masked, text), 6))
                           for text, target in unique_masks.items()}
            for rc in replay_clauses:
                rule, score = comparisons[rc['masked']]
                if rule:
                    hits.append(dict(kind='clause-vs-replay', request_id=r['request_id'], wording='paraphrase',
                                     clause_index=index, pool=rc['pool'], row_id=rc['row_id'], rule=rule))
                key = (rc['pool'], rc['row_id'], rc['clause'], rc['masked'])
                scores[key] = max(scores.get(key, 0), score)
        ordered = nsmallest(NEAREST, ((-score, *key) for key, score in scores.items()))
        entries = [dict(similarity=-score, pool=pool, row_id=rid, clause=clause, masked=masked)
                   for score, pool, rid, clause, masked in ordered]
        nearest[r['request_id']] = dict(masked_clauses=masked_parts, nearest=entries, nearest_sha256=digest(entries))
    roles = sorted({p['role'] for p in pools})
    missing = [r for r in REQUIRED_ROLES if r not in roles]
    insufficient = [rid for rid, value in nearest.items() if len(value['nearest']) < NEAREST]
    return dict(schema=AUDIT_SCHEMA, thresholds=dict(shingle=SHINGLE, jaccard=JACCARD, edit=EDIT, nearest=NEAREST),
                content_arms=list(WORDINGS), clause_arms=['paraphrase'], canonical_reuse='intentional',
                pools=pool_identities(pools), missing_roles=missing, insufficient_nearest=insufficient, hits=hits,
                rejected_request_ids=sorted({h['request_id'] for h in hits} | {h['other'] for h in hits if 'other' in h}),
                nearest=nearest, replay_clauses=len(replay_clauses),
                status='CLEAN' if not hits and not missing and not insufficient else ('REJECTED' if hits else 'PENDING'))


def check_inspection(inspection_rows, audit_doc, available_scope=False):
    """Every paraphrase's five nearest replay clauses must be manually inspected, bound by hash."""
    errors = []
    rows = {r.get('request_id'): r for r in inspection_rows}
    if len(rows) != len(inspection_rows): errors.append('duplicate nearest-inspection request ID')
    for request_id, value in audit_doc['nearest'].items():
        row = rows.get(request_id)
        if len(value['nearest']) != NEAREST: errors.append(request_id + ': five nearest replay clauses required')
        if row is None:
            errors.append(request_id + ': nearest replay clauses not inspected')
        elif row.get('nearest_sha256') != value['nearest_sha256']:
            errors.append(request_id + ': inspection bound to a different nearest list')
        elif row.get('verdict') != 'distinct' or not row.get('inspector'):
            errors.append(request_id + ': inspection must name an inspector and record verdict "distinct"')
        if row is not None and row.get('inventory_complete') is False and not available_scope:
            errors.append(request_id + ': inspection covers an incomplete inventory')
        if row is not None and 'audit_inventory_sha256' in row and row['audit_inventory_sha256'] != digest(audit_doc['pools']):
            errors.append(request_id + ': inspection bound to a different pool inventory')
    return errors


# ---------------------------------------------------------------------------------------------
# Panel, freeze and recheck.

def panel_rows(requests):
    """The 400 scored prompts. IDs are '<request_id>:<wording>'."""
    return [dict(id='%s:%s' % (r['request_id'], wording), request_id=r['request_id'], wording=wording,
                 primary_family=r['primary_family'], prompt=r[wording]['prompt'], constraints=r['constraints'])
            for r in sorted(requests, key=lambda r: r['request_id']) for wording in WORDINGS]


def exclusion_fingerprints(requests):
    """Normalised-text hashes that replay builders must exclude permanently."""
    import hashlib
    texts = set()
    for r in requests:
        texts.add(normal(r['content'])); texts.add(normal(r['witness']))
        for wording in WORDINGS:
            texts.add(normal(r[wording]['prompt'])); texts.add(normal(r[wording]['clause']))
        for m in r.get('mutations', []):
            texts.add(normal(m['text']))
    return sorted(hashlib.sha256(t.encode()).hexdigest() for t in texts)


def code_hashes():
    return {name: sha256_file(KIT / name) for name in ('sp_prepare.py', 'sp_score.py')}


def resource_hashes():
    return {name: sha256_file(RESOURCES / name) for name in
            ('checkers.py', 'templates.py', 'templates.json', 'compatibility.json', 'checker-edge-fixtures.json')}


def review_binding(requests, fixtures):
    return dict(requests_sha256=digest(requests), fixtures_sha256=digest(fixtures), resources=resource_hashes())


def check_reviews(documents, requests, fixtures):
    errors = []
    if len(documents) != 2: return ['PENDING: exactly two independent reviewer receipts required']
    binding = review_binding(requests, fixtures)
    identities = []
    expected = {r['request_id']: digest(r) for r in requests}
    for doc in documents:
        reviewer = doc.get('reviewer_id')
        if not isinstance(reviewer, str) or not re.fullmatch(r'[A-Za-z0-9_.-]+', reviewer):
            errors.append('PENDING: portable reviewer identity required')
        identities.append(reviewer)
        if doc.get('schema') != 'sp-review-receipt-v2' or doc.get('binding') != binding:
            errors.append('review receipt has stale or mismatched binding')
        if doc.get('independent_of_author_and_other_reviewer') is not True:
            errors.append('PENDING: reviewer independence must be attested')
        items = doc.get('items', [])
        if not isinstance(items, list):
            errors.append('review items must be a list'); continue
        indexed = {item.get('request_id'): item for item in items if isinstance(item, dict)}
        if len(indexed) != len(items) or set(indexed) != set(expected):
            errors.append('review receipt must cover every request exactly once')
        for rid, expected_hash in expected.items():
            item = indexed.get(rid, {})
            if item.get('request_sha256') != expected_hash:
                errors.append(rid + ': review bound to a different request')
            if not all(item.get(k) is True for k in ('semantic_equivalence', 'clarity', 'satisfiable', 'canonical_provenance')):
                errors.append('PENDING: ' + rid + ': independent review incomplete')
    if len(set(str(x) for x in identities)) != 2:
        errors.append('two distinct independent reviewer identities required')
    return errors


def construction(source=SOURCE):
    """Seal author fixtures and issue blank receipts; never declares the panel scored/frozen."""
    require(not (RESOURCES / 'FROZEN.json').exists(), 'cannot reconstruct an already frozen panel')
    ck = load_checkers(); requests, source_files = load_source(source)
    require(not validate_source(requests, ck), 'invalid authored source')
    require(json.loads((RESOURCES / 'compatibility.json').read_text()) == ck.compatibility_matrix(), 'compatibility matrix mismatch')
    fixtures, errors = [], []
    for row in requests:
        f, e = fixtures_for(row, ck); fixtures += f; errors += e
    require(not errors and not check_fixtures(fixtures, requests, ck), 'fixture qualification failed: ' + str(errors))
    edge = json.loads((RESOURCES / 'checker-edge-fixtures.json').read_text())
    require(all(ck.check(e['text'], e['constraint']) == e['expected'] for e in edge), 'edge fixture mismatch')
    write_jsonl(RESOURCES / 'authored-fixtures.jsonl', fixtures)
    binding = review_binding(requests, fixtures)
    for slot in (1, 2):
        receipt = dict(schema='sp-review-receipt-v2', reviewer_id=None,
                       independent_of_author_and_other_reviewer=None, binding=binding,
                       items=[dict(request_id=r['request_id'], request_sha256=digest(r),
                                   semantic_equivalence=None, clarity=None, satisfiable=None,
                                   canonical_provenance=None, notes='') for r in requests])
        receipt_path = RESOURCES / 'reviews' / ('reviewer-%d.template.json' % slot)
        # A template may now hold an independent review. Never overwrite it.
        if not receipt_path.exists():
            write_json(receipt_path, receipt)
    record = dict(schema='sp-construction-v2', status='AWAITING_INDEPENDENT_REVIEWS_AND_EXCLUSIONS',
                  binding=binding, source_files=source_files, code=code_hashes(),
                  source_sha256=sha256_file(Path(source) / 'requests.jsonl'),
                  fixtures_sha256=sha256_file(RESOURCES / 'authored-fixtures.jsonl'),
                  n_requests=len(requests), n_prompts=2 * len(requests), n_fixtures=len(fixtures),
                  raw_json_deviation='One raw JSON value; no code-fence stripping; reject NaN/Infinity.',
                  model_outputs_used=False)
    write_json(RESOURCES / 'CONSTRUCTION.json', record)
    return record


def build(source, pools_spec, inspection=None, reviews=()):
    """Audit and fixture work runs even while independent review remains pending."""
    ck = load_checkers(); requests, source_files = load_source(source)
    blockers = validate_source(requests, ck)
    require(not blockers, 'invalid source: ' + '\n'.join(blockers))
    require(json.loads((RESOURCES / 'compatibility.json').read_text()) == ck.compatibility_matrix(), 'compatibility matrix mismatch')
    fixtures = []; fixture_errors = []
    for r in requests:
        f, e = fixtures_for(r, ck); fixtures += f; fixture_errors += e
    blockers += fixture_errors
    blockers += [str(f) for f in check_fixtures(fixtures, requests, ck)]
    sealed = json.loads((RESOURCES / 'CONSTRUCTION.json').read_text())
    if sealed['binding'] != review_binding(requests, fixtures) or sealed['source_sha256'] != sha256_file(Path(source) / 'requests.jsonl'):
        blockers.append('construction binding differs; regenerate before independent reviews')
    if sealed['source_files'] != source_files:
        blockers.append('construction source-file snapshot changed; reseal before review')
    if sealed['fixtures_sha256'] != sha256_file(RESOURCES / 'authored-fixtures.jsonl'):
        blockers.append('sealed checker fixtures changed')
    pools = load_pools(pools_spec)
    audit_doc = audit(requests, pools, ck)
    blockers += ['leakage: %(kind)s %(request_id)s (%(rule)s)' % h for h in audit_doc['hits']]
    blockers += ['PENDING: pool role %s not supplied' % r for r in audit_doc['missing_roles']]
    blockers += ['PENDING: fewer than five replay clauses for ' + rid for rid in audit_doc['insufficient_nearest']]
    if inspection is None or not Path(inspection).exists(): blockers.append('PENDING: nearest-replay inspection file not supplied')
    else: blockers += check_inspection(read_jsonl(inspection), audit_doc)
    review_docs = [json.loads(Path(path).read_text()) for path in reviews]
    blockers += check_reviews(review_docs, requests, fixtures)
    return dict(ck=ck, requests=requests, source_files=source_files, fixtures=fixtures, audit=audit_doc,
                pools=pools, review_docs=review_docs, blockers=blockers)


def build_available(source, pools_spec, inspection, reviews):
    """Manager-authorized freeze of the audited local set; defer named node inputs.

    Reuse the completed exclusion computation only after checking its exact
    artifact hash, all 23 raw pool hashes/members, source, fixtures and reviews.
    The historical audit/inspection still truthfully say globally incomplete.
    """
    policy = json.loads((RESOURCES/'freeze-ruling.json').read_text())
    inventory_path = RESOURCES/'exclusions/inventory.available.json'
    audit_path = RESOURCES/'exclusions/audit.available.json.gz'
    require(policy['schema'] == 'sp-node-verification-policy-v1' and policy['required_before_scoring']
            and policy['verification_only'] and policy['pair_local_hits']
            and not policy['deletion_allowed'] and not policy['rescoring_allowed'], 'invalid available-freeze ruling')
    require(sha256_file(inventory_path) == policy['available_inventory_sha256']
            and sha256_file(audit_path) == policy['available_audit_sha256']
            and sha256_file(inspection) == policy['inspection_sha256']
            and sha256_file(Path(source)/'requests.jsonl') == policy['source_sha256'], 'available evidence binding changed')
    inventory = json.loads(inventory_path.read_text())
    audit_doc = json.loads(gzip.decompress(audit_path.read_bytes()))
    pools = load_pools(pools_spec, available_scope=True)
    require(pool_identities(pools) == audit_doc['pools'], 'available pool identities differ from audited snapshot')
    require(len(pools) == policy['available_pool_count']
            and sum(len(p['rows']) for p in pools) == policy['available_row_count'], 'available inventory coverage changed')
    require(not audit_doc['hits'] and not audit_doc['missing_roles'] and not audit_doc['insufficient_nearest'], 'available audit is not clean')
    ck = load_checkers(); requests, source_files = load_source(source)
    blockers = validate_source(requests, ck)
    fixtures = []
    for row in requests:
        built, errors = fixtures_for(row, ck); fixtures += built; blockers += errors
    blockers += [str(x) for x in check_fixtures(fixtures, requests, ck)]
    sealed = json.loads((RESOURCES/'CONSTRUCTION.json').read_text())
    require(sealed['binding'] == review_binding(requests, fixtures) and sealed['source_files'] == source_files
            and sealed['fixtures_sha256'] == sha256_file(RESOURCES/'authored-fixtures.jsonl'), 'construction binding changed')
    require(json.loads((RESOURCES/'compatibility.json').read_text()) == ck.compatibility_matrix(), 'compatibility mismatch')
    edge = json.loads((RESOURCES/'checker-edge-fixtures.json').read_text())
    require(all(ck.check(e['text'],e['constraint']) == e['expected'] for e in edge), 'checker edge fixture mismatch')
    inspections = read_jsonl(inspection)
    blockers += check_inspection(inspections, audit_doc, available_scope=True)
    binding = {r['request_id']:digest(r) for r in requests}
    require(len(inspections) == len(requests) and all(x.get('request_sha256') == binding.get(x['request_id']) for x in inspections),
            'inspection request binding changed')
    review_docs = [json.loads(Path(p).read_text()) for p in reviews]
    blockers += check_reviews(review_docs, requests, fixtures)
    # Preserve historical audit bytes on disk. The frozen copy declares the
    # authorized scope explicitly; it never asserts unavailable inputs passed.
    audit_doc = dict(audit_doc, status='CLEAN_AVAILABLE', node_inventory_verification='REQUIRED_BEFORE_SCORING')
    return dict(ck=ck, requests=requests, source_files=source_files, fixtures=fixtures,
                audit=audit_doc, pools=pools, review_docs=review_docs, blockers=blockers,
                node_policy=policy, inventory=inventory)


def freeze(out, source=SOURCE, pools_spec=None, inspection=None, reviews=(), available_only=False):
    """Write once only after genuine independent reviews and frozen-superset exclusions."""
    out = Path(out)
    state = (build_available(source, pools_spec, inspection, reviews) if available_only
             else build(source, pools_spec, inspection, reviews))
    if not (out / 'FROZEN.json').exists(): write_json(out / 'audit.json', state['audit'])
    require(not state['blockers'], 'SP panel cannot freeze:\n' + '\n'.join(state['blockers']))
    panel = panel_rows(state['requests'])
    files = {'panel.jsonl': panel, 'fixtures.jsonl': state['fixtures'], 'requests.jsonl': state['requests']}
    frozen = dict(schema=SCHEMA, checker_version=state['ck'].VERSION, checker_sha256=sha256_file(CHECKERS),
                  source_files=state['source_files'], requests_sha256=digest(state['requests']),
                  constraints_sha256=digest({r['request_id']: r['constraints'] for r in state['requests']}),
                  wording_sha256=digest({r['request_id']: [r[w] for w in WORDINGS] for r in state['requests']}),
                  witnesses_sha256=digest({r['request_id']: r['witness'] for r in state['requests']}),
                  pools=pool_identities(state['pools']), code=code_hashes(), resources=resource_hashes(),
                  review_binding=review_binding(state['requests'], state['fixtures']),
                  review_receipts_sha256=[digest(d) for d in state['review_docs']],
                  inspection_sha256=sha256_file(inspection), n_requests=len(state['requests']), n_prompts=len(panel),
                  exclusion_fingerprints=exclusion_fingerprints(state['requests']))
    if state.get('node_policy') is not None:
        frozen.update(status='FROZEN', freeze_scope='complete-locally-available-inventory',
                      node_verification=state['node_policy'],
                      node_verification_implementation_sha256=sha256_file(RESOURCES/'node_verification.py'))
    if (out / 'FROZEN.json').exists():
        existing = json.loads((out / 'FROZEN.json').read_text())
        require({k: v for k, v in existing.items() if k != 'files'} == frozen, 'SP panel already frozen with different content')
        verify_frozen(out); return existing
    for name, rows in files.items(): write_jsonl(out / name, rows)
    for index, receipt in enumerate(state['review_docs'], 1): write_json(out / ('review-%d.json' % index), receipt)
    frozen['files'] = {name: sha256_file(out / name) for name in (*files, 'audit.json', 'review-1.json', 'review-2.json')}
    if state.get('node_policy') is not None:
        write_jsonl(out/'nearest-inspection.jsonl', read_jsonl(inspection))
        write_json(out/'available-inventory.json', state['inventory'])
        frozen['files'].update({name:sha256_file(out/name) for name in ('nearest-inspection.jsonl','available-inventory.json')})
    # Exclusive creation enforces the scientific write-once boundary.
    with (out/'FROZEN.json').open('x', encoding='utf-8') as stream:
        json.dump(frozen, stream, sort_keys=True, indent=1); stream.write('\n')
    return frozen


def verify_frozen(root=RESOURCES):
    root = Path(root); path = root / 'FROZEN.json'
    if not path.exists(): raise FileNotFoundError('PENDING: SP panel not frozen')
    doc = json.loads(path.read_text())
    require(doc.get('schema') == SCHEMA, 'wrong SP freeze schema')
    for name, expected in doc['files'].items():
        require(sha256_file(root / name) == expected, 'frozen SP file changed: ' + name)
    require(resource_hashes() == doc['resources'], 'SP resource bytes differ from frozen hashes')
    require(code_hashes() == doc['code'], 'SP preparation/scoring code differs from frozen hashes')
    if doc.get('node_verification') is not None:
        require(sha256_file(RESOURCES/'node_verification.py') == doc['node_verification_implementation_sha256'],
                'node verification implementation changed')
    load_templates()  # Recheck the pinned replay-source bytes, not just copied metadata.
    return doc


def pair_verification_result(requests, problems, pools=()):
    """Localize overlap hits; incomplete coverage/identity never implies clean pairs."""
    ids = sorted(r['request_id'] for r in requests)
    affected = sorted({h[key] for h in problems for key in ('request_id','other') if h.get(key) in ids})
    incomplete = [h for h in problems if not h.get('request_id')]
    unverified = ids if incomplete else affected
    return dict(schema='sp-recheck-v2', pools=list(pools), hits=problems,
                status='UNVERIFIED' if problems else 'CLEAN', verification_complete=not incomplete,
                scoring_allowed=not incomplete, affected_request_ids=affected, unverified_request_ids=unverified,
                pair_status={rid:'UNVERIFIED' if rid in unverified else 'VERIFIED' for rid in ids},
                unverified_prompt_ids=[rid+':'+wording for rid in unverified for wording in WORDINGS],
                verification_only=True, panel_changed=False, rescoring_allowed=False, deletes=0)


def verify_returned_pools(doc, pools, requests, ck):
    """Containment/exclusion verification only. Never alter or reselect a frozen row."""
    bound = {p['name']: p for p in doc['pools']}
    problems = []
    missing = sorted({'broad', 'k', 'cr'} - {p['role'] for p in pools})
    for role in missing: problems.append(dict(kind='missing-return-role', role=role))
    for pool in pools:
        name = pool.get('candidate_superset_name')
        parent = bound.get(name)
        if parent is None or parent['role'] != pool['role'] or parent['sha256'] != pool.get('candidate_superset_sha256'):
            problems.append(dict(kind='candidate-superset-binding', pool=pool['name'])); continue
        extras = sorted(set(candidate_members(pool)) - set(parent['candidate_members']))
        if extras: problems.append(dict(kind='outside-frozen-candidate-superset', pool=pool['name'], prompt_sha256=extras))
    # The recheck can discover a response overlap, but can never repair the panel.
    checked = audit(requests, pools, ck)
    problems += checked['hits']
    return pair_verification_result(requests, problems, pool_identities(pools))


def recheck(frozen_root, pools_spec, out=None):
    requests = []
    try:
        doc = verify_frozen(frozen_root)
        ck = load_checkers(expected_sha256=doc['checker_sha256'])
        requests = read_jsonl(Path(frozen_root)/'requests.jsonl')
        if doc.get('node_verification') is not None:
            node_spec = json.loads(Path(pools_spec).read_text())
            require(node_spec.get('frozen_sha256') == sha256_file(Path(frozen_root)/'FROZEN.json'), 'node spec bound to different freeze')
            spec = importlib.util.spec_from_file_location('sp_node_verification', RESOURCES/'node_verification.py')
            node = importlib.util.module_from_spec(spec); spec.loader.exec_module(node)
            result = node.verify(doc, pools_spec, requests, ck)
        else:
            pools = load_pools(pools_spec, returned=True)
            result = verify_returned_pools(doc, pools, requests, ck)
    except (ValueError, OSError, KeyError, TypeError, ImportError):
        result = pair_verification_result(requests, [dict(kind='invalid-freeze-or-return-binding')])
    frozen_path = Path(frozen_root)/'FROZEN.json'
    result['frozen_sha256'] = sha256_file(frozen_path) if frozen_path.exists() else None
    if out is not None: write_json(out, result)
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('action', choices=('construction', 'audit', 'freeze', 'recheck'))
    p.add_argument('--source', type=Path, default=SOURCE)
    p.add_argument('--available-only', action='store_true', help='apply the bound manager ruling and mandatory node VERIFY-ONLY policy')
    p.add_argument('--pools', type=Path)
    p.add_argument('--reviews', type=Path, nargs=2, default=())
    p.add_argument('--inspection', type=Path)
    p.add_argument('--out', type=Path, default=RESOURCES)
    p.add_argument('--frozen', type=Path, default=RESOURCES)
    a = p.parse_args(argv)
    if a.action == 'construction':
        doc = construction(a.source)
        print('SP construction sealed:', doc['n_requests'], 'requests,', doc['n_fixtures'], 'fixtures; independent reviews and exclusions PENDING')
        return 0
    p.error('--pools is required for audit/freeze/recheck') if a.pools is None else None
    if a.action == 'recheck':
        result = recheck(a.frozen, a.pools, a.out)
        print('SP recheck:', result['status'], len(result['hits']), 'hits')
        return 0 if result['status'] == 'CLEAN' else 1
    if a.action == 'audit':
        state = build(a.source, a.pools, a.inspection, a.reviews)
        if state['audit'] is not None:
            write_json(a.out / 'audit.json', state['audit'])
        print('\n'.join(state['blockers']) or 'SP panel ready to freeze')
        return 0 if not state['blockers'] else 1
    doc = freeze(a.out, a.source, a.pools, a.inspection, a.reviews, available_only=a.available_only)
    print('SP panel frozen:', doc['n_requests'], 'requests,', doc['n_prompts'], 'prompts')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
