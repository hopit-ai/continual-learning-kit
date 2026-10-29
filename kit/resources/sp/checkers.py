"""SP checkers v2: pinned replay semantics, raw JSON deviation, no output repair.

Except strict JSON, the executable functions are extracted from the hash-bound
replay checker module without importing its unrelated dependencies. Required and
forbidden words use case-insensitive substring matches; exact frequency uses
lowercased regex word tokens. Word totals use whitespace splitting. Sentence
splitting, divider paragraphs, permissive marker bullets, lowercase and raw exact
suffix checks are exactly those of the pinned replay module. No universal empty
or language test is invented. See source/README.md and checker-edge-fixtures.json.
"""
from __future__ import annotations
import ast
import hashlib
import json
import re
from pathlib import Path

VERSION = 'sp-checkers-v2-raw-json'
SOURCE_SHA256 = 'afea345e3bc100e6d5588ca14e7269141e8810e1489e60ad64a59cd57360201c'
_SOURCE = Path(__file__).resolve().parents[1] / 'p3b' / 'if_functions.py'
raw = _SOURCE.read_bytes()
if hashlib.sha256(raw).hexdigest() != SOURCE_SHA256:
    raise ValueError('pinned replay checker changed')
_NAMES = ('verify_keywords', 'validate_forbidden_words', 'verify_keyword_frequency',
          'validate_word_constraint', 'verify_sentence_constraint', 'verify_paragraph_count',
          'verify_bullet_points', 'validate_lowercase', 'validate_end')
_tree = ast.parse(raw)
_nodes = [node for node in _tree.body if isinstance(node, ast.FunctionDef) and node.name in _NAMES]
assert len(_nodes) == len(_NAMES)
_legacy = {'re': re, 'json': json}
exec(compile(ast.Module(body=_nodes, type_ignores=[]), '<pinned-replay-checkers>', 'exec'), _legacy)

WORD = re.compile(r"\b\w+\b")
_SENTENCE_END = re.compile(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s")
FAMILIES = ('required_word', 'forbidden_word', 'word_frequency', 'word_count_interval',
            'sentence_count', 'paragraph_count', 'bullet_count', 'json_only', 'lowercase_only', 'required_suffix')
ARGUMENTS = {
    'required_word': ('word',), 'forbidden_word': ('word',), 'word_frequency': ('word', 'count'),
    'word_count_interval': ('min', 'max'), 'sentence_count': ('count',), 'paragraph_count': ('count',),
    'bullet_count': ('count',), 'json_only': (), 'lowercase_only': (), 'required_suffix': ('suffix',),
}
INCOMPATIBLE = frozenset(frozenset(p) for p in (
    ('json_only', 'sentence_count'), ('json_only', 'paragraph_count'),
    ('json_only', 'bullet_count'), ('json_only', 'required_suffix'),
    ('paragraph_count', 'bullet_count'),  # literal divider lines count as replay bullets
))


def _fold(text): return text.lower()
def words(text): return WORD.findall(text)
def word_spans(text): return [m.span() for m in WORD.finditer(text)]
def count_word(text, word): return WORD.findall(text.lower()).count(word.lower())
def sentences(text): return _SENTENCE_END.split(text)
def paragraphs(text): return [p.strip() for p in text.strip().split('* * *') if p.strip()]
def bullets(text): return [line.strip() for line in text.split('\n') if line.strip().startswith(('*', '-'))]


def _json(text):
    def reject(value): raise ValueError('nonfinite JSON constant')
    try: json.loads(text, parse_constant=reject)
    except ValueError: return False
    return True  # Any JSON value, including strings, numbers and null.


def json_fence_failure(text):
    """Descriptive flag only; never feeds a permissive correctness pass."""
    return not _json(text) and '```' in text


CHECKS = {
    'required_word': lambda r, a: _legacy['verify_keywords'](r, [a['word']]),
    'forbidden_word': lambda r, a: _legacy['validate_forbidden_words'](r, [a['word']]),
    'word_frequency': lambda r, a: _legacy['verify_keyword_frequency'](r, a['word'], a['count']),
    'word_count_interval': lambda r, a: _legacy['validate_word_constraint'](r, a['min'], 'at least') and _legacy['validate_word_constraint'](r, a['max'], 'at most'),
    'sentence_count': lambda r, a: _legacy['verify_sentence_constraint'](r, a['count'], 'at least') and _legacy['verify_sentence_constraint'](r, a['count'], 'at most'),
    'paragraph_count': lambda r, a: _legacy['verify_paragraph_count'](r, a['count']),
    'bullet_count': lambda r, a: _legacy['verify_bullet_points'](r, a['count']),
    'json_only': lambda r, a: _json(r),
    'lowercase_only': lambda r, a: _legacy['validate_lowercase'](r),
    'required_suffix': lambda r, a: _legacy['validate_end'](r, a['suffix']),
}


def compatibility_matrix():
    return dict(schema='sp-compatibility-v2', families=list(FAMILIES),
        matrix={a: {b: ('forbidden' if a == b or frozenset((a,b)) in INCOMPATIBLE else 'parameter-conditional') for b in FAMILIES} for a in FAMILIES},
        parameter_rules=['No repeated family.', 'No two word constraints on the same lowercased argument.',
                         'Lowercase requires lowercase word/suffix arguments as a conservative authoring restriction.',
                         'A passing witness and one single-constraint failure per constraint are mandatory.'],
        reasons={'json-format': 'JSON cannot pair with prose suffix, bullets, paragraphs or sentence counts.',
                 'divider-bullets': 'Replay paragraph dividers also count as bullet lines.'})


def validate_constraint(constraint):
    """Reject unknown families, missing/extra arguments and arguments a checker cannot honour."""
    if not isinstance(constraint, dict) or constraint.get("family") not in CHECKS:
        raise ValueError("unknown checker family: %r" % (constraint,))
    family, args = constraint["family"], constraint.get("args", {})
    if not isinstance(args, dict) or set(args) != set(ARGUMENTS[family]):
        raise ValueError("%s arguments must be exactly %s" % (family, ARGUMENTS[family]))
    if "word" in args:
        if not isinstance(args["word"], str) or WORD.fullmatch(args["word"]) is None:
            raise ValueError("%s word must be exactly one word token" % family)
    for key in ("count", "min", "max"):
        if key in args and (not isinstance(args[key], int) or isinstance(args[key], bool) or args[key] < 0):
            raise ValueError("%s %s must be a non-negative integer" % (family, key))
    if family in ("sentence_count", "paragraph_count", "bullet_count") and args["count"] < 1:
        raise ValueError(family + " count must be at least 1")
    if family == "word_frequency" and args["count"] < 1:
        raise ValueError("word_frequency count must be at least 1 (use forbidden_word for zero)")
    if family == "word_count_interval" and not 1 <= args["min"] <= args["max"]:
        raise ValueError("word_count_interval requires 1 <= min <= max")
    if family == "required_suffix":
        suffix = args["suffix"]
        if not isinstance(suffix, str) or not suffix or suffix != suffix.rstrip() or suffix != suffix.lstrip():
            raise ValueError("required_suffix must be non-empty without leading/trailing whitespace")
    return constraint


def validate_constraints(constraints):
    if not isinstance(constraints, list) or len(constraints) not in (1, 2):
        raise ValueError("a request carries one or two constraints")
    for c in constraints:
        validate_constraint(c)
    if len(constraints) == 2:
        a, b = constraints
        if a["family"] == b["family"]:
            raise ValueError("a request cannot repeat a checker family")
        if frozenset((a["family"], b["family"])) in INCOMPATIBLE:
            raise ValueError("incompatible checker families: %s + %s" % (a["family"], b["family"]))
        wa, wb = a["args"].get("word"), b["args"].get("word")
        if wa is not None and wb is not None and _fold(wa) == _fold(wb):
            raise ValueError("two word constraints on the same word")
        for c, other in ((a, b), (b, a)):
            if other["family"] == "lowercase_only":
                text = c["args"].get("word") or c["args"].get("suffix") or ""
                if text != text.lower():
                    raise ValueError("lowercase_only conflicts with a cased word/suffix argument")
    return constraints


def check(response, constraint):
    """One instruction verdict on unmodified raw text."""
    validate_constraint(constraint)
    if not isinstance(response, str):
        raise TypeError("response must be text")
    result = CHECKS[constraint["family"]](response, constraint["args"])
    if not isinstance(result, bool):
        raise TypeError("checker returned non-boolean")
    return result


def check_all(response, constraints):
    """Instruction-level verdicts in constraint order."""
    return [check(response, c) for c in validate_constraints(constraints)]
