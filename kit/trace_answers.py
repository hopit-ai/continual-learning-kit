"""Owner's v7i3 answer reading; registered metric adapters remain unchanged."""
from decimal import Decimal, InvalidOperation
import json
import re


def final_answer(text):
    lines = text.rstrip().splitlines()
    if not lines:
        return None
    # Formatting on the label is allowed, only on the final nonblank line.
    for line in (lines[-1], line_markup_removed(lines[-1])):
        match = re.fullmatch(r'\s*(?:Answer\s*:|(?P<mark>\*\*|__|\*|_|`)Answer'
                             r'(?:\s*:(?P=mark)|(?P=mark)\s*:))\s*(?P<value>.+?)\s*', line, re.I)
        if match:
            return match.group('value')
    return None


def line_markup_removed(line):
    """Remove balanced whole-line markdown only; never touch the answer value."""
    line = line.strip()
    while True:
        for mark in ('**', '__', '*', '_', '`'):
            if line.startswith(mark) and line.endswith(mark) and len(line) > 2*len(mark):
                line = line[len(mark):-len(mark)].strip()
                break
        else:
            return line


def unformatted(value, *, numeric=False):
    """Remove wrappers only; never erase operators inside an answer."""
    value = str(value).strip()
    while value:
        before = value
        if numeric:
            # A period outside a wrapper is punctuation, not part of a unit.
            value = re.sub(r'(?<=[\])}*_`])\.$', '', value).strip()
        else:
            value = value.rstrip('.!?,;:').strip()
        for left, right in (('$', '$'), (r'\(', r'\)'), (r'\[', r'\]')):
            if value.startswith(left) and value.endswith(right) and len(value) >= len(left)+len(right):
                value = value[len(left):-len(right)].strip()
                break
        if not numeric and value.startswith('$'):
            value = value[1:].strip()
        if not numeric and value.endswith('%'):
            value = value[:-1].strip()
        boxed = re.fullmatch(r'\\boxed\s*\{([^{}]*)\}', value)
        if boxed:
            value = boxed.group(1).strip()
        for left, right in (('(', ')'), ('[', ']'), ('{', '}'),
                            ('**', '**'), ('__', '__'), ('*', '*'), ('_', '_'), ('`', '`')):
            if value.startswith(left) and value.endswith(right) and len(value) >= len(left)+len(right):
                value = value[len(left):-len(right)].strip()
                break
        if value == before:
            break
    return value


def number(value):
    value = unformatted(value, numeric=True).replace('\N{MINUS SIGN}', '-')
    # One complete number and at most one explicitly permitted unit. Currency
    # prefix and suffix are alternatives; neither may hide a second unit.
    token = (r'[-+]?(?:(?:[0-9]+|[0-9]{1,3}(?:,[0-9]{3})+)(?:\.[0-9]*)?|\.[0-9]+)'
             r'(?:[eE][-+]?[0-9]+)?')
    unit = (r'g|grams?|kg|kilograms?|mg|milligrams?|mol|moles?|%|percent|dollars?|cents?|'
            r'l|liters?|litres?|ml|milliliters?|g/mol')
    match = re.fullmatch(r'(?:(?P<currency>\$)\s*)?(?P<number>'+token+r')'
                         r'(?:\s*(?P<unit>'+unit+r')\.?)?', value, re.I)
    try:
        result = (Decimal(match.group('number').replace(',', ''))
                  if match and not (match['currency'] and match['unit']) else None)
        # Reject absurd magnitudes instead of overflowing or silently clamping.
        return (result if result is not None and result.is_finite() and
                abs(result.adjusted()) <= 100 else None)
    except (InvalidOperation, ArithmeticError):
        return None


def fomc_labels(prompt):
    """Read option names from the released prompt, before its quoted Text."""
    header = prompt.split('\nText:', 1)[0]
    pairs = re.findall(r'\b([A-Z])\.\s*([^,.\n]+)', header)
    labels = {key: value.strip().casefold() for key, value in pairs}
    return labels if len(labels) >= 2 and len(pairs) == len(labels) and len(set(labels.values())) == len(labels) else {}


def label(value, prompt):
    labels = fomc_labels(prompt)
    if not labels:
        return None
    hits = set()
    for candidate in (str(value).strip(), unformatted(value)):
        pair = re.fullmatch(r'(.+?)\.?\s+(.+)', candidate)
        for key, name in labels.items():
            if candidate.casefold() in (key.casefold(), name) or (pair and
                    unformatted(pair.group(1)).casefold() == key.casefold() and
                    unformatted(pair.group(2)).casefold() == name):
                hits.add(key)
    return next(iter(hits)) if len(hits) == 1 else None


def agrees(task, value, gold, prompt=''):
    if value is None:
        return False
    if task == 'FOMC':
        predicted, expected = label(value, prompt), label(gold, prompt)
        return predicted is not None and predicted == expected
    if task == 'NumGLUE-ds':
        predicted, expected = number(value), number(gold)
        try:
            return (predicted is not None and expected is not None and
                    abs(predicted-expected) <= Decimal('.01')*abs(expected))
        except ArithmeticError:
            return False
    return False


def meetingbank_verdict(text):
    text = text.strip()
    if text.startswith('```'):
        match = re.fullmatch(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.S)
        if not match:
            return False
        text = match.group(1).strip()
    try:
        # Reject duplicate keys and JSON numbers (1 == True in Python).
        def object_pairs(pairs):
            if len({key for key, _ in pairs}) != len(pairs):
                raise ValueError('duplicate verifier key')
            return dict(pairs)
        value = json.loads(text, object_pairs_hook=object_pairs)
        return (isinstance(value, dict) and set(value)=={'supported','adequate_coverage'}
                and all(item is True for item in value.values()))
    except (ValueError, TypeError):
        return False


def exploratory_score(task, row, text):
    return 100 * int(agrees(task, final_answer(text), row['answer'], row['prompt']))
