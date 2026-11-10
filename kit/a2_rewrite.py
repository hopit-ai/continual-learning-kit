"""Frozen R parser, source-binding verifier and renderer. The API has no item input."""
from datetime import datetime
from decimal import Decimal
from fractions import Fraction
import itertools
import re
import json
from kit.a2_common import canonical, digest, RES
from kit.beds import finqa

SCALES = {'US dollars': 1, 'thousands of US dollars': 1000,
          'millions of US dollars': 1000000, 'billions of US dollars': 1000000000}
MARKER = re.compile(r'Statement: ([a-z_]+); accession (\S+), filed '
                    r'([0-9]{4}-[0-9]{2}-[0-9]{2}); source (\S+)\. Amounts in '
                    r'(US dollars|thousands of US dollars|millions of US dollars|'
                    r'billions of US dollars)\.')
FCF = ('How many percentage points did FCF/revenue change, current minus prior year? '
       'FCF is signed operating cash flow minus positive cash paid for "<CAPEX>"; '
       'use "<OCF>" and "<REVENUE>" for the matched durations. Answer in percentage '
       'points (signed), rounded to one decimal place; round only the final result.')
CASH = ('How many percentage points did cash conversion change, current minus prior year? '
        'Cash conversion is "<OCF>" divided by "<OPERATING_INCOME>", expressed as a '
        'percentage. Answer in percentage points (signed), rounded to one decimal '
        'place; round only the final result.')
FORMULAS = {'fcf_margin_change': '100*((a+b)/c-(d+e)/f)',
            'cash_conversion_change': '100*(a/b-c/d)'}
PROCEDURES = {
    'fcf_margin_change': 'Procedure: Match current and prior fiscal durations. Keep '
    'operating cash flow signed. Treat capital expenditure as a negative signed '
    'outflow. For each period, divide operating cash flow plus the signed '
    'capital-expenditure outflow by revenue. Subtract the prior ratio from the '
    'current ratio and multiply by 100. Report percentage points.',
    'cash_conversion_change': 'Procedure: Match current and prior fiscal durations. '
    'Keep operating cash flow signed. For each period, divide operating cash flow '
    'by operating income. Subtract the prior ratio from the current ratio and '
    'multiply by 100. Report percentage points.'}


class Fallback(ValueError):
    """Counted raw fallback, never a scientific failure or a retry request."""


def need(ok, reason):
    if not ok:
        raise Fallback(reason)


def question(text):
    for family, template in (('fcf_margin_change', FCF), ('cash_conversion_change', CASH)):
        names = re.findall(r'<([A-Z_]+)>', template)
        pattern = re.escape(template)
        for name in names:
            pattern = pattern.replace(re.escape('<'+name+'>'), r'([^"\n]+)', 1)
        match = re.fullmatch(pattern, text)
        if match:
            return family, dict(zip(names, match.groups()))
    raise Fallback('unsupported_question')


def decimal_string(value):
    text = format(value, 'f')
    if '.' in text:
        text = text.rstrip('0').rstrip('.')
    return '0' if Decimal(text) == 0 else text


def number(display, unit):
    text = re.sub(r'\[[0-9]+\]', '', display).strip()
    match = re.fullmatch(r'\$?\s*(\()?\s*\$?\s*([0-9,]+(?:\.[0-9]+)?)\s*(\))?', text)
    if not match or bool(match[1]) != bool(match[3]):
        return None, None
    try:
        # Decimal's default precision must not round arbitrary displayed digits.
        from decimal import localcontext
        with localcontext() as ctx:
            ctx.prec = max(50, len(match[2])+20)
            value = Decimal(match[2].replace(',', '')) * (-1 if match[1] else 1)
            return decimal_string(value), decimal_string(value*SCALES[unit])
    except ArithmeticError:
        return None, None


def period(text):
    match = re.fullmatch(r'([0-9]+)\s+Months?\s+Ended\s+(.+)', text, re.I)
    if not match or not 1 <= int(match[1]) <= 12:
        return None, None
    for fmt in ('%b. %d, %Y', '%b %d, %Y', '%B %d, %Y'):
        try:
            return int(match[1]), datetime.strptime(match[2].replace('Sept.', 'Sep.'), fmt).date().isoformat()
        except ValueError:
            pass
    return None, None


def parse(prompt, response, admitted, prompt_sha256):
    need(admitted is True, 'raw_admission_absent')
    need(isinstance(prompt, str) and digest(prompt.encode()) == prompt_sha256, 'prompt_missing_or_mismatch')
    before, delimiter, query = prompt.rpartition('\nQuestion: ')
    need(delimiter, 'question_delimiter_missing')
    family, _ = question(query)
    matches = list(MARKER.finditer(before))
    need(matches and before.count('Statement: ') == len(matches), 'statement_parse')
    cells = []
    for e, marker in enumerate(matches, 1):
        segment = before[marker.end():matches[e].start() if e < len(matches) else len(before)]
        start = segment.find('| ')
        need(start >= 0, 'table_missing')
        table = segment[start:]
        header, lf, remaining = table.partition('\n')
        need(lf and header.endswith(' |'), 'table_header')
        headings = header[2:-2].split(' | ')
        need(len(headings) >= 2 and headings[0] == '' and all('|' not in x for x in headings), 'table_header_width')
        rows = []
        while remaining.startswith('| '):
            # Read a row by the fixed width, allowing outside suffix after its terminator.
            pattern = r'\| ' + r' \| '.join([r'([^|\n]*)']*len(headings)) + r' \|'
            row = re.match(pattern, remaining)
            need(row is not None, 'table_row_width')
            rows.append(row.groups())
            remaining = remaining[row.end():]
            if remaining.startswith('\n| '):
                remaining = remaining[1:]
            else:
                break
        need(rows and '|' not in remaining, 'extra_or_malformed_table')
        for r, values in enumerate(rows, 1):
            for c in range(1, len(headings)):
                duration, end = period(headings[c])
                signed, usd = number(values[c], marker[5])
                cells.append(dict(id=f'e{e}:r{r}:c{c}', statement=marker[1], source=marker[4],
                                  label=values[0], period=headings[c], duration_months=duration,
                                  period_end=end, displayed_value=values[c], unit=marker[5],
                                  signed_value=signed, usd_value=usd))
    answers = list(finqa.ANSWER_LINE.finditer(response))
    need(answers, 'answer_line_missing')
    match = answers[-1]
    need('\n' not in match[0], 'answer_match_spans_lines')
    line = response[response.rfind('\n', 0, match.start())+1:]
    line = line.split('\n', 1)[0]
    return dict(family=family, question=query, cells=cells, response=response, answer_line=line)


def messages(trace):
    return [dict(role='system', content=(RES/'rewrite-system.txt').read_text()),
            dict(role='user', content='Convert this admitted trace using the fixed schema:\n'+canonical(trace).decode())]


def strict_json(text):
    def pairs(values):
        need(len(values) == len(dict(values)), 'duplicate_json_key')
        return dict(values)
    try:
        return json.loads(text, object_pairs_hook=pairs, parse_constant=lambda _: (_ for _ in ()).throw(Fallback('nonfinite_json')))
    except Fallback:
        raise
    except (ValueError, TypeError) as error:
        raise Fallback('malformed_json') from error


def clean_label(text):
    return re.sub(r'(?:\s*\[[0-9]+\])+$', '', text.strip()).strip()


def unique_binding(trace):
    family, labels = question(trace['question'])
    required = [('OCF', 'cash_flow'), ('CAPEX', 'cash_flow'), ('REVENUE', 'income')] if family == 'fcf_margin_change' else [('OCF', 'cash_flow'), ('OPERATING_INCOME', 'income')]
    pools = [[c for c in trace['cells'] if c['statement'] == statement and
              clean_label(c['label']) == clean_label(labels[label]) and
              all(c[k] is not None for k in ('duration_months', 'period_end', 'signed_value', 'usd_value'))]
             for label, statement in required]
    complete = []
    for current in itertools.product(*pools):
        if len({(c['duration_months'], c['period_end']) for c in current}) != 1:
            continue
        for prior in itertools.product(*pools):
            if len({(c['duration_months'], c['period_end']) for c in prior}) != 1:
                continue
            if current[0]['duration_months'] != prior[0]['duration_months']:
                continue
            days = (datetime.fromisoformat(current[0]['period_end'])-datetime.fromisoformat(prior[0]['period_end'])).days
            if 350 <= days <= 380:
                selected = (*current, *prior)
                if len({c['id'] for c in selected}) == len(selected):
                    complete.append(selected)
    need(complete, 'period_or_label_binding')
    latest = max(binding[0]['period_end'] for binding in complete)
    complete = [binding for binding in complete if binding[0]['period_end'] == latest]
    need(len(complete) == 1, 'ambiguous_binding')
    return dict(zip('abcdef', complete[0]))


def verify(trace, generated):
    obj = strict_json(generated)
    need(obj != {'failure': 'unresolved'}, 'model_unresolved')
    family = trace['family']
    need(isinstance(obj, dict) and set(obj) == {'family', 'cells'} and obj['family'] == family, 'binding_schema')
    expected = unique_binding(trace)
    need(isinstance(obj['cells'], dict) and set(obj['cells']) == set(expected), 'variable_set')
    need(obj['cells'] == {k: c['id'] for k, c in expected.items()}, 'incorrect_cells')
    values = {k: Fraction(c['usd_value']) for k, c in expected.items()}
    if family == 'fcf_margin_change':
        a, b, c, d, e, f = (values[k] for k in 'abcdef')
        need(b < 0 and e < 0 and c > 0 and f > 0, 'sign_or_denominator')
        result = 100*((a+b)/c-(d+e)/f)
    else:
        a, b, c, d = (values[k] for k in 'abcd')
        need(b > 0 and d > 0, 'sign_or_denominator')
        result = 100*(a/b-c/d)
    answer = finqa.ANSWER_LINE.findall(trace['response'])[-1].strip()
    last = answer.strip('`*. ').lower()
    need(not last.startswith(('yes', 'no')), 'nonnumeric_answer')
    need(not re.search(r'\$|dollars?|thousands?|millions?|billions?', answer, re.I) and
         '%' not in answer and not re.search(r'\bpercent\b(?!age points)', answer, re.I), 'conflicting_suffix')
    match = finqa.NUMBER.search(last)
    need(match is not None, 'nonnumeric_answer')
    token = match[0]
    digits = re.sub(r'[^0-9.]', '', token)
    try:
        numeric = Fraction(digits) * (-1 if '(' in token and ')' in token or token.strip().startswith('-') else 1)
    except ValueError as error:
        raise Fallback('nonnumeric_answer') from error
    need(abs(result-numeric) <= Fraction(1, 20), 'arithmetic_disagreement')
    return expected


def render(trace, binding):
    entries = []
    for variable, c in binding.items():
        q = lambda key: canonical(c[key]).decode()
        entries.append(f'{variable} = label={q("label")}; period={q("period")}; '
                       f'duration_months={c["duration_months"]}; displayed={q("displayed_value")}; '
                       f'unit={q("unit")}; value_usd={c["usd_value"]}; source={q("source")}')
    formula = re.sub(r'[a-f]', lambda m: '('+binding[m[0]]['usd_value']+')', FORMULAS[trace['family']])
    return '\n'.join([PROCEDURES[trace['family']], 'Bindings: '+' | '.join(entries),
                      'Application: '+formula, trace['answer_line']])


def rewrite(prompt, response, admitted, prompt_sha256, generate, fits):
    """One attempt, cached by caller. Both pre-generation and model failures fall back."""
    reason = None
    trace = None
    generated = None
    binding = None
    try:
        trace = parse(prompt, response, admitted, prompt_sha256)
        need(fits(messages(trace), None), 'rewrite_input_overflow')
        generated = generate(messages(trace))
        need(generated['finish_reason'] in ('stop', 'eos'), 'generation_truncation')
        binding = verify(trace, generated['response'])
        result = render(trace, binding)
        need(len(result.strip()) <= 2500 and fits(prompt, result), 'rewrite_output_or_teacher_cap')
    except Fallback as error:
        result, reason = response, str(error)
    return dict(response=result, accepted=reason is None, fallback_reason=reason,
                raw_sha256=digest(response.encode()), rewrite_sha256=digest(result.encode()),
                family=trace['family'] if trace else None, generation_attempted=generated is not None,
                trace_json_sha256=digest(canonical(trace)) if trace else None,
                rewrite_prompt_sha256=digest(canonical(messages(trace))) if trace else None,
                generated=generated, verified_bindings={k: c['id'] for k, c in binding.items()} if binding else None,
                model_revision='b968826d9c46dd6066d109eabc6255188de91218')
