"""Frozen pure v4 audit/scorer/retrieval functions, copied verbatim from the SEC slice builder. No IO/network."""
import re
from kit.beds import finqa
def load_kit(name):
    assert name == "beds/finqa"
    return finqa

VALUE = re.compile(r"^\$?\s*(\()?\s*\$?\s*([\d,]+(?:\.\d+)?)\s*(\))?$")

def parse_value(text: str):
    """'$ 94,036' -> 94036.0 ; '(171)' -> -171.0 ; '' -> None. Footnote markers like '[1]' are dropped first."""
    text = re.sub(r"\[\d+\]", "", text or "").strip()
    match = VALUE.match(text)
    if not match:
        return None
    value = float(match.group(2).replace(",", ""))
    return -value if match.group(1) and match.group(3) else value

def fmt(value: float) -> str:
    return ("%d" % value) if float(value).is_integer() else repr(value)

V4_FAMILIES = ("fcf_margin_change", "cash_conversion_change", "quarter_incremental_margin",
               "growth_acceleration", "segment_margin_divergence")

V4_SCALE = {"US dollars": 1, "thousands of US dollars": 1000,
            "millions of US dollars": 1000000, "billions of US dollars": 1000000000}

def v4_compute(formula, operands):
    """Decimal expression and separate linear program execution, no intermediate rounding."""
    import ast
    from decimal import Decimal, localcontext
    names = {chr(97 + k): Decimal(o['normalized_usd']) for k, o in enumerate(operands)}
    steps, results = [], []
    operations = {ast.Add: ('add', lambda a, b: a + b), ast.Sub: ('subtract', lambda a, b: a - b),
                  ast.Mult: ('multiply', lambda a, b: a * b), ast.Div: ('divide', lambda a, b: a / b)}
    def visit(node):
        if isinstance(node, ast.Name):
            return names[node.id], str(names[node.id])
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return Decimal(node.value), str(node.value)
        if not isinstance(node, ast.BinOp) or type(node.op) not in operations:
            raise ValueError('unsupported formula')
        a, at = visit(node.left); b, bt = visit(node.right)
        op, fn = operations[type(node.op)]
        value = fn(a, b)
        steps.append('%s(%s, %s)' % (op, at, bt))
        return value, '#' + str(len(steps) - 1)
    with localcontext() as ctx:
        ctx.prec = 50
        value, _ = visit(ast.parse(formula, mode='eval').body)
        for op, a, b in re.findall(r'(add|subtract|multiply|divide)\(([^,]+),\s*([^\)]+)\)', ', '.join(steps)):
            get = lambda x: results[int(x[1:])] if x.startswith('#') else Decimal(x)
            x, y = get(a), get(b)
            results.append({'add': lambda: x + y, 'subtract': lambda: x - y,
                            'multiply': lambda: x * y, 'divide': lambda: x / y}[op]())
        assert results[-1] == value
        from decimal import ROUND_HALF_UP
        rounded = value.quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)
    return value, format(rounded, '.1f'), ', '.join(steps)

def v4_excerpt_label(e):
    return 'Statement: %s; accession %s, filed %s; source %s. Amounts in %s.' % (
        e['kind'], e['accession'], e['filing_date'], e['r_page'], e['unit'])

def check_v4_item(item, finqa):
    s = item['sec']
    assert s['template'] in V4_FAMILIES and s['version'] == 4
    for o in s['operands']:
        e = next(e for e in s['excerpts'] if e['r_page'] == o['r_page'])
        assert e['table'][o['table_row']][0] == o['shown_label']
        assert e['table'][0][o['table_column']] == o['period']
        assert e['table'][o['table_row']][o['table_column']] == o['cell']
        assert o['unit'] == e['unit'] and parse_value(o['cell']) == o['value']
        from decimal import Decimal
        assert Decimal(str(o['value'])) * V4_SCALE[o['unit']] == Decimal(o['normalized_usd'])
        assert o['context']['start'] and o['context']['end'] and o['context_id'] and o['concept']
    value, answer, program = v4_compute(s['formula'], s['operands'])
    assert str(value) == s['unrounded_gold'] and float(value) == item['qa']['exe_ans']
    assert answer == item['qa']['answer'] and program == item['qa']['program']
    assert finqa.gold_consistent(item)

def v4_explicit_correct(item, response):
    """Preregistered: requested percent/pp number within 0.05 (one final decimal); no x100 tolerance.

    Numeric final answers need no unit suffix because the prompt specifies it. A conflicting explicit suffix fails.
    """
    from decimal import Decimal
    f = load_kit('beds/finqa')
    pred = f.extract_answer(response)
    if pred is None or isinstance(pred, str):
        return False
    last = f.ANSWER_LINE.findall(response)[-1].strip().lower()
    unit = item['sec']['answer_unit']
    if re.search(r'\$|dollars?|thousands?|millions?|billions?', last):
        return False
    if unit == 'percentage points' and ('%' in last or re.search(r'\bpercent\b(?!age points)', last)):
        return False
    if unit == 'percent' and re.search(r'\bpp\b|percentage points', last):
        return False
    return abs(Decimal(str(float(pred))) - Decimal(item['sec']['unrounded_gold'])) <= Decimal('.0500000001')

def v4_case_text(case, response, finqa):
    parts = []
    for e in case['sec']['excerpts']:
        rowids = {o['table_row'] for o in case['sec']['operands'] if o['r_page'] == e['r_page']}
        parts += [v4_excerpt_label(e), finqa.render_table([e['table'][0]] + [e['table'][r] for r in sorted(rowids)])]
    return '\n'.join([*parts, 'Question: ' + case['qa']['question'], 'Solution:', response.strip()])

MEMORY_HEADER = ("Solved examples from earlier filings (questions this assistant answered before, each checked "
                 "correct). Use them only as worked examples; the report excerpt below is a different filing.")

def label_words(item: dict) -> set:
    return {w for o in item["sec"]["operands"] for w in re.findall(r"[a-z]+", o["label"].lower()) if len(w) > 2}

def similarity(item: dict, case: dict) -> float:
    a, b = label_words(item), label_words(case)
    overlap = len(a & b) / len(a | b) if a | b else 0.0
    return 2.0 * (item["sec"]["template"] == case["sec"]["template"]) + overlap + 0.25 * (item["sec"]["form"] == case["sec"]["form"])
