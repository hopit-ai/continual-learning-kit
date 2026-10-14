"""3C's explicitly restricted rational-answer MATH bed; standard-library scorer."""
import ast
from fractions import Fraction
import json
from pathlib import Path
import re

DATA_SOURCE = 'math_screen'
PROMPT = 'Solve the following mathematics problem. Show your working and put the final answer in \\boxed{{...}}.\n\n{problem}'


def last_box(text):
    """Return (contents, start); a malformed last box is not an earlier answer."""
    starts = list(re.finditer(r'\\boxed\s*\{', text))
    if not starts:
        return None
    m = starts[-1]
    depth = 1
    for end in range(m.end(), len(text)):
        if text[end] == '{':
            depth += 1
        elif text[end] == '}':
            depth -= 1
            if depth == 0:
                return text[m.end():end], m.start()
    return None


def rational(text):
    """Exact scalar arithmetic, no eval, symbols, units, percentages or collections.

    Restricted syntax deliberately defines the population, rather than pretending
    to implement general mathematical equivalence. Bound input and integer growth.
    """
    if not isinstance(text, str) or len(text) > 4096:
        raise ValueError('unsupported rational answer')
    text = text.strip().replace('\\dfrac', '\\frac').replace('\\tfrac', '\\frac')
    for token in ('\\left', '\\right', '\\,', '\\!', '\\;', '\\ '):
        text = text.replace(token, '')
    text = text.replace('\\cdot', '*').replace('\\times', '*')
    text = text.replace('{,}', ',')

    def ungroup(match):
        number = match.group()
        if ',' not in number:
            return number
        if not re.fullmatch(r'\d{1,3}(?:,\d{3})+(?:\.\d+)?', number):
            raise ValueError('invalid thousands grouping')
        return number.replace(',', '')

    text = re.sub(r'[\d.,]+', ungroup, text)
    for _ in range(32):
        # TeX permits a single digit instead of a braced argument: \frac12.
        replaced = re.sub(r'\\frac\s*(?:\{([^{}]+)\}|([0-9]))\s*(?:\{([^{}]+)\}|([0-9]))',
                          lambda m: f'(({m[1] or m[2]})/({m[3] or m[4]}))', text)
        if replaced == text:
            break
        text = replaced
    text = text.replace('{', '(').replace('}', ')').replace('^', '**')
    if not re.fullmatch(r'[\d\s.()+*/\-]+', text):
        raise ValueError('unsupported rational syntax')
    try:
        tree = ast.parse(text.strip(), mode='eval')
        if len(list(ast.walk(tree))) > 128:
            raise ValueError('expression too large')

        def walk(node):
            if isinstance(node, ast.Constant) and type(node.value) in (int, float):
                value = Fraction(ast.get_source_segment(text.strip(), node))
            elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
                value = walk(node.operand) * (-1 if isinstance(node.op, ast.USub) else 1)
            elif isinstance(node, ast.BinOp):
                a, b = walk(node.left), walk(node.right)
                if isinstance(node.op, ast.Add): value = a + b
                elif isinstance(node.op, ast.Sub): value = a - b
                elif isinstance(node.op, ast.Mult): value = a * b
                elif isinstance(node.op, ast.Div): value = a / b
                elif isinstance(node.op, ast.Pow) and b.denominator == 1 and abs(b) <= 32:
                    value = a ** int(b)
                else: raise ValueError('unsupported operator')
            else:
                raise ValueError('unsupported expression')
            if value.numerator.bit_length() > 4096 or value.denominator.bit_length() > 4096:
                raise ValueError('answer too large')
            return value

        return walk(tree.body)
    except (SyntaxError, TypeError, ZeroDivisionError, OverflowError, RecursionError) as exc:
        raise ValueError('invalid rational answer') from exc


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    box = last_box(solution_str)
    try:
        correct = box is not None and rational(box[0]) == rational(ground_truth)
        supported = box is not None
    except ValueError:
        correct, supported = False, False
    return {'score': float(correct), 'acc': float(correct),
            'incorrect_format': float(not supported), 'pred': box[0] if box else ''}


def load(root, split):
    return [json.loads(line) for line in (Path(root) / (split + '.jsonl')).read_text().split('\n') if line.strip()]


def render_prompt(item):
    return item.get('prompt') or PROMPT.format(problem=item['problem'])


def gold_of(item):
    return item['gold']
