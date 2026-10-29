"""Exact replay-family comment templates, with source spans and deterministic bindings."""
from pathlib import Path
import hashlib
import json
import re

SOURCE_REVISION = '11826255077617a46919ce75cadf1f3d53f30dac'
SOURCE_SHA256 = 'afea345e3bc100e6d5588ca14e7269141e8810e1489e60ad64a59cd57360201c'
REPLAY_REVISION = '47c03c73621c4aab2b824b7818681117d662770e'
SOURCE = Path(__file__).resolve().parents[1] / 'p3b' / 'if_functions.py'
FUNCTIONS = {
    'required_word': ('verify_keywords', 'include keywords'),
    'forbidden_word': ('validate_forbidden_words', 'Forbidden Words'),
    'word_frequency': ('verify_keyword_frequency', 'Keyword Frequency'),
    'word_count_interval': ('validate_word_constraint', 'Number Words'),
    'sentence_count': ('verify_sentence_constraint', 'Number Sentences'),
    'paragraph_count': ('verify_paragraph_count', 'Number Paragraphs'),
    'bullet_count': ('verify_bullet_points', 'Number Bullets'),
    'json_only': ('validate_json_format', 'JSON Format'),
    'lowercase_only': ('validate_lowercase', 'All Lowercase'),
    'required_suffix': ('validate_end', 'End Checker'),
}


def inventory():
    raw = SOURCE.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA256:
        raise ValueError('replay template source hash mismatch')
    lines = raw.decode().splitlines()
    records = {}
    for family, (function, label) in FUNCTIONS.items():
        start = next(i for i, line in enumerate(lines) if re.match(r'# ' + re.escape(label) + r'\s*:', line))
        comment = [lines[start][2:].split(':', 1)[1].strip()]
        end = start + 1
        while end < len(lines) and lines[end].startswith('# '):
            comment.append(lines[end][2:]); end += 1
        # Preserve the explicit divider newline; join source-wrapped prose with a space.
        clause = ('\n' if family == 'paragraph_count' else ' ').join(comment)
        records[family] = dict(template_id='open-instruct:' + function + ':comment', function=function,
                               template=clause, source_lines=[start + 1, end],
                               source_comment='\n'.join(lines[start:end]))
    return dict(schema='sp-canonical-templates-v2', source_revision=SOURCE_REVISION,
                source_url='https://github.com/allenai/open-instruct/blob/' + SOURCE_REVISION + '/open_instruct/if_functions.py',
                source_sha256=SOURCE_SHA256, source_license='Apache-2.0',
                replay_dataset='allenai/RLVR-IFeval', replay_revision=REPLAY_REVISION,
                templates=records)


def render(constraints):
    templates = inventory()['templates']
    clauses, bindings = [], []
    for c in constraints:
        family, a = c['family'], c['args']
        template = templates[family]
        # The keyword slots denote a variadic keyword list; a singleton fills the entire list span.
        if family == 'required_word': substitutions = [{'{keyword1}, {keyword2}': a['word']}]
        elif family == 'forbidden_word': substitutions = [{'{forbidden words}': a['word']}]
        elif family == 'word_frequency': substitutions = [{'{word}': a['word'], '{N}': str(a['count'])}]
        elif family == 'word_count_interval':
            substitutions = [{'at least / around / at most': q, '{N}': str(a[k])} for q, k in [('at least', 'min'), ('at most', 'max')]]
        elif family == 'sentence_count':
            substitutions = [{'at least / around / at most': q, '{N}': str(a['count'])} for q in ('at least', 'at most')]
        elif family in ('paragraph_count', 'bullet_count'): substitutions = [{'{N}': str(a['count'])}]
        elif family == 'required_suffix': substitutions = [{'{end phrase}': json.dumps(a['suffix'], ensure_ascii=False)}]
        else: substitutions = [{}]
        for substitution in substitutions:
            clause = template['template']
            for old, new in substitution.items(): clause = clause.replace(old, new)
            clauses.append(clause)
            bindings.append(dict(template_id=template['template_id'], substitutions=substitution, clause=clause))
    return '\n'.join(clauses), bindings


if __name__ == '__main__':
    target = Path(__file__).with_name('templates.json')
    target.write_text(json.dumps(inventory(), indent=2, ensure_ascii=False) + '\n')
