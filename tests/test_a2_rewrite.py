"""R's frozen prompt-only boundary, exact arithmetic and raw-fallback semantics."""
import copy
import json
from decimal import Decimal
from pathlib import Path

import pytest
from kit.a2_common import digest, canonical, REGISTRATION
from kit import a2_rewrite as r
from kit.a2_sim import fixture
from kit.a2_common import checked, load
from kit.beds import finqa


@pytest.fixture(scope='module')
def traces(tmp_path_factory):
    cfg = fixture(tmp_path_factory.mktemp('r-parser')/'input', 'R')
    data = load(checked(cfg['inputs']))
    items = load(checked(data['slice']['2026-01']))
    return [i for i in items[:2]]


def parsed(item):
    prompt = finqa.render_prompt(item)
    raw = 'Verified arithmetic.\n  Answer: '+item['qa']['answer']+' percentage points  '
    return prompt, raw, r.parse(prompt, raw, True, digest(prompt.encode()))


@pytest.mark.parametrize('index', [0, 1])
def test_exact_parser_schema_verification_and_renderer(traces, index):
    prompt, raw, trace = parsed(traces[index])
    assert list(trace) == ['family', 'question', 'cells', 'response', 'answer_line']
    assert list(trace['cells'][0]) == ['id', 'statement', 'source', 'label', 'period',
        'duration_months', 'period_end', 'displayed_value', 'unit', 'signed_value', 'usd_value']
    assert trace['cells'][0]['id'] == 'e1:r1:c1'
    assert trace['cells'][0]['period_end'] == '2025-12-31'
    assert trace['answer_line'] == raw.split('\n')[-1]
    binding = r.unique_binding(trace)
    generated = canonical(dict(family=trace['family'], cells={k: c['id'] for k, c in binding.items()})).decode()
    assert r.verify(trace, generated) == binding
    text = r.render(trace, binding)
    assert text.startswith(r.PROCEDURES[trace['family']]+'\nBindings: a = label=')
    assert text.endswith(trace['answer_line']) and not text.endswith('\n')
    assert 'e1:r1' not in text and 'Application: 100*' in text
    assert r.messages(trace)[1]['content'] == 'Convert this admitted trace using the fixed schema:\n'+canonical(trace).decode()
    assert not r.messages(trace)[0]['content'].endswith('\n')


def test_prompt_only_and_midline_second_table(traces):
    item = traces[0]
    prompt, raw, trace = parsed(item)
    assert '. |  | 12 Months' in prompt
    for key in ('operands', 'concept', 'context_id', 'formula', 'unrounded_gold'):
        altered = copy.deepcopy(item)
        altered['sec'][key] = 'forbidden hidden value'
        assert r.parse(finqa.render_prompt(altered), raw, True, digest(prompt.encode())) == trace
    assert 'item' not in trace and set(trace) == {'family', 'question', 'cells', 'response', 'answer_line'}


@pytest.mark.parametrize('text,signed,usd', [
    ('($ 1,234.50)[2]', '-1234.5', '-1234500'),
    ('$ 0.00', '0', '0'), ('(0)', '0', '0'),
    ('(10', None, None), ('10)', None, None), ('-10', None, None),
    ('1.25', '1.25', '1250'),
    ('1234567890123456789012345678901234567890.12345',
     '1234567890123456789012345678901234567890.12345',
     '1234567890123456789012345678901234567890123.45')])
def test_exact_decimal_scales_and_sign_grammar(text, signed, usd):
    assert r.number(text, 'thousands of US dollars') == (signed, usd)


@pytest.mark.parametrize('text,want', [
    ('9 Months Ended Sept. 30, 2025', (9, '2025-09-30')),
    ('1 month ended January 31, 2026', (1, '2026-01-31')),
    ('12 Months Ended Dec 31, 2025', (12, '2025-12-31')),
    ('13 Months Ended Dec. 31, 2025', (None, None)),
    ('Fiscal year 2025', (None, None))])
def test_period_parsing_without_hidden_context(text, want):
    assert r.period(text) == want


@pytest.mark.parametrize('failure', ['hash', 'missing', 'admission', 'question', 'pipe', 'answer',
    'duplicate', 'extra', 'truncated', 'overflow', 'wrong_cells', 'percent', 'arithmetic'])
def test_counted_fallback_no_retry_or_gold(traces, failure):
    prompt, raw, trace = parsed(traces[0])
    admission = True
    expected_hash = digest(prompt.encode())
    calls = []
    binding = r.unique_binding(trace)
    obj = dict(family=trace['family'], cells={k: c['id'] for k, c in binding.items()})
    generated = canonical(obj).decode()
    if failure == 'hash': expected_hash = '0'*64
    if failure == 'missing': prompt = None
    if failure == 'admission': admission = False
    if failure == 'question': prompt = prompt.replace('round only the final result.', 'round everything.')
    if failure == 'pipe': prompt = prompt.replace('| 200 |', '| 200|bad |')
    if failure == 'answer': raw = 'No final numeric answer.'
    if failure == 'duplicate': generated = generated[:-1]+',"family":"fcf_margin_change"}'
    if failure == 'extra': generated = generated[:-1]+',"extra":1}'
    if failure == 'wrong_cells': obj['cells']['a'] = obj['cells']['b']; generated = canonical(obj).decode()
    if failure == 'percent': raw = raw.replace('percentage points', '%')
    if failure == 'arithmetic': raw = 'Answer: 99.0'
    if failure not in ('hash', 'missing'): expected_hash = digest(prompt.encode())
    def generate(messages):
        calls.append(messages)
        return dict(response=generated, finish_reason='length' if failure == 'truncated' else 'stop')
    result = r.rewrite(prompt, raw, admission, expected_hash, generate,
        lambda prompt, response: failure != 'overflow')
    assert not result['accepted'] and result['response'] == raw and result['fallback_reason']
    assert len(calls) <= 1


def test_latest_common_period_and_ambiguity_are_fail_closed(traces):
    _, _, trace = parsed(traces[0])
    repeated = copy.deepcopy(trace['cells'][0])
    repeated['id'] = 'e3:r1:c1'
    trace['cells'].append(repeated)
    with pytest.raises(r.Fallback, match='ambiguous_binding'):
        r.unique_binding(trace)


def test_exact_prompt_matches_frozen_registration():
    source = (r.RES/'registrations.md').read_bytes()
    assert digest(source) == REGISTRATION == '276f4f6e04c3d0c228a47e04f80fa1d8de22335a5af5aee1ee68dabab98bf789'
    section = source.decode().split('**Exact rewrite prompt.**')[1].split('```text\n')[1].split('\n```')[0]
    assert digest(section.encode()) == 'f2a03f62408e094dc3712503afc8cbf89c655e41244614f9a317f5d88889ac85'
    assert digest((r.RES/'rewrite-system.txt').read_bytes()) == digest(section.encode())
    assert (r.RES/'rewrite-system.txt').read_text() == section


PRIVATE_REGISTRATION = (Path(__file__).resolve().parents[1]/
                        'docs/phase4/a2-registrations-20261108/registrations.md')


@pytest.mark.skipif(not PRIVATE_REGISTRATION.exists(), reason='private registration document is not exported')
def test_private_registration_matches_shipped_copy():
    assert PRIVATE_REGISTRATION.read_bytes() == (r.RES/'registrations.md').read_bytes()


def test_multiline_answer_match_cannot_be_a_single_preserved_line(traces):
    prompt, raw, _ = parsed(traces[0])
    raw = raw.replace('Answer: ', 'Answer:\n')
    with pytest.raises(r.Fallback, match='answer_match_spans_lines'):
        r.parse(prompt, raw, True, digest(prompt.encode()))
