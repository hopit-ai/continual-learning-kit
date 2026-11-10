"""Gold-free parser coverage on shipped SEC examples via the real 4A generator."""
from collections import Counter
from pathlib import Path

import pytest
from kit import a2_rewrite as r, next_sec_core as sec
from kit.a2_common import canonical, digest, load, atomic, KIT
from kit.beds import finqa


@pytest.mark.parametrize('month,expected', [
    ('2026-07', dict(total=61, parsed=60, bound=60, extra_or_malformed_table=1)),
    ('2026-08', dict(total=80, parsed=77, bound=76, extra_or_malformed_table=3, period_or_label_binding=1)),
    ('2026-09', dict(total=52, parsed=51, bound=51, extra_or_malformed_table=1)),
])
def test_real_generator_prompts_parse_and_unique_binding(month, expected, tmp_path, monkeypatch):
    items = [i for i in load(KIT/f'resources/next/sec/batch-{month}.json')
             if i['sec']['template'] in r.FORMULAS]
    counts = Counter(); header_counts = Counter(); failure_ids = {}
    shapes = set(); statements = set(); families = set(); units = set()
    company_lines = footnotes = numeric_footnotes = label_differences = 0
    for item in items:
        source = item['sec']
        excerpts = {}
        for excerpt in source['excerpts']:
            table = dict(excerpt, statement=dict(unit=excerpt['unit'], footnotes=excerpt['footnotes']),
                         xml=dict(facts=[]))
            excerpts[excerpt['r_page']] = table
        # The original XBRL cache is not shipped. Bind the already audited cells
        # at that boundary; run v4_make_item's real pre-lines/table/footnote code.
        refs = [(excerpts[o['r_page']], o['table_row'], o['table_column']-1) for o in source['operands']]
        def operand(table, row, col):
            return next(o for o in source['operands'] if
                        (o['r_page'], o['table_row'], o['table_column']) == (table['r_page'], row, col+1))
        monkeypatch.setattr(sec, 'v4_operand', operand)
        query = item['qa']['question'].removesuffix(' Answer in percentage points (signed), rounded to one decimal place; round only the final result.')
        regenerated = sec.v4_make_item(dict(record=source, screened_document=source['screened_document']),
                                      source['template'], query, source['formula'], refs)
        prompt = finqa.render_prompt(regenerated)
        assert prompt == finqa.render_prompt(item)
        assert '(CIK ' in prompt
        company_lines += 1
        families.add(source['template'])
        for excerpt in source['excerpts']:
            statements.add(excerpt['kind']); units.add(excerpt['unit'])
            shapes.add(len(excerpt['table'][0])-1)
            for heading in excerpt['table'][0][1:]:
                header_counts[heading] += 1
                # Exercise every shipped heading, even in an extra-table fallback.
                duration, end = r.period(heading)
                assert duration is not None and end is not None
            footnotes += len(excerpt['footnotes'])
            numeric_footnotes += sum('[' in str(v) for row in excerpt['table'] for v in row)
        label_differences += sum(o['label'] != o.get('shown_label', o['label']) for o in source['operands'])
        counts['total'] += 1
        raw = 'Unedited admitted fixture response.\nAnswer: 0.0'
        reason = None
        try:
            trace = r.parse(prompt, raw, True, digest(prompt.encode()))
            counts['parsed'] += 1
            # Verify every cell's visible source and stable enumeration, including
            # the second table that the generator joins on the marker's line.
            expected_cells = [(e, row, col, excerpt) for e, excerpt in enumerate(source['excerpts'], 1)
                              for row in range(1, len(excerpt['table'])) for col in range(1, len(excerpt['table'][0]))]
            assert len(trace['cells']) == len(expected_cells)
            for cell, (e, row, col, excerpt) in zip(trace['cells'], expected_cells):
                assert cell['id'] == f'e{e}:r{row}:c{col}'
                assert cell['label'] == excerpt['table'][row][0]
                assert cell['displayed_value'] == excerpt['table'][row][col]
                assert cell['statement'] == excerpt['kind']
                assert cell['period'] == excerpt['table'][0][col]
            r.unique_binding(trace)
            counts['bound'] += 1
        except r.Fallback as error:
            reason = str(error); counts[reason] += 1; failure_ids[item['id']] = reason
        if reason:
            calls = []
            def generate(messages):
                calls.append(messages)
                # Verification must reject an impossible label/period binding.
                return dict(response=canonical(dict(family=source['template'], cells={})).decode(), finish_reason='stop')
            result = r.rewrite(prompt, raw, True, digest(prompt.encode()), generate, lambda *a: True)
            assert not result['accepted'] and result['response'] == raw
            assert result['fallback_reason'] == reason and len(calls) <= 1
    report = dict(month=month, counts=dict(counts), headers=dict(header_counts),
                  numeric_column_counts=sorted(shapes), statements=sorted(statements), units=sorted(units),
                  company_lines=company_lines, footnotes=footnotes, numeric_footnote_cells=numeric_footnotes,
                  shown_label_differences=label_differences, failure_ids=failure_ids,
                  source='shipped examples; real v4_make_item, archived audited operands; no model output',
                  generator_sha256=digest((KIT/'next_sec_core.py').read_bytes()))
    atomic(tmp_path/f'sec-prompt-coverage-{month}.json', report)
    assert counts == Counter(expected)
    assert statements == {'cash_flow', 'income'} and families == set(r.FORMULAS)
    assert numeric_footnotes


@pytest.mark.parametrize('family', list(r.FORMULAS))
def test_real_generator_footnotes_and_shown_label_variants(family, monkeypatch):
    import copy
    item = copy.deepcopy(next(i for i in load(KIT/'resources/next/sec/batch-2026-07.json')
                              if i['sec']['template'] == family))
    source = item['sec']; excerpts = {}
    for excerpt in source['excerpts']:
        excerpt['footnotes'] = ['[7] The displayed figures preserve the signed USD scale.']
        for row in excerpt['table'][1:]:
            row[0] += ' [7]'
        excerpts[excerpt['r_page']] = dict(excerpt,
            statement=dict(unit=excerpt['unit'], footnotes=excerpt['footnotes']), xml=dict(facts=[]))
    def operand(table, row, col):
        return next(o for o in source['operands'] if
                    (o['r_page'], o['table_row'], o['table_column']) == (table['r_page'], row, col+1))
    monkeypatch.setattr(sec, 'v4_operand', operand)
    refs = [(excerpts[o['r_page']], o['table_row'], o['table_column']-1) for o in source['operands']]
    query = item['qa']['question'].removesuffix(' Answer in percentage points (signed), rounded to one decimal place; round only the final result.')
    generated = sec.v4_make_item(dict(record=source, screened_document=source['screened_document']),
                               family, query, source['formula'], refs)
    prompt = finqa.render_prompt(generated)
    assert len(generated['pre_text']) == 2 and '[7] The displayed figures' in prompt
    assert '[7] The displayed figures' in ' '.join(generated['post_text'])
    trace = r.parse(prompt, 'Answer: 0.0', True, digest(prompt.encode()))
    binding = r.unique_binding(trace)
    assert {c['statement'] for c in binding.values()} == {'cash_flow', 'income'}
    assert all(c['label'].endswith(' [7]') for c in binding.values())
    assert '[7]' not in trace['question']
    # Unregistered case/word changes must still fail rather than be repaired.
    changed = prompt.replace(' [7] |', ' different visible label |')
    parsed = r.parse(changed, 'Answer: 0.0', True, digest(changed.encode()))
    with pytest.raises(r.Fallback, match='period_or_label_binding'):
        r.unique_binding(parsed)
