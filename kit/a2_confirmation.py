"""Dormant confirmation support. Refuse to open October-November before release."""
from datetime import date
from kit.a2_common import *
from kit.a2_bind import batch, ledger


def bind_confirmation(cfg, doc, out):
    if not cfg.get('allow_confirmation'):
        raise NotReady('confirmation not released; October-November remains unread')
    if not doc['technical_synthetic'] and date.today() < date(2026, 12, 15):
        raise NotReady('confirmation build/release window has not opened')
    # Check the frozen terminal identity before opening the release receipt.
    freeze_doc = load(Path(out)/'frozen-june.json')
    require(freeze_doc['frozen_before_confirmation'], 'June systems not frozen')
    value = load(checked(cfg.get('confirmation')))
    require(value.get('schema') == 'a2-confirmation.v1' and
            value.get('cutoff') == '2026-12-14' and
            '2026-12-15' <= value.get('build_date', '') <= '2026-12-31', 'confirmation protocol dates')
    require(value.get('frozen_june_sha256') == s.file_hash(Path(out)/'frozen-june.json'),
            'confirmation release is not bound to frozen June systems')
    audit = load(checked(value.get('audit')))
    require(audit.get('source_disjoint') is True and audit.get('fixed_exclusions') is True and
            audit.get('audit_verified') is True and audit.get('release_controls') is True,
            'confirmation audit/exclusions/release invalid')
    require(set(value['slice']) == {'2026-10', '2026-11'}, 'confirmation month set')
    used = {i['id'] for month in MONTHS for i in batch(doc, month)}
    supply = {}
    for month in ('2026-10', '2026-11'):
        rows = load(checked(value['slice'][month]))
        require(not used.intersection(i['id'] for i in rows), 'confirmation item overlap')
        require(all(i['sec']['accepted'].startswith(month) and i['sec']['template'] in FAMILIES
                    for i in rows), 'confirmation chronology/family')
        holdouts = [i for i in rows if i['sec']['issuer_holdout']]
        supply[month] = dict(questions=len(rows), holdouts=len(holdouts),
                             components=len({i['sec']['issuer_group'] for i in holdouts}))
    result = dict(value, supply=supply, terminal_june_frozen=True, audit_verified=True)
    if any(v['questions'] < 40 or v['holdouts'] < 10 or v['components'] < 5 for v in supply.values()):
        raise NotReady('insufficient confirmation evidence')
    bind(Path(out)/'confirmation-binding.json', result)
    return result


def prepare(cfg, doc, out, attempt):
    from kit.a2_read import confirmation_plan
    from kit.a2_pipeline import state, members, tok_for, model_identity
    from kit.a4_memory import build
    value = load(Path(out)/'confirmation-binding.json')
    pool = ledger(doc, 6)
    tok = tok_for(doc)
    specs = []
    for key in confirmation_plan(doc['arm']):
        kind, model, seed, n, panel, mode = key
        record = state(out, doc, model, seed, 6)
        if panel in ('2026-10', '2026-11'):
            rows = load(checked(value['slice'][panel]))
            data = [dict(id=i['id'], panel=panel, item=i,
                         prompt=build(i, pool, tok)['prompt'], context={})
                    for i in rows if i['sec']['issuer_holdout']]
        else:
            data = members(out, doc, panel, mode, True)
        specs.append(dict(key=key, record=record, model=model_identity(record, doc), data=data))
    atomic(attempt/'prepared.json', specs)
    return [attempt/'prepared.json']
