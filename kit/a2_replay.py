"""P copies I0's realised legal-stage generic positions and ordering, without shuffling."""
from kit.a2_common import *


def transfer(task, source, seed, evaluation):
    from kit.c_common import overlap
    require(len(task) == 1280, 'control schedule must precede replay substitution')
    spec = load(checked(source['maps'][str(seed)]))
    pool = t.read_jsonl(checked(source['pool']))
    by_id = {str(r['id']): r for r in pool}
    require(len(pool) == len(by_id), 'duplicate I0 generic IDs')
    targets = load(checked(source['stored_targets']))
    require(all(str(r['id']) in targets and r['response'] == targets[str(r['id'])]
                for r in pool), 'stored I0 targets changed')
    _, excluded = overlap(pool, evaluation)
    excluded_ids = {str(r['id']) for r in excluded}
    receipt = load(checked(source['overlap_receipt']))
    if set(receipt.get('excluded_ids', [])) != excluded_ids:
        raise NotReady('I0 overlap exclusions not bound to SEC/general-v1')
    count = sum(map(len, spec['slots']))
    # The upstream receipt supplies the realised stream, including repetitions.
    # Walking it only passes excluded entries; it never shuffles or re-runs draws.
    accepted = []
    passed = []
    for index, item_id in enumerate(spec['accepted_order']):
        if len(accepted) == count:
            break
        item_id = str(item_id)
        require(item_id in by_id, 'I0 ordering references an absent target')
        if item_id in excluded_ids:
            passed.append(dict(id=item_id, source_index=index))
        else:
            accepted.append(item_id)
    if len(accepted) != count:
        raise NotReady('copied I0 ordering cannot fill registered map after exclusions')
    rows = list(task)
    ids = iter(accepted)
    for update, slots in enumerate(spec['slots']):
        for position in sorted(slots):
            item_id = next(ids)
            rows[update*32+position-1] = dict(by_id[item_id], p3b_replay=True, gold='')
    return rows, dict(generic_positions=count, task_positions=1280-count,
                      ids=accepted, excluded_ids=sorted(excluded_ids), passed_over=passed,
                      copied_skipped_draws=spec['skipped_draws'], slots=spec['slots'])


def applicable(source):
    return any(sum(map(len, load(checked(source['maps'][str(seed)]))['slots']))
               for seed in SEEDS)
