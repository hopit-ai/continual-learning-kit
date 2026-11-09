"""Two-phase rollout adapter. Finish every incumbent draw before auxiliary work."""
from copy import deepcopy

import numpy as np

from kit import ts_arms as ts
from kit.nm_common import need


def install(manager, config):
    """Wrap a TS-owned manager instance, leaving the incumbent call untouched.

    The auxiliary agent uses request-local seeds. Its outputs are carried as
    plain records and never concatenated into the actor's response tensor.
    """
    original = manager.generate_sequences

    def generate(prompts):
        saved = deepcopy(prompts)
        result = original(prompts)
        if config['arm'] == 'D' or saved.meta_info.get('validate', False):
            return result
        snapshots = np.empty(len(saved), dtype=object)
        snapshots[:] = [deepcopy(p) for p in saved.non_tensor_batch['raw_prompt']]
        result.non_tensor_batch['ts_student_prompt'] = snapshots
        extras = saved.non_tensor_batch['extra_info']
        indices = [i for i, extra in enumerate(extras) if not extra.get('m3_replay')]
        need(indices, 'TS generation batch has no task rows')
        auxiliary = saved[indices]
        auxiliary.non_tensor_batch['agent_name'] = np.array(['ts_auxiliary'] * len(indices), dtype=object)
        plans = np.empty(len(indices), dtype=object)
        for i, index in enumerate(indices):
            extra = extras[index]
            plans[i] = ts.attempt_plan(config['arm'], config['seed'], config['stage'],
                extra['nm_update'], extra['nm_position'], extra['index'])[1:]
        auxiliary.non_tensor_batch['ts_plan'] = plans
        with ts.isolated_rng():
            output = original(auxiliary)
        need(len(output) == len(indices) and 'ts_attempts' in output.non_tensor_batch,
             'auxiliary generation output missing')
        groups = np.empty(len(result), dtype=object)
        groups[:] = None
        for i, index in enumerate(indices):
            group = output.non_tensor_batch['ts_attempts'][i]
            need(isinstance(group, list) and len(group) == 7, 'seven auxiliary outputs required')
            groups[index] = group
        result.non_tensor_batch['ts_attempts'] = groups
        return result

    manager.generate_sequences = generate


def record(tokenizer, token_ids, eos_token_id, finish_reason=None):
    """Preserve the exact response and the pinned naive manager's finish gate.

    The raw vLLM reason is telemetry. The incumbent reward gate tests membership
    of tokenizer.eos_token_id in the valid response, including alternate stop
    tokens. The valid response is decoded in full.
    """
    ids = tuple(int(x) for x in token_ids)
    return ts.Attempt(tokenizer.decode(ids, skip_special_tokens=True), ids, eos_token_id not in ids, finish_reason)


def checked_signal(tokenizer, stage, arm, ids, finish_reason, records, gold, demonstration, reward):
    """Model-origin edge cases mask one scheduled row and retain a named reason.

    Missing records or request identities are validated by the caller as
    infrastructure errors. Checker/parser exceptions produce named zero-mask rows.
    """
    need(stage in ts.STAGES and arm in ('M', 'S', 'H'), 'invalid signal request')
    need(isinstance(demonstration, str) and demonstration.strip(), 'verified D solution missing')
    need(isinstance(records, list) and len(records) == 7 and
         all(isinstance(r, dict) and 'token_ids' in r and 'finish_reason' in r for r in records),
         'invalid auxiliary record schema')
    group = []
    labels = [0] * 8
    try:
        group = [record(tokenizer, ids, tokenizer.eos_token_id, finish_reason)]
        group += [record(tokenizer, r['token_ids'], tokenizer.eos_token_id, r.get('finish_reason')) for r in records]
        if any(a.finish_reason not in ('stop', 'length', 'abort') for a in group):
            return group, ts.masked_signal('unknown_finish_reason', labels)
        labels = [ts.check_attempt(stage, a, gold) for a in group]
        if float(reward) != labels[0]:
            return group, ts.masked_signal('incumbent_checker_disagreement', labels)
        return group, ts.choose_signal(arm, group, labels, demonstration)
    except MemoryError:
        raise
    except Exception as exc:
        # Retain raw finish telemetry even if token validation/decoding failed.
        # Placeholder tokens are used only for zero-labelled diagnostics. The
        # actor still receives its original, unmodified response tensor.
        if len(group) != 8:
            group = []
            for raw in [dict(token_ids=ids, finish_reason=finish_reason), *records]:
                tokens = raw.get('token_ids')
                valid = tokens if isinstance(tokens, (list, tuple)) else ()
                group.append(ts.Attempt('', (0,), tokenizer.eos_token_id not in valid, raw.get('finish_reason')))
        # The exception type is carried so a text-only return names the cause.
        return group, ts.masked_signal('invalid_model_output_or_checker_error', labels, type(exc).__name__)
