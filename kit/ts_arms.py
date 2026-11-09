"""Unregistered teacher-source arms. Incumbent D modules remain independent.

Only attempt zero enters the objective. Auxiliary request seeds are explicit,
domain-separated, and independent of outcomes and generation order.
"""
from contextlib import contextmanager
from dataclasses import dataclass
from copy import deepcopy
import random

import numpy as np
import torch

from kit import sdft_objective as objective
from kit.nm_common import KIT, SEEDS, STEPS, digest, need, sha


ARMS = ('D', 'M', 'S', 'H')
STAGES = ('finqa', 'contract_nli', 'math_hard')
BINDINGS = ('I0', 'CR')
POSITIVE_FEEDBACK = 'The attempted answer passed verification.'


def settings(binding, stage):
    need(binding in BINDINGS and stage in STAGES, 'unknown TS recipe/stage')
    number = STAGES.index(stage)
    long = binding == 'CR' or stage == 'contract_nli'
    task, replay = (28, 4) if binding == 'I0' else (32, (8, 16, 24)[number])
    return dict(task_rows=task, replay_rows=replay, rows_per_update=task + replay,
                divisor=32, updates=40, exports=list(STEPS), lr=1e-5,
                warmup_steps=10, weight_decay=.01, grad_clip=1., ema_rate=.01,
                caps=dict(prompt=10240 if long else 2048, response=2048,
                          teacher=14336 if long else 6144))


def attempt_plan(arm, seed, stage, update, position, question_id):
    need(arm in ARMS and seed in SEEDS and stage in STAGES, 'unknown TS identity')
    need(type(update) is int and 1 <= update <= 40 and type(position) is int and position > 0,
         'invalid scheduled position')
    # None means the exact incumbent stream, without a new seed or generator.
    plan = [dict(index=0, seed=None, loss_bearing=True)]
    for index in range(1, 1 if arm == 'D' else 8):
        identity = ['ts-aux-v1', seed, stage, update, position, str(question_id), index]
        value = int(digest(identity)[:15], 16)
        plan.append(dict(index=index, seed=value, loss_bearing=False))
    need(len({p['seed'] for p in plan}) == len(plan), 'auxiliary seed collision')
    return plan


@contextmanager
def isolated_rng():
    """Protect caller RNG state even when an auxiliary callback fails.

    Generation backends must also use explicit per-request seeds. Saving driver
    RNG state alone cannot isolate a remote inference engine's sampling stream.
    """
    python_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng(devices=[]):
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def generate_group(plan, incumbent, auxiliary):
    """CPU/backend protocol. Never short circuit on success or resample failures."""
    attempts = [incumbent()]
    with isolated_rng(), torch.no_grad():
        for request in plan[1:]:
            attempts.append(auxiliary(request['seed']))
    need(len(attempts) == len(plan), 'incomplete attempt group')
    return attempts


@dataclass(frozen=True)
class Attempt:
    text: str
    token_ids: tuple
    truncated: bool
    finish_reason: str | None = None

    def __post_init__(self):
        need(isinstance(self.text, str) and isinstance(self.token_ids, tuple)
             and 0 < len(self.token_ids) <= 2048, 'invalid attempt response/cap')
        need(all(type(x) is int and x >= 0 for x in self.token_ids), 'invalid response tokens')
        need(type(self.truncated) is bool, 'missing finish telemetry')


def check_attempt(stage, attempt, gold):
    """Only frozen binary labels leave the checker. Gold never enters feedback.

    FinQA has its inherited finish gate. Legal and rational MATH retain their
    existing parser semantics, including correctly formatted capped answers.
    """
    need(stage in STAGES and isinstance(attempt, Attempt), 'invalid checker request')
    if stage == 'finqa':
        from kit.nm_entry import compute_score
        return int(compute_score(stage, attempt.text, gold, {'truncated': attempt.truncated})['score'])
    if stage == 'contract_nli':
        from kit.c_reward import compute_score
        return int(compute_score(stage, attempt.text, gold)['score'])
    from kit.beds.math_screen import compute_score
    return int(compute_score('math_screen', attempt.text, gold)['score'])


def checker_hashes():
    names = ('beds/finqa.py', 'beds/rewards.py', 'c_reward.py',
             'resources/c/legal_headroom.py', 'beds/math_screen.py')
    return {name: sha(KIT / name) for name in names}


@dataclass(frozen=True)
class Signal:
    mask: int
    context: str | None
    route: str
    sibling_index: int | None
    labels: tuple
    reason: str | None = None
    detail: str | None = None


def masked_signal(reason, labels=(), detail=None):
    return Signal(0, None, 'model-output-error', None, tuple(labels), reason, detail)


def normalized_text(text):
    return ' '.join(text.split())


def near_duplicate(left, right):
    """Linear-time character-trigram Jaccard, on whitespace-normalised text."""
    a = {left[i:i + 3] for i in range(max(1, len(left) - 2))}
    b = {right[i:i + 3] for i in range(max(1, len(right) - 2))}
    return len(a & b) / len(a | b) >= .9


def distinct_sibling(attempts, labels):
    first = attempts[0]
    return next((i for i in range(1, len(attempts)) if labels[i]
                 and attempts[i].token_ids != first.token_ids
                 and normalized_text(attempts[i].text) != normalized_text(first.text)), None)


def choose_signal(arm, attempts, labels, demonstration):
    need(arm in ARMS and len(attempts) == len(labels) == (1 if arm == 'D' else 8),
         'complete fixed attempt group/checker output required')
    need(all(type(label) is int and label in (0, 1) for label in labels), 'checker output must be binary')
    need(isinstance(demonstration, str) and demonstration.strip(), 'verified D solution missing')
    # A repeated copy of the target is never privileged sibling context.
    sibling = distinct_sibling(attempts, labels)
    mask = int(any(labels)) if arm in ('M', 'S') else 1
    if arm in ('D', 'M'):
        return Signal(mask, demonstration, 'demonstration', None, tuple(labels))
    if sibling is not None:
        return Signal(mask, attempts[sibling].text, 'sibling', sibling, tuple(labels))
    if not labels[0] and any(labels[1:]):
        sibling = next(i for i in range(1, len(labels)) if labels[i])
        return Signal(mask, attempts[sibling].text, 'nondistinct-success', sibling, tuple(labels))
    if arm == 'H':
        return Signal(1, demonstration, 'fallback', None, tuple(labels))
    if not mask:
        return Signal(0, None, 'empty', None, tuple(labels))
    if labels[0] != 1:
        return masked_signal('unhandled_signal_selection', labels)
    return Signal(1, POSITIVE_FEEDBACK, 'self-success', None, tuple(labels))


def teacher_inputs(tokenizer, prompts, signals, responses, response_mask, caps, *, student_prompts=None):
    """D's literal wrapper for every context route, plain prompt for empty S.

    Padding is left-sided. The loss-bearing completion follows unchanged.
    Every prompt is checked before padding. Overlength contexts are errors.
    """
    need(len(prompts) == len(signals) == len(responses) == len(response_mask) > 0,
         'teacher batch geometry differs')
    need(responses.shape == response_mask.shape and responses.shape[-1] <= caps['response'],
         'response cap/geometry differs')
    eligible = response_mask.new_tensor([s.mask for s in signals])
    need(bool(((response_mask == 0) | (response_mask == 1)).all())
         and bool(((response_mask.sum(-1) > 0) | (eligible == 0)).all()), 'invalid response mask')
    frozen = deepcopy(prompts) if student_prompts is None else student_prompts
    need(len(frozen) == len(prompts), 'student prompt snapshot geometry differs')
    ids = []
    for index, (prompt, signal) in enumerate(zip(prompts, signals)):
        need(list(prompt) == list(frozen[index]), 'student prompt changed after incumbent generation')
        student = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        need(0 < len(tokenizer(student, add_special_tokens=False)['input_ids']) <= caps['prompt'],
             'student prompt over cap; truncation forbidden')
        auxiliary_context = signal.route in ('sibling', 'nondistinct-success', 'self-success')
        if auxiliary_context:
            # The incumbent draw used the frozen raw prompt before any signal
            # existed. Only the teacher copy receives the context. Short legal
            # labels and table numbers may legitimately occur in the question.
            messages = deepcopy(list(prompt))
            need(messages and messages[-1]['role'] == 'user'
                 and all(isinstance(m['content'], str) for m in messages), 'invalid student message structure')
            messages[-1]['content'] = objective.TEACHER_TEMPLATE.substitute(
                orig_content=messages[-1]['content'], output_text=signal.context)
        else:
            messages = objective.teacher_messages(prompt, signal.context) if signal.context is not None else prompt
        try:
            rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            tokens = tokenizer(rendered, add_special_tokens=False)['input_ids']
        except (ValueError, TypeError, RuntimeError, UnicodeError):
            if not auxiliary_context:
                raise
            signals[index] = masked_signal('auxiliary_teacher_render_error', signal.labels)
            tokens = tokenizer(student, add_special_tokens=False)['input_ids']
        if auxiliary_context and not 0 < len(tokens) <= caps['teacher']:
            signals[index] = masked_signal('auxiliary_teacher_prompt_over_cap', signal.labels)
            tokens = tokenizer(student, add_special_tokens=False)['input_ids']
        need(0 < len(tokens) <= caps['teacher'], 'teacher prompt over cap; truncation forbidden')
        need(list(prompt) == list(frozen[index]), 'teacher construction mutated student prompt')
        ids.append(tokens)
    width = max(map(len, ids))
    padded = torch.full((len(ids), width), tokenizer.pad_token_id or 0, dtype=torch.long)
    mask = torch.zeros_like(padded, dtype=response_mask.dtype)
    for i, tokens in enumerate(ids):
        padded[i, -len(tokens):] = torch.tensor(tokens)
        mask[i, -len(tokens):] = 1
    attention = torch.cat((mask, response_mask.cpu()), -1)
    return dict(teacher_input_ids=torch.cat((padded, responses.cpu()), -1),
                teacher_attention_mask=attention,
                teacher_position_ids=(attention.cumsum(-1) - 1).clamp(min=0),
                self_distillation_mask=torch.tensor([s.mask for s in signals], dtype=torch.float32),
                teacher_prompt_lengths=list(map(len, ids)))


def per_row_loss(kl, response_mask, old_log_probs, rollout_log_probs):
    need(kl.shape == response_mask.shape == old_log_probs.shape == rollout_log_probs.shape,
         'task loss geometry differs')
    mask = objective.loss_mask(response_mask)
    count = mask.sum(-1).clamp(min=1)
    weights = objective.reference_importance_weights(old_log_probs, rollout_log_probs)
    return (kl * mask).sum(-1) / count * (weights * mask).sum(-1) / count


def masked_loss(task_losses, masks, replay_losses):
    need(task_losses.ndim == masks.ndim == replay_losses.ndim == 1 and task_losses.shape == masks.shape,
         'loss vectors differ')
    need(bool(((masks == 0) | (masks == 1)).all()), 'nonbinary task mask')
    need(bool(torch.isfinite(task_losses).all() and torch.isfinite(replay_losses).all()), 'nonfinite loss')
    return ((task_losses * masks.detach()).sum() + replay_losses.sum()) / 32


def telemetry(signals, groups, response_mask, task_losses):
    need(len(signals) == len(groups) == len(response_mask) == len(task_losses), 'telemetry geometry differs')
    masks = response_mask.new_tensor([s.mask for s in signals])
    loss_tokens = objective.loss_mask(response_mask) * masks[:, None]
    success = [any(s.labels) for s in signals]
    siblings = [distinct_sibling(g, s.labels) is not None for g, s in zip(groups, signals)]
    comparisons = [(normalized_text(g[0].text), normalized_text(a.text))
                   for g, s in zip(groups, signals) if s.reason is None for a in g[1:]]
    near_duplicates = sum(near_duplicate(a, b) for a, b in comparisons)
    reasons = sorted({s.reason for s in signals if s.reason})
    return dict(attempted_rows=len(signals), attempted_trajectories=sum(map(len, groups)),
                eligible_rows=sum(s.mask for s in signals), success_rows=sum(success),
                distinct_sibling_rows=sum(siblings),
                nondistinct_success_context=sum(s.route == 'nondistinct-success' for s in signals),
                near_duplicate_pairs=near_duplicates, sibling_pairs=len(comparisons),
                near_duplicate_excluded_pairs=sum(len(g) - 1 for g, s in zip(groups, signals) if s.reason is not None),
                near_duplicate_rate=near_duplicates / len(comparisons) if comparisons else None,
                model_output_masked_rows=sum(s.reason is not None for s in signals),
                **{'finish_reason_' + reason: sum(a.finish_reason == reason for g in groups for a in g)
                   for reason in ('stop', 'length', 'abort')},
                **{'masked_reason_' + reason: sum(s.reason == reason for s in signals) for reason in reasons},
                singleton_success_count=sum(sum(s.labels) == 1 for s in signals),
                self_only_success_count=sum(sum(s.labels) == 1 and s.labels[0] == 1 for s in signals),
                q8=sum(success) / len(signals) if signals else None,
                q7=sum(siblings) / len(signals) if signals else None,
                loss_bearing_tokens=int(loss_tokens.sum()),
                actual_task_loss_mass=float((task_losses.detach() * masks).sum()) / 32,
                empty_batches=int(not any(s.mask for s in signals)),
                capped_outputs=sum(a.truncated for group in groups for a in group))
