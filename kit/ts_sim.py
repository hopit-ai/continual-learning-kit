"""CPU-only TS rehearsal. Toy logits, genuine checker/mask/loss/step arithmetic."""
import torch

from kit import ts_arms as ts, ts_masks
from kit.nm_common import seal, digest


def train(row, incoming, root, *, missing_checker=False, crash_after=None, resume_step=0):
    shape = ts.settings(row['binding'], row['stage'])
    student = torch.nn.Parameter(torch.tensor(incoming['weights'], dtype=torch.float32))
    teacher = student.detach().clone()
    optimizer = torch.optim.AdamW([student], lr=1e-5, weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(step / 10, 1.))
    base = dict(finqa=('Answer: 42', '42'), contract_nli=('Answer: entailment', 'entailment'),
                math_hard=(r'\boxed{42}', '42'))
    correct, gold = base[row['stage']]
    history, snapshots = [], {'0': dict(incoming)}
    if resume_step:
        state = torch.load(root / 'native' / f'global_step_{resume_step}' / 'cpu-state.pt', weights_only=True)
        with torch.no_grad():
            student.copy_(state['student'])
            teacher.copy_(state['teacher'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        history, snapshots = state['history'], state['snapshots']
        torch.set_rng_state(state['rng'])
    if row['arm'] != 'D':
        ts_masks.supersede(root, resume_step)
    response_mask = torch.ones(shape['task_rows'], 7)
    for update in range(resume_step + 1, 41):
        groups, signals = [], []
        for position in range(1, shape['task_rows'] + 1):
            plan = ts.attempt_plan(row['arm'], row['seed'], row['stage'], update, position, str(position))
            # Update one is intentionally all failure for every M/S chain.
            passed = [] if update == 1 else [0] if position % 3 == 0 else [2, 5] if position % 3 == 1 else []
            group = [ts.Attempt(correct if p['index'] in passed else 'invalid response',
                                (p['index'] + 1, 9), False) for p in plan]
            labels = [ts.check_attempt(row['stage'], a, gold) for a in group]
            if missing_checker:
                labels = labels[:-1]
            signals.append(ts.choose_signal(row['arm'], group, labels, 'Verified worked solution.'))
            groups.append(group)
        if row['arm'] != 'D':
            extras = [dict(nm_position=i + 1, index=str(i + 1)) for i in range(shape['task_rows'])]
            ts_masks.record(root, update, extras, signals, groups)
        logits = student.expand(shape['task_rows'], 7, -1)
        target = teacher.expand_as(logits).log_softmax(-1).detach()
        kl = (target.exp() * (target - logits.log_softmax(-1))).sum(-1)
        old, rollout = torch.zeros_like(response_mask), torch.zeros_like(response_mask)
        task_losses = ts.per_row_loss(kl, response_mask, old, rollout)
        replay_logits = student.expand(shape['replay_rows'], 5, -1)
        replay = -replay_logits.log_softmax(-1)[..., 2].mean(-1)
        masks = torch.tensor([s.mask for s in signals], dtype=torch.float32)
        loss = ts.masked_loss(task_losses, masks, replay)
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_([student], 1.)
        optimizer.step(); scheduler.step()
        teacher.mul_(.99).add_(student.detach(), alpha=.01)
        history.append(dict(update=update, **ts.telemetry(signals, groups, response_mask, task_losses),
                            replay_loss=float(replay.detach().sum()) / 32, loss=float(loss.detach())))
        if update in ts.STEPS:
            weights = student.detach().tolist()
            snapshots[str(update)] = dict(weights=weights, sha256=digest(weights), technical_synthetic=True)
            checkpoint = root / 'native' / f'global_step_{update}'
            checkpoint.mkdir(parents=True, exist_ok=True)
            torch.save(dict(student=student.detach(), teacher=teacher, optimizer=optimizer.state_dict(),
                            scheduler=scheduler.state_dict(), rng=torch.get_rng_state(),
                            history=history, snapshots=snapshots), checkpoint / 'cpu-state.pt')
            (root / 'native/latest_checkpointed_iteration.txt').write_text(str(update))
        if update == crash_after:
            raise RuntimeError('CPU crash after update ' + str(update))
    result = dict(status='COMPLETE', technical_synthetic=True, updates=40,
                  steps=dict(optimizer=40, scheduler=40, ema=40), history=history, exports=snapshots)
    seal(root / 'native-receipt.json', result)
    return result


def base(seed):
    generator = torch.Generator().manual_seed(seed)
    weights = torch.randn(13, generator=generator).tolist()
    return dict(weights=weights, sha256=digest(weights), technical_synthetic=True)
