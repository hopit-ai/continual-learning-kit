"""Opt-in replay CE. D's ordinary task loss stays in sdft_actor.update_policy."""
import torch
from kit.sdft_actor import _model_forward


def mixed_loss(task, replay): return (28 * task.mean() + 4 * replay.mean()) / 32


def replay_loss(model, inputs, target_ids):
    """One sequence: frozen base labels including EOS, token mean, no IS correction."""
    from torch.utils.checkpoint import checkpoint
    width = inputs['responses'].shape[1]
    prompt = inputs['input_ids'][:, :-width]
    target = torch.tensor(target_ids, dtype=torch.long, device=prompt.device)[None, :]
    mask = torch.cat((inputs['attention_mask'][:, :-width], torch.ones_like(target)), -1)
    values = dict(input_ids=torch.cat((prompt, target), -1), attention_mask=mask,
                  position_ids=(mask.cumsum(-1)-1).clamp(min=0))
    def head(hidden, weight):
        hidden = hidden[:, -target.shape[1]-1:-1]
        total = hidden.new_zeros((), dtype=torch.float32)
        def ce(h, w, y): return torch.nn.functional.cross_entropy((h @ w.T).float().reshape(-1, w.shape[0]), y.reshape(-1), reduction='sum')
        for start in range(0, target.shape[1], 256):
            total = total + checkpoint(ce, hidden[:,start:start+256], weight, target[:,start:start+256], use_reentrant=False)
        return total / target.numel()
    with torch.autocast(device_type=prompt.device.type, dtype=torch.bfloat16, enabled=prompt.device.type == 'cuda'):
        return _model_forward(model, values, head)


def install_sft(trainer):
    """F's optimizer and loss unchanged; ordered sampler and explicit probe scheduler."""
    import os
    from kit.v4_phase0_timing import distributed_defaults
    distributed_defaults()
    from torch.utils.data.distributed import DistributedSampler
    from kit.v4_sft import registered_scheduler
    def ordered(*args, **kwargs):
        kwargs['shuffle'] = False; return DistributedSampler(*args, **kwargs)
    def constant(optimizer, **kwargs): return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.)
    trainer.DistributedSampler = ordered
    trainer.get_wsd_schedule_with_warmup = constant if os.environ.get('P3B_RECOVERY') == '1' else registered_scheduler
    torch.manual_seed(int(os.environ['KIT_SFT_SEED']))
    original_run = trainer.run_sft
    def checked(config):
        from pathlib import Path
        from kit.p3b_train import check_data
        from kit.p3b_common import require
        check_data(Path(os.environ['P3B_DATA']))
        require(config.optim.lr==1e-5 and list(config.optim.betas)==[.9,.999] and config.optim.weight_decay==.01 and config.optim.clip_grad==1.,'F optimizer drift')
        require(config.data.train_batch_size==(16 if os.environ.get('P3B_RECOVERY')=='1' else 32),'wrong F batch')
        return original_run(config)
    trainer.run_sft = checked
    old = trainer.FSDPSFTTrainer.training_step
    def counted(self, batch):
        lr = self.optimizer.param_groups[0]['lr']; step = self.optimizer.step
        counts=torch.stack((batch['loss_mask'].sum(),batch['attention_mask'].sum())).to(self.device_name)
        if torch.distributed.is_initialized(): torch.distributed.all_reduce(counts)
        def update(*args, **kwargs):
            value = step(*args, **kwargs); self._v4_updates = getattr(self, '_v4_updates', 0)+1; return value
        self.optimizer.step = update
        try: result = old(self, batch)
        finally: self.optimizer.step = step
        result.update({'p3b/response_tokens':int(counts[0]),'p3b/total_tokens':int(counts[1]),'train/lr':lr, 'p3b/completed_optimizer_updates':getattr(self, '_v4_updates', 0)})
        return result
    trainer.FSDPSFTTrainer.training_step = counted
    from kit.v4_state import install_sft
    install_sft(trainer.FSDPSFTTrainer)
    from kit.v4_timing import install_telemetry
    install_telemetry(trainer.FSDPSFTTrainer,updates='training_step',exports='save_checkpoint',memory_methods=('__init__','save_checkpoint'))


def mixed_teacher_inputs(tokenizer, raw_prompts, demonstrations, responses, response_mask, *, replay, **kwargs):
    """Build task contexts; pad replay with a task copy for discarded FSDP forwards."""
    from kit import sdft_objective as objective
    tasks=[i for i,flag in enumerate(replay) if not flag]
    if not tasks: raise ValueError('mixed D batch needs task items')
    built=objective.build_teacher_inputs(tokenizer,[raw_prompts[i] for i in tasks],
        [demonstrations[i] for i in tasks],responses[tasks],response_mask[tasks],**kwargs)
    for key in ('teacher_input_ids','teacher_attention_mask','teacher_position_ids'):
        task_tensor=built[key]; full=task_tensor[:1].expand(len(replay),*task_tensor.shape[1:]).clone()
        full[tasks]=task_tensor; built[key]=full
    lengths=[built['teacher_prompt_lengths'][0]]*len(replay)
    for i,n in zip(tasks,built['teacher_prompt_lengths']): lengths[i]=n
    built['teacher_prompt_lengths']=lengths
    built['self_distillation_mask']=torch.ones(len(replay),dtype=torch.float32)
    return built
