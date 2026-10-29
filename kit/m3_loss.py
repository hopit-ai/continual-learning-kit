"""M3 EOS-inclusive sequence means and bounded full-vocabulary replay losses."""
from __future__ import annotations
import torch
from kit.m3_contract import PROJECTION_TOKENS, GLOBAL_DIVISOR, geometry
from kit.sdft_actor import _model_forward, teacher_projection


class _ReplayProjection(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, base_hidden, base_weight, targets, chunk, mode):
        h = hidden.reshape(-1, hidden.shape[-1])
        bh = base_hidden.detach().reshape_as(h) if mode == 'kl' else base_hidden
        out = torch.empty(h.shape[0], dtype=torch.float32, device=h.device)
        for start in range(0, len(h), chunk):
            end = start + chunk
            s = (h[start:end] @ weight.T).float().log_softmax(-1)
            if mode == 'kl':
                b = (bh[start:end] @ base_weight.detach().T).float().log_softmax(-1)
                out[start:end] = (b.exp() * (b - s)).sum(-1)
            else:
                out[start:end] = -s.gather(-1, targets.reshape(-1)[start:end, None]).squeeze(-1)
        ctx.save_for_backward(hidden, weight, base_hidden.detach(), base_weight.detach(), targets)
        ctx.chunk, ctx.mode = chunk, mode
        return out.reshape(hidden.shape[:-1])

    @staticmethod
    def backward(ctx, gradient):
        hidden, weight, base_hidden, base_weight, targets = ctx.saved_tensors
        h = hidden.reshape(-1, hidden.shape[-1])
        bh = base_hidden.reshape_as(h) if ctx.mode == 'kl' else base_hidden
        gh = torch.empty_like(h)
        gw = torch.zeros(weight.shape, dtype=torch.float32, device=weight.device)
        for start in range(0, len(h), ctx.chunk):
            end = start + ctx.chunk
            g = (h[start:end] @ weight.T).float().softmax(-1)
            if ctx.mode == 'kl':
                g = g - (bh[start:end] @ base_weight.T).float().softmax(-1)
            else:
                g.scatter_add_(-1, targets.reshape(-1)[start:end, None], -torch.ones_like(g[:, :1]))
            g = g * gradient.reshape(-1)[start:end, None]
            with torch.autocast(device_type=h.device.type, enabled=False):
                gh[start:end] = (g @ weight.float()).to(h.dtype)
                gw.add_(g.T @ h[start:end].float())
        return gh.reshape_as(hidden), gw.to(weight.dtype), None, None, None, None, None


def projected_forward_kl(hidden, weight, base_hidden, base_weight, chunk_tokens=256):
    """KL(base || student), T=1; at most 256 *total* prediction tokens/project.

    No reference gradients, vocabulary approximation or temperature multiplier.
    The only saved tensors are hidden states and parameter-sized head weights.
    """
    if not 1 <= chunk_tokens <= PROJECTION_TOKENS:
        raise ValueError('projection bound is 1..256 tokens')
    if hidden.shape != base_hidden.shape or weight.shape != base_weight.shape:
        raise ValueError('base/student projection geometry differs')
    targets = torch.empty(0, dtype=torch.long, device=hidden.device)
    return _ReplayProjection.apply(hidden, weight, base_hidden, base_weight, targets, chunk_tokens, 'kl')


def projected_ce(hidden, weight, targets, chunk_tokens=256):
    if targets.shape != hidden.shape[:-1] or not 1 <= chunk_tokens <= PROJECTION_TOKENS:
        raise ValueError('invalid CE projection geometry')
    empty = hidden.new_empty(0)
    return _ReplayProjection.apply(hidden, weight, empty, empty, targets, chunk_tokens, 'ce')


def sequence_mean(tokens, mask):
    if tokens.shape != mask.shape or bool((mask.sum(-1) == 0).any()):
        raise ValueError('each sequence needs at least one response token')
    mask = mask.float()
    return ((tokens.float() * mask).sum(-1) / mask.sum(-1)).mean()


def response_prediction_mask(input_ids, attention_mask, prompt_width, eos_token_id):
    """Prediction-position mask: prompt/pad excluded, first EOS included, no skip."""
    if input_ids.shape != attention_mask.shape or prompt_width < 1 or prompt_width >= input_ids.shape[1]:
        raise ValueError('invalid prompt/response geometry')
    response = input_ids[:, prompt_width:]
    active = attention_mask[:, prompt_width:].bool()
    eos = (response == eos_token_id) & active
    before_or_first_eos = (eos.cumsum(-1) - eos.to(torch.long)) == 0
    return (active & before_or_first_eos).float()


def replay_inputs(inputs, target_ids, eos_token_id):
    """Replace sampled response with the identical stored prefix for both models."""
    if inputs['input_ids'].shape[0] != 1:
        raise ValueError('M3 microbatch must be one row')
    ids = list(target_ids)
    if not ids or ids[-1] != eos_token_id or eos_token_id in ids[:-1] or len(ids) > 2048:
        raise ValueError('stored replay must terminate in one EOS within response cap')
    if any(type(value) is not int or value < 0 for value in ids):
        raise ValueError('target IDs must be nonnegative integers')
    width = inputs['responses'].shape[-1]
    prompt = inputs['input_ids'][:, :-width]
    pmask = inputs['attention_mask'][:, :-width]
    if prompt.shape[1] < 1 or int(pmask.sum()) < 1:
        raise ValueError('empty replay prompt')
    target = torch.tensor(ids, dtype=torch.long, device=prompt.device)[None]
    full = torch.cat((prompt, target), -1)
    attention = torch.cat((pmask, torch.ones_like(target)), -1)
    mask = response_prediction_mask(full, attention, prompt.shape[-1], eos_token_id)
    assert int(mask.sum()) == len(ids)
    return dict(input_ids=full, attention_mask=attention,
                position_ids=(attention.cumsum(-1)-1).clamp(min=0), responses=target, response_mask=mask)


def replay_loss(model, values, base=None, projection=None):
    """Stored-prefix CE, or forward KL against a separate frozen base."""
    width = values['responses'].shape[-1]
    if base is not None and projection is None:
        projection = teacher_projection(base, values)
    def head(hidden, weight):
        hidden = hidden[:, -width-1:-1]
        per_token = (projected_forward_kl(hidden, weight, projection['hidden'], projection['weight'])
                     if projection is not None else projected_ce(hidden, weight, values['responses']))
        return sequence_mean(per_token, values['response_mask'])
    with torch.autocast(device_type=values['input_ids'].device.type, dtype=torch.bfloat16,
                        enabled=values['input_ids'].device.type == 'cuda'):
        return _model_forward(model, values, head)


def compose_loss(task_losses, replay_losses, arm, stage):
    """Independent global objective oracle; never divide CR/G by total rows."""
    counts = geometry(arm, stage)
    if task_losses.numel() != counts['task_rows'] or replay_losses.numel() != counts['replay_rows']:
        raise ValueError('arm row counts differ')
    return (task_losses.sum() + replay_losses.sum()) / GLOBAL_DIVISOR


def audit_long_context_rows(tokenizer, rows, *, prompt_limit=10240, response_limit=2048,
                            reprompt_limit=14336, context_limit=16384):
    """CPU-render admitted rows through the actual rollout/teacher templates.

    Every row is checked (never filtered); maxima per domain bind the longest
    legal and FinQA rows to the common stage-3 configuration. No model is loaded.
    The full possible sampled response is reserved in both contexts.
    """
    import hashlib
    import json
    from kit.sdft_objective import teacher_messages
    from kit.m3_contract import PROMPT_CAP, RESPONSE_CAP, REPROMPT_CAP, CONTEXT_CAP
    if (prompt_limit<PROMPT_CAP or response_limit!=RESPONSE_CAP or
            reprompt_limit<REPROMPT_CAP or context_limit<CONTEXT_CAP):
        raise ValueError('all arms/stages require the common long-context limits')
    maxima={}
    count=0
    for row in rows:
        count+=1
        prompt=row['prompt']
        if isinstance(prompt,str): prompt=json.loads(prompt) if prompt.startswith('[') else [dict(role='user',content=prompt)]
        extra=row['extra_info']
        demonstration=extra['demonstration']
        def render(messages):
            text=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
            return tokenizer.encode(text,add_special_tokens=False)
        prompt_ids=render(prompt)
        target_ids=tokenizer.encode(demonstration+tokenizer.eos_token,add_special_tokens=False)
        teacher_ids=(render(teacher_messages(prompt,demonstration))
                     if not extra.get('m3_replay') else prompt_ids)
        if (not 0<len(prompt_ids)<=prompt_limit or not 0<len(target_ids)<=response_limit or
                not target_ids or target_ids[-1]!=tokenizer.eos_token_id or
                tokenizer.eos_token_id in target_ids[:-1]):
            raise ValueError('admitted row exceeds common input/response cap or EOS convention; no truncation')
        if (len(teacher_ids)>reprompt_limit or len(teacher_ids)+response_limit>context_limit or
                len(prompt_ids)+response_limit>context_limit):
            raise ValueError('fully rendered teacher/rollout context exceeds stage-3 allowance; no truncation')
        if list(extra['m3_target_ids'])!=list(target_ids):
            raise ValueError('stored target differs from actual local tokenizer encoding')
        domain=extra['m3_domain']
        values=dict(prompt_tokens=len(prompt_ids),response_including_eos=len(target_ids),
                    teacher_prompt_tokens=len(teacher_ids),teacher_full_context=len(teacher_ids)+response_limit)
        previous=maxima.setdefault(domain,dict.fromkeys(values,0))
        for key,value in values.items(): previous[key]=max(previous[key],value)
    if count==0: raise ValueError('no admitted rows to audit')
    digest=hashlib.sha256(json.dumps(maxima,sort_keys=True).encode()).hexdigest()
    return dict(rows=count,skipped_draw_count=0,longest_by_domain=maxima,length_audit_sha256=digest,
                prompt_cap=prompt_limit,response_cap=response_limit,reprompt_cap=reprompt_limit,context_cap=context_limit)


def cuda_full_vocab_check():
    """CUDA fp32 full-vocabulary oracle, V=151936 and T=300 (>256).

    No CPU fallback: an unavailable device leaves GPU qualification incomplete.
    TF32/autocast are disabled for the registered 1e-5 absolute tolerances.
    """
    if not torch.cuda.is_available():raise RuntimeError('CUDA full-vocabulary qualification is INCOMPLETE')
    previous=torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32=False
    try:
        generator=torch.Generator(device='cuda').manual_seed(101)
        with torch.autocast(device_type='cuda',enabled=False):
            h=(torch.randn((1,300,4),generator=generator,device='cuda')*.1).requires_grad_()
            w=(torch.randn((151936,4),generator=generator,device='cuda')*.1).requires_grad_()
            bh=torch.randn(h.shape,generator=generator,device='cuda')*.1
            bw=torch.randn(w.shape,generator=generator,device='cuda')*.1
            got=projected_forward_kl(h,w,bh,bw,256)
            s=(h@w.T).log_softmax(-1);b=(bh@bw.T).log_softmax(-1)
            reference=(b.exp()*(b-s)).sum(-1)
            value_error=float((got-reference).abs().max())
            a=torch.autograd.grad(got.mean(),(h,w),retain_graph=True)
            r=torch.autograd.grad(reference.mean(),(h,w))
            gradient_error=max(float((x-y).abs().max()) for x,y in zip(a,r))
        if max(value_error,gradient_error)>1e-5:raise RuntimeError('CUDA full-vocabulary KL agreement exceeds 1e-5')
        return dict(passed=True,vocabulary=151936,tokens=300,value_error=value_error,gradient_error=gradient_error)
    finally:
        torch.backends.cuda.matmul.allow_tf32=previous
        torch.cuda.empty_cache()
