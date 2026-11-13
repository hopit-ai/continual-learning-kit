"""Sequential full-sequence OpenCompass likelihoods and subset aggregation."""
import math
import warnings
from contextlib import contextmanager

from kit.trace_common import need
from kit.trace_general_build import config_value, VENDOR


def prompt_text(tokenizer, row, *, ppl=False):
    if row['render_mode'] == 'raw':
        return row['prompt']
    return tokenizer.apply_chat_template(row['messages'], tokenize=False,
        add_generation_prompt=not ppl, enable_thinking=False)


def select_label(labels, losses):
    need(labels and len(labels) == len(losses) and
         all(math.isfinite(x) for x in losses), 'invalid PPL label losses')
    return labels[losses.index(min(losses))]


def label_texts(tokenizer, row):
    """PPLInferencer's mutable ICE list, carried forward in label order."""
    shot_ids = list(row.get('shot_ids', []))
    shots = row.get('shot_prompts', [])
    if shot_ids:
        need(len(shots) == len(shot_ids) and isinstance(row.get('ice_prefix'), str),
             'MMLU shot rendering metadata missing; rebuild staged general rows')
        prefix = row['ice_prefix']
        original_ice = prefix + '\n'.join(shots) + '\n'
    for prompt in row['label_prompts']:
        text = prompt_text(tokenizer, prompt, ppl=True)
        if shots:
            need(text.startswith(original_ice), 'MMLU shot rendering differs')
            query = text[len(original_ice):]
            # BaseRetriever keeps ice_eos_token even with zero examples.
            text = prefix + '\n'.join(shots[:len(shot_ids)]) + '\n' + query
            while shot_ids and len(tokenizer.encode(text)) > 2048:
                shot_ids = shot_ids[:-1]
                ice = '\n'.join(shots[:len(shot_ids)]) + '\n'
                text = prefix + ice + query
        yield text, list(shot_ids)


@contextmanager
def likelihood_runtime(model, tokenizer_path):
    """Keep one local bf16 HF model loaded across a state's PPL panels."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from kit.m2_gpu import assert_current
    assert_current()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    net = AutoModelForCausalLM.from_pretrained(model, local_files_only=True,
            torch_dtype=torch.bfloat16).to('cuda').eval()
    def logits_fn(ids):
        return net(input_ids=ids, use_cache=False).logits
    logits_fn.device = 'cuda'
    try:
        yield tokenizer, logits_fn
    finally:
        del net, logits_fn
        torch.cuda.empty_cache()


def likelihood(model, rows, tokenizer_path, runtime=None):
    """Runtime injection is (tokenizer, logits_fn(input_ids)); CPU when injected.

    Matches HF._get_ppl: float32 shifted logits, ignore pad targets, divide by
    the full non-pad input length (NOT the number of shifted targets).
    Tokenization is HF.get_logits' one-sequence call, with its default specials
    and 2048 token limit; no normalizing_str or mask_length.
    """
    import torch
    if runtime is None:
        with likelihood_runtime(model, tokenizer_path) as loaded:
            return likelihood(model, rows, tokenizer_path, runtime=loaded)
    tokenizer, logits_fn = runtime
    device = getattr(logits_fn, 'device', 'cpu')
    records = []
    with torch.inference_mode():
        for row in rows:
            losses, lengths, truncated, used_shots = [], [], [], []
            for text, shot_ids in label_texts(tokenizer, row):
                used_shots.append(shot_ids)
                uncut = tokenizer([text], padding=False, truncation=False)['input_ids'][0]
                ids = tokenizer([text], padding=False, truncation=True,
                                max_length=2048)['input_ids'][0]
                need(len(ids) >= 2, 'PPL sequence requires two tokens')
                inputs = torch.tensor([ids], device=device)
                logits = logits_fn(inputs)
                logits = logits[..., :-1, :].contiguous().float()
                targets = inputs[..., 1:].contiguous()
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)), targets.view(-1),
                    reduction='none', ignore_index=tokenizer.pad_token_id).view(targets.size())
                n = (inputs != tokenizer.pad_token_id).sum(-1).cpu().numpy()
                need(n[0] > 0, 'PPL sequence has no non-pad tokens')
                value = (loss.sum(-1).cpu().detach().numpy()/n)[0]
                losses.append(float(value))
                lengths.append(len(uncut))
                truncated.append(max(0, len(uncut)-len(ids)))
            records.append(dict(id=row['id'], losses=losses,
                selected=select_label(row['labels'], losses), uncut_lengths=lengths,
                shot_ids_by_label=dict(zip(row['labels'], used_shots)),
                truncated_tokens_by_label=dict(zip(row['labels'], truncated)),
                prompt_cap_hits=sum(x > 0 for x in truncated),
                truncated_tokens=sum(truncated),
                truncation_side=getattr(tokenizer, 'truncation_side', 'right')))
    hits = sum(r['prompt_cap_hits'] for r in records)
    if hits:
        warnings.warn(f'TRACE general PPL: {hits} label prompts truncated at 2048 tokens', stacklevel=2)
    return records


def aggregate(panel, rows, values):
    from kit.trace_score import aggregate as basic
    need(len(rows) == len(values), 'general score rows missing')
    by_subset = {}
    for row, value in zip(rows, values):
        by_subset.setdefault(row['subset'], []).append(value)
    scores = {key: basic(items) for key, items in by_subset.items()}
    # sum/n must represent the primary naive mean in the existing TRACE return.
    result = basic([v['mean'] for v in scores.values()])
    result.update(aggregation='OpenCompass naive subset mean', subsets=scores, pooled=basic(values))
    if panel == 'MMLU':
        import ast
        from kit.trace_general_build import _data
        tree = ast.parse((VENDOR/'configs/summarizers/groups/mmlu.py').read_text())
        weights = next(_data(node.value, {}) for node in tree.body
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == '_mmlu_weights'
                                                   for t in node.targets))
        need(set(weights) == set(scores), 'MMLU weighted group requires all 57 subjects')
        result['mmlu-weighted'] = math.fsum(scores[k]['mean']*w for k, w in weights.items())/sum(weights.values())
    return result
