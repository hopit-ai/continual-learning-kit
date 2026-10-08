"""Plan v4, arm D: demonstration-conditioned SDFT, as a pure objective the verl trainer calls (no verl import here).

What arm D is (docs/plan-v4-teacher-20261006.md, section 3): the student answers from the QUESTION ALONE (ONE on-policy
sample a prompt, as SDFT's reference: num_generations=1); a teacher with the student's own 8B weights (an EMA copy at the reference's rate 0.01) sees the
question PLUS a verified demonstration and scores the student's generated tokens; the loss is per-token FORWARD KL from
that teacher to the student over the FULL vocabulary, with the teacher stop-gradient. Every prompt group gets a target,
including groups where no student answer was right: there is no "successful sibling" condition anywhere.

Everything here is copied from, and tested against, the SDFT reference (references/Self-Distillation, pinned d775732):

  - the teacher prompt: main.py:23-46 and 49-77, `string.Template` substituted with the student's question and the
    demonstration, the student's system message (if any) kept in front, as the reference's science loader does;
  - the loss: distil_trainer.py:1599-1688 with the arguments main.py:100-131 passes -- forward KL
    (`kl_div(student, teacher, log_target=True)`, alpha=0, distil_config.py:486 and README "Updates 04/07/26"), summed
    over the vocabulary; the first `num_loss_tokens_to_skip=3` completion tokens out of the loss (main.py:129); each
    sequence's per-token loss multiplied by the sequence mean of its truncated importance ratio between the trainer
    and the sampler (vllm_importance_sampling_correction=True, cap 2.0, distil_config.py:620-636); a token mean per
    sequence, then a mean over sequences.

Production memory path: kit/sdft_actor.py patches the actor update AND sampled
log-probability forwards. It chunks the model-head projection, normalization,
KL, entropy and backward together and recomputes the reference importance
ratio without verl's lower clamp. The functions accepting precomputed full
log-probability tensors below are CPU numerical diagnostics; their gradients
necessarily have the full input shape and they are not used by D's update.
"""
from __future__ import annotations

import json
import os
import sys
from string import Template
from typing import Any, Mapping, Optional

from pathlib import Path
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

TEACHER_INPUT_CAP = 6144

# -------------------------------------------------------------------------------------------------- reference settings
#: references/Self-Distillation/main.py:30-37 (tool use) and 56-63 (science): the same template in both loaders, byte for byte.
TEACHER_TEMPLATE = Template("""
$orig_content

This is an example for a response to the question:
$output_text

Now answer with a response of your own, including the thinking process.
""")
#: main.py:129 `num_loss_tokens_to_skip = 3`.
NUM_LOSS_TOKENS_TO_SKIP = 3
#: distil_config.py:630 `vllm_importance_sampling_cap` default; main.py:128 turns the correction on.
IMPORTANCE_SAMPLING_CAP = 2.0
#: main.py:16 `--ref_model_mixup_alpha` default 0.01, applied after every optimizer step (main.py:125-127); in verl this is
#: `self_distillation.teacher_update_rate` with the EMA teacher (dp_actor.py:132-151), 0 = frozen.
REFERENCE_TEACHER_RATE = 0.01
#: Arm D's registered settings (plan v4 section 3, frozen 7 October). kit/run_sdft.sh writes the same values as constants,
#: and check_trainer_config refuses a resolved configuration with any other. One sample a prompt: a group is one sample,
#: so nothing on D's path may depend on a group of several (the loss reads no advantage; see tests/test_kit_sdft.py).
REGISTERED = {
    "trainer.total_training_steps": 40,
    "data.max_response_length": 2048,
    "data.train_batch_size": 32,
    "actor_rollout_ref.rollout.n": 1,                                      # main.py:119 num_generations=1
    "actor_rollout_ref.actor.self_distillation.teacher_update_rate": REFERENCE_TEACHER_RATE,
    "actor_rollout_ref.actor.optim.lr": 1e-5,
    "actor_rollout_ref.actor.optim.lr_warmup_steps": 10,
    "actor_rollout_ref.actor.optim.lr_scheduler_type": "constant",
    "actor_rollout_ref.actor.optim.weight_decay": 0.01,
}
#: Tokens of the response handled at once by forward_kl_per_token: 256 x 151,936 x 4 bytes = 156 MB per chunk tensor.
DEFAULT_CHUNK_TOKENS = 256


class SdftConfigError(ValueError):
    """A trainer setting that would make this run something other than arm D."""


class SdftDataError(ValueError):
    """A data row that cannot be trained as arm D (no demonstration, or the demonstration in the student's prompt)."""


# ------------------------------------------------------------------------------------------------- the teacher prompt
def demonstration_of(extra_info: Any) -> str:
    """The row's verified demonstration (`extra_info["demonstration"]`), or SdftDataError. Never a default: a row without
    one would get no distillation target, which is the successful-sibling condition by another route."""
    demonstration = extra_info.get("demonstration") if isinstance(extra_info, Mapping) else None
    if not isinstance(demonstration, str) or not demonstration.strip():
        index = extra_info.get("index") if isinstance(extra_info, Mapping) else None
        raise SdftDataError("row %r has no extra_info.demonstration: arm D trains only on covered questions" % (index,))
    return demonstration


def _contents(messages) -> list:
    return [m.get("content") if isinstance(m, Mapping) else None for m in messages]


def check_student_prompt_clean(messages, demonstration: str) -> None:
    """SdftDataError if the demonstration (or its stripped text) appears anywhere in the student's own prompt."""
    needle = demonstration.strip()
    for content in _contents(messages):
        if not isinstance(content, str):
            raise SdftDataError("a student message has non-text content; arm D's prompts are plain text")
        if needle and needle in content:
            raise SdftDataError("the demonstration is inside the student's prompt: the student must answer from the question alone")


def teacher_messages(messages, demonstration: str) -> list:
    """The teacher's chat: every message before the student's last one unchanged (the system prompt), then the reference's
    template with the student's last message as `orig_content` and the demonstration as `output_text`."""
    messages = list(messages)
    if not messages or not isinstance(messages[-1], Mapping) or messages[-1].get("role") != "user":
        raise SdftDataError("the student's prompt must end with a user message")
    check_student_prompt_clean(messages, demonstration)
    text = TEACHER_TEMPLATE.substitute(orig_content=messages[-1]["content"], output_text=demonstration)
    return [dict(m) for m in messages[:-1]] + [{"role": "user", "content": text}]


def with_demonstration(row: Mapping, demonstration: str) -> dict:
    """A copy of a verl data row with `extra_info.demonstration` set; the prompt is untouched."""
    if not isinstance(demonstration, str) or not demonstration.strip():
        raise SdftDataError("an empty demonstration is not a demonstration")
    out = dict(row)
    out["extra_info"] = dict(row.get("extra_info") or {}, demonstration=demonstration)
    check_student_prompt_clean(out["prompt"], demonstration)
    return out


def check_rows(train_rows, test_rows) -> dict:
    """What the launcher checks before taking a GPU: every training row carries a demonstration that is not in its prompt,
    and NO held-out row carries one (no teacher material from held-out questions). Returns counts; raises SdftDataError."""
    train_rows, test_rows = list(train_rows), list(test_rows)
    from kit.v4_contract import check_synthetic_rows
    check_synthetic_rows(train_rows, os.environ.get("STEPS", 40), os.environ.get("V4_PROFILE", "scientific"))
    check_synthetic_rows(test_rows)
    if not train_rows:
        raise SdftDataError("the training file has no rows")
    if any(row.get("data_source") in ("sciknoweval", "finqa") for row in test_rows):
        from kit.v4_contract import check_heldout_rows
        check_heldout_rows(test_rows)
    for row in train_rows:
        teacher_messages(row["prompt"], demonstration_of(row.get("extra_info")))
    leaked = [i for i, row in enumerate(test_rows) if isinstance(row.get("extra_info"), Mapping)
              and row["extra_info"].get("demonstration") not in (None, "")]
    if leaked:
        raise SdftDataError("%d held-out rows carry a demonstration (first at row %d): refused" % (len(leaked), leaked[0]))
    return {"train_rows": len(train_rows), "train_rows_with_demonstration": len(train_rows), "test_rows": len(test_rows),
            "test_rows_with_demonstration": 0}


def build_teacher_inputs(tokenizer, raw_prompts, demonstrations, responses: torch.Tensor, response_mask: torch.Tensor, *,
                         chat_template_kwargs: Optional[Mapping] = None, max_reprompt_len: int) -> dict:
    """The tensors verl's actor reads for the teacher (ray_trainer.py:762-795's keys): the teacher prompt rendered with the
    student's own chat-template settings, LEFT padded, followed by the student's sampled response ids unchanged.

    The teacher prompt is rendered to text and tokenised without special tokens, as the reference does
    (distil_trainer.py:1363-1375). A teacher prompt longer than `max_reprompt_len` is refused, never truncated.
    `self_distillation_mask` is all ones: every sample has a target."""
    raw_prompts, demonstrations = list(raw_prompts), list(demonstrations)
    if not (len(raw_prompts) == len(demonstrations) == responses.shape[0] == response_mask.shape[0]):
        raise ValueError("prompts, demonstrations and responses disagree in number")
    kwargs = dict(chat_template_kwargs or {})
    ids = []
    for messages, demonstration in zip(raw_prompts, demonstrations):
        text = tokenizer.apply_chat_template(teacher_messages(messages, demonstration), tokenize=False,
                                             add_generation_prompt=True, **kwargs)
        row = tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(row) > max_reprompt_len:
            raise SdftDataError("a teacher prompt has %d tokens, over max_reprompt_len=%d: refused rather than truncated"
                                % (len(row), max_reprompt_len))
        ids.append(row)
    width = max(len(r) for r in ids)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    prompt_ids = torch.full((len(ids), width), pad, dtype=torch.long)
    prompt_mask = torch.zeros((len(ids), width), dtype=response_mask.dtype)
    for i, row in enumerate(ids):
        if row:
            prompt_ids[i, width - len(row):] = torch.tensor(row, dtype=torch.long)
            prompt_mask[i, width - len(row):] = 1
    responses = responses.to("cpu")
    response_mask = response_mask.to("cpu")
    input_ids = torch.cat([prompt_ids, responses.to(torch.long)], dim=1)
    attention_mask = torch.cat([prompt_mask, response_mask], dim=1)
    position_ids = torch.clip(torch.cumsum(attention_mask, dim=-1) - 1, min=0, max=None)   # verl/utils/model.py:226
    return {"teacher_input_ids": input_ids, "teacher_attention_mask": attention_mask, "teacher_position_ids": position_ids,
            "self_distillation_mask": torch.ones(len(ids), dtype=torch.float32),
            "teacher_prompt_lengths": [len(r) for r in ids]}


def group_metrics(uids, sequence_scores, success_threshold: float) -> dict:
    """Informational only (nothing here gates the loss): the share of prompt groups with no rewarded attempt, which the
    plan's mechanism variable reports, and the share of samples that got a target (always 1)."""
    groups: dict = {}
    for uid, score in zip(list(uids), list(sequence_scores)):
        groups.setdefault(uid, []).append(float(score) >= success_threshold)
    n = max(len(groups), 1)
    return {"self_distillation/success_group_fraction": sum(any(v) for v in groups.values()) / n,
            "sdft/no_success_group_fraction": sum(not any(v) for v in groups.values()) / n,
            "self_distillation/reprompt_sample_fraction": 1.0}


# ----------------------------------------------------------------------------------------------------------- the loss
class _ForwardKL(torch.autograd.Function):
    """sum_v exp(t) * (t - s) per token, i.e. `kl_div(s, t, reduction="none", log_target=True).sum(-1)`, in token chunks.

    Backward: d/ds = -exp(t) (the teacher gets no gradient). Nothing of size (B, T, V) is saved beyond the inputs."""

    @staticmethod
    def forward(ctx, student, teacher, chunk):
        out = student.new_empty(student.shape[:-1])
        for b in range(student.shape[0]):
            for i in range(0, student.shape[1], chunk):
                t = teacher[b, i:i + chunk]
                out[b, i:i + chunk] = (t.exp() * (t - student[b, i:i + chunk])).sum(-1)
        ctx.save_for_backward(teacher)
        ctx.chunk = chunk
        return out

    @staticmethod
    def backward(ctx, grad_out):
        (teacher,) = ctx.saved_tensors
        grad = torch.empty_like(teacher)
        for b in range(teacher.shape[0]):
            for i in range(0, teacher.shape[1], ctx.chunk):
                grad[b, i:i + ctx.chunk] = -teacher[b, i:i + ctx.chunk].exp() * grad_out[b, i:i + ctx.chunk].unsqueeze(-1)
        return grad, None, None


def forward_kl_per_token(student_logps: torch.Tensor, teacher_logps: torch.Tensor,
                         chunk_tokens: int = DEFAULT_CHUNK_TOKENS) -> torch.Tensor:
    """(B, T) forward KL(teacher || student) over the full vocabulary, from (B, T, V) log-probabilities. The teacher is
    detached here whatever the caller did: no gradient can reach it."""
    if student_logps.dim() != 3 or student_logps.shape != teacher_logps.shape:
        raise ValueError("student and teacher log-probabilities must both be (B, T, V), got %s and %s"
                         % (tuple(student_logps.shape), tuple(teacher_logps.shape)))
    if chunk_tokens < 1:
        raise ValueError("chunk_tokens must be positive")
    return _ForwardKL.apply(student_logps, teacher_logps.detach().to(student_logps.dtype), int(chunk_tokens))


def reference_importance_weights(old_log_probs, rollout_log_probs):
    """Reference's min(exp(old-rollout), 2), with no artificial lower clamp."""
    return torch.exp(old_log_probs.detach() - rollout_log_probs.detach()).clamp(max=IMPORTANCE_SAMPLING_CAP)


def loss_mask(response_mask: torch.Tensor, skip_tokens: int = NUM_LOSS_TOKENS_TO_SKIP, dtype=torch.float32) -> torch.Tensor:
    """distil_trainer.py:1599-1606: the response mask with its first `skip_tokens` positions zeroed."""
    mask = response_mask.to(dtype)
    if skip_tokens > 0:
        positions = torch.arange(mask.shape[-1], device=mask.device).unsqueeze(0)
        mask = mask * (positions >= skip_tokens).to(dtype)
    return mask


def sdft_loss(student_logps: torch.Tensor, teacher_logps: torch.Tensor, response_mask: torch.Tensor,
              is_weights: Optional[torch.Tensor], *, skip_tokens: int = NUM_LOSS_TOKENS_TO_SKIP,
              chunk_tokens: int = DEFAULT_CHUNK_TOKENS) -> tuple:
    """The reference's loss for one micro-batch, before verl's 1/gradient-accumulation scaling: (loss, metrics).

    `is_weights` is verl's per-token truncated ratio (already min(exp(old - rollout), cap) and zero on padding); it enters
    as its mean over each sequence's loss tokens, as distil_trainer.py:1678-1682 does. None means no correction."""
    mask = loss_mask(response_mask, skip_tokens, dtype=student_logps.dtype)
    per_token = forward_kl_per_token(student_logps, teacher_logps, chunk_tokens)
    counts = mask.sum(-1).clamp(min=1.0)
    metrics = {}
    if is_weights is not None:
        sequence_weight = (is_weights.detach().to(student_logps.dtype) * mask).sum(-1) / counts
        per_token = per_token * sequence_weight.unsqueeze(-1)
        metrics["sdft/is_sequence_weight_mean"] = float(sequence_weight.mean().item())
    loss = ((per_token * mask).sum(-1) / counts).mean()
    with torch.no_grad():
        metrics["sdft/forward_kl_token_mean"] = float(((per_token.detach() * mask).sum() / mask.sum().clamp(min=1.0)).item())
        metrics["sdft/loss_tokens"] = float(mask.sum().item())
    return loss, metrics


# ------------------------------------------------------------------------------------------- what the trainer must be
def _get(tree: Any, dotted: str, default: Any = KeyError):
    node = tree
    for part in dotted.split("."):
        if isinstance(node, Mapping):
            if part not in node:
                if default is KeyError:
                    raise SdftConfigError("the trainer configuration has no %s" % dotted)
                return default
            node = node[part]
        elif hasattr(node, part):
            node = getattr(node, part)
        else:
            if default is KeyError:
                raise SdftConfigError("the trainer configuration has no %s" % dotted)
            return default
    return node


def check_self_distillation(sd: Any, loss_agg_mode: str) -> None:
    """The settings the loss itself depends on; SdftConfigError on any other. Called by the trainer before it starts and
    by the loss on every call, so a top-k, a reverse KL or a JSD can never be trained under arm D's name."""
    if _get(sd, "distillation_topk", None) is not None:
        raise SdftConfigError("distillation_topk=%r: arm D is full-vocabulary forward KL, never top-k"
                              % _get(sd, "distillation_topk", None))
    if _get(sd, "full_logit_distillation") is not True:
        raise SdftConfigError("full_logit_distillation must be True: arm D distils the full distribution")
    if float(_get(sd, "alpha")) != 0.0:
        raise SdftConfigError("alpha=%r: arm D is forward KL (alpha 0), as the reference's paper results used"
                              % _get(sd, "alpha"))
    if _get(sd, "is_clip", None) is not None:
        raise SdftConfigError("is_clip must be null: the reference has no student/old ratio clip in its loss")
    if _get(sd, "teacher_regularization") != "ema":
        raise SdftConfigError("teacher_regularization must be ema (the reference's moving teacher)")
    if _get(sd, "include_environment_feedback", False):
        raise SdftConfigError("include_environment_feedback must be False: arm D's teacher sees the demonstration only")
    if loss_agg_mode != "seq-mean-token-mean":
        raise SdftConfigError("loss_agg_mode=%r: the reference takes a token mean per sequence, then a sequence mean"
                              % loss_agg_mode)


def check_trainer_config(config: Any) -> None:
    """Every whole-run setting arm D depends on, read from the resolved trainer configuration (a mapping)."""
    from kit.v4_contract import profile
    import os
    profile(_get(config, "trainer.total_training_steps"), os.environ.get("V4_PROFILE", "scientific"))
    for key, expected in (("data.max_response_length", 2048), ("data.train_batch_size", 32),
                          ("data.max_prompt_length", 2048), ("data.filter_overlong_prompts", False),
                          ("actor_rollout_ref.actor.self_distillation.max_reprompt_len", 6144)):
        if _get(config, key) != expected: raise SdftConfigError(key + " differs from registered recipe")
    actor = _get(config, "actor_rollout_ref.actor")
    if _get(actor, "policy_loss.loss_mode") != "sdpo":
        raise SdftConfigError("policy_loss.loss_mode must be sdpo: verl's self-distillation path carries arm D")
    check_self_distillation(_get(actor, "self_distillation"), _get(actor, "loss_agg_mode"))
    if float(_get(actor, "entropy_coeff", 0)) != 0.0:
        raise SdftConfigError("entropy_coeff must be 0: the reference adds no entropy term")
    if _get(actor, "use_kl_loss", False):
        raise SdftConfigError("use_kl_loss must be False")
    if int(_get(actor, "ppo_epochs", 1)) != 1:
        raise SdftConfigError("ppo_epochs must be 1: one update per batch of samples, as the reference's num_iterations=1")
    if int(_get(actor, "ppo_mini_batch_size")) != int(_get(config, "data.train_batch_size")):
        raise SdftConfigError("ppo_mini_batch_size must equal data.train_batch_size: one optimizer step and one teacher "
                              "update per batch of prompts, as the reference")
    correction = _get(config, "algorithm.rollout_correction")
    if _get(correction, "rollout_is", None) != "token" or float(_get(correction, "rollout_is_threshold", 0)) != IMPORTANCE_SAMPLING_CAP:
        raise SdftConfigError("algorithm.rollout_correction must be rollout_is=token with threshold %s: it supplies the "
                              "reference's truncated importance ratio" % IMPORTANCE_SAMPLING_CAP)
    if (_get(correction, "rollout_rs", None) is not None or _get(correction, "bypass_mode", False)
            or _get(correction, "rollout_is_batch_normalize", False)):
        raise SdftConfigError("rollout rejection sampling, bypass mode and batch-normalised ratios are not part of the reference")
    for dotted, registered in REGISTERED.items():
        if dotted == "trainer.total_training_steps" and os.environ.get("V4_PROFILE") == "technical-smoke":
            continue  # profile() above bounds this explicitly labelled engineering dose
        found = _get(config, dotted)
        if (found != registered) if isinstance(registered, str) else (float(found) != float(registered)):
            raise SdftConfigError("%s=%r: registered for arm D at %r (plan v4 section 3)" % (dotted, found, registered))
    if _get(config, "actor_rollout_ref.rollout.calculate_log_probs", False) is not True:
        raise SdftConfigError("rollout.calculate_log_probs must be True: the importance ratio needs the sampler's log-probs")


def self_distillation_loss(student_log_probs, teacher_log_probs, response_mask, self_distillation_config,
                           old_log_probs=None, student_all_log_probs=None, teacher_all_log_probs=None,
                           student_topk_log_probs=None, teacher_topk_log_probs=None, self_distillation_mask=None,
                           loss_agg_mode="token-mean", rollout_is_weights=None):
    """A drop-in for verl's `compute_self_distillation_loss` (core_algos.py:1085, same signature, called by
    dp_actor.py:833) that computes the SDFT reference's loss, and refuses everything else."""
    check_self_distillation(self_distillation_config, loss_agg_mode)
    if student_topk_log_probs is not None or teacher_topk_log_probs is not None:
        raise SdftConfigError("top-k log-probabilities reached the loss: arm D is full-vocabulary")
    if student_all_log_probs is None or teacher_all_log_probs is None:
        raise SdftConfigError("full-vocabulary log-probabilities are missing: the actor did not take the full-logit path")
    if self_distillation_mask is not None and bool((self_distillation_mask != 1).any()):
        raise SdftConfigError("a sample without a distillation target reached the loss: arm D has no successful-sibling "
                              "condition, every sample has a target")
    if rollout_is_weights is None:
        raise SdftConfigError("rollout_is_weights are missing: the reference applies the truncated importance ratio")
    return sdft_loss(student_all_log_probs, teacher_all_log_probs, response_mask, rollout_is_weights)


# ----------------------------------------------------------------------------------------------- the launcher's check
def _read_parquet(path: str) -> list:
    import pyarrow.parquet as pq                                                              # noqa: PLC0415
    return pq.read_table(path).to_pylist()


def main(argv) -> int:
    """`python sdft_objective.py check-data TRAIN.parquet TEST.parquet OUT.json`: kit/run_sdft.sh's data check."""
    if argv in (["--help"], ["-h"]):
        print("usage: sdft_objective.py check-data TRAIN.parquet TEST.parquet OUT.json")
        return 0
    if len(argv) != 4 or argv[0] != "check-data":
        print("usage: sdft_objective.py check-data TRAIN.parquet TEST.parquet OUT.json", file=sys.stderr)
        return 2
    try:
        rows = _read_parquet(argv[1])
        from kit.v4_contract import check_schedule
        import os
        check_schedule(rows, int(os.environ.get("STEPS", "40")))
        counts = check_rows(rows, _read_parquet(argv[2]))
    except (SdftDataError, KeyError, TypeError) as error:
        print("arm D data check failed: %s" % error, file=sys.stderr)
        return 2
    with open(argv[3], "w") as handle:
        json.dump(dict(counts, train_file=argv[1], test_file=argv[2]), handle, indent=1)
    print("arm D data check: %d training rows, each with a demonstration not in its prompt; %d held-out rows, none with one"
          % (counts["train_rows"], counts["test_rows"]))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
