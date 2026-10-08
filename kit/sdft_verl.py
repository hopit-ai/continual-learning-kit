"""Plan v4, arm D inside the pinned verl trainer (references/SDPO at 7c457fc), with no file of that checkout edited.

WHY A RUNTIME PATCH AND NOT A PATCH FILE. verl's SDPO path already does everything arm D needs except two functions:
the actor's loss (`compute_self_distillation_loss`) and the trainer's teacher context (`_maybe_build_self_distillation_batch`,
which shows the teacher a SUCCESSFUL SIBLING and masks every sample that has none). verl has a documented hook for
replacing the driver-side runner (`run_ppo(config, task_runner_class=...)`, main_ppo.py:49-56, "For recipe to change
TaskRunner"; it reads the module's `TaskRunner` when no class is passed), and the runner chooses the worker class it
hands to Ray (main_ppo.py:123-180). So:

  - `SdftTaskRunner` (in the TaskRunner process) checks the resolved configuration, replaces
    `RayPPOTrainer._maybe_build_self_distillation_batch` with `build_sdft_teacher_batch` (teacher = question + the row's
    `extra_info.demonstration`, the SDFT reference's template; every sample gets a target), and registers
  - `SdftActorRolloutRefWorker`, the pinned `AsyncActorRolloutRefWorker` unchanged except that constructing it (in each
    GPU worker process) replaces `dp_actor.compute_self_distillation_loss` with kit/sdft_objective.py's
    `self_distillation_loss` (full-vocabulary forward KL, the reference's reduction). dp_actor calls that name at run
    time (dp_actor.py:833), so the replacement is what every micro-batch uses.

Ray imports this module in those processes because the two classes are pickled by reference (they live in an importable
module, `kit.sdft_verl`, on the PYTHONPATH kit/run_sdft.sh exports). Before patching anything, `check_pinned` compares the
sha256 of every verl file whose behaviour this relies on with the pinned commit's: a different verl fails closed.

Launched by kit/run_sdft.sh as `python kit/sdft_entry.py --config-name sdpo <overrides>`; verl's own Hydra entry
(`main_ppo.main`) composes the configuration exactly as `python -m verl.trainer.main_ppo` does (the same pattern as
kit/sft_entry.py).
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import ray
from omegaconf import OmegaConf

from kit import sdft_objective as objective
import verl.trainer.main_ppo as main_ppo
import verl.trainer.ppo.ray_trainer as ray_trainer
import verl.workers.actor.dp_actor as dp_actor
from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker

#: sha256 of each pinned file (references/SDPO at 7c457fc1b1f6) this module replaces a function of, calls into, or
#: relies on the call pattern of. tests/test_kit_sdft.py recomputes them from the reference checkout.
PINNED_SHA256 = {
    "verl/trainer/main_ppo.py": "0e9888efd2ff36d92ba73935a5ed964dbaf480772130042c0860611c8a4c789e",
    "verl/trainer/ppo/ray_trainer.py": "bb7312f2c503e62e219b4d7efc7fde196872eea7f34445430850a1436608a388",
    "verl/workers/actor/dp_actor.py": "cfb30b67b30f61e63966e84a3bdcc894937e5d9eba0bcc42e6210030c03e7fbd",
    "verl/trainer/ppo/core_algos.py": "4f949c323f005b4945f1983bdf89a9ddbd53865d0d7511993c37fbe7c47d6903",
    "verl/workers/fsdp_workers.py": "cbb13e80f4df109a6bb5b1771551bf7965b75a2935d5f619d54b871f83c2be80",
    "verl/workers/config/actor.py": "46b46587bdfcee7967f237f74c42094683f5cd65d86239847e610a4f7322ee93",
}


def check_pinned(root: Path = None) -> None:
    """RuntimeError unless every file in PINNED_SHA256 under the imported verl's checkout has the pinned bytes."""
    root = Path(root) if root is not None else Path(main_ppo.__file__).resolve().parents[2]
    for relative, expected in PINNED_SHA256.items():
        path = root / relative
        found = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else "missing"
        if found != expected:
            raise RuntimeError("arm D refuses this verl: %s is %s, not the pinned %s" % (path, found, expected))


# ------------------------------------------------------------------------------------------ the teacher's context
_SHOWN = False                                                    # the first teacher prompt is printed once a run


def build_sdft_teacher_batch(self, batch, reward_tensor, reward_extra_infos_dict=None):
    """Replaces RayPPOTrainer._maybe_build_self_distillation_batch (ray_trainer.py:672-796) for arm D.

    The teacher sees the student's question plus the row's verified demonstration, in the SDFT reference's template;
    the student's sampled tokens follow unchanged. No sibling, no feedback, no reward enters the context, and
    `self_distillation_mask` is all ones. Rewards are read only for the logged group metrics."""
    from verl import DataProto                                                                # noqa: PLC0415

    actor = self.config.actor_rollout_ref.actor
    if actor.policy_loss.get("loss_mode", "vanilla") != "sdpo":
        return None
    sd = actor.self_distillation
    raw_prompts = batch.non_tensor_batch["raw_prompt"]
    extra_infos = batch.non_tensor_batch["extra_info"]
    demonstrations = [objective.demonstration_of(extra) for extra in extra_infos]
    chat_kwargs = self.config.data.get("apply_chat_template_kwargs", None)
    chat_kwargs = OmegaConf.to_container(chat_kwargs, resolve=True) if chat_kwargs is not None else {}
    device = batch.batch["input_ids"].device
    built = objective.build_teacher_inputs(self.tokenizer, raw_prompts, demonstrations, batch.batch["responses"],
                                           batch.batch["response_mask"], chat_template_kwargs=chat_kwargs,
                                           max_reprompt_len=int(sd.max_reprompt_len))
    scores = reward_tensor.sum(dim=-1).detach().cpu().tolist()
    metrics = objective.group_metrics(batch.non_tensor_batch["uid"], scores, float(sd.success_reward_threshold))
    metrics["sdft/teacher_prompt_tokens_max"] = float(max(built["teacher_prompt_lengths"]))
    global _SHOWN
    if not _SHOWN:                                   # once a run: the smoke reads the teacher's exact context in console.log
        _SHOWN = True
        width = built["teacher_input_ids"].shape[1] - batch.batch["responses"].shape[1]
        ids = built["teacher_input_ids"][0, :width][built["teacher_attention_mask"][0, :width].bool()]
        print("arm D teacher prompt, sample 0 of the first batch:\n%s\n[end of teacher prompt]" % self.tokenizer.decode(ids), flush=True)
    tensors = {key: built[key].to(device) for key in
               ("teacher_input_ids", "teacher_attention_mask", "teacher_position_ids", "self_distillation_mask")}
    return DataProto.from_dict(tensors=tensors), metrics


def install_trainer_patch() -> None:
    check_pinned()
    ray_trainer.RayPPOTrainer._maybe_build_self_distillation_batch = build_sdft_teacher_batch
    print("arm D (kit/sdft_verl.py): teacher context = question + extra_info.demonstration, every sample a target "
          "[pid %d]" % os.getpid(), flush=True)


def install_actor_patch() -> None:
    check_pinned()
    from kit.sdft_actor import update_policy, forward_micro_batch
    dp_actor.DataParallelPPOActor.update_policy = update_policy
    dp_actor.DataParallelPPOActor._forward_micro_batch = forward_micro_batch
    # Retained diagnostic callable; production updates never request full logps.
    dp_actor.compute_self_distillation_loss = objective.self_distillation_loss
    print("arm D (kit/sdft_verl.py): loss = full-vocabulary forward KL(teacher || student), SDFT reference reduction "
          "[pid %d]" % os.getpid(), flush=True)


# ------------------------------------------------------------------------------------------------ the two classes
class SdftActorRolloutRefWorker(AsyncActorRolloutRefWorker):
    """The pinned worker; constructing it installs arm D's loss in its process."""

    def __init__(self, *args, **kwargs):
        install_actor_patch()
        from kit.v4_timing import install_telemetry
        install_telemetry(dp_actor.DataParallelPPOActor,updates='update_policy')
        install_telemetry(AsyncActorRolloutRefWorker,memory_methods=('init_model','save_checkpoint'))
        from kit.v4_state import install_worker
        install_worker(AsyncActorRolloutRefWorker)
        super().__init__(*args, **kwargs)


class SdftTaskRunner(main_ppo.TaskRunner):
    """The pinned runner; it refuses a configuration that is not arm D, installs the teacher context and registers the
    arm-D worker in place of the pinned one."""

    def run(self, config):
        objective.check_trainer_config(OmegaConf.to_container(config, resolve=True))
        from kit.v4_contract import check_resolved
        from kit.v4_run import check_data
        check_resolved(OmegaConf.to_container(config, resolve=True), 'D', profile_name=os.environ.get('V4_PROFILE','scientific'))
        check_data({**os.environ, 'ARM':'D'})
        install_trainer_patch()
        from kit.v4_timing import install_telemetry
        from kit.v4_restore_check import install_trainer_probe
        install_trainer_probe(ray_trainer.RayPPOTrainer,save='_save_checkpoint',load='_load_checkpoint')
        install_telemetry(ray_trainer.RayPPOTrainer,exports='_save_checkpoint')
        return super().run(config)

    def add_actor_rollout_worker(self, config):
        actor_rollout_cls, group_cls = super().add_actor_rollout_worker(config)
        role = ray_trainer.Role.ActorRolloutRef
        if actor_rollout_cls is not AsyncActorRolloutRefWorker or role not in self.role_worker_mapping:
            raise RuntimeError("arm D needs the legacy FSDP worker with its colocated teacher (actor.strategy fsdp/fsdp2, "
                               "use_legacy_worker_impl auto), got %r" % (actor_rollout_cls,))
        self.role_worker_mapping[role] = ray.remote(SdftActorRolloutRefWorker)
        return SdftActorRolloutRefWorker, group_cls


def main() -> None:
    check_pinned()
    main_ppo.TaskRunner = SdftTaskRunner          # run_ppo reads this name when no runner class is passed (main_ppo.py:79-80)
    main_ppo.main()


__all__ = ["PINNED_SHA256", "SdftActorRolloutRefWorker", "SdftTaskRunner", "build_sdft_teacher_batch", "check_pinned",
           "install_actor_patch", "install_trainer_patch", "main"]
