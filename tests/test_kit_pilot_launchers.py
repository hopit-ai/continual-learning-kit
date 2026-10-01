"""The 8B launchers' recipe knobs (plan v3, package 3): DATASET, LR, MINI_BATCH on the GRPO launcher and DATASET, LR,
TEACHER_RATE on the SDPO launcher.

Both launchers ran on the partner's node before these knobs existed (K0, K4a, K1c). The contract here is that an
unset knob leaves the command exactly as it was, that each knob changes only the tokens it names, and that a value
the trainer would choke on is refused before a GPU is taken.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

KIT = Path(__file__).resolve().parents[1] / "kit"
GRPO, SDPO = KIT / "run_grpo_toolalpaca.sh", KIT / "run_sdpo_toolalpaca.sh"
KNOBS = ("DATASET", "LR", "MINI_BATCH", "TEACHER_RATE", "KEEP_TRAINER_CKPT", "LORA", "FEEDBACK", "SOFT", "TEMP", "SEED", "STEPS", "TEST_FREQ", "NGPU", "TP", "OFFLOAD")


def dry(script: Path, **extra):
    env = {k: v for k, v in os.environ.items() if k not in KNOBS}
    env.update({"SDPO_DIR": "/ref/SDPO", "MODEL_DIR": "/models/Qwen3-8B", "NAME": "run-a1", "WORK": "/work", "DRY_RUN": "1", **extra})
    done = subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True)
    return done.returncode, done.stdout.split("\n")[:-1], done.stderr


def changed(script: Path, **extra) -> list:
    _, plain, _ = dry(script)
    code, argv, err = dry(script, **extra)
    assert code == 0, err
    assert len(argv) == len(plain), "a recipe knob replaces values, it adds no key"
    return [(a, b) for a, b in zip(plain, argv) if a != b]


def test_unset_knobs_name_the_values_the_partner_already_ran():
    _, grpo, _ = dry(GRPO)
    assert "data.train_files=[/ref/SDPO/datasets/tooluse/train.parquet]" in grpo and "vars.task=datasets/tooluse" in grpo
    assert "actor_rollout_ref.actor.optim.lr=1e-5" in grpo and "actor_rollout_ref.actor.ppo_mini_batch_size=32" in grpo
    _, sdpo, _ = dry(SDPO)
    assert "data.val_files=[/ref/SDPO/datasets/tooluse/test.parquet]" in sdpo and "actor_rollout_ref.actor.optim.lr=1e-5" in sdpo
    assert "actor_rollout_ref.actor.self_distillation.teacher_update_rate=0.05" in sdpo
    assert dry(GRPO, DATASET="datasets/tooluse", MINI_BATCH="32")[1] == grpo and dry(GRPO, LR="1e-5")[1] == grpo
    assert dry(SDPO, DATASET="datasets/tooluse", LR="1e-5", TEACHER_RATE="0.05")[1] == sdpo


def test_the_tuned_grpo_recipe_changes_the_rate_and_the_minibatch_and_nothing_else():
    assert changed(GRPO, LR="1e-6", MINI_BATCH="8") == [
        ("actor_rollout_ref.actor.optim.lr=1e-5", "actor_rollout_ref.actor.optim.lr=1e-6"),
        ("actor_rollout_ref.actor.ppo_mini_batch_size=32", "actor_rollout_ref.actor.ppo_mini_batch_size=8")]
    _, lora, _ = dry(GRPO, LORA="1")
    assert "actor_rollout_ref.actor.optim.lr=1e-4" in lora, "the LoRA arm keeps its own default rate when LR is unset"


def test_a_frozen_teacher_is_one_value():
    assert changed(SDPO, TEACHER_RATE="0") == [
        ("actor_rollout_ref.actor.self_distillation.teacher_update_rate=0.05", "actor_rollout_ref.actor.self_distillation.teacher_update_rate=0")]
    assert changed(SDPO, LR="1e-6") == [("actor_rollout_ref.actor.optim.lr=1e-5", "actor_rollout_ref.actor.optim.lr=1e-6")]


@pytest.mark.parametrize("script", [GRPO, SDPO])
def test_the_dataset_knob_moves_the_three_places_that_name_the_task(script):
    assert [a.split("=")[0] for a, _ in changed(script, DATASET="datasets/sciknoweval/chemistry")] == ["data.train_files", "data.val_files", "vars.task"]
    _, argv, _ = dry(script, DATASET="datasets/sciknoweval/chemistry")
    assert "data.train_files=[/ref/SDPO/datasets/sciknoweval/chemistry/train.parquet]" in argv and "vars.task=datasets/sciknoweval/chemistry" in argv
    text = script.read_text()
    assert 'TASK="$DATASET"' in text and '"$SDPO_DIR/$DATASET/$f"' in text, "the exported TASK and the parquet check follow the knob too"
    assert '"dataset": "$DATASET"' in text and '"model_dir": "$MODEL_DIR"' in text


def test_a_second_stage_starts_from_the_first_stages_merged_model():
    """Chaining needs no knob: MODEL_DIR is the previous run's hf-step folder, and under SDPO the teacher starts there too."""
    for script in (GRPO, SDPO):
        _, argv, _ = dry(script, MODEL_DIR="/work/runs/chem-a1/hf-step40")
        assert "actor_rollout_ref.model.path=/work/runs/chem-a1/hf-step40" in argv


@pytest.mark.parametrize("script, bad", [
    (GRPO, {"DATASET": "../secrets"}), (GRPO, {"DATASET": "/abs/path"}), (GRPO, {"DATASET": "datasets//x"}), (GRPO, {"MINI_BATCH": "7"}),
    (GRPO, {"MINI_BATCH": "0"}), (GRPO, {"MINI_BATCH": "64"}), (GRPO, {"LR": "fast"}), (SDPO, {"TEACHER_RATE": "2"}), (SDPO, {"TEACHER_RATE": "-0.1"}),
    (SDPO, {"TEACHER_RATE": "five"}), (SDPO, {"LR": "1e"}), (SDPO, {"DATASET": "tooluse"})])
def test_a_value_the_trainer_would_choke_on_is_refused_before_a_gpu_is_taken(script, bad):
    code, argv, err = dry(script, **bad)
    assert code == 2 and argv == [] and "must" in err, (bad, err)


@pytest.mark.parametrize("rate", ["0", "0.0", "0.05", "0.5", "1", "1.0", ".25"])
def test_teacher_rates_from_zero_to_one_are_accepted(rate):
    assert dry(SDPO, TEACHER_RATE=rate)[0] == 0


def test_the_teacher_rate_is_a_recipe_not_an_arm():
    """The one-arm rule still counts only FEEDBACK, SOFT and TEMP: a frozen teacher with feedback is one arm."""
    assert dry(SDPO, TEACHER_RATE="0", FEEDBACK="1")[0] == 0
    assert dry(SDPO, FEEDBACK="1", SOFT="1")[0] == 2


# ------------------------------------------------------------------------- KEEP_TRAINER_CKPT (the review of send 4, finding 4)
def test_the_trainer_checkpoint_knob_changes_no_argument_and_refuses_what_it_cannot_do():
    for script in (GRPO, SDPO):
        assert dry(script, KEEP_TRAINER_CKPT="0")[1] == dry(script)[1], "it acts after the merge; the trainer's command is untouched"
        code, argv, err = dry(script, KEEP_TRAINER_CKPT="yes")
        assert code == 2 and argv == [] and "KEEP_TRAINER_CKPT must be 0 or 1" in err
    code, _argv, err = dry(GRPO, KEEP_TRAINER_CKPT="0", LORA="1")
    assert code == 2 and "full training only" in err, "the LoRA arm's adapter lives in the trainer's checkpoint"


FAKE_TRAINER = """import sys
from pathlib import Path
cfg = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
ckpt = Path(cfg["trainer.default_local_dir"]) / ("global_step_%s" % cfg["trainer.total_training_steps"]) / "actor"
ckpt.mkdir(parents=True)
(ckpt / "model_world_size_8_rank_0.pt").write_text("full-precision shard")
"""
FAKE_MERGER = """import json, os, sys
from pathlib import Path
target = Path(sys.argv[sys.argv.index("--target_dir") + 1])
target.mkdir(parents=True)
mode = os.environ.get("FAKE_MERGE", "complete")
(target / "config.json").write_text("{}")
if mode == "no-weights":
    sys.exit(0)
(target / "model-00001-of-00002.safetensors").write_text("merged weights, shard 1")
if mode == "dies-after-one-shard":
    sys.exit(1)                                  # config and one shard on disk, no index, no tokenizer
(target / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"a": "model-00001-of-00002.safetensors", "b": "model-00002-of-00002.safetensors"}}))
if mode != "shard-missing":
    (target / "model-00002-of-00002.safetensors").write_text("merged weights, shard 2")
if mode == "no-tokenizer":
    sys.exit(0)
(target / "tokenizer_config.json").write_text("{}")
sys.exit(3 if mode == "exits-nonzero-after-writing-everything" else 0)
"""


def run_for_real(tmp_path: Path, script: Path, name: str, **extra) -> tuple:
    """The launcher itself, past its dry run, against a stand-in checkout whose trainer and merger only write files."""
    if shutil.which("git") is None or shutil.which("python") is None:
        pytest.skip("needs git and a `python` on PATH, as the launchers do")
    checkout, model, work = tmp_path / "SDPO", tmp_path / "model", tmp_path / "work"
    if not checkout.exists():
        (checkout / "verl" / "trainer").mkdir(parents=True)
        (checkout / "verl" / "__init__.py").write_text("")
        (checkout / "verl" / "trainer" / "__init__.py").write_text("")
        (checkout / "verl" / "trainer" / "main_ppo.py").write_text(FAKE_TRAINER)
        (checkout / "verl" / "model_merger.py").write_text(FAKE_MERGER)
        (checkout / "verl" / "utils" / "reward_score" / "feedback").mkdir(parents=True)
        (checkout / "verl" / "utils" / "reward_score" / "feedback" / "__init__.py").write_text("")
        (checkout / "datasets" / "tooluse").mkdir(parents=True)
        for split in ("train", "test"):
            (checkout / "datasets" / "tooluse" / ("%s.parquet" % split)).write_text("")
        git = ["git", "-C", str(checkout), "-c", "user.name=t", "-c", "user.email=t@localhost"]
        for command in (["init", "-q"], ["add", "-A"], ["commit", "-q", "-m", "stand-in"]):
            subprocess.run(git + command, check=True, capture_output=True)
        model.mkdir()
        (model / "config.json").write_text("{}")
    env = {k: v for k, v in os.environ.items() if k not in KNOBS and k != "PYTHONPATH"}
    env.update({"SDPO_DIR": str(checkout), "MODEL_DIR": str(model), "NAME": name, "WORK": str(work), "STEPS": "2", **extra})
    env.pop("FAKE_MERGE", None) if "FAKE_MERGE" not in extra else None
    done = subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True)
    run = work / "runs" / name
    return done, run, json.loads((run / "run-summary.json").read_text()) if (run / "run-summary.json").is_file() else None


@pytest.mark.parametrize("script, folder", [(GRPO, "tool-grpo"), (SDPO, "tool-sdpo")])
def test_the_trainer_checkpoint_goes_only_when_asked_and_only_after_a_complete_merge(tmp_path, script, folder):
    shard = Path(folder) / "global_step_2" / "actor" / "model_world_size_8_rank_0.pt"
    done, run, summary = run_for_real(tmp_path, script, "kept")
    assert done.returncode == 0, done.stderr[-800:]
    assert (run / shard).is_file(), "the default keeps it, as every earlier campaign did"
    assert (summary["merged"], summary["merge_returncode"], summary["trainer_checkpoint_kept"]) == (1, 0, 1)
    done, run, summary = run_for_real(tmp_path, script, "dropped", KEEP_TRAINER_CKPT="0")
    assert done.returncode == 0, done.stderr[-800:]
    assert not (run / folder / "global_step_2").exists() and (run / "hf-step2" / "model-00002-of-00002.safetensors").is_file()
    assert (summary["merged"], summary["trainer_checkpoint_kept"]) == (1, 0)


@pytest.mark.parametrize("script, folder", [(GRPO, "tool-grpo"), (SDPO, "tool-sdpo")])
@pytest.mark.parametrize("mode", ["no-weights", "dies-after-one-shard", "shard-missing", "no-tokenizer", "exits-nonzero-after-writing-everything"])
def test_a_merge_that_is_not_complete_is_not_merged_fails_the_run_and_never_costs_the_checkpoint(tmp_path, script, folder, mode):
    """Round 2 of the review: a merger that wrote a config and one shard and then died was recorded as merged, and with
    KEEP_TRAINER_CKPT=0 the only recoverable copy of the weights was deleted."""
    done, run, summary = run_for_real(tmp_path, script, mode, KEEP_TRAINER_CKPT="0", FAKE_MERGE=mode)
    assert (run / folder / "global_step_2" / "actor" / "model_world_size_8_rank_0.pt").is_file(), "the checkpoint is still there"
    assert summary["returncode"] == 0 and summary["merged"] == 0 and summary["trainer_checkpoint_kept"] == 1
    assert summary["merge_returncode"] == {"dies-after-one-shard": 1, "exits-nonzero-after-writing-everything": 3}.get(mode, 0)
    assert done.returncode == 4 and "left no complete model" in done.stderr, "the row fails, so nothing is scored from it"
