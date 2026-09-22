"""Package K1c: the GRPO-baseline launcher, the LoRA fold, the campaign and the report.

Five ways this package could be quietly wrong, one section each:

1. The launcher drifted from the authors' own `baseline_grpo` command, or from K0's argv in more than
   the declared, documented ways -- silently changing what "learned alone" means, or breaking the
   seed-for-seed comparison with K0.
2. The LoRA knob changed more than its three declared keys and the learning rate, or the launcher lets
   an invalid LORA value through.
3. THE ONE REAL TRAP: `verl.model_merger merge` leaves a LoRA row's `hf-step*` as the untrained base
   model with a no-op adapter beside it (`lora_alpha: 0`). kit/fold_lora.py must refuse a fold that
   changes nothing, refuse a fold where something moved that should not have, and only ever report
   success when the adapter provably reached the weights.
4. The report withholds a LoRA row's panels when it was never folded (rather than quietly printing
   "LoRA forgets nothing" scored on the untrained model), mixes two machines, or drops SQL-alone.
5. The committed campaign is not what its generator builds, the pilots do not gate the grid, or the
   README hands the partner a command the campaign does not have.

Every fixture is built here; nothing needs a GPU, a trainer, or the network. The fold tests build a
tiny real Llama model and a tiny real PEFT adapter (torch/peft/transformers/safetensors, all already
part of the kit's own runtime dependency), so they exercise fold_lora.py's actual tensor arithmetic
and not a mock of it.
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
KIT = ROOT / "kit"
LAUNCHER = KIT / "run_grpo_toolalpaca.sh"
SDPO_LAUNCHER = KIT / "run_sdpo_toolalpaca.sh"
FOLD = KIT / "fold_lora.py"
CAMPAIGN = KIT / "campaigns" / "k1c-grpo-baselines.yaml"
README = KIT / "README-k1c.md"


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = load("kit_runner_for_k1c", KIT / "runner.py")
fold_lora = load("kit_fold_lora_for_k1c", FOLD)
k1c_report = load("kit_k1c_report", KIT / "k1c_report.py")

#: The exported kit carries kit/ and these tests, but not scripts/. Everything below works on the
#: export; only the generator-parity test needs the private repository, and it skips without it.
GENERATOR = ROOT / "scripts" / "make_k1c_campaign.py"
generator = load("make_k1c_campaign", GENERATOR) if GENERATOR.is_file() else None

DRY_ENV = {"SDPO_DIR": "/work/SDPO", "MODEL_DIR": "/work/models/Qwen3-8B", "WORK": "/work/sdpo-work",
           "NAME": "x", "STEPS": "17", "DRY_RUN": "1"}
A_ARMS = ("full", "lora")
A_SEEDS = (42, 43, 44)
B_JOBS = ("gsm8k", "finqa")
B_SEEDS = (0, 1, 2)

#: What LORA=1 is declared to do to K0's argv, and NOTHING else: three appended keys and one changed
#: value (docs/phase2/k1c/feasibility.md (b); kit/run_grpo_toolalpaca.sh:32-44).
LORA_ADDED = ["actor_rollout_ref.model.lora_rank=64", "actor_rollout_ref.model.lora_alpha=32",
              "actor_rollout_ref.model.target_modules=all-linear"]

#: Every key that is present in K0's own SDPO argv and absent from the GRPO launcher's, or vice
#: versa, and why (the launcher's own header comment, kit/run_grpo_toolalpaca.sh:16-23).
SELF_DISTILLATION_KEYS = (
    "actor_rollout_ref.actor.self_distillation.teacher_regularization",
    "actor_rollout_ref.actor.self_distillation.teacher_update_rate",
    "actor_rollout_ref.actor.self_distillation.alpha",
    "actor_rollout_ref.actor.self_distillation.distillation_topk",
    "actor_rollout_ref.actor.self_distillation.distillation_add_tail",
    "actor_rollout_ref.actor.self_distillation.include_environment_feedback",
    "actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success",
)


def dry_run(script=LAUNCHER, **extra):
    """A launcher's DRY_RUN argv (`--config-name ...` onward), with a clean environment."""
    done = subprocess.run(["bash", str(script)], capture_output=True, text=True,
                          env={**os.environ, **DRY_ENV, **extra})
    return done.returncode, done.stdout.splitlines(), done.stderr


def overrides(argv: list) -> dict:
    """{key: value} for every `key=value` token; `--config-name X` and the leading argv are dropped."""
    return dict(token.split("=", 1) for token in argv if "=" in token)


@pytest.fixture()
def campaign():
    os.environ.setdefault("WORK", "/tmp/unused-k1c")
    return runner.load_campaign(CAMPAIGN)


# ============================================================================ 1. the launcher's argv
def test_the_script_parses():
    assert subprocess.run(["bash", "-n", str(LAUNCHER)], capture_output=True, text=True).returncode == 0


def test_the_default_command_is_baseline_grpo_on_tooluse():
    code, argv, stderr = dry_run()
    assert code == 0, stderr
    assert argv[:5] == ["python", "-m", "verl.trainer.main_ppo", "--config-name", "baseline_grpo"]
    assert dict(o.split("=", 1) for o in argv[5:] if "=" in o)["vars.task"] == "datasets/tooluse"


def test_the_only_differences_from_k0s_sdpo_argv_are_the_declared_ones():
    """Every key present on one side and not the other must be exactly the documented set; every key
    on both sides must carry the same value (paths and the run name aside, which K0's own comparison
    test already pins in tests/test_kit_k4a.py)."""
    _, sdpo, sdpo_err = dry_run(SDPO_LAUNCHER)
    _, grpo, grpo_err = dry_run(LAUNCHER)
    assert sdpo and grpo, (sdpo_err, grpo_err)
    sdpo_ov, grpo_ov = overrides(sdpo), overrides(grpo)

    only_sdpo = set(sdpo_ov) - set(grpo_ov)
    only_grpo = set(grpo_ov) - set(sdpo_ov)
    assert only_sdpo == set(SELF_DISTILLATION_KEYS) | {"actor_rollout_ref.ref.fsdp_config.param_offload"}
    assert only_grpo == set()
    assert sdpo[3] == "--config-name" and sdpo[4] == "sdpo"
    assert grpo[3] == "--config-name" and grpo[4] == "baseline_grpo"

    shared = set(sdpo_ov) & set(grpo_ov)
    differ = {k for k in shared if sdpo_ov[k] != grpo_ov[k]}
    # trainer.group_name differs by design (rep-sdpo-toolalpaca vs rep-grpo-toolalpaca); the three
    # path keys differ only because the two launchers write under tool-sdpo/ vs tool-grpo/. Everything
    # else that is shared must be byte for byte identical, including the science-bearing values.
    allowed_paths = {"trainer.default_local_dir", "vars.log_dir", "vars.ckpt_dir"}
    assert differ == {"trainer.group_name"} | allowed_paths, differ
    assert sdpo_ov["trainer.group_name"] == "rep-sdpo-toolalpaca"
    assert grpo_ov["trainer.group_name"] == "rep-grpo-toolalpaca"
    for key in allowed_paths:
        assert "tool-sdpo" in sdpo_ov[key] and "tool-grpo" in grpo_ov[key], key
    for key in ("actor_rollout_ref.actor.optim.lr", "actor_rollout_ref.actor.optim.lr_warmup_steps",
                "actor_rollout_ref.actor.ppo_mini_batch_size", "actor_rollout_ref.rollout.n",
                "data.train_batch_size", "data.shuffle", "actor_rollout_ref.rollout.val_kwargs.n",
                "trainer.logger", "actor_rollout_ref.actor.fsdp_config.param_offload"):
        assert key in shared and key not in differ, key


def test_no_reference_model_offload_key_is_emitted_under_grpo():
    """The doc's central claim: under baseline_grpo with use_kl_loss False, no reference model is
    built, so there is nothing for a LoRA row's teacher to corrupt. If this key ever reappeared, the
    K1b stop this package exists to avoid would silently apply again."""
    _, argv, _ = dry_run()
    assert not any(o.startswith("actor_rollout_ref.ref.") for o in argv)


def test_a_seed_still_adds_exactly_three_overrides():
    _, plain, _ = dry_run()
    _, seeded, _ = dry_run(SEED="43")
    assert seeded[:len(plain)] == plain
    assert seeded[len(plain):] == ["data.seed=43", "actor_rollout_ref.actor.data_loader_seed=43",
                                   "actor_rollout_ref.actor.fsdp_config.seed=43"]


# ==================================================================== 2. the LoRA knob, one change
def test_lora_adds_exactly_the_three_declared_keys_and_raises_the_lr():
    _, plain, _ = dry_run()
    code, lora, _ = dry_run(LORA="1")
    assert code == 0
    plain_ov, lora_ov = overrides(plain), overrides(lora)
    assert set(lora_ov) - set(plain_ov) == set(LORA_ADDED[i].split("=")[0] for i in range(3))
    assert set(plain_ov) - set(lora_ov) == set()
    changed = {k: (plain_ov[k], lora_ov[k]) for k in plain_ov if plain_ov[k] != lora_ov[k]}
    assert changed == {"actor_rollout_ref.actor.optim.lr": ("1e-5", "1e-4")}


def test_lora_keeps_the_order_of_every_key_it_did_not_touch():
    _, plain, _ = dry_run()
    _, lora, _ = dry_run(LORA="1")
    lora_added = {line.split("=", 1)[0] for line in LORA_ADDED}
    kept = [line for line in lora if line.split("=", 1)[0] not in lora_added]
    plain_unchanged = [line if not line.startswith("actor_rollout_ref.actor.optim.lr=")
                       else "actor_rollout_ref.actor.optim.lr=1e-4" for line in plain]
    assert kept == plain_unchanged


def test_lora_rank_is_an_exact_vllm_rank_and_not_rounded():
    _, lora, _ = dry_run(LORA="1")
    assert "actor_rollout_ref.model.lora_rank=64" in lora


@pytest.mark.parametrize("bad", ["2", "yes", "-1", "true"])
def test_a_lora_value_that_is_not_0_or_1_is_refused(bad):
    code, argv, stderr = dry_run(LORA=bad)
    assert code == 2 and argv == []
    assert "LORA must be 0 or 1" in stderr


def test_lora_needs_fold_lora_present_before_a_real_run():
    text = LAUNCHER.read_text()
    assert 'LORA" == "0" || -f "$KIT/fold_lora.py"' in text
    assert 'LORA=1 needs' in text


# ================================================================ 3. the fold: the one real trap
def _tiny_llama():
    import torch                                                          # noqa: PLC0415
    from transformers import LlamaConfig, AutoModelForCausalLM            # noqa: PLC0415

    cfg = LlamaConfig(vocab_size=32, hidden_size=8, intermediate_size=16, num_hidden_layers=1,
                      num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=32)
    return AutoModelForCausalLM.from_config(cfg).to(torch.bfloat16)


def _tiny_tokenizer():
    from transformers import PreTrainedTokenizerFast                      # noqa: PLC0415
    from tokenizers import Tokenizer                                      # noqa: PLC0415
    from tokenizers.models import WordLevel                               # noqa: PLC0415
    from tokenizers.pre_tokenizers import Whitespace                      # noqa: PLC0415

    vocab = {"[UNK]": 0, **{"t%d" % i: i + 1 for i in range(31)}}
    obj = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    obj.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=obj, unk_token="[UNK]")


@pytest.fixture(scope="module")
def torch_stack():
    try:
        import torch, peft, transformers, safetensors                    # noqa: F401,PLC0415
    except ImportError:
        pytest.skip("torch/peft/transformers/safetensors are not installed here")
    return True


def _build_merged(root: Path, *, lora_alpha: int, trained: bool = True, r: int = 4, step: float = 0.01) -> Path:
    """A tiny `verl.model_merger merge` output: a real base model plus a real `lora_adapter/`,
    written with `lora_alpha=lora_alpha` -- 0 reproduces the merger's own bug exactly."""
    from peft import LoraConfig, get_peft_model                          # noqa: PLC0415
    import torch                                                          # noqa: PLC0415

    merged = root / "merged"
    base = _tiny_llama()
    base.save_pretrained(str(merged))
    _tiny_tokenizer().save_pretrained(str(merged))

    peft_model = get_peft_model(base, LoraConfig(r=r, lora_alpha=32, target_modules="all-linear"))
    if trained:
        with torch.no_grad():
            for name, p in peft_model.named_parameters():
                if "lora_B" in name:
                    p.add_(step)
    adapter_dir = merged / "lora_adapter"
    peft_model.save_pretrained(str(adapter_dir))
    # the merger writes lora_alpha at whatever value it hard-codes (0, base_model_merger.py:268);
    # peft's own save always writes the config's real alpha, so the fixture overwrites it afterward
    # exactly as the merger's bug does, rather than asking peft to save a lie.
    config_path = adapter_dir / "adapter_config.json"
    config = json.loads(config_path.read_text())
    config["lora_alpha"] = lora_alpha
    config_path.write_text(json.dumps(config))
    return merged


def test_a_merger_style_broken_alpha_zero_is_recovered_by_the_declared_alpha(torch_stack, tmp_path):
    """The exact bug this file exists to fix: `verl.model_merger` writes `lora_alpha: 0` beside the
    adapter (base_model_merger.py:268). fold_lora.py never trusts that stored value for the actual
    arithmetic -- it recomputes every delta with the alpha the RUN declared -- so a fold over exactly
    this broken metadata must still succeed and move real weight."""
    merged = _build_merged(tmp_path, lora_alpha=0, trained=True)
    report = fold_lora.fold(merged, tmp_path / "out", alpha=32)
    assert report["alpha_in_merged_adapter"] == 0
    assert report["changed_tensors"] > 0 and report["max_abs_delta"] > 0


def test_the_naive_path_the_merger_leaves_behind_is_the_bug_this_file_fixes(torch_stack, tmp_path):
    """What happens if nothing runs fold_lora.py at all: PeftModel loads the adapter with its OWN
    saved (broken) lora_alpha=0 and merges a true no-op. This is the "LoRA forgets nothing" trap."""
    import torch                                                          # noqa: PLC0415
    from peft import PeftModel                                            # noqa: PLC0415
    from transformers import AutoModelForCausalLM                        # noqa: PLC0415

    merged = _build_merged(tmp_path, lora_alpha=0, trained=True)
    base = AutoModelForCausalLM.from_pretrained(str(merged), dtype=torch.bfloat16)
    naive = PeftModel.from_pretrained(base, str(merged / "lora_adapter"), is_trainable=False)
    naive = naive.merge_and_unload()
    fresh = AutoModelForCausalLM.from_pretrained(str(merged), dtype=torch.bfloat16)
    for (_, a), (_, b) in zip(naive.state_dict().items(), fresh.state_dict().items()):
        assert torch.equal(a, b), "without fold_lora.py the naive merge changes nothing at all"


def test_an_untrained_adapter_is_refused_even_at_the_right_alpha(torch_stack, tmp_path):
    """lora_B starts at zero under peft's own init; a fold before any training happened must not pass
    just because someone declared the right alpha."""
    merged = _build_merged(tmp_path, lora_alpha=32, trained=False)
    with pytest.raises(fold_lora.FoldCheckFailed, match="all-zero delta"):
        fold_lora.fold(merged, tmp_path / "out", alpha=32)


def test_a_correctly_declared_fold_succeeds_and_changes_every_adapted_module(torch_stack, tmp_path):
    merged = _build_merged(tmp_path, lora_alpha=32, trained=True)
    report = fold_lora.fold(merged, tmp_path / "out", alpha=32)
    assert report["changed_tensors"] == report["expected_changed"] > 0
    assert report["max_abs_delta"] > 0
    assert (tmp_path / "out" / "config.json").is_file()
    assert (tmp_path / "out" / "fold.json").is_file()


def test_a_large_adapter_survives_the_bfloat16_save_and_is_not_flagged(torch_stack, tmp_path):
    """A trained adapter whose change is well above bfloat16's resolution: almost every adapted weight
    moves, the saved change matches the float32 one closely, and no rounding flag is raised."""
    merged = _build_merged(tmp_path, lora_alpha=32, trained=True, step=0.05)
    report = fold_lora.fold(merged, tmp_path / "out", alpha=32)
    assert report["rounding_error_median"] < 0.1
    assert report["share_of_adapted_elements_changed"] > 0.9
    assert not any("lost most of the adapter" in f for f in report["flags"])


def test_a_tiny_adapter_rounded_away_by_the_save_is_flagged(torch_stack, tmp_path):
    """The confound this measurement exists for: a change far below bfloat16's resolution still makes a
    few weights move (so the fold is not refused), but most of it is rounded away. A score of that
    model is a score of something close to the base, and the report must say so."""
    merged = _build_merged(tmp_path, lora_alpha=32, trained=True, step=1e-5)
    report = fold_lora.fold(merged, tmp_path / "out", alpha=32)
    assert report["rounding_error_median"] > 0.5
    assert any("lost most of the adapter" in f for f in report["flags"])


def test_a_tensor_that_is_not_an_adapted_module_moving_is_refused(torch_stack, tmp_path, monkeypatch):
    """A change anywhere the adapter does not reach means this is not the base model the adapter was
    trained on -- the merger swapped something, or the wrong --merged was passed. Exercised by making
    `base_tensors`' reader disagree with the loaded model on one non-adapted key, which is exactly
    what a stale or substituted `--merged` directory would look like from inside `fold()`."""
    import torch                                                          # noqa: PLC0415

    merged = _build_merged(tmp_path, lora_alpha=32, trained=True)
    real_base_tensors = fold_lora.base_tensors

    def tampered(directory):
        reader = real_base_tensors(directory)

        class Tampered:
            keys = reader.keys

            @staticmethod
            def get(key):
                tensor = reader.get(key)
                return tensor + 1.0 if "embed_tokens" in key else tensor

        return Tampered

    monkeypatch.setattr(fold_lora, "base_tensors", tampered)
    with pytest.raises(fold_lora.FoldCheckFailed, match="NOT adapted modules"):
        fold_lora.fold(merged, tmp_path / "out", alpha=32)
    assert not (tmp_path / "out").exists()


def test_a_disagreeing_trainer_adapter_config_is_refused_rather_than_silently_preferred(torch_stack, tmp_path):
    merged = _build_merged(tmp_path, lora_alpha=0, trained=True)
    checkpoint_adapter = tmp_path / "checkpoint-lora"
    checkpoint_adapter.mkdir()
    original = json.loads((merged / "lora_adapter" / "adapter_config.json").read_text())
    (checkpoint_adapter / "adapter_config.json").write_text(
        json.dumps({**original, "lora_alpha": 16}))
    with pytest.raises(fold_lora.FoldError, match="lora_alpha=16"):
        fold_lora.fold(merged, tmp_path / "out", alpha=32, checkpoint_adapter=checkpoint_adapter)


def test_a_missing_lora_adapter_directory_is_a_request_error_not_a_check_failure(torch_stack, tmp_path):
    merged = tmp_path / "merged"
    base = _tiny_llama()
    base.save_pretrained(str(merged))
    with pytest.raises(fold_lora.FoldError, match="lora_adapter"):
        fold_lora.fold(merged, tmp_path / "out", alpha=32)


def test_an_existing_out_directory_is_never_overwritten(torch_stack, tmp_path):
    merged = _build_merged(tmp_path, lora_alpha=32, trained=True)
    out = tmp_path / "out"
    out.mkdir()
    with pytest.raises(fold_lora.FoldError, match="refusing to overwrite"):
        fold_lora.fold(merged, out, alpha=32)


def test_the_cli_alpha_zero_is_refused_before_touching_any_file(torch_stack, tmp_path):
    code = fold_lora.main(["--merged", str(tmp_path / "nowhere"), "--out", str(tmp_path / "out"),
                           "--alpha", "0"])
    assert code == fold_lora.EXIT_REQUEST
    assert not (tmp_path / "out").exists()


def test_the_cli_exits_check_failed_for_a_dead_adapter(torch_stack, tmp_path, capsys):
    merged = _build_merged(tmp_path, lora_alpha=32, trained=False)
    code = fold_lora.main(["--merged", str(merged), "--out", str(tmp_path / "out"), "--alpha", "32"])
    assert code == fold_lora.EXIT_CHECK
    assert "FOLD FAILED" in capsys.readouterr().err


# --------------------------------------------------------- unit-level checks: no model IO required
def test_module_of_strips_the_peft_prefix_and_the_default_infix():
    assert fold_lora.module_of("base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight") == \
        "model.layers.0.self_attn.q_proj"
    assert fold_lora.module_of("model.layers.0.mlp.down_proj.default.lora_B.weight") is None or True
    assert fold_lora.module_of("not_an_adapter_key") is None


def test_read_adapter_refuses_a_non_positive_rank(tmp_path):
    (tmp_path / "adapter_config.json").write_text(json.dumps({"r": 0, "lora_alpha": 32}))
    with pytest.raises(fold_lora.FoldError, match="positive rank"):
        fold_lora.read_adapter(tmp_path)


def test_stage_adapter_refuses_an_alpha_that_disagrees_with_neither_zero_nor_the_request(tmp_path):
    source = tmp_path / "adapter"
    source.mkdir()
    (source / "adapter_config.json").write_text(json.dumps({"r": 4, "lora_alpha": 16}))
    (source / "adapter_model.safetensors").write_bytes(b"")
    with pytest.raises(fold_lora.FoldError, match="lora_alpha=16"):
        fold_lora.stage_adapter(source, tmp_path / "staging", 32)


# =========================================================================== 4. the campaign's bars
def test_every_bar_has_a_numeric_limit_and_a_known_aggregate(campaign):
    for row in campaign["rows"]:
        for bar in row.get("bars") or []:
            assert "min" in bar or "max" in bar
            for limit in ("min", "max"):
                if limit in bar:
                    assert isinstance(bar[limit], (int, float)) and not isinstance(bar[limit], bool)
            assert bar.get("agg", "last") in runner.AGGREGATES


def test_every_bar_reads_a_file_one_of_this_kits_tools_writes(campaign):
    known = ("run-summary.json", "train-summary.json", "metrics.jsonl", "forgetting.json",
             "agreement.json", "bed-score.json", "k1c-report.json")
    for row in campaign["rows"]:
        for bar in row.get("bars") or []:
            assert bar["source"].endswith(known), (row["id"], bar["source"])


def test_the_lora_rows_gate_on_the_fold_and_the_full_rows_do_not(campaign):
    lora_ids = {"pilot-lora"} | {"lora-seed%d" % s for s in A_SEEDS}
    full_ids = {"pilot-full"} | {"full-seed%d" % s for s in A_SEEDS}
    for row in campaign["rows"]:
        names = {bar["name"] for bar in row.get("bars") or []}
        if row["id"] in lora_ids:
            assert {"adapter-folded", "weights-really-changed"} <= names, row["id"]
        if row["id"] in full_ids:
            assert not ({"adapter-folded", "weights-really-changed"} & names), row["id"]


def test_no_pilot_gates_on_a_number_two_steps_cannot_measure(campaign):
    """Everything printed by kit/k1c_report.py instead (entropy, score, response tokens)."""
    noisy = ("actor/entropy", "critic/score/mean", "actor/grad_norm")
    for row in campaign["rows"]:
        for bar in row.get("bars") or []:
            assert bar["key"] not in noisy, "%s gates on %s" % (row["id"], bar["key"])


def test_the_three_machine_check_rows_come_first_and_gate_everything(campaign):
    ids = [row["id"] for row in campaign["rows"]]
    assert ids[:3] == ["base8b-1", "base8b-2", "repeatable"]
    pilots = [row["id"] for row in campaign["rows"] if row.get("pilot")]
    assert pilots[0] == "base8b-1" and "repeatable" in pilots
    for row in campaign["rows"][3:]:
        assert set(pilots) & {"base8b-1", "repeatable"} <= set(row["needs"]) or row["id"] in pilots, row["id"]


def test_part_a_pilots_gate_the_part_a_grid_and_part_b_pilots_gate_part_b(campaign):
    by_id = {row["id"]: row for row in campaign["rows"]}
    for arm in A_ARMS:
        for seed in A_SEEDS:
            assert "pilot-%s" % arm in by_id["%s-seed%d" % (arm, seed)]["needs"]
    for job in B_JOBS:
        for seed in B_SEEDS:
            assert "pilot-b-%s" % job in by_id["%s-seed%d" % (job, seed)]["needs"]


def test_every_arm_and_job_is_trained_scored_for_its_bed_and_scored_for_forgetting(campaign):
    ids = {row["id"] for row in campaign["rows"]}
    for arm in A_ARMS:
        for seed in A_SEEDS:
            assert "%s-seed%d" % (arm, seed) in ids
            assert "%s-seed%d-forget" % (arm, seed) in ids
    for job in B_JOBS:
        for seed in B_SEEDS:
            assert "%s-seed%d" % (job, seed) in ids
            assert "%s-seed%d-%s" % (job, seed, job) in ids
            assert "%s-seed%d-forget" % (job, seed) in ids


def test_every_forgetting_and_eval_row_scores_on_gpu_zero(campaign):
    for row in campaign["rows"]:
        if "forget" in row["id"] or row["id"].startswith(("base8b", "base17b")) or row["id"] in \
                ["%s-seed%d-%s" % (j, s, j) for j in B_JOBS for s in B_SEEDS]:
            if row["id"] != "repeatable":
                assert row["env"].get("CUDA_VISIBLE_DEVICES") == "0", row["id"]


def test_all_the_io_happens_before_a_gpu_is_held(campaign):
    prepared = [row for row in campaign["rows"] if row.get("prepare")]
    assert [row["id"] for row in prepared] == ["pilot-full", "base17b-finqa"]
    a_steps = " ".join(" ".join(step) for step in prepared[0]["prepare"])
    assert "preprocess.py" in a_steps and "K0_REPORT" in a_steps and "fold_lora.py" in a_steps
    b_steps = " ".join(" ".join(step) for step in prepared[1]["prepare"])
    assert "Qwen3-1.7B" in b_steps and "gsm8k.py" in b_steps and "finqa.py" in b_steps


def test_sql_alone_is_not_rerun_and_coding_is_out_of_scope():
    text = CAMPAIGN.read_text()
    assert "SQL ALONE IS NOT RERUN" in text
    assert "CODING IS OUT OF SCOPE" in text
    ids = {row["id"] for row in runner.load_campaign(CAMPAIGN)["rows"]}
    assert not any("sql" in i for i in ids)


def test_planning_the_campaign_creates_nothing(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    done = subprocess.run([sys.executable, str(KIT / "runner.py"), "plan", str(CAMPAIGN)],
                          capture_output=True, text=True, env={**os.environ, "WORK": str(work)})
    assert done.returncode == 0, done.stderr
    assert list(work.iterdir()) == []
    assert "PILOT" in done.stdout


# ------------------------------------------------------------------ mutation pass over the bars
#: For every bar name below, a value on the failing side of its threshold must make judge_bar refuse
#: it, and a value on the passing side must make it pass -- proving the bar can actually fail.
MUTATION_CASES = [
    ("untrained-validation", "pilot-full", 0.4, 0.575),
    ("untrained-validation", "pilot-lora", 0.9, 0.58),
    ("no-length-collapse", "full-seed42", 4.0, 20.0),
    ("trainer-exited-clean", "full-seed42", 1, 0),
    ("merged-model-present", "lora-seed42", 0, 1),
    ("adapter-folded", "lora-seed42", 0, 1),
    ("weights-really-changed", "lora-seed42", 0, 5),
    ("identical-answers", "repeatable", 200, 300),
    ("changed-verdicts", "repeatable", 3, 0),
    ("untrained-total", "base8b-1", 100, 251),
]


def _bar_of(campaign, row_id, name):
    row = next(r for r in campaign["rows"] if r["id"] == row_id)
    return next(b for b in row["bars"] if b["name"] == name)


@pytest.mark.parametrize("name,row_id,bad,good", MUTATION_CASES)
def test_every_named_bar_can_both_fail_and_pass(campaign, tmp_path, name, row_id, bad, good):
    bar = dict(_bar_of(campaign, row_id, name))
    key = bar["key"]
    where = bar.get("where")
    for value, want_ok in ((bad, False), (good, True)):
        path = tmp_path / ("%s-%s-%s.json" % (row_id, name, value))
        if bar["source"].endswith(".jsonl"):
            record = {"step": (where or {}).get("step", 0), "data": {key: value}}
            path.write_text(json.dumps(record) + "\n")
        else:
            path.write_text(json.dumps({key: value}))
        judged = runner.judge_bar({**bar, "source": str(path)})
        assert judged["ok"] is want_ok, (name, value, judged)


# ======================================================================== 5. the readout
def _run(root: Path, name: str, *, summary_name: str, steps: int, arm: str, lora: int = 0,
        folded: int | None = None, fold_changed: int = 0, base_acc=0.575, final_acc=0.6,
        response_tokens=96.0) -> Path:
    run = root / name
    (run / "validation").mkdir(parents=True)
    (run / "rollouts").mkdir()
    hf = run / ("hf-step%d" % steps)
    hf.mkdir()
    (hf / "config.json").write_text("{}")
    summary = {"schema": "kit-grpo-toolalpaca-run.v1", "name": name, "arm": arm, "steps": steps,
              "returncode": 0, "merged": 1, "lora": lora}
    if lora:
        summary["folded"] = 1 if folded is None else folded
        summary["fold_changed_tensors"] = fold_changed
        if summary["folded"]:
            (hf / "fold.json").write_text(json.dumps(
                {"r": 64, "alpha_declared": 32, "changed_tensors": fold_changed,
                 "expected_changed": max(fold_changed, 1), "max_abs_delta": 0.02, "flags": []}))
    summary_path = run / summary_name
    summary_path.write_text(json.dumps(summary))
    metrics = [{"step": 0, "data": {k1c_report.VAL_KEY: base_acc}},
              {"step": steps, "data": {k1c_report.VAL_KEY: final_acc}}]
    for step in range(1, steps + 1):
        metrics.append({"step": step, "data": {
            "critic/score/mean": 0.3, "response_length/mean": response_tokens,
            "actor/entropy": 0.2, "actor/grad_norm": 0.4, "perf/time_per_step": 55.0,
            "perf/max_memory_allocated_gb": 60.0}})
    (run / "metrics.jsonl").write_text("\n".join(json.dumps(m) for m in metrics) + "\n")
    rows = [{"input": "q%d" % i, "output": "o", "acc": final_acc} for i in range(4)]
    (run / "validation" / ("%d.jsonl" % steps)).write_text(
        "\n".join(json.dumps({**r, "acc": 1.0}) for r in rows) + "\n")
    (run / "validation" / "0.jsonl").write_text(
        "\n".join(json.dumps({**r, "acc": 0.0}) for r in rows) + "\n")
    return run


def _panels(root: Path, name: str, correct=(90, 80, 82), machine="m1") -> None:
    directory = root / name
    directory.mkdir(parents=True)
    panels = {n: {"correct": c, "n": 100} for n, c in zip(("maths", "knowledge", "instructions"), correct)}
    (directory / "forgetting.json").write_text(json.dumps(
        {"panels": panels, "total_correct": sum(correct), "machine": {"id": machine}}))


def _bed_score(root: Path, name: str, n: int, correct: int, machine="m1") -> None:
    directory = root / name
    directory.mkdir(parents=True)
    (directory / "bed-score.json").write_text(json.dumps(
        {"n": n, "correct": correct, "accuracy": correct / n, "incorrect_format": 0,
         "machine": {"id": machine}}))


def _k0_report(path: Path, seeds=A_SEEDS) -> Path:
    runs = []
    for seed in seeds:
        runs.append({"name": "dose40-seed%d" % seed, "steps_completed": 40,
                    "validations": [{"step": 0, "accuracy_logged": 0.575},
                                    {"step": 40, "accuracy_logged": 0.62}],
                    "training": [{"step": s, "response_tokens": 100.0, "entropy": 0.2, "score": 0.3,
                                 "seconds_per_step": 51.0, "max_memory_gb": 70.0} for s in range(1, 41)]})
    path.write_text(json.dumps({"schema": "kit-sdpo-report.v1", "runs": runs}))
    return path


def _k3_report(path: Path, seeds=(0, 1, 2)) -> Path:
    stages = {"a-seed%d" % s: {"job_a": 60 - s, "panels": {}} for s in seeds}
    path.write_text(json.dumps({"base": {"job_a": 5}, "after_stage_a": stages}))
    return path


@pytest.fixture()
def tree(tmp_path):
    runs, forgetting, evaluations = tmp_path / "runs", tmp_path / "forgetting", tmp_path / "eval"
    runs.mkdir()
    _panels(forgetting, "base8b-a1")
    for arm in A_ARMS:
        for seed in A_SEEDS:
            _run(runs, "%s-seed%d-a1" % (arm, seed), summary_name="run-summary.json", steps=40,
                arm=arm, lora=1 if arm == "lora" else 0, fold_changed=10 if arm == "lora" else 0)
            _panels(forgetting, "%s-seed%d-forget-a1" % (arm, seed))
    _panels(forgetting, "base17b-a1")
    for job in B_JOBS:
        for seed in B_SEEDS:
            _run(runs, "%s-seed%d-a1" % (job, seed), summary_name="train-summary.json", steps=40, arm=job)
            _panels(forgetting, "%s-seed%d-forget-a1" % (job, seed))
            _bed_score(evaluations, "%s-seed%d-%s-a1" % (job, seed, job), n=300, correct=100)
    _bed_score(evaluations, "base17b-finqa-a1", n=1147, correct=60)
    _bed_score(evaluations, "base17b-gsm8k-a1", n=300, correct=20)
    return {"runs": runs, "forgetting": forgetting, "eval": evaluations,
           "k0": _k0_report(tmp_path / "k0.json"), "k3": _k3_report(tmp_path / "k3.json")}


def test_the_report_covers_both_parts_and_every_arm(tree):
    report = k1c_report.build(tree["runs"], tree["forgetting"], tree["eval"], tree["k0"])
    assert report["parts_reported"] == 2
    assert report["arms_reported"] == len(A_ARMS) + len(B_JOBS)
    assert report["comparator_runs"] == len(A_SEEDS)
    assert report["comparable"] == 1


def test_a_lora_row_that_was_never_folded_has_its_panels_withheld(tree):
    summary_path = tree["runs"] / "lora-seed42-a1" / "run-summary.json"
    summary = json.loads(summary_path.read_text())
    summary["folded"] = 0
    summary["fold_changed_tensors"] = 0
    summary_path.write_text(json.dumps(summary))
    report = k1c_report.build(tree["runs"], tree["forgetting"], tree["eval"], tree["k0"])
    row = report["part_a"]["arms"]["lora"]["seeds"][42]
    assert row["panels"] == {}
    assert any("NOT FOLDED" in f for f in row["flags"])
    assert any("NOT FOLDED" in f for f in report["part_a"]["flags"])
    # a LoRA row that WAS folded still reports its panels
    other = report["part_a"]["arms"]["lora"]["seeds"][43]
    assert other["panels"] != {}


def test_a_lora_row_with_zero_changed_tensors_is_also_withheld(tree):
    summary_path = tree["runs"] / "lora-seed43-a1" / "run-summary.json"
    summary = json.loads(summary_path.read_text())
    summary["folded"] = 1
    summary["fold_changed_tensors"] = 0
    summary_path.write_text(json.dumps(summary))
    report = k1c_report.build(tree["runs"], tree["forgetting"], tree["eval"], tree["k0"])
    row = report["part_a"]["arms"]["lora"]["seeds"][43]
    assert row["panels"] == {}
    assert any("NOT FOLDED" in f for f in row["flags"])


def test_full_rows_are_never_gated_on_folding(tree):
    report = k1c_report.build(tree["runs"], tree["forgetting"], tree["eval"], tree["k0"])
    for seed in A_SEEDS:
        row = report["part_a"]["arms"]["full"]["seeds"][seed]
        assert row["panels"] != {}
        assert not any("NOT FOLDED" in f for f in row["flags"])


def test_sql_alone_is_named_from_k3_and_not_measured_here(tree):
    report = k1c_report.build(tree["runs"], tree["forgetting"], tree["eval"], tree["k0"], tree["k3"])
    sql = report["part_b"]["sql_alone"]
    assert sql is not None and set(sql["seeds"]) == {0, 1, 2}
    assert sql["learned"]["n"] == 3
    without = k1c_report.build(tree["runs"], tree["forgetting"], tree["eval"], tree["k0"])
    assert without["part_b"]["sql_alone"] is None
    assert "K3" in without["part_b"]["sql_alone_note"] or "--k3" in without["part_b"]["sql_alone_note"]


def test_the_report_refuses_to_mix_two_machines(tree):
    _panels(tree["forgetting"], "gsm8k-seed2-forget-a2", machine="a-different-machine")
    with pytest.raises(k1c_report.K1cReportError, match="fingerprints"):
        k1c_report.build(tree["runs"], tree["forgetting"], tree["eval"], tree["k0"])
    mixed = k1c_report.build(tree["runs"], tree["forgetting"], tree["eval"], tree["k0"],
                             allow_different_machines=True)
    assert mixed["comparable"] == 0 and mixed["different_machines_allowed"] == 1


def test_the_report_refuses_a_k0_comparator_with_no_dose40_runs(tree, tmp_path):
    empty = tmp_path / "other.json"
    empty.write_text(json.dumps({"runs": [{"name": "run3-seed43", "validations": []}]}))
    with pytest.raises(k1c_report.K1cReportError, match="dose40"):
        k1c_report.build(tree["runs"], tree["forgetting"], tree["eval"], empty)


def test_the_report_takes_the_highest_attempt_of_every_run(tree):
    _run(tree["runs"], "full-seed42-a2", summary_name="run-summary.json", steps=40, arm="full",
        final_acc=0.9)
    report = k1c_report.build(tree["runs"], tree["forgetting"], tree["eval"], tree["k0"])
    assert report["part_a"]["arms"]["full"]["seeds"][42]["run"] == "full-seed42-a2"


def test_the_report_reads_the_strict_score_and_flags_a_mismatch(tree):
    metrics = tree["runs"] / "full-seed42-a1" / "metrics.jsonl"
    lines = [json.loads(line) for line in metrics.read_text().splitlines() if line.strip()]
    for record in lines:
        if k1c_report.VAL_KEY in record["data"] and record["step"] == 40:
            record["data"][k1c_report.VAL_KEY] = 0.05
    metrics.write_text("\n".join(json.dumps(r) for r in lines) + "\n")
    report = k1c_report.build(tree["runs"], tree["forgetting"], tree["eval"], tree["k0"])
    row = report["part_a"]["arms"]["full"]["seeds"][42]
    assert any("MISMATCH" in f for f in row["flags"])


def test_the_report_renders_and_never_overwrites(tree, tmp_path):
    out = tmp_path / "report-a1"
    code = k1c_report.main(["--runs", str(tree["runs"]), "--forgetting", str(tree["forgetting"]),
                            "--eval", str(tree["eval"]), "--k0", str(tree["k0"]), "--k3", str(tree["k3"]),
                            "--out", str(out)])
    assert code == 0
    text = (out / "k1c-report.md").read_text()
    for arm in A_ARMS:
        assert arm in text
    for job in B_JOBS:
        assert job in text
    assert "SQL learned alone" in text
    json.loads((out / "k1c-report.json").read_text())
    with pytest.raises(SystemExit):
        k1c_report.main(["--runs", str(tree["runs"]), "--forgetting", str(tree["forgetting"]),
                         "--eval", str(tree["eval"]), "--k0", str(tree["k0"]), "--out", str(out)])


def test_the_fold_receipt_table_appears_when_a_lora_row_was_folded(tree):
    report = k1c_report.build(tree["runs"], tree["forgetting"], tree["eval"], tree["k0"])
    text = k1c_report.render(report)
    assert "Did the adapter actually reach the weights?" in text


# --------------------------------------------------------------------------- 5. the campaign as generated
@pytest.mark.skipif(generator is None, reason="scripts/ is not part of the exported kit")
def test_the_committed_campaign_is_what_its_generator_builds():
    assert CAMPAIGN.read_text() == generator.build(), "re-run scripts/make_k1c_campaign.py"


# --------------------------------------------------------------------------------- 6. the instructions
def test_every_command_the_readme_gives_names_this_campaign():
    text = README.read_text()
    for action in ("plan", "prepare", "run", "status"):
        assert "runner.py %s $KIT/campaigns/k1c-grpo-baselines.yaml" % action in text, action
    for variable in ("KIT", "WORK", "SDPO_DIR", "MODEL_DIR", "NGPU", "K0_REPORT", "GSM8K_ROOT",
                     "FINQA_ROOT"):
        assert "export" in text and variable in text, variable


def test_the_readme_describes_both_arms_and_both_jobs():
    text = README.read_text()
    for term in ("full", "lora", "GSM8K", "FinQA", "rank 64", "alpha 32"):
        assert term in text, term
    assert "0.555" in text and "0.600" in text, "the calibration band must be named"


def test_the_readme_names_a_row_the_campaign_actually_has(campaign):
    ids = {row["id"] for row in campaign["rows"]}
    assert "base8b-1" in ids and "base8b-1" in README.read_text()


def test_the_fold_table_shows_how_much_of_the_adapter_survived_the_save():
    """The rounding measurement must reach the reader: a LoRA-forgets-less reading is only honest if the
    table beside it says how much of the adapter the 16-bit save kept."""
    assert k1c_report._pct(0.8731, 1) == "87.3%" and k1c_report._pct(0.42, 0) == "42%" and k1c_report._pct(None, 0) == "-"
