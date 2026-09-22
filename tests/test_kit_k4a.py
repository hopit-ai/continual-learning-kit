"""Package K4a: the launcher's arm knobs, the soft reward, the campaign and the report.

Six ways this package could be quietly wrong, one section each:

1. The K0 command moved. Every K4a number is compared with a K0 run; if the default argv drifted by
   one character the comparison would be against something else. The whole argv is pinned here.
2. An arm changed more than its one declared thing, or two arms ran in one command, so a difference
   in the result could not be attributed to anything.
3. The soft reward leaked into the measurement. The validation metric is built from `acc`; if
   kit/beds/tooluse_soft.py ever changed `acc`, or reimplemented the authors' parser instead of
   loading it, K4a would be measuring its own reward.
4. A bar reads a key nothing writes, or reads it as something other than a number, so a run that
   went wrong passes its gate.
5. The committed campaign is not what its generator builds, or the pilots do not gate the grid.
6. The README tells the partner a command the campaign does not have.

Every fixture is built here; nothing needs a GPU, a trainer, or the network. The soft-reward tests
load the REAL authors' checker out of references/SDPO, so they fail if that checkout moves.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
KIT = ROOT / "kit"
LAUNCHER = KIT / "run_sdpo_toolalpaca.sh"
CAMPAIGN = KIT / "campaigns" / "k4a-stuck-problems.yaml"
README = KIT / "README-k4a.md"
REFERENCE_TOOLUSE = ROOT / "references" / "SDPO" / "verl" / "utils" / "reward_score" / "feedback" / "tooluse.py"


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = load("kit_runner_for_k4a", KIT / "runner.py")
k4a_report = load("kit_k4a_report", KIT / "k4a_report.py")

#: The exported kit carries kit/ and these tests, but not scripts/. Everything below works on the
#: export; only the generator-parity test needs the private repository, and it skips without it.
GENERATOR = ROOT / "scripts" / "make_k4a_campaign.py"
generator = load("make_k4a_campaign", GENERATOR) if GENERATOR.is_file() else None

ARMS = ("feedback", "soft", "variation")
SEEDS = (42, 43, 44)

DRY_ENV = {"SDPO_DIR": "/work/SDPO", "MODEL_DIR": "/work/models/Qwen3-8B", "WORK": "/work/sdpo-work",
           "NAME": "x", "STEPS": "17", "DRY_RUN": "1"}

#: K0's command, key for key and value for value, as receipt 201 proved it equals run 3's. This list
#: is the contract with every number the partner has already sent us: it may not move.
K0_ARGV = [
    "python", "-m", "verl.trainer.main_ppo", "--config-name", "sdpo",
    "data.train_files=[/work/SDPO/datasets/tooluse/train.parquet]",
    "data.val_files=[/work/SDPO/datasets/tooluse/test.parquet]",
    "actor_rollout_ref.model.path=/work/models/Qwen3-8B",
    "actor_rollout_ref.actor.strategy=fsdp2",
    "actor_rollout_ref.actor.optim.lr=1e-5",
    "actor_rollout_ref.actor.optim.lr_warmup_steps=10",
    "actor_rollout_ref.actor.ppo_mini_batch_size=32",
    "actor_rollout_ref.rollout.n=8",
    "data.train_batch_size=32",
    "data.shuffle=True",
    "trainer.experiment_name=x",
    "trainer.default_local_dir=/work/sdpo-work/runs/x/tool-sdpo",
    "trainer.save_freq=17",
    "trainer.max_actor_ckpt_to_keep=4",
    "trainer.total_epochs=1",
    "trainer.val_before_train=True",
    "actor_rollout_ref.actor.checkpoint.save_contents=[model]",
    "trainer.resume_mode=disable",
    "trainer.logger=[console,file]",
    "trainer.project_name=r99-reference-runtime",
    "trainer.group_name=rep-sdpo-toolalpaca",
    "vars.dir=/work/SDPO",
    "vars.task=datasets/tooluse",
    "vars.log_dir=/work/sdpo-work/runs/x/tool-sdpo/logs",
    "vars.ckpt_dir=/work/sdpo-work/runs/x/tool-sdpo",
    "custom_reward_function.path=/work/SDPO/verl/utils/reward_score/feedback/__init__.py",
    "custom_reward_function.name=compute_score",
    "actor_rollout_ref.actor.self_distillation.teacher_regularization=ema",
    "actor_rollout_ref.actor.self_distillation.teacher_update_rate=0.05",
    "actor_rollout_ref.actor.self_distillation.alpha=0.5",
    "actor_rollout_ref.actor.self_distillation.distillation_topk=100",
    "actor_rollout_ref.actor.self_distillation.distillation_add_tail=True",
    "actor_rollout_ref.actor.self_distillation.include_environment_feedback=False",
    "actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=True",
    "trainer.total_training_steps=17",
    "actor_rollout_ref.rollout.val_kwargs.n=16",
    "trainer.test_freq=17",
    "actor_rollout_ref.actor.calculate_entropy=True",
    "actor_rollout_ref.rollout.gpu_memory_utilization=0.55",
    "actor_rollout_ref.actor.fsdp_config.param_offload=False",
    "actor_rollout_ref.actor.fsdp_config.optimizer_offload=False",
    "actor_rollout_ref.ref.fsdp_config.param_offload=False",
    "trainer.rollout_data_dir=/work/sdpo-work/runs/x/rollouts",
    "trainer.validation_data_dir=/work/sdpo-work/runs/x/validation",
    "trainer.n_gpus_per_node=4",
    "trainer.nnodes=1",
    "actor_rollout_ref.rollout.tensor_model_parallel_size=2",
]

#: What each arm's knob is allowed to do to that command, and NOTHING else.
DECLARED = {
    "feedback": {"env": {"FEEDBACK": "1"},
                 "removed": ["actor_rollout_ref.actor.self_distillation.include_environment_feedback=False"],
                 "added": ["actor_rollout_ref.actor.self_distillation.include_environment_feedback=True"]},
    "soft": {"env": {"SOFT": "1"},
             "removed": ["custom_reward_function.path=/work/SDPO/verl/utils/reward_score/feedback/__init__.py"],
             "added": ["custom_reward_function.path=%s/beds/tooluse_soft.py" % KIT]},
    "variation": {"env": {"TEMP": "1.2"}, "removed": [],
                  "added": ["actor_rollout_ref.rollout.temperature=1.2"]},
}


def dry_run(**extra):
    """The launcher's argv, with everything that would touch a disk skipped."""
    done = subprocess.run(["bash", str(LAUNCHER)], capture_output=True, text=True,
                          env={**os.environ, **DRY_ENV, **extra})
    return done.returncode, done.stdout.splitlines(), done.stderr


@pytest.fixture()
def campaign():
    os.environ.setdefault("WORK", "/tmp/unused-k4a")
    return runner.load_campaign(CAMPAIGN)


# --------------------------------------------------------------- 1. the K0 command has not moved
def test_the_default_argv_is_still_k0s_command_exactly():
    code, argv, _ = dry_run()
    assert code == 0
    assert argv == K0_ARGV, "the default command moved; every K0 comparison would be against something else"


def test_a_seed_still_adds_exactly_three_overrides():
    _, plain, _ = dry_run()
    _, seeded, _ = dry_run(SEED="43")
    assert seeded[:len(plain)] == plain
    assert seeded[len(plain):] == ["data.seed=43", "actor_rollout_ref.actor.data_loader_seed=43",
                                   "actor_rollout_ref.actor.fsdp_config.seed=43"]


def test_offload_still_changes_exactly_the_three_offload_keys():
    _, plain, _ = dry_run()
    _, offloaded, _ = dry_run(OFFLOAD="1")
    changed = [(a, b) for a, b in zip(plain, offloaded) if a != b]
    assert len(plain) == len(offloaded)
    assert [b for _, b in changed] == ["actor_rollout_ref.actor.fsdp_config.param_offload=True",
                                       "actor_rollout_ref.actor.fsdp_config.optimizer_offload=True",
                                       "actor_rollout_ref.ref.fsdp_config.param_offload=True"]


# ------------------------------------------------------------------- 2. one arm, one change, once
@pytest.mark.parametrize("arm", ARMS)
def test_each_arm_changes_exactly_its_declared_keys(arm):
    _, plain, _ = dry_run()
    code, armed, _ = dry_run(**DECLARED[arm]["env"])
    assert code == 0
    assert sorted(set(plain) - set(armed)) == sorted(DECLARED[arm]["removed"])
    assert sorted(set(armed) - set(plain)) == sorted(DECLARED[arm]["added"])


@pytest.mark.parametrize("arm", ARMS)
def test_an_arm_keeps_the_order_of_every_key_it_did_not_touch(arm):
    _, plain, _ = dry_run()
    _, armed, _ = dry_run(**DECLARED[arm]["env"])
    untouched = [line for line in plain if line not in DECLARED[arm]["removed"]]
    assert [line for line in armed if line not in DECLARED[arm]["added"]] == untouched


@pytest.mark.parametrize("first,second", [("feedback", "soft"), ("feedback", "variation"),
                                          ("soft", "variation")])
def test_two_arms_in_one_run_are_refused(first, second):
    code, argv, stderr = dry_run(**DECLARED[first]["env"], **DECLARED[second]["env"])
    assert code == 2, "a run that changed two things answers neither question"
    assert argv == [], "the refusal must come before the command is printed"
    assert "one run is one arm" in stderr


def test_all_three_arms_at_once_are_refused():
    env = {k: v for arm in ARMS for k, v in DECLARED[arm]["env"].items()}
    code, _, stderr = dry_run(**env)
    assert code == 2 and "3 arms" in stderr


@pytest.mark.parametrize("bad", [{"FEEDBACK": "yes"}, {"SOFT": "2"}, {"TEMP": "hot"}, {"TEMP": "-1"}])
def test_a_knob_that_is_not_a_value_it_understands_is_refused(bad):
    code, argv, _ = dry_run(**bad)
    assert code == 2 and argv == []


def test_the_soft_arm_points_at_a_reward_file_that_exists():
    _, argv, _ = dry_run(SOFT="1")
    path = next(line.split("=", 1)[1] for line in argv if line.startswith("custom_reward_function.path="))
    assert Path(path).is_file()


# -------------------------------------------------------------------------- 3. the soft reward
@pytest.fixture(scope="module")
def soft():
    if not REFERENCE_TOOLUSE.is_file():
        pytest.skip("references/SDPO is not checked out")
    os.environ["SDPO_TOOLUSE"] = str(REFERENCE_TOOLUSE)
    return load("kit_tooluse_soft", KIT / "beds" / "tooluse_soft.py")


def call(actions_and_inputs) -> str:
    return "\n".join("Action: %s\nAction Input: %s" % (a, json.dumps(i)) for a, i in actions_and_inputs)


GROUND_TRUTH = json.dumps([
    {"Action": "get_weather", "Action_Input": json.dumps({"city": "Zurich", "unit": "C"})},
    {"Action": "send_email", "Action_Input": json.dumps({"to": "ada@example.com"})},
])
RIGHT = [("get_weather", {"city": "Zurich", "unit": "C"}), ("send_email", {"to": "ada@example.com"})]


def test_the_soft_reward_loads_the_authors_checker_and_never_a_copy_of_its_own(soft):
    assert Path(soft.AUTHORS.__file__).resolve() == REFERENCE_TOOLUSE.resolve()
    with pytest.raises(soft.AuthorsCheckerMissing):
        old = dict(os.environ)
        try:
            os.environ.pop("SDPO_TOOLUSE", None)
            os.environ.pop("SDPO_DIR", None)
            if "verl" in sys.modules:
                pytest.skip("verl is importable here, so the fallback cannot be exercised")
            soft.load_authors_checker()
        finally:
            os.environ.clear()
            os.environ.update(old)


@pytest.mark.parametrize("solution", [
    call(RIGHT),
    call([("get_weather", {"city": "Bern", "unit": "C"}), ("send_email", {"to": "ada@example.com"})]),
    call([("get_time", {"city": "Zurich"})]),
    "no tool call here at all",
    "Action: get_weather\nAction Input: {not json",
])
def test_every_key_except_score_is_the_authors_own(soft, solution):
    """The validation metric reads `acc`. If this ever differs, K4a stops measuring K0's score."""
    theirs = soft.AUTHORS.compute_score(solution, GROUND_TRUTH)
    ours = soft.compute_score("tooluse", solution, GROUND_TRUTH)
    assert set(ours) == set(theirs), "the five keys the trainer dumps must not change"
    for key in ("acc", "pred", "incorrect_format", "feedback"):
        assert ours[key] == theirs[key], key


def test_a_correct_answer_scores_exactly_one_under_both(soft):
    ours = soft.compute_score("tooluse", call(RIGHT), GROUND_TRUTH)
    assert ours["acc"] == 1.0 and ours["score"] == 1.0


@pytest.mark.parametrize("solution,expected", [
    # exactly right by the authors' checker
    (call(RIGHT), 1.0),
    # every tool right, two of three argument pairs right -> the share of the arguments
    (call([("get_weather", {"city": "Zurich", "unit": "F"}), ("send_email", {"to": "ada@example.com"})]), 2 / 3),
    # every tool right, one of three
    (call([("get_weather", {"city": "Zurich", "unit": "F"}), ("send_email", {"to": "bob@example.com"})]), 1 / 3),
    # every tool right, none
    (call([("get_weather", {"city": "Bern", "unit": "F"}), ("send_email", {"to": "bob@example.com"})]), 0.0),
    # every tool right, every EXPECTED argument right, plus one the question never asked for: the
    # share is over the longer of the two argument sets, so an invented argument costs a quarter here
    (call([("get_weather", {"city": "Zurich", "unit": "C", "lang": "de"}),
           ("send_email", {"to": "ada@example.com"})]), 0.75),
    # HALF THE TOOLS is not a near-miss, however perfect its arguments
    (call([("get_weather", {"city": "Zurich", "unit": "C"})]), 0.0),
    # a spurious extra call is not a near-miss either
    (call(RIGHT + [("delete_all", {})]), 0.0),
    # one tool called twice when it should be called once
    (call([("get_weather", {"city": "Zurich", "unit": "C"}), ("get_weather", {"city": "Zurich", "unit": "C"})]), 0.0),
    # no tool right, arguments irrelevant
    (call([("delete_all", {"city": "Zurich", "unit": "C", "to": "ada@example.com"})]), 0.0),
    # no Action/Action Input pair at all
    ("I looked it up: it is 19 degrees in Zurich.", 0.0),
])
def test_the_partial_credit_rule_is_the_one_written_at_the_top_of_the_file(soft, solution, expected):
    assert soft.compute_score("tooluse", solution, GROUND_TRUTH)["score"] == pytest.approx(expected, abs=1e-6)


def test_calling_only_some_of_the_right_tools_is_never_shown_to_the_teacher(soft):
    """The rule's whole point: a different answer is not a near-miss, however good its arguments."""
    partial = [call([("get_weather", {"city": "Zurich", "unit": "C"})]),
               call([("send_email", {"to": "ada@example.com"})]),
               call(RIGHT + [("send_email", {"to": "ada@example.com"})])]
    for solution in partial:
        result = soft.compute_score("tooluse", solution, GROUND_TRUTH)
        assert result["score"] == 0.0, solution
        assert result["score"] < soft.SUCCESS_REWARD_THRESHOLD


def test_only_a_strictly_correct_answer_can_reach_the_eligibility_ceiling(soft):
    """An answer the authors call wrong must never score 1.0: 1.0 is reserved for their verdict.

    The last case is the one that matters most: every expected argument is right, so a share taken
    over the EXPECTED arguments alone would read 1.0 and make an answer with an invented argument
    indistinguishable from a correct one.
    """
    wrong = [call([("get_weather", {"city": "Bern", "unit": "C"}), ("send_email", {"to": "ada@example.com"})]),
             call([("get_weather", {"city": "Zurich", "unit": "C"})]),
             call([("get_weather", {"city": "Zurich", "unit": "C"}), ("send_email", {"to": "ada@example.com"}),
                   ("send_email", {"to": "ada@example.com"})]),
             call([("get_weather", {"city": "Zurich", "unit": "C", "lang": "de"}),
                   ("send_email", {"to": "ada@example.com"})])]
    for solution in wrong:
        result = soft.compute_score("tooluse", solution, GROUND_TRUTH)
        assert result["acc"] == 0.0, solution
        assert result["score"] < 1.0, solution


def test_the_soft_score_is_never_below_the_authors_score(soft):
    for solution in (call(RIGHT), call([("get_weather", {"city": "Bern"})]), "nothing"):
        result = soft.compute_score("tooluse", solution, GROUND_TRUTH)
        assert result["score"] >= result["acc"]


def test_the_eligibility_line_is_exactly_the_right_tools_and_half_the_arguments(soft):
    """SDPO shows an attempt to the teacher at score >= 0.5, so the rule must sit either side of it."""
    assert soft.SUCCESS_REWARD_THRESHOLD == 0.5
    # right tools, two of three arguments -> 0.67, shown
    shown = soft.compute_score("tooluse", call([("get_weather", {"city": "Zurich", "unit": "C"}),
                                                ("send_email", {"to": "bob@example.com"})]), GROUND_TRUTH)
    assert shown["score"] >= soft.SUCCESS_REWARD_THRESHOLD and shown["acc"] == 0.0
    # right tools, one of three arguments -> 0.33, ignored
    ignored = soft.compute_score("tooluse", call([("get_weather", {"city": "Zurich", "unit": "F"}),
                                                  ("send_email", {"to": "bob@example.com"})]), GROUND_TRUTH)
    assert ignored["score"] < soft.SUCCESS_REWARD_THRESHOLD
    # exactly half is on the line, and the line includes it
    half = json.dumps([{"Action": "f", "Action_Input": json.dumps({"a": 1, "b": 2})}])
    on_the_line = soft.compute_score("tooluse", call([("f", {"a": 1, "b": 99})]), half)
    assert on_the_line["score"] == pytest.approx(0.5)
    assert on_the_line["score"] >= soft.SUCCESS_REWARD_THRESHOLD


def test_a_row_from_another_bed_raises_rather_than_scoring_zero(soft):
    with pytest.raises(ValueError):
        soft.compute_score("gsm8k", "Action: add\nAction Input: {}", GROUND_TRUTH)


def test_unparseable_ground_truth_does_not_invent_credit(soft):
    result = soft.compute_score("tooluse", call(RIGHT), "{not a list")
    assert result["score"] == 0.0 and result["acc"] == 0.0


# ------------------------------------------------------------------------------ 4. the bars
def test_every_bar_has_a_numeric_limit_and_a_known_aggregate(campaign):
    for row in campaign["rows"]:
        for bar in row.get("bars") or []:
            assert "min" in bar or "max" in bar
            for limit in ("min", "max"):
                if limit in bar:
                    assert isinstance(bar[limit], (int, float)) and not isinstance(bar[limit], bool)
            assert bar.get("agg", "last") in runner.AGGREGATES


def test_every_bar_reads_a_file_one_of_this_kits_tools_writes(campaign):
    known = ("run-summary.json", "metrics.jsonl", "forgetting.json", "agreement.json", "k4a-report.json")
    for row in campaign["rows"]:
        for bar in row.get("bars") or []:
            assert bar["source"].endswith(known), (row["id"], bar["source"])


def _metrics(tmp_path: Path, steps, **overrides) -> Path:
    """A metrics.jsonl shaped like the trainer's file logger writes it."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "metrics.jsonl"
    lines = [json.dumps({"step": 0, "data": {"val-core/tooluse/acc/mean@16": 0.5744}})]
    for step in steps:
        data = {"critic/score/mean": 0.32, "response_length/mean": 96.0,
                "self_distillation/success_group_fraction": 0.37,
                "self_distillation/empty_target_batch": 0.63, "actor/entropy": 0.21,
                "actor/grad_norm": 0.4, "perf/time_per_step": 55.0,
                "perf/max_memory_allocated_gb": 77.2}
        data.update(overrides)
        lines.append(json.dumps({"step": step, "data": data}))
    path.write_text("\n".join(lines) + "\n")
    return path


def _bar_of(campaign, row_id, name):
    row = next(r for r in campaign["rows"] if r["id"] == row_id)
    return next(b for b in row["bars"] if b["name"] == name)


def test_the_feedback_pilots_mechanism_bar_reads_numbers_and_judges_them(campaign, tmp_path):
    """It must PASS on what that arm should produce and FAIL on K0's own values."""
    bar = dict(_bar_of(campaign, "pilot-feedback", "targets-now-exist"))
    bar["source"] = str(_metrics(tmp_path / "k0", [1, 2]))
    judged = runner.judge_bar(bar)
    assert judged["value"] is not None and not judged["ok"], "it passes on K0's own numbers"
    bar["source"] = str(_metrics(tmp_path / "armed", [1, 2],
                                 **{"self_distillation/empty_target_batch": 0.04}))
    assert runner.judge_bar(bar)["ok"]


def test_no_pilot_gates_on_a_number_that_is_noise_over_two_steps(campaign):
    """One pilot gates all three arms, so a bar that can fail by luck costs a round trip for nothing.

    K0's own per-step success share runs 0.125 to 0.625 and its entropy 0.062 to 0.402, so a two-step
    mean of either is a coin flip. They belong in the report, which kit/k4a_report.py prints.
    """
    noisy = ("self_distillation/success_group_fraction", "actor/entropy", "critic/score/mean")
    for row in campaign["rows"]:
        for bar in row.get("bars") or []:
            assert bar["key"] not in noisy, "%s gates on %s, which two steps cannot measure" % (
                row["id"], bar["key"])


def test_the_only_mechanism_bar_is_the_feedback_arms(campaign):
    gated = {row["id"]: sorted(bar["name"] for bar in row["bars"])
             for row in campaign["rows"] if row["id"].startswith("pilot-")}
    common = ["merged-model-present", "no-length-collapse", "trainer-exited-clean", "untrained-validation"]
    assert gated["pilot-feedback"] == sorted(common + ["targets-now-exist"])
    assert gated["pilot-soft"] == common
    assert gated["pilot-variation"] == common


def test_the_calibration_bar_reads_the_untrained_step_only(campaign, tmp_path):
    bar = dict(_bar_of(campaign, "pilot-feedback", "untrained-validation"))
    path = tmp_path / "cal" / "metrics.jsonl"
    path.parent.mkdir()
    path.write_text("\n".join([
        json.dumps({"step": 0, "data": {"val-core/tooluse/acc/mean@16": 0.5744}}),
        json.dumps({"step": 2, "data": {"val-core/tooluse/acc/mean@16": 0.9}}),
    ]) + "\n")
    bar["source"] = str(path)
    assert runner.judge_bar(bar)["ok"] and runner.judge_bar(bar)["value"] == pytest.approx(0.5744)


def test_a_run_that_did_not_merge_fails_its_bar(campaign, tmp_path):
    for merged, ok in ((1, True), (0, False)):
        bar = dict(_bar_of(campaign, "pilot-soft", "merged-model-present"))
        path = tmp_path / ("summary-%d" % merged)
        path.mkdir()
        (path / "run-summary.json").write_text(json.dumps(
            {"schema": "kit-sdpo-run.v1", "returncode": 0, "merged": merged}))
        bar["source"] = str(path / "run-summary.json")
        assert runner.judge_bar(bar)["ok"] is ok


def test_the_launcher_writes_the_keys_those_bars_read():
    text = LAUNCHER.read_text()
    for key in ('"returncode":', '"merged":', '"arm":', '"steps":'):
        assert key in text, key
    assert 'exit "$STATUS"' in text, "a failed trainer must still leave a summary and a non-zero exit"


# ------------------------------------------------------------------ 5. the campaign, as generated
@pytest.mark.skipif(generator is None, reason="scripts/ is not part of the exported kit")
def test_the_committed_campaign_is_what_its_generator_builds():
    assert CAMPAIGN.read_text() == generator.build(), "re-run scripts/make_k4a_campaign.py"


def test_the_three_pilots_come_first_and_gate_every_later_row(campaign):
    ids = [row["id"] for row in campaign["rows"]]
    pilots = [row["id"] for row in campaign["rows"] if row.get("pilot")]
    assert pilots[:3] == ["pilot-%s" % arm for arm in sorted(ARMS)]
    assert ids[:3] == pilots[:3]
    for row in campaign["rows"][3:]:
        assert set(pilots[:3]).issubset(set(row["needs"])), row["id"]


def test_every_arm_and_seed_is_trained_scored_and_reported(campaign):
    ids = {row["id"] for row in campaign["rows"]}
    for arm in ARMS:
        for seed in SEEDS:
            assert "%s-seed%d" % (arm, seed) in ids
            assert "%s-seed%d-forget" % (arm, seed) in ids
    report = next(row for row in campaign["rows"] if row["id"] == "report")
    for arm in ARMS:
        for seed in SEEDS:
            assert "%s-seed%d" % (arm, seed) in report["needs"]
            assert "%s-seed%d-forget" % (arm, seed) in report["needs"]


def test_every_training_row_carries_its_arms_knob_and_no_other(campaign):
    trainers = {"pilot-%s" % arm: arm for arm in ARMS}
    trainers.update({"%s-seed%d" % (arm, seed): arm for arm in ARMS for seed in SEEDS})
    for row in campaign["rows"]:
        knobs = {k: v for k, v in row["env"].items() if k in ("FEEDBACK", "SOFT", "TEMP")}
        want = DECLARED[trainers[row["id"]]]["env"] if row["id"] in trainers else {}
        assert knobs == want, row["id"]
    assert len(trainers) == len(ARMS) * (1 + len(SEEDS))


def test_the_grid_runs_k0s_dose_and_the_pilots_run_two_steps(campaign):
    for row in campaign["rows"]:
        if row["id"].startswith("pilot-"):
            assert row["env"]["STEPS"] == "2"
        elif any(row["id"] == "%s-seed%d" % (arm, seed) for arm in ARMS for seed in SEEDS):
            assert row["env"]["STEPS"] == "40" and row["env"]["TEST_FREQ"] == "5"
            assert row["env"]["SEED"] == row["id"].split("seed")[-1]


def test_every_forgetting_row_scores_on_gpu_zero(campaign):
    for row in campaign["rows"]:
        if "forget" in row["id"] or row["id"].startswith("base-"):
            assert row["env"].get("CUDA_VISIBLE_DEVICES") == "0"


def test_all_the_io_happens_before_a_gpu_is_held(campaign):
    prepared = [row for row in campaign["rows"] if row.get("prepare")]
    assert [row["id"] for row in prepared] == ["pilot-feedback"], "preparation belongs to the first row"
    steps = " ".join(" ".join(step) for step in prepared[0]["prepare"])
    assert "preprocess.py" in steps and "MODEL_DIR" in steps and "tooluse_soft.py" in steps
    assert "K0_REPORT" in steps, "the control must be on disk before 8 hours of GPU are spent"


def test_planning_the_campaign_creates_nothing(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    done = subprocess.run([sys.executable, str(KIT / "runner.py"), "plan", str(CAMPAIGN)],
                          capture_output=True, text=True, env={**os.environ, "WORK": str(work)})
    assert done.returncode == 0, done.stderr
    assert list(work.iterdir()) == []
    assert "PILOT" in done.stdout


# --------------------------------------------------------------------------- 6. the instructions
def test_every_command_the_readme_gives_names_this_campaign():
    text = README.read_text()
    for action in ("plan", "prepare", "run", "status"):
        assert "runner.py %s $KIT/campaigns/k4a-stuck-problems.yaml" % action in text, action
    for variable in ("KIT", "WORK", "SDPO_DIR", "MODEL_DIR", "NGPU", "K0_REPORT"):
        assert "export" in text and variable in text, variable


def test_the_readme_describes_the_arms_the_launcher_actually_has():
    text = README.read_text()
    for arm in ARMS:
        assert arm in text
    for env in ("FEEDBACK=1", "SOFT=1", "TEMP=1.2"):
        assert env in text, env
    assert "0.555" in text and "0.600" in text, "the stop rule must name the calibration band"


def test_the_readme_names_a_row_the_campaign_has(campaign):
    ids = {row["id"] for row in campaign["rows"]}
    assert "--row base-1" in README.read_text() and "base-1" in ids


# ----------------------------------------------------------------------------- 7. the report
def qid(prompt: str) -> str:
    """How both reports name a held-out question: the sha1 of its prompt (kit/make_report.py:68)."""
    return hashlib.sha1(prompt.encode()).hexdigest()[:12]


def _run_dir(root: Path, name: str, *, steps, arm, acc_of=lambda q, step: 1.0) -> Path:
    run = root / name
    (run / "validation").mkdir(parents=True)
    (run / "rollouts").mkdir()
    (run / ("hf-step%d" % steps)).mkdir()
    (run / ("hf-step%d" % steps) / "config.json").write_text("{}")
    (run / "run-summary.json").write_text(json.dumps(
        {"schema": "kit-sdpo-run.v1", "name": name, "arm": arm, "steps": steps,
         "returncode": 0, "merged": 1}))
    questions = ["q%d" % i for i in range(4)]
    metrics = []
    for step in (0, steps):
        # `score` deliberately differs from `acc` wherever the answer was wrong: that is what the
        # soft arm does to a validation dump, and a report that read `score` must not go unnoticed.
        rows = [{"input": q, "output": "o", "acc": acc_of(q, step),
                 "score": acc_of(q, step) if acc_of(q, step) >= 1.0 else 0.7}
                for q in questions for _ in range(2)]
        (run / "validation" / ("%d.jsonl" % step)).write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n")
        accuracy = sum(acc_of(q, step) for q in questions) / len(questions)
        metrics.append({"step": step, "data": {"val-core/tooluse/acc/mean@16": accuracy}})
    for step in range(1, steps + 1):
        metrics.append({"step": step, "data": {
            "critic/score/mean": 0.4, "response_length/mean": 96.0,
            "self_distillation/success_group_fraction": 0.8,
            "self_distillation/empty_target_batch": 0.2, "actor/entropy": 0.21,
            "actor/grad_norm": 0.4, "perf/time_per_step": 55.0,
            "perf/max_memory_allocated_gb": 77.2}})
        (run / "rollouts" / ("%d.jsonl" % step)).write_text("\n".join(
            json.dumps({"input": q, "output": "o", "acc": 1.0 if q == "q0" else 0.0, "score": 0.7})
            for q in questions for _ in range(2)) + "\n")
    (run / "metrics.jsonl").write_text("\n".join(json.dumps(m) for m in metrics) + "\n")
    return run


def _panels(root: Path, name: str, correct=(90, 80, 82), machine="m1") -> None:
    directory = root / name
    directory.mkdir(parents=True)
    panels = {n: {"correct": c, "n": 100, "per_member": {}, "median_output_chars": 40}
              for n, c in zip(("maths", "knowledge", "instructions"), correct)}
    (directory / "forgetting.json").write_text(json.dumps(
        {"panels": panels, "total_correct": sum(correct), "machine": {"id": machine}}))


def _control(path: Path, seeds=SEEDS) -> Path:
    runs = []
    for seed in seeds:
        runs.append({
            "name": "dose40-seed%d" % seed,
            "steps_completed": 40,
            "validations": [{"step": 0, "accuracy_logged": 0.5744}, {"step": 40, "accuracy_logged": 0.62}],
            # deliberately NOT flat: a two-step pilot must be held against the control's own first
            # two steps, and a mean over all 40 would be a different number.
            "training": [{"step": s,
                          "empty_target_batch": 0.63 if s <= 2 else 0.50,
                          "success_group_fraction": 0.37 if s <= 2 else 0.55,
                          "response_tokens": 96.0 if s <= 2 else 140.0,
                          "entropy": 0.21 if s <= 2 else 0.31,
                          "score": 0.32 if s <= 2 else 0.45} for s in range(1, 41)],
            # q0 and q1 were solved by the control; q2 and q3 are the stuck ones. Both reports key a
            # question by the sha1 of its prompt, which is how the two sides line up at all.
            "per_question": {"40": {qid(q): a for q, a in
                                    (("q0", 1.0), ("q1", 0.5), ("q2", 0.0), ("q3", 0.0))}},
        })
    path.write_text(json.dumps({"schema": "kit-sdpo-report.v1", "runs": runs}))
    return path


@pytest.fixture()
def tree(tmp_path):
    runs, forgetting = tmp_path / "runs", tmp_path / "forgetting"
    runs.mkdir()
    _panels(forgetting, "base-a1")
    for arm in ARMS:
        for seed in SEEDS:
            _run_dir(runs, "%s-seed%d-a1" % (arm, seed), steps=40, arm=arm,
                     acc_of=lambda q, step: 1.0 if (q in ("q0", "q1") or step == 0) else 0.5)
            _panels(forgetting, "%s-seed%d-forget-a1" % (arm, seed))
    _run_dir(runs, "pilot-feedback-a1", steps=2, arm="FEEDBACK=1")
    return {"runs": runs, "forgetting": forgetting, "k0": _control(tmp_path / "k0.json")}


def test_the_report_puts_every_arm_beside_the_control_with_the_same_seed(tree):
    report = k4a_report.build(tree["runs"], tree["forgetting"], tree["k0"])
    assert report["arms_reported"] == len(ARMS) and report["control_runs"] == len(SEEDS)
    assert report["comparable"] == 1
    for arm in ARMS:
        for seed in SEEDS:
            row = report["arms"][arm]["seeds"][seed]
            assert row["control"]["name"] == "dose40-seed%d" % seed
            assert row["change_vs_control"] == pytest.approx(row["final"] - 0.62, abs=1e-6)


def test_the_report_reads_the_strict_score_and_flags_a_metric_that_is_not_it(tree, tmp_path):
    report = k4a_report.build(tree["runs"], tree["forgetting"], tree["k0"])
    row = report["arms"]["soft"]["seeds"][42]
    assert row["final"] == pytest.approx(0.75)                 # q0,q1 at 1.0; q2,q3 at 0.5
    assert row["flags"] == []
    # now make the trainer's logged number disagree with the per-question `acc`
    metrics = tree["runs"] / "soft-seed42-a1" / "metrics.jsonl"
    lines = [json.loads(line) for line in metrics.read_text().splitlines() if line.strip()]
    for record in lines:
        if "val-core/tooluse/acc/mean@16" in record["data"]:
            record["data"]["val-core/tooluse/acc/mean@16"] = 0.99
    metrics.write_text("\n".join(json.dumps(r) for r in lines) + "\n")
    flagged = k4a_report.build(tree["runs"], tree["forgetting"], tree["k0"])
    assert any("MISMATCH" in flag for flag in flagged["arms"]["soft"]["seeds"][42]["flags"])


def test_the_report_recomputes_a_strict_success_share_that_no_reward_can_move(tree):
    report = k4a_report.build(tree["runs"], tree["forgetting"], tree["k0"])
    row = report["arms"]["soft"]["seeds"][42]
    assert row["strict_success"]["success_group_fraction"] == pytest.approx(0.25)   # only q0 is right
    assert row["means"]["success_group_fraction"] == pytest.approx(0.8)             # what the trainer counted
    assert row["strict_success"]["empty_target_fraction"] == pytest.approx(0.75)


def test_the_report_splits_the_questions_the_control_could_not_do(tree):
    report = k4a_report.build(tree["runs"], tree["forgetting"], tree["k0"])
    block = report["arms"]["feedback"]["seeds"][43]["by_k0_difficulty"]
    assert block["stuck_questions"] == 2 and block["already_solved_questions"] == 2
    assert block["stuck_accuracy_now"] == pytest.approx(0.5)
    assert block["already_solved_accuracy_now"] == pytest.approx(1.0)
    assert block["control_already_solved_accuracy"] == pytest.approx(0.75)


def test_a_split_over_no_shared_questions_is_flagged_rather_than_rendered_empty(tree, tmp_path):
    """If the two reports ever named questions differently, every other table would still render."""
    control = json.loads(tree["k0"].read_text())
    for run in control["runs"]:
        run["per_question"] = {"40": {"a-name-from-somewhere-else": 1.0}}
    tree["k0"].write_text(json.dumps(control))
    report = k4a_report.build(tree["runs"], tree["forgetting"], tree["k0"])
    row = report["arms"]["feedback"]["seeds"][43]
    assert row["by_k0_difficulty"]["shared_questions"] == 0
    assert any("NO SHARED QUESTIONS" in flag for flag in row["flags"])


def test_the_report_refuses_to_mix_two_machines(tree):
    _panels(tree["forgetting"], "feedback-seed44-forget-a2", machine="a-different-machine")
    with pytest.raises(k4a_report.K4aReportError, match="fingerprints"):
        k4a_report.build(tree["runs"], tree["forgetting"], tree["k0"])
    mixed = k4a_report.build(tree["runs"], tree["forgetting"], tree["k0"], allow_different_machines=True)
    assert mixed["comparable"] == 0 and mixed["different_machines_allowed"] == 1


def test_the_report_refuses_a_control_with_no_dose40_runs(tree, tmp_path):
    empty = tmp_path / "other.json"
    empty.write_text(json.dumps({"runs": [{"name": "run3-seed43", "validations": []}]}))
    with pytest.raises(k4a_report.K4aReportError, match="dose40"):
        k4a_report.build(tree["runs"], tree["forgetting"], empty)


def test_the_report_takes_the_highest_attempt_of_every_run(tree):
    _run_dir(tree["runs"], "feedback-seed42-a2", steps=40, arm="feedback",
             acc_of=lambda q, step: 0.25)
    report = k4a_report.build(tree["runs"], tree["forgetting"], tree["k0"])
    assert report["arms"]["feedback"]["seeds"][42]["run"] == "feedback-seed42-a2"
    assert report["arms"]["feedback"]["seeds"][42]["final"] == pytest.approx(0.25)


def test_the_report_renders_and_never_overwrites(tree, tmp_path):
    out = tmp_path / "report-a1"
    code = k4a_report.main(["--runs", str(tree["runs"]), "--forgetting", str(tree["forgetting"]),
                            "--k0", str(tree["k0"]), "--out", str(out)])
    assert code == 0
    text = (out / "k4a-report.md").read_text()
    for arm in ARMS:
        assert arm in text
    assert "K0 control" in text and "strict" in text
    json.loads((out / "k4a-report.json").read_text())
    with pytest.raises(SystemExit):
        k4a_report.main(["--runs", str(tree["runs"]), "--forgetting", str(tree["forgetting"]),
                         "--k0", str(tree["k0"]), "--out", str(out)])


def test_the_report_carries_the_pilot_checks_the_campaign_no_longer_gates(tree):
    """What the soft and temperature pilots moved, printed beside K0's own first two steps."""
    checks = k4a_report.build(tree["runs"], tree["forgetting"], tree["k0"])["pilot_checks"]
    assert set(checks["arms"]) == {"feedback"}                    # the only pilot in this fixture
    assert checks["arms"]["feedback"]["success_group_fraction"] == pytest.approx(0.8)
    assert checks["arms"]["feedback"]["entropy"] == pytest.approx(0.21)
    # the reference column is the control's own FIRST TWO steps, not a mean over all 40 of them
    reference = checks["control_first_two_steps"]
    assert reference["success_group_fraction"]["mean"] == pytest.approx(0.37)
    assert reference["entropy"]["mean"] == pytest.approx(0.21)
    assert reference["empty_target_batch"]["mean"] == pytest.approx(0.63)
    control = k4a_report.build(tree["runs"], tree["forgetting"], tree["k0"])["control"][42]
    assert control["means"]["success_group_fraction"] != pytest.approx(0.37), "fixture is flat"
    assert checks["control_runs"] == ["dose40-seed%d" % seed for seed in SEEDS]


def test_the_pilot_check_table_is_labelled_a_check_and_not_a_verdict(tree, tmp_path):
    out = tmp_path / "report-a1"
    k4a_report.main(["--runs", str(tree["runs"]), "--forgetting", str(tree["forgetting"]),
                     "--k0", str(tree["k0"]), "--out", str(out)])
    text = (out / "k4a-report.md").read_text()
    assert "a CHECK, not a gate" in text
    assert "do not treat it as a" in text


def test_a_run_with_no_rollout_dumps_says_so_rather_than_comparing_anyway(tree):
    for path in (tree["runs"] / "variation-seed42-a1" / "rollouts").glob("*.jsonl"):
        path.unlink()
    report = k4a_report.build(tree["runs"], tree["forgetting"], tree["k0"])
    row = report["arms"]["variation"]["seeds"][42]
    assert row["strict_success"] is None
    assert any("NO ROLLOUT DUMPS" in flag for flag in row["flags"])
