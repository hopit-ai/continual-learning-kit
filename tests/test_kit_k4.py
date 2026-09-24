"""Package K4: the hint pipeline, the SDPO bed launcher, the campaign and the report.

Eight ways this package could be quietly wrong, one section each:

1. The SDPO command moved. `teacher-hint` is K0's own command with bed paths; if the argv drifted, the
   arm would be a different method. Both launchers are dry-run and their argvs compared key for key.
2. The hint reached the student in the `teacher-hint` arm, or the teacher in no arm at all. The pinned
   trainer's own feedback path is pinned here, and the reward function is exercised.
3. A hint contained the answer. Every rule of the leakage filter, per bed, on worked cases.
4. The hinted file differs from the control's somewhere other than the hinted questions, or the faded
   arm's two halves are not the same questions the other arms see, or a hint silently shortened the
   dose by pushing a prompt over verl's prompt limit.
5. A bar reads a key nothing writes, or a number two steps cannot measure.
6. The committed campaign is not what its generator builds, or the pilots do not gate the grid, or a
   bed's dose is not its control's dose.
7. The report names a folder the campaign does not write (the K4a lesson, receipt 225), subtracts
   counts from two machines, or turns a missing scoring into a dash that reads like a zero.
8. The README tells the partner a command the campaign does not have.

Every fixture is built here; nothing needs a GPU, a trainer, a served model or the network. The
trainer-path test reads references/SDPO at its pinned commit and skips without it.
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
BED_LAUNCHER = KIT / "run_sdpo_bed.sh"
TOOL_LAUNCHER = KIT / "run_sdpo_toolalpaca.sh"
CAMPAIGN = KIT / "campaigns" / "k4-hints.yaml"
README = KIT / "README-k4.md"
RAY_TRAINER = ROOT / "references" / "SDPO" / "verl" / "trainer" / "ppo" / "ray_trainer.py"
NAIVE_MANAGER = ROOT / "references" / "SDPO" / "verl" / "workers" / "reward_manager" / "naive.py"


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = load("kit_runner_for_k4", KIT / "runner.py")
hints = load("kit_hints", KIT / "hints.py")
k4_report = load("kit_k4_report", KIT / "k4_report.py")

#: The exported kit carries kit/ and these tests, but not scripts/. Everything below works on the
#: export; only the generator-parity test needs the private repository, and it skips without it.
GENERATOR = ROOT / "scripts" / "make_k4_campaign.py"
generator = load("make_k4_campaign", GENERATOR) if GENERATOR.is_file() else None

BEDS = ("spider", "finqa")
ARMS = ("hint", "hint-faded", "teacher-none", "teacher-hint")
#: The two SDPO arms and the switch position each one IS. `teacher-none` is the SDPO route's control:
#: same command, same data, same reward function, same dose, one trainer key apart.
SDPO_ARMS = {"teacher-none": "0", "teacher-hint": "1"}
SEEDS = (0, 1, 2, 3, 4)
DOSE = {"spider": (20, 10), "finqa": (40, 20)}          # (steps, fade), each bed's own control's dose

SPIDER_SOURCE, FINQA_SOURCE = "spider1_execution", "finqa"

#: The only differences allowed between the two SDPO launchers' commands: the data, the reward and the
#: output. `vars.task` is a data key (it names the dataset directory the config's defaults are built
#: from) and `trainer.group_name` is a label on the output.
ALLOWED_DIFFERENCES = {
    "data.train_files", "data.val_files", "custom_reward_function.path", "vars.task",
    "trainer.group_name", "trainer.default_local_dir", "vars.log_dir", "vars.ckpt_dir",
}

COMMON = {"SDPO_DIR": "/work/SDPO", "MODEL_DIR": "/work/models/Qwen3-1.7B", "WORK": "/work/k4-work",
          "NAME": "x", "STEPS": "40", "TEST_FREQ": "40", "DRY_RUN": "1", "NGPU": "4"}
BED_ENV = {"TRAIN_FILE": "/work/data/spider/train.parquet",
           "VAL_FILE": "/work/data/spider/heldout.parquet",
           "HINTS_FILE": "/work/hints.jsonl", "TASK": "datasets/spider_sql"}


def dry(script: Path, **extra):
    done = subprocess.run(["bash", str(script)], capture_output=True, text=True,
                          env={**os.environ, **COMMON, **extra})
    return done.returncode, [line for line in done.stdout.split("\n") if line], done.stderr


def as_keys(argv: list) -> dict:
    return dict(item.split("=", 1) for item in argv if "=" in item)


@pytest.fixture()
def campaign():
    os.environ.setdefault("WORK", "/tmp/unused-k4")
    return runner.load_campaign(CAMPAIGN)


# ------------------------------------------------- 1. the SDPO command is still K0's, with bed paths
@pytest.mark.parametrize("feedback", ["0", "1"])
def test_the_bed_launcher_differs_from_k0s_command_only_in_data_reward_and_output(feedback):
    """AT BOTH SWITCH POSITIONS. `teacher-none` is a control only if its command is the reference's
    own at FEEDBACK=0, exactly as `teacher-hint`'s is at FEEDBACK=1."""
    code, bed, err = dry(BED_LAUNCHER, VAL_BEFORE="1", FEEDBACK=feedback, **BED_ENV)
    assert code == 0, err
    tool_code, tool, tool_err = dry(TOOL_LAUNCHER, FEEDBACK=feedback)
    assert tool_code == 0, tool_err
    assert [item.split("=", 1)[0] for item in bed] == [item.split("=", 1)[0] for item in tool], \
        "the two commands must carry the same keys in the same order"
    mine, theirs = as_keys(bed), as_keys(tool)
    differ = {key for key in mine if mine[key] != theirs[key]}
    assert differ == ALLOWED_DIFFERENCES, \
        "the bed launcher may differ from K0's command only in data, reward and output"


def test_the_one_mechanism_key_is_the_feedback_switch():
    """Against the reference launcher's own default (feedback off) exactly one further key differs,
    and it is the one the hint arm turns on."""
    _, bed, _ = dry(BED_LAUNCHER, VAL_BEFORE="1", FEEDBACK="1", **BED_ENV)
    _, tool, _ = dry(TOOL_LAUNCHER)
    mine, theirs = as_keys(bed), as_keys(tool)
    differ = {key for key in mine if mine[key] != theirs[key]}
    assert differ - ALLOWED_DIFFERENCES == {
        "actor_rollout_ref.actor.self_distillation.include_environment_feedback"}
    assert mine["actor_rollout_ref.actor.self_distillation.include_environment_feedback"] == "True"


def test_the_two_sdpo_arms_commands_differ_in_that_key_and_nothing_else():
    """The whole point of `teacher-none`: one key apart from `teacher-hint`, so a difference in the
    result is the hint and not the trainer, the data, the reward function, the dose or the seed."""
    _, treated, _ = dry(BED_LAUNCHER, FEEDBACK="1", **BED_ENV)
    _, control, _ = dry(BED_LAUNCHER, FEEDBACK="0", **BED_ENV)
    mine, theirs = as_keys(treated), as_keys(control)
    assert [item.split("=", 1)[0] for item in treated] == [item.split("=", 1)[0] for item in control]
    assert {key for key in mine if mine[key] != theirs[key]} == {
        "actor_rollout_ref.actor.self_distillation.include_environment_feedback"}
    assert theirs["actor_rollout_ref.actor.self_distillation.include_environment_feedback"] == "False"


def test_the_switch_is_what_names_the_arm_in_the_runs_own_summary():
    """The summary is what kit/k4_report.py reads: a run that recorded the wrong arm would put the
    control's numbers in the treatment's row."""
    text = BED_LAUNCHER.read_text()
    assert 'ARM=teacher-hint' in text and 'ARM=teacher-none' in text
    assert '"arm": "$ARM"' in text and '"feedback": $FEEDBACK' in text


def test_the_campaigns_settings_turn_the_in_trainer_validation_off_and_the_shuffle_with_it():
    _, argv, _ = dry(BED_LAUNCHER, SHUFFLE="0", TEST_FREQ="-1", **BED_ENV)
    keys = as_keys(argv)
    assert keys["trainer.val_before_train"] == "False"
    assert keys["trainer.test_freq"] == "-1"
    assert keys["data.shuffle"] == "False", "the control runs read their file in order; so must this arm"


def test_a_seed_still_adds_exactly_three_overrides():
    _, plain, _ = dry(BED_LAUNCHER, **BED_ENV)
    _, seeded, _ = dry(BED_LAUNCHER, SEED="3", **BED_ENV)
    assert seeded[:len(plain)] == plain
    assert seeded[len(plain):] == ["data.seed=3", "actor_rollout_ref.actor.data_loader_seed=3",
                                   "actor_rollout_ref.actor.fsdp_config.seed=3"]


@pytest.mark.parametrize("bad", [{"SHUFFLE": "yes"}, {"VAL_BEFORE": "2"}, {"FEEDBACK": "off"},
                                 {"FEEDBACK": "2"}])
def test_a_knob_that_is_not_a_value_it_understands_is_refused(bad):
    code, argv, _ = dry(BED_LAUNCHER, **{**BED_ENV, **bad})
    assert code == 2 and argv == []


@pytest.mark.parametrize("feedback", ["0", "1"])
def test_the_launcher_refuses_to_run_either_sdpo_arm_with_no_hints(tmp_path, feedback):
    """A hint run whose reward function found no hints would be an ordinary SDPO run calling itself an
    arm; a control run without them would be scored differently from the arm it is the control for."""
    for name in ("train.parquet", "heldout.parquet"):
        (tmp_path / name).write_text("not really a parquet, but it is on disk")
    (tmp_path / "model").mkdir()
    (tmp_path / "model" / "config.json").write_text("{}")
    env = {**COMMON, "DRY_RUN": "0", "WORK": str(tmp_path / "work"), "FEEDBACK": feedback,
           "MODEL_DIR": str(tmp_path / "model"), "TRAIN_FILE": str(tmp_path / "train.parquet"),
           "VAL_FILE": str(tmp_path / "heldout.parquet"), "TASK": "datasets/spider_sql"}
    done = subprocess.run(["bash", str(BED_LAUNCHER)], capture_output=True, text=True,
                          env={**os.environ, **env})
    assert done.returncode != 0 and "HINTS_FILE" in done.stderr
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    done = subprocess.run(["bash", str(BED_LAUNCHER)], capture_output=True, text=True,
                          env={**os.environ, **env, "HINTS_FILE": str(empty)})
    assert done.returncode == 2 and "missing or empty" in done.stderr


def test_the_launcher_writes_the_keys_the_campaigns_bars_read():
    text = BED_LAUNCHER.read_text()
    for key in ('"returncode":', '"merged":', '"steps":', '"hints":', '"feedback":', '"arm":'):
        assert key in text, key
    assert 'exit "$STATUS"' in text, "a failed trainer must still leave a summary and a non-zero exit"


# --------------------------------------------- 2. the hint reaches the teacher, and only the teacher
@pytest.mark.skipif(not RAY_TRAINER.is_file(), reason="references/SDPO is not checked out")
def test_the_pinned_trainer_still_takes_the_teachers_feedback_from_the_reward_functions_dict():
    """If this moves, `teacher-hint` is not the arm it is named for. docs/phase2/k4/feasibility.md (a)."""
    trainer = RAY_TRAINER.read_text()
    assert 'raw_feedback = reward_extra_infos_dict.get("feedback", [])' in trainer
    assert "if raw_feedback[i] and isinstance(raw_feedback[i], str) and raw_feedback[i].strip():" in trainer
    assert "feedback_section = self_distillation_cfg.feedback_template.format(" in trainer
    # ... and it goes into the TEACHER's tensors, not the student's.
    assert "teacher_input_ids = torch.cat([teacher_prompt[\"input_ids\"].to(device), responses], dim=1)" \
        in trainer
    manager = NAIVE_MANAGER.read_text()
    assert "for key, value in score.items():" in manager and "reward_extra_info[key].append(value)" in manager
    assert 'extra_info = data_item.non_tensor_batch.get("extra_info", {})' in manager


def row(source: str, index: str, prompt: str, ground_truth: str, **extra) -> dict:
    return {"data_source": source, "prompt": [{"role": "user", "content": prompt}],
            "ability": source, "reward_model": {"style": "rule", "ground_truth": ground_truth},
            "extra_info": {"split": "train", "index": index, "problem": prompt, **extra}}


GOLD_SQL = "SELECT name FROM singer WHERE age > 30"
SPIDER_TRUTH = json.dumps({"db": "concert_singer", "gold_sql": GOLD_SQL}, sort_keys=True)
PLAN = "Find the singer table.\nFilter the rows by age.\nSelect the name column."


def spider_rows(n: int = 6) -> list:
    return [row(SPIDER_SOURCE, "spider-train_spider-concert_singer-%d" % i,
                "Schema:\nCREATE TABLE singer (name text, age int)\n\nQuestion: who is old?",
                SPIDER_TRUTH) for i in range(n)]


def finqa_rows(n: int = 6, gold="0.1446") -> list:
    return [row(FINQA_SOURCE, "ADI/2009/page_%d.pdf-1" % i,
                "Report excerpt: revenue rose.\n\nQuestion: by what percent?", gold,
                gold_consistent=True) for i in range(n)]


@pytest.fixture()
def served(tmp_path, monkeypatch):
    """kit/hints.py as the reward file the trainer would load, with two questions hinted."""
    path = tmp_path / "hints-filtered.jsonl"
    path.write_text("".join(json.dumps({"index": index, "hint": PLAN}) + "\n" for index in
                            ("ADI/2009/page_0.pdf-1", "ADI/2009/page_1.pdf-1")))
    monkeypatch.setenv(hints.HINTS_FILE_ENV, str(path))
    hints._HINTS = None
    yield path
    hints._HINTS = None


def test_the_reward_function_returns_the_hint_for_a_stuck_question_answered_wrongly(served):
    result = hints.feedback(FINQA_SOURCE, "Answer: 99", "0.1446",
                            {"index": "ADI/2009/page_0.pdf-1"})
    assert result["feedback"] == PLAN
    assert result["score"] == 0.0 and result["acc"] == 0.0


def test_the_reward_function_is_empty_everywhere_else(served):
    assert hints.feedback(FINQA_SOURCE, "Answer: 99", "0.1446", {"index": "no-hint-here"})["feedback"] == ""
    # and on an attempt the bed calls right, because feedback on a correct answer is not this arm
    right = hints.feedback(FINQA_SOURCE, "Answer: 0.1446", "0.1446", {"index": "ADI/2009/page_0.pdf-1"})
    assert right["acc"] == 1.0 and right["feedback"] == ""


def test_the_reward_function_always_returns_a_feedback_key(served):
    """_collect_feedback indexes the feedback list positionally: a missing key would shift every hint
    onto the wrong question (docs/phase2/k4/feasibility.md (a))."""
    for solution in ("Answer: 99", "no answer at all", "", "Answer: yes"):
        for index in ("ADI/2009/page_0.pdf-1", "somewhere-else"):
            result = hints.feedback(FINQA_SOURCE, solution, "0.1446", {"index": index})
            assert "feedback" in result and isinstance(result["feedback"], str)
            assert {"score", "acc", "pred", "incorrect_format", "feedback"} <= set(result)


def test_the_reward_function_replaces_the_beds_own_complaint_rather_than_adding_to_it(served):
    """One declared change: the teacher sees the hint. Passing the bed's message too is a second one."""
    bed = load("kit_bed_finqa_for_k4", KIT / "beds" / "finqa.py")
    theirs = bed.compute_score(FINQA_SOURCE, "Answer: 99", "0.1446", None)["feedback"]
    ours = hints.feedback(FINQA_SOURCE, "Answer: 99", "0.1446", {"index": "ADI/2009/page_0.pdf-1"})
    assert theirs and theirs not in ours["feedback"]
    assert ours["feedback"] == PLAN


def test_compute_score_is_the_same_function_so_the_launchers_default_name_works():
    assert hints.compute_score is hints.feedback


def test_an_unknown_data_source_raises_rather_than_scoring_zero(served):
    with pytest.raises(hints.HintsError):
        hints.feedback("some-other-bed", "Answer: 1", "1", {"index": "x"})


def test_a_run_with_no_hints_file_raises_rather_than_serving_nothing(monkeypatch):
    monkeypatch.delenv(hints.HINTS_FILE_ENV, raising=False)
    hints._HINTS = None
    with pytest.raises(hints.HintsError, match="HINTS_FILE"):
        hints.feedback(FINQA_SOURCE, "Answer: 1", "1", {"index": "x"})
    hints._HINTS = None


def test_the_reward_files_self_check_proves_it_serves_and_stays_quiet(tmp_path, monkeypatch):
    rows = finqa_rows(4)
    rows_path = tmp_path / "train.jsonl"
    rows_path.write_text(hints.as_jsonl(rows))
    hint_path = tmp_path / "hints-filtered.jsonl"
    hint_path.write_text(hints.as_jsonl([{"index": rows[0]["extra_info"]["index"], "hint": PLAN}]))
    hints._HINTS = None
    code = hints.main(["feedback", "--hints", str(hint_path), "--rows", str(rows_path),
                       "--out", str(tmp_path / "serve")])
    hints._HINTS = None
    assert code == 0
    check = json.loads((tmp_path / "serve" / "feedback-check.json").read_text())
    assert check["served"] == 1 and check["empty_elsewhere"] == 1
    assert sorted(check["keys"]) == ["acc", "feedback", "incorrect_format", "pred", "score"]


# ------------------------------------------------------------------------- 3. the leakage filter
@pytest.mark.parametrize("hint,reason", [
    (PLAN, None),
    ("", "empty"),
    ("Only one line here.\nAnd a second.", "too_few_lines"),
    ("a\nb\nc\nd\ne\nf", "too_many_lines"),
    ("Find the table.\nThen run %s to get it.\nRead the name column." % GOLD_SQL, "contains_gold_query"),
    ("Find the table.\nWrite SELECT name FROM singer to list them.\nFilter by age.", "contains_sql_statement"),
    ("Find the table.\nFilter by age.\nAnswer: the two oldest singers.", "writes_answer_line"),
])
def test_every_spider_rule_drops_what_it_is_for(hint, reason):
    assert hints.drop_reason(hint, spider_rows(1)[0]) == reason


# ---- reasoning models think inside the reply (smoke attempt 2: Qwen3-8B, 147 of 171 hints too long) ----

def test_a_thinking_block_is_stripped_before_the_plan_is_counted_or_filtered():
    reply = "<think>\nThe gold is SELECT name FROM singer WHERE age > 30.\n</think>\n" + PLAN
    assert hints.tidy(reply) == PLAN
    assert hints.drop_reason(hints.tidy(reply), spider_rows(1)[0]) is None
    thinking, plan = hints.split_thinking(reply)
    assert thinking.startswith("<think>") and thinking.endswith("</think>") and plan.strip() == PLAN


def test_an_unclosed_thinking_block_yields_no_plan_rather_than_the_thinking():
    reply = "<think>\nStill thinking when max_tokens ran out.\nFind the table.\nFilter.\nSort."
    assert hints.tidy(reply) == ""
    assert hints.drop_reason(hints.tidy(reply), spider_rows(1)[0]) == "empty"


def test_a_reply_without_thinking_is_unchanged_by_the_strip():
    assert hints.tidy(PLAN + "\n") == PLAN


def test_a_plan_written_as_one_paragraph_is_reshaped_to_one_sentence_a_line():
    paragraph = "Look at the `city` table. Filter rows where `state_name` is 'California'. Sum the `population` values."
    assert hints.tidy(paragraph).split("\n") == ["Look at the `city` table.", "Filter rows where `state_name` is 'California'.",
                                                  "Sum the `population` values."]
    assert hints.was_reshaped(paragraph) and hints.drop_reason(hints.tidy(paragraph), spider_rows(1)[0]) is None


def test_a_plan_already_written_as_lines_is_not_reshaped_even_with_two_sentences_on_a_line():
    plan = "Find the table. Then the column.\nFilter by age.\nSort by name."
    assert hints.tidy(plan) == plan and not hints.was_reshaped(plan)


def test_a_decimal_inside_a_sentence_does_not_split_it():
    assert hints.tidy("Divide by 3.5 to scale. Then round. Report it.").split("\n")[0] == "Divide by 3.5 to scale."


@pytest.mark.parametrize("text,is_sql", [
    ("Write SELECT name FROM singer to list them.", True),
    ("select name from artist where age > 30", True),
    ("Run `select count(*) from singer` on the table.", True),
    ("Select the publication titles from the publication table using the matched ids.", False),
    ("Select publication titles from the publication table.", False),
    ("Select rows with age over 30 from each of the two tables and count them.", False),
    ("Filter by age, then sort by name.", False),
])
def test_the_sql_rule_drops_statements_and_keeps_english_sentences_that_start_with_select(text, is_sql):
    assert hints.looks_like_sql_statement(text) is is_sql
    plan = "Find the singer table.\n%s\nReport the names." % text
    assert (hints.drop_reason(plan, spider_rows(1)[0]) == "contains_sql_statement") is is_sql



def test_the_request_asks_the_server_not_to_think_and_falls_back_once_on_a_400(monkeypatch):
    import io
    import urllib.error
    import urllib.request
    seen = []

    def fake_urlopen(request, timeout):
        body = json.loads(request.data.decode("utf-8"))
        seen.append(body)
        if "chat_template_kwargs" in body:
            raise urllib.error.HTTPError(request.full_url, 400, "unknown field", {}, io.BytesIO(b""))
        payload = {"choices": [{"message": {"content": PLAN}, "finish_reason": "stop"}], "usage": {}}
        return io.BytesIO(json.dumps(payload).encode("utf-8"))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    answer = hints.post_chat("http://x/v1", "m", "q", api_key=None, max_tokens=10, temperature=0.0,
                             timeout=1.0, retries=3)
    assert seen[0]["chat_template_kwargs"] == {"enable_thinking": False}
    assert "chat_template_kwargs" not in seen[1] and len(seen) == 2
    assert answer["text"] == PLAN and answer["thinking_switch"] == "rejected"


@pytest.mark.parametrize("hint,reason", [
    (PLAN, None),
    ("Take the revenue rows.\nDivide the later by the earlier.\nSubtract one.", None),
    ("Take the revenue rows.\nThe growth is 0.1446 of the base.\nReport it.", "contains_gold_number"),
    ("Take the revenue rows.\nIt comes to about 14.46 percent.\nReport it.", "contains_gold_number"),
    ("Take the revenue rows.\nThe ratio is 0.14455 here.\nReport it.", "contains_gold_number"),
    ("Take the revenue rows.\nDivide them.\nAnswer: 0.2", "writes_answer_line"),
])
def test_every_numeric_rule_drops_what_it_is_for(hint, reason):
    assert hints.drop_reason(hint, finqa_rows(1)[0]) == reason


def test_a_number_outside_one_percent_of_the_gold_is_kept():
    kept = "Take the revenue rows.\nThe table has 2 columns and 9 rows.\nDivide the later by the earlier."
    assert hints.drop_reason(kept, finqa_rows(1)[0]) is None


def test_a_yes_no_answer_is_dropped_only_on_a_yes_no_question():
    yes_no = finqa_rows(1, gold="yes")[0]
    assert hints.drop_reason("Compare the two years.\nThe answer is yes.\nSay so.", yes_no) == "states_yes_no"
    assert hints.drop_reason("Compare the two years.\nCheck which is larger.\nSay so.", yes_no) is None


def test_the_literal_rule_of_the_design_note_is_available_and_counted():
    """--allow-sql-statements keeps a SELECT that is not the gold query; the stricter default does not."""
    hint = "Find the table.\nWrite SELECT age FROM singer to look.\nThen filter."
    member = spider_rows(1)[0]
    assert hints.drop_reason(hint, member) == "contains_sql_statement"
    assert hints.drop_reason(hint, member, allow_sql_statements=True) is None


def test_the_filter_counts_every_drop_by_reason_and_computes_coverage(tmp_path):
    rows = spider_rows(4)
    given = [{"index": rows[0]["extra_info"]["index"], "hint": PLAN},
             {"index": rows[1]["extra_info"]["index"], "hint": "one line"},
             {"index": rows[2]["extra_info"]["index"],
              "hint": "Find it.\nRun %s.\nRead it." % GOLD_SQL},
             {"index": "a-question-from-somewhere-else", "hint": PLAN}]
    kept, dropped, counts = hints.filter_hints(given, rows)
    assert [entry["index"] for entry in kept] == [rows[0]["extra_info"]["index"]]
    assert counts["too_few_lines"] == 1 and counts["contains_gold_query"] == 1
    assert counts["no_such_question"] == 1
    assert len(dropped) == 3
    rows_path = tmp_path / "train.jsonl"
    rows_path.write_text(hints.as_jsonl(rows))
    hints_path = tmp_path / "hints.jsonl"
    hints_path.write_text(hints.as_jsonl(given))
    assert hints.main(["filter", "--hints", str(hints_path), "--rows", str(rows_path),
                       "--stuck-questions", "4", "--out", str(tmp_path / "f")]) == 0
    record = json.loads((tmp_path / "f" / "filter.json").read_text())
    assert record["kept"] == 1 and record["coverage"] == 0.25
    assert record["dropped_by_reason"]["contains_gold_query"] == 1


def test_two_hints_for_one_question_are_refused():
    rows = spider_rows(2)
    twice = [{"index": rows[0]["extra_info"]["index"], "hint": PLAN}] * 2
    with pytest.raises(hints.HintsError, match="two hints"):
        hints.filter_hints(twice, rows)


def test_the_coverage_is_a_share_of_the_stuck_set_not_of_the_hints_that_came_back(tmp_path):
    """A `generate` that stopped early must not flatter the coverage bar."""
    rows = spider_rows(10)
    stuck = tmp_path / "stuck.json"
    stuck.write_text(json.dumps({"schema": hints.SCHEMA, "stage": "stuck", "stuck": 10,
                                 "stuck_ids": [r["extra_info"]["index"] for r in rows]}))
    rows_path = tmp_path / "train.jsonl"
    rows_path.write_text(hints.as_jsonl(rows))
    given = [{"index": rows[i]["extra_info"]["index"], "hint": PLAN} for i in range(2)]
    hints_path = tmp_path / "hints.jsonl"
    hints_path.write_text(hints.as_jsonl(given))
    hints.main(["filter", "--hints", str(hints_path), "--rows", str(rows_path),
                "--stuck", str(stuck), "--out", str(tmp_path / "f")])
    record = json.loads((tmp_path / "f" / "filter.json").read_text())
    assert record["kept"] == 2 and record["coverage"] == 0.2


# --------------------------------------------------- 4. the hinted files, and the dose that survives
def prepared(tmp_path, rows, hinted_indices, **extra):
    rows_path = tmp_path / "train.jsonl"
    rows_path.write_text(hints.as_jsonl(rows))
    hints_path = tmp_path / "hints-filtered.jsonl"
    hints_path.write_text(hints.as_jsonl([{"index": index, "hint": PLAN} for index in hinted_indices]))
    argv = ["apply", "--rows", str(rows_path), "--hints", str(hints_path),
            "--out", str(tmp_path / "applied")]
    for key, value in extra.items():
        argv += ["--%s" % key.replace("_", "-")] + ([] if value is True else [str(value)])
    code = hints.main(argv)
    manifest = json.loads((tmp_path / "applied" / "apply.manifest.json").read_text())
    written = [json.loads(line) for line in
               (tmp_path / "applied" / "train.jsonl").read_text().split("\n") if line.strip()]
    return code, manifest, written


def test_the_hint_is_appended_to_the_stuck_prompts_and_to_nothing_else(tmp_path):
    rows = spider_rows(4)
    marked = [rows[0]["extra_info"]["index"], rows[2]["extra_info"]["index"]]
    code, manifest, written = prepared(tmp_path, rows, marked)
    assert code == 0 and manifest["hinted_rows"] == 2 and manifest["unhinted_rows"] == 2
    for original, after in zip(rows, written):
        before = original["prompt"][0]["content"]
        now = after["prompt"][0]["content"]
        if original["extra_info"]["index"] in marked:
            assert now == before + hints.HINT_HEADER + PLAN
            assert now.startswith(before), "the question itself must be untouched"
        else:
            assert json.dumps(after, sort_keys=True) == json.dumps(original, sort_keys=True)
    # the ground truth, the data source and the reward model are never touched, hinted or not
    assert [r["reward_model"] for r in written] == [r["reward_model"] for r in rows]


def test_a_row_that_changed_without_a_hint_is_a_refusal(tmp_path, monkeypatch):
    rows = spider_rows(3)
    monkeypatch.setattr(hints, "apply_hints",
                        lambda rows_, hints_: ([{**r, "ability": "tampered"} for r in rows_],
                                               [rows_[0]["extra_info"]["index"]]))
    with pytest.raises(SystemExit, match="row with no hint changed"):
        prepared(tmp_path, rows, [rows[0]["extra_info"]["index"]])


def test_a_hint_for_a_question_this_file_does_not_hold_is_a_refusal(tmp_path):
    """Otherwise the hinted file would silently equal the control's and the arm would measure nothing."""
    with pytest.raises(SystemExit, match="not in this training file"):
        prepared(tmp_path, spider_rows(2), ["a-question-from-another-bed"])


def test_an_empty_hints_file_is_a_refusal(tmp_path):
    with pytest.raises(SystemExit, match="has no rows"):
        prepared(tmp_path, spider_rows(2), [])


def test_the_faded_arm_gets_two_files_that_between_them_are_the_other_arms_questions(tmp_path):
    rows = spider_rows(8)
    marked = [r["extra_info"]["index"] for r in rows[:3]]
    code, manifest, written = prepared(tmp_path, rows, marked, fade_at_step=2, steps=4, batch=2)
    assert code == 0
    part1 = [json.loads(l) for l in (tmp_path / "applied" / "train-part1.jsonl").read_text().split("\n") if l.strip()]
    part2 = [json.loads(l) for l in (tmp_path / "applied" / "train-part2.jsonl").read_text().split("\n") if l.strip()]
    assert manifest["split_after_rows"] == 4
    assert len(part1) == 4 and len(part2) == 4
    # the same questions in the same order as the `hint` arm's single file, each exactly once
    assert [r["extra_info"]["index"] for r in part1 + part2] == [r["extra_info"]["index"] for r in rows]
    # part1 carries the hints, part2 carries none of them
    assert all(hints.HINT_HEADER in r["prompt"][0]["content"] for r in part1[:3])
    assert not any(hints.HINT_HEADER in r["prompt"][0]["content"] for r in part2)
    # and part1 is the first rows of the hinted file, unchanged
    assert json.dumps(part1, sort_keys=True) == json.dumps(written[:4], sort_keys=True)


def test_a_hinted_file_too_short_for_the_dose_is_refused_before_a_gpu_is_taken(tmp_path):
    rows = spider_rows(4)
    with pytest.raises(SystemExit, match="the trainer would run out of data"):
        prepared(tmp_path, rows, [rows[0]["extra_info"]["index"]], steps=4, batch=2)


#: Two short prompts and two long ones, with a budget that fits all four plain but only the short ones
#: once a hint is appended: `int(len / 3) + 1` tokens, the estimate kit/hints.py uses with no tokenizer.
MIXED_BUDGET = 120


def mixed_rows() -> list:
    rows = spider_rows(4)
    for index, member in enumerate(rows):
        member["prompt"][0]["content"] = "q" * (60 if index < 2 else 300)
    return rows


def test_a_hint_that_would_push_a_prompt_over_verls_limit_is_refused_by_default(tmp_path):
    rows = mixed_rows()
    marked = [r["extra_info"]["index"] for r in rows]
    with pytest.raises(SystemExit, match="the trainer would run out of data"):
        prepared(tmp_path, rows, marked, steps=2, batch=2, max_prompt_tokens=MIXED_BUDGET)


def test_the_declared_fallback_keeps_the_dose_and_counts_what_it_cost(tmp_path):
    rows = mixed_rows()
    marked = [r["extra_info"]["index"] for r in rows]
    code, manifest, written = prepared(tmp_path, rows, marked, steps=2, batch=2,
                                       max_prompt_tokens=MIXED_BUDGET, drop_hints_over_budget=True)
    assert code == 0
    assert manifest["hinted_rows"] == 2, "only the short prompts can carry a hint here"
    assert manifest["hints_dropped_over_budget"] == 2
    assert manifest["drop_hints_over_budget"] == 1
    assert manifest["rows_within_budget_after"] == 4, "the dose survives, at the cost of two hints"
    # the two questions whose hint was dropped come out byte-identical to the control's rows
    assert json.dumps(written[2:], sort_keys=True) == json.dumps(rows[2:], sort_keys=True)


def test_the_manifest_records_how_the_prompt_lengths_were_counted(tmp_path):
    rows = finqa_rows(4)
    _, manifest, _ = prepared(tmp_path, rows, [rows[0]["extra_info"]["index"]])
    assert manifest["prompt_tokens_from"].startswith("characters/")
    assert manifest["longest_prompt_after"] > manifest["longest_prompt_before"]
    assert manifest["rows_pushed_over_budget"] == 0
    assert manifest["header_sha256"] == hints.sha256_text(hints.HINT_HEADER)


def test_nothing_is_ever_overwritten(tmp_path):
    rows = spider_rows(4)
    prepared(tmp_path, rows, [rows[0]["extra_info"]["index"]])
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        prepared(tmp_path, rows, [rows[0]["extra_info"]["index"]])


def test_the_stuck_record_is_the_gates_arithmetic():
    """Stuck is 0 of 8 and nothing else: a question solved once in eight is not stuck, and hinting it
    would put a hint on a question the model can already sometimes do."""
    rows = spider_rows(4)
    per_question = [{"index": r["extra_info"]["index"], "wins": w, "attempts": 8}
                    for r, w in zip(rows, (0, 1, 8, 4))]
    record = hints.stuck_record(rows, per_question, attempts=8, temperature=1.0, model="m",
                               seconds=1.0, extra={})
    assert record["stuck"] == 1 and record["stuck_share"] == 0.25
    assert record["stuck_ids"] == [rows[0]["extra_info"]["index"]]
    assert record["solved_every_time"] == 1
    assert record["mean_win_rate"] == pytest.approx(13 / 32)
    assert record["wins_histogram"] == [1, 1, 0, 0, 1, 0, 0, 0, 1]
    assert record["attempts"] == 8 and record["temperature"] == 1.0


def test_the_header_that_joins_a_hint_to_a_question_is_pinned():
    """The header is part of the treatment: a run whose header moved is a different experiment from
    the one the numbers were measured in, so it is pinned here as a literal and by its sha256."""
    assert hints.HINT_HEADER == "\n\nA hint from a stronger model. It is a plan, not the answer:\n"
    assert hints.sha256_text(hints.HINT_HEADER) == \
        "906ad7dc754ac7dbe5f3fd7b864185fb71f182fd622aa3030f584d3c3a4754a9"


def test_the_fade_splits_where_the_dose_is_filled_and_not_at_the_nominal_row(tmp_path):
    """verl drops an overlong prompt, so the first H steps are filled by MORE rows than H x batch when
    some are dropped. A split at the nominal row would give the first half a short dose and hand the
    rows it never saw to the second half."""
    rows = spider_rows(8)
    for index, member in enumerate(rows):
        member["prompt"][0]["content"] = "q" * (600 if index < 2 else 60)
    marked = [rows[index]["extra_info"]["index"] for index in (2, 3)]
    code, manifest, _ = prepared(tmp_path, rows, marked, fade_at_step=1, steps=2, batch=2,
                                 max_prompt_tokens=MIXED_BUDGET)
    assert code == 0
    assert manifest["rows_over_budget_after"] == 2
    assert manifest["split_after_rows"] == 4, "two rows are dropped, so two steps need four rows"
    assert manifest["part1_rows_within_budget"] == 2 and manifest["part2_rows_within_budget"] == 4


def test_the_faded_arms_second_half_has_no_hint_in_it_even_where_the_hint_arm_had_one(tmp_path):
    rows = spider_rows(8)
    marked = [rows[index]["extra_info"]["index"] for index in (0, 1, 5)]
    code, manifest, written = prepared(tmp_path, rows, marked, fade_at_step=2, steps=4, batch=2)
    assert code == 0 and manifest["split_after_rows"] == 4
    part2 = [json.loads(l) for l in
             (tmp_path / "applied" / "train-part2.jsonl").read_text().split("\n") if l.strip()]
    assert hints.HINT_HEADER in written[5]["prompt"][0]["content"], "the hint arm hints row 5"
    assert not any(hints.HINT_HEADER in member["prompt"][0]["content"] for member in part2)
    assert json.dumps(part2, sort_keys=True) == json.dumps(rows[4:], sort_keys=True)


def test_the_hint_request_names_the_rules_the_filter_enforces():
    text = hints.HINT_INSTRUCTION.lower()
    assert "three to five lines" in text
    assert "do not state the final answer" in text
    assert "do not write a sql statement" in text and "answer:" in text


# ------------------------------------------------------------------------------ 5. the bars
def test_every_bar_has_a_numeric_limit_and_a_known_aggregate(campaign):
    for line in campaign["rows"]:
        for bar in line.get("bars") or []:
            assert "min" in bar or "max" in bar, (line["id"], bar)
            for limit in ("min", "max"):
                if limit in bar:
                    assert isinstance(bar[limit], (int, float)) and not isinstance(bar[limit], bool)
            assert bar.get("agg", "last") in runner.AGGREGATES


def test_every_bar_reads_a_file_one_of_this_kits_tools_writes(campaign):
    known = ("train-summary.json", "metrics.jsonl", "forgetting.json", "agreement.json",
             "bed-score.json", "stuck.json", "hints.manifest.json", "filter.json",
             "apply.manifest.json", "feedback-check.json", "k4-report.json")
    for line in campaign["rows"]:
        for bar in line.get("bars") or []:
            assert bar["source"].endswith(known), (line["id"], bar["source"])


def test_no_bar_gates_on_a_number_two_steps_cannot_measure(campaign):
    """One pilot stops every later row, so a bar that can fail by luck costs a round trip for nothing."""
    noisy = ("critic/score/mean", "actor/entropy", "self_distillation/success_group_fraction",
             "self_distillation/empty_target_batch")
    for line in campaign["rows"]:
        for bar in line.get("bars") or []:
            assert bar["key"] not in noisy, "%s gates on %s" % (line["id"], bar["key"])


def test_the_coverage_bar_is_the_design_notes_ninety_percent(campaign):
    for bed in BEDS:
        bar = next(b for b in next(r for r in campaign["rows"] if r["id"] == "%s-filter" % bed)["bars"]
                   if b["key"] == "coverage")
        assert bar["min"] == 0.90


def test_the_stuck_share_bands_bracket_the_gates_own_measurement(campaign):
    measured = {"spider": 171 / 640, "finqa": 200 / 640}
    for bed in BEDS:
        bar = next(b for b in next(r for r in campaign["rows"] if r["id"] == "%s-stuck" % bed)["bars"]
                   if b["key"] == "stuck_share")
        assert bar["min"] < measured[bed] < bar["max"], bed


def _summary(tmp_path: Path, **values) -> Path:
    path = tmp_path / "train-summary.json"
    path.write_text(json.dumps({"schema": "kit-sdpo-bed-run.v1", "returncode": 0, "merged": 1,
                                "steps": 40, "hints": 200, "feedback": 1, **values}))
    return path


def test_a_run_that_served_no_hints_fails_its_bar(campaign, tmp_path):
    bar = dict(next(b for b in next(r for r in campaign["rows"]
                                    if r["id"] == "spider-teacher-hint-seed0")["bars"]
                    if b["name"] == "hints-were-served"))
    for hints_served, ok in ((200, True), (0, False)):
        path = tmp_path / str(hints_served)
        path.mkdir()
        bar["source"] = str(_summary(path, hints=hints_served))
        assert runner.judge_bar(bar)["ok"] is ok


@pytest.mark.parametrize("arm,bar_name,passes", [
    ("teacher-hint", "teacher-saw-the-feedback", 1),
    ("teacher-none", "teacher-saw-no-feedback", 0),
])
def test_an_sdpo_run_at_the_wrong_switch_position_fails_its_bar(campaign, tmp_path, arm, bar_name,
                                                                passes):
    """The control and the treatment are one trainer key apart, so each row gates on its own position:
    a `teacher-none` run that trained with the teacher reading hints is not a control at all."""
    bar = dict(next(b for b in next(r for r in campaign["rows"]
                                    if r["id"] == "spider-%s-seed0" % arm)["bars"]
                    if b["name"] == bar_name))
    for feedback in (0, 1):
        path = tmp_path / ("%s-%d" % (arm, feedback))
        path.mkdir()
        bar["source"] = str(_summary(path, feedback=feedback, arm=arm))
        assert runner.judge_bar(bar)["ok"] is (feedback == passes)


# ------------------------------------------------------------------ 6. the campaign, as generated
@pytest.mark.skipif(generator is None, reason="scripts/ is not part of the exported kit")
def test_the_committed_campaign_is_what_its_generator_builds():
    assert CAMPAIGN.read_text() == generator.build(), "re-run scripts/make_k4_campaign.py"


def test_every_bed_arm_and_seed_is_trained_scored_and_reported(campaign):
    ids = {line["id"] for line in campaign["rows"]}
    report = next(line for line in campaign["rows"] if line["id"] == "report")
    for bed in BEDS:
        for arm in ARMS:
            for seed in SEEDS:
                point = "%s-%s-seed%d" % (bed, arm, seed)
                assert point in ids, point
                assert "%s-eval" % point in ids
                assert "%s-forget" % point in ids
                assert "%s-eval" % point in report["needs"]
                assert "%s-forget" % point in report["needs"]
        assert "base-%s" % bed in ids and "base-%s" % bed in report["needs"]


def test_the_reused_none_arm_is_never_retrained(campaign):
    """K3 and K1c already ran it. A K4 row that trained the GRPO control would be paying twice for it.

    `teacher-none` is a different arm: it is the SDPO route's control, which no earlier package ran,
    and it IS trained here.
    """
    for line in campaign["rows"]:
        for bed in BEDS:
            assert not line["id"].startswith("%s-none-" % bed), line["id"]
        assert "a-seed" != line["id"][:6]
    text = CAMPAIGN.read_text()
    assert "$K3_ROOT" in text and "$K1C_ROOT" in text
    ids = {line["id"] for line in campaign["rows"]}
    for bed in BEDS:
        assert {"%s-teacher-none-seed%d" % (bed, seed) for seed in SEEDS} <= ids


def test_each_beds_dose_is_its_own_controls_dose(campaign):
    for bed in BEDS:
        steps, fade = DOSE[bed]
        for seed in SEEDS:
            hint = next(r for r in campaign["rows"] if r["id"] == "%s-hint-seed%d" % (bed, seed))
            assert hint["env"]["STEPS"] == str(steps)
            first = next(r for r in campaign["rows"] if r["id"] == "%s-hint-faded-seed%d-first" % (bed, seed))
            second = next(r for r in campaign["rows"] if r["id"] == "%s-hint-faded-seed%d" % (bed, seed))
            assert first["env"]["STEPS"] == str(fade)
            assert second["env"]["STEPS"] == str(steps - fade)
            assert int(first["env"]["STEPS"]) + int(second["env"]["STEPS"]) == steps
            sdpo = [next(r for r in campaign["rows"] if r["id"] == "%s-%s-seed%d" % (bed, arm, seed))
                    for arm in SDPO_ARMS]
            for line in sdpo:
                assert line["env"]["STEPS"] == str(steps), line["id"]
            for line in (hint, first, second, *sdpo):
                assert line["env"]["SEED"] == str(seed)


def test_the_faded_arm_restarts_from_its_own_first_halfs_checkpoint(campaign):
    for bed in BEDS:
        steps, fade = DOSE[bed]
        for seed in SEEDS:
            second = next(r for r in campaign["rows"] if r["id"] == "%s-hint-faded-seed%d" % (bed, seed))
            command = " ".join(second["command"])
            assert "%s-hint-faded-seed%d-first-a*/hf-step%d" % (bed, seed, fade) in command
            assert "train-part2.parquet" in command
            assert "%s-hint-faded-seed%d-first" % (bed, seed) in second["needs"]


def test_the_teacher_arms_train_on_the_controls_own_file_and_the_others_on_a_hinted_one(campaign):
    for bed in BEDS:
        for arm in SDPO_ARMS:
            teacher = next(r for r in campaign["rows"] if r["id"] == "%s-%s-seed0" % (bed, arm))
            assert "hints-filtered.jsonl" in " ".join(teacher["command"]), arm
            assert "applied" not in teacher["env"].get("TRAIN_FILE", ""), arm
            assert teacher["env"]["TRAIN_FILE"].endswith("train.parquet"), arm
        hint = next(r for r in campaign["rows"] if r["id"] == "%s-hint-seed0" % bed)
        assert "%s-applied-a*" % bed in " ".join(hint["command"])


def test_the_two_sdpo_arms_differ_only_in_the_switch_and_each_row_gates_on_its_own_position(campaign):
    """Every row of the pair is the same row but for FEEDBACK, and each one PROVES which position it
    ran at: a control that ran with the switch on would otherwise be a second treatment arm."""
    for bed in BEDS:
        for seed in SEEDS:
            rows = {arm: next(r for r in campaign["rows"] if r["id"] == "%s-%s-seed%d" % (bed, arm, seed))
                    for arm in SDPO_ARMS}
            for arm, switch in SDPO_ARMS.items():
                assert rows[arm]["env"]["FEEDBACK"] == switch, (bed, arm)
            treated, control = rows["teacher-hint"], rows["teacher-none"]
            assert treated["command"] == control["command"], "one command, one switch"
            differ = {key for key in treated["env"]
                      if treated["env"][key] != control["env"].get(key)}
            assert differ == {"FEEDBACK", "NAME"}, (bed, seed, differ)
            bars = {arm: {bar["name"]: bar for bar in rows[arm]["bars"]} for arm in SDPO_ARMS}
            assert bars["teacher-hint"]["teacher-saw-the-feedback"]["min"] == 1
            assert bars["teacher-none"]["teacher-saw-no-feedback"]["max"] == 0
            for arm in SDPO_ARMS:
                assert "hints-were-served" in bars[arm], arm
                assert bars[arm]["hints-were-served"]["min"] == bars["teacher-hint"][
                    "hints-were-served"]["min"], "both arms score the attempts the same way"


def test_each_sdpo_pilot_runs_and_gates_on_its_own_switch_position(campaign):
    """Two steps of each SDPO arm before fifty runs of either, each at the position it is named for:
    a pilot that ran the other arm would pass while proving nothing about this one."""
    for bed in BEDS:
        for arm, switch in SDPO_ARMS.items():
            pilot = next(r for r in campaign["rows"] if r["id"] == "pilot-%s-%s" % (bed, arm))
            assert pilot["pilot"] is True
            assert pilot["env"]["FEEDBACK"] == switch, (bed, arm)
            names = {bar["name"] for bar in pilot["bars"]}
            assert ("teacher-saw-the-feedback" in names) is (switch == "1"), (bed, arm)
            assert ("teacher-saw-no-feedback" in names) is (switch == "0"), (bed, arm)


def test_no_required_path_and_no_env_value_hides_a_shell_expansion(campaign):
    """`requires` is tested with Path.exists and `env` values are literal: a `$(...)` in either would
    refuse every row for ever, or reach the trainer as eight characters."""
    for line in campaign["rows"]:
        for path in line.get("requires") or []:
            assert "$(" not in path and "*" not in path, (line["id"], path)
        for key, value in (line.get("env") or {}).items():
            assert "$(" not in value, (line["id"], key, value)


def test_the_pilots_come_first_and_gate_every_later_row(campaign):
    ids = [line["id"] for line in campaign["rows"]]
    pilots = [line["id"] for line in campaign["rows"] if line.get("pilot")]
    assert ids[0] == "base-forget-1" == pilots[0]
    for bed in BEDS:
        for name in ("%s-stuck" % bed, "%s-filter" % bed, "pilot-%s-hint" % bed,
                     "pilot-%s-hint-faded" % bed, "pilot-%s-teacher-none" % bed,
                     "pilot-%s-teacher-hint" % bed, "base-%s" % bed):
            assert name in pilots, name
    for line in campaign["rows"]:
        earlier = pilots[:pilots.index(line["id"])] if line["id"] in pilots else \
            [p for p in pilots if ids.index(p) < ids.index(line["id"])]
        assert set(earlier).issubset(set(line["needs"])), line["id"]


def test_spider_comes_before_finqa_so_a_finqa_failure_costs_no_spider_row(campaign):
    ids = [line["id"] for line in campaign["rows"]]
    assert ids.index("spider-teacher-hint-seed4") < ids.index("finqa-stuck")


def test_all_the_io_happens_before_a_gpu_is_held(campaign):
    prepared_rows = [line for line in campaign["rows"] if line.get("prepare")]
    assert [line["id"] for line in prepared_rows] == ["base-forget-1"], \
        "preparation belongs to the first row"
    steps = " ".join(" ".join(step) for step in prepared_rows[0]["prepare"])
    assert "beds/spider.py" in steps and "beds/finqa.py" in steps
    assert "Qwen3-1.7B" in steps
    assert "HINT_BASE_URL" in steps, "the served hint-giver must answer before a GPU is taken"
    assert "K3_ROOT" in steps and "K1C_ROOT" in steps, \
        "the control must be on disk before the whole grid runs"


def test_the_hint_pipeline_holds_at_most_one_gpu_and_only_for_the_stuck_set(campaign):
    for bed in BEDS:
        stuck = next(r for r in campaign["rows"] if r["id"] == "%s-stuck" % bed)
        assert stuck["env"]["CUDA_VISIBLE_DEVICES"] == "0"
        for name in ("%s-hints" % bed, "%s-filter" % bed, "%s-apply" % bed, "%s-hints-serve" % bed):
            line = next(r for r in campaign["rows"] if r["id"] == name)
            assert "CUDA_VISIBLE_DEVICES" not in line["env"], name


def test_every_scoring_row_scores_on_gpu_zero(campaign):
    for line in campaign["rows"]:
        if line["id"].endswith(("-eval", "-forget")) or line["id"].startswith("base-"):
            if line["id"] == "base-repeatable":
                continue
            assert line["env"].get("CUDA_VISIBLE_DEVICES") == "0", line["id"]


def test_planning_the_campaign_creates_nothing(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    done = subprocess.run([sys.executable, str(KIT / "runner.py"), "plan", str(CAMPAIGN)],
                          capture_output=True, text=True, env={**os.environ, "WORK": str(work)})
    assert done.returncode == 0, done.stderr
    assert list(work.iterdir()) == []
    assert "PILOT" in done.stdout


# ------------------------------------------------------------------------------ 7. the report
def _bed_score(directory: Path, stem: str, per_item: dict, machine="m1", n=None, attempt=1) -> None:
    out = directory / ("%s-a%d" % (stem, attempt))
    out.mkdir(parents=True)
    correct = sum(1 for value in per_item.values() if value >= 1)
    (out / "bed-score.json").write_text(json.dumps(
        {"schema": "kit-bed-score.v1", "n": n or len(per_item), "correct": correct,
         "accuracy": round(correct / max(1, len(per_item)), 6), "incorrect_format": 0,
         "per_item": per_item, "machine": {"id": machine}, "model": stem}))


def _panels(directory: Path, stem: str, correct=(90, 80, 82), machine="m1", attempt=1) -> None:
    out = directory / ("%s-a%d" % (stem, attempt))
    out.mkdir(parents=True)
    names = ("instructions", "knowledge", "maths")
    (out / "forgetting.json").write_text(json.dumps(
        {"panels": {name: {"correct": value, "n": 100} for name, value in zip(names, correct)},
         "total_correct": sum(correct), "machine": {"id": machine}}))


def _run(runs: Path, point: str, steps: int, attempt: int = 1, **values) -> None:
    out = runs / ("%s-a%d" % (point, attempt))
    out.mkdir(parents=True)
    (out / "train-summary.json").write_text(json.dumps(
        {"schema": "kit-grpo-run.v1", "name": point, "steps": steps, "returncode": 0, "merged": 1,
         **values}))
    (out / "metrics.jsonl").write_text("\n".join(json.dumps(
        {"step": step, "data": {"critic/score/mean": 0.4, "response_length/mean": 96.0,
                                "actor/entropy": 0.2, "actor/grad_norm": 0.4,
                                "perf/time_per_step": 15.0, "perf/max_memory_allocated_gb": 40.0}})
        for step in range(1, steps + 1)) + "\n")


ITEMS = {"spider": ["spider-train_spider-db-%d" % i for i in range(10)],
         "finqa": ["ADI/2009/page_%d.pdf-1" % i for i in range(10)]}
#: The untrained model answers the first four of every ten; the last six are its stuck set.
UNTRAINED = {bed: {item: (1 if index < 4 else 0) for index, item in enumerate(ITEMS[bed])}
             for bed in BEDS}


def _scored(bed: str, correct: int, machine="m1") -> dict:
    """A per-item map with `correct` of the ten items right, the first four first."""
    return {item: (1 if index < correct else 0) for index, item in enumerate(ITEMS[bed])}


#: What each arm answers of the ten held-out items in the fixture below. The control arms -- `none`,
#: reused, and `teacher-none`, trained here -- answer 4 and 5; the two hinted arms answer 7. So the
#: GRPO route's effect is +0.3 and the SDPO route's is +0.2, and the two cannot be confused.
ARM_CORRECT = {"hint": 7, "hint-faded": 5, "teacher-none": 5, "teacher-hint": 7}
NONE_CORRECT = 4


@pytest.fixture()
def tree(tmp_path):
    """A K4 work tree built from the CAMPAIGN FILE'S OWN OUT templates, so the report cannot look up a
    folder the campaign does not write (receipt 225: the K4a report's forgetting table was all
    dashes because the two names had drifted apart)."""
    import yaml
    spec = yaml.safe_load(CAMPAIGN.read_text())
    work = tmp_path / "work"
    outs = [(line["id"], (line.get("env") or {}).get("OUT", "")) for line in spec["rows"]]
    root, runs = work / "k4", work / "runs"
    for row_id, out in outs:
        if not out or "{attempt}" not in out:
            continue
        path = Path(out.replace("{work}", str(work)).replace("{attempt}", "1"))
        if "/eval/" in out and row_id.endswith("-eval"):
            point = path.name[:-3]
            named = k4_report.POINT.match(point)
            assert named, point
            _bed_score(root / "eval", point,
                       _scored(named["bed"], ARM_CORRECT[named["arm"]]))
        elif "/eval/" in out and row_id.startswith("base-"):
            bed = row_id[len("base-"):]
            _bed_score(root / "eval", "base-%s" % bed, UNTRAINED[bed])
        elif "/forgetting/" in out and row_id.endswith("-forget"):
            _panels(root / "forgetting", path.name[:-3])
        elif out.endswith("/forgetting/base-a{attempt}"):
            _panels(root / "forgetting", "base")
    for bed in BEDS:
        steps = DOSE[bed][0]
        for arm in ARMS:
            for seed in SEEDS:
                _run(runs, "%s-%s-seed%d" % (bed, arm, seed),
                     DOSE[bed][1] if arm == "hint-faded" else steps,
                     arm=arm, hints=200 if arm in SDPO_ARMS else None,
                     feedback=int(SDPO_ARMS[arm]) if arm in SDPO_ARMS else None)
    # the two finished packages the `none` arm is read from, in their own names
    none_roots = {}
    for bed in BEDS:
        package = tmp_path / ("%s-package" % bed)
        names = k4_report.NONE_TREES[bed]
        for seed in SEEDS:
            _bed_score(package / "eval", names["eval"] % seed, _scored(bed, NONE_CORRECT))
            _panels(package / "forgetting", names["forget"][0] % seed)
        none_roots[bed] = package
    rows = tmp_path / "finqa-test.jsonl"
    rows.write_text("".join(json.dumps(
        {"data_source": "finqa", "extra_info": {"index": item, "gold_consistent": index != 9}}) + "\n"
        for index, item in enumerate(ITEMS["finqa"])))
    return {"root": root, "runs": runs, "none_roots": none_roots, "finqa_rows": rows, "work": work}


def build(tree, **extra):
    return k4_report.build(tree["root"], tree["runs"],
                           none_roots={bed: str(tree["none_roots"][bed]) for bed in BEDS},
                           finqa_rows=str(tree["finqa_rows"]), **extra)


def test_the_report_joins_every_scoring_the_campaign_writes(tree):
    report = build(tree)
    assert report["beds_reported"] == 2
    assert report["arms_reported"] == len(ARMS) + 1 and report["none_runs"] == 2 * len(SEEDS)
    assert report["comparable"] == 1
    assert report["flags"] == [], report["flags"]
    for bed in BEDS:
        for arm in ("none",) + ARMS:
            block = report["beds"][bed]["arms"][arm]
            assert block["seeds_scored"] == len(SEEDS), (bed, arm)
            for seed in SEEDS:
                row = block["seeds"][str(seed)]
                assert row["eval"]["found"] == 1 and row["panels"]["found"] == 1
                assert row["panel_change"], (bed, arm, seed)


def test_the_gain_is_paired_seed_for_seed_against_the_reused_control(tree):
    report = build(tree)
    # spider: hint answers 7 of 10, the control 4 of 10 -> +0.3; finqa excludes one flagged item
    spider = report["beds"]["spider"]["arms"]["hint"]
    assert spider["gain_per_seed"]["0"] == pytest.approx(0.3)
    assert spider["gain_paired"]["mean"] == pytest.approx(0.3)
    assert spider["gain_paired"]["n"] == len(SEEDS)
    assert report["beds"]["spider"]["verdict"]["verdict"] == "BAR MET"
    assert report["beds"]["spider"]["verdict"]["gain"] == pytest.approx(0.3)


def test_the_verdict_uses_the_three_point_bar_and_nothing_else(tree):
    report = build(tree)
    assert report["gain_bar"] == 0.03
    for bed in BEDS:
        verdict = report["beds"][bed]["verdict"]
        assert verdict["bar"] == 0.03 and verdict["seeds"] == len(SEEDS)
        # it is a bar on `hint`, never on the other three arms
        assert verdict["gain"] == pytest.approx(report["beds"][bed]["arms"]["hint"]["gain_paired"]["mean"])
        assert report["beds"][bed]["routes"]["grpo"]["is_the_verdict"] == 1
        assert all(report["beds"][bed]["routes"][name]["is_the_verdict"] == 0
                   for name in ("sdpo", "trainer-change"))


def test_a_gain_under_the_bar_is_bar_not_met_with_its_reason(tree):
    for bed in BEDS:
        for seed in SEEDS:
            _bed_score(tree["root"] / "eval", "%s-hint-seed%d" % (bed, seed), _scored(bed, 4), attempt=2)
    report = build(tree)
    for bed in BEDS:
        assert report["beds"][bed]["verdict"]["verdict"] == "BAR NOT MET"
        assert "under the 0.03 bar" in report["beds"][bed]["verdict"]["reason"]


def test_a_panel_loss_larger_than_the_gain_fails_the_bar_even_when_the_gain_is_big(tree):
    for seed in SEEDS:
        _panels(tree["root"] / "forgetting", "spider-hint-seed%d" % seed, correct=(90, 80, 70),
                attempt=2)
    report = build(tree)
    verdict = report["beds"]["spider"]["verdict"]
    assert verdict["verdict"] == "BAR NOT MET" and verdict["panel_ok"] == 0
    assert "past the floor" in verdict["reason"]


def test_a_gain_against_none_from_the_other_route_is_labelled_not_a_controlled_comparison(tree):
    """`teacher-hint` minus `none` is the hint AND the change of trainer, and stays labelled as such
    even now that the SDPO route has its own control."""
    report = build(tree)
    for bed in BEDS:
        arms = report["beds"][bed]["arms"]
        for arm in SDPO_ARMS:
            assert arms[arm]["controlled_comparison"] == 0, arm
            assert arms[arm]["route"] == "sdpo" and arms[arm]["compared_against"] == "none"
        for arm in ("hint", "hint-faded"):
            assert arms[arm]["controlled_comparison"] == 1 and arms[arm]["route"] == "grpo"
    text = k4_report.render(report)
    assert "changes the trainer (SDPO, with a teacher) as well as the hint" in text
    assert "is not the verdict" in text


def test_each_route_gets_its_own_effect_against_its_own_control(tree):
    """The addition K4 was held up for: `hint` minus `none` is the GRPO route, `teacher-hint` minus
    `teacher-none` is the SDPO route, and neither is read as the other."""
    report = build(tree)
    spider = report["beds"]["spider"]["routes"]
    assert spider["grpo"]["treated"] == "hint" and spider["grpo"]["control"] == "none"
    assert spider["grpo"]["paired"]["mean"] == pytest.approx(0.3)      # 7 of 10 against 4 of 10
    assert spider["sdpo"]["treated"] == "teacher-hint" and spider["sdpo"]["control"] == "teacher-none"
    assert spider["sdpo"]["paired"]["mean"] == pytest.approx(0.2)      # 7 of 10 against 5 of 10
    assert spider["sdpo"]["per_seed"]["0"] == pytest.approx(0.2)
    assert spider["sdpo"]["paired"]["n"] == len(SEEDS)
    # and the change of trainer itself, with no hint on either side, is its own line and not a hint
    assert spider["trainer-change"]["treated"] == "teacher-none"
    assert spider["trainer-change"]["control"] == "none"
    assert spider["trainer-change"]["paired"]["mean"] == pytest.approx(0.1)
    # the three are consistent: (teacher-hint - none) = trainer change + the SDPO route's hint effect
    crossed = report["beds"]["spider"]["arms"]["teacher-hint"]["gain_paired"]["mean"]
    assert crossed == pytest.approx(spider["trainer-change"]["paired"]["mean"]
                                    + spider["sdpo"]["paired"]["mean"])
    # finqa excludes one flagged item, so its route effects are ninths and not tenths
    finqa = report["beds"]["finqa"]["routes"]
    assert finqa["sdpo"]["paired"]["mean"] == pytest.approx(2 / 9)


def test_the_report_says_which_route_every_number_belongs_to(tree):
    text = k4_report.render(build(tree))
    assert "## Which route does each number belong to?" in text
    assert "`teacher-hint` - `teacher-none`" in text and "`hint` - `none`" in text
    assert "SDPO instead of GRPO, with no hint on either side" in text
    assert "**(the verdict)**" in text


def test_a_seed_missing_on_one_side_of_a_route_is_dropped_from_that_route_and_not_zeroed(tree):
    import shutil
    shutil.rmtree(tree["root"] / "eval" / "spider-teacher-none-seed3-a1")
    report = build(tree)
    route = report["beds"]["spider"]["routes"]["sdpo"]
    assert route["per_seed"]["3"] is None
    assert route["paired"]["n"] == len(SEEDS) - 1
    assert route["paired"]["mean"] == pytest.approx(0.2)


def test_an_sdpo_run_filed_under_the_wrong_switch_position_is_a_flag(tree):
    """A control that trained with the teacher reading hints is not a control, and the accuracy table
    alone cannot show it: the run's own record of the switch is checked against the arm's name."""
    _run(tree["runs"], "spider-teacher-none-seed0", DOSE["spider"][0], attempt=2,
         arm="teacher-hint", hints=200, feedback=1)
    report = build(tree)
    row = report["beds"]["spider"]["arms"]["teacher-none"]["seeds"]["0"]
    assert row["run"]["feedback_declared"] == 1
    assert any("WRONG FEEDBACK SWITCH" in flag for flag in row["flags"])
    assert any("WRONG ARM" in flag for flag in row["flags"])
    assert any("WRONG FEEDBACK SWITCH" in flag for flag in report["flags"])


def test_the_report_splits_the_held_out_items_by_what_the_untrained_model_could_do(tree):
    report = build(tree)
    block = report["beds"]["spider"]["arms"]["hint"]
    split = block["seeds"]["0"]["split"]
    assert split["already_solved_items"] == 4 and split["stuck_items"] == 6
    # hint answers 7 of 10: the 4 it already had, plus 3 of the 6 stuck ones
    assert split["already_solved_accuracy_now"] == pytest.approx(1.0)
    assert split["stuck_accuracy_now"] == pytest.approx(0.5)
    assert block["stuck_accuracy"]["mean"] == pytest.approx(0.5)


def test_finqas_flagged_items_are_excluded_from_every_number(tree):
    report = build(tree)
    block = report["beds"]["finqa"]
    assert block["excluded_items"] == 1
    # nine judged items, of which `hint` answers seven
    row = block["arms"]["hint"]["seeds"]["0"]
    assert row["judged"]["n"] == 9 and row["judged"]["correct"] == 7
    assert row["all_items"]["n"] == 10
    assert row["judged"]["accuracy"] == pytest.approx(7 / 9)
    assert report["beds"]["spider"]["excluded_items"] is None


def test_without_the_finqa_rows_the_exclusion_is_not_silently_skipped(tree):
    report = k4_report.build(tree["root"], tree["runs"],
                             none_roots={bed: str(tree["none_roots"][bed]) for bed in BEDS})
    assert report["finqa_rows"] is None
    assert report["beds"]["finqa"]["arms"]["hint"]["seeds"]["0"]["judged"]["n"] == 10


def test_the_report_refuses_to_subtract_counts_from_two_machines(tree):
    _bed_score(tree["root"] / "eval", "spider-hint-seed4", _scored("spider", 8),
               machine="another-machine", attempt=2)
    with pytest.raises(k4_report.K4ReportError, match="fingerprints"):
        build(tree)
    mixed = build(tree, allow_different_machines=True)
    assert mixed["comparable"] == 0 and mixed["different_machines_allowed"] == 1


def test_a_missing_scoring_is_a_flag_with_the_path_it_looked_for_not_a_dash(tree):
    import shutil
    shutil.rmtree(tree["root"] / "eval" / "spider-hint-seed2-a1")
    report = build(tree)
    row = report["beds"]["spider"]["arms"]["hint"]["seeds"]["2"]
    assert row["eval"]["found"] == 0
    assert any("NO HELD-OUT SCORING" in flag for flag in row["flags"])
    assert "spider-hint-seed2-a<N>" in row["eval"]["looked_for"]
    assert any("NO HELD-OUT SCORING" in flag for flag in report["flags"])
    assert report["beds"]["spider"]["arms"]["hint"]["gain_paired"]["n"] == len(SEEDS) - 1


def test_a_missing_control_tree_is_a_flag_and_never_a_zero(tree):
    report = k4_report.build(tree["root"], tree["runs"], none_roots={"spider": None, "finqa": None})
    assert report["none_runs"] == 0
    assert any("NO CONTROL TREE" in flag for flag in report["flags"])
    assert report["beds"]["spider"]["verdict"]["verdict"] == "NO VERDICT"


def test_the_faded_arms_steps_are_its_two_halves_and_not_just_the_second(tree):
    """A table showing 10 where the arm trained 20 would read as a shorter arm, not a faded one."""
    for bed in BEDS:
        steps, fade = DOSE[bed]
        for seed in SEEDS:
            _run(tree["runs"], "%s-hint-faded-seed%d-first" % (bed, seed), fade, arm="hint-faded")
        row = build(tree)["beds"][bed]["arms"]["hint-faded"]["seeds"]["0"]
        assert row["run"]["steps"] == steps - fade
        assert row["run"]["steps_total"] == steps
        assert row["run"]["first_half"]["steps"] == fade


def test_a_faded_arm_whose_first_half_is_missing_does_not_report_a_step_count(tree):
    row = build(tree)["beds"]["spider"]["arms"]["hint-faded"]["seeds"]["0"]
    assert row["run"]["first_half"]["found"] == 0
    assert row["run"]["steps_total"] is None


def test_the_report_takes_the_highest_attempt_of_every_scoring(tree):
    _bed_score(tree["root"] / "eval", "spider-hint-seed0", _scored("spider", 9), attempt=2)
    report = build(tree)
    row = report["beds"]["spider"]["arms"]["hint"]["seeds"]["0"]
    assert row["eval"]["attempt"] == 2 and row["judged"]["correct"] == 9


def test_the_report_renders_and_never_overwrites(tree, tmp_path):
    out = tmp_path / "report-a1"
    argv = ["--root", str(tree["root"]), "--runs", str(tree["runs"]),
            "--spider-none-root", str(tree["none_roots"]["spider"]),
            "--finqa-none-root", str(tree["none_roots"]["finqa"]),
            "--finqa-rows", str(tree["finqa_rows"]), "--out", str(out)]
    assert k4_report.main(argv) == 0
    text = (out / "k4-report.md").read_text()
    for bed in BEDS:
        assert bed.upper() in text
    for arm in ARMS:
        assert arm in text
    assert "BAR MET" in text and "reused" in text
    json.loads((out / "k4-report.json").read_text())
    with pytest.raises(SystemExit):
        k4_report.main(argv)


# ------------------------------------------------------------------------- 8. the instructions
def test_every_command_the_readme_gives_names_this_campaign():
    text = README.read_text()
    for action in ("plan", "prepare", "run", "status"):
        assert "runner.py %s $KIT/campaigns/k4-hints.yaml" % action in text, action
    for variable in ("KIT", "WORK", "SDPO_DIR", "SPIDER_ROOT", "FINQA_ROOT", "HINT_BASE_URL",
                     "HINT_MODEL", "K3_ROOT", "K1C_ROOT"):
        assert "export" in text and variable in text, variable


def test_the_readme_names_the_arms_the_campaign_actually_has_and_the_bars_it_gates_on():
    text = README.read_text()
    for arm in ARMS + ("none",):
        assert arm in text, arm
    assert "90" in text, "the coverage bar must be named"
    assert "3 points" in text or "3-point" in text
    assert "20 steps" in text and "40 steps" in text, "the two doses must both be stated"


def test_the_readme_is_honest_about_the_two_things_a_reader_could_misread():
    text = README.read_text()
    assert "teacher-hint" in text
    # each route's number is read against its OWN control, and the reader is told which is which
    assert "`teacher-hint` minus `teacher-none`" in text
    assert "`hint` minus `none`" in text
    assert "the change of trainer" in text
    assert "same machine" in text


def test_the_readme_arms_table_gives_every_arm_and_the_route_it_ran_on():
    """The table is what the partner reads before starting. An arm missing from it, or an arm whose
    route is not stated, is how a number gets compared with the wrong control."""
    text = README.read_text()
    routes = {"none": "GRPO", "hint": "GRPO", "hint-faded": "GRPO",
              "teacher-none": "SDPO", "teacher-hint": "SDPO"}
    for arm, route in routes.items():
        line = next((l for l in text.split("\n") if l.startswith("| **%s**" % arm)), None)
        assert line, "no arms-table row for %s" % arm
        assert "| %s |" % route in line, (arm, route)


def test_the_readme_prices_the_arm_it_asks_for():
    """The control arm is a third more GPU time than the package without it: a partner who is asked
    for it must be able to read what it costs before starting."""
    text = README.read_text()
    assert "teacher-none" in text
    assert "65 to 75 GPU-hours" in text and "about 43 are the training" in text
    assert "20 runs at 20 steps" in text and "20 runs at 40 steps" in text


def test_the_readme_names_a_row_the_campaign_has(campaign):
    ids = {line["id"] for line in campaign["rows"]}
    text = README.read_text()
    named = [word for word in text.split() if word.startswith("--row")]
    assert named, "the README should show how to run one row"
    for line in text.split("\n"):
        if "--row " in line:
            row_id = line.split("--row ", 1)[1].split()[0]
            assert row_id in ids, row_id


def test_normalise_sql_ignores_spacing_so_a_quoted_gold_query_cannot_slip_past_the_filter():
    assert hints.normalise_sql("SELECT   count(*) FROM t  WHERE a = 1 ;") == hints.normalise_sql('select count(*) from t where a=1')
    assert hints.normalise_sql('SELECT "x" FROM t') == hints.normalise_sql("select 'x' from t")
