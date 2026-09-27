"""The K3 stage-A dose probe: its campaign, kit/subset.py and kit/k3_dose_report.py.

The probe exists because K3's stage A moved the Spider held-out score from 70 to 62 against a +5
bar, with 21 right -> wrong and 13 wrong -> right. Six things are checked here, and each of them is a
way the probe could quietly answer the wrong question:

(a) every training row's environment is K3's stage-A environment except the six keys an arm is
    ALLOWED to change. An arm that also moved the seed, the GPU count or the KL switch would be two
    changes at once and could not be read as a dose.
(b) the four `*-delta` rows are leaves and the report does not need them, so an arm that fails its
    delta rows record the number; the +5 verdict is the report's, so nothing halts `run --all`.
(c) kit/subset.py drops exactly the questions recorded with wins == attempts, refuses a recorded
    question the training file does not hold, refuses a subset too short to fill one step, and never
    overwrites.
(d) the report's paired churn and its exact McNemar p, against hand-computed numbers: 21 and 13 must
    give a delta of -8 and p = 0.229, and an arm at +7 must clear the bar.
(e) a run with no metrics.jsonl still produces a report, with the missing file named.
(f) the density cells appear when a scoring carries `output_tokens_total` and are `-` when it does not.
(g) probe 2 (kit/campaigns/k3-dose-2.yaml): four arms of three runs, the runs of an arm identical but
    for their SEED label, every run probe 1's `ref20` but for the keys a budgeted dose may change, the
    budget runs carrying LENGTH_BUDGET 800 and `ref20`'s none, every run with its own scoring and
    leaf delta row; the report with --campaign groups the runs into arms and applies the arm rule
    (mean +5, mean density within 1.5x, 2 of 3 runs at +5) -- while without --campaign it renders
    probe 1's fixture byte for byte as before.

Everything builds its own fixtures; nothing here needs a GPU, a network or a trained model.
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
CAMPAIGN = KIT / "campaigns" / "k3-dose-probe.yaml"
K3_CAMPAIGN = KIT / "campaigns" / "k3-replay.yaml"


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = load("kit_runner_for_k3_dose", KIT / "runner.py")
subset = load("kit_subset", KIT / "subset.py")
hints = load("kit_hints_for_k3_dose", KIT / "hints.py")
sequence = load("kit_sequence_for_k3_dose", KIT / "sequence.py")
report_tool = load("kit_k3_dose_report", KIT / "k3_dose_report.py")

ARMS = ("ref20", "steps60", "lr3x", "hard20")
#: The keys an arm of this probe is allowed to differ from K3's stage A in. Everything else -- the
#: seed, the GPU count, the KL switch, the validation file, the work directory -- must be identical.
ALLOWED_TO_DIFFER = {"NAME", "TRAIN_FILE", "STEPS", "SAVE_FREQ", "LR", "FILE_LOG"}


@pytest.fixture()
def campaign():
    os.environ.setdefault("WORK", "/tmp/unused-k3-dose")
    return runner.load_campaign(CAMPAIGN)


@pytest.fixture()
def k3_campaign():
    os.environ.setdefault("WORK", "/tmp/unused-k3-dose")
    return runner.load_campaign(K3_CAMPAIGN)


def row_of(campaign: dict, row_id: str) -> dict:
    return next(row for row in campaign["rows"] if row["id"] == row_id)


# ============================================================ (a) the launcher-parity of every arm
def test_every_arms_environment_is_k3_stage_as_but_for_the_six_keys_a_dose_may_change(campaign,
                                                                                      k3_campaign):
    """One change per arm. A row that also moved SEED, NGPU, KL, WORK or VAL_FILE would be two."""
    stage_a = row_of(k3_campaign, "a-seed0")["env"]
    for arm in ARMS:
        env = row_of(campaign, arm)["env"]
        shared = {key: value for key, value in env.items() if key not in ALLOWED_TO_DIFFER}
        theirs = {key: value for key, value in stage_a.items() if key not in ALLOWED_TO_DIFFER}
        assert shared == theirs, arm
        assert env["SEED"] == "0" and env["NGPU"] == "8" and env["KL"] == "0", arm
        assert env["FILE_LOG"] == "1", "%s: the report reads metrics.jsonl, which needs FILE_LOG" % arm
        assert env["SAVE_FREQ"] == env["STEPS"], arm
        assert env["NAME"] == "%s-seed0-a{attempt}" % arm


def test_each_arm_is_the_dose_it_is_named_for(campaign):
    doses = {"ref20": ("20", "1e-5"), "steps60": ("60", "1e-5"),
             "lr3x": ("20", "3e-5"), "hard20": ("20", "1e-5")}
    for arm, (steps, lr) in doses.items():
        env = row_of(campaign, arm)["env"]
        assert (env["STEPS"], env["LR"]) == (steps, lr), arm
        assert any(bar["key"] == "steps" and bar["min"] == int(steps)
                   for bar in row_of(campaign, arm)["bars"]), arm


def test_each_arm_trains_on_the_file_its_change_needs(campaign):
    """The reference file for ref20 and lr3x, the 1,920-row pool for steps60, the subset for hard20."""
    commands = {arm: " ".join(row_of(campaign, arm)["command"]) for arm in ARMS}
    for arm in ("ref20", "lr3x"):
        assert "TRAIN_FILE=\"{work}/data/spider/train.parquet\"" in commands[arm], arm
    assert "spider1920-a*" in commands["steps60"] and "pool1920" in row_of(campaign, "steps60")["needs"]
    assert "hard640-a*" in commands["hard20"] and "subset" in row_of(campaign, "hard20")["needs"]
    for arm in ARMS:
        assert "run_grpo.sh" in commands[arm] and "models/Qwen3-1.7B" in commands[arm], arm


def test_the_longer_file_is_built_and_not_asked_of_the_trainer(campaign):
    """data.shuffle=False and total_epochs=1: 60 steps x 32 is 1,920 rows, written by sequence.py."""
    pooled = " ".join(row_of(campaign, "pool1920")["command"])
    assert "sequence.py\" pool" in pooled and "--rows 1920" in pooled and "--seed 0" in pooled
    launcher = (KIT / "run_grpo.sh").read_text()
    assert "data.shuffle=False" in launcher and "trainer.total_epochs=1" in launcher
    assert int(row_of(campaign, "steps60")["env"]["STEPS"]) * 32 == 1920
    hard = " ".join(row_of(campaign, "subset")["command"])
    assert "subset.py\" drop-solved" in hard and "--rows 640 --seed 0" in hard
    assert int(row_of(campaign, "hard20")["env"]["STEPS"]) * 32 == 640


def test_the_stuck_row_samples_the_trainers_own_way_on_one_gpu(campaign):
    row = row_of(campaign, "stuck")
    command = " ".join(row["command"])
    assert "hints.py\" stuck" in command and "--attempts 8" in command and "--temperature 1.0" in command
    assert row["env"]["CUDA_VISIBLE_DEVICES"] == "0" and "NGPU" not in row["env"]
    assert (hints.ATTEMPTS, hints.TEMPERATURE) == (8, 1.0), "the row must be hints.py's own sampling"


def test_training_uses_every_gpu_and_scoring_uses_only_the_first(campaign):
    for row in campaign["rows"]:
        env, command = row["env"], " ".join(row["command"])
        if "NGPU" in env:
            assert env["NGPU"] == "8" and "CUDA_VISIBLE_DEVICES" not in env, row["id"]
        elif "eval_bed" in command or "hints.py\" stuck" in command:
            assert env.get("CUDA_VISIBLE_DEVICES") == "0", row["id"]


def test_every_row_has_a_bar_that_reads_a_number_some_tool_writes(campaign):
    written_as_numbers = {
        "n": "eval_bed.py writes it in bed-score.json",
        "questions": "hints.py stuck", "solved_every_time": "hints.py stuck",
        "kept": "subset.py", "rows_total": "sequence.py pool",
        "returncode": "run_grpo.sh", "merged": "run_grpo.sh", "steps": "run_grpo.sh",
        "delta": "delta.py", "same_machine_flag": "delta.py",
        "arms_reported": "k3_dose_report.py",
    }
    for row in campaign["rows"]:
        assert row["bars"], "%s has no bar: nothing would judge it" % row["id"]
        for bar in runner.resolve(row, campaign, 1)["bars"]:
            assert bar["key"] in written_as_numbers, (row["id"], bar["key"])
            assert "min" in bar or "max" in bar, (row["id"], bar)
            assert bar["source"].endswith(".json"), (row["id"], bar)


def test_nothing_needs_a_private_model_or_repository(campaign):
    text = CAMPAIGN.read_text()
    assert "HopitAI" not in text and "hf auth" not in text
    downloads = [line for line in text.split("\n") if "hf download" in line]
    assert downloads and all("Qwen/Qwen3-1.7B" in line for line in downloads), downloads
    for row in campaign["rows"]:
        ran = " ".join(row["command"]) + " " + " ".join(" ".join(s) for s in row.get("prepare") or [])
        assert "continual." not in ran and "continual/" not in ran, row["id"]
        assert "sdft" not in ran and "modal" not in ran.lower(), row["id"]
        for script in ("runner.py", "run_grpo.sh", "eval_bed.py", "delta.py", "hints.py",
                       "subset.py", "sequence.py", "k3_dose_report.py", "beds/spider.py"):
            if script in ran:
                assert (KIT / script).is_file(), script


def test_the_probe_reuses_k3s_own_paths_so_one_work_tree_serves_both(campaign, k3_campaign):
    text = CAMPAIGN.read_text()
    for path in ("{work}/data/spider", "{work}/models/Qwen3-1.7B"):
        assert path in text and path in K3_CAMPAIGN.read_text()
    for row in campaign["rows"]:
        for key, value in row["env"].items():
            if key == "WORK" or not value.startswith("{work}"):
                continue
            if "/models/" in value or "/data/spider" in value:
                continue
            assert value.startswith("{work}/k3dose/"), (row["id"], value)


# ================================================== (b) the delta rows are leaves, the report is not
def test_the_delta_rows_are_leaves_and_the_report_does_not_need_them(campaign):
    ids = [row["id"] for row in campaign["rows"]]
    deltas = ["%s-delta" % arm for arm in ARMS]
    assert set(deltas) <= set(ids)
    for row in campaign["rows"]:
        assert not set(row["needs"]) & set(deltas), "%s needs a readout row" % row["id"]
    report = row_of(campaign, "report")
    assert sorted(report["needs"]) == sorted(["base-spider"] + ["%s-spider" % arm for arm in ARMS])
    assert ids[-4:] == deltas, "the readouts run last, after the report"


def test_the_probe_has_no_gate(campaign):
    """No pilot row, so no bar on this probe can refuse any other row: it measures, it does not gate."""
    assert [row["id"] for row in campaign["rows"] if row.get("pilot")] == []


def test_each_readout_compares_the_untrained_model_with_its_own_arm(campaign):
    for arm in ARMS:
        row = row_of(campaign, "%s-delta" % arm)
        command = " ".join(row["command"])
        assert "base-spider-a*" in command and "%s-spider-a*" % arm in command
        assert "--key correct" in command
        assert sorted(row["needs"]) == sorted(["base-spider", "%s-spider" % arm])
        bars = {bar["name"]: bar for bar in row["bars"]}
        assert bars["delta-recorded"]["key"] == "delta" and bars["delta-recorded"]["min"] == -100, "the +5 verdict is the report's, not a halting bar"
        assert bars["same-machine"]["key"] == "same_machine_flag" and bars["same-machine"]["min"] == 1


def test_a_readout_bar_passes_on_any_recorded_delta_so_a_missed_arm_halts_nothing(campaign, tmp_path):
    bar = dict(next(b for b in row_of(campaign, "hard20-delta")["bars"] if b["name"] == "delta-recorded"))
    for value, ok in ((7, True), (4, True), (-8, True)):
        path = tmp_path / ("delta%d.json" % value)
        path.write_text(json.dumps({"schema": "kit-delta.v1", "key": "correct", "a": 70.0,
                                    "b": 70.0 + value, "delta": float(value), "same_machine_flag": 1}))
        bar["source"] = str(path)
        assert runner.judge_bar(bar)["ok"] is ok


def test_the_plan_prints_every_row_and_executes_nothing(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    done = subprocess.run([sys.executable, str(KIT / "runner.py"), "plan", str(CAMPAIGN)],
                          env={**os.environ, "WORK": str(work), "SDPO_DIR": "/s", "MODEL_DIR": "/m"},
                          capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    printed = [line for line in done.stdout.split("\n") if line.startswith("[")]
    assert len(printed) == 17
    assert "campaign k3-dose-probe (17 rows)" in done.stdout
    assert list(work.iterdir()) == [], "planning must create nothing"


# ================================================================= (c) kit/subset.py drops the right rows
def _row(index: str) -> dict:
    return {"data_source": "spider_sql", "extra_info": {"index": index, "split": "train"},
            "prompt": [{"role": "user", "content": "question %s" % index}],
            "reward_model": {"ground_truth": json.dumps({"db": "d", "gold_sql": "SELECT 1"})}}


def _training_file(tmp_path: Path, ids: list, name: str = "train") -> Path:
    path = tmp_path / ("%s.jsonl" % name)
    path.write_text("".join(json.dumps(_row(i), sort_keys=True) + "\n" for i in ids), encoding="utf-8")
    return path


def _stuck_file(tmp_path: Path, wins: dict, *, attempts: int = 8, name: str = "stuck.json") -> Path:
    per_question = [{"index": index, "wins": won, "attempts": attempts} for index, won in wins.items()]
    rows = [_row(index) for index in wins]
    record = hints.stuck_record(rows, per_question, attempts=attempts, temperature=1.0,
                               model="/models/Qwen3-1.7B", seconds=1.0,
                               extra={"per_question": per_question})
    path = tmp_path / name
    path.write_text(json.dumps(record, indent=1, sort_keys=True), encoding="utf-8")
    return path


def test_subset_names_a_question_exactly_as_hints_py_does():
    """The ids in a stuck.json are hints.question_id's; a second rule here would match nothing."""
    row = _row("q7")
    assert subset.question_id(row) == hints.question_id(row) == "q7"
    with pytest.raises(subset.SubsetError, match="extra_info.index"):
        subset.question_id({"prompt": []})


def test_subset_reads_a_trainer_file_the_way_sequence_does(tmp_path):
    """The two readers are duplicated on purpose; they must still agree, row for row."""
    path = _training_file(tmp_path, ["q%d" % i for i in range(5)])
    assert subset.read_rows(path) == sequence.read_rows(path)
    assert subset.columns_of(subset.read_rows(path), path) == \
        sequence.columns_of(sequence.read_rows(path), path)


def test_it_drops_exactly_the_questions_won_every_time(tmp_path):
    ids = ["q%d" % i for i in range(40)]
    # 6 solved every time, 4 never, the rest in between
    wins = {index: (8 if i < 6 else 0 if i < 10 else 3) for i, index in enumerate(ids)}
    manifest = subset.build_subset(_training_file(tmp_path, ids), _stuck_file(tmp_path, wins),
                                   tmp_path / "out", allow_no_parquet=True)
    assert manifest["input_rows"] == 40 and manifest["dropped_always_solved"] == 6
    assert manifest["kept"] == 34 and manifest["never_solved"] == 4
    assert manifest["wins_histogram"][8] == 6 and manifest["wins_histogram"][0] == 4
    assert manifest["wins_histogram_kept"][8] == 0 and manifest["wins_histogram_kept"][0] == 4
    assert sum(manifest["wins_histogram"]) == 40 and sum(manifest["wins_histogram_kept"]) == 34
    written = [json.loads(line) for line in
               (tmp_path / "out" / "train.jsonl").read_text().split("\n") if line.strip()]
    assert [subset.question_id(row) for row in written] == ids[6:], "kept rows, in the input's order"
    assert manifest["outputs"]["jsonl_sha256"] == subset.sha256_file(tmp_path / "out" / "train.jsonl")


def test_a_recorded_question_the_file_does_not_hold_is_a_refusal(tmp_path):
    ids = ["q%d" % i for i in range(40)]
    wins = {index: 3 for index in ids}
    wins["not-in-the-file"] = 8
    with pytest.raises(subset.SubsetError, match="not in this training file"):
        subset.build_subset(_training_file(tmp_path, ids), _stuck_file(tmp_path, wins),
                            tmp_path / "out", allow_no_parquet=True)
    assert not (tmp_path / "out").exists(), "a refusal must leave nothing behind"


def test_a_subset_too_short_to_fill_one_step_is_a_refusal(tmp_path):
    ids = ["q%d" % i for i in range(40)]
    wins = {index: (8 if i >= 31 else 3) for i, index in enumerate(ids)}     # 31 rows would be left
    assert subset.BATCH == 32
    with pytest.raises(subset.SubsetError, match="one step needs 32"):
        subset.build_subset(_training_file(tmp_path, ids), _stuck_file(tmp_path, wins),
                            tmp_path / "out", allow_no_parquet=True)
    assert not (tmp_path / "out").exists()


def test_it_never_overwrites_and_refuses_a_stuck_file_of_another_kind(tmp_path):
    ids = ["q%d" % i for i in range(40)]
    wins = {index: 3 for index in ids}
    rows, stuck = _training_file(tmp_path, ids), _stuck_file(tmp_path, wins)
    subset.build_subset(rows, stuck, tmp_path / "out", allow_no_parquet=True)
    with pytest.raises(subset.SubsetError, match="refusing to overwrite"):
        subset.build_subset(rows, stuck, tmp_path / "out", allow_no_parquet=True)
    other = tmp_path / "filter.json"
    other.write_text(json.dumps({"schema": "kit-hints.v1", "stage": "filter", "kept": 3}))
    with pytest.raises(subset.SubsetError, match="not a stuck.json"):
        subset.build_subset(rows, other, tmp_path / "out2", allow_no_parquet=True)
    assert not (tmp_path / "out2").exists()


def test_two_rows_with_one_question_id_are_refused(tmp_path):
    ids = ["q%d" % i for i in range(40)] + ["q0"]
    wins = {index: 3 for index in ids}
    with pytest.raises(subset.SubsetError, match="two training rows carry"):
        subset.build_subset(_training_file(tmp_path, ids), _stuck_file(tmp_path, wins),
                            tmp_path / "out", allow_no_parquet=True)


def test_the_subset_cli_writes_a_manifest_a_campaign_bar_can_read(tmp_path):
    ids = ["q%d" % i for i in range(40)]
    wins = {index: (8 if i < 5 else 1) for i, index in enumerate(ids)}
    code = subset.main(["drop-solved", "--from", str(_training_file(tmp_path, ids)),
                        "--stuck", str(_stuck_file(tmp_path, wins)),
                        "--out", str(tmp_path / "out"), "--allow-no-parquet"])
    assert code == 0
    manifest = json.loads((tmp_path / "out" / "subset.manifest.json").read_text())
    assert manifest["schema"] == "kit-subset.v1"
    assert isinstance(manifest["kept"], int) and manifest["kept"] == 35


# ================================================================== the report's fixture tree
PER_ITEM_IDS = ["item-%03d" % i for i in range(100)]


def _verdicts(right: list) -> dict:
    return {index: (1 if index in set(right) else 0) for index in PER_ITEM_IDS}


def _score(path: Path, *, correct_ids: list, truncated: int = 0, machine: str | None = "m1",
           tokens: int | None = None) -> None:
    path.mkdir(parents=True)
    per_item = _verdicts(correct_ids)
    payload = {"schema": "kit-bed-score.v1", "bed": "spider", "split": "heldout", "n": 100,
               "correct": len(correct_ids), "total_correct": len(correct_ids),
               "accuracy": round(len(correct_ids) / 100, 6), "incorrect_format": 0,
               "truncated_at_max_tokens": truncated, "per_item": per_item}
    if machine is not None:
        payload["machine"] = {"id": machine, "deterministic": True}
    if tokens is not None:
        payload["output_tokens_total"] = tokens
    (path / "bed-score.json").write_text(json.dumps(payload), encoding="utf-8")


def _metrics(path: Path, steps: int, *, reward=lambda step: 0.63, grad=lambda step: 0.5) -> None:
    lines = []
    for step in range(1, steps + 1):
        lines.append(json.dumps({"step": step, "data": {
            "critic/score/mean": reward(step), "response_length/mean": 300.0 + step,
            "actor/entropy": 0.8, "actor/grad_norm": grad(step)}}))
    (path / "metrics.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run(runs: Path, arm: str, *, steps: int, lr: str, train_file: str, attempt: int = 1,
         metrics: bool = True, seed: int = 0, summary: dict | None = None, **kwargs) -> Path:
    path = runs / ("%s-seed%d-a%d" % (arm, seed, attempt))
    path.mkdir(parents=True)
    (path / "train-summary.json").write_text(json.dumps({
        "schema": "kit-grpo-run.v1", "name": path.name, "steps": steps, "save_freq": steps,
        "returncode": 0, "merged": 1, "kl": 0, "lr": lr, "n_gpus": 8, "seconds": 600,
        "train_file": train_file, **(summary or {})}), encoding="utf-8")
    if metrics:
        _metrics(path, steps, **kwargs)
    return path


#: The partner's own stage-A numbers: 70 untrained, 21 right -> wrong and 13 wrong -> right, so 62.
UNTRAINED_RIGHT = PER_ITEM_IDS[:70]
REF20_RIGHT = PER_ITEM_IDS[21:70] + PER_ITEM_IDS[70:83]          # 49 kept + 13 gained = 62
PLUS_SEVEN = PER_ITEM_IDS[:70] + PER_ITEM_IDS[70:77]             # 77: seven wrong -> right, none lost


def _tree(tmp_path: Path, *, tokens=None, metrics=True, arms=ARMS) -> tuple:
    """(root, runs): the tree the campaign writes, with hand-computed churn on every arm."""
    root, runs = tmp_path / "k3dose", tmp_path / "runs"
    after = {"ref20": REF20_RIGHT, "steps60": PLUS_SEVEN,
             "lr3x": PER_ITEM_IDS[:64], "hard20": PER_ITEM_IDS[:78]}
    token_of = {"base": 4000, "ref20": 5000, "steps60": 6000, "lr3x": 4400, "hard20": 9000}
    _score(root / "eval" / "base-spider-a1", correct_ids=UNTRAINED_RIGHT, truncated=0,
           tokens=token_of["base"] if tokens else None)
    files = {"ref20": str(tmp_path / "data" / "spider" / "train.parquet"),
             "steps60": str(tmp_path / "k3dose" / "data" / "spider1920-a1" / "train.parquet"),
             "lr3x": str(tmp_path / "data" / "spider" / "train.parquet"),
             "hard20": str(tmp_path / "k3dose" / "data" / "hard640-a1" / "train.parquet")}
    for arm in arms:
        _score(root / "eval" / ("%s-spider-a1" % arm), correct_ids=after[arm],
               truncated=1 if arm == "ref20" else 0, tokens=token_of[arm] if tokens else None)
        _run(runs, arm, steps=60 if arm == "steps60" else 20,
             lr="3e-5" if arm == "lr3x" else "1e-5", train_file=files[arm], metrics=metrics)
    # the two manifests the report reads for "rows" and "passes", and the bed's own jsonl
    (tmp_path / "data" / "spider").mkdir(parents=True)
    (tmp_path / "data" / "spider" / "train.jsonl").write_text(
        "".join(json.dumps(_row("q%d" % i)) + "\n" for i in range(640)), encoding="utf-8")
    for name, rows, distinct in (("spider1920-a1", 1920, 640), ("hard640-a1", 640, 340)):
        folder = root / "data" / name
        folder.mkdir(parents=True)
        (folder / "pool.manifest.json").write_text(json.dumps({
            "schema": "kit-pool.v1", "rows_total": rows, "inputs_pooled": 1, "seed": 0,
            "distinct_per_input": [distinct], "rows_per_input": [rows]}), encoding="utf-8")
    return root, runs


# ==================================================== (d) the churn, the McNemar p and the +5 bar
def test_the_report_recomputes_the_partners_own_churn_and_its_exact_mcnemar_p(tmp_path):
    root, runs = _tree(tmp_path)
    report = report_tool.build(root, runs)
    assert report["base"]["correct"] == 70 and report["base"]["n"] == 100
    ref = report["arms"]["ref20"]
    assert (ref["untrained"], ref["after"], ref["delta"]) == (70, 62, -8)
    assert (ref["right_to_wrong"], ref["wrong_to_right"], ref["unchanged"]) == (21, 13, 66)
    assert ref["items_compared"] == 100 and ref["changed"] == 34
    assert round(ref["mcnemar_p"], 3) == 0.229
    assert ref["clears_bar"] == 0 and "does not clear the +5 bar (-8)" in ref["verdict"]
    assert "policy moved: 34 of 100 answers changed" in ref["verdict"]


def test_an_arm_at_plus_seven_clears_the_bar_and_is_named_as_what_to_do_next(tmp_path):
    root, runs = _tree(tmp_path)
    report = report_tool.build(root, runs)
    arm = report["arms"]["steps60"]
    assert arm["delta"] == 7 and arm["clears_bar"] == 1
    assert (arm["right_to_wrong"], arm["wrong_to_right"], arm["unchanged"]) == (0, 7, 93)
    assert arm["mcnemar_p"] == pytest.approx(2 * 0.5 ** 7)
    assert "clears the +5 bar (+7)" in arm["verdict"]
    assert report["arms_clearing_the_bar"] == ["steps60", "hard20"]
    assert report["arms_reported"] == 4
    text = report_tool.render(report)
    assert "## What to do next" in text
    assert "`steps60`, `hard20`" in text and "steps60 +7" in text
    assert "stage A of K3 should be re-dosed" in text and "ONE seed" in text


def test_the_mcnemar_p_is_the_exact_two_sided_binomial():
    assert report_tool.mcnemar_exact(21, 13) == pytest.approx(0.229481, abs=1e-6)
    assert report_tool.mcnemar_exact(13, 21) == pytest.approx(0.229481, abs=1e-6)
    assert report_tool.mcnemar_exact(0, 0) == 1.0
    assert report_tool.mcnemar_exact(5, 5) == 1.0
    assert report_tool.mcnemar_exact(10, 0) == pytest.approx(2 * 0.5 ** 10)


def test_the_report_reads_the_dose_the_file_and_the_passes_from_the_run_itself(tmp_path):
    root, runs = _tree(tmp_path)
    report = report_tool.build(root, runs)
    assert report["warmup_steps"] == 10 and report["bar"] == 5
    assert report["arms"]["ref20"]["steps"] == 20 and report["arms"]["ref20"]["lr"] == "1e-5"
    assert report["arms"]["lr3x"]["lr"] == "3e-5"
    assert report["arms"]["steps60"]["steps"] == 60
    assert report["arms"]["ref20"]["training_file"]["rows"] == 640
    assert report["arms"]["ref20"]["training_file"]["passes"] == 1.0
    assert report["arms"]["steps60"]["training_file"] == {"rows": 1920, "distinct": 640, "passes": 3.0,
                                                          "source": "pool.manifest.json"}
    assert report["arms"]["hard20"]["training_file"]["passes"] == pytest.approx(1.88, abs=1e-9)
    assert report["arms"]["ref20"]["truncated_before"] == 0
    assert report["arms"]["ref20"]["truncated_after"] == 1
    training = report["arms"]["steps60"]["training"]
    assert training["steps_logged"] == 60 and training["missing"] is None
    assert training["reward"]["first_quarter"] == pytest.approx(0.63)
    assert training["reward"]["last_quarter"] == pytest.approx(0.63)
    assert training["grad_norm"]["mean"] == pytest.approx(0.5)
    assert training["entropy"]["mean"] == pytest.approx(0.8)
    assert training["response_tokens"]["mean"] == pytest.approx(330.5)


def test_the_first_and_last_quarter_of_a_climbing_signal_differ(tmp_path):
    root, runs = _tree(tmp_path, arms=("ref20",))
    (runs / "ref20-seed0-a1" / "metrics.jsonl").unlink()
    _metrics(runs / "ref20-seed0-a1", 20, reward=lambda step: 0.5 + 0.01 * step,
             grad=lambda step: float(step))
    training = report_tool.build(root, runs)["arms"]["ref20"]["training"]
    assert training["reward"]["first_quarter"] == pytest.approx(0.5 + 0.01 * 3)      # steps 1..5
    assert training["reward"]["last_quarter"] == pytest.approx(0.5 + 0.01 * 18)      # steps 16..20
    assert training["grad_norm"]["first_quarter"] == pytest.approx(3.0)
    assert training["grad_norm"]["last_quarter"] == pytest.approx(18.0)
    assert training["grad_norm"]["mean"] == pytest.approx(10.5)


def test_the_report_writes_both_files_and_never_overwrites(tmp_path):
    root, runs = _tree(tmp_path)
    out = tmp_path / "report" / "a1"
    assert report_tool.main(["--root", str(root), "--runs", str(runs), "--out", str(out)]) == 0
    written = json.loads((out / "report.json").read_text())
    assert written["arms_reported"] == 4 and written["schema"] == "kit-k3-dose-report.v1"
    assert isinstance(written["arms_reported"], int) and not isinstance(written["arms_reported"], bool)
    text = (out / "report.md").read_text()
    assert "# K3 stage A: which dose clears the +5 bar?" in text
    assert "| 70 | 62 | -8 | 21 | 13 | 66 | 0.229 |" in text
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        report_tool.main(["--root", str(root), "--runs", str(runs), "--out", str(out)])


def test_the_report_reads_the_highest_attempt_of_a_scoring_and_of_a_run(tmp_path):
    root, runs = _tree(tmp_path)
    _score(root / "eval" / "ref20-spider-a2", correct_ids=PER_ITEM_IDS[:80])
    _run(runs, "ref20", steps=20, lr="1e-5", train_file="/late.parquet", attempt=2)
    report = report_tool.build(root, runs)
    assert report["arms"]["ref20"]["after"] == 80 and report["arms"]["ref20"]["delta"] == 10
    assert report["arms"]["ref20"]["run"] == "ref20-seed0-a2"


def test_an_empty_tree_is_a_refusal_not_an_empty_report(tmp_path):
    (tmp_path / "k3dose").mkdir()
    with pytest.raises(report_tool.K3DoseReportError, match="no untrained scoring"):
        report_tool.build(tmp_path / "k3dose", tmp_path / "runs")


# ================================================ (e) a missing file is named, never invented
def test_a_run_with_no_metrics_still_reports_and_flags_the_missing_file(tmp_path):
    root, runs = _tree(tmp_path, metrics=False)
    report = report_tool.build(root, runs)
    assert report["arms_reported"] == 4, "the held-out numbers do not depend on metrics.jsonl"
    assert report["arms"]["ref20"]["delta"] == -8
    training = report["arms"]["ref20"]["training"]
    assert training["steps_logged"] == 0
    assert "no metrics.jsonl" in training["missing"] and "FILE_LOG=1" in training["missing"]
    assert all(training[key]["mean"] is None for key in report_tool.TRAIN_KEYS)
    assert any("no metrics.jsonl" in item for item in report["missing"])
    text = report_tool.render(report)
    assert "## Missing" in text and "no metrics.jsonl" in text
    assert "| ref20 | 0 | - | - | - | - | - | - | - | - |" in text


def test_a_missing_arm_is_reported_as_missing_and_not_counted(tmp_path):
    root, runs = _tree(tmp_path, arms=("ref20", "steps60", "lr3x"))
    report = report_tool.build(root, runs)
    assert report["arms_reported"] == 3
    missing = report["arms"]["hard20"]
    assert missing["after"] is None and missing["delta"] is None and missing["clears_bar"] == 0
    assert "no held-out scoring" in missing["missing"][0]
    assert "no run directory" in missing["missing"][1]
    assert missing["verdict"] == "no held-out scoring, so this arm has no verdict"
    text = report_tool.render(report)
    assert "| hard20 |" in text and "## Missing" in text


# =============================================== (f) the density column, present and absent
def test_the_density_cells_appear_when_the_scorings_carry_token_counts(tmp_path):
    root, runs = _tree(tmp_path, tokens=True)
    report = report_tool.build(root, runs)
    assert report["density"]["measured"] == 1 and report["density"]["note"] is None
    ref = report["arms"]["ref20"]["density"]
    # 4000 tokens over 70 correct untrained; 5000 over 62 after
    assert ref["untrained"] == pytest.approx(4000 / 70)
    assert ref["trained"] == pytest.approx(5000 / 62)
    assert ref["ratio"] == pytest.approx((5000 / 62) / (4000 / 70))
    assert ref["verdict"] == "within bar" and ref["bar"] == 1.5
    hard = report["arms"]["hard20"]["density"]
    assert hard["ratio"] == pytest.approx((9000 / 78) / (4000 / 70)) and hard["verdict"] == "over bar"
    text = report_tool.render(report)
    assert "## What a correct answer costs (tokens per correct answer)" in text
    assert "| ref20 | 57.1 | 80.6 | 1.41 | within bar |" in text
    assert "| hard20 | 57.1 | 115.4 | 2.02 | over bar |" in text


def test_the_density_cells_are_dashes_when_no_scoring_carries_them(tmp_path):
    root, runs = _tree(tmp_path)
    report = report_tool.build(root, runs)
    assert report["density"]["measured"] == 0
    assert "carried token counts" in report["density"]["note"]
    for arm in ARMS:
        assert report["arms"][arm]["density"] == {"untrained": None, "trained": None, "ratio": None,
                                                  "bar": 1.5, "verdict": "unknown"}
    text = report_tool.render(report)
    assert "| ref20 | - | - | - | unknown |" in text
    assert "carried token counts" in text
    assert "| 70 | 62 | -8 | 21 | 13 | 66 | 0.229 |" in text, "the held-out table is unchanged"


# ================================================== (g) probe 2: the length-budgeted reward
CAMPAIGN2 = KIT / "campaigns" / "k3-dose-2.yaml"
ARMS2 = ("budget20", "budget40", "budget60", "ref20")
RUNS_PER_ARM = 3
RUNS2 = tuple("%s-r%d" % (arm, r) for arm in ARMS2 for r in range(1, RUNS_PER_ARM + 1))
#: The keys a probe-2 run may differ from probe 1's `ref20` in. NGPU, KL, WORK and VAL_FILE may not.
ALLOWED_TO_DIFFER_2 = {"NAME", "TRAIN_FILE", "STEPS", "SAVE_FREQ", "SEED", "LENGTH_BUDGET", "LR", "FILE_LOG"}


@pytest.fixture()
def campaign2():
    os.environ.setdefault("WORK", "/tmp/unused-k3-dose")
    return runner.load_campaign(CAMPAIGN2)


def runs_of(arm: str) -> list:
    return ["%s-r%d" % (arm, r) for r in range(1, RUNS_PER_ARM + 1)]


def test_probe_2_parses_and_has_no_gate(campaign2):
    ids = [row["id"] for row in campaign2["rows"]]
    assert campaign2["name"] == "k3-dose-2"
    assert len(ids) == 3 + len(ARMS2) * RUNS_PER_ARM * 3 + 1 == 40
    assert len(set(ids)) == len(ids)
    assert ids[:3] == ["base-spider", "pool1280", "pool1920"] and row_of(campaign2, "base-spider")["prepare"]
    assert [row["id"] for row in campaign2["rows"] if row.get("pilot")] == []
    for row in ("stuck", "subset", "ref20-s1"):
        assert row not in ids, "probe 2 needs no stuck set and no subset, and ref20-s1 is now ref20"


def test_probe_2_has_twelve_training_runs_in_four_arms_of_three(campaign2):
    training = [row for row in campaign2["rows"] if "run_grpo.sh" in " ".join(row["command"])]
    assert [row["id"] for row in training] == list(RUNS2)
    for arm in ARMS2:
        rows = [row_of(campaign2, run) for run in runs_of(arm)]
        assert [row["env"]["SEED"] for row in rows] == ["1", "2", "3"], arm
        for r, row in enumerate(rows, start=1):
            assert row["env"]["NAME"] == "%s-r%d-seed%d-a{attempt}" % (arm, r, r), row["id"]
        same = [{k: v for k, v in row["env"].items() if k not in ("NAME", "SEED")} for row in rows]
        assert same[0] == same[1] == same[2], "%s: its runs differ in more than the label" % arm
        assert len({row["description"] for row in rows}) == 1 and rows[0]["description"], arm
        assert rows[0]["command"] == rows[1]["command"] == rows[2]["command"], arm
        assert rows[0]["needs"] == rows[1]["needs"] == rows[2]["needs"], arm
        assert [bar for bar in rows[0]["bars"]] == [bar for bar in rows[2]["bars"]], arm


def test_probe_2s_runs_are_probe_1s_ref20_but_for_the_keys_a_budgeted_dose_may_change(campaign,
                                                                                    campaign2):
    ref20 = row_of(campaign, "ref20")["env"]
    theirs = {key: value for key, value in ref20.items() if key not in ALLOWED_TO_DIFFER_2}
    for run in RUNS2:
        row = row_of(campaign2, run)
        env = row["env"]
        assert {key: value for key, value in env.items() if key not in ALLOWED_TO_DIFFER_2} == theirs, run
        assert set(env) - set(ref20) <= {"LENGTH_BUDGET"}, run
        assert env["NGPU"] == "8" and env["KL"] == "0" and env["LR"] == "1e-5", run
        assert env["FILE_LOG"] == "1" and env["SAVE_FREQ"] == env["STEPS"], run
        assert "run_grpo.sh" in " ".join(row["command"]), run
        assert any(bar["key"] == "steps" and bar["min"] == int(env["STEPS"]) for bar in row["bars"]), run


def test_the_budget_runs_carry_800_and_ref20_carries_none(campaign2):
    steps = {"budget20": "20", "budget40": "40", "budget60": "60", "ref20": "20"}
    for arm, count in steps.items():
        for run in runs_of(arm):
            assert row_of(campaign2, run)["env"]["STEPS"] == count, run
    for arm in ("budget20", "budget40", "budget60"):
        for run in runs_of(arm):
            row = row_of(campaign2, run)
            assert row["env"]["LENGTH_BUDGET"] == "800", run
            bar = next(b for b in row["bars"] if b["name"] == "budget-applied")
            assert (bar["key"], bar["min"], bar["max"]) == ("length_budget_chars", 800, 800), run
    for run in runs_of("ref20"):
        ref = row_of(campaign2, run)
        assert "LENGTH_BUDGET" not in ref["env"] and "LENGTH_BUDGET" not in " ".join(ref["command"])
        assert not any(bar["key"] == "length_budget_chars" for bar in ref["bars"])


def test_probe_2_trains_on_the_file_each_dose_needs(campaign2):
    for arm in ("budget20", "ref20"):
        for run in runs_of(arm):
            assert "TRAIN_FILE=\"{work}/data/spider/train.parquet\"" in " ".join(row_of(campaign2, run)["command"])
    for arm, pool in (("budget40", "1280"), ("budget60", "1920")):
        for run in runs_of(arm):
            assert "k3dose2/data/spider%s-a*" % pool in " ".join(row_of(campaign2, run)["command"])
            assert "pool%s" % pool in row_of(campaign2, run)["needs"]
    for pool, rows in (("pool1280", 1280), ("pool1920", 1920)):
        command = " ".join(row_of(campaign2, pool)["command"])
        assert "sequence.py\" pool" in command and "--rows %d --seed 0" % rows in command


def test_every_run_has_its_own_scoring_and_delta_row(campaign2):
    for run in RUNS2:
        env = row_of(campaign2, run)["env"]
        scoring = row_of(campaign2, "%s-spider" % run)
        assert scoring["needs"] == [run] and scoring["env"]["CUDA_VISIBLE_DEVICES"] == "0"
        assert scoring["env"]["OUT"] == "{work}/k3dose2/eval/%s-spider-a{attempt}" % run
        assert "runs/%s-seed%s-a*/hf-step%s" % (run, env["SEED"], env["STEPS"]) in " ".join(scoring["command"])
        delta = row_of(campaign2, "%s-delta" % run)
        assert sorted(delta["needs"]) == sorted(["base-spider", "%s-spider" % run])
        command = " ".join(delta["command"])
        assert "k3dose2/eval/base-spider-a*" in command and "k3dose2/eval/%s-spider-a*" % run in command
        assert delta["env"]["OUT"] == "{work}/k3dose2/deltas/%s-a{attempt}.json" % run
        bars = {bar["name"]: bar for bar in delta["bars"]}
        assert bars["delta-recorded"]["key"] == "delta" and bars["delta-recorded"]["min"] == -100
        assert bars["same-machine"]["key"] == "same_machine_flag" and bars["same-machine"]["min"] == 1


def test_probe_2_writes_only_under_k3dose2(campaign2):
    for row in campaign2["rows"]:
        for key, value in row["env"].items():
            if key == "WORK" or not value.startswith("{work}") or "/models/" in value or "/data/spider" in value:
                continue
            assert value.startswith("{work}/k3dose2/"), (row["id"], value)
        command = " ".join(row["command"])
        assert "/k3dose/" not in command, "%s reads probe 1's tree" % row["id"]


def test_probe_2s_delta_rows_are_leaves_and_the_report_reads_the_campaign(campaign2):
    ids = [row["id"] for row in campaign2["rows"]]
    deltas = ["%s-delta" % run for run in RUNS2]
    assert ids[-12:] == deltas, "the readouts run last, after the report"
    for row in campaign2["rows"]:
        assert not set(row["needs"]) & set(deltas), "%s needs a readout row" % row["id"]
    report = row_of(campaign2, "report")
    assert sorted(report["needs"]) == sorted(["base-spider"] + ["%s-spider" % run for run in RUNS2])
    assert "--campaign \"{kit}/campaigns/k3-dose-2.yaml\"" in " ".join(report["command"])
    assert next(bar for bar in report["bars"] if bar["key"] == "arms_reported")["min"] == 4


def test_probe_2s_bars_read_numbers_its_tools_write(campaign2):
    written = {"n", "rows_total", "returncode", "merged", "steps", "length_budget_chars", "delta",
               "same_machine_flag", "arms_reported"}
    for row in campaign2["rows"]:
        assert row["bars"], row["id"]
        for bar in runner.resolve(row, campaign2, 1)["bars"]:
            assert bar["key"] in written and bar["source"].endswith(".json"), (row["id"], bar)
    assert "length_budget_chars" in (KIT / "run_grpo.sh").read_text()


def test_probe_2s_plan_prints_every_row_and_executes_nothing(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    done = subprocess.run([sys.executable, str(KIT / "runner.py"), "plan", str(CAMPAIGN2)],
                          env={**os.environ, "WORK": str(work), "SDPO_DIR": "/s", "MODEL_DIR": "/m"},
                          capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert "campaign k3-dose-2 (40 rows)" in done.stdout
    assert len([line for line in done.stdout.split("\n") if line.startswith("[")]) == 40
    assert list(work.iterdir()) == []


def test_the_arm_list_is_read_from_either_campaign_file():
    probe2 = report_tool.arms_from_campaign(CAMPAIGN2)
    assert probe2["name"] == "k3-dose-2"
    assert [a["arm"] for a in probe2["arms"]] == list(RUNS2)
    assert [a["group"] for a in probe2["arms"]] == [arm for arm in ARMS2 for _ in range(RUNS_PER_ARM)]
    by_run = {a["arm"]: a for a in probe2["arms"]}
    assert by_run["ref20-r2"]["run"] == "ref20-r2-seed2" and by_run["ref20-r2"]["seed"] == 2
    assert by_run["ref20-r2"]["scoring"] == "ref20-r2-spider"
    assert by_run["budget40-r3"]["run"] == "budget40-r3-seed3" and "800-character" in by_run["budget40-r3"]["what"]
    probe1 = report_tool.arms_from_campaign(CAMPAIGN)
    assert probe1["arms"] == report_tool.default_arms(), "probe 1's file must name exactly its four arms"


def test_a_campaign_with_no_training_row_is_a_refusal(tmp_path):
    path = tmp_path / "c.json"
    path.write_text(json.dumps({"schema": "kit-campaign.v1", "name": "x",
                                "rows": [{"id": "a", "command": ["true"]}]}))
    with pytest.raises(report_tool.K3DoseReportError, match="no row that runs kit/run_grpo.sh"):
        report_tool.arms_from_campaign(path)


#: Per run: (delta, tokens per correct answer against the untrained model's). The untrained model
#: scores 70 at 7,000 tokens, 100 per correct answer, so a run at +d with ratio x spent 100x(70+d)x.
PROBE2 = {
    "budget20": ((7, 1.1), (6, 1.1), (2, 1.1)),     # mean +5, 2 of 3 at +5, 1.1x: CLEARS
    "budget40": ((9, 2.0), (9, 2.0), (9, 2.0)),     # mean +9, 3 of 3, but 2.0x: does not
    "budget60": ((5, 1.0), (5, 1.0), (5, 1.0)),     # mean +5, 3 of 3, 1.0x: CLEARS
    "ref20": ((8, 1.0), (1, 1.0), (1, 1.0)),        # mean +3.3, 1 of 3: does not
}


def _tree2(tmp_path: Path, *, tokens=True, skip=()) -> tuple:
    """Probe 2's tree, three runs per arm, each run +d with no answer lost (PROBE2 above)."""
    root, runs = tmp_path / "k3dose2", tmp_path / "runs"
    dose = {"budget20": (20, 800, "data/spider/train.parquet"),
            "budget40": (40, 800, "k3dose2/data/spider1280-a1/train.parquet"),
            "budget60": (60, 800, "k3dose2/data/spider1920-a1/train.parquet"),
            "ref20": (20, None, "data/spider/train.parquet")}
    _score(root / "eval" / "base-spider-a1", correct_ids=UNTRAINED_RIGHT, tokens=7000 if tokens else None)
    for arm in ARMS2:
        steps, budget, train_file = dose[arm]
        for r, (delta, ratio) in enumerate(PROBE2[arm], start=1):
            run = "%s-r%d" % (arm, r)
            if run in skip:
                continue
            correct = 70 + delta
            _score(root / "eval" / ("%s-spider-a1" % run), correct_ids=PER_ITEM_IDS[:correct],
                   tokens=round(100 * correct * ratio) if tokens else None)
            _run(runs, run, steps=steps, lr="1e-5", seed=r, train_file=str(tmp_path / train_file),
                 summary={"length_budget_chars": budget})
    (tmp_path / "data" / "spider").mkdir(parents=True)
    (tmp_path / "data" / "spider" / "train.jsonl").write_text(
        "".join(json.dumps(_row("q%d" % i)) + "\n" for i in range(640)), encoding="utf-8")
    for name, rows in (("spider1280-a1", 1280), ("spider1920-a1", 1920)):
        folder = root / "data" / name
        folder.mkdir(parents=True)
        (folder / "pool.manifest.json").write_text(json.dumps({
            "schema": "kit-pool.v1", "rows_total": rows, "inputs_pooled": 1, "seed": 0,
            "distinct_per_input": [640], "rows_per_input": [rows]}), encoding="utf-8")
    return root, runs


def test_the_report_groups_runs_into_arms_and_applies_the_arm_rule(tmp_path):
    root, runs = _tree2(tmp_path)
    report = report_tool.build(root, runs, report_tool.arms_from_campaign(CAMPAIGN2))
    assert report["arm_order"] == list(RUNS2) and report["group_order"] == list(ARMS2)
    assert report["arms_reported"] == 4 and report["runs_reported"] == 12
    assert report["seeds"] == [1, 2, 3] and report["campaign"]["name"] == "k3-dose-2"
    groups = report["groups"]
    assert {name: group["runs"] for name, group in groups.items()} == {arm: runs_of(arm) for arm in ARMS2}
    assert {name: group["deltas"] for name, group in groups.items()} == {
        arm: [delta for delta, _ in PROBE2[arm]] for arm in ARMS2}
    assert groups["budget20"]["mean_delta"] == pytest.approx(5.0)
    assert groups["ref20"]["mean_delta"] == pytest.approx(10 / 3, abs=1e-4)
    assert [groups[a]["runs_at_bar"] for a in ARMS2] == [2, 3, 3, 1]
    assert groups["budget20"]["mean_density_ratio"] == pytest.approx(1.1, abs=1e-3)
    assert groups["budget40"]["mean_density_ratio"] == pytest.approx(2.0, abs=1e-3)
    assert [groups[a]["decision"] for a in ARMS2] == ["clears", "does not", "clears", "does not"]
    assert report["arms_clearing_the_decision"] == ["budget20", "budget60"], "ties keep the file's order"
    runs_ = report["arms"]
    assert [runs_[r]["decision"] for r in runs_of("budget20")] == ["clears", "clears", "does not"]
    assert runs_["ref20-r2"]["length_budget_chars"] is None and runs_["budget60-r3"]["length_budget_chars"] == 800
    assert runs_["ref20-r2"]["seed"] == 2 and runs_["ref20-r2"]["run"] == "ref20-r2-seed2-a1"
    assert runs_["budget40-r1"]["training_file"]["passes"] == 2.0
    assert runs_["budget40-r1"]["mcnemar_p"] == pytest.approx(2 * 0.5 ** 9, abs=1e-6)
    assert (runs_["ref20-r1"]["right_to_wrong"], runs_["ref20-r1"]["wrong_to_right"]) == (0, 8)


def test_the_report_renders_a_bold_mean_row_per_arm_and_names_the_clearing_arms(tmp_path):
    root, runs = _tree2(tmp_path)
    out = tmp_path / "report" / "a1"
    assert report_tool.main(["--root", str(root), "--runs", str(runs), "--campaign", str(CAMPAIGN2),
                             "--out", str(out)]) == 0
    text = (out / "report.md").read_text()
    assert "# K3 stage A, `k3-dose-2`: which dose clears the +5 bar at a cost within 1.5x?" in text
    assert "4 arms, 12 runs" in text and "at least 2 of its runs are individually at +5" in text
    assert ("| arm | what it changes | rows | passes | steps | lr | seed | length budget | warm-up | "
            "untrained | after |") in text
    assert "| budget40-r2 | two passes (1,280 rows, 40 steps) with an 800-character length budget | 1280 | 2.00 | 40 | 1e-5 | 2 | 800 chars | 10 | 70 | 79 | +9 |" in text
    assert "| 640 | 1.00 | 20 | 1e-5 | 3 | none | 10 | 70 | 71 | +1 |" in text
    assert ("| **budget20: mean of 3 runs** | the reference dose (640 rows, 20 steps) with an 800-character "
            "length budget |  |  |  |  |  |  |  | 70 | **75.0** | **+5.0** |  |  |  |  |  |  | **clears**: "
            "runs +7, +6, +2; 2 of 3 at +5 or more; mean tokens per correct 1.10x the untrained model's |") in text
    assert ("**budget40: mean of 3 runs**" in text
            and "| **does not**: runs +9, +9, +9; 3 of 3 at +5 or more; mean tokens per correct 2.00x" in text)
    assert "| **clears**: runs +5, +5, +5; 3 of 3 at +5 or more; mean tokens per correct 1.00x" in text
    assert "| 70 | **73.3** | **+3.3** |" in text
    assert "| **does not**: runs +8, +1, +1; 1 of 3 at +5 or more;" in text
    lines = text.split("\n")
    mean_row = next(i for i, line in enumerate(lines) if line.startswith("| **budget20: mean"))
    assert [lines[mean_row - k].split(" | ")[0] for k in (3, 2, 1)] == ["| budget20-r1", "| budget20-r2", "| budget20-r3"]
    assert "| budget20-r1 | 800 chars | 20 |" in text and "| ref20-r3 | none | 20 |" in text
    assert "| budget40-r1 | 100.0 | 200.0 | 2.00 | over bar |" in text
    assert "`budget20`, `budget60` clear the arm rule" in text and "budget20 mean +5.0, budget60 mean +5.0" in text
    assert "`budget20`, `budget60` share the largest mean gain, so this probe does not choose" in text
    assert "has the largest mean gain" not in text
    assert text.endswith("4 of 4 arms reported (12 of 12 runs), seeds 1, 2, 3 (labels only), warm-up 10 "
                         "steps on every arm.\n")
    written = json.loads((out / "report.json").read_text())
    assert written["arms_reported"] == 4 and written["groups"]["budget60"]["decision"] == "clears"


def test_with_no_arm_clearing_the_report_names_the_best_mean(tmp_path):
    root, runs = _tree2(tmp_path, tokens=False)
    report = report_tool.build(root, runs, report_tool.arms_from_campaign(CAMPAIGN2))
    groups = report["groups"]
    assert [groups[a]["decision"] for a in ARMS2] == ["density unknown"] * 3 + ["does not"]
    assert report["arms_clearing_the_decision"] == []
    text = report_tool.render(report)
    assert "No arm cleared the arm rule" in text
    assert "The best mean is `budget40` at +9.0 (runs +9, +9, +9; 3 of 3 at +5 or more" in text
    assert "`budget20`, `budget40`, `budget60` met the delta half of the rule but carried no token counts" in text


def test_an_arm_missing_a_run_is_incomplete_and_cannot_clear(tmp_path):
    root, runs = _tree2(tmp_path, skip=("budget60-r2",))
    report = report_tool.build(root, runs, report_tool.arms_from_campaign(CAMPAIGN2))
    group = report["groups"]["budget60"]
    assert group["runs_reported"] == 2 and group["decision"] == "incomplete"
    assert report["arms_reported"] == 3 and report["runs_reported"] == 11
    assert report["arms_clearing_the_decision"] == ["budget20"]
    assert any(item.startswith("budget60-r2: no held-out scoring") for item in report["missing"])
    text = report_tool.render(report)
    assert "`budget60` is missing a run's held-out scoring" in text
    assert "`budget20` clears the arm rule" in text and "`budget20` has the largest mean gain" in text
    assert "| **budget60: mean of 3 runs** |" in text and "**incomplete**: runs +5, +5; 2 of 3 at +5" in text


def test_without_a_campaign_the_probe_1_report_is_unchanged(tmp_path):
    """No seed column, no budget column, no decision: the probe-1 report exactly as it was."""
    root, runs = _tree(tmp_path, tokens=True)
    report = report_tool.build(root, runs)
    for key in ("campaign", "arm_order", "arms_clearing_the_decision", "groups", "group_order", "runs_reported"):
        assert key not in report
    assert all("length_budget_chars" not in arm and "decision" not in arm and "group" not in arm
               for arm in report["arms"].values())
    text = report_tool.render(report)
    assert text.startswith("# K3 stage A: which dose clears the +5 bar?\n")
    assert ("| arm | what it changes | rows | passes | steps | lr | warm-up | untrained | after | delta | "
            "right -> wrong | wrong -> right | unchanged | McNemar p | truncated before | truncated after "
            "| verdict |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n") in text
    assert ("| arm | steps logged | reward first quarter | reward last quarter | reward mean | grad norm "
            "first quarter | grad norm last quarter | grad norm mean | entropy mean | response tokens mean |"
            "\n|---|---|---|---|---|---|---|---|---|---|\n") in text
    assert "length budget" not in text and "seeds 0" not in text and "mean of" not in text
    assert text.endswith("4 of 4 arms reported, seed 0, warm-up 10 steps on every arm.\n")


#: sha256 of probe 1's report.md on `_tree(tokens=True)` with the clock and the paths pinned, as
#: kit/k3_dose_report.py rendered it before probe 2's runs were grouped into arms.
PROBE1_RENDER_SHA256 = "568681030cc4769f6db27e15083b076c6074da902fffee3e357a3dba5397a292"


def test_probe_1s_render_is_byte_for_byte_what_it_was(tmp_path):
    import hashlib
    root, runs = _tree(tmp_path, tokens=True)
    report = report_tool.build(root, runs)
    report["generated_at"] = "2026-09-27T00:00:00Z"
    text = report_tool.render(report).replace(str(tmp_path.resolve()), "<tmp>")
    assert hashlib.sha256(text.encode("utf-8")).hexdigest() == PROBE1_RENDER_SHA256


def test_probe_1s_tree_reads_the_same_with_or_without_its_campaign(tmp_path):
    root, runs = _tree(tmp_path, tokens=True)
    plain = report_tool.build(root, runs)
    via = report_tool.build(root, runs, report_tool.arms_from_campaign(CAMPAIGN))
    for arm in ARMS:
        extra = {"length_budget_chars", "decision", "group"}
        assert {k: v for k, v in via["arms"][arm].items() if k not in extra} == plain["arms"][arm], arm
