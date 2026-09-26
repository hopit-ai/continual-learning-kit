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
         metrics: bool = True, **kwargs) -> Path:
    path = runs / ("%s-seed0-a%d" % (arm, attempt))
    path.mkdir(parents=True)
    (path / "train-summary.json").write_text(json.dumps({
        "schema": "kit-grpo-run.v1", "name": path.name, "steps": steps, "save_freq": steps,
        "returncode": 0, "merged": 1, "kl": 0, "lr": lr, "n_gpus": 8, "seconds": 600,
        "train_file": train_file}), encoding="utf-8")
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
