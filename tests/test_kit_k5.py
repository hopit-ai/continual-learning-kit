"""kit/campaigns/k5-sequence.yaml, its generator, kit/sequence.py and kit/k5_report.py.

K5 is the headline package: four jobs learned one after another, two orders, five seeds, every bed
and the forgetting panel scored after every stage. Five things are checked here, and each of them is
a way the package could be quietly wrong:

1. The committed campaign is exactly what `scripts/make_k5_campaign.py` builds, it covers every
   order, arm, seed and stage once, and each stage really starts from the previous stage's merged
   checkpoint. A hand-edited row would be a setting nobody generated and nobody reviewed.
2. The pilot gates everything, with K3's two bars.
3. EVERY DIRECTORY THE CAMPAIGN WRITES IS MATCHED BY THE TOOLS THAT READ IT -- built from the
   campaign file itself, not from a list copied into the test. That is the K4a lesson (receipt 225):
   a report whose folder pattern did not match the campaign's folder names reported nothing at all,
   silently.
4. `kit/sequence.py` pools deterministically, refuses a missing cell instead of writing a manifest
   with a hole in it, and its estimate is the campaign's own arithmetic.
5. `kit/k5_report.py` computes what it claims to and refuses to subtract a count made on one machine
   from a count made on another.

Everything builds its own fixtures FROM the campaign, through the real runner's own resolver, so the
fixtures cannot drift from the campaign either. Nothing here needs a GPU, a network or a model.
"""
from __future__ import annotations

import importlib.util
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
KIT = ROOT / "kit"
CAMPAIGN = KIT / "campaigns" / "k5-sequence.yaml"


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# The exported kit carries kit/ and these tests, but not scripts/: the generator-parity test skips there.
GENERATOR = ROOT / "scripts" / "make_k5_campaign.py"
generator = load("make_k5_campaign", GENERATOR) if GENERATOR.is_file() else None

needs_generator = pytest.mark.skipif(generator is None, reason="scripts/ is not part of the exported kit")
runner = load("kit_runner_for_k5", KIT / "runner.py")
sequence = load("kit_sequence", KIT / "sequence.py")
k5_report = load("kit_k5_report", KIT / "k5_report.py")
scorecard = load("kit_scorecard_for_k5", KIT / "scorecard.py")
plasticity = load("kit_plasticity_for_k5", KIT / "plasticity.py")
mix = load("kit_mix_for_k5", KIT / "mix.py")

ARMS = ("none", "rehearse10")
SEEDS = (0, 1, 2, 3, 4)
ORDERS = {"sqlfirst": ("spider", "code", "gsm8k", "finqa"),
          "mathsfirst": ("gsm8k", "finqa", "code", "spider")}
BEDS = ("code", "finqa", "gsm8k", "spider")
HELD_OUT = {"spider": 100, "code": 51, "gsm8k": 300, "finqa": 1147}


@pytest.fixture()
def campaign():
    os.environ.setdefault("WORK", "/tmp/unused-k5")
    return runner.load_campaign(CAMPAIGN)


def rows_by_id(campaign) -> dict:
    return {row["id"]: row for row in campaign["rows"]}


# ------------------------------------------------------------------------ 1. generated, not edited
def test_the_committed_campaign_is_what_its_generator_builds():
    if generator is None:
        pytest.skip("scripts/ is not part of the exported kit")
    assert CAMPAIGN.read_text() == generator.build(), "re-run scripts/make_k5_campaign.py"


def test_it_covers_every_order_arm_seed_and_stage_once(campaign):
    ids = [row["id"] for row in campaign["rows"]]
    assert len(ids) == len(set(ids))
    for order, jobs in ORDERS.items():
        for seed in SEEDS:
            shared = "%s-shared-seed%d-stage1" % (order, seed)
            assert shared in ids
            for arm in ARMS:
                for stage in range(2, len(jobs) + 1):
                    assert "%s-%s-seed%d-stage%d" % (order, arm, seed, stage) in ids
    assert ids[-1] == "report"


def test_every_point_scores_every_bed_the_panels_and_the_probe(campaign):
    ids = set(row["id"] for row in campaign["rows"])
    points = ["base"] + ["%s-shared-seed%d-stage1" % (order, seed) for order in ORDERS for seed in SEEDS]
    points += ["%s-%s-seed%d-stage%d" % (order, arm, seed, stage)
               for order, jobs in ORDERS.items() for arm in ARMS for seed in SEEDS
               for stage in range(2, len(jobs) + 1)]
    for point in points:
        for bed in BEDS:
            assert "%s-%s" % (point, bed) in ids, (point, bed)
        assert "%s-plasticity" % point in ids, point
        if point != "base":
            assert "%s-forget" % point in ids, point
    assert "base-forget-1" in ids and "base-forget-2" in ids      # the untrained panels, scored twice


def test_each_stage_starts_from_the_previous_stages_merged_checkpoint(campaign):
    rows = rows_by_id(campaign)
    for order, jobs in ORDERS.items():
        for arm in ARMS:
            for seed in SEEDS:
                first = " ".join(rows["%s-shared-seed%d-stage1" % (order, seed)]["command"])
                assert 'MODEL_DIR="{work}/models/Qwen3-1.7B"' in first
                for stage in range(2, len(jobs) + 1):
                    row = rows["%s-%s-seed%d-stage%d" % (order, arm, seed, stage)]
                    previous = "shared" if stage == 2 else arm
                    wanted = ("{work}/arms/%s-%s/runs/%s-pos%d-seed%d-a*/hf-step40"
                              % (order, previous, jobs[stage - 2], stage - 1, seed))
                    assert wanted in " ".join(row["command"]), row["id"]
                    assert row["needs"][-1] == "%s-%s-seed%d-stage%d" % (order, previous, seed, stage - 1)


def test_every_run_is_named_the_way_the_plasticity_probe_reads_names(campaign):
    """<job>-pos<POSITION>-seed<SEED>: kit/plasticity.py compares position 1 with position 4."""
    seen = {}
    for row in campaign["rows"]:
        name = (row.get("env") or {}).get("NAME")
        if not name or "run_grpo" not in " ".join(row["command"]):
            continue
        stem = name.replace("-a{attempt}", "")
        point = plasticity.POINT.match(stem)
        assert point, name
        work = row["env"]["WORK"]
        assert (work, stem) not in seen, "two runs would write to %s/runs/%s" % (work, stem)
        seen[(work, stem)] = row["id"]
    # the pair plan 4b's bar needs: one job at position 1 and at position 4, under one seed
    positions = {(plasticity.POINT.match(stem)["job"], int(plasticity.POINT.match(stem)["position"]))
                 for _work, stem in seen}
    assert ("spider", plasticity.POSITION_EARLY) in positions
    assert ("spider", plasticity.POSITION_LATE) in positions


def test_the_two_orders_never_put_one_job_in_the_same_position_twice():
    """Otherwise one arm's run directories would collide, and the probe could not tell them apart."""
    pairs = [(job, stage + 1) for jobs in ORDERS.values() for stage, job in enumerate(jobs)]
    assert len(pairs) == len(set(pairs))


# ------------------------------------------------------------------------------- 2. the pilot gate
def test_the_pilot_rows_gate_every_later_row(campaign):
    pilots = [row["id"] for row in campaign["rows"] if row.get("pilot")]
    assert pilots == ["base-forget-1", "base-repeatable", "pilot-stage1", "pilot-learned-sql",
                      "pilot-stage2", "pilot-damaged-sql"]
    ids = [row["id"] for row in campaign["rows"]]
    for row in campaign["rows"]:
        earlier = {pilot for pilot in pilots if ids.index(pilot) < ids.index(row["id"])}
        assert earlier <= set(row["needs"]), "%s can run before %s" % (row["id"], sorted(earlier - set(row["needs"])))
    bars = {row["id"]: row["bars"] for row in campaign["rows"]}
    improved = next(bar for bar in bars["pilot-learned-sql"] if bar["key"] == "delta")
    damaged = next(bar for bar in bars["pilot-damaged-sql"] if bar["key"] == "delta")
    assert improved["min"] == 5 and "max" not in improved
    assert damaged["max"] == -5 and "min" not in damaged
    for row_id in ("pilot-learned-sql", "pilot-damaged-sql"):
        assert any(bar["key"] == "same_machine_flag" and bar["min"] == 1 for bar in bars[row_id]), row_id


def test_the_pilot_is_two_jobs_one_method_one_seed_and_compares_the_right_scorings(campaign):
    rows = rows_by_id(campaign)
    assert rows["pilot-stage1"]["env"]["NAME"].startswith("spider-pos1-seed0")
    assert rows["pilot-stage2"]["env"]["NAME"].startswith("gsm8k-pos2-seed0")
    for stage in ("pilot-stage1", "pilot-stage2"):
        assert rows[stage]["env"]["KL"] == "0", stage          # one method: plain GRPO
        assert rows[stage]["env"]["SEED"] == "0", stage        # one seed
    learned = " ".join(rows["pilot-learned-sql"]["command"])
    damaged = " ".join(rows["pilot-damaged-sql"]["command"])
    assert "base-spider-a*" in learned and "pilot-stage1-spider-a*" in learned
    assert "pilot-stage1-spider-a*" in damaged and "pilot-stage2-spider-a*" in damaged
    assert "--key correct" in learned and "--key correct" in damaged
    # the pilot must not depend on the bed another worker is writing
    for row in campaign["rows"]:
        if row["id"].startswith("pilot-"):
            assert "code" not in (row.get("tags") or []), row["id"]
            assert "beds/code.py" not in " ".join(" ".join(step) for step in row.get("prepare") or [])


def test_nothing_needs_a_private_model_or_our_own_code(campaign):
    text = CAMPAIGN.read_text()
    assert "HopitAI" not in text and "hf auth" not in text
    downloads = [line for line in text.split("\n") if "hf download" in line]
    assert downloads and all("Qwen/Qwen3-1.7B" in line for line in downloads), downloads
    for row in campaign["rows"]:
        ran = " ".join(row["command"]) + " " + " ".join(" ".join(step) for step in row.get("prepare") or [])
        assert "continual." not in ran and "continual/" not in ran, row["id"]
        assert "modal" not in ran.lower(), row["id"]
        for script in ("runner.py", "run_grpo.sh", "eval_bed.py", "score_forgetting.py", "delta.py",
                       "mix.py", "sequence.py", "scorecard.py", "plasticity.py", "k5_report.py",
                       "beds/spider.py", "beds/gsm8k.py", "beds/finqa.py"):
            if script in ran:
                assert (KIT / script).is_file(), script


# ------------------------------------------------------- 3. the tools match the campaign's folders
def resolved(campaign, row) -> dict:
    return runner.resolve(row, campaign, 1)


def out_dirs(campaign) -> list:
    return [(row["id"], resolved(campaign, row)["env"]["OUT"])
            for row in campaign["rows"] if (row.get("env") or {}).get("OUT")]


def test_every_folder_the_campaign_writes_is_matched_by_the_tools_that_read_it(campaign, tmp_path):
    """The K4a lesson, built FROM the campaign file: a folder no tool matches is a silent hole."""
    os.environ["WORK"] = str(tmp_path)
    campaign = runner.load_campaign(CAMPAIGN)
    points = {"base"}
    for row_id, out in out_dirs(campaign):
        name = Path(out).name.replace("-a1", "")
        if "/eval/" in out:
            beds = [bed for bed in BEDS if name.endswith("-" + bed)]
            assert len(beds) == 1, out
            stem = name[: -len(beds[0]) - 1]
            if not row_id.startswith("pilot-"):
                assert k5_report.POINT.match(stem), out
                assert sequence.POINT.match(stem) or stem == "base", out
                points.add(stem)
        elif "/forgetting/" in out and name not in ("base", "base-repeat"):
            assert k5_report.POINT.match(name), out
            points.add(name)
        elif "/plasticity/" in out:
            assert k5_report.POINT.match(name), out
        elif "/plasticity-compare/" in out:
            assert k5_report.COMPARE.match(name), out
    # and every point named by the grid is one of the points the report will look for
    for order, jobs in ORDERS.items():
        for arm in ARMS:
            for seed in SEEDS:
                for stage in range(1, len(jobs) + 1):
                    assert k5_report.point_name(order, arm, seed, stage) in points


# ------------------------------------------------------------------- bars, doses and the hardware
WRITTEN_AS_NUMBERS = {
    "total_correct": "score_forgetting.py writes it in forgetting.json",
    "identical_answers": "score_forgetting.py agree",
    "changed_verdicts": "score_forgetting.py agree",
    "same_machine_flag": "score_forgetting.py agree, and delta.py",
    "n": "eval_bed.py writes it in bed-score.json",
    "delta": "delta.py",
    "returncode": "run_grpo.sh writes it in train-summary.json",
    "merged": "run_grpo.sh",
    "steps": "run_grpo.sh",
    "kl": "run_grpo.sh",
    "prompts_measured": "plasticity.py probe",
    "comparable": "plasticity.py compare, scorecard.py and k5_report.py",
    "cells_named": "sequence.py manifest",
    "seeds_named": "sequence.py manifest",
    "cells_read": "scorecard.py",
    "seeds_reported": "scorecard.py and k5_report.py",
    "orders_reported": "k5_report.py",
    "arms_reported": "k5_report.py",
}


def test_every_bar_reads_a_key_a_tool_writes_as_a_number(campaign):
    used = {(row["id"], bar["key"]) for row in campaign["rows"] for bar in row["bars"]}
    assert [pair for pair in sorted(used) if pair[1] not in WRITTEN_AS_NUMBERS] == []
    for row in campaign["rows"]:
        assert row["bars"], "%s has no bar: nothing would judge it" % row["id"]
        for bar in row["bars"]:
            assert "min" in bar or "max" in bar, (row["id"], bar)
            assert bar["source"].endswith(".json"), (row["id"], bar)


@needs_generator
def test_the_dose_is_forty_steps_of_thirty_two_questions_with_the_share_exact_over_them():
    assert generator.ROWS == generator.STEPS * generator.BATCH == 1280
    share = generator.ARMS["rehearse10"]["share"]
    job_rows = int(round(generator.ROWS * (1 - share)))
    assert job_rows == 1152
    added = mix.rows_to_add(job_rows, share)
    assert added == 128 and job_rows + added == generator.ROWS
    assert added / generator.ROWS == share
    # FinQA needs a margin because verl drops prompts over 2048 tokens (K1c's FINQA_TRAIN_ROWS)
    assert generator.stage_rows(generator.ROWS, "finqa") == 1600
    assert generator.stage_rows(job_rows, "finqa") == 1440
    assert generator.stage_rows(generator.ROWS, "gsm8k") == 1280


def test_each_stages_rows_come_from_the_files_its_prepare_step_writes(campaign):
    rows = rows_by_id(campaign)
    for order, jobs in ORDERS.items():
        for arm in ARMS:
            for seed in SEEDS:
                for stage in range(1, len(jobs) + 1):
                    row = rows[("%s-shared-seed%d-stage1" % (order, seed)) if stage == 1
                               else "%s-%s-seed%d-stage%d" % (order, arm, seed, stage)]
                    train = row["env"]["TRAIN_FILE"]
                    assert train in row["requires"] and row["env"]["VAL_FILE"] in row["requires"]
                    prepared = " ".join(" ".join(step) for step in row["prepare"])
                    assert train.rsplit("/", 1)[0] in prepared, row["id"]
                    if arm == "rehearse10" and stage > 1:
                        assert "/mix/" in train
                        assert "--share 0.10 --seed %d" % seed in prepared
                        for earlier in jobs[:stage - 1]:
                            assert "/data/%s/train.parquet" % earlier in prepared, (row["id"], earlier)
                    else:
                        assert "/stage/" in train
                    assert row["env"]["STEPS"] == "40" and row["env"]["SEED"] == str(seed)


def test_training_uses_every_gpu_logs_its_metrics_and_scoring_uses_only_the_first(campaign):
    for row in campaign["rows"]:
        env, command = row["env"], " ".join(row["command"])
        if "run_grpo" in command:
            assert env["NGPU"] == "8" and "CUDA_VISIBLE_DEVICES" not in env, row["id"]
            assert env["FILE_LOG"] == "1", row["id"]        # entropy is logged from here on (Q9)
            assert env["TEST_FREQ"] == "-1", row["id"]      # K3's setting: no in-trainer validation
        elif "eval_bed" in command or "score_forgetting.py\" generate" in command or "plasticity.py\" probe" in command:
            assert env.get("CUDA_VISIBLE_DEVICES") == "0", row["id"]


@needs_generator
def test_the_coding_rows_are_all_tagged_and_can_be_left_out(campaign, tmp_path):
    for row in campaign["rows"]:
        ran = " ".join(row["command"]) + " " + " ".join(" ".join(s) for s in row.get("prepare") or [])
        touches_code = "beds/code.py" in ran or "--bed code " in ran or "CODE_ROOT" in ran
        assert touches_code == ("code" in (row.get("tags") or [])), row["id"]
    without = generator.build(jobs=tuple(j for j in generator.JOBS if j != "code"))
    path = tmp_path / "k5-without-code.yaml"
    path.write_text(without)
    os.environ.setdefault("WORK", "/tmp/unused-k5")
    fallback = runner.load_campaign(path)
    for row in fallback["rows"]:
        ran = " ".join(row["command"]) + " " + " ".join(" ".join(s) for s in row.get("prepare") or [])
        assert not any(marker in ran for marker in generator.CODE_MARKERS), row["id"]
        assert "code" not in (row.get("tags") or []), row["id"]
    assert any("--bed spider" in " ".join(row["command"]) for row in fallback["rows"])
    assert any("--bed finqa" in " ".join(row["command"]) for row in fallback["rows"])


@needs_generator
def test_the_coding_rows_call_the_coding_bed_the_way_that_bed_is_written(campaign):
    """Read FROM kit/beds/code.py, so that a change to its interface fails here and not on a GPU."""
    bed = KIT / "beds" / "code.py"
    if not bed.is_file():
        pytest.skip("kit/beds/code.py has not landed yet; the coding rows are tagged `code`")
    source = bed.read_text()
    spec = generator.JOBS["code"]
    assert 'prepare.add_argument("%s"' % spec["prepare_flag"] in source
    assert spec["root_env"] in source                     # the bed reads its root from this variable
    assert "CODE_TESTS" in source                         # ... and its tests from this one
    assert '"heldout"' in source and "SPLITS" in source
    split = json.loads((KIT / "beds" / "code-split-v1.json").read_text())
    assert spec["held_out"] == split["counts"]["panel"], "the panel is 51 problems, and a smaller " \
                                                         "panel is not the panel"
    # every row carrying coding questions exports the right tests file: the training subset while
    # training, every test on the panel. A rehearsal stage after the coding stage carries coding
    # questions too, and its reward needs the tests as much as the coding stage's does.
    for row in campaign["rows"]:
        ran = " ".join(row["command"]) + " " + " ".join(" ".join(s) for s in row.get("prepare") or [])
        tests = (row.get("env") or {}).get("CODE_TESTS")
        if "beds/code.py" in ran or "--bed code " in ran:
            assert tests is not None, "%s trains or scores on coding rows with no CODE_TESTS" % row["id"]
        if tests is None:
            continue
        wanted = "heldout" if "eval_bed.py" in " ".join(row["command"]) else "train"
        assert tests.endswith("/%s.tests.jsonl" % wanted), row["id"]
        assert tests in row["requires"], row["id"]


def test_the_coding_bed_is_not_reachable_from_the_kits_two_entry_points_yet():
    """A tripwire, not a check: the day someone wires it in, this starts checking instead of skipping.

    The campaign scores the coding bed through `kit/eval_bed.py generate --bed code` and trains on
    coding rows through `kit/beds/rewards.py`, which dispatches on `data_source`. Neither file knows
    the bed yet, and neither belongs to this package.
    """
    eval_bed = (KIT / "eval_bed.py").read_text()
    rewards = (KIT / "beds" / "rewards.py").read_text()
    missing = []
    if '"code":' not in eval_bed.split("BED_FILES", 1)[-1][:400]:
        missing.append("kit/eval_bed.py needs `code` in BED_FILES, DEFAULT_SPLIT, SPLITS and "
                       "MAX_NEW_TOKENS, and a branch in items_of/score_one for its prompts and its "
                       "sandboxed checker")
    if '"code":' not in rewards.split("BED_FILES", 1)[-1][:400]:
        missing.append("kit/beds/rewards.py needs `code` in BED_FILES, or a coding row raises "
                       "UnknownDataSource('livecodebench') inside the trainer")
    if missing:
        pytest.skip("; ".join(missing))
    assert "--bed code " in CAMPAIGN.read_text()


def test_the_coding_bed_is_always_scored_last_at_a_point(campaign):
    """`run --all` stops at the first failure, and the coding bed does not exist yet."""
    ids = [row["id"] for row in campaign["rows"]]
    for point in ("base", "sqlfirst-shared-seed0-stage1", "mathsfirst-none-seed3-stage4"):
        here = [index for index, row_id in enumerate(ids) if row_id.startswith(point + "-")]
        code_row = ids.index("%s-code" % point)
        assert code_row == max(here), point


# ------------------------------------------------------------------------------- the arms
@needs_generator
def test_an_arm_the_kit_cannot_run_refuses_and_says_what_is_missing():
    for arm in ("sdft", "isdft"):
        with pytest.raises(generator.ArmError) as exc:
            generator.build(arms=("none", arm))
        assert "launcher" in str(exc.value) and "Nothing has been written" in str(exc.value)
    with pytest.raises(generator.ArmError):
        generator.build(arms=("nosucharm",))
    with pytest.raises(generator.ArmError):
        generator.build(arms=())


@needs_generator
def test_an_unavailable_arm_can_be_written_as_a_row_that_fails_rather_than_silence(tmp_path):
    text = generator.build(arms=("none", "sdft"), refusing_rows=True)
    path = tmp_path / "sdft.yaml"
    path.write_text(text)
    os.environ["WORK"] = str(tmp_path)
    campaign = runner.load_campaign(path)
    row = rows_by_id(campaign)["sdft-unavailable"]
    assert row["tags"] == ["sdft"]
    done = subprocess.run(row["command"], capture_output=True, text=True)
    assert done.returncode == 2 and "launcher" in done.stderr
    assert "sdft" not in " ".join(r["id"] for r in campaign["rows"] if r["id"] != "sdft-unavailable")


@needs_generator
def test_the_kl_arm_is_buildable_and_turns_the_loss_on_only_after_the_first_stage(tmp_path):
    text = generator.build(arms=("none", "kl"))
    path = tmp_path / "kl.yaml"
    path.write_text(text)
    os.environ["WORK"] = str(tmp_path)
    campaign = runner.load_campaign(path)
    rows = rows_by_id(campaign)
    assert rows["sqlfirst-shared-seed0-stage1"]["env"]["KL"] == "0"
    assert rows["sqlfirst-kl-seed0-stage2"]["env"]["KL"] == "1"
    assert rows["sqlfirst-none-seed0-stage2"]["env"]["KL"] == "0"
    kl_bar = next(bar for bar in rows["sqlfirst-kl-seed0-stage2"]["bars"] if bar["key"] == "kl")
    assert kl_bar["min"] == 1


# ------------------------------------------------------------------------------- the plan runs dry
def test_the_plan_prints_every_row_and_executes_nothing(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    done = subprocess.run([sys.executable, str(KIT / "runner.py"), "plan", str(CAMPAIGN)],
                          env={**os.environ, "WORK": str(work), "SDPO_DIR": "/s", "MODEL_DIR": "/m"},
                          capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    printed = [line for line in done.stdout.split("\n") if line.startswith("[")]
    assert len(printed) == CAMPAIGN.read_text().count("  - id:")
    assert list(work.iterdir()) == [], "planning must create nothing"


# ------------------------------------------------------------------------------- 4. kit/sequence.py
def _rows(source: str, n: int) -> list:
    return [{"data_source": source, "prompt": [{"role": "user", "content": "q%d" % i}],
             "ability": source, "reward_model": {"style": source, "ground_truth": str(i)},
             "extra_info": {"index": i}} for i in range(n)]


def _write_jsonl(path: Path, rows: list) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    return path


def test_pool_writes_exactly_the_rows_asked_for_and_repeats_only_when_it_must(tmp_path):
    source = _write_jsonl(tmp_path / "spider.jsonl", _rows("spider", 640))
    manifest = sequence.build_pool([str(source)], 1280, 0, tmp_path / "out", allow_no_parquet=True)
    written = [json.loads(line) for line in (tmp_path / "out" / "train.jsonl").read_text().split("\n") if line]
    assert manifest["rows_total"] == len(written) == 1280
    assert manifest["distinct_per_input"] == [640] and manifest["repeats_per_input"] == [640]
    assert sorted(row["extra_info"]["index"] for row in written) == sorted(list(range(640)) * 2)
    assert written[:640] != written[640:], "the second pass is a different order, not a copy"


def test_pool_draws_equally_from_every_job_it_is_given(tmp_path):
    a = _write_jsonl(tmp_path / "a.jsonl", _rows("spider", 100))
    b = _write_jsonl(tmp_path / "b.jsonl", _rows("gsm8k", 40))
    manifest = sequence.build_pool([str(a), str(b)], None, 0, tmp_path / "pool", allow_no_parquet=True)
    assert manifest["rows_total"] == 80 and manifest["rows_per_input"] == [40, 40]
    assert manifest["data_source_counts"] == {"gsm8k": 40, "spider": 40}
    assert manifest["repeats_per_input"] == [0, 0]


def test_pool_is_deterministic_in_its_seed_and_never_overwrites(tmp_path):
    source = _write_jsonl(tmp_path / "a.jsonl", _rows("spider", 50))
    first = sequence.build_pool([str(source)], 60, 7, tmp_path / "one", allow_no_parquet=True)
    again = sequence.build_pool([str(source)], 60, 7, tmp_path / "two", allow_no_parquet=True)
    other = sequence.build_pool([str(source)], 60, 8, tmp_path / "three", allow_no_parquet=True)
    assert first["outputs"]["jsonl_sha256"] == again["outputs"]["jsonl_sha256"]
    assert first["outputs"]["jsonl_sha256"] != other["outputs"]["jsonl_sha256"]
    with pytest.raises(sequence.SequenceError, match="never replaced"):
        sequence.build_pool([str(source)], 60, 7, tmp_path / "one", allow_no_parquet=True)


def test_pool_refuses_inputs_that_are_not_one_table(tmp_path):
    a = _write_jsonl(tmp_path / "a.jsonl", _rows("spider", 10))
    odd = [dict(row, surprise=1) for row in _rows("gsm8k", 10)]
    b = _write_jsonl(tmp_path / "b.jsonl", odd)
    with pytest.raises(sequence.SequenceError, match="ONE table"):
        sequence.build_pool([str(a), str(b)], None, 0, tmp_path / "out", allow_no_parquet=True)
    with pytest.raises(sequence.SequenceError, match="at least one --from"):
        sequence.build_pool([], None, 0, tmp_path / "out2", allow_no_parquet=True)


def test_pool_reads_a_trainer_file_exactly_as_mix_does():
    """kit/mix.py is the kit's reference for what a trainer row file is; these two must not drift."""
    ours = (KIT / "sequence.py").read_text()
    theirs = (KIT / "mix.py").read_text()
    for function in ("def read_rows(path) -> list:", "def columns_of(rows: list, path) -> tuple:"):
        assert function in ours and function in theirs
        body_ours = " ".join(ours.split(function, 1)[1].split("\ndef ", 1)[0].split())
        body_theirs = " ".join(theirs.split(function, 1)[1].split("\ndef ", 1)[0].split())
        assert body_ours.replace("SequenceError", "X") == body_theirs.replace("MixError", "X"), function


# ------------------------------------------------------------- the tree the campaign would write
BASE_SCORES = {"spider": 40, "code": 10, "gsm8k": 100, "finqa": 200}
PANELS = {"ifeval": 80, "knowledge": 70, "math": 90}
LEARNED, DECAY = 20, 5


def _score_of(bed: str, jobs, stage: int, seed: int) -> int:
    """A made-up but LAWFUL tree: a job gains 20 when it is its turn and loses 5 a stage afterwards."""
    value = BASE_SCORES[bed] + seed
    if bed in jobs[:stage]:
        position = jobs.index(bed) + 1
        value += LEARNED - DECAY * (stage - position)
    return value


def _point_fields(point: str):
    if point == "base":
        return None
    match = k5_report.POINT.match(point)
    return (match["order"], match["arm"], int(match["seed"]), int(match["stage"]))


def _write_tree(tmp_path: Path, *, machine_of=lambda point: "m1", skip=()) -> Path:
    """Every directory the campaign writes, filled with a lawful scoring. Built FROM the campaign."""
    os.environ["WORK"] = str(tmp_path)
    campaign = runner.load_campaign(CAMPAIGN)
    for row in campaign["rows"]:
        out = (resolved(campaign, row)["env"] or {}).get("OUT")
        if not out or row["id"] in skip:
            continue
        path, name = Path(out), Path(out).name.replace("-a1", "")
        if "/eval/" in out:
            bed = next((bed for bed in BEDS if name.endswith("-" + bed)), None)
            point = name[: -len(bed) - 1] if bed else None
            if point is None or (point != "base" and not k5_report.POINT.match(point)):
                continue
            fields = _point_fields(point)
            jobs = ORDERS[fields[0]] if fields else ORDERS["sqlfirst"]
            stage, seed = (fields[3], fields[2]) if fields else (0, 0)
            correct = _score_of(bed, jobs, stage, seed)
            path.mkdir(parents=True)
            payload = {"schema": "kit-bed-score.v1", "bed": bed, "n": HELD_OUT[bed], "correct": correct,
                       "total_correct": correct, "accuracy": round(correct / HELD_OUT[bed], 6),
                       "incorrect_format": 0}
            if machine_of(point) is not None:
                payload["machine"] = {"id": machine_of(point), "deterministic": True}
            (path / "bed-score.json").write_text(json.dumps(payload), encoding="utf-8")
        elif "/forgetting/" in out and (name == "base" or k5_report.POINT.match(name)):
            fields = _point_fields(name)
            stage = fields[3] if fields else 0
            panels = {panel: value - stage for panel, value in PANELS.items()}
            path.mkdir(parents=True)
            payload = {"schema": "kit-forgetting.v1",
                       "panels": {panel: {"correct": value, "n": 100} for panel, value in panels.items()},
                       "total_correct": sum(panels.values())}
            if machine_of(name) is not None:
                payload["machine"] = {"id": machine_of(name), "deterministic": True}
            (path / "forgetting.json").write_text(json.dumps(payload), encoding="utf-8")
        elif "/plasticity/" in out and (name == "base" or k5_report.POINT.match(name)):
            fields = _point_fields(name)
            stage = fields[3] if fields else 0
            path.mkdir(parents=True)
            (path / "plasticity.json").write_text(json.dumps(
                {"schema": "kit-plasticity.v1", "mode": "probe", "prompts_measured": 200,
                 "dormant_share": 0.10 + 0.01 * stage, "effective_rank": 100 - stage,
                 "machine": {"id": "m1"}}), encoding="utf-8")
    return tmp_path / "k5"


def test_the_manifest_is_built_from_the_campaign_and_feeds_the_scorecard(tmp_path):
    root = _write_tree(tmp_path)
    out = tmp_path / "sequences" / "sqlfirst-rehearse10-a1.json"
    manifest = sequence.build_manifest(CAMPAIGN, root, "sqlfirst", "rehearse10")
    out.parent.mkdir(parents=True)
    out.write_text(json.dumps(manifest), encoding="utf-8")
    assert manifest["jobs"] == list(ORDERS["sqlfirst"])
    assert manifest["seeds_named"] == len(SEEDS)
    assert manifest["cells_named"] == (len(ORDERS["sqlfirst"]) + 1) * len(ORDERS["sqlfirst"]) * len(SEEDS)
    assert manifest["runs"][0]["stages"][0]["point"] == "sqlfirst-shared-seed0-stage1"
    assert manifest["runs"][0]["stages"][1]["point"] == "sqlfirst-rehearse10-seed0-stage2"
    card = scorecard.build(out, key="accuracy")
    assert card["jobs"] == list(ORDERS["sqlfirst"]) and card["seeds"] == list(SEEDS)
    assert card["comparable"] == 1 and card["cells_read"] == manifest["cells_named"]
    # spider is learned at stage 1 and loses 5 a stage for the three that follow
    assert card["per_seed"]["0"]["per_job_backward"]["spider"] == pytest.approx(-15 / HELD_OUT["spider"])


def test_the_manifest_bar_in_the_campaign_is_the_number_the_manifest_writes(campaign, tmp_path):
    rows = rows_by_id(campaign)
    bar = next(bar for bar in rows["sqlfirst-none-manifest"]["bars"] if bar["key"] == "cells_named")
    root = _write_tree(tmp_path)
    manifest = sequence.build_manifest(CAMPAIGN, root, "sqlfirst", "none")
    assert manifest["cells_named"] == bar["min"]


def test_a_missing_cell_is_a_refusal_that_names_it(tmp_path):
    root = _write_tree(tmp_path, skip=("sqlfirst-none-seed2-stage3-gsm8k",))
    with pytest.raises(sequence.SequenceError) as exc:
        sequence.build_manifest(CAMPAIGN, root, "sqlfirst", "none")
    assert "seed 2 stage 3" in str(exc.value) and "gsm8k" in str(exc.value)
    assert "missing" in str(exc.value)
    sequence.build_manifest(CAMPAIGN, root, "sqlfirst", "rehearse10")     # the other arm is untouched


def test_the_manifest_reads_the_highest_attempt_of_each_scoring(tmp_path):
    root = _write_tree(tmp_path)
    first = json.loads((root / "eval" / "base-spider-a1" / "bed-score.json").read_text())
    second = dict(first, correct=first["correct"] + 7)
    (root / "eval" / "base-spider-a2").mkdir()
    (root / "eval" / "base-spider-a2" / "bed-score.json").write_text(json.dumps(second))
    manifest = sequence.build_manifest(CAMPAIGN, root, "sqlfirst", "none")
    assert manifest["runs"][0]["untrained"]["scores"]["spider"].endswith("base-spider-a2")


def test_the_estimate_adds_up_the_campaigns_own_arithmetic():
    estimate = sequence.build_estimate(CAMPAIGN)
    kinds = estimate["by_kind"]
    assert kinds["train"]["rows"] == 72                      # 70 grid stages + the pilot's two
    assert kinds["train"]["gpu_hours"] == pytest.approx(72 * 40 * 15 * 8 / 3600, abs=0.1)
    assert kinds["score"]["gpu_hours"] > 0 and kinds["probe"]["gpu_hours"] > 0
    assert estimate["gpu_hours"] == pytest.approx(sum(entry["gpu_hours"] for entry in kinds.values()), abs=0.1)
    assert 0 < estimate["gpu_hours_tagged_code"] < estimate["gpu_hours"]
    assert estimate["gpu_hours_without_code"] == pytest.approx(
        estimate["gpu_hours"] - estimate["gpu_hours_tagged_code"], abs=0.1)
    assert "MEASURED" in estimate["assumptions"]["train_seconds_per_step_source"]
    assert "ASSUMPTION" in estimate["assumptions"]["score_seconds_per_answer_source"]
    assert estimate["assumptions"]["panel_questions"] == 300


# ------------------------------------------------------------------------------- 5. the K5 report
def _readout(tmp_path: Path, **kwargs) -> Path:
    """The whole tree, plus the manifests the report reads the stage order from."""
    root = _write_tree(tmp_path, **kwargs)
    for order in ORDERS:
        for arm in ARMS:
            manifest = sequence.build_manifest(CAMPAIGN, root, order, arm)
            path = root / "sequences" / ("%s-%s-a1.json" % (order, arm))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(manifest), encoding="utf-8")
    return root


def test_the_report_joins_every_order_arm_and_seed(tmp_path):
    root = _readout(tmp_path)
    report = k5_report.build(root)
    assert report["orders_reported"] == 2 and report["arms_reported"] == 2
    assert report["seeds_reported"] == len(SEEDS) and report["comparable"] == 1
    assert report["beds"] == list(BEDS)
    assert report["base"]["scores"]["spider"] == BASE_SCORES["spider"]
    arm = report["orders"]["sqlfirst"]["arms"]["none"]
    assert arm["seeds_reported"] == len(SEEDS)
    # the first job is read against the untrained model, which carries no per-seed jitter, so its
    # gain carries the whole of it: the mean and the sample spread are both exercised here
    assert arm["learned"]["spider"]["mean"] == LEARNED + statistics.fmean(SEEDS)
    assert arm["learned"]["spider"]["sd"] == pytest.approx(statistics.stdev(SEEDS), abs=1e-4)
    # a later job is read against the stage before it, where the jitter cancels
    assert arm["learned"]["finqa"] == {"n": len(SEEDS), "mean": LEARNED, "sd": 0.0}
    assert arm["kept"]["spider"]["mean"] == -DECAY * 3        # three stages followed it
    assert arm["kept"]["spider"]["sd"] == 0                   # the jitter cancels in a difference
    assert arm["kept"]["code"]["mean"] == -DECAY * 2
    assert arm["general_delta"]["math"]["mean"] == -len(ORDERS["sqlfirst"])
    assert arm["missing_points"] == []
    seed0 = arm["seeds"]["0"]
    assert seed0["stages"][0]["point"] == "base" and seed0["stages"][1]["point"] == "sqlfirst-shared-seed0-stage1"
    assert seed0["final"]["spider"] == BASE_SCORES["spider"] + LEARNED - DECAY * 3


def test_the_report_refuses_to_subtract_counts_made_on_two_machines(tmp_path):
    root = _readout(tmp_path, machine_of=lambda point: "m2" if point == "base" else "m1")
    with pytest.raises(k5_report.K5ReportError, match="different machine-and-mode fingerprints"):
        k5_report.build(root)
    allowed = k5_report.build(root, allow_different_machines=True)
    assert allowed["comparable"] == 0 and allowed["different_machines_allowed"] == 1
    assert "not comparable" in k5_report.render(allowed)


def test_the_report_will_not_guess_which_job_a_stage_learned(tmp_path):
    root = _write_tree(tmp_path)
    with pytest.raises(k5_report.K5ReportError, match="no sequence manifests"):
        k5_report.build(root)


def test_the_report_writes_both_files_and_never_overwrites(tmp_path):
    root = _readout(tmp_path)
    out = tmp_path / "report-a1"
    assert k5_report.main(["--root", str(root), "--out", str(out)]) == 0
    written = json.loads((out / "k5-report.json").read_text())
    assert written["orders_reported"] == 2
    text = (out / "k5-report.md").read_text()
    assert "## sqlfirst: spider then code then gsm8k then finqa" in text
    assert "### arm `rehearse10`" in text
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        k5_report.main(["--root", str(root), "--out", str(out)])


def test_an_empty_tree_is_a_refusal_not_an_empty_report(tmp_path):
    (tmp_path / "k5").mkdir()
    with pytest.raises(k5_report.K5ReportError, match="eval/base"):
        k5_report.build(tmp_path / "k5")
