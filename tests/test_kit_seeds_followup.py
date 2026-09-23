"""The two seed follow-ups: kit/campaigns/k3-seeds-3-4.yaml, kit/campaigns/k4a-seeds-45-46.yaml,
their generator, and kit/pilots_passed.py.

A follow-up is a dangerous shape of file: it is read as though it were the original campaign, and
every number it produces is put in the same table as numbers from a run that happened weeks earlier.
Five ways it could be quietly wrong, one section each:

1. A seed row is not the original's row. If one setting drifted -- a step count, a mixture share, a
   bar -- seed 3 would be a different experiment from seeds 0 to 2, and the report would average it
   with them anyway. Every seed row here is compared, key for key, with the row of the original
   campaign it repeats, with only the seed substituted.
2. The follow-up runs a pilot of its own, or repeats the original's. Its only pilot must train
   nothing: it reads the original's pilot verdicts out of the runner's own state in THIS work
   directory, and refuses everything below unless all of them passed.
3. `kit/pilots_passed.py` calls a campaign passed that was not. It must count a failed pilot, an
   unfinished attempt and a campaign that never ran here as NOT passed, always exit 0 so the bar is
   what decides, and read the latest attempt of each row.
4. The naming drifted, so the original's report tool does not see the new seeds. Each report tool is
   run here over a synthetic tree holding the original's three seeds and the follow-up's two, and
   must report five.
5. The follow-up writes over the original's report, or collides with its runner state.

Everything builds its own fixtures; nothing needs a GPU, a trainer, or the network.
"""
from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
KIT = ROOT / "kit"
README = KIT / "README-seeds.md"
GATE = "original-pilots-passed"

#: (the follow-up, the campaign it repeats, the seed of the original whose rows every new seed row
#: is a copy of, the new seeds). Seed 0 of K3 is the pilot's own seed and has a different shape, so
#: seed 1 is the template there; K4a's three seeds are all the same shape.
PACKAGES = {
    "k3": {"followup": KIT / "campaigns" / "k3-seeds-3-4.yaml",
           "original": KIT / "campaigns" / "k3-replay.yaml",
           "template_seed": 1, "new_seeds": (3, 4), "rows_per_seed": 20},
    "k4a": {"followup": KIT / "campaigns" / "k4a-seeds-45-46.yaml",
            "original": KIT / "campaigns" / "k4a-stuck-problems.yaml",
            "template_seed": 42, "new_seeds": (45, 46), "rows_per_seed": 6},
}


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = load("kit_runner_for_seeds", KIT / "runner.py")
pilots_passed = load("kit_pilots_passed", KIT / "pilots_passed.py")
k3_report = load("kit_k3_report_for_seeds", KIT / "k3_report.py")
k4a_report = load("kit_k4a_report_for_seeds", KIT / "k4a_report.py")

#: The exported kit carries kit/ and these tests, but not scripts/. Everything below works on the
#: export; only the generator-parity test needs the private repository, and it skips without it.
GENERATOR = ROOT / "scripts" / "make_seeds_followup.py"
generator = load("make_seeds_followup", GENERATOR) if GENERATOR.is_file() else None


def rows_of(path: Path) -> dict:
    """The campaign's rows exactly as written, without the runner's added `needs`."""
    return {row["id"]: row for row in yaml.safe_load(path.read_text())["rows"]}


@pytest.fixture(params=sorted(PACKAGES), ids=sorted(PACKAGES))
def package(request):
    os.environ.setdefault("WORK", "/tmp/unused-seeds-followup")
    spec = dict(PACKAGES[request.param])
    spec["name"] = request.param
    spec["campaign"] = runner.load_campaign(spec["followup"])
    spec["original_campaign"] = runner.load_campaign(spec["original"])
    return spec


# ------------------------------------------------------------------------ 1. generated, not edited
@pytest.mark.skipif(generator is None, reason="scripts/ is not part of the exported kit")
def test_both_committed_campaigns_are_what_the_generator_builds():
    assert PACKAGES["k3"]["followup"].read_text() == generator.build_k3(), "re-run scripts/make_seeds_followup.py"
    assert PACKAGES["k4a"]["followup"].read_text() == generator.build_k4a(), "re-run scripts/make_seeds_followup.py"


@pytest.mark.skipif(generator is None, reason="scripts/ is not part of the exported kit")
def test_the_originals_were_not_touched():
    """The two published campaigns may be running on the partner's machine as this is read."""
    for key in ("k3", "k4a"):
        original = PACKAGES[key]["original"]
        built = {"k3": generator.k3.build, "k4a": generator.k4a.build}[key]()
        assert original.read_text() == built, "%s changed; it is published and may be running" % original.name


# ------------------------------------------------------- 2. a seed row is the original's seed row
def with_seed(value, old: int, new: int, key=None):
    """The original's row, with its seed and nothing else replaced.

    A bare number is only ever a seed under the env variable named SEED: `KL: "1"` is an arm, not
    seed 1, and a substitution that moved it would make this test pass over a changed experiment.
    """
    if isinstance(value, list):
        return [with_seed(item, old, new, key) for item in value]
    if isinstance(value, dict):
        return {name: with_seed(item, old, new, name) for name, item in value.items()}
    if not isinstance(value, str):
        return value
    if key == "SEED" and value == str(old):
        return str(new)
    return value.replace("seed%d" % old, "seed%d" % new).replace("--seed %d" % old, "--seed %d" % new)


@pytest.mark.parametrize("key", sorted(PACKAGES))
def test_every_seed_row_is_the_originals_row_with_only_the_seed_changed(key):
    spec = PACKAGES[key]
    old, new_rows = spec["template_seed"], rows_of(spec["followup"])
    original = rows_of(spec["original"])
    covered = set()
    for seed in spec["new_seeds"]:
        mine = {row_id: row for row_id, row in new_rows.items() if "seed%d" % seed in row_id}
        assert len(mine) == spec["rows_per_seed"], (seed, sorted(mine))
        for row_id, row in mine.items():
            template = row_id.replace("seed%d" % seed, "seed%d" % old)
            assert template in original, "%s repeats %s, which the original does not have" % (row_id, template)
            assert with_seed(original[template], old, seed) == row, row_id
            covered.add(template)
    assert covered == {row_id for row_id in original if "seed%d" % old in row_id}, \
        "the follow-up dropped a row the original runs at every seed"


@pytest.mark.parametrize("key", sorted(PACKAGES))
def test_the_follow_up_is_the_seed_rows_a_gate_and_a_report_and_nothing_else(key):
    spec = PACKAGES[key]
    ids = list(rows_of(spec["followup"]))
    assert ids[0] == GATE and ids[-1] == "report"
    assert len(ids) == len(spec["new_seeds"]) * spec["rows_per_seed"] + 2
    assert len(set(ids)) == len(ids)


def test_nothing_of_the_originals_own_seeds_is_run_again(package):
    """Re-running seed 0 would spend a day to write a number we already have, under a new attempt."""
    original_seeds = {row["id"] for row in package["original_campaign"]["rows"]}
    for row in package["campaign"]["rows"]:
        if row["id"] in (GATE, "report"):
            continue
        assert row["id"] not in original_seeds, row["id"]


def test_the_new_runs_and_folders_use_the_originals_naming_scheme(package):
    """What lets the original's report tool read five seeds without being changed.

    A run is named after its row and a scoring lands in a folder the original also wrote into. A
    follow-up that invented a folder of its own would run perfectly and report three seeds.
    """
    original = package["original"].read_text()
    for row in package["campaign"]["rows"]:
        name = row["env"].get("NAME")
        if name is not None:
            assert name == "%s-a{attempt}" % row["id"], row["id"]
        out = row["env"].get("OUT")
        if out is not None:
            assert out.rsplit("/", 1)[0] + "/" in original, (row["id"], out)
    for root in ("{work}/runs/", {"k3": "{work}/k3/eval/", "k4a": "{work}/k4a/forgetting/"}[package["name"]]):
        assert root in package["followup"].read_text(), root


# ----------------------------------------------------------------------------- 3. the gate row
def test_the_gate_row_is_a_pilot_that_trains_nothing(package):
    gate = package["campaign"]["rows"][0]
    assert gate["id"] == GATE and gate["pilot"] is True
    assert [row["id"] for row in package["campaign"]["rows"] if row.get("pilot")] == [GATE], \
        "a follow-up may not train a pilot of its own"
    command = " ".join(gate["command"])
    assert "pilots_passed.py" in command and (KIT / "pilots_passed.py").is_file()
    for script in ("run_grpo.sh", "run_sdpo_toolalpaca.sh", "eval_bed.py", "score_forgetting.py"):
        assert script not in command, script
    assert gate.get("prepare") is None and gate.get("requires") is None


def test_the_gate_rows_bars_are_numbers_that_demand_every_pilot(package):
    gate = package["campaign"]["rows"][0]
    expected = sum(1 for row in package["original_campaign"]["rows"] if row.get("pilot"))
    assert expected >= 1
    keys = {}
    for bar in gate["bars"]:
        assert bar["source"].endswith("/pilots-passed.json"), bar
        for limit in ("min", "max"):
            assert isinstance(bar[limit], int) and not isinstance(bar[limit], bool), bar
        assert bar["min"] == bar["max"] == expected, bar
        keys[bar["key"]] = bar
    assert set(keys) == {"passed_count", "expected_count"}, \
        "passed_count == expected_count is what the two bars together say"


def test_the_gate_names_the_original_campaign_file(package):
    command = " ".join(package["campaign"]["rows"][0]["command"])
    assert package["original"].name in command
    assert package["campaign"]["name"] != package["original_campaign"]["name"], \
        "two campaigns with one name would share the runner's state folder"


def test_every_row_below_the_gate_is_refused_until_it_passes(package):
    for row in package["campaign"]["rows"][1:]:
        assert GATE in row["needs"], row["id"]


def test_the_report_row_runs_the_originals_own_tool_beside_the_originals_report(package):
    report = package["campaign"]["rows"][-1]
    tool = {"k3": "k3_report.py", "k4a": "k4a_report.py"}[package["name"]]
    command = " ".join(report["command"])
    assert tool in command and (KIT / tool).is_file()
    original_report = next(row for row in package["original_campaign"]["rows"] if row["id"] == "report")
    assert " ".join(original_report["command"]).replace(original_report["env"]["OUT"], "") == \
           command.replace(report["env"]["OUT"], ""), "it must read the same tree, the same way"
    assert report["env"]["OUT"] != original_report["env"]["OUT"], \
        "the original's report folder is never written over"
    for row in package["campaign"]["rows"]:
        assert row["bars"], "%s has no bar: nothing would judge it" % row["id"]
        for bar in row["bars"]:
            assert "min" in bar or "max" in bar, (row["id"], bar)
            assert bar.get("agg", "last") in runner.AGGREGATES, (row["id"], bar)


def test_planning_a_follow_up_creates_nothing(package, tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    done = subprocess.run([sys.executable, str(KIT / "runner.py"), "plan", str(package["followup"])],
                          capture_output=True, text=True, env={**os.environ, "WORK": str(work)})
    assert done.returncode == 0, done.stderr
    assert list(work.iterdir()) == [], "planning must create nothing"
    assert "PILOT" in done.stdout


# ------------------------------------------------------------------------- 4. pilots_passed.py
def state(work: Path, campaign: str, verdicts: dict) -> None:
    """The runner's own state folder: {work}/campaign/<name>/<row>/attempt-N/verdict.json."""
    for row_id, attempts in verdicts.items():
        for attempt, verdict in attempts.items():
            directory = work / "campaign" / campaign / row_id / ("attempt-%d" % attempt)
            directory.mkdir(parents=True)
            if verdict is None:                         # an attempt that started and wrote no verdict
                continue
            (directory / "verdict.json").write_text(json.dumps(
                {"schema": "kit-campaign.v1", "row": row_id, "attempt": attempt, "verdict": verdict,
                 "reason": None if verdict == "PASS" else "a bar was missed"}))


def run_gate(package, work: Path, out: Path) -> dict:
    """The gate row's command, as the campaign runs it, and the JSON it left behind."""
    os.environ["WORK"] = str(work)
    assert pilots_passed.main(["--campaign", str(package["original"]), "--out", str(out)]) == 0
    return json.loads((out / "pilots-passed.json").read_text())


def judge_gate(package, out: Path) -> list:
    """The gate row's own bars, judged against that JSON exactly as the runner would."""
    judged = []
    for bar in package["campaign"]["rows"][0]["bars"]:
        judged.append(runner.judge_bar({**bar, "source": str(out / "pilots-passed.json")}))
    return judged


def pilot_ids(package) -> list:
    return [row["id"] for row in package["original_campaign"]["rows"] if row.get("pilot")]


def test_every_pilot_passed_is_the_only_case_the_gate_lets_through(package, tmp_path, monkeypatch):
    monkeypatch.setenv("WORK", str(tmp_path / "work"))
    pilots = pilot_ids(package)
    state(tmp_path / "work", package["original_campaign"]["name"], {row: {1: "PASS"} for row in pilots})
    payload = run_gate(package, tmp_path / "work", tmp_path / "out")
    assert payload["passed_count"] == payload["expected_count"] == len(pilots)
    assert payload["failed_count"] == payload["missing_count"] == 0 and payload["reason"] is None
    assert [pilot["row"] for pilot in payload["pilots"]] == pilots
    assert payload["state_dir"].endswith("campaign/%s" % package["original_campaign"]["name"])
    assert all(judged["ok"] for judged in judge_gate(package, tmp_path / "out")), judge_gate(package, tmp_path / "out")


def test_one_failed_pilot_stops_the_follow_up(package, tmp_path, monkeypatch):
    monkeypatch.setenv("WORK", str(tmp_path / "work"))
    pilots = pilot_ids(package)
    verdicts = {row: {1: "PASS"} for row in pilots}
    verdicts[pilots[-1]] = {1: "FAIL"}
    state(tmp_path / "work", package["original_campaign"]["name"], verdicts)
    payload = run_gate(package, tmp_path / "work", tmp_path / "out")
    assert payload["passed_count"] == len(pilots) - 1 and payload["failed_count"] == 1
    assert "passed in this WORK" in payload["reason"]
    judged = judge_gate(package, tmp_path / "out")
    assert not judged[0]["ok"] and judged[0]["value"] == len(pilots) - 1
    assert judged[1]["ok"], "the count of pilots is still right; it is passed_count that failed"


def test_an_original_that_never_ran_here_stops_the_follow_up(package, tmp_path, monkeypatch):
    monkeypatch.setenv("WORK", str(tmp_path / "work"))
    (tmp_path / "work").mkdir()
    payload = run_gate(package, tmp_path / "work", tmp_path / "out")
    assert payload["passed_count"] == 0 and payload["failed_count"] == 0
    assert payload["missing_count"] == payload["expected_count"] == len(pilot_ids(package))
    assert all(pilot["verdict"] == "NOT_RUN" for pilot in payload["pilots"])
    assert not judge_gate(package, tmp_path / "out")[0]["ok"]


def test_the_latest_attempt_of_a_pilot_is_the_one_that_counts(package, tmp_path, monkeypatch):
    """A pilot that failed and was fixed has passed; a pilot that passed and was re-run has not."""
    monkeypatch.setenv("WORK", str(tmp_path / "work"))
    pilots = pilot_ids(package)
    verdicts = {row: {1: "PASS"} for row in pilots}
    verdicts[pilots[0]] = {1: "FAIL", 2: "PASS"}
    state(tmp_path / "work", package["original_campaign"]["name"], verdicts)
    assert run_gate(package, tmp_path / "work", tmp_path / "out")["passed_count"] == len(pilots)

    verdicts[pilots[0]] = {1: "PASS", 2: "FAIL"}
    state(tmp_path / "second", package["original_campaign"]["name"], verdicts)
    payload = run_gate(package, tmp_path / "second", tmp_path / "out-2")
    assert payload["passed_count"] == len(pilots) - 1 and payload["failed_count"] == 1


def test_an_attempt_that_wrote_no_verdict_is_not_a_pass(package, tmp_path, monkeypatch):
    """A crash mid-pilot leaves start.json and nothing else; that is not evidence of anything."""
    monkeypatch.setenv("WORK", str(tmp_path / "work"))
    pilots = pilot_ids(package)
    verdicts = {row: {1: "PASS"} for row in pilots}
    verdicts[pilots[0]] = {1: None}
    state(tmp_path / "work", package["original_campaign"]["name"], verdicts)
    payload = run_gate(package, tmp_path / "work", tmp_path / "out")
    assert payload["passed_count"] == len(pilots) - 1
    assert payload["pilots"][0]["verdict"] == "NO_VERDICT"


def test_it_exits_zero_even_when_it_can_read_nothing_at_all(package, tmp_path, monkeypatch):
    """The bar decides, not the exit code -- so a missing campaign still leaves numbers on disk."""
    monkeypatch.setenv("WORK", str(tmp_path / "work"))
    out = tmp_path / "out"
    assert pilots_passed.main(["--campaign", str(tmp_path / "not-a-campaign.yaml"), "--out", str(out)]) == 0
    payload = json.loads((out / "pilots-passed.json").read_text())
    assert payload["expected_count"] == payload["passed_count"] == 0 and payload["reason"]
    assert not judge_gate(package, out)[0]["ok"] and not judge_gate(package, out)[1]["ok"]


def test_it_is_run_as_a_script_the_way_the_campaign_runs_it(package, tmp_path, monkeypatch):
    monkeypatch.setenv("WORK", str(tmp_path / "work"))
    state(tmp_path / "work", package["original_campaign"]["name"],
          {row: {1: "PASS"} for row in pilot_ids(package)})
    done = subprocess.run([sys.executable, str(KIT / "pilots_passed.py"), "--campaign", str(package["original"]),
                           "--out", str(tmp_path / "out")], capture_output=True, text=True,
                          env={**os.environ, "WORK": str(tmp_path / "work")})
    assert done.returncode == 0, done.stderr
    assert "pilots of %s passed" % package["original_campaign"]["name"] in done.stdout
    assert (tmp_path / "out" / "pilots-passed.json").is_file()


# -------------------------------------------------- 5. the original report tools, over five seeds
K3_SEEDS = (0, 1, 2, 3, 4)
K3_ARMS = ("none", "rehearse10", "rehearse30", "kl")
K4A_SEEDS = (42, 43, 44, 45, 46)
K4A_ARMS = ("feedback", "soft", "variation")
K0_SEEDS = (42, 43, 44)                       # the control ran three, and only three


def _bed_score(path: Path, bed: str, correct: int, n: int) -> None:
    path.mkdir(parents=True)
    (path / "bed-score.json").write_text(json.dumps(
        {"schema": "kit-bed-score.v1", "bed": bed, "n": n, "correct": correct,
         "accuracy": round(correct / n, 6), "machine": {"id": "m1", "deterministic": True}}))


def _forgetting(path: Path, correct=(90, 70, 80)) -> None:
    path.mkdir(parents=True)
    (path / "forgetting.json").write_text(json.dumps(
        {"schema": "kit-forgetting.v1",
         "panels": {name: {"correct": value, "n": 100}
                    for name, value in zip(("ifeval", "knowledge", "math"), correct)},
         "total_correct": sum(correct), "machine": {"id": "m1", "deterministic": True}}))


def _k3_point(root: Path, point: str, spider: int, gsm8k: int) -> None:
    _bed_score(root / "eval" / ("%s-spider-a1" % point), "spider", spider, 100)
    _bed_score(root / "eval" / ("%s-gsm8k-a1" % point), "gsm8k", gsm8k, 300)
    _forgetting(root / "forgetting" / ("%s-a1" % point))


@pytest.fixture()
def k3_tree(tmp_path):
    """The original's three seeds and the follow-up's two, in one $WORK/k3, named the same way."""
    root = tmp_path / "k3"
    _k3_point(root, "base", 40, 30)
    for seed in K3_SEEDS:
        _k3_point(root, "a-seed%d" % seed, 60 + seed, 31)
        for arm in K3_ARMS:
            _k3_point(root, "b-%s-seed%d" % (arm, seed), 45 + seed, 70 + seed)
    return root


def test_the_k3_report_reads_all_five_seeds_from_one_tree(k3_tree):
    report = k3_report.build(k3_tree)
    assert report["seeds_reported"] == 5 and report["arms_reported"] == len(K3_ARMS)
    assert report["comparable"] == 1
    for arm in K3_ARMS:
        assert sorted(report["arms"][arm]["seeds"]) == list(K3_SEEDS)
        assert report["arms"][arm]["job_a_change"]["n"] == 5
        assert report["arms"][arm]["job_a_change"]["sd"] is not None, "five seeds have a spread"
    assert sorted(report["after_stage_a"]) == ["a-seed%d" % seed for seed in K3_SEEDS]


def test_the_k3_follow_ups_report_row_passes_its_bars_on_that_tree(k3_tree, tmp_path):
    out = tmp_path / "report-5seeds-a1"
    assert k3_report.main(["--root", str(k3_tree), "--out", str(out)]) == 0
    campaign = runner.load_campaign(PACKAGES["k3"]["followup"])
    for bar in campaign["rows"][-1]["bars"]:
        judged = runner.judge_bar({**bar, "source": str(out / "k3-report.json")})
        assert judged["ok"], judged
    assert "| none | 4 |" in (out / "k3-report.md").read_text(), "the new seeds are in the table"


def test_the_k3_report_would_fail_the_seeds_bar_with_the_original_three(tmp_path):
    """The bar is what says the follow-up's seeds arrived; three seeds must not pass it."""
    root = tmp_path / "k3"
    _k3_point(root, "base", 40, 30)
    for seed in (0, 1, 2):
        _k3_point(root, "a-seed%d" % seed, 60, 31)
        for arm in K3_ARMS:
            _k3_point(root, "b-%s-seed%d" % (arm, seed), 45, 70)
    out = tmp_path / "report-a1"
    assert k3_report.main(["--root", str(root), "--out", str(out)]) == 0
    campaign = runner.load_campaign(PACKAGES["k3"]["followup"])
    bar = next(b for b in campaign["rows"][-1]["bars"] if b["key"] == "seeds_reported")
    judged = runner.judge_bar({**bar, "source": str(out / "k3-report.json")})
    assert not judged["ok"] and judged["value"] == 3


def _k4a_run(runs: Path, name: str, *, steps=40, accuracy=0.62) -> None:
    run = runs / name
    (run / ("hf-step%d" % steps)).mkdir(parents=True)
    (run / ("hf-step%d" % steps) / "config.json").write_text("{}")
    (run / "run-summary.json").write_text(json.dumps(
        {"schema": "kit-sdpo-run.v1", "name": name, "steps": steps, "returncode": 0, "merged": 1}))
    metrics = [{"step": 0, "data": {"val-core/tooluse/acc/mean@16": 0.5744}}]
    metrics += [{"step": step, "data": {"critic/score/mean": 0.4, "response_length/mean": 96.0,
                                        "self_distillation/success_group_fraction": 0.8,
                                        "self_distillation/empty_target_batch": 0.2,
                                        "actor/entropy": 0.21, "actor/grad_norm": 0.4,
                                        "perf/time_per_step": 55.0, "perf/max_memory_allocated_gb": 77.2}}
                for step in range(1, steps + 1)]
    metrics.append({"step": steps, "data": {"val-core/tooluse/acc/mean@16": accuracy}})
    (run / "metrics.jsonl").write_text("\n".join(json.dumps(record) for record in metrics) + "\n")


@pytest.fixture()
def k4a_tree(tmp_path):
    """Five seeds of every arm in one $WORK/runs, against the K0 control's three."""
    runs, forgetting = tmp_path / "runs", tmp_path / "k4a" / "forgetting"
    runs.mkdir()
    _forgetting(forgetting / "base-a1")
    for arm in K4A_ARMS:
        for seed in K4A_SEEDS:
            _k4a_run(runs, "%s-seed%d-a1" % (arm, seed), accuracy=0.6 + seed / 1000)
            _forgetting(forgetting / ("%s-seed%d-forget-a1" % (arm, seed)))
    control = {"schema": "kit-sdpo-report.v1", "runs": [
        {"name": "dose40-seed%d" % seed, "steps_completed": 40,
         "validations": [{"step": 0, "accuracy_logged": 0.5744}, {"step": 40, "accuracy_logged": 0.62}],
         "training": [{"step": step, "empty_target_batch": 0.63, "success_group_fraction": 0.37,
                       "response_tokens": 96.0, "entropy": 0.21, "score": 0.32} for step in range(1, 41)],
         "per_question": {"40": {"q0": 1.0, "q1": 0.0}}} for seed in K0_SEEDS]}
    k0 = tmp_path / "k0.json"
    k0.write_text(json.dumps(control))
    return {"runs": runs, "forgetting": forgetting, "k0": k0}


def test_the_k4a_report_reads_all_five_seeds_from_one_runs_directory(k4a_tree):
    report = k4a_report.build(k4a_tree["runs"], k4a_tree["forgetting"], k4a_tree["k0"])
    assert report["arms_reported"] == len(K4A_ARMS) and report["comparable"] == 1
    seeds = {seed for arm in report["arms"].values() for seed in arm["seeds"]}
    assert sorted(seeds) == list(K4A_SEEDS)
    for arm in K4A_ARMS:
        assert report["arms"][arm]["final"]["n"] == 5, "each arm's own score is now over five seeds"
    assert "over 5 seeds" in k4a_report.render(report)


def test_the_two_new_k4a_seeds_have_no_control_and_the_report_says_so(k4a_tree):
    """K0 ran dose40-seed42/43/44 only. The follow-up must not borrow another seed's control."""
    report = k4a_report.build(k4a_tree["runs"], k4a_tree["forgetting"], k4a_tree["k0"])
    assert report["control_runs"] == len(K0_SEEDS)
    for arm in K4A_ARMS:
        for seed in (45, 46):
            row = report["arms"][arm]["seeds"][seed]
            assert row["control"] is None and "change_vs_control" not in row
            assert row["by_k0_difficulty"] is None
        for seed in K0_SEEDS:
            assert report["arms"][arm]["seeds"][seed]["control"]["name"] == "dose40-seed%d" % seed
        assert report["arms"][arm]["change_vs_control"]["n"] == len(K0_SEEDS), \
            "the paired difference stays a comparison over three seeds"


def test_the_k4a_follow_ups_report_row_passes_its_bars_on_that_tree(k4a_tree, tmp_path):
    out = tmp_path / "report-5seeds-a1"
    assert k4a_report.main(["--runs", str(k4a_tree["runs"]), "--forgetting", str(k4a_tree["forgetting"]),
                            "--k0", str(k4a_tree["k0"]), "--out", str(out)]) == 0
    campaign = runner.load_campaign(PACKAGES["k4a"]["followup"])
    for bar in campaign["rows"][-1]["bars"]:
        judged = runner.judge_bar({**bar, "source": str(out / "k4a-report.json")})
        assert judged["ok"], judged
    text = (out / "k4a-report.md").read_text()
    assert "| feedback | 46 |" in text, "the new seeds are in the table"


# ------------------------------------------------------------------------- 6. the instructions
def test_the_readme_gives_commands_these_campaigns_have():
    text = "\n".join(" ".join(line.split()) for line in README.read_text().splitlines())
    for key in sorted(PACKAGES):
        name = PACKAGES[key]["followup"].name
        for action in ("plan", "prepare", "run", "status"):
            assert "runner.py %s $KIT/campaigns/%s" % (action, name) in text, (action, name)


def test_every_campaign_file_the_readme_names_is_one_this_kit_ships():
    """A file name that is one character off costs the partner a round trip to find out."""
    named = set(re.findall(r"campaigns/([A-Za-z0-9._-]+\.yaml)", README.read_text()))
    assert {PACKAGES[key]["followup"].name for key in PACKAGES} <= named
    for name in sorted(named):
        assert (KIT / "campaigns" / name).is_file(), name


def test_the_readme_names_the_one_condition_and_what_comes_back():
    text = README.read_text()
    for campaign in ("k3-replay", "k4a-stuck-problems"):
        assert "$WORK/campaign/%s/" % campaign in text, campaign
    for key in sorted(PACKAGES):
        out = next(row for row in yaml.safe_load(PACKAGES[key]["followup"].read_text())["rows"]
                   if row["id"] == "report")["env"]["OUT"]
        assert out.replace("{work}", "$WORK").replace("-a{attempt}", "-a1") in text, key
    assert "same machine" in text and "same `$WORK`" in text
