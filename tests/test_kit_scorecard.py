"""kit/scorecard.py: the four numbers for a sequence of jobs, and the K3 adapter that feeds it.

What this file is here to catch, in the order a reader of the scorecard would worry about it:

1. THE ARITHMETIC. The definitions were written before any number existed, so they are testable
   against a grid computed by hand: a three-job sequence whose average accuracy, backward transfer,
   forward transfer and general delta are worked out in the test itself and asserted digit for digit.
   A scorecard that quietly measured backward transfer against the UNTRAINED model instead of against
   the moment each job was learned would still look like a plausible negative number, which is exactly
   why the hand-computed case exists.
2. THE HOLES. Every one of the (T+1) x T cells must be named and readable. A cell carried over from a
   neighbouring stage, or a zero for a missing file, would turn an unfinished campaign into a
   publishable-looking table. Every refusal here is checked to leave NO output behind.
3. THE MACHINE RULE. Every number on the scorecard is a difference of two counts, and counts from two
   machines differ by up to 3 points before any training (receipts 204 and 209). A mixed scorecard is
   refused unless the departure is asked for, and then it is recorded.
4. THE ADAPTER. `from-k3` must find the real campaign tree's shape -- highest attempt, `base`,
   `a-seed<S>`, `b-<arm>-seed<S>` -- and must leave a hole as a hole, so that an unfinished tree is
   refused by `build` and not averaged over what happens to be there.
5. THE STANDALONE RULE. The partner runs this file on its own: it may import nothing of ours and
   nothing that needs a GPU.

Everything here builds its own fixtures; no test needs a GPU, a model or data from any machine.
"""
from __future__ import annotations

import ast
import importlib.util
import json
import statistics
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
KIT = ROOT / "kit"


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


scorecard = load("kit_scorecard", KIT / "scorecard.py")


# ============================================================================ fixtures
def write_score(directory: Path, correct: float, *, n: float = 100, machine: str | None = "m1",
                bed: str = "job", key_value: dict | None = None, tokens: int | None = None) -> Path:
    """One `bed-score.json` in the shape kit/eval_bed.py writes (`tokens` = output_tokens_total, optional)."""
    payload = {"schema": "kit-bed-score.v1", "bed": bed, "n": n, "correct": correct,
               "total_correct": correct, "accuracy": round(correct / n, 6), "incorrect_format": 0}
    if tokens is not None:
        payload["output_tokens_total"] = tokens
    if machine is not None:
        payload["machine"] = {"id": machine, "deterministic": True}
    payload.update(key_value or {})
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "bed-score.json").write_text(json.dumps(payload, indent=1, sort_keys=True), encoding="utf-8")
    return directory


def write_panels(directory: Path, scores: dict, *, machine: str | None = "m1", tokens: dict | None = None) -> Path:
    """One `forgetting.json` in the shape kit/score_forgetting.py writes (`tokens` = {panel: output_tokens_total})."""
    payload = {"schema": "kit-forgetting.v1", "total_correct": sum(scores.values()),
               "panels": {name: {"n": 100, "correct": correct, "parsed": 100, "per_member": {},
                                 "median_output_chars": 200, "empty_outputs": 0,
                                 **({"output_tokens_total": tokens[name]} if tokens and name in tokens else {})}
                          for name, correct in scores.items()}}
    if machine is not None:
        payload["machine"] = {"id": machine, "deterministic": True}
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "forgetting.json").write_text(json.dumps(payload, indent=1, sort_keys=True), encoding="utf-8")
    return directory


def a_run(root: Path, seed: int, jobs: list, grid: list, panels: list, *,
          machine: str | None = "m1", n=100) -> dict:
    """One seed's sequence: grid[i][j] is s[i][j], panels[i] is that stage's panel scores or None."""
    sizes = n if isinstance(n, dict) else {job: n for job in jobs}
    points = []
    for index, row in enumerate(grid):
        entry = {"scores": {}}
        for job, value in zip(jobs, row):
            where = write_score(root / ("seed%d-stage%d-%s" % (seed, index, job)), value, n=sizes[job],
                                machine=machine, bed=job)
            entry["scores"][job] = str(where)
        if panels[index] is not None:
            entry["forgetting"] = str(write_panels(root / ("seed%d-stage%d-panels" % (seed, index)),
                                                   panels[index], machine=machine))
        points.append(entry)
    return {"seed": seed, "untrained": points[0],
            "stages": [dict(point, job=job) for point, job in zip(points[1:], jobs)]}


def a_manifest(path: Path, runs: list, *, name: str = "seq", schema: str = "kit-sequence.v1") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schema": schema, "name": name, "runs": runs}, indent=1), encoding="utf-8")
    return path


# The hand-computed example. Three jobs a, b, c, each scored on 100 held-out questions.
#
#   after stage        a    b    c
#   0 (untrained)     10   20   30
#   1 (learned a)     60   22   28
#   2 (learned b)     45   70   35
#   3 (learned c)     40   55   80
#
# average accuracy  = mean(40, 55, 80)                       = 175 / 3 = 58.3333
# backward transfer = mean(s[3][a] - s[1][a], s[3][b] - s[2][b])
#                   = mean(40 - 60, 55 - 70) = mean(-20, -15)          = -17.5
# forward transfer  = mean(s[1][b] - s[0][b], s[2][c] - s[0][c])
#                   = mean(22 - 20, 35 - 30) = mean(2, 5)              = +3.5
# general delta     = (89, 72, 85) - (91, 72, 88)            = gsm8k -2, mmlu 0, ifeval -3
HAND_JOBS = ["a", "b", "c"]
HAND_GRID = [[10, 20, 30], [60, 22, 28], [45, 70, 35], [40, 55, 80]]
HAND_BEFORE = {"gsm8k": 91, "mmlu": 72, "ifeval": 88}
HAND_AFTER = {"gsm8k": 89, "mmlu": 72, "ifeval": 85}


@pytest.fixture()
def hand_manifest(tmp_path) -> Path:
    run = a_run(tmp_path / "tree", 0, HAND_JOBS, HAND_GRID, [HAND_BEFORE, None, None, HAND_AFTER])
    return a_manifest(tmp_path / "seq.json", [run], name="hand")


# ============================================================================ 1. the arithmetic
def test_the_four_numbers_are_what_the_definitions_say_they_are(hand_manifest):
    card = scorecard.build(hand_manifest)
    seed = card["per_seed"]["0"]
    assert seed["average_accuracy"] == 58.3333
    assert seed["backward_transfer"] == -17.5
    assert seed["forward_transfer"] == 3.5
    assert seed["general_delta"] == {"gsm8k": -2.0, "ifeval": -3.0, "mmlu": 0.0}
    assert card["jobs"] == ["a", "b", "c"] and card["stages"] == 3
    assert card["cells_read"] == 12, "a three-job sequence is a 4 x 3 grid"
    assert seed["matrix"] == {"a": [10.0, 60.0, 45.0, 40.0], "b": [20.0, 22.0, 70.0, 55.0],
                              "c": [30.0, 28.0, 35.0, 80.0]}


def test_backward_transfer_is_measured_from_the_moment_each_job_was_learned(hand_manifest):
    """Against the untrained model it would be +30 over the same grid: a gain where there is damage."""
    card = scorecard.build(hand_manifest)
    seed = card["per_seed"]["0"]
    assert seed["per_job_backward"] == {"a": -20.0, "b": -15.0}, "job c is the last job: it is not in the mean"
    against_untrained = statistics.fmean([40 - 10, 55 - 20])
    assert seed["backward_transfer"] != against_untrained
    assert seed["per_job_forward"] == {"b": 2.0, "c": 5.0}, "job a is the first job: it has no forward term"


def test_every_seed_is_reported_and_averaged_with_its_spread(tmp_path):
    """Two seeds whose numbers are 10 and 20: the mean is 15 and the sample spread is 10 / sqrt(2)."""
    jobs = ["a", "b"]
    runs = [a_run(tmp_path / "tree", 0, jobs, [[0, 0], [20, 0], [10, 10]], [{"p": 50}, None, {"p": 49}]),
            a_run(tmp_path / "tree", 1, jobs, [[0, 0], [40, 0], [20, 20]], [{"p": 50}, None, {"p": 47}])]
    card = scorecard.build(a_manifest(tmp_path / "seq.json", runs))
    assert card["seeds"] == [0, 1] and card["seeds_reported"] == 2
    assert card["per_seed"]["0"]["average_accuracy"] == 10.0
    assert card["per_seed"]["1"]["average_accuracy"] == 20.0
    assert card["average"]["average_accuracy"] == {"n": 2, "mean": 15.0, "sd": 7.0711}
    assert card["average"]["backward_transfer"] == {"n": 2, "mean": -15.0, "sd": 7.0711}
    assert card["average"]["forward_transfer"] == {"n": 2, "mean": 0.0, "sd": 0.0}
    assert card["average"]["general_delta"] == {"p": {"n": 2, "mean": -2.0, "sd": 1.4142}}
    assert card["per_seed"]["0"]["general_delta"] == {"p": -1.0}
    assert card["per_seed"]["1"]["general_delta"] == {"p": -3.0}


def test_one_seed_has_a_mean_but_no_spread(hand_manifest):
    card = scorecard.build(hand_manifest)
    assert card["average"]["average_accuracy"] == {"n": 1, "mean": 58.3333, "sd": None}
    assert card["average"]["general_delta"]["gsm8k"] == {"n": 1, "mean": -2.0, "sd": None}


def test_a_one_job_sequence_has_no_transfer_to_report(tmp_path):
    """With one job there is no j < T and no j > 1: the two means are null, not zero."""
    run = a_run(tmp_path / "tree", 0, ["a"], [[10], [60]], [{"p": 50}, {"p": 48}])
    card = scorecard.build(a_manifest(tmp_path / "seq.json", [run]))
    seed = card["per_seed"]["0"]
    assert seed["average_accuracy"] == 60.0
    assert seed["backward_transfer"] is None and seed["forward_transfer"] is None
    assert seed["general_delta"] == {"p": -2.0}
    assert card["average"]["backward_transfer"] == {"n": 0, "mean": None, "sd": None}
    assert "not defined for a sequence of one job" in scorecard.render(card)


def test_the_key_being_read_is_the_one_asked_for(tmp_path):
    """Jobs with held-out sets of different sizes are averaged on `accuracy`, not on counts."""
    jobs = ["small", "big"]
    run = a_run(tmp_path / "tree", 0, jobs, [[10, 30], [80, 30], [40, 240]], [{"p": 50}, None, {"p": 50}],
                n={"small": 100, "big": 300})
    manifest = a_manifest(tmp_path / "seq.json", [run])
    counts = scorecard.build(manifest)
    assert counts["per_seed"]["0"]["average_accuracy"] == 140.0            # mean(40, 240)
    assert counts["job_questions"] == {"small": 100, "big": 300}
    assert "different sizes" in scorecard.render(counts)
    shares = scorecard.build(manifest, key="accuracy")
    assert shares["key"] == "accuracy"
    assert shares["per_seed"]["0"]["average_accuracy"] == 0.6             # mean(40/100, 240/300)
    assert shares["per_seed"]["0"]["backward_transfer"] == -0.4           # 0.4 - 0.8
    assert "different sizes" not in scorecard.render(shares)


def test_what_each_job_gained_when_it_was_learned_is_reported_too(hand_manifest):
    """Not part of the scorecard, but a sequence where nothing was learned has nothing to forget."""
    card = scorecard.build(hand_manifest)
    assert card["per_seed"]["0"]["learned"] == {"a": 50.0, "b": 48.0, "c": 45.0}


# ============================================================================ 2. the holes
def test_a_missing_cell_is_refused_and_nothing_is_written(tmp_path):
    run = a_run(tmp_path / "tree", 0, HAND_JOBS, HAND_GRID, [HAND_BEFORE, None, None, HAND_AFTER])
    del run["stages"][1]["scores"]["c"]
    manifest = a_manifest(tmp_path / "seq.json", [run])
    with pytest.raises(scorecard.ScorecardError, match="has no scoring of job 'c'"):
        scorecard.build(manifest)
    with pytest.raises(SystemExit, match="missing cell is a refusal"):
        scorecard.main(["build", "--manifest", str(manifest), "--out", str(tmp_path / "out")])
    assert not (tmp_path / "out").exists()


def test_the_missing_cell_is_named_by_seed_stage_and_job(tmp_path):
    run = a_run(tmp_path / "tree", 4, HAND_JOBS, HAND_GRID, [HAND_BEFORE, None, None, HAND_AFTER])
    del run["untrained"]["scores"]["b"]
    with pytest.raises(scorecard.ScorecardError, match=r"seed 4 stage 0 \(untrained\) has no scoring of job 'b'"):
        scorecard.build(a_manifest(tmp_path / "seq.json", [run]))
    run = a_run(tmp_path / "tree2", 4, HAND_JOBS, HAND_GRID, [HAND_BEFORE, None, None, HAND_AFTER])
    del run["stages"][2]["scores"]["a"]
    with pytest.raises(scorecard.ScorecardError, match=r"seed 4 stage 3 \(c\) has no scoring of job 'a'"):
        scorecard.build(a_manifest(tmp_path / "seq2.json", [run]))


def test_the_panels_before_and_after_the_sequence_are_both_required(tmp_path):
    for drop, message in ((0, r"stage 0 \(untrained\)"), (3, r"stage 3 \(c\)")):
        panels = [HAND_BEFORE, None, None, HAND_AFTER]
        panels[drop] = None
        run = a_run(tmp_path / ("tree%d" % drop), 0, HAND_JOBS, HAND_GRID, panels)
        with pytest.raises(scorecard.ScorecardError, match="names no forgetting.json for %s" % message):
            scorecard.build(a_manifest(tmp_path / ("seq%d.json" % drop), [run]))


def test_a_cell_that_cannot_be_read_is_named_rather_than_traced(tmp_path):
    def manifest_with(job_path) -> Path:
        run = a_run(tmp_path / "tree", 0, HAND_JOBS, HAND_GRID, [HAND_BEFORE, None, None, HAND_AFTER])
        run["stages"][0]["scores"]["b"] = str(job_path)
        return a_manifest(tmp_path / "seq.json", [run])

    (tmp_path / "bare").mkdir()
    (tmp_path / "broken").mkdir()
    (tmp_path / "broken" / "bed-score.json").write_text("{not json}", encoding="utf-8")
    (tmp_path / "list").mkdir()
    (tmp_path / "list" / "bed-score.json").write_text(json.dumps([1, 2]), encoding="utf-8")
    for where, message in ((tmp_path / "nowhere", "no such result file"), (tmp_path / "bare", "holds none of"),
                           (tmp_path / "broken", "is not JSON"), (tmp_path / "list", "not a JSON object")):
        (tmp_path / "seq.json").unlink(missing_ok=True)
        with pytest.raises(scorecard.ScorecardError, match=message):
            scorecard.build(manifest_with(where))


def test_a_missing_key_and_a_non_number_are_each_refused(tmp_path):
    run = a_run(tmp_path / "tree", 0, HAND_JOBS, HAND_GRID, [HAND_BEFORE, None, None, HAND_AFTER])
    manifest = a_manifest(tmp_path / "seq.json", [run])
    with pytest.raises(scorecard.ScorecardError, match="no top-level key 'f1'"):
        scorecard.build(manifest, key="f1")
    write_score(tmp_path / "text", 0, key_value={"correct": "forty"})
    run["stages"][0]["scores"]["a"] = str(tmp_path / "text")
    with pytest.raises(scorecard.ScorecardError, match="not a number"):
        scorecard.build(a_manifest(tmp_path / "seq2.json", [run]))


def test_a_forgetting_file_without_panels_is_refused(tmp_path):
    run = a_run(tmp_path / "tree", 0, ["a"], [[10], [60]], [{"p": 50}, {"p": 48}])
    (tmp_path / "flat").mkdir()
    (tmp_path / "flat" / "forgetting.json").write_text(json.dumps({"total_correct": 200}), encoding="utf-8")
    run["stages"][0]["forgetting"] = str(tmp_path / "flat")
    with pytest.raises(scorecard.ScorecardError, match="carries no `panels` object"):
        scorecard.build(a_manifest(tmp_path / "seq.json", [run]))


def test_panels_that_do_not_cover_the_same_questions_are_refused(tmp_path):
    run = a_run(tmp_path / "tree", 0, ["a"], [[10], [60]], [{"p": 50, "q": 50}, {"p": 48}])
    with pytest.raises(scorecard.ScorecardError, match="was scored on panels"):
        scorecard.build(a_manifest(tmp_path / "seq.json", [run]))
    runs = [a_run(tmp_path / "t2", 0, ["a"], [[10], [60]], [{"p": 50}, {"p": 48}]),
            a_run(tmp_path / "t2", 1, ["a"], [[10], [60]], [{"q": 50}, {"q": 48}])]
    with pytest.raises(scorecard.ScorecardError, match="seeds cannot be averaged"):
        scorecard.build(a_manifest(tmp_path / "seq2.json", runs))


def test_a_job_whose_held_out_set_changed_size_is_refused(tmp_path):
    """100 correct of 100 and 100 of 300 are not the same ability, and their difference is not forgetting."""
    run = a_run(tmp_path / "tree", 0, ["a", "b"], [[10, 10], [50, 10], [40, 60]], [{"p": 50}, None, {"p": 50}])
    write_score(tmp_path / "tree" / "seed0-stage2-a", 40, n=300, bed="a")
    with pytest.raises(scorecard.ScorecardError, match="not the same held-out set"):
        scorecard.build(a_manifest(tmp_path / "seq.json", [run]))
    runs = [a_run(tmp_path / "t2", 0, ["a"], [[10], [60]], [{"p": 50}, {"p": 48}], n=100),
            a_run(tmp_path / "t2", 1, ["a"], [[10], [60]], [{"p": 50}, {"p": 48}], n=200)]
    with pytest.raises(scorecard.ScorecardError, match="those seeds cannot be averaged"):
        scorecard.build(a_manifest(tmp_path / "seq2.json", runs))


@pytest.mark.parametrize("wreck, message", [
    (lambda m: m.update(schema="kit-scorecard.v1"), "is not a sequence manifest"),
    (lambda m: m.update(runs=[]), "lists no runs"),
    (lambda m: m.update(runs="all of them"), "lists no runs"),
    (lambda m: m["runs"][0].pop("seed"), "has no integer `seed`"),
    (lambda m: m["runs"][0].update(seed="zero"), "has no integer `seed`"),
    (lambda m: m["runs"][0].update(stages=[]), "lists no stages"),
    (lambda m: m["runs"][0]["stages"][0].pop("job"), "names no `job`"),
    (lambda m: m["runs"][0].pop("untrained"), "has no `untrained` scorings"),
    (lambda m: m["runs"][0]["stages"][0].pop("scores"), "names no `scores`"),
])
def test_a_manifest_that_is_not_one_sequence_is_refused(tmp_path, wreck, message):
    run = a_run(tmp_path / "tree", 0, HAND_JOBS, HAND_GRID, [HAND_BEFORE, None, None, HAND_AFTER])
    manifest = {"schema": "kit-sequence.v1", "name": "seq", "runs": [run]}
    wreck(manifest)
    path = tmp_path / "seq.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(scorecard.ScorecardError, match=message):
        scorecard.build(path)


def test_two_seeds_that_learned_different_sequences_are_refused(tmp_path):
    runs = [a_run(tmp_path / "tree", 0, ["a", "b"], [[1, 1], [2, 1], [2, 2]], [{"p": 50}, None, {"p": 50}]),
            a_run(tmp_path / "tree", 1, ["b", "a"], [[1, 1], [1, 2], [2, 2]], [{"p": 50}, None, {"p": 50}])]
    with pytest.raises(scorecard.ScorecardError, match="one scorecard is one sequence"):
        scorecard.build(a_manifest(tmp_path / "seq.json", runs))


def test_a_repeated_seed_and_a_repeated_job_are_refused(tmp_path):
    run = a_run(tmp_path / "tree", 0, ["a", "b"], [[1, 1], [2, 1], [2, 2]], [{"p": 50}, None, {"p": 50}])
    with pytest.raises(scorecard.ScorecardError, match="seed 0 appears twice"):
        scorecard.build(a_manifest(tmp_path / "seq.json", [run, run]))
    twice = a_run(tmp_path / "t2", 0, ["a", "b"], [[1, 1], [2, 1], [2, 2]], [{"p": 50}, None, {"p": 50}])
    twice["stages"][1]["job"] = "a"
    with pytest.raises(scorecard.ScorecardError, match="learns 'a' twice"):
        scorecard.build(a_manifest(tmp_path / "seq2.json", [twice]))


def test_a_manifest_that_is_not_json_or_not_there_is_named(tmp_path):
    (tmp_path / "bad.json").write_text("{", encoding="utf-8")
    with pytest.raises(scorecard.ScorecardError, match="is not JSON"):
        scorecard.build(tmp_path / "bad.json")
    with pytest.raises(scorecard.ScorecardError, match="cannot read the manifest"):
        scorecard.build(tmp_path / "nowhere.json")
    (tmp_path / "list.json").write_text(json.dumps([1]), encoding="utf-8")
    with pytest.raises(scorecard.ScorecardError, match="is not a JSON object"):
        scorecard.build(tmp_path / "list.json")


def test_a_relative_path_is_read_against_the_manifests_own_directory(tmp_path):
    """The partner sends a folder back; the manifest inside it must still point at its own scorings."""
    run = a_run(tmp_path / "pack" / "tree", 0, ["a"], [[10], [60]], [{"p": 50}, {"p": 48}])
    for point in [run["untrained"]] + run["stages"]:
        point["scores"] = {job: str(Path(where).relative_to(tmp_path / "pack"))
                           for job, where in point["scores"].items()}
        if "forgetting" in point:
            point["forgetting"] = str(Path(point["forgetting"]).relative_to(tmp_path / "pack"))
    card = scorecard.build(a_manifest(tmp_path / "pack" / "seq.json", [run]))
    assert card["per_seed"]["0"]["average_accuracy"] == 60.0


# ============================================================================ 3. the machine rule
def test_scorings_from_two_machines_are_refused_unless_the_departure_is_asked_for(tmp_path):
    run = a_run(tmp_path / "tree", 0, HAND_JOBS, HAND_GRID, [HAND_BEFORE, None, None, HAND_AFTER])
    write_score(tmp_path / "tree" / "seed0-stage3-a", 40, machine="m2", bed="a")
    manifest = a_manifest(tmp_path / "seq.json", [run])
    with pytest.raises(scorecard.ScorecardError, match="2 different machine-and-mode fingerprints"):
        scorecard.build(manifest)
    card = scorecard.build(manifest, allow_different_machines=True)
    assert card["comparable"] == 0 and card["different_machines_allowed"] == 1
    assert card["machine_ids"] == ["m1", "m2"]
    assert card["per_seed"]["0"]["backward_transfer"] == -17.5
    assert "do not share one machine-and-mode fingerprint" in scorecard.render(card)


def test_a_scoring_with_no_fingerprint_at_all_is_not_comparable(tmp_path):
    """A `score` re-scoring of saved answers carries no machine; it cannot vouch for the comparison."""
    run = a_run(tmp_path / "tree", 0, ["a"], [[10], [60]], [{"p": 50}, {"p": 48}], machine=None)
    manifest = a_manifest(tmp_path / "seq.json", [run])
    with pytest.raises(scorecard.ScorecardError, match="1 different machine-and-mode fingerprints"):
        scorecard.build(manifest)
    card = scorecard.build(manifest, allow_different_machines=True)
    assert card["machine_ids"] == [None] and card["comparable"] == 0


def test_one_machine_throughout_is_comparable(hand_manifest):
    card = scorecard.build(hand_manifest)
    assert card["comparable"] == 1 and card["machine_ids"] == ["m1"]
    assert card["different_machines_allowed"] == 0
    assert "do not share one machine" not in scorecard.render(card)


# ============================================================================ 4. the two files it writes
def test_the_command_line_writes_both_files_and_never_overwrites(tmp_path, hand_manifest, capsys):
    out = tmp_path / "card"
    assert scorecard.main(["build", "--manifest", str(hand_manifest), "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    card = json.loads((out / "scorecard.json").read_text())
    assert card["schema"] == "kit-scorecard.v1" and card["name"] == "hand"
    assert card == scorecard.build(hand_manifest)
    text = (out / "scorecard.md").read_text()
    assert text == scorecard.render(card) and text in printed
    assert "| average accuracy | 58.33 | 58.33 | - |" in text, "a level carries no sign; a difference does"
    assert "| backward transfer | -17.50 | -17.50 | - |" in text
    assert "| forward transfer | +3.50 | +3.50 | - |" in text
    assert "| general delta: ifeval | -3 | -3.00 | - |" in text
    before = (out / "scorecard.json").read_bytes()
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        scorecard.main(["build", "--manifest", str(hand_manifest), "--out", str(out)])
    assert (out / "scorecard.json").read_bytes() == before


def test_the_readout_shows_the_grid_it_read(hand_manifest):
    text = scorecard.render(scorecard.build(hand_manifest))
    assert "| stage 0 (untrained) | 10 | 20 | 30 |" in text
    assert "| stage 3 (c) | 40 | 55 | 80 |" in text
    assert "12 cells read from" in text


# ============================================================================ 5. the K3 adapter
def a_k3_tree(root: Path, values: dict, *, panels: dict | None = None, machine: str = "m1") -> Path:
    """A K3 campaign tree: eval/<point>-<bed>-aN/bed-score.json, forgetting/<point>-aN/forgetting.json."""
    for point, (spider, gsm8k, attempt) in values.items():
        write_score(root / "eval" / ("%s-spider-a%d" % (point, attempt)), spider, n=100, machine=machine,
                    bed="spider")
        write_score(root / "eval" / ("%s-gsm8k-a%d" % (point, attempt)), gsm8k, n=300, machine=machine,
                    bed="gsm8k")
        if panels and point in panels:
            write_panels(root / "forgetting" / ("%s-a%d" % (point, attempt)), panels[point], machine=machine)
    return root


K3_VALUES = {"base": (80, 60, 1), "a-seed0": (92, 58, 1), "b-none-seed0": (61, 190, 1),
             "a-seed1": (90, 57, 1), "b-none-seed1": (66, 186, 1),
             "b-rehearse10-seed0": (84, 181, 1), "b-rehearse10-seed1": (86, 179, 1)}
K3_PANELS = {"base": {"gsm8k": 91, "mmlu": 72, "ifeval": 88},
             "a-seed0": {"gsm8k": 90, "mmlu": 72, "ifeval": 86},
             "b-none-seed0": {"gsm8k": 93, "mmlu": 70, "ifeval": 84},
             "a-seed1": {"gsm8k": 91, "mmlu": 71, "ifeval": 87},
             "b-none-seed1": {"gsm8k": 92, "mmlu": 71, "ifeval": 83},
             "b-rehearse10-seed0": {"gsm8k": 92, "mmlu": 71, "ifeval": 86},
             "b-rehearse10-seed1": {"gsm8k": 91, "mmlu": 72, "ifeval": 85}}


def test_the_adapter_turns_a_k3_arm_into_a_two_job_sequence(tmp_path):
    root = a_k3_tree(tmp_path / "k3", K3_VALUES, panels=K3_PANELS)
    manifest = scorecard.from_k3(root, "none")
    assert manifest["schema"] == "kit-sequence.v1" and manifest["name"] == "k3-none"
    assert manifest["jobs"] == ["spider", "gsm8k"] and manifest["seeds_found"] == [0, 1]
    assert manifest["arms_found"] == ["none", "rehearse10"]
    assert manifest["source"] == {"campaign": "k3-replay", "root": str(root.resolve()), "arm": "none"}
    assert [stage["job"] for stage in manifest["runs"][0]["stages"]] == ["spider", "gsm8k"]
    assert scorecard.missing_cells(manifest) == []


def test_the_scorecard_of_a_k3_arm_is_the_numbers_in_the_tree(tmp_path):
    """Seed 0: SQL 80 -> 92 -> 61, maths 60 -> 58 -> 190; panels 91/72/88 -> 93/70/84."""
    root = a_k3_tree(tmp_path / "k3", K3_VALUES, panels=K3_PANELS)
    path = tmp_path / "seq-none.json"
    path.write_text(json.dumps(scorecard.from_k3(root, "none")), encoding="utf-8")
    card = scorecard.build(path)
    seed = card["per_seed"]["0"]
    assert seed["matrix"] == {"spider": [80.0, 92.0, 61.0], "gsm8k": [60.0, 58.0, 190.0]}
    assert seed["average_accuracy"] == round((61 + 190) / 2, 4)
    assert seed["backward_transfer"] == -31.0                       # 61 - 92, and only Spider is an earlier job
    assert seed["forward_transfer"] == -2.0                         # maths after stage A, 58, minus 60
    assert seed["general_delta"] == {"gsm8k": 2.0, "mmlu": -2.0, "ifeval": -4.0}
    assert card["average"]["backward_transfer"] == {"n": 2, "mean": -27.5, "sd": 4.9497}   # -31 and -24
    assert card["seeds"] == [0, 1] and card["comparable"] == 1
    assert "different sizes" in scorecard.render(card), "Spider is scored on 100 questions and GSM8K on 300"


def test_the_adapter_reads_the_highest_attempt_of_every_scoring(tmp_path):
    root = a_k3_tree(tmp_path / "k3", K3_VALUES, panels=K3_PANELS)
    write_score(root / "eval" / "a-seed0-spider-a2", 95, n=100, bed="spider")
    write_score(root / "eval" / "a-seed0-spider-a10", 97, n=100, bed="spider")
    write_panels(root / "forgetting" / "b-none-seed0-a3", {"gsm8k": 1, "mmlu": 2, "ifeval": 3})
    path = tmp_path / "seq.json"
    path.write_text(json.dumps(scorecard.from_k3(root, "none")), encoding="utf-8")
    card = scorecard.build(path)
    assert card["per_seed"]["0"]["matrix"]["spider"] == [80.0, 97.0, 61.0], "a10 is a later attempt than a2"
    assert card["per_seed"]["0"]["general_delta"] == {"gsm8k": -90.0, "mmlu": -70.0, "ifeval": -85.0}


def test_the_adapter_leaves_a_hole_as_a_hole(tmp_path):
    """An unfinished tree must be refused by `build`, not averaged over whatever happens to be there."""
    values = dict(K3_VALUES)
    del values["a-seed1"]
    root = a_k3_tree(tmp_path / "k3", values, panels=K3_PANELS)
    manifest = scorecard.from_k3(root, "none")
    gaps = scorecard.missing_cells(manifest)
    assert gaps == ["seed 1 stage 1 (spider): spider", "seed 1 stage 1 (spider): gsm8k"]
    path = tmp_path / "seq.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(scorecard.ScorecardError, match=r"seed 1 stage 1 \(spider\) has no scoring of job 'spider'"):
        scorecard.build(path)


def test_the_adapter_refuses_an_arm_or_a_tree_that_is_not_there(tmp_path):
    root = a_k3_tree(tmp_path / "k3", K3_VALUES, panels=K3_PANELS)
    with pytest.raises(scorecard.ScorecardError, match="no stage-B arm 'kl'"):
        scorecard.from_k3(root, "kl")
    with pytest.raises(scorecard.ScorecardError, match="no scorings under"):
        scorecard.from_k3(tmp_path / "empty", "none")
    bare = a_k3_tree(tmp_path / "k3-no-base", {k: v for k, v in K3_VALUES.items() if k != "base"},
                     panels=K3_PANELS)
    with pytest.raises(scorecard.ScorecardError, match="no `base` point"):
        scorecard.from_k3(bare, "none")


def test_the_adapter_ignores_directories_the_campaign_did_not_write(tmp_path):
    root = a_k3_tree(tmp_path / "k3", K3_VALUES, panels=K3_PANELS)
    write_score(root / "eval" / "scratch-spider-a1", 7, bed="spider")
    write_score(root / "eval" / "b-none-seed0-spider", 7, bed="spider")       # no attempt suffix
    (root / "eval" / "notes").mkdir()
    manifest = scorecard.from_k3(root, "none")
    assert manifest["arms_found"] == ["none", "rehearse10"]
    path = tmp_path / "seq.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    assert scorecard.build(path)["per_seed"]["0"]["matrix"]["spider"][2] == 61.0


def test_the_adapter_command_line_writes_a_manifest_and_never_overwrites(tmp_path, capsys):
    root = a_k3_tree(tmp_path / "k3", K3_VALUES, panels=K3_PANELS)
    out = tmp_path / "manifests" / "seq-none.json"
    assert scorecard.main(["from-k3", "--root", str(root), "--arm", "none", "--out", str(out)]) == 0
    assert "arm none: 2 seeds (0, 1), 2 jobs" in capsys.readouterr().out
    assert json.loads(out.read_text())["name"] == "k3-none"
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        scorecard.main(["from-k3", "--root", str(root), "--arm", "none", "--out", str(out)])
    assert scorecard.main(["from-k3", "--root", str(root), "--arm", "rehearse10", "--name", "k3-r10",
                           "--out", str(tmp_path / "r10.json")]) == 0
    assert json.loads((tmp_path / "r10.json").read_text())["name"] == "k3-r10"


def test_the_adapter_command_line_says_which_cells_are_not_there_yet(tmp_path, capsys):
    values = {k: v for k, v in K3_VALUES.items() if k != "a-seed1"}
    root = a_k3_tree(tmp_path / "k3", values, panels=K3_PANELS)
    assert scorecard.main(["from-k3", "--root", str(root), "--arm", "none",
                           "--out", str(tmp_path / "seq.json")]) == 0
    printed = capsys.readouterr().out
    assert "2 cells are not in the tree yet" in printed
    assert "seed 1 stage 1 (spider): spider" in printed


# ============================================================================ 6. the standalone rule
def test_scorecard_imports_nothing_of_ours_and_nothing_that_needs_a_gpu():
    tree = ast.parse((KIT / "scorecard.py").read_text(encoding="utf-8"), str(KIT / "scorecard.py"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".")[0])
    assert not imported & {"continual", "sdft", "kit", "modal", "torch", "vllm", "transformers"}, imported
    # importlib loads kit/density.py from beside the file (the density column, plan 4c); still nothing of ours by name
    assert imported <= {"argparse", "json", "re", "statistics", "sys", "pathlib", "__future__", "importlib"}, imported


def function_ast(path: Path, name: str) -> str:
    """One function's code, with its docstring dropped, as a comparable string."""
    tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
    found = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name]
    assert len(found) == 1, "%s defines %d functions named %s" % (path.name, len(found), name)
    node = found[0]
    body = node.body[1:] if (isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant)
                             and isinstance(node.body[0].value.value, str)) else node.body
    return ast.dump(ast.Module(body=body, type_ignores=[]))


def test_the_spread_and_the_tree_reading_are_the_same_as_the_k3_readouts():
    """Duplicated on purpose, so each file stands alone; this test is what keeps the duplicates equal."""
    k3_report = load("kit_k3_report_t", KIT / "k3_report.py")
    assert function_ast(KIT / "scorecard.py", "spread") == function_ast(KIT / "k3_report.py", "spread")
    assert scorecard.POINT.pattern == k3_report.POINT.pattern
    assert scorecard.ATTEMPT.pattern == k3_report.ATTEMPT.pattern
    assert scorecard.spread([1.0, 2.0, 6.0]) == k3_report.spread([1.0, 2.0, 6.0])


# ---- what a correct answer costs (plan 4c): the density column ------------------------------------------

def density_manifest(tmp_path: Path, *, tokens: bool) -> Path:
    """A one-seed, two-job sequence: job a doubles its tokens per correct answer, job b stays flat; panel p
    goes over the bar and panel q stays within. Without `tokens`, no scoring carries token counts."""
    root = tmp_path / "d"
    tok = (lambda v: v) if tokens else (lambda v: None)
    stages = []
    # untrained: a 20 correct / 2,000 tokens (100 per correct); b 30 / 3,000 (100); p 50 / 5,000 (100); q 50 / 5,000
    write_score(root / "s0-a", 20, bed="a", tokens=tok(2000)); write_score(root / "s0-b", 30, bed="b", tokens=tok(3000))
    write_panels(root / "s0-f", {"p": 50, "q": 50}, tokens={"p": 5000, "q": 5000} if tokens else None)
    # after a: a 40 / 8,000 (200: x2, over); b 30 / 3,000
    write_score(root / "s1-a", 40, bed="a", tokens=tok(8000)); write_score(root / "s1-b", 30, bed="b", tokens=tok(3000))
    write_panels(root / "s1-f", {"p": 50, "q": 50}, tokens={"p": 6000, "q": 5000} if tokens else None)
    # after b: a 40 / 8,000 (x2, over); b 60 / 6,000 (100: x1, within); p 50 / 8,000 (x1.6, over); q 50 / 6,000 (x1.2)
    write_score(root / "s2-a", 40, bed="a", tokens=tok(8000)); write_score(root / "s2-b", 60, bed="b", tokens=tok(6000))
    write_panels(root / "s2-f", {"p": 50, "q": 50}, tokens={"p": 8000, "q": 6000} if tokens else None)
    for i, job in enumerate((None, "a", "b")):
        stages.append({"scores": {"a": "d/s%d-a" % i, "b": "d/s%d-b" % i}, "forgetting": "d/s%d-f" % i,
                       **({"job": job} if job else {})})
    return a_manifest(tmp_path / "manifest.json", [{"seed": 0, "untrained": stages[0], "stages": stages[1:]}])


def test_the_density_column_reads_tokens_per_correct_answer_and_judges_it_against_the_bar(tmp_path):
    card = scorecard.build(density_manifest(tmp_path, tokens=True))
    d = card["per_seed"]["0"]["density"]
    assert d["bar"] == 1.5
    assert d["jobs"]["a"]["untrained"] == 100 and d["jobs"]["a"]["trained"] == 200 and d["jobs"]["a"]["verdict"] == "over bar"
    assert d["jobs"]["b"]["ratio"] == 1 and d["jobs"]["b"]["verdict"] == "within bar"
    assert d["panels"]["p"]["ratio"] == 1.6 and d["panels"]["p"]["verdict"] == "over bar"
    assert d["panels"]["q"]["ratio"] == 1.2 and d["panels"]["q"]["verdict"] == "within bar"
    assert d["over_bar"] == ["a", "panel p"] and d["known"] == d["cells"] == 10
    text = scorecard.render(card)
    assert "## What a correct answer costs" in text and "| 0 | 100 to 200 (x2.00, over) | 100 to 100 (x1.00) | 100 to 160 (x1.60, over) | 100 to 120 (x1.20) | a, panel p |" in text
    assert "carried no token counts" not in text


def test_a_scoring_without_token_counts_gives_a_dash_and_never_a_refusal(tmp_path):
    with_tokens = scorecard.build(density_manifest(tmp_path / "w", tokens=True))
    without = scorecard.build(density_manifest(tmp_path / "wo", tokens=False))
    for name in ("average_accuracy", "backward_transfer", "forward_transfer", "general_delta"):
        assert with_tokens["per_seed"]["0"][name] == without["per_seed"]["0"][name]
    d = without["per_seed"]["0"]["density"]
    assert d["known"] == 0 and d["cells"] == 10 and d["over_bar"] == []
    assert all(c["verdict"] == "unknown" for c in list(d["jobs"].values()) + list(d["panels"].values()))
    text = scorecard.render(without)
    assert "| 0 | - | - | - | - | none |" in text and "10 of 10 scorings carried no token counts" in text


def test_token_counts_beside_the_result_are_read_when_the_result_has_none(tmp_path):
    manifest = density_manifest(tmp_path, tokens=False)
    root = tmp_path / "d"
    for stage, tokens in ((0, 2000), (2, 8000)):                       # a: 20 -> 40 correct, 100 -> 200 per correct
        rows = [{"id": i, "output_tokens": tokens // 100} for i in range(100)]
        (root / ("s%d-a" % stage) / "responses.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    prow = [{"panel": "p", "id": i, "output_tokens": 50} for i in range(100)] + [{"panel": "q", "id": i, "output_tokens": 50} for i in range(100)]
    (root / "s0-f" / "responses.jsonl").write_text("".join(json.dumps(r) + "\n" for r in prow), encoding="utf-8")
    (root / "s2-f" / "responses.jsonl").write_text("".join(json.dumps(dict(r, output_tokens=80 if r["panel"] == "p" else 60)) + "\n" for r in prow), encoding="utf-8")
    d = scorecard.build(manifest)["per_seed"]["0"]["density"]
    assert d["jobs"]["a"]["ratio"] == 2 and d["jobs"]["a"]["verdict"] == "over bar"
    assert d["jobs"]["b"]["verdict"] == "unknown"
    assert d["panels"]["p"]["ratio"] == 1.6 and d["panels"]["q"]["ratio"] == 1.2

