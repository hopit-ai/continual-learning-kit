"""kit/control_report.py: the small-model control's pre-registered decisions, with no GPU, network or model.

The work tree is built by hand in the layout kit/campaigns/k1c-control.yaml writes, with sweep.json files in the shape
kit/cap_sweep.py writes them. GSM8K has 300 questions, so the threshold t is ceil(5 x 300 / 100) = 15. The reference
scores S(2048) = 150, E(2048) = 155, S(8192) = 160, E(8192) = 165, and 400 tokens per strict-correct answer at 2048.
Every run, unless a test says otherwise, scores S(2048) = 170 (gain +20: learns, preserves), 400 tokens per correct
(cost ratio 1.0) and a mean training length of 100 at step 1 and 150 at step 40 (ratio 1.5): it does not drift.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

KIT = Path(__file__).resolve().parents[1] / "kit"


def _load(name: str):
    spec = importlib.util.spec_from_file_location("kit_%s_control_test" % name, KIT / ("%s.py" % name))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


control_report = _load("control_report")
cr = control_report

B, H, N = 2048, 8192, 300
ITEMS = "g" * 64
KEYS = ["%s-seed%d" % (cell, seed) for cell in ("lr6", "gate", "lr6-gate") for seed in (0, 1, 2)]
DEFAULT = {"sB": 170, "tpc": 400.0, "lengths": (100.0, 150.0), "machine": "m1", "run_attempt": 1, "sweep_attempt": 1,
           "model_attempt": None}


# ------------------------------------------------------------------------------------------ the tree
def sweep(work: Path, stem: str, attempt: int, sB: int, tpc=400.0, machine="m1", model="/w/models/Qwen3-1.7B", n=N, items=ITEMS):
    """A bed sweep with S(B) = sB, E(B) = sB + 5, S(H) = sB + 10, E(H) = sB + 15."""
    folder = work / "k1c" / "control" / "report-sweep" / ("%s-a%d" % (stem, attempt))
    folder.mkdir(parents=True)
    points = {512: (sB - 50, sB - 45, 80), B: (sB, sB + 5, 10), H: (sB + 10, sB + 15, 0)}
    per_budget = [{"budget": b, "n": n, "correct_strict": s, "correct_canonical": e, "cut": c, "finished": n - c,
                   "tokens_total": 1000, "tokens_per_correct_strict": tpc if b == B else 500.0}
                  for b, (s, e, c) in sorted(points.items())]
    body = {"schema": "kit-cap-sweep.v1", "kind": "bed", "bed": "gsm8k", "n": n, "cap": H, "budgets": sorted(points),
            "items_sha256": items, "model": model, "canonical_rule": "kit-canonical.v3", "per_budget": per_budget,
            "scoring": "/node/work/k1c/control/eval/%s-a%d" % (stem, attempt)}
    if machine is not None:
        body["machine"] = {"id": machine}
    (folder / "sweep.json").write_text(json.dumps(body))
    return folder


def run(work: Path, key: str, attempt: int, merged=1, lengths=(100.0, 150.0)) -> Path:
    """A run folder; `lengths` is (mean length at step 1, at step 40), None for a missing one, or None for no metrics."""
    folder = work / "runs" / ("%s-a%d" % (key, attempt))
    folder.mkdir(parents=True)
    (folder / "train-summary.json").write_text(json.dumps({
        "returncode": 0 if merged else 1, "merged": merged, "steps": 40, "lr": "1e-6", "max_response_length": 8192,
        "finish_gate": 0, "n_gpus": 8, "seconds": 3600, "model_dir": str(folder / "hf-step40")}))
    if lengths is not None:
        first, last = lengths
        lines = [{"step": 40, "data": {} if last is None else {"response_length/mean": last}},          # written backwards
                 {"step": 20, "data": {"response_length/mean": 999.0}},
                 {"step": 1, "data": {} if first is None else {"response_length/mean": first}},
                 {"step": 41, "data": {"critic/score/mean": 0.5}}]                                     # no length: not last
        (folder / "metrics.jsonl").write_text("\n".join(json.dumps(r) for r in lines) + '\n{"step": 42, "da')
    return folder


def build_tree(work: Path, overrides=None, skip=(), reference=True, check=True) -> Path:
    work.mkdir(parents=True, exist_ok=True)
    if reference:
        sweep(work, "base17b-gsm8k", 1, 150)
        if check:
            prefix_check(work, 1.0)                    # the campaign's check of the reference, passing and of this scoring
    for key in KEYS:
        if key in skip:
            continue
        spec = {**DEFAULT, **(overrides or {}).get(key, {})}
        run(work, key, spec["run_attempt"], lengths=spec["lengths"])
        model = "/w/runs/%s-a%d/hf-step40" % (key, spec["model_attempt"] or spec["run_attempt"])
        sweep(work, "%s-gsm8k" % key, spec["sweep_attempt"], spec["sB"], tpc=spec["tpc"], machine=spec["machine"], model=model)
    return work


def prefix_check(work: Path, agreement: float, attempt=1, **extra) -> Path:
    folder = work / "k1c" / "control" / "report-prefix"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / ("prefix-base17b-gsm8k-a%d.json" % attempt)
    path.write_text(json.dumps({"compared": 300, "text_agreement": agreement, "same_machine": True, "same_max_model_len": True,
                                "long": "/moved/eval/base17b-gsm8k-a1", "short": "/moved/eval/direct-base17b-gsm8k-a1", "short_cap": B, "long_cap": H, **extra}))
    return path


def report_of(work: Path, out: Path, *extra) -> dict:
    assert cr.main(["--work", str(work), "--serving", str(B), "--diagnostic", str(H), "--out", str(out), *extra]) == 0
    return json.loads((out / "control-report.json").read_text())


def _run(report, key):
    (row,) = [r for r in report["runs"] if r["key"] == key]
    return row


# ------------------------------------------------------------------------------------------- whole tree
def test_a_full_tree_decides_every_cell_and_the_bar_keys_are_ints(tmp_path):
    report = report_of(build_tree(tmp_path / "work"), tmp_path / "out")
    for key in ("runs_expected", "runs_reported", "cells_decided", "accounting_exact", "lineage_ok"):
        assert type(report[key]) is int, key
    assert (report["runs_expected"], report["runs_reported"], report["cells_decided"]) == (9, 9, 3)
    assert (report["accounting_exact"], report["lineage_ok"]) == (1, 1)
    assert report["threshold"] == 15 and report["n"] == 300
    assert report["reference_prefix_check"]["passed"] is True and report["reference_prefix_check"]["bound"] is True
    assert report["not_reported"] == [] and report["lineage_problems"] == []
    r = _run(report, "lr6-seed0")
    assert r["reported"] == 1 and r["run_attempt"] == 1 and r["sweep_attempt"] == 1
    assert r["scores"] == {"strict_serving": 170, "canonical_serving": 175, "strict_diagnostic": 180, "canonical_diagnostic": 185}
    # F = 150 - 170 = -20 = residual (165 - 185 = -20) + budget ((185 - 175) - (165 - 155) = 0) + extraction (5 - 5 = 0)
    assert (r["F"], r["residual"], r["budget"], r["extraction"]) == (-20, -20, 0, 0)
    assert (r["gain"], r["learns"], r["preserves"]) == (20, True, True)
    # the length at the first and last step that has it: 100 at step 1, 150 at step 40 (step 41 has none)
    assert (r["length_first"], r["length_first_step"], r["length_last"], r["length_last_step"], r["length_ratio"]) == (100.0, 1, 150.0, 40, 1.5)
    assert r["metrics_unreadable_lines"] == 1
    assert (r["drift_by_length"], r["drift_by_cost"], r["drifts"], r["tokens_per_correct_ratio"]) == (False, False, False, 1.0)
    assert r["label"] == "intact" and r["label_canonical"] == "intact"
    assert r["lr"] == "1e-6" and r["merged"] == 1 and r["n_gpus"] == 8
    cells = report["cells"]
    for cell in cells.values():
        assert (cell["free_of_drift"], cell["learns"], cell["preserves"], cell["runs_reported"]) == (True, True, True, 3)
    assert cells["lr6"]["reading"] == cr.READ_LR6
    assert cells["gate"]["reading"] == cr.READ_GATE
    assert cells["lr6-gate"]["reading"] == cr.READ_NONE
    assert report["no_cell_learns"] is False and report["no_cell_learns_sentence"] is None
    md = (tmp_path / "out" / "control-report.md").read_text()
    assert cr.MECHANICAL in md and cr.ONE_TASK in md and cr.NO_CELL_LEARNS not in md
    assert "| lr6 | 0 | 170 | 175 | 180 | 185 | +20 | yes | yes | 100.0 | 150.0 | 1.50 | 1.00 | no (length no, cost no) | -20 = -20 + +0 + +0 | intact (canonical intact) |" in md
    assert "None: every run is reported." in md


def test_the_verbatim_sentences():
    assert cr.MECHANICAL == "These are the pre-registered rules applied mechanically; they decide nothing by themselves."
    assert cr.ONE_TASK == "This is one task: it says nothing about keeping a task while learning another."
    assert cr.READ_LR6 == "The drift is recipe-specific to the higher learning rate, subject to the K1c runs being a fair comparison."
    assert cr.READ_GATE == ("The combined training-limit-and-gate change stabilises a single task at the old rate; it cannot say "
                            "which of the two parts did it.")
    assert cr.READ_NOT_USEFUL == "Its usefulness is not established; that alone does not show that learning was suppressed."
    assert cr.READ_NONE == "No pre-registered reading applies; the numbers stand as reported."
    assert cr.NO_CELL_LEARNS == ("No tested setting demonstrated learning on GSM8K at this dose; the reason is not established "
                                 "and it is not evidence about model size.")


# --------------------------------------------------------------------------------------- one run
def test_learns_and_preserves_at_their_boundaries(tmp_path):
    work = build_tree(tmp_path / "work", {"lr6-seed0": {"sB": 165}, "lr6-seed1": {"sB": 164},
                                          "lr6-seed2": {"sB": 135}, "gate-seed0": {"sB": 134}})
    report = report_of(work, tmp_path / "out")
    assert (_run(report, "lr6-seed0")["gain"], _run(report, "lr6-seed0")["learns"]) == (15, True)       # exactly t
    assert (_run(report, "lr6-seed1")["gain"], _run(report, "lr6-seed1")["learns"]) == (14, False)      # one below
    assert (_run(report, "lr6-seed2")["gain"], _run(report, "lr6-seed2")["preserves"]) == (-15, True)   # exactly S_ref - t
    assert (_run(report, "gate-seed0")["gain"], _run(report, "gate-seed0")["preserves"]) == (-16, False)
    assert _run(report, "lr6-seed2")["learns"] is False and _run(report, "lr6-seed1")["preserves"] is True


def test_drift_by_length_and_by_cost_at_their_boundaries(tmp_path):
    work = build_tree(tmp_path / "work", {
        "lr6-seed0": {"lengths": (100.0, 200.0)},          # exactly 2x: not drift
        "lr6-seed1": {"lengths": (100.0, 200.5)},          # over 2x: drift
        "lr6-seed2": {"tpc": 600.0},                       # exactly 1.5: not drift
        "gate-seed0": {"tpc": 600.5},                      # over 1.5: drift
        "gate-seed1": {"lengths": None},                   # no metrics: length unknown, cost False -> None
        "gate-seed2": {"lengths": None, "tpc": 900.0},     # length unknown, cost True -> True
        "lr6-gate-seed0": {"tpc": None},                   # cost unknown, length False -> None
        "lr6-gate-seed1": {"tpc": None, "lengths": (100.0, 300.0)}})  # cost unknown, length True -> True
    report = report_of(work, tmp_path / "out")
    want = {"lr6-seed0": (False, False, False), "lr6-seed1": (True, False, True), "lr6-seed2": (False, False, False),
            "gate-seed0": (False, True, True), "gate-seed1": (None, False, None), "gate-seed2": (None, True, True),
            "lr6-gate-seed0": (False, None, None), "lr6-gate-seed1": (True, None, True), "lr6-gate-seed2": (False, False, False)}
    for key, (by_length, by_cost, drifts) in want.items():
        r = _run(report, key)
        assert (r["drift_by_length"], r["drift_by_cost"], r["drifts"]) == (by_length, by_cost, drifts), key
    assert _run(report, "lr6-seed0")["length_ratio"] == 2.0 and _run(report, "lr6-seed2")["tokens_per_correct_ratio"] == 1.5
    assert _run(report, "gate-seed1")["length_first"] is None and _run(report, "gate-seed2")["length_first"] is None
    assert _run(report, "lr6-gate-seed0")["tokens_per_correct_ratio"] is None
    # gate: drifts [True, None, True] -> two drift: not free; lr6-gate: [None, True, False] -> undecided
    assert report["cells"]["gate"]["free_of_drift"] is False
    assert report["cells"]["lr6-gate"]["free_of_drift"] is None
    assert report["cells"]["lr6-gate"]["reading"] == "Not decided: free_of_drift (drift not known for lr6-gate-seed0)."
    assert report["cells_decided"] == 2


# --------------------------------------------------------------------------------------- one cell
def fake(key, reported=1, drifts=False, gain=20, preserves=True):
    return {"key": key, "reported": reported, "drifts": drifts if reported else None,
            "gain": gain if reported else None, "preserves": preserves if reported else None}


def cell_of(cell, *runs):
    return cr.cell_decisions(cell, list(runs), 15)


def test_two_of_three():
    assert cr.two_of_three([True, True, False]) is True
    assert cr.two_of_three([False, False, True]) is False
    assert cr.two_of_three([True, False, None]) is None
    assert cr.two_of_three([True, None, None]) is None
    assert cr.two_of_three([None, None, None]) is None
    assert cr.two_of_three([False, None, None]) is None
    # "can no longer reach two": with three values it coincides with "at least two False"; with two it stands alone
    assert cr.two_of_three([True, False]) is False
    assert cr.two_of_three([True, None]) is None


def test_the_length_endpoints_are_step_1_and_the_last_step_and_nothing_stands_in_for_them(tmp_path):
    """The pre-registration says step 1 and step 40. The first version took the first and last steps that HAD a length, so
    a run with no step 1 was read from step 20 (999 tokens) to 150 as "no drift" (round 2 of the review)."""
    report = report_of(build_tree(tmp_path / "work", {"lr6-seed0": {"lengths": (None, 150.0)}, "lr6-seed1": {"lengths": (100.0, None)},
                                                      "lr6-seed2": {"lengths": (100.0, 250.0)}}), tmp_path / "out")
    r = _run(report, "lr6-seed0")
    assert (r["length_first"], r["length_first_step"], r["length_last"], r["length_last_step"], r["length_ratio"], r["drift_by_length"]) == (None, None, 150.0, 40, None, None)
    r = _run(report, "lr6-seed1")
    assert (r["length_first"], r["length_last"], r["length_ratio"], r["drift_by_length"]) == (100.0, None, None, None)
    r = _run(report, "lr6-seed2")
    assert (r["length_first_step"], r["length_last_step"], r["length_ratio"], r["drift_by_length"]) == (1, 40, 2.5, True), "step 20's 999 is not an endpoint"


def test_free_of_drift():
    assert cell_of("lr6-gate", fake("a"), fake("b"), fake("c", drifts=True))["free_of_drift"] is True
    assert cell_of("lr6-gate", fake("a", drifts=True), fake("b", drifts=True), fake("c"))["free_of_drift"] is False
    assert cell_of("lr6-gate", fake("a"), fake("b", drifts=True), fake("c", drifts=None))["free_of_drift"] is None
    assert cell_of("lr6-gate", fake("a"), fake("b", reported=0), fake("c", reported=0))["free_of_drift"] is None
    # one drifts, one unreported, one unknown: no-drift can still reach... 0 + 2 = 2: None
    assert cell_of("lr6-gate", fake("a", drifts=True), fake("b", reported=0), fake("c", drifts=None))["free_of_drift"] is None
    # two drift and one unreported: no-drift can no longer reach two
    assert cell_of("lr6-gate", fake("a", drifts=True), fake("b", drifts=True), fake("c", reported=0))["free_of_drift"] is False


def test_cell_learns():
    assert cell_of("x", fake("a", gain=15), fake("b", gain=15), fake("c", gain=15))["learns"] is True          # mean exactly t
    assert cell_of("x", fake("a", gain=15), fake("b", gain=15), fake("c", gain=14))["learns"] is False         # mean 44/3
    assert cell_of("x", fake("a", gain=45), fake("b", gain=0), fake("c", gain=0))["learns"] is False           # one gain above 0
    assert cell_of("x", fake("a", gain=40), fake("b", gain=5), fake("c", gain=0))["learns"] is True            # two above 0
    one_missing = cell_of("x", fake("a", gain=90), fake("b", gain=90), fake("c", reported=0))
    assert one_missing["learns"] is None and one_missing["mean_gain"] is None and one_missing["runs_reported"] == 2
    assert cell_of("x", fake("a", gain=40), fake("b", gain=5), fake("c", gain=0))["mean_gain"] == 15.0


def test_cell_preserves():
    assert cell_of("x", fake("a"), fake("b"), fake("c", preserves=False))["preserves"] is True
    assert cell_of("x", fake("a", preserves=False), fake("b", preserves=False), fake("c"))["preserves"] is False
    assert cell_of("x", fake("a"), fake("b", preserves=False), fake("c", reported=0))["preserves"] is None
    assert cell_of("x", fake("a", preserves=False), fake("b", reported=0), fake("c", reported=0))["preserves"] is None
    assert cell_of("x", fake("a", preserves=False), fake("b", preserves=False), fake("c", reported=0))["preserves"] is False
    assert cell_of("x", fake("a"), fake("b", reported=0), fake("c", reported=0))["preserves"] is None


def test_each_reading():
    drifting = (fake("a", drifts=True, gain=0), fake("b", drifts=True, gain=0), fake("c", gain=0))
    # lr6 free of drift: its reading wins even when the cell does not learn
    assert cell_of("lr6", fake("a", gain=0), fake("b", gain=0), fake("c", gain=0))["reading"] == cr.READ_LR6
    assert cell_of("gate", fake("a"), fake("b"), fake("c"))["reading"] == cr.READ_GATE
    # gate free of drift but not preserving: the gate reading does not apply
    assert cell_of("gate", *(fake(k, preserves=False) for k in "abc"))["reading"] == cr.READ_NOT_USEFUL
    assert cell_of("lr6", *drifting)["reading"] == cr.READ_NOT_USEFUL               # learns False
    assert cell_of("lr6-gate", fake("a", gain=0), fake("b", gain=0), fake("c", gain=0))["reading"] == cr.READ_NOT_USEFUL
    assert cell_of("lr6-gate", fake("a"), fake("b"), fake("c"))["reading"] == cr.READ_NONE
    assert cell_of("lr6", *(fake(k, drifts=True) for k in "abc"))["reading"] == cr.READ_NONE   # drifts, learns, preserves
    assert cell_of("lr6-gate", fake("a"), fake("b"), fake("c", reported=0))["reading"] == \
        "Not decided: learns (runs not reported: c)."
    undecided = cell_of("gate", fake("a", drifts=True), fake("b", drifts=None), fake("c", reported=0))
    assert undecided["reading"] == "Not decided: free_of_drift, learns (runs not reported: c; drift not known for b)."
    assert undecided["decided"] == 0 and undecided["preserves"] is True


def test_no_cell_learns(tmp_path):
    report = report_of(build_tree(tmp_path / "work", {k: {"sB": 150} for k in KEYS}), tmp_path / "out")
    assert all(c["learns"] is False for c in report["cells"].values())
    assert report["no_cell_learns"] is True and report["no_cell_learns_sentence"] == cr.NO_CELL_LEARNS
    assert report["cells"]["lr6"]["reading"] == cr.READ_LR6
    assert report["cells"]["gate"]["reading"] == cr.READ_GATE
    assert report["cells"]["lr6-gate"]["reading"] == cr.READ_NOT_USEFUL
    assert cr.NO_CELL_LEARNS in (tmp_path / "out" / "control-report.md").read_text()


def test_one_undecided_cell_is_not_no_cell_learns(tmp_path):
    overrides = {k: {"sB": 150} for k in KEYS}
    report = report_of(build_tree(tmp_path / "work", overrides, skip={"lr6-seed2"}), tmp_path / "out")
    # three-valued (round 2): two cells do not learn and one is unknown, so "no cell learns" is unknown, not false
    assert report["cells"]["lr6"]["learns"] is None and report["no_cell_learns"] is None and report["no_cell_learns_sentence"] is None
    assert report["runs_reported"] == 8
    (missing,) = report["not_reported"]
    assert missing["key"] == "lr6-seed2" and missing["reason"].startswith("missing:")
    assert report["lineage_ok"] == 1           # a missing sweep is not a lineage problem


# ----------------------------------------------------------------------------------------- lineage
def test_lineage_matches():
    assert cr.lineage_matches("/w/runs/lr6-seed0-a2/hf-step40", "lr6-seed0", 2)
    assert cr.lineage_matches("/w/runs/lr6-seed0-a2/hf-step40/", "lr6-seed0", 2)
    assert cr.lineage_matches("runs/lr6-seed0-a2/hf-step7", "lr6-seed0", 2)
    assert not cr.lineage_matches("/w/runs/lr6-seed0-a1/hf-step40", "lr6-seed0", 2)
    assert not cr.lineage_matches("/w/runs/lr6-seed0-a12/hf-step40", "lr6-seed0", 1)
    assert not cr.lineage_matches("/w/runs/lr6-gate-seed0-a2/hf-step40", "gate-seed0", 2)
    assert not cr.lineage_matches("/w/myruns/gate-seed0-a2/hf-step40", "gate-seed0", 2)
    assert not cr.lineage_matches("/w/runs/lr6-seed0-a2/hf-step40/model", "lr6-seed0", 2)
    assert not cr.lineage_matches("/w/runs/lr6-seed0-a2/hf-stepX", "lr6-seed0", 2)
    assert not cr.lineage_matches(None, "lr6-seed0", 2)


def test_a_sweep_of_an_older_attempt_is_not_reported(tmp_path):
    work = build_tree(tmp_path / "work", {"lr6-seed0": {"run_attempt": 2, "model_attempt": 1}})
    run(work, "lr6-seed0", 1)                                  # attempt 1 merged too, but 2 is the attempt of record
    report = report_of(work, tmp_path / "out")
    r = _run(report, "lr6-seed0")
    assert r["reported"] == 0 and r["run_attempt"] == 2 and r["reason"].startswith("lineage:")
    assert r["gain"] is None and r["drifts"] is None and r["learns"] is None
    assert report["lineage_ok"] == 0 and type(report["lineage_ok"]) is int
    assert [p["key"] for p in report["lineage_problems"]] == ["lr6-seed0"]
    assert report["runs_reported"] == 8 and report["cells"]["lr6"]["learns"] is None
    md = (tmp_path / "out" / "control-report.md").read_text()
    assert "- lr6-seed0: lineage:" in md and "lineage clean: NO" in md


def test_the_merged_attempt_wins_over_a_later_unmerged_one(tmp_path):
    work = build_tree(tmp_path / "work")
    run(work, "lr6-seed0", 2, merged=0)                        # a retry that failed: attempt 1 stays the run of record
    report = report_of(work, tmp_path / "out")
    r = _run(report, "lr6-seed0")
    assert r["reported"] == 1 and r["run_attempt"] == 1 and r["attempts_found"] == [1, 2] and report["lineage_ok"] == 1


def test_the_highest_sweep_attempt_is_the_one_checked(tmp_path):
    work = build_tree(tmp_path / "work", {"lr6-seed0": {"run_attempt": 2}})
    run(work, "lr6-seed0", 1)
    sweep(work, "lr6-seed0-gsm8k", 2, 10, model="/w/runs/lr6-seed0-a1/hf-step40")    # newer sweep, older model
    report = report_of(work, tmp_path / "out")
    r = _run(report, "lr6-seed0")
    assert r["sweep_attempt"] == 2 and r["reported"] == 0 and r["reason"].startswith("lineage:")


def test_a_sweep_without_a_merged_run_is_a_lineage_problem(tmp_path):
    work = build_tree(tmp_path / "work", skip={"gate-seed1"})
    run(work, "gate-seed1", 1, merged=0)
    sweep(work, "gate-seed1-gsm8k", 1, 170, model="/w/runs/gate-seed1-a1/hf-step40")
    report = report_of(work, tmp_path / "out")
    r = _run(report, "gate-seed1")
    assert r["reported"] == 0 and r["run_attempt"] is None and r["reason"].startswith("lineage:")
    assert report["lineage_ok"] == 0


# ----------------------------------------------------------------------------------------- machines
def test_another_machine_is_not_reported_unless_allowed(tmp_path):
    work = build_tree(tmp_path / "work", {"lr6-seed0": {"machine": "m2"}, "lr6-seed1": {"machine": None}})
    report = report_of(work, tmp_path / "out")
    for key in ("lr6-seed0", "lr6-seed1"):
        assert _run(report, key)["reported"] == 0 and _run(report, key)["reason"].startswith("another machine"), key
    assert report["different_machines_allowed"] is False and report["lineage_ok"] == 1 and report["runs_reported"] == 7
    allowed = report_of(work, tmp_path / "out2", "--allow-different-machines")
    assert allowed["different_machines_allowed"] is True and allowed["runs_reported"] == 9
    assert _run(allowed, "lr6-seed0")["same_machine"] is False and _run(allowed, "lr6-seed2")["same_machine"] is True
    assert "--allow-different-machines was given" in (tmp_path / "out2" / "control-report.md").read_text()


def test_other_items_are_not_reported(tmp_path):
    work = build_tree(tmp_path / "work", skip={"gate-seed2"})
    run(work, "gate-seed2", 1)
    sweep(work, "gate-seed2-gsm8k", 1, 170, model="/w/runs/gate-seed2-a1/hf-step40", items="x" * 64)
    r = _run(report_of(work, tmp_path / "out"), "gate-seed2")
    assert r["reported"] == 0 and r["reason"].startswith("other items")


# ----------------------------------------------------------------------------------------- refusals
def test_the_reference_prefix_check(tmp_path):
    work = build_tree(tmp_path / "work", check=False)
    prefix_check(work, 0.99)
    report = report_of(work, tmp_path / "out")
    assert report["reference_prefix_check"]["passed"] is True and report["reference_prefix_check"]["text_agreement"] == 0.99
    prefix_check(work, 0.98, attempt=2)                        # the newer check wins, and fails
    with pytest.raises(SystemExit, match="prefix check .*prefix-base17b-gsm8k-a2.json does not qualify it .failed"):
        cr.main(["--work", str(work), "--serving", str(B), "--diagnostic", str(H), "--out", str(tmp_path / "o2")])
    assert not (tmp_path / "o2").exists()


@pytest.mark.parametrize("extra, said", [({"same_max_model_len": False}, "failed"), ({"long": "/moved/eval/base17b-gsm8k-a9"}, "of another scoring"),
                                         ({"short_cap": 1024}, "of another scoring"), ({"long_cap": 4096}, "of another scoring")])
def test_a_prefix_check_on_another_context_or_another_scoring_does_not_qualify(tmp_path, extra, said):
    work = build_tree(tmp_path / "work", check=False)
    prefix_check(work, 1.0, **extra)
    with pytest.raises(SystemExit, match="does not qualify it .%s" % said):
        cr.main(["--work", str(work), "--serving", str(B), "--diagnostic", str(H), "--out", str(tmp_path / "o")])


def test_the_reference_check_is_required(tmp_path):
    """A missing check used to be recorded and to decide nothing (round 2 of the review): the campaign schedules it, so
    without it no budget read against the reference is qualified."""
    work = build_tree(tmp_path / "work", check=False)
    with pytest.raises(SystemExit, match="does not qualify it .missing"):
        cr.main(["--work", str(work), "--serving", str(B), "--diagnostic", str(H), "--out", str(tmp_path / "o")])
    assert not (tmp_path / "o").exists()


def test_a_missing_reference_is_an_error(tmp_path):
    work = build_tree(tmp_path / "work", reference=False)
    with pytest.raises(SystemExit, match="no sweep of the untrained model"):
        cr.main(["--work", str(work), "--serving", str(B), "--diagnostic", str(H), "--out", str(tmp_path / "o")])
    assert not (tmp_path / "o").exists()


def test_other_refusals(tmp_path):
    work = build_tree(tmp_path / "work")
    (tmp_path / "taken").mkdir()
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        cr.main(["--work", str(work), "--serving", str(B), "--diagnostic", str(H), "--out", str(tmp_path / "taken")])
    assert list((tmp_path / "taken").iterdir()) == []
    with pytest.raises(SystemExit, match="must be below"):
        cr.main(["--work", str(work), "--serving", str(H), "--diagnostic", str(B), "--out", str(tmp_path / "o")])
    with pytest.raises(SystemExit, match="no such work tree"):
        cr.main(["--work", str(tmp_path / "nowhere"), "--serving", str(B), "--diagnostic", str(H), "--out", str(tmp_path / "o")])
    with pytest.raises(SystemExit, match="did not read budget 4096"):
        cr.main(["--work", str(work), "--serving", "4096", "--diagnostic", str(H), "--out", str(tmp_path / "o")])
    assert not (tmp_path / "o").exists()


def test_the_report_needs_only_budget_report(tmp_path):
    """kit/ exported with just these two files still works: no import of pilot_report or anything else of the kit."""
    import shutil
    kit = tmp_path / "kit"
    kit.mkdir()
    for name in ("control_report.py", "budget_report.py"):
        shutil.copy(KIT / name, kit / name)
    spec = importlib.util.spec_from_file_location("kit_control_report_alone", kit / "control_report.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    work = build_tree(tmp_path / "work")
    assert module.main(["--work", str(work), "--serving", str(B), "--diagnostic", str(H), "--out", str(tmp_path / "o")]) == 0
