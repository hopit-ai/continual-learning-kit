"""The two-task pilot's report tools: kit/rollout_stats.py and kit/pilot_report.py, with no GPU, network or transformers.

The tokenizer is replaced by one whose tokens are the whitespace-separated words, so every length below can be read
off the test text. The work tree is built by hand in the fixed layout the pilot campaign writes, with sweep.json files
in the shape kit/cap_sweep.py writes them, and every expected number is worked out in the comments beside it.
"""
from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest

KIT = Path(__file__).resolve().parents[1] / "kit"


def _load(name: str):
    spec = importlib.util.spec_from_file_location("kit_%s_pilot_test" % name, KIT / ("%s.py" % name))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


rollout_stats = _load("rollout_stats")
pilot_report = _load("pilot_report")


class WordTokenizer:
    """One token per whitespace-separated word."""

    def encode(self, text, add_special_tokens=True):
        assert add_special_tokens is False
        return text.split()


@pytest.fixture(autouse=True)
def fake_tokenizer(monkeypatch):
    monkeypatch.setattr(rollout_stats, "load_tokenizer", lambda model_dir: WordTokenizer())
    monkeypatch.setitem(sys.modules, "transformers", None)        # any import of transformers would now fail


# ---------------------------------------------------------------------------------- rollout stats
def rollout(prompt, output, score, step, **extra):
    return {"input": prompt, "output": output, "score": score, "step": step, "acc": float(score > 0), **extra}


def write_rollouts(run: Path) -> Path:
    """Three steps, written as 1, 2 and 10 so a string sort would put 10 before 2."""
    folder = run / "rollouts"
    folder.mkdir(parents=True)
    loop = "w w w w w\nw w w w w"                    # 10 words; one line twice: a loop
    step1 = ([rollout("P1", "a\nb\nc", 1.0, 1) for _ in range(4)]          # three distinct lines: not a loop
             + [rollout("P2", loop, 0.0, 1, truncated=True) for _ in range(2)]
             + [rollout("P2", "y", 0.0, 1, truncated=False) for _ in range(2)])
    step2 = ([rollout("P1", "a", s, 2) for s in (1.0, 0.0, 1.0, 0.0)] + [rollout("P2", "", 0.0, 2) for _ in range(4)])
    step10 = [rollout("P1", "q q", 1.0, 10), rollout("P1", "q", 0.0, 10)]
    for step, rows in ((1, step1), (2, step2), (10, step10)):
        (folder / ("%d.jsonl" % step)).write_text("".join(json.dumps(r) + "\n" for r in rows))
    return run


def stats_of(run: Path, out: Path, *extra):
    assert rollout_stats.main(["--run", str(run), "--model", "unused", "--out", str(out), *extra]) == 0
    return json.loads((out / "rollout-stats.json").read_text())


def test_rollout_stats_per_step(tmp_path):
    report = stats_of(write_rollouts(tmp_path / "run"), tmp_path / "out", "--cap", "10")
    assert report["schema"] == rollout_stats.SCHEMA and report["cap"] == 10 and report["cap_source"] == "--cap"
    assert [s["step"] for s in report["steps"]] == [1, 2, 10]
    one, two, ten = report["steps"]
    assert (one["rollouts"], one["prompts"], one["score_mean"]) == (8, 2, 0.5)
    assert one["groups"] == {"n": 2, "same_score_share": 1.0, "all_zero_share": 0.5, "all_positive_share": 0.5}
    # lengths [3,3,3,3,10,10,1,1]: mean 4.25, median 3, p90 the 8th of 8 (ceil 7.2), max 10
    assert one["tokens"]["all"] == {"n": 8, "mean": 4.25, "median": 3.0, "p90": 10, "max": 10}
    assert one["tokens"]["positive"]["mean"] == 3.0 and one["tokens"]["zero"]["mean"] == 5.5
    assert one["at_cap_share"] == {"all": 0.25, "positive": 0.0, "zero": 0.5}
    assert one["truncated_share"] == 0.5 and one["truncated_known"] == 4
    assert one["looping_share"] == 0.25                 # the two repeated-line answers; three distinct lines are not a loop
    assert two["groups"] == {"n": 2, "same_score_share": 0.5, "all_zero_share": 0.5, "all_positive_share": 0.0}
    assert two["truncated_share"] is None and two["truncated_known"] == 0
    assert two["tokens"]["positive"] == {"n": 2, "mean": 1.0, "median": 1.0, "p90": 1, "max": 1}
    assert ten["groups"]["same_score_share"] == 0.0 and ten["score_mean"] == 0.5
    totals = report["totals"]
    assert totals["steps"] == 3 and totals["rollouts"] == 18 and totals["prompts"] == 2 and totals["prompt_groups"] == 5
    assert totals["groups"]["same_score_share"] == 0.6                 # 2 + 1 + 0 of 5 groups
    md = (tmp_path / "out" / "rollout-stats.md").read_text()
    assert "| 1 | 8 | 2 | 0.500 | 100.0% / 50.0% / 50.0% |" in md and "| all | 18 |" in md


def test_the_loop_measures():
    assert rollout_stats.compress_ratio("") is None
    assert rollout_stats.compress_ratio("the cat sat " * 200) < rollout_stats.compress_ratio("a quick brown fox jumps over the lazy dog")
    assert not rollout_stats.is_looping("Thought: t\nAction: a\nAction Input: {}")       # every line once
    assert rollout_stats.is_looping("x\ny\nx\nz\nw\nv")                              # 2 of 6 = 33%, twice
    assert not rollout_stats.is_looping("x\ny\nx\nz\nw\nv\nu")                       # 2 of 7 = 28.6%
    assert not rollout_stats.is_looping("\n\n  \n")


def test_the_cap_comes_from_the_run_summary_else_8192(tmp_path):
    run = write_rollouts(tmp_path / "run")
    assert stats_of(run, tmp_path / "o1")["cap"] == 8192
    (run / "run-summary.json").write_text(json.dumps({"max_response_length": 7}))
    report = stats_of(run, tmp_path / "o2")
    assert report["cap"] == 7 and "run-summary.json" in report["cap_source"]
    (run / "train-summary.json").write_text(json.dumps({"max_response_length": 9}))
    assert stats_of(run, tmp_path / "o3")["cap"] == 9


def test_rollout_stats_refusals(tmp_path):
    (tmp_path / "bare").mkdir()
    with pytest.raises(SystemExit, match="no rollouts folder"):
        rollout_stats.main(["--run", str(tmp_path / "bare"), "--model", "unused", "--out", str(tmp_path / "o")])
    assert not (tmp_path / "o").exists()
    run = write_rollouts(tmp_path / "run")
    (tmp_path / "taken").mkdir()
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        rollout_stats.main(["--run", str(run), "--model", "unused", "--out", str(tmp_path / "taken")])
    (run / "rollouts" / "3.jsonl").write_text('{"input": "P", "output": "x"}\n')
    with pytest.raises(SystemExit, match="numeric `score`"):
        rollout_stats.main(["--run", str(run), "--model", "unused", "--out", str(tmp_path / "o4")])


# ------------------------------------------------------------------------------------ the work tree
B, H = 2048, 8192
N = {"chemistry": 100, "toolalpaca": 50}
ITEMS = {"chemistry": "c" * 64, "toolalpaca": "t" * 64}
BASE_MODEL = "/models/Qwen3-8B"
FIRST = {"chemtool": "chem", "toolchem": "tool"}                   # a stage-2 chain -> the chain it starts from


def model_of(key: str, attempt=1) -> str:
    """The model folder a sweep of `key` scored: the untrained model, or attempt `attempt` of the run's checkpoint."""
    return BASE_MODEL if key == "base8b" else "/work/k8b-work/runs/%s-a%d/hf-step40" % (key, attempt)


def pts(at512, at2048, at8192, tpc=400.0, cut_h=0, extra=None):
    """(strict, canonical) at 512, 2048 and 8192 -> {budget: (strict, canonical, cut, tokens per strict correct)}."""
    points = {512: (*at512, 40, tpc), 2048: (*at2048, 10, tpc), 8192: (*at8192, cut_h, tpc)}
    points.update(extra or {})
    return points


# Stage 1 against the base (chemistry n = 100, toolalpaca n = 50; gain at S(2048)):
#   g8-chem-r1   60 - 30 = 30, 30 per 100, acquired; tokens per correct 500 / 400 = 1.25
#   g8-tool-r1   30 - 15 = 15, 30 per 100;  g8-chem-r2 30;  g32-tool-r1 10, 20 per 100;  sema-chem-r1 20
#   g32-chem-r1  32 - 30 = 2, 2 per 100: NOT acquired (and at E(8192) 35 - 35 = 0: not in the diagnostic cohort either)
# Stage 2, the strict loss S_ref(N) - S_after(N) at 512 / 2048 / 8192, per 100:
#   g8-chemtool-r1   20 / 20 / 4        (F 20 = residual 4 + budget 13 + extraction 3; the loss shrinks at H)
#   g8-toolchem-r1   2 / 5 / 3 questions of 50 = 4 / 10 / 6
#   g8-chemtool-r2   10 / 10 / 2        (F 10 = residual 3 + budget 8 + extraction -1)
#   g32-toolchem-r1  1 / 5 / 1 of 50 = 2 / 10 / 2
#   sema-chemtool-r1 5 / 5 / 8          (the loss grows at H; F 5 = residual 9 + budget -3 + extraction -1)
#   g32-chemtool-r1  F 7, but chemistry was not acquired: not defined
# Own means per 100 at 2048: g8 40/3 = 13.333333, g32 10, sema 5; at 8192: g8 4, g32 2, sema 8.
# Shared rows: g8 and sema share chemtool-r1 (20 vs 5 at 2048, 4 vs 8 at 8192: reversed); g8 and g32 share toolchem-r1
# (10 vs 10 at 2048: a tie; 6 vs 2 at 8192); g32 and sema share none.
TREE = {
    ("base8b", "chemistry"): pts((20, 22), (30, 33), (32, 35), tpc=400.0, cut_h=3),
    ("base8b", "toolalpaca"): pts((10, 10), (15, 16), (15, 16), tpc=120.0),
    ("g8-chem-r1", "chemistry"): pts((40, 42), (60, 62), (62, 64), tpc=500.0),
    ("g8-chemtool-r1", "chemistry"): pts((20, 25), (40, 45), (58, 60)),
    ("g8-chemtool-r1", "toolalpaca"): pts((20, 20), (28, 29), (29, 30)),
    ("g8-tool-r1", "toolalpaca"): pts((20, 20), (30, 31), (30, 31)),
    ("g8-toolchem-r1", "toolalpaca"): pts((18, 18), (25, 26), (27, 28)),
    ("g8-chem-r2", "chemistry"): pts((40, 42), (60, 62), (62, 64)),
    ("g8-chemtool-r2", "chemistry"): pts((30, 31), (50, 51), (60, 61)),
    ("g32-chem-r1", "chemistry"): pts((20, 21), (32, 34), (33, 35)),
    ("g32-chemtool-r1", "chemistry"): pts((15, 16), (25, 27), (30, 31)),
    ("g32-tool-r1", "toolalpaca"): pts((15, 15), (25, 26), (26, 27)),
    ("g32-toolchem-r1", "toolalpaca"): pts((14, 14), (20, 21), (25, 26)),
    ("sema-chem-r1", "chemistry"): pts((35, 36), (50, 52), (55, 57)),
    ("sema-chemtool-r1", "chemistry"): pts((30, 31), (45, 46), (47, 48), extra={4096: (46, 47, 5, 400.0)}),
}


def bed_sweep(work: Path, key: str, task: str, points: dict, attempt=1, machine="m1", run=1) -> Path:
    """`run`: the run attempt whose checkpoint the sweep scored (its `model`); None writes no `model` at all."""
    folder = work / "k8b" / "report-sweep" / ("%s-%s-a%d" % (key, task, attempt))
    folder.mkdir(parents=True)
    n = N[task]
    per_budget = [{"budget": b, "n": n, "correct_strict": s, "correct_canonical": e, "cut": c, "finished": n - c,
                   "tokens_total": 1000, "tokens_per_correct_strict": t} for b, (s, e, c, t) in sorted(points.items())]
    sweep = {"schema": "kit-cap-sweep.v1", "kind": "bed", "bed": task, "n": n, "cap": 8192, "budgets": sorted(points),
             "items_sha256": ITEMS[task], "machine": {"id": machine}, "canonical_rule": "kit-canonical.v3", "per_budget": per_budget,
             "scoring": "/node/work/k8b/eval/%s-%s-a%d" % (key, task, attempt)}       # the scoring folder the sweep read
    if run is not None:
        sweep["model"] = model_of(key, run)
    (folder / "sweep.json").write_text(json.dumps(sweep))
    return folder


def panel_sweep(work: Path, key: str, correct: dict, attempt=1, machine="m1", run=1) -> Path:
    """correct: {panel: (at 2048, at 8192)}, 100 questions a panel."""
    folder = work / "k8b" / "report-sweep" / ("%s-panel-a%d" % (key, attempt))
    folder.mkdir(parents=True)
    per_budget = []
    for index, budget in enumerate((B, H)):
        panels = {name: {"n": 100, "correct": pair[index], "cut": 0, "tokens_total": 500} for name, pair in correct.items()}
        per_budget.append({"budget": budget, "panels": panels,
                           "total": {k: sum(p[k] for p in panels.values()) for k in ("n", "correct", "cut", "tokens_total")}})
    (folder / "sweep.json").write_text(json.dumps({
        "schema": "kit-cap-sweep.v1", "kind": "panel", "n": 100 * len(correct), "cap": 8192, "budgets": [B, H],
        "panel_file_sha256": "p" * 64, "panels": list(correct), "machine": {"id": machine}, "model": model_of(key, run),
        "per_budget": per_budget}))
    return folder


def edit_sweep(work: Path, stem: str, attempt: int, change) -> None:
    path = work / "k8b" / "report-sweep" / ("%s-a%d" % (stem, attempt)) / "sweep.json"
    sweep = json.loads(path.read_text())
    change(sweep)
    path.write_text(json.dumps(sweep))


def write_run(work: Path, key: str, attempt: int, summary=None, metrics=None) -> Path:
    run = work / "runs" / ("%s-a%d" % (key, attempt))
    run.mkdir(parents=True, exist_ok=True)
    if summary is not None:
        (run / "run-summary.json").write_text(json.dumps(summary))
    if metrics is not None:
        (run / "metrics.jsonl").write_text(metrics)
    return run


def merged_attempt(work: Path, key: str):
    found = [int(path.parent.name.rsplit("-a", 1)[1]) for path in (work / "runs").glob("%s-a*/run-summary.json" % key)
             if json.loads(path.read_text()).get("merged") == 1]
    return max(found) if found else None


def start_of(work: Path, key: str) -> str:
    """The model a run starts from: the untrained model, or for a stage-2 chain its first stage's merged checkpoint."""
    recipe, chain, rep = key.split("-")
    if chain not in FIRST:
        return BASE_MODEL
    first = "%s-%s-%s" % (recipe, FIRST[chain], rep)
    return model_of(first, merged_attempt(work, first) or 1)


def merge_runs(work: Path, keys) -> None:
    """A merged attempt 1 for every run key that has no merged attempt, stage 1 first so stage 2 can name its start."""
    for key in sorted({k for k in keys if k != "base8b"}, key=lambda k: k.split("-")[1] in FIRST):
        if merged_attempt(work, key) is None:
            write_run(work, key, 1, summary={"returncode": 0, "merged": 1, "model_dir": start_of(work, key)})


def metrics_text() -> str:
    """Steps 1-25, written backwards; the mean length is 100 + 10 x step except a peak of 999 at step 22."""
    lines = []
    for step in range(25, 0, -1):
        data = {"response_length/mean": 999.0 if step == 22 else 100.0 + 10 * step, "response_length/max": 2000.0,
                "response_length/clip_ratio": round(step / 100, 2), "critic/score/mean": round(step / 100, 2), "actor/entropy": 0.5}
        if step in (10, 25):
            data["val-core/sciknoweval/acc/mean@16"] = {10: 0.41, 25: 0.52}[step]
        lines.append(json.dumps({"step": step, "data": data}))
    return "\n".join(lines) + '\n{"step": 26, "data": {"response_len'        # a line cut while it was written


def build_tree(work: Path, tree=TREE, machines=None) -> Path:
    """Every sweep scored the merged attempt of its run (g8-chem-r1's is attempt 2, every other run's attempt 1), and
    every stage-2 run started from its first stage's merged attempt."""
    machines = machines or {}
    for (key, task), points in tree.items():
        run = 2 if key == "g8-chem-r1" else 1
        bed_sweep(work, key, task, points, attempt=run, machine=machines.get((key, task), "m1"), run=run)
    bed_sweep(work, "g8-chem-r1", "chemistry", pts((0, 0), (0, 0), (0, 0)), attempt=1, run=1)     # an older attempt: loses
    panel_sweep(work, "base8b", {"math": (30, 35), "knowledge": (40, 40)})
    panel_sweep(work, "g8-chem-r1", {"math": (28, 34), "knowledge": (41, 40)}, machine=machines.get(("g8-chem-r1", "panel"), "m1"), run=2)
    write_run(work, "g8-chem-r1", 1, summary={"seconds": 1, "n_gpus": 1})                     # never merged
    run = write_run(work, "g8-chem-r1", 2, metrics=metrics_text(),
                    summary={"name": "g8-chem-r1", "arm": "grpo", "steps": 25, "returncode": 0, "merged": 1, "n_gpus": 4,
                             "seconds": 7200, "learning_rate": "1e-6", "dataset": "sciknoweval", "model_dir": BASE_MODEL, "mini_batch": 8})
    write_rollouts(run)
    assert rollout_stats.main(["--run", str(run), "--model", "unused", "--out", str(run / "report-rollouts")]) == 0
    write_run(work, "g8-chemtool-r1", 1, summary={"returncode": 0, "merged": 1, "model_dir": model_of("g8-chem-r1", 2)})
    write_run(work, "g8-chemtool-r1", 2, summary={"returncode": 1, "merged": 0})     # a retry that failed: attempt 1 stays of record
    write_run(work, "sfrz-chem-r1", 1)                    # a run folder holding nothing yet
    merge_runs(work, [key for key, _task in tree])
    return work


def write_check(work: Path, name: str, compared=100, agreement=1.0, same_machine=True, same_length=True, long=None,
                short_cap=2048, long_cap=8192) -> None:
    """A prefix check file. `long` is the long scoring it was made on; by default the scoring folder its own name says."""
    folder = work / "k8b" / "report-prefix"
    folder.mkdir(parents=True, exist_ok=True)
    long = long or "/node/work/k8b/eval/%s" % name[len("prefix-"):-len(".json")]
    (folder / name).write_text(json.dumps({"compared": compared, "text_agreement": agreement, "same_machine": same_machine,
                                           "same_max_model_len": same_length, "long": long, "short": "/node/work/k8b/eval/direct",
                                           "short_cap": short_cap, "long_cap": long_cap}))


def qualify(work: Path) -> None:
    """A passing check, of the very scoring that was swept, for every scheduled pair whose sweep exists and has no check
    file yet: the state of a campaign whose prefix rows all passed. Tests about the checks write their own first."""
    for key, task in pilot_report.REQUIRED_PREFIX:
        sweeps = sorted((work / "k8b" / "report-sweep").glob("%s-%s-a*" % (key, task)), key=lambda f: int(f.name.rsplit("-a", 1)[1]))
        sweeps = [f for f in sweeps if (f / "sweep.json").is_file()]
        if sweeps and not list((work / "k8b" / "report-prefix").glob("prefix-%s-%s-a*.json" % (key, task))):
            write_check(work, "prefix-%s-%s-a1.json" % (key, task), long=json.loads((sweeps[-1] / "sweep.json").read_text())["scoring"])


def report_of(work: Path, out: Path, *extra, qualified=True) -> dict:
    if qualified:
        qualify(work)
    assert pilot_report.main(["--work", str(work), "--serving", str(B), "--diagnostic", str(H), "--out", str(out), *extra]) == 0
    return json.loads((out / "pilot-report.json").read_text())


@pytest.fixture()
def full(tmp_path):
    work = build_tree(tmp_path / "work")
    return work, report_of(work, tmp_path / "report"), (tmp_path / "report" / "pilot-report.md").read_text()


def _row(rows, key):
    (row,) = [r for r in rows if r["key"] == key]
    return row


def test_the_bar_keys(full):
    _work, report, _md = full
    assert report["schema"] == pilot_report.SCHEMA and report["runs_expected"] == 24
    names = ("runs_found", "stage1_scored", "stage2_scored", "retention_rows_defined", "retention_rows_defined_diagnostic",
             "accounting_exact", "lineage_ok", "prefix_checks_found", "prefix_checks_failed", "prefix_checks_missing_count")
    bars = {k: report[k] for k in names}
    # 12 run keys hold sweeps, and sfrz-chem-r1 has an empty run folder; the four scheduled prefix checks are there
    assert bars == {"runs_found": 13, "stage1_scored": 6, "stage2_scored": 6, "retention_rows_defined": 5,
                    "retention_rows_defined_diagnostic": 5, "accounting_exact": 1, "lineage_ok": 1,
                    "prefix_checks_found": 4, "prefix_checks_failed": 0, "prefix_checks_missing_count": 0}
    assert all(c["bound"] is True and c["used"] is True for c in report["prefix_checks"])
    assert all(type(v) is int for v in bars.values())
    assert report["lineage_problems"] == [] and len(report["prefix_checks"]) == 4 and report["different_machines_allowed"] == 0


def test_base_and_acquisition(full):
    _work, report, md = full
    chem = report["base"]["chemistry"]
    assert chem["scores"] == {"strict_serving": 30, "canonical_serving": 33, "strict_diagnostic": 32, "canonical_diagnostic": 35}
    assert chem["cut_share_diagnostic"] == 0.03 and chem["tokens_per_correct_serving"] == 400.0
    rows = report["acquisition"]
    assert len(rows) == 12                                        # 6 recipe-replicates x 2 tasks
    g8 = _row(rows, "g8-chem-r1")
    assert g8["attempt"] == 2, "the highest attempt wins"
    assert (g8["gain_serving_strict"], g8["gain_per_100"], g8["acquired"]) == (30, 30.0, True)
    assert g8["tokens_per_correct_ratio_to_base"] == 1.25 and g8["cut_share_diagnostic"] == 0.0
    assert (g8["gain_serving_canonical"], g8["passes_canonical_serving"], g8["gain_diagnostic_canonical"], g8["acquired_diagnostic"]) == (29, True, 29, True)
    assert "passes_at_diagnostic" not in g8
    assert (_row(rows, "g8-tool-r1")["gain_per_100"], _row(rows, "g32-tool-r1")["gain_per_100"]) == (30.0, 20.0)
    g32 = _row(rows, "g32-chem-r1")
    assert (g32["gain_serving_strict"], g32["gain_per_100"], g32["acquired"]) == (2, 2.0, False)
    assert (g32["gain_serving_canonical"], g32["passes_canonical_serving"], g32["acquired_diagnostic"]) == (1, False, False)
    assert g32["note"] is None, "no canonical reading passes either"
    missing = _row(rows, "sfrz-chem-r1")
    assert missing["found"] == 0 and missing["acquired"] is None and missing["note"] == "missing"
    assert "| g32-chem-r1 | chemistry | 32 | 34 | 33 | 35 | +2 | +2.0 | no | +1.0 | no | +0.0 | no |" in md
    assert "| sfrz-tool-r1 | toolalpaca | - | - | - | - | - | - | - |" in md


def test_retention_rows_and_the_undefined_marking(full):
    _work, report, md = full
    rows = report["retention"]
    assert len(rows) == 12
    g8 = _row(rows, "g8-chemtool-r1")
    assert (g8["F"], g8["residual"], g8["budget"], g8["extraction"]) == (20, 4, 13, 3)
    assert g8["F"] == g8["residual"] + g8["budget"] + g8["extraction"]
    assert g8["scores"] == {"strict_serving": 40, "canonical_serving": 45, "strict_diagnostic": 58, "canonical_diagnostic": 60}
    assert g8["first_task_acquired"] is True and g8["defined"] == 1 and g8["retained_share"] == 0.6667 and g8["status"] == "defined"
    assert g8["reference"] == "g8-chem-r1" and g8["task"] == "chemistry"
    assert g8["first_task_acquired_diagnostic"] is True and g8["defined_diagnostic"] == 1
    assert g8["access_sensitive"] is True                       # 20 of 100 >= 5 per 100, and 2 x (13 + 3) = 32 >= 20
    toolchem = _row(rows, "g8-toolchem-r1")
    assert toolchem["task"] == "toolalpaca" and toolchem["reference"] == "g8-tool-r1" and toolchem["F"] == 5
    assert toolchem["access_sensitive"] is False                # 2 x (budget 2 + extraction 0) = 4 < 5
    not_learned = _row(rows, "g32-chemtool-r1")
    assert not_learned["F"] == 7 and not_learned["scored"] == 1, "the accounting is still printed"
    assert not_learned["defined"] == 0 and not_learned["retained_share"] is None and not_learned["access_sensitive"] is None
    assert not_learned["status"] == "not acquired: retention is not defined"
    assert not_learned["first_task_acquired_diagnostic"] is False and not_learned["defined_diagnostic"] == 0
    assert "| g32-chemtool-r1 | chemistry | g32-chem-r1 | no | +7 |" in md
    assert md.count("not acquired: retention is not defined") >= 2                 # the definition and the row
    gone = _row(rows, "g8-toolchem-r2")
    assert gone["scored"] == 0 and gone["defined"] == 0 and gone["status"].startswith("missing: ")
    assert _row(rows, "sfrz-toolchem-r1")["status"].startswith("missing: ")


def test_second_task(full):
    _work, report, _md = full
    row = _row(report["second_task"], "g8-chemtool-r1")
    assert row["task"] == "toolalpaca" and row["alone"] == "g8-tool-r1"
    assert (row["difference_serving_strict"], row["difference_per_100"]) == (-2, -4.0) and row["note"] is None
    assert row["chain_scores"]["scores"]["strict_serving"] == 28 and row["alone_scores"]["scores"]["strict_serving"] == 30
    other = _row(report["second_task"], "g8-toolchem-r1")
    assert other["task"] == "chemistry" and other["alone"] == "g8-chem-r1" and other["difference_serving_strict"] is None
    assert other["note"] == "missing" and other["chain_scores"]["found"] == 0


def test_retention_by_budget_per_cohort_and_pairs_on_shared_rows(full):
    _work, report, md = full
    rb = report["retention_by_budget"]
    assert set(rb) == {"serving_cohort", "diagnostic_cohort"}
    t = rb["serving_cohort"]
    assert t["budgets"] == [512, 2048, 8192], "4096 is read by one sweep only, so it is not a common budget"
    assert "order_serving" not in t and "order_diagnostic" not in t and "reversed_pairs" not in t
    assert set(t["recipes"]) == {"g8", "g32", "sema"}
    g8 = t["recipes"]["g8"]
    assert g8["rows"] == 3 and sorted(g8["keys"]) == ["g8-chemtool-r1", "g8-chemtool-r2", "g8-toolchem-r1"]
    assert g8["comparable"] is False
    assert g8["mean_loss_per_100"] == {"512": 11.333333, "2048": 13.333333, "8192": 4.0}
    assert g8["mean_loss"] == {"512": 10.666667, "2048": 11.666667, "8192": 3.0}          # questions: (20+2+10)/3 ...
    assert t["recipes"]["g32"]["mean_loss_per_100"] == {"512": 2.0, "2048": 10.0, "8192": 2.0}
    assert t["recipes"]["sema"]["mean_loss_per_100"] == {"512": 5.0, "2048": 5.0, "8192": 8.0}
    assert _row(t["rows"], "g8-toolchem-r1")["loss_per_100"] == {"512": 4.0, "2048": 10.0, "8192": 6.0}
    pairs = {(p["a"], p["b"]): p for p in t["pairs"]}
    assert len(pairs) == 6                                         # every pair of the four recipes, in RECIPES order
    reverses = pairs[("g8", "sema")]
    assert reverses["shared"] == ["chemtool-r1"] and reverses["n_shared"] == 1
    assert reverses["mean_loss_per_100"] == {"g8": {"512": 20.0, "2048": 20.0, "8192": 4.0}, "sema": {"512": 5.0, "2048": 5.0, "8192": 8.0}}
    assert (reverses["order_serving"], reverses["order_diagnostic"], reverses["reversed"]) == ("sema", "g8", True)
    tie = pairs[("g8", "g32")]
    assert tie["shared"] == ["toolchem-r1"] and (tie["order_serving"], tie["order_diagnostic"], tie["reversed"]) == ("tie", "g32", False)
    none = pairs[("g32", "sema")]
    assert none == {"a": "g32", "b": "sema", "shared": [], "n_shared": 0, "mean_loss_per_100": None,
                    "order_serving": None, "order_diagnostic": None, "reversed": None}
    assert t["ranking_differs"] is True
    assert [r["key"] for r in rb["diagnostic_cohort"]["rows"]] == [r["key"] for r in t["rows"]]
    assert "descriptive" in md and "needs independent confirmation" in md
    assert md.count("is over its own rows and is not comparable across recipes") == 2      # once per cohort
    assert "| g8 | sema | chemtool-r1 | +20.00 | +5.00 | +4.00 | +8.00 | sema | g8 | yes |" in md
    assert "| g32 | sema | none | - | - | - | - | - | - | - |" in md


def test_a_ranking_that_does_not_differ(tmp_path):
    work = build_tree(tmp_path / "work", {k: v for k, v in TREE.items() if not k[0].startswith("sema")})
    t = report_of(work, tmp_path / "report")["retention_by_budget"]["serving_cohort"]
    assert set(t["recipes"]) == {"g8", "g32"}
    assert [(p["a"], p["b"]) for p in t["pairs"] if p["n_shared"]] == [("g8", "g32")]
    pair = next(p for p in t["pairs"] if (p["a"], p["b"]) == ("g8", "g32"))
    assert (pair["order_serving"], pair["reversed"]) == ("tie", False), "a tie is not a reversal"
    assert t["ranking_differs"] is False


def test_panel_against_base(full):
    _work, report, md = full
    panel = report["panel"]
    assert panel["panels"] == ["math", "knowledge"]
    g8 = _row(panel["rows"], "g8-chem-r1")
    assert g8["panels"]["math"] == {"serving": 28, "diagnostic": 34, "serving_minus_base": -2, "diagnostic_minus_base": -1}
    assert g8["panels"]["total"]["serving"] == 69 and g8["panels"]["total"]["serving_minus_base"] == -1
    assert _row(panel["rows"], "g32-chem-r1")["found"] == 0
    assert "| g8-chem-r1 | 28 / 34 (-2, -1) | 41 / 40 (+1, +0) | 69 / 74 (-1, -1) |" in md


def test_training_from_metrics_summary_and_rollout_stats(full):
    _work, report, md = full
    t = _row(report["training"], "g8-chem-r1")
    assert t["attempt"] == 2 and t["merged_attempt"] == 2 and t["gpu_hours"] == 8.0 and t["seconds"] == 7200 and t["mini_batch"] == 8
    assert (t["length_mean_first"], t["length_mean_step_20"], t["length_mean_last"]) == (110.0, 300.0, 350.0)
    assert (t["length_mean_max"], t["length_mean_max_step"], t["clip_ratio_max"]) == (999.0, 22, 0.25)
    assert (t["score_first"], t["score_last"], t["metrics_steps"], t["metrics_unreadable_lines"]) == (0.01, 0.25, 25, 1)
    assert [(v["step"], v["accuracy"]) for v in t["validation"]] == [(10, 0.41), (25, 0.52)]
    assert (t["no_spread_first"], t["no_spread_last"]) == (1.0, 0.0)
    retried = _row(report["training"], "g8-chemtool-r1")
    assert (retried["attempt"], retried["returncode"], retried["merged"]) == (1, 0, 1), "the merged attempt, not the failed retry"
    empty = _row(report["training"], "sfrz-chem-r1")
    assert empty["found"] == 1 and empty["attempt"] == 1 and empty["merged_attempt"] is None, "no merged attempt: the highest folder"
    assert empty["metrics_steps"] == 0 and empty["gpu_hours"] is None and empty["length_mean_first"] is None
    assert _row(report["training"], "sfrz-tool-r1")["found"] == 0
    assert "sampled, 16 answers a question, the authors' protocol; not comparable with the greedy scores above" in md
    assert "| g8-chem-r1 | 10: 0.4100, 25: 0.5200 |" in md


def test_the_text_states_and_never_concludes(full):
    _work, _report, md = full
    low = md.lower()
    assert "proves" not in low and "caused" not in low and "learned" not in low
    for heading in ("## Prefix checks", "## 1. Base", "## 2. Acquisition", "## 3. Retention", "## 4. Second task",
                    "## 5. Retention by budget", "## 6. Panel", "## 7. Training", "## 8. Decisions"):
        assert heading in md
    assert md.index("## Prefix checks") < md.index("## 1. Base")
    assert "| base8b | chemistry | 1 | 100 | 1.0000 | yes | yes | yes | pass |" in md


def test_the_two_caveats_are_printed_verbatim(full):
    _work, _report, md = full
    tool = ("ToolAlpaca has 68 held-out questions: a paired difference under about 12 per 100 is inside noise, and every "
            "ToolAlpaca acquisition is to be read with that.")
    recipes = ("The four recipes differ in learning rate, minibatch and teacher at once, and have one or two runs each: "
               "nothing here identifies an effect of an algorithm.")
    assert md.count(tool) == 1 and md.index("## 2. Acquisition") < md.index(tool) < md.index("## 3. Retention")
    assert md.count(recipes) == 3, "at the top, and above the pairs table of each cohort"
    assert md.index(recipes) < md.index("## Prefix checks")
    after_5 = md[md.index("## 5. Retention by budget"):]
    assert after_5.count(recipes) == 2 and after_5.index(recipes) < after_5.index("| a | b | shared rows |")


def test_a_partial_tree_still_reports(tmp_path):
    work = tmp_path / "work"
    bed_sweep(work, "base8b", "chemistry", TREE[("base8b", "chemistry")])
    bed_sweep(work, "g8-chem-r1", "chemistry", TREE[("g8-chem-r1", "chemistry")])
    merge_runs(work, ["g8-chem-r1"])
    (work / "k8b" / "report-sweep" / "g8-tool-r1-toolalpaca-a1").mkdir()            # a sweep folder with no sweep.json
    report = report_of(work, tmp_path / "report")
    assert (report["runs_found"], report["stage1_scored"], report["stage2_scored"], report["retention_rows_defined"]) == (1, 1, 0, 0)
    assert report["accounting_exact"] == 1 and report["lineage_ok"] == 1 and report["base"]["toolalpaca"]["found"] == 0
    for cohort in ("serving_cohort", "diagnostic_cohort"):
        t = report["retention_by_budget"][cohort]
        assert t["budgets"] == [] and t["ranking_differs"] is None and t["recipes"] == {}
    assert _row(report["acquisition"], "g8-tool-r1")["found"] == 0
    md = (tmp_path / "report" / "pilot-report.md").read_text()
    assert "No retention row is defined" in md and "not defined (no pair of recipes shares a defined row)" not in md


def test_one_recipe_gives_no_ranking(tmp_path):
    work = build_tree(tmp_path / "work", {k: v for k, v in TREE.items() if k[0] == "base8b" or k[0].startswith("g8-")})
    t = report_of(work, tmp_path / "report")["retention_by_budget"]["serving_cohort"]
    assert set(t["recipes"]) == {"g8"} and t["ranking_differs"] is None
    assert all(p["n_shared"] == 0 for p in t["pairs"])
    assert "not defined (no pair of recipes shares a defined row)" in (tmp_path / "report" / "pilot-report.md").read_text()


def test_pilot_report_refusals(tmp_path):
    work = tmp_path / "work"
    bed_sweep(work, "g8-chem-r1", "chemistry", TREE[("g8-chem-r1", "chemistry")])
    with pytest.raises(SystemExit, match="no sweep of base8b"):
        pilot_report.main(["--work", str(work), "--serving", "2048", "--diagnostic", "8192", "--out", str(tmp_path / "o1")])
    assert not (tmp_path / "o1").exists()
    bed_sweep(work, "base8b", "chemistry", TREE[("base8b", "chemistry")])
    with pytest.raises(SystemExit, match="must be below"):
        pilot_report.main(["--work", str(work), "--serving", "8192", "--diagnostic", "2048", "--out", str(tmp_path / "o2")])
    (tmp_path / "taken").mkdir()
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        pilot_report.main(["--work", str(work), "--serving", "2048", "--diagnostic", "8192", "--out", str(tmp_path / "taken")])
    with pytest.raises(SystemExit, match="no such work tree"):
        pilot_report.main(["--work", str(tmp_path / "nowhere"), "--serving", "2048", "--diagnostic", "8192", "--out", str(tmp_path / "o3")])


def test_other_items_are_not_compared(tmp_path):
    work = tmp_path / "work"
    bed_sweep(work, "base8b", "chemistry", TREE[("base8b", "chemistry")])
    folder = bed_sweep(work, "g32-chem-r1", "chemistry", TREE[("g32-chem-r1", "chemistry")])
    merge_runs(work, ["g32-chem-r1"])
    sweep = json.loads((folder / "sweep.json").read_text())
    sweep["items_sha256"] = "x" * 64
    (folder / "sweep.json").write_text(json.dumps(sweep))
    g32 = _row(report_of(work, tmp_path / "report")["acquisition"], "g32-chem-r1")
    assert g32["scored"] == 0 and g32["acquired"] is None and "other items" in g32["note"]


# ------------------------------------------------------------------------------------- one machine
def test_different_machines_are_never_compared_unless_allowed(tmp_path):
    machines = {("g8-chem-r1", "chemistry"): "m2", ("g8-chemtool-r1", "toolalpaca"): "m3", ("g32-chem-r1", "chemistry"): None,
                ("g8-chem-r1", "panel"): "m2"}
    work = build_tree(tmp_path / "work", machines=machines)
    report = report_of(work, tmp_path / "refused")
    md = (tmp_path / "refused" / "pilot-report.md").read_text()
    assert report["different_machines_allowed"] == 0 and report["machine_ids"] == ["m1", "m2", "m3"]
    acquired = _row(report["acquisition"], "g8-chem-r1")
    assert acquired["scored"] == 0 and acquired["acquired"] is None and acquired["acquired_diagnostic"] is None
    assert acquired["note"] == "not compared: scored on different machines (m1 vs m2)"
    assert acquired["scores"]["strict_serving"] == 60, "the run's own scores stay visible"
    assert _row(report["acquisition"], "g32-chem-r1")["note"] == "not compared: scored on different machines (m1 vs None)"
    kept = _row(report["retention"], "g8-chemtool-r1")
    assert kept["scored"] == 0 and kept["defined"] == 0 and kept["status"] == "not compared: scored on different machines (m2 vs m1)"
    assert _row(report["retention"], "g32-chemtool-r1")["status"] == "not compared: scored on different machines (None vs m1)"
    assert _row(report["retention"], "g8-chemtool-r2")["defined"] == 1, "one machine: still compared"
    second = _row(report["second_task"], "g8-chemtool-r1")
    assert second["difference_serving_strict"] is None and second["note"] == "not compared: scored on different machines (m1 vs m3)"
    assert second["chain_scores"]["scores"]["strict_serving"] == 28
    panel = _row(report["panel"]["rows"], "g8-chem-r1")
    assert panel["compared_with_base"] is False and panel["note"] == "not compared: scored on different machines (m1 vs m2)"
    assert panel["panels"]["math"]["serving"] == 28 and panel["panels"]["math"]["serving_minus_base"] is None
    t = report["retention_by_budget"]["serving_cohort"]
    assert "g8-chemtool-r1" not in [r["key"] for r in t["rows"]], "no ranking reads a row across machines"
    assert "more than one machine" in md

    allowed = report_of(work, tmp_path / "allowed", "--allow-different-machines")
    assert allowed["different_machines_allowed"] == 1
    acquired = _row(allowed["acquisition"], "g8-chem-r1")
    assert acquired["scored"] == 1 and acquired["acquired"] is True and acquired["same_machine"] is False and "another machine" in acquired["note"]
    kept = _row(allowed["retention"], "g8-chemtool-r1")
    assert kept["scored"] == 1 and kept["defined"] == 1 and kept["same_machine"] is False
    second = _row(allowed["second_task"], "g8-chemtool-r1")
    assert second["difference_serving_strict"] == -2 and second["note"] == "scored on different machines (m1 vs m3)"
    panel = _row(allowed["panel"]["rows"], "g8-chem-r1")
    assert panel["compared_with_base"] is True and panel["panels"]["math"]["serving_minus_base"] == -2
    assert "--allow-different-machines was given" in (tmp_path / "allowed" / "pilot-report.md").read_text()


# ----------------------------------------------------------------------------------------- lineage
def test_lineage_reads_the_end_of_a_model_path():
    assert pilot_report.lineage("/work/k8b-work/runs/g8-chem-r1-a2/hf-step40") == ("g8-chem-r1", 2)
    assert pilot_report.lineage("/work/k8b-work/runs/g8-chem-r1-a12/hf-step40/") == ("g8-chem-r1", 12)
    assert pilot_report.lineage("/work/k8b-work/runs/g8-chem-r1-a2/hf-step40/extra") is None
    assert pilot_report.lineage(BASE_MODEL) is None and pilot_report.lineage(None) is None


def test_a_sweep_of_an_earlier_attempt_of_a_retried_run_is_not_used(tmp_path):
    work = build_tree(tmp_path / "work")
    write_run(work, "g8-tool-r1", 2, summary={"returncode": 0, "merged": 1, "model_dir": BASE_MODEL})   # retried and merged again
    edit_sweep(work, "g32-tool-r1-toolalpaca", 1, lambda s: s.pop("model"))
    edit_sweep(work, "base8b-toolalpaca", 1, lambda s: s.pop("model"))       # a base sweep with no `model` is still the base
    report = report_of(work, tmp_path / "report")
    retried = "g8-tool-r1 toolalpaca sweep a1 scored attempt 1 of the run; the run's merged attempt is 2"
    unnamed = "g32-tool-r1 toolalpaca sweep a1 has no `model` field"
    # ... and the stage that was trained on top of the OLD attempt is no longer a continuation of the run of record
    orphan = "g8-toolchem-r1 was trained from /work/k8b-work/runs/g8-tool-r1-a1/hf-step40, but the scored attempt of g8-tool-r1 is 2"
    assert report["lineage_ok"] == 0 and sorted(report["lineage_problems"]) == sorted([retried, unnamed, orphan])
    assert report["base"]["toolalpaca"]["found"] == 1
    row = _row(report["acquisition"], "g8-tool-r1")
    assert row["found"] == 0 and row["scored"] == 0 and row["note"] == "lineage: " + retried
    assert _row(report["acquisition"], "g32-tool-r1")["note"] == "lineage: " + unnamed
    assert _row(report["retention"], "g8-toolchem-r1")["status"].startswith("lineage: ")
    second = _row(report["second_task"], "g8-chemtool-r1")
    assert second["difference_serving_strict"] is None and second["note"] == "lineage: " + retried
    assert _row(report["training"], "g8-tool-r1")["attempt"] == 2
    assert "**Lineage: these sweeps or rows are not used:" in (tmp_path / "report" / "pilot-report.md").read_text()


def test_a_stage_two_run_trained_from_an_earlier_attempt_of_its_first_stage(tmp_path):
    work = build_tree(tmp_path / "work")
    (work / "runs" / "g8-chemtool-r1-a1" / "run-summary.json").write_text(json.dumps(
        {"returncode": 0, "merged": 1, "model_dir": model_of("g8-chem-r1", 1)}))
    report = report_of(work, tmp_path / "report")
    why = ("g8-chemtool-r1 was trained from /work/k8b-work/runs/g8-chem-r1-a1/hf-step40, but the scored attempt of "
           "g8-chem-r1 is 2")
    assert report["lineage_ok"] == 0 and report["lineage_problems"] == [why]
    row = _row(report["retention"], "g8-chemtool-r1")
    assert row["scored"] == 0 and row["defined"] == 0 and row["defined_diagnostic"] == 0 and "F" not in row
    assert row["status"] == "lineage: " + why
    assert report["retention_rows_defined"] == 4
    # round 2 of the review: the parent is checked before the run enters ANY comparison, not only its retention row
    second = _row(report["second_task"], "g8-chemtool-r1")
    assert second["difference_serving_strict"] is None and second["note"] == "lineage: " + why
    # with g8's first-run chemtool row unscored and every r2 row but one missing, whether a loss repeats is not settled
    assert report["decision"]["on_to_package_4"] is None


# ------------------------------------------------------------------------------------ prefix checks
def test_the_prefix_check_rule(tmp_path):
    work = tmp_path / "work"
    write_check(work, "prefix-base8b-chemistry-a1.json", agreement=0.99)                     # exactly the bar: passes
    write_check(work, "prefix-base8b-toolalpaca-a1.json", same_machine=False)
    write_check(work, "prefix-g8-chem-r1-chemistry-a1.json", compared=0, agreement=None)
    write_check(work, "prefix-g8-tool-r1-toolalpaca-a1.json", same_length=False)
    write_check(work, "prefix-g32-chem-r1-chemistry-a1.json", agreement=0.98)
    write_check(work, "repeat-base8b-chemistry-a1.json", compared=0)                          # not a prefix check
    checks = pilot_report.read_prefix_checks(work / "k8b" / "report-prefix")
    assert {k: c["passed"] for k, c in checks.items()} == {
        ("base8b", "chemistry"): True, ("base8b", "toolalpaca"): False, ("g8-chem-r1", "chemistry"): False,
        ("g8-tool-r1", "toolalpaca"): False, ("g32-chem-r1", "chemistry"): False}


def test_a_failed_prefix_check_leaves_that_sweep_out_and_keeps_the_others(tmp_path):
    work = build_tree(tmp_path / "work")
    write_check(work, "prefix-base8b-chemistry-a1.json")
    write_check(work, "prefix-g8-tool-r1-toolalpaca-a1.json")
    write_check(work, "prefix-g8-tool-r1-toolalpaca-a2.json", compared=50, agreement=0.95)     # the highest attempt wins
    write_check(work, "repeat-base8b-chemistry-a1.json", compared=0)
    report = report_of(work, tmp_path / "report")
    md = (tmp_path / "report" / "pilot-report.md").read_text()
    assert (report["prefix_checks_found"], report["prefix_checks_failed"], report["lineage_ok"]) == (5, 1, 1)
    failed = [c for c in report["prefix_checks"] if not c["passed"]]
    assert [(c["key"], c["task"], c["attempt"], c["text_agreement"]) for c in failed] == [("g8-tool-r1", "toolalpaca", 2, 0.95)]
    why = "prefix reuse failed for g8-tool-r1 on toolalpaca (agreement 0.95)"
    row = _row(report["acquisition"], "g8-tool-r1")
    assert row["found"] == 0 and row["note"] == why
    assert _row(report["acquisition"], "g8-chem-r1")["scored"] == 1, "the base chemistry sweep passed and is used"
    assert report["stage1_scored"] == 5
    assert _row(report["retention"], "g8-toolchem-r1")["status"] == why
    assert _row(report["second_task"], "g8-chemtool-r1")["note"] == why
    assert "| g8-tool-r1 | toolalpaca | 2 | 50 | 0.9500 | yes | yes | no | FAIL |" in md
    assert "**Left out of every table because prefix reuse failed: the toolalpaca sweep of g8-tool-r1.**" in md


# -------------------------------------------------------------------------------------- the cohorts
def test_a_task_learned_only_in_long_answers_is_marked_and_stays_not_acquired(tmp_path):
    """Found by the full dry run (scripts/simulate_campaigns.py): a run whose answers outgrow the serving budget while
    it trains shows no gain at S(2048) and a large one at E(8192). `acquired` must stay the pre-registered S(B) rule;
    the canonical readings are printed beside it, the row says where its gain is visible, and its retention row enters
    the diagnostic cohort only, never the serving one."""
    tree = {("base8b", "chemistry"): pts((20, 22), (30, 33), (32, 35)),
            ("g32-chem-r1", "chemistry"): pts((20, 21), (28, 30), (70, 75)),          # S(B) -2; E(B) 30 - 33 = -3; E(H) 75 - 35 = +40
            ("g32-chemtool-r1", "chemistry"): pts((15, 16), (20, 22), (60, 66)),
            ("g8-chem-r1", "chemistry"): pts((40, 42), (60, 62), (62, 64)),             # S(B) +30; E(H) 64 - 35 = +29
            ("sema-chem-r1", "chemistry"): pts((20, 21), (33, 40), (34, 38))}          # S(B) +3; E(B) 40 - 33 = +7; E(H) +3
    work = tmp_path / "work"
    for (key, task), points in tree.items():
        bed_sweep(work, key, task, points)
    merge_runs(work, [key for key, _task in tree])
    report = report_of(work, tmp_path / "report")
    md = (tmp_path / "report" / "pilot-report.md").read_text()
    g32, g8, sema = (_row(report["acquisition"], k) for k in ("g32-chem-r1", "g8-chem-r1", "sema-chem-r1"))
    assert (g32["acquired"], g32["gain_per_100"], g32["gain_serving_canonical"], g32["passes_canonical_serving"]) == (False, -2.0, -3, False)
    assert (g32["gain_diagnostic_canonical"], g32["gain_diagnostic_per_100"], g32["acquired_diagnostic"]) == (40, 40.0, True)
    assert g32["note"] == "gain visible only at E(8192)" and "passes_at_diagnostic" not in g32
    assert (g8["acquired"], g8["gain_diagnostic_canonical"], g8["acquired_diagnostic"], g8["note"]) == (True, 29, True, None)
    assert (sema["acquired"], sema["gain_serving_canonical"], sema["passes_canonical_serving"], sema["acquired_diagnostic"]) == (False, 7, True, False)
    assert sema["note"] == "gain visible at E(2048), not at S(2048)"
    assert "| g32-chem-r1 | chemistry | 28 | 30 | 70 | 75 | -2 | -2.0 | no | -3.0 | no | +40.0 | yes |" in md
    assert md.count("gain visible only at E(8192)") == 1 and md.count("gain visible at E(2048), not at S(2048)") == 1
    assert "learned" not in md.replace(report["work"], "").lower(), "the tmp folder carries this test's name"
    kept =_row(report["retention"], "g32-chemtool-r1")
    assert kept["status"] == "not acquired: retention is not defined" and kept["defined"] == 0 and kept["retained_share"] is None
    assert (kept["F"], kept["residual"]) == (8, 9), "its accounting is still printed"
    assert kept["first_task_acquired_diagnostic"] is True and kept["defined_diagnostic"] == 1
    assert (report["retention_rows_defined"], report["retention_rows_defined_diagnostic"]) == (0, 1)
    rb = report["retention_by_budget"]
    assert rb["serving_cohort"]["rows"] == [] and rb["serving_cohort"]["budgets"] == []
    (row,) = rb["diagnostic_cohort"]["rows"]
    assert row["key"] == "g32-chemtool-r1" and row["loss"] == {"512": 5, "2048": 8, "8192": 10}
    assert rb["diagnostic_cohort"]["ranking_differs"] is None
    assert "| g32-chemtool-r1 | chemistry | g32-chem-r1 | no | +8 |" in md and "| defined | not acquired: retention is not defined |" in md


# ------------------------------------------------------------------------------------- the decisions
def stage1(values: dict) -> list:
    """Rows of table 2: {key: True / False (scored) or None (not scored)}."""
    return [{"key": key, "scored": int(value is not None), "acquired": value} for key, value in values.items()]


def stage2(key: str, F=None, n=100, budget=0, extraction=0, first=True) -> dict:
    """A row of table 3, defined when F is given and its first task was acquired."""
    recipe, chain, rep = key.split("-")
    defined = int(F is not None and first is True)
    row = {"key": key, "recipe": recipe, "chain": chain, "replicate": rep, "defined": defined, "first_task_acquired": first,
           "F": F, "n": n, "budget": budget, "extraction": extraction}
    row["access_sensitive"] = pilot_report.access_sensitive(row) if defined else None
    return row


BOTH_G8 = {"g8-chem-r1": True, "g8-tool-r1": True, "sema-chem-r1": True, "sema-tool-r1": False}
SETTLED = [stage2("g8-toolchem-r1", first=False), stage2("sema-chemtool-r1", F=0), stage2("sema-toolchem-r1", first=False)]


def test_on_to_package_4_when_both_halves_hold(full):
    d = pilot_report.decision_of(stage1(BOTH_G8), [stage2("g8-chemtool-r1", F=10, budget=5), stage2("g8-chemtool-r2", F=6)], B)
    assert d["on_to_package_4"] is True and d["published_acquire_both_first_run"] == {"g8": True, "sema": False}
    entry = next(e for e in d["loss_repeats"] if (e["recipe"], e["chain"]) == ("g8", "chemtool"))
    assert entry == {"recipe": "g8", "chain": "chemtool", "both_defined": True, "both_lose_5_per_100": True,
                     "both_access_sensitive": False}                  # r1: 2 x 5 >= 10; r2: 0 < 6
    assert (d["any_defined_loss_5_per_100"], d["losses_only_in_g32"]) == (True, False)
    # the whole tree: g8 acquires both in r1, and its chemtool rows lose 20 and 10 per 100
    _work, report, md = full
    decision = report["decision"]
    assert decision["on_to_package_4"] is True and decision["published_acquire_both_first_run"] == {"g8": True, "sema": None}
    assert next(e for e in decision["loss_repeats"] if (e["recipe"], e["chain"]) == ("g8", "chemtool"))["both_access_sensitive"] is True
    assert "These are the pre-registered rules" in md and "they decide nothing by themselves" in md
    assert "- On to package 4: yes." in md


def test_not_on_when_no_published_recipe_acquires_both():
    d = pilot_report.decision_of(stage1({"g8-chem-r1": True, "g8-tool-r1": False, "sema-chem-r1": False, "sema-tool-r1": True}), [], B)
    assert d["on_to_package_4"] is False and "no published recipe acquires both tasks" in d["why"]
    assert d["published_acquire_both_first_run"] == {"g8": False, "sema": False} and d["any_defined_loss_5_per_100"] is None


def test_not_on_when_no_loss_repeats():
    rows = [stage2("g8-chemtool-r1", F=10), stage2("g8-chemtool-r2", F=2)] + SETTLED       # r2 loses 2 per 100: not 5
    d = pilot_report.decision_of(stage1(BOTH_G8), rows, B)
    assert d["on_to_package_4"] is False and "no acquired task loses 5 per 100 in both runs of a recipe" in d["why"]
    assert (d["any_defined_loss_5_per_100"], d["losses_only_in_g32"]) == (True, False)


def test_not_on_when_the_only_losses_are_in_g32():
    rows = [stage2("g8-chemtool-r1", F=2), stage2("g8-chemtool-r2", F=2), stage2("g32-chemtool-r1", F=20)] + SETTLED
    d = pilot_report.decision_of(stage1(BOTH_G8), rows, B)
    assert d["on_to_package_4"] is False and "the only losses are in g32" in d["why"]
    assert (d["any_defined_loss_5_per_100"], d["losses_only_in_g32"]) == (True, True)


def test_not_decided_when_data_are_missing():
    d = pilot_report.decision_of(stage1({"g8-chem-r1": True}), [], B)
    assert d["on_to_package_4"] is None and d["published_acquire_both_first_run"] == {"g8": None, "sema": None}
    assert "missing: whether g8 acquires both tasks in its first run (g8-tool-r1 not scored)" in d["why"]
    d = pilot_report.decision_of(stage1(BOTH_G8), [stage2("g8-chemtool-r1", F=10)], B)        # r2 of every chain unknown
    assert d["on_to_package_4"] is None
    assert any(w.startswith("missing: whether a loss repeats in g8 chemtool, g8 toolchem") for w in d["why"])


# ------------------------------------------------------------------------------------------- export
def test_the_report_needs_only_the_kit(tmp_path):
    """kit/ exported alone: pilot_report loads budget_report by path and imports nothing from scripts/."""
    kit = tmp_path / "kit"
    kit.mkdir()
    for name in ("pilot_report", "budget_report", "rollout_stats"):
        shutil.copy(KIT / ("%s.py" % name), kit / ("%s.py" % name))
    spec = importlib.util.spec_from_file_location("kit_pilot_report_exported", kit / "pilot_report.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert Path(module.budget_report.__file__).resolve().parent == kit.resolve()


def test_rollout_statistics_are_found_where_the_campaign_writes_them(tmp_path, monkeypatch):
    """kit/campaigns/k8b-pilot.yaml writes them under <work>/k8b/report-rollouts/<key>-a<N>/, the highest attempt wins,
    and the file carries `steps_read` at the top level for the campaign's bar."""
    work = tmp_path / "work"
    run = work / "runs" / "g8-chem-r1-a1"
    (run / "rollouts").mkdir(parents=True)
    (run / "run-summary.json").write_text(json.dumps({"name": "g8-chem-r1-a1", "returncode": 0, "merged": 1, "n_gpus": 8, "seconds": 3600}))
    (run / "metrics.jsonl").write_text(json.dumps({"step": 1, "data": {"response_length/mean": 100.0, "critic/score/mean": 0.5}}) + "\n")
    lines = [{"input": "q%d" % (i // 2), "output": "one two three", "score": float(i % 2), "step": 1} for i in range(8)]
    (run / "rollouts" / "1.jsonl").write_text("".join(json.dumps(line) + "\n" for line in lines))
    import types
    monkeypatch.setattr(rollout_stats, "load_tokenizer", lambda model: types.SimpleNamespace(encode=lambda text, add_special_tokens=False: text.split()))
    out = work / "k8b" / "report-rollouts" / "g8-chem-r1-a2"
    assert rollout_stats.main(["--run", str(run), "--model", "unused", "--out", str(out)]) == 0
    stats = json.loads((out / "rollout-stats.json").read_text())
    assert stats["steps_read"] == 1 and not isinstance(stats["steps_read"], bool)
    row = pilot_report.training_of(pilot_report.Work(work), "g8-chem-r1")
    assert row["rollout_stats_found"] == 1


# ------------------------------------------------------------- round 2 of the review: required, bound checks; parents; three values
def test_a_scheduled_prefix_check_that_is_missing_leaves_its_sweep_out(tmp_path):
    """A missing check used to disqualify nothing, so a direct scoring that failed before writing its check let the
    unqualified sweep into the tables while the two base checks satisfied the count bar."""
    work = build_tree(tmp_path / "work")
    write_check(work, "prefix-base8b-chemistry-a1.json")
    write_check(work, "prefix-base8b-toolalpaca-a1.json")
    write_check(work, "prefix-sema-chemtool-r1-chemistry-a1.json")                      # g8-chemtool-r1's is the one missing
    report = report_of(work, tmp_path / "report", qualified=False)
    assert report["prefix_checks_missing"] == ["g8-chemtool-r1 on chemistry"] and report["prefix_checks_missing_count"] == 1
    assert _row(report["retention"], "g8-chemtool-r1")["status"] == "prefix check missing for g8-chemtool-r1 on chemistry: its curve is not qualified"
    assert _row(report["retention"], "g8-chemtool-r1")["scored"] == 0 and _row(report["retention"], "sema-chemtool-r1")["scored"] == 1
    assert _row(report["acquisition"], "g8-tool-r1")["scored"] == 1, "a sweep the campaign schedules no check for is used without one"
    assert "**Left out of every table because its scheduled prefix check is missing: g8-chemtool-r1 on chemistry.**" in (tmp_path / "report" / "pilot-report.md").read_text()


def test_with_no_check_of_the_untrained_model_nothing_is_compared(tmp_path):
    work = build_tree(tmp_path / "work")
    report = report_of(work, tmp_path / "report", qualified=False)
    assert report["stage1_scored"] == 0 and report["prefix_checks_missing_count"] == 4
    assert report["retention_rows_defined"] == 0 and report["retention_rows_defined_diagnostic"] == 0, "no acquisition, so no cohort"
    assert report["base"]["chemistry"]["found"] == 0


@pytest.mark.parametrize("kwargs, said", [
    ({"long": "/node/work/k8b/eval/base8b-chemistry-a7"}, "base8b-chemistry-a7"),                   # another attempt's scoring
    ({"long": "/node/work/k8b/pilot/prefix-chemistry-long-a1"}, "prefix-chemistry-long-a1"),       # another generation altogether
    ({"short_cap": 1024}, "caps 1024 and 8192"), ({"long_cap": 4096}, "caps 2048 and 4096")])
def test_a_check_counts_only_for_the_scoring_it_was_made_on(tmp_path, kwargs, said):
    """An old passing check must not qualify a newer scoring, and the identity must survive the tree being packed and
    moved: it is the scoring folder's NAME and the two caps, never an absolute path."""
    work = build_tree(tmp_path / "work")
    write_check(work, "prefix-base8b-chemistry-a1.json", **kwargs)
    report = report_of(work, tmp_path / "report")
    check = next(c for c in report["prefix_checks"] if (c["key"], c["task"]) == ("base8b", "chemistry"))
    assert check["passed"] is True and check["bound"] is False and check["used"] is False and report["prefix_checks_failed"] == 1
    assert report["base"]["chemistry"]["found"] == 0 and said in _row(report["acquisition"], "g8-chem-r1")["note"]
    assert "NOT OF THIS SCORING" in (tmp_path / "report" / "pilot-report.md").read_text()


def test_the_check_binds_by_folder_name_so_a_moved_tree_still_qualifies(tmp_path):
    work = build_tree(tmp_path / "work")
    write_check(work, "prefix-base8b-chemistry-a1.json", long="/somewhere/else/entirely/base8b-chemistry-a1")
    report = report_of(work, tmp_path / "report")
    assert report["prefix_checks_failed"] == 0 and report["base"]["chemistry"]["found"] == 1


def test_a_wrong_parent_is_caught_even_when_the_reference_sweep_is_missing(tmp_path):
    """The parent check ran only inside the retention accounting, when both sweeps existed: with the first-task
    reference sweep gone, a stage-2 run trained from the wrong attempt still entered the second-task table."""
    work = build_tree(tmp_path / "work")
    (work / "runs" / "g8-chemtool-r1-a1" / "run-summary.json").write_text(json.dumps(
        {"returncode": 0, "merged": 1, "model_dir": model_of("g8-chem-r1", 1)}))
    import shutil
    for folder in (work / "k8b" / "report-sweep").glob("g8-chem-r1-chemistry-a*"):       # the reference sweep is unavailable
        shutil.rmtree(folder)
    report = report_of(work, tmp_path / "report")
    why = "g8-chemtool-r1 was trained from /work/k8b-work/runs/g8-chem-r1-a1/hf-step40, but the scored attempt of g8-chem-r1 is 2"
    assert report["lineage_ok"] == 0 and why in report["lineage_problems"]
    second = _row(report["second_task"], "g8-chemtool-r1")
    assert second["difference_serving_strict"] is None and second["note"] == "lineage: " + why


def test_acquires_both_is_false_as_soon_as_one_task_is_scored_and_not_acquired():
    """False AND unknown is false: a recipe that failed one task has not acquired both, whatever the other row is."""
    def acq(key, scored, acquired):
        return {"key": key, "scored": scored, "acquired": acquired}
    rows = [acq("g8-chem-r1", 1, False), acq("sema-tool-r1", 1, False)]                  # g8-tool-r1 and sema-chem-r1 are missing
    decision = pilot_report.decision_of(rows, [], 2048)
    assert decision["published_acquire_both_first_run"] == {"g8": False, "sema": False} and decision["on_to_package_4"] is False
    rows = [acq("g8-chem-r1", 1, True), acq("sema-tool-r1", 1, False)]                   # g8 may still acquire both: unknown
    assert pilot_report.decision_of(rows, [], 2048)["published_acquire_both_first_run"] == {"g8": None, "sema": False}


def test_access_sensitivity_is_decided_in_either_cohort(tmp_path):
    tree = {("base8b", "chemistry"): pts((20, 22), (30, 33), (32, 35)),
            ("g32-chem-r1", "chemistry"): pts((20, 21), (28, 30), (70, 75)),             # acquired at E(H) only
            ("g32-chemtool-r1", "chemistry"): pts((15, 16), (20, 22), (60, 66))}
    work = tmp_path / "work"
    for (key, task), points in tree.items():
        bed_sweep(work, key, task, points)
    merge_runs(work, ["g32-chem-r1", "g32-chemtool-r1"])
    report = report_of(work, tmp_path / "report")
    row = _row(report["retention"], "g32-chemtool-r1")
    assert (row["defined"], row["defined_diagnostic"]) == (0, 1) and (row["F"], row["budget"], row["extraction"]) == (8, -1, 0)
    assert row["access_sensitive"] is False, "F is 8 of 100 and the budget and extraction terms carry none of it: decided, not left blank"
    assert report["retention_rows_defined"] == 0, "and it is still not in the serving cohort"
