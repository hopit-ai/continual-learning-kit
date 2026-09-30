"""kit/runner.py: the pilot gate, the records, and the refusals. Pure CPU, under a second each."""
from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("kit_runner", ROOT / "kit" / "runner.py")
runner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(runner)
TOY = ROOT / "kit" / "campaigns" / "toy.yaml"


@pytest.fixture()
def work(tmp_path, monkeypatch):
    monkeypatch.setenv("WORK", str(tmp_path))
    return tmp_path


def variant(tmp_path: Path, **pilot_env) -> Path:
    """The toy campaign with the pilot's environment changed (a wrong base score, a crash)."""
    campaign = yaml.safe_load(TOY.read_text())
    campaign["rows"][0]["env"].update(pilot_env)
    for row in campaign["rows"]:
        row["command"] = [part.replace("{kit}", str(ROOT / "kit")) for part in row["command"]]
    path = tmp_path / "campaign.yaml"
    path.write_text(yaml.safe_dump(campaign))
    return path


def verdict(work: Path, row: str, attempt: int = 1) -> dict:
    return json.loads((work / "campaign" / "toy" / row / ("attempt-%d" % attempt) / "verdict.json").read_text())


def test_big_row_is_refused_before_the_pilot_has_run(work, capsys):
    assert runner.main(["run", str(TOY), "--row", "big"]) == runner.EXIT_REFUSED
    assert "needs pilot, which has not run" in capsys.readouterr().out
    assert not (work / "campaign" / "toy" / "big").exists(), "a refusal must launch and record nothing"
    assert not (work / "runs").exists()


def test_pilot_pass_unblocks_the_big_row(work):
    assert runner.main(["prepare", str(TOY), "--all"]) == runner.EXIT_OK
    assert runner.main(["run", str(TOY), "--all"]) == runner.EXIT_OK
    assert verdict(work, "pilot")["verdict"] == verdict(work, "big")["verdict"] == "PASS"
    start = json.loads((work / "campaign" / "toy" / "big" / "attempt-1" / "start.json").read_text())
    assert start["needs"] == ["pilot"] and start["campaign_sha256"] and start["env"]["STEPS"] == "17"


def test_pilot_outside_its_band_fails_and_blocks(work, tmp_path, capsys):
    path = variant(tmp_path, BASE="0.40")
    assert runner.main(["run", str(path), "--all"]) == runner.EXIT_FAILED
    v = verdict(work, "pilot")
    assert v["verdict"] == "FAIL" and "calibration" in v["reason"] and "0.4 is outside [0.555, 0.6]" in v["reason"]
    assert runner.main(["run", str(path), "--row", "big"]) == runner.EXIT_REFUSED
    assert "latest verdict is FAIL" in capsys.readouterr().out
    assert not (work / "runs" / "big-1").exists()


def test_a_crash_is_a_fail_even_when_the_bars_would_pass(work, tmp_path):
    assert runner.main(["run", str(variant(tmp_path, CRASH="1")), "--row", "pilot"]) == runner.EXIT_FAILED
    v = verdict(work, "pilot")
    assert v["verdict"] == "FAIL" and v["reason"] == "command exited 1" and all(b["ok"] for b in v["bars"])


def test_a_missing_metrics_file_is_a_fail_never_a_pass(work, tmp_path):
    campaign = yaml.safe_load(variant(tmp_path).read_text())
    campaign["rows"][0]["bars"][0]["source"] = "{work}/runs/nowhere/metrics.jsonl"
    path = tmp_path / "missing.yaml"
    path.write_text(yaml.safe_dump(campaign))
    assert runner.main(["run", str(path), "--row", "pilot"]) == runner.EXIT_FAILED
    assert "calibration: file not found" in verdict(work, "pilot")["reason"]


def test_a_second_attempt_is_a_new_directory_and_the_gate_reads_the_latest(work, tmp_path):
    assert runner.main(["run", str(variant(tmp_path, BASE="0.40")), "--row", "pilot"]) == runner.EXIT_FAILED
    fixed = variant(tmp_path, BASE="0.58")
    assert runner.main(["run", str(fixed), "--row", "pilot", "--attempt", "1"]) == runner.EXIT_REFUSED
    assert runner.main(["run", str(fixed), "--row", "pilot"]) == runner.EXIT_OK
    assert verdict(work, "pilot", 1)["verdict"] == "FAIL" and verdict(work, "pilot", 2)["verdict"] == "PASS"
    assert runner.main(["prepare", str(fixed), "--row", "big"]) == runner.EXIT_OK
    assert runner.main(["run", str(fixed), "--row", "big"]) == runner.EXIT_OK


def test_an_attempt_that_never_wrote_a_verdict_blocks(work, tmp_path):
    (work / "campaign" / "toy" / "pilot" / "attempt-1").mkdir(parents=True)
    assert runner.main(["run", str(TOY), "--row", "big"]) == runner.EXIT_REFUSED


def test_a_row_is_refused_until_its_io_is_done_and_prepare_needs_no_pilot(work, capsys):
    assert runner.main(["run", str(TOY), "--row", "pilot"]) == runner.EXIT_OK
    assert runner.main(["run", str(TOY), "--row", "big"]) == runner.EXIT_REFUSED          # pilot passed, inputs absent
    out = capsys.readouterr().out
    assert "its prepare steps have not run" in out and "no GPU needed" in out and "required paths are missing" in out
    assert not (work / "campaign" / "toy" / "big" / "attempt-1").exists(), "nothing may launch while inputs are missing"
    assert runner.main(["prepare", str(TOY), "--all"]) == runner.EXIT_OK
    assert (work / "data" / "big-inputs.txt").read_text() == "ready"
    assert runner.main(["run", str(TOY), "--row", "big"]) == runner.EXIT_OK


def test_prepare_runs_before_any_pilot_and_a_failed_preparation_blocks(work, tmp_path, capsys):
    assert runner.main(["prepare", str(TOY), "--all"]) == runner.EXIT_OK                   # no pilot has run yet
    campaign = yaml.safe_load(TOY.read_text())
    campaign["rows"][1]["prepare"] = [["python3", "-c", "raise SystemExit(3)"]]
    campaign["rows"][1]["requires"] = []
    campaign["name"] = "toy-broken-io"
    for row in campaign["rows"]:
        row["command"] = [part.replace("{kit}", str(ROOT / "kit")) for part in row["command"]]
        row["env"] = {k: v.replace("campaign/toy/", "campaign/toy-broken-io/") for k, v in row["env"].items()}
    path = tmp_path / "broken.yaml"
    path.write_text(yaml.safe_dump(campaign))
    assert runner.main(["prepare", str(path), "--all"]) == runner.EXIT_FAILED
    assert runner.main(["run", str(path), "--row", "pilot"]) == runner.EXIT_OK
    assert runner.main(["run", str(path), "--row", "big"]) == runner.EXIT_REFUSED
    assert "its latest preparation is FAIL" in capsys.readouterr().out


def test_plan_executes_nothing(work, capsys):
    assert runner.main(["plan", str(TOY)]) == runner.EXIT_OK
    out = capsys.readouterr().out
    assert "[pilot] PILOT" in out and "[big] needs: pilot" in out and not any(work.iterdir())
    assert "prepare (no GPU):" in out and "requires:" in out


@pytest.mark.parametrize("mutate, message", [
    (lambda c: c["rows"][0].pop("bars"), "a pilot without bars cannot gate anything"),
    (lambda c: c["rows"][0].update(needs=["big"]), "needs rows that do not come before it"),
    (lambda c: c["rows"][1].update(id="pilot"), "unique, non-empty ids"),
    (lambda c: c.update(schema="other"), "schema must be"),
    (lambda c: c["rows"][0]["bars"][0].pop("min") and c["rows"][0]["bars"][0].pop("max"), "needs min and/or max"),
])
def test_a_wrong_campaign_file_refuses(work, tmp_path, mutate, message, capsys):
    campaign = yaml.safe_load(TOY.read_text())
    mutate(campaign)
    path = tmp_path / "wrong.yaml"
    path.write_text(yaml.safe_dump(campaign))
    assert runner.main(["plan", str(path)]) == runner.EXIT_REFUSED
    assert message in capsys.readouterr().err


def test_k0_campaign_matches_the_partner_instructions(work, monkeypatch):
    campaign = runner.load_campaign(ROOT / "kit" / "campaigns" / "k0-sdpo-toolalpaca.yaml")
    rows = {r["id"]: runner.resolve(r, campaign, 1) for r in campaign["rows"]}
    assert [r["id"] for r in campaign["rows"] if r.get("pilot")] == ["smoke"]
    assert all(r["needs"] == ["smoke"] for r in campaign["rows"][1:])
    env = {k: v["env"] for k, v in rows.items()}
    assert env["smoke"]["STEPS"] == "2" and "SEED" not in env["smoke"]
    assert (env["run3-seed43"]["SEED"], env["run3-seed43"]["STEPS"]) == ("43", "17")
    assert (env["run3-seed44"]["SEED"], env["run3-seed44"]["STEPS"]) == ("44", "17")
    for seed in ("42", "43", "44"):
        e = env["dose40-seed%s" % seed]
        assert (e["SEED"], e["STEPS"], e["TEST_FREQ"]) == (seed, "40", "5")
    assert len(rows["smoke"]["prepare"]) == 2 and "preprocess.py" in " ".join(rows["smoke"]["prepare"][0])
    band = rows["smoke"]["bars"][0]
    assert (band["min"], band["max"], band["where"]) == (0.555, 0.600, {"step": 0})
    assert band["source"] == str(work / "runs" / "smoke-a1" / "metrics.jsonl")


def test_no_kit_tool_reads_json_lines_with_splitlines():
    """str.splitlines() also breaks on the Unicode line separators (code points 0x2028, 0x2029, 0x85), which a model can
    write INSIDE an answer; the kit writes answers with ensure_ascii=False. One such character would split a record in two
    and crash the reader on the partner's machine. Found 20 September while reviewing the K3 beds."""
    import re as _re
    record = json.dumps({"response": "one" + chr(0x2028) + "two" + chr(0x85) + "three"}, ensure_ascii=False) + "\n"
    assert len(record.splitlines()) > 1 and len([x for x in record.split("\n") if x]) == 1      # the hazard, and the fix
    offenders = []
    for path in sorted((ROOT / "kit").rglob("*.py")):
        for number, line in enumerate(path.read_text().split("\n"), 1):
            if ".splitlines()" in line and _re.search(r"json\.loads|responses|\.jsonl", line):
                offenders.append("%s:%d" % (path.relative_to(ROOT), number))
    assert offenders == [], offenders


def test_the_runner_reads_a_metrics_line_that_contains_a_unicode_line_separator(work, tmp_path):
    source = tmp_path / "metrics.jsonl"
    source.write_text(json.dumps({"step": 1, "note": "a" + chr(0x2028) + "b", "data": {"x": 3.0}}, ensure_ascii=False) + "\n")
    judged = runner.judge_bar({"name": "x", "source": str(source), "key": "x", "min": 1})
    assert judged["ok"] and judged["value"] == 3.0


# ------------------------------------------------------------------------------------ deferred preparation, batch
WRITE_OUT = ["python3", "-c", "import os, pathlib; p = pathlib.Path(os.environ['OUT']); "
             "p.parent.mkdir(parents=True, exist_ok=True); p.write_text('made')"]


def campaign_file(tmp_path: Path, name: str, rows: list) -> Path:
    path = tmp_path / ("%s.json" % name)
    path.write_text(json.dumps({"schema": runner.SCHEMA, "name": name, "workdir_env": "WORK", "rows": rows}))
    return path


def chain(tmp_path: Path, name: str = "chain", **extra_b) -> Path:
    """Row `a` writes {work}/made/a.txt (a GPU row, in real life); row `b` requires it, needs `a`, and has a prepare step."""
    b = {"id": "b", "needs": ["a"], "requires": ["{work}/made/a.txt"], "command": ["true"],
         "prepare": [["python3", "-c", "open('{work}/b-prepared.txt', 'w').write('y')"]], **extra_b}
    return campaign_file(tmp_path, name, [{"id": "a", "env": {"OUT": "{work}/made/a.txt"}, "command": WRITE_OUT}, b])


def test_prepare_all_defers_a_row_whose_input_an_earlier_row_writes(work, tmp_path, capsys):
    path = chain(tmp_path)
    assert runner.main(["prepare", str(path), "--all"]) == runner.EXIT_OK
    out = capsys.readouterr().out
    assert "deferred: b waits for %s (produced by a or later)" % (work.resolve() / "made" / "a.txt") in out
    assert not (work / "campaign" / "chain" / "b").exists(), "a deferred row records nothing that could gate it"
    assert not (work / "b-prepared.txt").exists(), "a deferred row's prepare steps wait too"
    assert runner.main(["status", str(path)]) == runner.EXIT_OK
    status = capsys.readouterr().out.splitlines()
    assert status[1].split() == ["b", "BLOCKED", "needs", "a,", "which", "has", "not", "run"], "blocked by its needs alone"


def test_run_all_prepares_a_deferred_row_right_before_running_it(work, tmp_path, capsys):
    path = chain(tmp_path)
    assert runner.main(["prepare", str(path), "--all"]) == runner.EXIT_OK
    assert runner.main(["run", str(path), "--all"]) == runner.EXIT_OK
    out = capsys.readouterr().out
    assert out.index("RUN a") < out.index("PREPARE b now") < out.index("PASS prepare b") < out.index("RUN b")
    assert (work / "b-prepared.txt").read_text() == "y"
    assert json.loads((work / "campaign" / "chain" / "b" / "prepare-1.json").read_text())["verdict"] == "PASS"
    assert json.loads((work / "campaign" / "chain" / "b" / "attempt-1" / "verdict.json").read_text())["verdict"] == "PASS"


def test_once_the_input_exists_prepare_all_prepares_the_row_as_usual(work, tmp_path, capsys):
    path = chain(tmp_path)
    (work / "made").mkdir()
    (work / "made" / "a.txt").write_text("made")                                  # as if row a had run
    assert runner.main(["prepare", str(path), "--all"]) == runner.EXIT_OK
    assert "deferred" not in capsys.readouterr().out
    assert json.loads((work / "campaign" / "chain" / "b" / "prepare-1.json").read_text())["verdict"] == "PASS"
    assert runner.main(["run", str(path), "--all"]) == runner.EXIT_OK
    assert len(list((work / "campaign" / "chain" / "b").glob("prepare-*.json"))) == 1, "a prepared row is not prepared twice"


def test_run_all_does_not_prepare_a_deferred_row_behind_a_failed_pilot(work, tmp_path):
    pilot = {"id": "p", "pilot": True, "command": ["false"],
             "bars": [{"name": "x", "source": "{work}/nowhere.json", "key": "x", "min": 0}]}
    rows = json.loads(chain(tmp_path).read_text())["rows"]
    rows[1]["needs"] = ["a", "p"]
    path = campaign_file(tmp_path, "gated", [rows[0], pilot, rows[1]])
    assert runner.main(["prepare", str(path), "--all"]) == runner.EXIT_OK
    assert runner.main(["run", str(path), "--all"]) == runner.EXIT_FAILED
    assert (work / "made" / "a.txt").exists()
    assert not (work / "b-prepared.txt").exists() and not (work / "campaign" / "gated" / "b").exists()


def test_prepare_row_is_unchanged_for_a_row_whose_input_is_not_written_yet(work, tmp_path, capsys):
    assert runner.main(["prepare", str(chain(tmp_path)), "--row", "b"]) == runner.EXIT_FAILED
    assert "FAIL prepare b: required paths are missing after preparation" in capsys.readouterr().out


@pytest.mark.parametrize("requires", [
    ["{work}/nobody-writes-this.txt"],                                  # under {work}, but no row writes it
    ["/nonexistent-dataset-root/train.json"],                           # outside {work}: a dataset root
    ["{work}/made/a.txt", "/nonexistent-model/config.json"],            # one produced, one external: still loud
])
def test_an_input_nothing_in_the_campaign_writes_still_fails_prepare_loudly(work, tmp_path, capsys, requires):
    path = chain(tmp_path, requires=requires)
    assert runner.main(["prepare", str(path), "--all"]) == runner.EXIT_FAILED
    out = capsys.readouterr().out
    assert "deferred" not in out and "FAIL prepare b: required paths are missing after preparation" in out
    assert json.loads((work / "campaign" / "chain" / "b" / "prepare-1.json").read_text())["verdict"] == "FAIL"


def test_k2_defers_exactly_the_rows_that_wait_on_an_earlier_gpu_row(work):
    campaign = runner.load_campaign(ROOT / "kit" / "campaigns" / "k2-recovery-test.yaml")
    waiting = {row["id"]: sorted(set(runner.deferred(campaign, row).values())) for row in campaign["rows"]}
    assert {k: v for k, v in waiting.items() if v} == {
        "small-before": ["small-make-damaged"], "small-repair": ["small-targets"], "spider60-repair": ["q3-targets"],
        "bird-before": ["bird-make-damaged"], "bird-repair": ["q25-targets"]}


def tiny(tmp_path: Path, name: str, command: list) -> Path:
    return campaign_file(tmp_path, name, [{"id": "only", "command": command}])


def summary_lines(out: str) -> list:
    return out[out.index("BATCH SUMMARY"):].strip().split("\n")[1:]


def test_batch_runs_campaigns_in_the_order_given(work, tmp_path, capsys):
    first, second = tiny(tmp_path, "first", ["true"]), tiny(tmp_path, "second", ["true"])
    assert runner.main(["batch", str(second), str(first)]) == runner.EXIT_OK
    out = capsys.readouterr().out
    assert out.index("=== BATCH second: prepare --all") < out.index("=== BATCH second: run --all") < out.index("=== BATCH first")
    lines = summary_lines(out)
    assert [line.split()[0] for line in lines] == ["second", "first"]
    assert all(line.endswith("PASS (run --all exited 0)") for line in lines)
    for name in ("first", "second"):
        assert json.loads((work / "campaign" / name / "only" / "attempt-1" / "verdict.json").read_text())["verdict"] == "PASS"


def test_batch_stops_at_the_first_campaign_that_fails(work, tmp_path, capsys):
    bad, good = tiny(tmp_path, "bad", ["false"]), tiny(tmp_path, "good", ["true"])
    assert runner.main(["batch", str(bad), str(good)]) == runner.EXIT_FAILED
    lines = summary_lines(capsys.readouterr().out)
    assert lines[0].split()[0] == "bad" and lines[0].endswith("FAIL (run --all exited 1)")
    assert lines[1].split()[0] == str(good) and "NOT RUN" in lines[1]
    assert not (work / "campaign" / "good").exists()


def test_batch_returns_the_refusal_code_and_stops_on_a_wrong_campaign_file(work, tmp_path, capsys):
    pilot = {"id": "p", "pilot": True, "command": ["true"],
             "bars": [{"name": "x", "source": "{work}/nowhere.json", "key": "x", "min": 0}]}
    blocked = campaign_file(tmp_path, "blocked", [pilot, {"id": "big", "needs": ["p"], "command": ["true"]}])
    (work / "campaign" / "blocked" / "p" / "attempt-1").mkdir(parents=True)          # a pilot attempt with no verdict
    assert runner.main(["run", str(blocked), "--row", "big"]) == runner.EXIT_REFUSED
    broken = tmp_path / "broken.json"
    broken.write_text(json.dumps({"schema": "other", "rows": []}))
    assert runner.main(["batch", str(broken), str(tiny(tmp_path, "later", ["true"]))]) == runner.EXIT_REFUSED
    captured = capsys.readouterr()
    assert "CAMPAIGN ERROR in %s" % broken in captured.err and "NOT RUN" in captured.out
    assert not (work / "campaign" / "later").exists()


def test_batch_plan_prints_every_plan_and_executes_nothing(work, tmp_path, capsys):
    one, two = tiny(tmp_path, "one", ["true"]), chain(tmp_path)
    assert runner.main(["batch", "--plan", str(one), str(two)]) == runner.EXIT_OK
    out = capsys.readouterr().out
    assert "campaign one (1 rows)" in out and "campaign chain (2 rows)" in out and "[b] needs: a" in out
    assert sorted(p.name for p in work.iterdir()) == ["chain.json", "one.json"], "only the campaign files themselves"


# ------------------------------------------------------------------------------------ implicit order
# A row that requires what another row's prepare writes depends on that row. The runner gets the order free from
# file order; a parallel dispatcher does not (K1c, 30 September: `base17b-gsm8k` required the GSM8K files
# `base17b-finqa`'s prepare builds, named no edge, and the partner's dispatcher started both at once).
BUILD = ["python3", "-c", "import pathlib; p = pathlib.Path('{work}/data/x'); p.mkdir(parents=True, exist_ok=True); "
         "(p / 'train.parquet').write_text('t')", "--out", "{work}/data/x"]


def ordered(tmp_path: Path, name: str, *, stated: bool) -> Path:
    """`maker`'s prepare builds {work}/data/x; `explicit` and `self-made` are fine; `reader` names `maker` only if stated."""
    wanted = ["{work}/data/x/train.parquet"]
    return campaign_file(tmp_path, name, [
        {"id": "maker", "prepare": [BUILD], "requires": wanted, "command": ["true"]},
        {"id": "explicit", "needs": ["maker"], "requires": wanted, "command": ["true"]},
        {"id": "through-explicit", "needs": ["explicit"], "requires": wanted, "command": ["true"]},
        {"id": "self-made", "prepare": [BUILD], "requires": wanted, "command": ["true"]},
        {"id": "reader", "needs": ["maker"] if stated else [], "requires": wanted, "command": ["true"]}])


def test_implicit_edges_finds_the_unstated_order_and_only_that(work, tmp_path):
    edges = runner.implicit_edges(runner.load_campaign(ordered(tmp_path, "implied", stated=False)))
    assert edges == [("reader", str(work.resolve() / "data" / "x" / "train.parquet"), "maker", "prepare")]
    assert runner.implicit_edges(runner.load_campaign(ordered(tmp_path, "stated", stated=True))) == []


def test_implicit_edges_reads_a_command_output_too(work, tmp_path):
    rows = json.loads(chain(tmp_path).read_text())["rows"]
    del rows[1]["needs"]
    edges = runner.implicit_edges(runner.load_campaign(campaign_file(tmp_path, "unstated", rows)))
    assert edges == [("b", str(work.resolve() / "made" / "a.txt"), "a", "command")]


def test_plan_exits_non_zero_on_an_implicit_edge_and_prints_nothing_new_without_one(work, tmp_path, capsys):
    assert runner.main(["plan", str(ordered(tmp_path, "implied", stated=False))]) == runner.EXIT_REFUSED
    out = capsys.readouterr().out
    assert ("IMPLICIT ORDER: reader requires %s, written by maker's prepare, but does not need maker"
            % (work.resolve() / "data" / "x" / "train.parquet")) in out
    assert out.count("IMPLICIT ORDER") == 1
    assert runner.main(["plan", str(ordered(tmp_path, "stated", stated=True))]) == runner.EXIT_OK
    assert "IMPLICIT" not in capsys.readouterr().out
    assert runner.main(["batch", "--plan", str(ordered(tmp_path, "implied", stated=False))]) == runner.EXIT_REFUSED


@pytest.mark.parametrize("action", [["prepare", "--all"], ["prepare", "--row", "maker"], ["run", "--all"],
                                    ["run", "--row", "maker"], ["batch"]])
def test_prepare_run_and_batch_refuse_an_implicit_edge_before_doing_anything(work, tmp_path, capsys, action):
    path = ordered(tmp_path, "implied", stated=False)
    argv = ["batch", str(path)] if action == ["batch"] else [action[0], str(path)] + action[1:]
    assert runner.main(argv) == runner.EXIT_REFUSED
    assert "IMPLICIT ORDER: reader requires" in capsys.readouterr().out
    assert not (work / "data").exists() and not (work / "campaign").exists(), "a refusal prepares and records nothing"


@pytest.mark.parametrize("path", sorted((ROOT / "kit" / "campaigns").glob("*.yaml")), ids=lambda p: p.name)
def test_every_committed_campaign_states_its_order(work, path):
    assert runner.implicit_edges(runner.load_campaign(path)) == []


GENERATORS = sorted((ROOT / "scripts").glob("make_*_campaign.py"))


@pytest.mark.skipif(not GENERATORS, reason="scripts/ is not part of the exported kit")
@pytest.mark.parametrize("path", GENERATORS, ids=lambda p: p.name)
def test_every_generated_campaign_is_what_its_generator_builds(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)
    assert generator.OUT.read_text() == generator.build(), "re-run %s" % path.relative_to(ROOT)


# ------------------------------------------------------------------------------------ the pilot gate, stated
# The runner enforces "every row needs every earlier pilot" by itself; the campaign file must say so too, or a
# dispatcher that reads only `needs` starts rows beside a pilot (30 September: `base-forget-2` beside `base-forget-1`).
CAMPAIGNS = sorted((ROOT / "kit" / "campaigns").glob("*.yaml"))
PASSING_PILOT = {"id": "p", "pilot": True, "command": ["true"],
                 "bars": [{"name": "x", "source": "{work}/nowhere.json", "key": "x", "min": 0}]}


@pytest.mark.parametrize("path", CAMPAIGNS, ids=lambda p: p.name)
def test_every_committed_campaign_states_its_pilot_edges(work, path):
    """Each pilot needs the previous pilot and every other row the most recent pilot before it, in the file."""
    last = None
    for row in runner.load_campaign(path)["rows"]:
        if last:
            assert last in row["_stated_needs"], "%s does not state the pilot %s" % (row["id"], last)
        if row.get("pilot"):
            last = row["id"]


def test_a_row_after_a_pilot_that_does_not_need_it_is_refused(work, tmp_path, capsys):
    rows = [PASSING_PILOT, {"id": "second", "command": ["true"]}, {"id": "third", "needs": ["second"], "command": ["true"]},
            {"id": "q", "pilot": True, "needs": ["third"], "command": ["true"], "bars": PASSING_PILOT["bars"]}]
    path = campaign_file(tmp_path, "unstated-pilot", rows)
    assert runner.implicit_edges(runner.load_campaign(path)) == [
        ("second", None, "p", "pilot"), ("third", None, "p", "pilot"), ("q", None, "p", "pilot")], "the gate's own pilots do not count"
    assert runner.main(["plan", str(path)]) == runner.EXIT_REFUSED
    assert "IMPLICIT ORDER: second comes after pilot p but does not need it" in capsys.readouterr().out
    for argv in (["run", str(path), "--all"], ["prepare", str(path), "--all"], ["batch", str(path)]):
        assert runner.main(argv) == runner.EXIT_REFUSED
    assert not (work / "campaign").exists(), "a refusal runs and records nothing"
    rows[1]["needs"] = ["p"]                                                      # third and q now reach p through second
    assert runner.implicit_edges(runner.load_campaign(campaign_file(tmp_path, "stated-pilot", rows))) == []


# ------------------------------------------------------------------------------------ wants
def seeded(tmp_path: Path, name: str, *, failing=(), report: dict | None = None) -> Path:
    """`w-seed0` and `w-seed1` each write {work}/made/<id>.txt; `report` wants both and needs neither."""
    rows = [{"id": "w-seed%d" % s, "env": {"OUT": "{work}/made/w-seed%d.txt" % s},
             "command": ["false"] if s in failing else WRITE_OUT} for s in (0, 1)]
    return campaign_file(tmp_path, name, rows + [report or {"id": "report", "wants": ["w-seed0", "w-seed1"], "command": ["true"]}])


def row_verdict(work: Path, campaign: str, row: str) -> dict:
    return json.loads((work / "campaign" / campaign / row / "attempt-1" / "verdict.json").read_text())


def test_a_wanted_row_runs_first_when_it_is_scheduled(work, tmp_path, capsys):
    path = seeded(tmp_path, "wanting")
    assert runner.main(["plan", str(path)]) == runner.EXIT_OK
    assert "[report] needs: nothing\n  wants: w-seed0, w-seed1" in capsys.readouterr().out
    assert runner.main(["run", str(path), "--all"]) == runner.EXIT_OK
    out = capsys.readouterr().out
    assert out.index("RUN w-seed0") < out.index("RUN w-seed1") < out.index("RUN report")
    assert row_verdict(work, "wanting", "report")["wants_not_passed"] == []


def test_a_want_states_a_file_order_as_a_need_does(work, tmp_path):
    reader = {"id": "report", "requires": ["{work}/made/w-seed0.txt"], "command": ["true"]}
    edges = runner.implicit_edges(runner.load_campaign(seeded(tmp_path, "unordered", report=reader)))
    assert edges == [("report", str(work.resolve() / "made" / "w-seed0.txt"), "w-seed0", "command")]
    wanting = seeded(tmp_path, "ordered", report={**reader, "wants": ["w-seed0"]})
    assert runner.implicit_edges(runner.load_campaign(wanting)) == []


def test_a_failed_seed_run_does_not_stop_run_all_and_the_report_records_the_want(work, tmp_path, capsys):
    path = seeded(tmp_path, "failing", failing=(0,))
    assert runner.main(["run", str(path), "--all"]) == runner.EXIT_FAILED               # the first failure's code, at the end
    out = capsys.readouterr().out
    assert "continuing: w-seed0 failed; no later row needs it (wanted by: report)\n" in out
    assert "note report: runs without 1 wanted row(s) that have not passed: w-seed0" in out
    assert row_verdict(work, "failing", "w-seed1")["verdict"] == "PASS"
    report = row_verdict(work, "failing", "report")
    assert report["verdict"] == "PASS" and report["wants_not_passed"] == ["w-seed0"]
    start = json.loads((work / "campaign" / "failing" / "report" / "attempt-1" / "start.json").read_text())
    assert start["wants_not_passed"] == ["w-seed0"]


def test_a_failed_row_a_later_row_needs_still_stops_run_all_naming_that_row(work, tmp_path, capsys):
    path = seeded(tmp_path, "needed", failing=(0,), report={"id": "report", "needs": ["w-seed0"], "command": ["true"]})
    assert runner.main(["run", str(path), "--all"]) == runner.EXIT_FAILED
    out = capsys.readouterr().out
    assert "STOP: w-seed0 did not pass and report needs it; nothing after it runs" in out and "continuing" not in out
    assert not (work / "campaign" / "needed" / "w-seed1").exists() and not (work / "campaign" / "needed" / "report").exists()
    assert runner.main(["run", str(path), "--row", "report"]) == runner.EXIT_REFUSED


def test_a_need_through_another_row_stops_run_all_too_and_so_does_batch(work, tmp_path, capsys):
    """`sum` needs `mid`, which needs the failing `w-seed0`: the need is transitive, and `batch` applies the same rule."""
    def rows(mid: dict) -> list:
        return [{"id": "w-seed0", "command": ["false"]}, {"id": "w-seed1", "command": ["true"]},
                {"id": "mid", "command": ["true"], **mid}, {"id": "sum", "needs": ["mid"], "command": ["true"]},
                {"id": "report", "wants": ["w-seed0", "w-seed1"], "command": ["true"]}]
    through = campaign_file(tmp_path, "through", rows({"needs": ["w-seed0"]}))
    campaign = runner.load_campaign(through)
    assert runner.needed_later(campaign, campaign["rows"], "w-seed0") == ["mid", "sum"]
    assert runner.main(["batch", str(through)]) == runner.EXIT_FAILED
    assert "STOP: w-seed0 did not pass and mid needs it; nothing after it runs" in capsys.readouterr().out
    assert not (work / "campaign" / "through" / "w-seed1").exists()
    around = campaign_file(tmp_path, "around", rows({"wants": ["w-seed0"]}))
    assert runner.main(["batch", str(around)]) == runner.EXIT_FAILED
    assert "continuing: w-seed0 failed; no later row needs it (wanted by: mid, report)" in capsys.readouterr().out
    assert row_verdict(work, "around", "sum")["verdict"] == "PASS"
    assert row_verdict(work, "around", "report")["wants_not_passed"] == ["w-seed0"]


def test_a_refused_row_nothing_needs_does_not_stop_run_all(work, tmp_path, capsys):
    rows = [{"id": "lost", "requires": ["/nowhere/at/all"], "command": ["true"]},
            {"id": "after", "wants": ["lost"], "command": ["true"]}]
    assert runner.main(["run", str(campaign_file(tmp_path, "refusing", rows)), "--all"]) == runner.EXIT_REFUSED
    out = capsys.readouterr().out
    assert "REFUSED lost" in out and "continuing: lost failed; no later row needs it (wanted by: after)" in out
    assert row_verdict(work, "refusing", "after")["wants_not_passed"] == ["lost"]


@pytest.mark.parametrize("mutate, message", [
    (lambda rows: rows[0].update(wants=["report"]), "wants rows that do not come before it"),
    (lambda rows: rows[2].update(wants="w-seed0"), "wants must be a list"),
])
def test_a_wrong_wants_refuses(work, tmp_path, capsys, mutate, message):
    rows = json.loads(seeded(tmp_path, "wrong").read_text())["rows"]
    mutate(rows)
    assert runner.main(["plan", str(campaign_file(tmp_path, "wrong", rows))]) == runner.EXIT_REFUSED
    assert message in capsys.readouterr().err


# ------------------------------------------------------------------------------------ --seeds
def test_seeds_parse_lists_and_ranges():
    assert runner.parse_seeds("0,1,2,3,4") == runner.parse_seeds("0-4") == [0, 1, 2, 3, 4]
    assert runner.parse_seeds("7, 0-2") == [0, 1, 2, 7]
    for bad in ("a", "0-x", ",", "0-", "4-0"):
        with pytest.raises(runner.CampaignError):
            runner.parse_seeds(bad)


def test_the_seed_filter_reads_the_id_then_the_name_and_keeps_rows_with_no_seed(work, tmp_path, capsys):
    rows = [{"id": "a-seed3", "command": ["true"]}, {"id": "a-seed12-forget", "command": ["true"]},
            {"id": "train", "env": {"NAME": "run-seed7-a{attempt}"}, "command": ["true"]},
            {"id": "b-seed3", "env": {"NAME": "b-seed7-a{attempt}"}, "command": ["true"]},       # the id wins
            {"id": "plain", "env": {"NAME": "plain-a{attempt}"}, "command": ["true"]}]
    campaign = runner.load_campaign(campaign_file(tmp_path, "filtered", rows))
    assert runner.filter_seeds(campaign, [3, 12]) == ["train"]
    assert runner.filter_seeds(campaign, [7]) == ["a-seed3", "a-seed12-forget", "b-seed3"]
    assert runner.filter_seeds(campaign, None) == []
    capsys.readouterr()
    assert runner.main(["plan", str(campaign_file(tmp_path, "filtered", rows)), "--seeds", "7"]) == runner.EXIT_OK
    out = capsys.readouterr().out
    assert out.count("skipped (seed filter): a-seed3\n") == 1 and "[a-seed3]" not in out and "[train]" in out and "[plain]" in out
    assert runner.main(["plan", str(campaign_file(tmp_path, "filtered", rows)), "--seeds", "x"]) == runner.EXIT_REFUSED


def test_a_filtered_out_want_does_not_block_and_the_verdict_records_the_filter(work, tmp_path, capsys):
    path = seeded(tmp_path, "reduced")
    assert runner.main(["prepare", str(path), "--all", "--seeds", "1"]) == runner.EXIT_OK
    assert runner.main(["run", str(path), "--all", "--seeds", "1"]) == runner.EXIT_OK
    out = capsys.readouterr().out
    assert out.count("skipped (seed filter): w-seed0") == 2, "once per command"
    assert not (work / "campaign" / "reduced" / "w-seed0").exists() and not (work / "made" / "w-seed0.txt").exists()
    verdict = row_verdict(work, "reduced", "report")
    assert verdict["verdict"] == "PASS" and verdict["seeds"] == [1] and verdict["wants_not_passed"] == ["w-seed0"]
    assert row_verdict(work, "reduced", "w-seed1")["seeds"] == [1]
    assert runner.main(["run", str(seeded(tmp_path, "full")), "--all"]) == runner.EXIT_OK
    assert row_verdict(work, "full", "report")["seeds"] is None


def test_a_row_that_needs_a_skipped_row_is_refused_naming_it(work, tmp_path, capsys):
    path = seeded(tmp_path, "needing", report={"id": "sum", "needs": ["w-seed0"], "command": ["true"]})
    assert runner.main(["plan", str(path), "--seeds", "1"]) == runner.EXIT_OK
    assert "refused unless it already passed: needs w-seed0, which the seed filter skips" in capsys.readouterr().out
    assert runner.main(["run", str(path), "--all", "--seeds", "1"]) == runner.EXIT_REFUSED
    assert "REFUSED sum: needs w-seed0, which the seed filter skips (--seeds 1) and which has not passed" in capsys.readouterr().out
    assert row_verdict(work, "needing", "w-seed1")["verdict"] == "PASS" and not (work / "campaign" / "needing" / "sum").exists()
    assert runner.main(["run", str(path), "--row", "w-seed0"]) == runner.EXIT_OK             # an earlier, unfiltered run
    assert runner.main(["run", str(path), "--all", "--seeds", "1"]) == runner.EXIT_OK        # the need has passed: not refused


def test_batch_takes_the_seed_filter(work, tmp_path, capsys):
    path = seeded(tmp_path, "batched")
    assert runner.main(["batch", "--seeds", "0", str(path)]) == runner.EXIT_OK
    assert "skipped (seed filter): w-seed1" in capsys.readouterr().out
    assert not (work / "campaign" / "batched" / "w-seed1").exists()
    assert row_verdict(work, "batched", "report")["seeds"] == [0]
    assert runner.main(["batch", "--seeds", "0-", str(path)]) == runner.EXIT_REFUSED


def test_a_stated_seed_wins_over_the_name_and_plan_prints_it(work, tmp_path, capsys):
    rows = [{"id": "x-r1", "seed": 1, "env": {"NAME": "x-r1-seed9-a{attempt}"}, "command": ["true"]},
            {"id": "x-r1-spider", "seed": 1, "needs": ["x-r1"], "command": ["true"]},
            {"id": "old-seed2", "command": ["true"]}, {"id": "plain", "command": ["true"]}]
    path = campaign_file(tmp_path, "stated", rows)
    campaign = runner.load_campaign(path)
    assert [runner.seed_of(row) for row in campaign["rows"]] == [1, 1, 2, None]
    assert runner.filter_seeds(campaign, [9]) == ["x-r1", "x-r1-spider", "old-seed2"], "the field first, the name only without it"
    capsys.readouterr()
    assert runner.main(["plan", str(path)]) == runner.EXIT_OK
    out = capsys.readouterr().out
    assert "[x-r1] needs: nothing\n  seed: 1\n" in out and "[old-seed2] needs: nothing\n  seed: 2 (from its name)\n" in out
    assert "[plain] needs: nothing\n  command:" in out


@pytest.mark.parametrize("seed", ["1", -1, True, 1.5])
def test_a_seed_that_is_not_a_whole_number_refuses(work, tmp_path, capsys, seed):
    path = campaign_file(tmp_path, "badseed", [{"id": "a", "seed": seed, "command": ["true"]}])
    assert runner.main(["plan", str(path)]) == runner.EXIT_REFUSED
    assert "row a: seed must be a whole number" in capsys.readouterr().err


# ------------------------------------------------------------------------------------ `seed:` in the committed campaigns
SEEDED = re.compile(r"seed(\d+)|-r(\d+)(?:-|$)")


@pytest.mark.parametrize("path", CAMPAIGNS, ids=lambda p: p.name)
def test_every_seeded_row_states_its_seed(path):
    """A row whose id or run NAME is built per seed (`seed<N>`) or per run (`-r<N>`) carries `seed:`, and agrees."""
    for row in yaml.safe_load(path.read_text())["rows"]:
        named = [m for text in (row["id"], str((row.get("env") or {}).get("NAME", ""))) for m in SEEDED.finditer(text)]
        if named:
            assert isinstance(row.get("seed"), int), "%s carries no seed: field" % row["id"]
            assert {int(m.group(1) or m.group(2)) for m in named} == {row["seed"]}, row["id"]


def test_seeds_1_2_on_the_anchor_plan_the_base_the_r1_and_r2_rows_and_the_report(work, capsys):
    assert runner.main(["plan", str(ROOT / "kit" / "campaigns" / "k3-anchor.yaml"), "--seeds", "1-2"]) == runner.EXIT_OK
    planned = re.findall(r"(?m)^\[([^\]]+)\]", capsys.readouterr().out)
    spider = ["%s-r%d%s" % (arm, run, part) for arm in ("ref20", "kl01", "kl10", "ent01", "lr06", "lr36", "lr06kl") for run in (1, 2)
              for part in ("", "-spider", "-delta")]
    gsm8k = ["%s-r%d%s" % (arm, run, part) for arm in ("g-ref", "g-lr06", "g-lr06kl", "g-kl01") for run in (1, 2)
             for part in ("", "-gsm8k", "-delta")]
    assert sorted(planned) == sorted(["base-spider", "base-gsm8k", "report"] + spider + gsm8k) and len(planned) == 69


def test_the_dose_probe_filters_each_arm_with_its_scoring_and_delta(work, capsys):
    """Every probe arm is one run at seed 0: `--seeds 1` drops the run, its scoring and its delta together."""
    assert runner.main(["plan", str(ROOT / "kit" / "campaigns" / "k3-dose-probe.yaml"), "--seeds", "1"]) == runner.EXIT_OK
    out = capsys.readouterr().out
    assert re.findall(r"(?m)^\[([^\]]+)\]", out) == ["base-spider", "stuck", "subset", "pool1920", "report"]
    assert "refused unless it already passed" not in out


# ------------------------------------------------------------------------------------ report rows in the committed campaigns
# A report row NEEDS only pilots and the untrained (base) scorings and WANTS every per-seed row it reads, so a
# reduced-seed run (`--seeds 0-4`) still writes a report and records its verdict. The count pins each one.
REPORT_WANTS = {"k1a-forgetting-of-k0.yaml": 5, "k1c-grpo-baselines.yaml": 100, "k2-recovery-test.yaml": 0,
                "k2b-route.yaml": 60, "k3-anchor.yaml": 33, "k3-dose-2.yaml": 12, "k3-dose-probe.yaml": 4,
                "k3-replay.yaml": 140, "k4-hints.yaml": 200, "k4a-stuck-problems.yaml": 18, "k5-sequence.yaml": 24}


def test_every_campaign_with_a_report_row_is_listed():
    assert sorted(p.name for p in CAMPAIGNS if "\n  - id: report\n" in p.read_text()) == sorted(REPORT_WANTS)


@pytest.mark.parametrize("name", sorted(REPORT_WANTS))
def test_every_report_row_needs_only_pilots_and_base_scorings_and_wants_its_seed_rows(work, name):
    campaign = runner.load_campaign(ROOT / "kit" / "campaigns" / name)
    rows = {row["id"]: row for row in campaign["rows"]}
    memo: dict = {}

    def per_seed(rid: str) -> bool:
        """The row carries a seed, or needs (through rows that are not pilots) one that does."""
        if rid not in memo:
            row = rows[rid]
            memo[rid] = runner.seed_of(row) is not None or (not row.get("pilot") and any(map(per_seed, row["_stated_needs"])))
        return memo[rid]

    report = rows["report"]
    assert [n for n in report["_stated_needs"] if rows[n].get("pilot") or not per_seed(n)] == report["_stated_needs"]
    assert all(per_seed(w) for w in report["wants"]) and len(report["wants"]) == REPORT_WANTS[name]
    assert any(per_seed(rid) for rid in rows) == bool(report["wants"])
