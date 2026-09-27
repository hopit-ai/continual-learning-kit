"""kit/runner.py: the pilot gate, the records, and the refusals. Pure CPU, under a second each."""
from __future__ import annotations

import importlib.util
import json
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
    """Row `a` writes {work}/made/a.txt (a GPU row, in real life); row `b` requires it and has a prepare step."""
    b = {"id": "b", "requires": ["{work}/made/a.txt"], "command": ["true"],
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
    assert "BLOCKED" not in capsys.readouterr().out


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
    blocked = campaign_file(tmp_path, "blocked", [pilot, {"id": "big", "command": ["true"]}])
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
    assert "campaign one (1 rows)" in out and "campaign chain (2 rows)" in out and "[b] needs: nothing" in out
    assert sorted(p.name for p in work.iterdir()) == ["chain.json", "one.json"], "only the campaign files themselves"
