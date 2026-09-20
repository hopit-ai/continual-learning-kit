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
