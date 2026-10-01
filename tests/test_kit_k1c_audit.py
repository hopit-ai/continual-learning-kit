"""kit/campaigns/k1c-audit.yaml: plan v3 package 1, the budget audit of the ten K1c Part B checkpoints.

What can go wrong with a campaign file costs the partner a day each time, so the file is generated and this test
pins what a reader of the runbook is told: one GPU, a pilot first, nothing implicit, every command valid shell, every
output where the collector will pack it, every bar reading a number that the tool really writes.
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
CAMPAIGN = KIT / "campaigns" / "k1c-audit.yaml"
GENERATOR = ROOT / "scripts" / "make_k1c_audit_campaign.py"


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


generator = load("make_k1c_audit_campaign", GENERATOR) if GENERATOR.is_file() else None
needs_generator = pytest.mark.skipif(generator is None, reason="scripts/ is not part of the exported kit")
collect = load("kit_collect_audit", KIT / "collect.py")
ROWS = yaml.safe_load(CAMPAIGN.read_text())["rows"]
BY_ID = {row["id"]: row for row in ROWS}
GPU = [row for row in ROWS if "CUDA_VISIBLE_DEVICES" in row["env"]]


@needs_generator
def test_the_committed_campaign_is_what_the_generator_writes():
    assert CAMPAIGN.read_text() == generator.build()


def test_the_shape_the_runbook_promises():
    assert len(ROWS) == 66 and len(GPU) == 26
    assert ROWS[0]["id"] == "pilot-cap" and ROWS[0]["pilot"] is True and "--limit 8" in ROWS[0]["command"][2]
    assert all("pilot-cap" in row.get("needs", []) for row in ROWS[1:]), "every row needs the pilot"
    assert {row["env"]["CUDA_VISIBLE_DEVICES"] for row in GPU} == {"0"}, "one physical GPU"
    chain = GPU[1:]
    assert all(later.get("wants") == [earlier["id"]] for earlier, later in zip(chain, chain[1:])), "scorings run one after another"
    long = [row for row in GPU if row["id"].endswith("-long")]
    assert len(long) == 23 and all("--max-new-tokens 8192 --max-model-len 12288" in row["command"][2] for row in long)
    direct = [row for row in GPU if row["id"].startswith("direct-")]
    assert [row["id"] for row in direct] == ["direct-gsm8k-seed0", "direct-finqa-seed1"]
    assert all("--max-new-tokens 2048 --max-model-len 12288" in row["command"][2] for row in direct), "same context, shorter cap"
    assert sum(row["id"].startswith("check-same-gpu-") for row in ROWS) == 2 and sum(row["id"].startswith("check-k1c-") for row in ROWS) == 12
    assert sum(row["id"].startswith("sweep-") for row in ROWS) == 23 and {"budget-gsm8k", "budget-finqa", "report-8k"} <= set(BY_ID)


def test_no_cpu_row_takes_a_gpu_and_every_seeded_row_states_its_seed():
    for row in ROWS:
        if row["id"].startswith(("check-", "sweep-", "budget-", "report-")):
            assert "CUDA_VISIBLE_DEVICES" not in row["env"], row["id"]
        match = re.search(r"seed(\d)", row["id"])
        assert (row.get("seed") == int(match.group(1))) if match else ("seed" not in row), row["id"]


def test_every_row_that_reads_a_scoring_needs_the_row_that_writes_it():
    for row in ROWS:
        rid = row["id"]
        if rid.startswith("sweep-") and rid.endswith("-panel"):
            who = rid[len("sweep-"):-len("-panel")]
            assert ("%s-panel-long" % who) in row["needs"], rid
        elif rid.startswith("sweep-"):
            assert ("%s-long" % rid[len("sweep-"):]) in row["needs"], rid
        elif rid.startswith("check-k1c-"):
            assert ("%s-long" % rid[len("check-k1c-"):]) in row["needs"], rid
        elif rid.startswith("check-same-gpu-"):
            bed, seed = rid[len("check-same-gpu-"):].split("-seed")
            assert {"%s-seed%s-%s-long" % (bed, seed, bed), "direct-%s-seed%s" % (bed, seed)} <= set(row["needs"]), rid
    for bed in ("gsm8k", "finqa"):
        row = BY_ID["budget-%s" % bed]
        checked = {"gsm8k": 0, "finqa": 1}[bed]                      # the seed of this bed that has a same-GPU prefix check
        assert "sweep-base17b-%s" % bed in row["needs"]
        assert row["wants"] == ["sweep-%s-seed%d-%s" % (bed, s, bed) for s in range(5)] + ["check-same-gpu-%s-seed%d" % (bed, checked)]
        # the pre-registered rule is enforced, not just written: the check's file is handed to the report
        command = row["command"][2]
        assert 'case " %d " in *" $s "*)' % checked in command, "only the seed that has a scheduled check"
        assert "same-gpu-%s-seed$s-%s-a*.json" % (bed, bed) in command and '"${QUAL[@]}"' in command
        # a check file that is not there is still named, so the report leaves the seed out instead of reading it unqualified
        assert '--qualify "seed$s=${c:-{work}/k1c/cap8192/report-prefix/same-gpu-%s-seed$s-%s-a1.json}"' % (bed, bed) in command


def test_every_command_is_valid_shell_and_quotes_the_work_directory():
    for row in ROWS:
        text = row["command"][2]
        for name, value in (("{kit}", "/k it"), ("{work}", "/w ork"), ("{attempt}", "1")):
            text = text.replace(name, value)
        assert subprocess.run(["bash", "-n", "-c", text], capture_output=True).returncode == 0, row["id"]
        assert not re.search(r"ls -d \{work\}", row["command"][2]) and not re.search(r'ls -d "\{work\}/[^"]*\*', row["command"][2]), row["id"]


def test_the_runner_plans_it_with_nothing_implicit(tmp_path):
    env = {**os.environ, "WORK": str(tmp_path), "KIT": str(KIT), "GSM8K_ROOT": "/x", "FINQA_ROOT": "/x", "K0_REPORT": "/x"}
    done = subprocess.run([sys.executable, str(KIT / "runner.py"), "plan", str(CAMPAIGN)], capture_output=True, text=True, env=env)
    assert done.returncode == 0, done.stderr
    assert "IMPLICIT ORDER" not in done.stdout + done.stderr and done.stdout.count("\n[") + done.stdout.startswith("[") >= 66


def test_everything_the_campaign_writes_is_somewhere_the_collector_packs():
    """The review of receipt 241 found a campaign whose scorings the collector would have left behind."""
    for row in ROWS:
        if row["id"] == "pilot-cap":
            continue
        out = row["env"]["OUT"].replace("{work}/", "").replace("{attempt}", "1")
        probes = [out] if out.endswith(".json") else [out + "/" + name for name in {
            "eval": ("bed-score.json", "responses.jsonl", "tokens.jsonl"), "forgetting": ("forgetting.json", "responses.jsonl", "tokens.jsonl"),
        }.get(next((part for part in Path(out).parts if part in ("eval", "forgetting")), ""), ("sweep.json", "per_item.jsonl", "sweep.md", "budget-report.json", "k1c-report.md"))]
        for probe in probes:
            assert collect.wanted(Path(probe), Path(".")), "%s writes %s, which the collector would not pack" % (row["id"], probe)


def test_every_bar_reads_a_number_its_tool_really_writes():
    written = {"bed-score.json": {"n", "max_new_tokens"}, "forgetting.json": {"total_correct", "max_new_tokens"},
               "sweep.json": {"n", "reproduces_scoring_mismatches"}, "budget-report.json": {"scorings_reported", "accounting_exact"},
               "k1c-report.json": {"comparable"}}
    for row in ROWS:
        for bar in row["bars"]:
            name = Path(bar["source"]).name
            keys = written.get(name, {"compared", "text_agreement"} if "report-prefix" in bar["source"] else set())
            assert bar["key"] in keys, (row["id"], bar)
    sweep, report, eb, sf = ((KIT / f).read_text() for f in ("cap_sweep.py", "budget_report.py", "eval_bed.py", "score_forgetting.py"))
    for key in ("reproduces_scoring_mismatches", '"compared"', '"text_agreement"'):
        assert key in sweep, key
    assert '"scorings_reported"' in report and '"accounting_exact"' in report
    assert '"max_new_tokens": cap' in eb and '"max_new_tokens": cap' in sf
