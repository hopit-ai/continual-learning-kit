"""kit/campaigns/k8b-pilot.yaml: plan v3 package 3, the 8B discovery pilot.

Generated, and pinned here: what the runbook promises (pilots first, four recipes, the second run fixed in advance,
one GPU for scoring, a second stage that starts from the first stage's own merged model), valid shell, nothing
implicit, and every output where the collector packs it.
"""
from __future__ import annotations

import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
KIT = ROOT / "kit"
CAMPAIGN = KIT / "campaigns" / "k8b-pilot.yaml"
GENERATOR = ROOT / "scripts" / "make_pilot_campaign.py"


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


generator = load("make_pilot_campaign", GENERATOR) if GENERATOR.is_file() else None
needs_generator = pytest.mark.skipif(generator is None, reason="scripts/ is not part of the exported kit")
collect = load("kit_collect_pilot", KIT / "collect.py")
ROWS = yaml.safe_load(CAMPAIGN.read_text())["rows"]
BY_ID = {row["id"]: row for row in ROWS}
PILOTS = [row for row in ROWS if row.get("pilot")]
TRAIN = [row for row in ROWS if re.fullmatch(r"(g8|g32|sema|sfrz)-(chem|tool|chemtool|toolchem)-r[12]", row["id"])]
SCORING = [row for row in ROWS if "CUDA_VISIBLE_DEVICES" in row["env"] and not row.get("pilot")]


@needs_generator
def test_the_committed_campaign_is_what_the_generator_writes():
    assert CAMPAIGN.read_text() == generator.build()


def test_pilots_come_first_and_gate_everything():
    assert [row["id"] for row in ROWS[:9]] == ["pilot-score", "pilot-prefix-chemistry", "pilot-prefix-toolalpaca", "pilot-g8", "pilot-g32", "pilot-sema",
                                               "pilot-sfrz", "pilot-chain", "pilot-score-chain"]
    assert [row["id"] for row in PILOTS] == [row["id"] for row in ROWS[:9]]
    for earlier, later in zip(PILOTS, PILOTS[1:]):
        assert earlier["id"] in later["needs"], later["id"]
    assert all("pilot-score-chain" in row["needs"] for row in ROWS[9:]), "every later row needs the last pilot"
    chem = "datasets/sciknoweval/chemistry"
    assert all(BY_ID["pilot-%s" % r]["env"]["DATASET"] == chem and BY_ID["pilot-%s" % r]["env"]["STEPS"] == "2" for r in ("g8", "g32", "sema", "sfrz"))
    assert 'runs/pilot-g8-a*/hf-step2' in BY_ID["pilot-chain"]["command"][2] and BY_ID["pilot-chain"]["env"]["DATASET"] == "datasets/tooluse"
    assert 'runs/pilot-chain-a*/hf-step2' in BY_ID["pilot-score-chain"]["command"][2] and "--bed toolalpaca" in BY_ID["pilot-score-chain"]["command"][2]
    prepare = " ".join(step[2] for step in BY_ID["pilot-score"]["prepare"])
    assert "data/preprocess.py --data_source datasets/sciknoweval/chemistry" in prepare and "data/preprocess.py --data_source datasets/tooluse" in prepare


def test_twenty_four_runs_four_recipes_and_the_second_run_fixed_in_advance():
    assert len(TRAIN) == 24
    first = [row["id"] for row in TRAIN if row["id"].endswith("-r1")]
    second = [row["id"] for row in TRAIN if row["id"].endswith("-r2")]
    assert len(first) == 16 and len(second) == 8 and {i.split("-")[0] for i in second} == {"g8", "sema"}, "the two published recipes"
    assert {row["env"]["SEED"] for row in TRAIN if row["id"].endswith("-r1")} == {"42"} and {row["env"]["SEED"] for row in TRAIN if row["id"].endswith("-r2")} == {"43"}
    assert all(row["env"]["STEPS"] == "40" and row["env"]["TEST_FREQ"] == "20" for row in TRAIN)
    assert all(row["seed"] == int(row["id"][-1]) for row in TRAIN), "`seed:` is the run number in the name (r1, r2); the trainer's SEED is 42 or 43"
    settings = {"g8": {"LR": "1e-6", "MINI_BATCH": "8"}, "g32": {"LR": "1e-5", "MINI_BATCH": "32"},
                "sema": {"LR": "1e-5", "TEACHER_RATE": "0.05"}, "sfrz": {"LR": "1e-5", "TEACHER_RATE": "0"}}
    for row in TRAIN:
        recipe, chain, _rep = row["id"].split("-")
        launcher = "run_grpo_toolalpaca.sh" if recipe.startswith("g") else "run_sdpo_toolalpaca.sh"
        assert launcher in " ".join(row["command"]), row["id"]
        assert {k: row["env"][k] for k in settings[recipe]} == settings[recipe], row["id"]
        assert row["env"]["DATASET"] == ("datasets/tooluse" if chain in ("tool", "chemtool") else "datasets/sciknoweval/chemistry"), row["id"]


def test_a_second_stage_starts_from_its_own_recipes_first_stage_and_needs_it():
    for row in TRAIN:
        recipe, chain, rep = row["id"].split("-")
        if chain in ("chem", "tool"):
            assert row["command"] == ["bash", "{kit}/%s" % ("run_grpo_toolalpaca.sh" if recipe.startswith("g") else "run_sdpo_toolalpaca.sh")], "stage 1 starts from $MODEL_DIR"
            continue
        first = "%s-%s-%s" % (recipe, "chem" if chain == "chemtool" else "tool", rep)
        assert first in row["needs"], row["id"]
        assert 'MODEL_DIR="$(ls -d "{work}"/runs/%s-a*/hf-step40 | sort -V | tail -1)"' % first in row["command"][2], row["id"]


def test_every_checkpoint_is_scored_on_both_tasks_and_the_panel_on_one_gpu_in_sequence():
    assert {row["env"]["CUDA_VISIBLE_DEVICES"] for row in SCORING} == {"0"}
    assert all(later.get("wants") == [earlier["id"]] for earlier, later in zip(SCORING, SCORING[1:])), "scorings run one after another"
    assert all(row.get("order") == row.get("wants") for row in SCORING[1:]), "and say that the scoring before is order, not input"
    assert all("order" not in row for row in ROWS if row not in SCORING), "every other wanted row is read"
    direct = [row for row in SCORING if row["id"].startswith("direct-")]
    assert [row["id"] for row in direct] == ["direct-g8-chemtool-r1-chemistry", "direct-sema-chemtool-r1-chemistry"]
    assert all("--max-new-tokens 2048 --max-model-len 12288" in row["command"][2] for row in direct), "the serving budget in the SAME context"
    assert all("--max-new-tokens 8192 --max-model-len 12288" in row["command"][2] for row in SCORING if row not in direct)
    for key in ["base8b"] + [row["id"] for row in TRAIN]:
        for suffix in ("chemistry", "toolalpaca", "panel"):
            assert "sweep-%s-%s" % (key, suffix) in BY_ID, (key, suffix)
            if key == "base8b" and suffix != "panel":
                # the untrained model's task scorings are the prefix pilots' long generations: one scoring, checked and swept
                assert "%s-%s" % (key, suffix) not in BY_ID and BY_ID["pilot-prefix-%s" % suffix]["env"]["OUT_LONG"].endswith("/k8b/eval/base8b-%s-a{attempt}" % suffix)
                assert 'eval/base8b-%s-a*' % suffix in BY_ID["sweep-base8b-%s" % suffix]["command"][2]
                assert BY_ID["sweep-base8b-%s" % suffix]["wants"] == ["pilot-prefix-%s" % suffix], "it reads the pilot's scoring: a retried pilot re-runs it"
                continue
            assert "%s-%s" % (key, suffix) in BY_ID["sweep-%s-%s" % (key, suffix)]["needs"]
        if key != "base8b":
            assert key in BY_ID["%s-chemistry" % key]["needs"] and "stats-%s" % key in BY_ID
            # a failed run's partial rollouts are still summarised: the statistics WANT the run, they do not need it
            assert BY_ID["stats-%s" % key]["wants"] == [key] and key not in BY_ID["stats-%s" % key]["needs"]
    assert BY_ID["check-repeat"]["bars"][1] == {"name": "repeatable", "source": BY_ID["check-repeat"]["env"]["OUT"], "key": "text_agreement", "min": 0.99}


def test_the_report_waits_for_nothing_but_the_pilots_and_the_untrained_model():
    report = ROWS[-1]
    assert report["id"] == "report" and set(report["needs"]) == {"pilot-score-chain", "sweep-base8b-chemistry", "sweep-base8b-toolalpaca", "sweep-base8b-panel"}
    assert {row["id"] for row in TRAIN} <= set(report["wants"]) and sum(w.startswith("sweep-") for w in report["wants"]) == 72
    assert report["wants"][:2] == ["pilot-prefix-chemistry", "pilot-prefix-toolalpaca"], "it reads their checks, so it wants them"
    assert len(report["wants"]) == 198 and all(BY_ID[w].get("seed") in (1, 2) for w in report["wants"][2:]), "the rest are the per-run rows only"
    bars = {bar["key"]: bar for bar in report["bars"]}
    assert bars["lineage_ok"]["min"] == 1 and bars["prefix_checks_failed"]["max"] == 0 and bars["prefix_checks_found"]["min"] == 2
    assert bars["prefix_checks_missing_count"]["max"] == 0
    assert "CUDA_VISIBLE_DEVICES" not in report["env"]


def test_every_command_is_valid_shell_and_quotes_the_work_directory():
    for row in ROWS:
        text = " ".join(row["command"]) if row["command"][0] == "bash" and row["command"][1] != "-c" else row["command"][2]
        for name, value in (("{kit}", "/k it"), ("{work}", "/w ork"), ("{attempt}", "1")):
            text = text.replace(name, value)
        if row["command"][1] == "-c":
            assert subprocess.run(["bash", "-n", "-c", text], capture_output=True).returncode == 0, row["id"]
            assert "ls -d {work}" not in row["command"][2], row["id"]
        for step in row.get("prepare", []):
            assert subprocess.run(["bash", "-n", "-c", step[2].replace("{kit}", "/k")], capture_output=True).returncode == 0, row["id"]


def test_the_runner_plans_it_with_nothing_implicit(tmp_path):
    env = {**os.environ, "WORK": str(tmp_path), "KIT": str(KIT), "SDPO_DIR": "/x", "MODEL_DIR": "/x"}
    done = subprocess.run([sys.executable, str(KIT / "runner.py"), "plan", str(CAMPAIGN)], capture_output=True, text=True, env=env)
    assert done.returncode == 0, done.stderr
    assert "IMPLICIT ORDER" not in done.stdout + done.stderr


def test_everything_the_campaign_writes_is_somewhere_the_collector_packs():
    for row in ROWS:
        out = row["env"].get("OUT")
        if out is None or "/pilot/" in out:
            continue
        out = out.replace("{work}/", "").replace("{attempt}", "1")
        folder = next((part for part in Path(out).parts if part in ("eval", "forgetting")), "")
        names = {"eval": ("bed-score.json", "responses.jsonl", "tokens.jsonl"), "forgetting": ("forgetting.json", "responses.jsonl", "tokens.jsonl")}.get(
            folder, ("sweep.json", "per_item.jsonl", "rollout-stats.json", "pilot-report.json", "pilot-report.md"))
        for probe in ([out] if out.endswith(".json") else [out + "/" + name for name in names]):
            assert collect.wanted(Path(probe), Path(".")), "%s writes %s, which the collector would not pack" % (row["id"], probe)
    for name in ("run-summary.json", "metrics.jsonl", "env/argv.txt"):
        assert collect.wanted(Path("runs/g8-chem-r1-a1") / name, Path("."))


def test_prefix_reuse_is_qualified_on_this_model_before_anything_is_trained():
    """Every score at a shorter budget is read from a prefix of the long generation. The review of the send found that
    the only 8B check compared a long generation with another long one, which cannot show that. Now: for each task, a
    PILOT generates the untrained model at 8,192 and at 2,048 in one context on one GPU and requires 99 percent
    agreement; and two trained checkpoints are checked the same way against the very scoring that is swept."""
    for bed, n in (("chemistry", 210), ("toolalpaca", 68)):
        row = BY_ID["pilot-prefix-%s" % bed]
        long, short, check = row["command"][2].split(" && ")
        assert "--max-new-tokens 8192 --max-model-len 12288" in long and "--max-new-tokens 2048 --max-model-len 12288" in short
        assert "cap_sweep.py\" check" in check and row["env"]["OUT_LONG"] in check and row["env"]["OUT_SHORT"] in check
        assert row["env"]["CUDA_VISIBLE_DEVICES"] == "0" and row["env"]["OUT"].endswith("/k8b/report-prefix/prefix-base8b-%s-a{attempt}.json" % bed)
        assert row["env"]["OUT_LONG"].endswith("/k8b/eval/base8b-%s-a{attempt}" % bed), "the check is of the scoring that is swept"
        assert row["bars"] == [{"name": "compared", "source": row["env"]["OUT"], "key": "compared", "min": n},
                               {"name": "prefix-agrees", "source": row["env"]["OUT"], "key": "text_agreement", "min": 0.99}]
    for key in ("g8-chemtool-r1", "sema-chemtool-r1"):
        row = BY_ID["prefix-%s-chemistry" % key]
        assert {"%s-chemistry" % key, "direct-%s-chemistry" % key} <= set(row["needs"]) and row["seed"] == 1
        assert 'eval/%s-chemistry-a*' % key in row["command"][2] and 'eval/direct-%s-chemistry-a*' % key in row["command"][2]
        assert row["env"]["OUT"].endswith("/k8b/report-prefix/prefix-%s-chemistry-a{attempt}.json" % key) and row["bars"][1]["min"] == 0.99
        assert row["id"] in ROWS[-1]["wants"]


def test_disk_is_checked_before_anything_runs_and_trainer_checkpoints_are_not_kept(tmp_path):
    first = BY_ID["pilot-score"]["prepare"][0][2]
    assert "shutil.disk_usage" in first and "K8B_MIN_FREE_GB" in first and "'650'" in first and first.rstrip().endswith('"{work}"')
    # ... on a work folder that does not exist yet, which is what the README's command starts from (review, round 2)
    fresh = tmp_path / "not" / "there" / "yet"
    done = subprocess.run(["bash", "-c", first.replace("{work}", str(fresh))], env={**os.environ, "K8B_MIN_FREE_GB": "0"}, capture_output=True, text=True)
    assert done.returncode == 0 and fresh.is_dir(), done.stderr
    done = subprocess.run(["bash", "-c", first.replace("{work}", str(fresh))], env={**os.environ, "K8B_MIN_FREE_GB": "99999999"}, capture_output=True, text=True)
    assert done.returncode == 1 and "GB free under" in done.stderr
    training = TRAIN + [BY_ID["pilot-%s" % name] for name in ("g8", "g32", "sema", "sfrz", "chain")]
    assert len(training) == 29 and all(row["env"]["KEEP_TRAINER_CKPT"] == "0" for row in training)


def test_every_row_that_reads_a_pilots_output_wants_that_pilot():
    """Round 3 of the review: a row that only NEEDS a pilot is gated by it and not re-run when the pilot is retried,
    which is right for a row that merely waits and wrong for one that reads what the pilot wrote."""
    readers = {"pilot-chain": "pilot-g8", "pilot-score-chain": "pilot-chain", "check-repeat": "pilot-prefix-chemistry",
               "sweep-base8b-chemistry": "pilot-prefix-chemistry", "sweep-base8b-toolalpaca": "pilot-prefix-toolalpaca"}
    for reader, producer in readers.items():
        assert producer in BY_ID[reader]["wants"], reader
    assert {"pilot-prefix-chemistry", "pilot-prefix-toolalpaca"} <= set(BY_ID["report"]["wants"])
    # and nothing else names a pilot among its wanted rows: a retried pilot must not re-run a training row
    assert all(not any(w.startswith("pilot-") for w in row.get("wants") or []) for row in ROWS if row["id"] not in list(readers) + ["report"])
