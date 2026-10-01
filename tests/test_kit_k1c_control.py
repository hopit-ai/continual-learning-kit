"""kit/campaigns/k1c-control.yaml: plan v3 package 2, the small-model control on GSM8K.

Three cells of a two-by-two whose fourth cell is K1c's own five runs. The file is generated; this pins what makes
the comparison readable: K1c's data, dose and command with only the cell's settings changed, a pilot on the two new
settings, one GPU for scoring, and outputs the collector packs.
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
KIT = ROOT / "kit"
CAMPAIGN = KIT / "campaigns" / "k1c-control.yaml"
GENERATOR = ROOT / "scripts" / "make_k1c_control_campaign.py"


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


generator = load("make_k1c_control_campaign", GENERATOR) if GENERATOR.is_file() else None
needs_generator = pytest.mark.skipif(generator is None, reason="scripts/ is not part of the exported kit")
collect = load("kit_collect_control", KIT / "collect.py")
ROWS = yaml.safe_load(CAMPAIGN.read_text())["rows"]
BY_ID = {row["id"]: row for row in ROWS}
TRAIN = [row for row in ROWS if "TRAIN_FILE" in row["env"] and not row.get("pilot")]
CELLS = {"lr6": {"LR": "1e-6"}, "gate": {"LR": "1e-5", "MAX_RESPONSE": "2048", "FINISH_GATE": "1"}, "lr6-gate": {"LR": "1e-6", "MAX_RESPONSE": "2048", "FINISH_GATE": "1"}}


@needs_generator
def test_the_committed_campaign_is_what_the_generator_writes():
    assert CAMPAIGN.read_text() == generator.build()


def test_nine_runs_in_three_cells_with_only_the_cells_settings_changed_from_k1c():
    assert [row["id"] for row in TRAIN] == ["%s-seed%d" % (cell, seed) for cell in CELLS for seed in (0, 1, 2)]
    k1c = yaml.safe_load((KIT / "campaigns" / "k1c-grpo-baselines.yaml").read_text())["rows"]
    theirs = next(row for row in k1c if row["id"] == "gsm8k-seed0")
    for row in TRAIN:
        cell = row["id"].rsplit("-seed", 1)[0]
        assert row["command"] == theirs["command"], "the same launcher call, from the same untrained model"
        same = {k: v for k, v in row["env"].items() if k not in ("NAME", "SEED", "DUMP_ATTEMPTS", "LR", "MAX_RESPONSE", "FINISH_GATE")}
        assert same == {k: v for k, v in theirs["env"].items() if k not in ("NAME", "SEED")}, row["id"]
        assert {k: row["env"].get(k) for k in ("LR", "MAX_RESPONSE", "FINISH_GATE")} == {"LR": CELLS[cell]["LR"], "MAX_RESPONSE": CELLS[cell].get("MAX_RESPONSE"), "FINISH_GATE": CELLS[cell].get("FINISH_GATE")}
        assert row["env"]["DUMP_ATTEMPTS"] == "1" and row["seed"] == int(row["env"]["SEED"])
        bars = {bar["name"]: bar for bar in row["bars"]}
        cap, gate = int(CELLS[cell].get("MAX_RESPONSE", 8192)), int(CELLS[cell].get("FINISH_GATE", 0))
        assert (bars["training-cap"]["min"], bars["training-cap"]["max"]) == (cap, cap) and (bars["finish-gate"]["min"], bars["finish-gate"]["max"]) == (gate, gate)


def test_the_pilot_exercises_the_two_new_settings_and_gates_every_row():
    pilot = ROWS[0]
    assert pilot["id"] == "pilot-gate" and pilot["pilot"] is True and pilot["env"]["STEPS"] == "2"
    assert pilot["env"]["MAX_RESPONSE"] == "2048" and pilot["env"]["FINISH_GATE"] == "1"
    assert all("pilot-gate" in row["needs"] for row in ROWS[1:])
    launcher = (KIT / "run_grpo.sh").read_text()
    assert '"max_response_length": ${MAX_RESPONSE:-8192},' in launcher and '"finish_gate": $FINISH_GATE,' in launcher, "the bars read numbers the launcher writes"


def test_scoring_is_one_gpu_in_sequence_at_the_long_cap_and_the_reference_is_scored_here():
    scoring = [row for row in ROWS if "CUDA_VISIBLE_DEVICES" in row["env"]]
    assert len(scoring) == 21 and {row["env"]["CUDA_VISIBLE_DEVICES"] for row in scoring} == {"0"}
    assert [row["id"] for row in scoring[:3]] == ["base17b-gsm8k", "base17b-panel", "direct-base17b-gsm8k"]
    assert all(later.get("wants") == [earlier["id"]] for earlier, later in zip(scoring, scoring[1:]))
    assert all(row.get("order") == row.get("wants") for row in scoring[1:]) and all("order" not in row for row in ROWS if row not in scoring)
    direct = BY_ID["direct-base17b-gsm8k"]
    assert "--max-new-tokens 2048 --max-model-len 12288" in direct["command"][2], "the serving budget in the SAME context"
    assert all("--max-new-tokens 8192 --max-model-len 12288" in row["command"][2] for row in scoring if row is not direct)
    for cell in CELLS:
        row = BY_ID["budget-%s" % cell]
        assert {"sweep-base17b-gsm8k", "prefix-base17b-gsm8k"} <= set(row["needs"]) and row["wants"] == ["sweep-%s-seed%d-gsm8k" % (cell, s) for s in (0, 1, 2)]
        assert '--ref "$(ls -d "{work}/k1c/control"/report-sweep/base17b-gsm8k-a*' in row["command"][2]
        assert '--qualify "ref=$(ls -d "{work}/k1c/control"/report-prefix/prefix-base17b-gsm8k-a*.json | sort -V | tail -1)"' in row["command"][2]


def test_prefix_reuse_is_checked_on_the_reference_and_a_failed_run_costs_only_itself():
    check = BY_ID["prefix-base17b-gsm8k"]
    assert {"base17b-gsm8k", "direct-base17b-gsm8k"} <= set(check["needs"]) and "cap_sweep.py\" check" in check["command"][2]
    assert check["bars"] == [{"name": "compared", "source": check["env"]["OUT"], "key": "compared", "min": 300},
                             {"name": "prefix-agrees", "source": check["env"]["OUT"], "key": "text_agreement", "min": 0.99}]
    for row in TRAIN:
        stats = BY_ID["stats-%s" % row["id"]]
        assert stats["wants"] == [row["id"]] and row["id"] not in stats["needs"], "a failed run's rollouts are still summarised"


def test_the_last_row_applies_the_pre_registered_rules():
    report = ROWS[-1]
    assert report["id"] == "report" and "control_report.py" in report["command"][2] and "--serving 2048 --diagnostic 8192" in report["command"][2]
    assert set(report["needs"]) == {"pilot-gate", "sweep-base17b-gsm8k", "prefix-base17b-gsm8k"}
    assert len(report["wants"]) == 54 and all(BY_ID[w].get("seed") in (0, 1, 2) for w in report["wants"])
    assert {bar["key"] for bar in report["bars"]} == {"runs_reported", "accounting_exact", "lineage_ok"}
    assert (KIT / "control_report.py").is_file()


def test_shell_runner_and_collector(tmp_path):
    for row in ROWS:
        text = row["command"][2].replace("{kit}", "/k it").replace("{work}", "/w ork").replace("{attempt}", "1")
        assert subprocess.run(["bash", "-n", "-c", text], capture_output=True).returncode == 0, row["id"]
        out = row["env"].get("OUT")
        if out:
            out = out.replace("{work}/", "").replace("{attempt}", "1")
            folder = next((part for part in Path(out).parts if part in ("eval", "forgetting")), "")
            names = {"eval": ("bed-score.json", "tokens.jsonl"), "forgetting": ("forgetting.json", "tokens.jsonl")}.get(folder, ("sweep.json", "rollout-stats.json", "budget-report.json", "control-report.json", "control-report.md"))
            probes = [Path(out)] if out.endswith(".json") else [Path(out) / name for name in names]
            assert all(collect.wanted(probe, Path(".")) for probe in probes), row["id"]
    env = {**os.environ, "WORK": str(tmp_path), "KIT": str(KIT), "SDPO_DIR": "/x", "GSM8K_ROOT": "/x"}
    done = subprocess.run([sys.executable, str(KIT / "runner.py"), "plan", str(CAMPAIGN)], capture_output=True, text=True, env=env)
    assert done.returncode == 0 and "IMPLICIT ORDER" not in done.stdout + done.stderr, done.stderr
