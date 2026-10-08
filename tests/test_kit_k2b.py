"""Package K2b: the SFT launcher, the GSM8K bed's SFT export, the generated campaign and the between-route report.

Nothing here trains, generates or scores a model. The launcher is checked by its DRY_RUN argv and by one run
with `python` stubbed on PATH; the export and the report run on small fixture trees."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import random
import re
import subprocess
import sys
from pathlib import Path

import pytest


# ------------------------------------------------------------------ the shell that runs pytest decides nothing here
def _launcher_variables() -> frozenset:
    """Every variable a kit launcher reads from its environment (`${NAME:-default}` or `${NAME:?required}`), collected
    from the launchers themselves so that a new knob is covered the day it is added, plus the two the launchers
    export for the reward function."""
    kit = __import__("pathlib").Path(__file__).resolve().parents[1] / "kit"
    found = set()
    for script in sorted(kit.glob("*.sh")):
        found |= set(__import__("re").findall(r"\$\{([A-Z_][A-Z0-9_]*):[-?]", script.read_text()))
    return frozenset(found - {"PYTHONPATH", "USER"}) | {"KIT_FINISH_GATE", "KIT_LENGTH_BUDGET_CHARS"}


def shell() -> dict:
    """os.environ WITHOUT any launcher variable. A test passes every setting it means and inherits none: the partner
    README tells people to `export NGPU=8`, and a suite run in that shell failed a test that expects the launcher's
    default of four GPUs (found on 1 October 2026, verifying the public tag from a fresh clone)."""
    names = _launcher_variables()
    return {k: v for k, v in __import__("os").environ.items() if k not in names}

ROOT = Path(__file__).resolve().parents[1]
KIT = ROOT / "kit"
SCRIPT = KIT / "run_sft.sh"
CAMPAIGN = KIT / "campaigns" / "k2b-route.yaml"
# The pinned fork, if a checkout of it sits in references/ here or in a repository this checkout lives inside.
PINNED = next((base / "references" / "SDPO" for base in (ROOT, *ROOT.parents)
               if (base / "references" / "SDPO" / "verl/trainer/config/sft_trainer.yaml").is_file()), None)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gsm8k = _load("kit_bed_gsm8k_for_k2b", KIT / "beds" / "gsm8k.py")
generator = _load("make_k2b_campaign", ROOT / "scripts" / "make_k2b_campaign.py") if (ROOT / "scripts" / "make_k2b_campaign.py").is_file() else None
needs_generator = pytest.mark.skipif(generator is None, reason="scripts/ is not part of the exported kit")
k2b_report = _load("kit_k2b_report", KIT / "k2b_report.py")
runner = _load("kit_runner_for_k2b", KIT / "runner.py")


def campaign() -> dict:
    os.environ.setdefault("WORK", "/tmp/unused")
    return runner.load_campaign(CAMPAIGN)


def rows() -> dict:
    return {row["id"]: row for row in campaign()["rows"]}


# ------------------------------------------------------------------------------------ the campaign
@needs_generator
def test_the_committed_campaign_is_what_its_generator_builds():
    assert CAMPAIGN.read_text() == generator.build(), "re-run scripts/make_k2b_campaign.py"


def test_the_pilot_gates_every_later_row_and_each_route_must_damage_spider_by_5():
    found = campaign()["rows"]
    pilots = [r["id"] for r in found if r.get("pilot")]
    assert pilots == ["q17-original", "q17-repeatable", "rl-seed0", "sft-seed0", "pilot-rl-damaged-spider", "pilot-sft-damaged-spider"]
    last = max(i for i, r in enumerate(found) if r.get("pilot"))
    for row in found[last + 1:]:
        assert set(pilots) <= set(row["needs"]), "%s can run before the pilot has passed" % row["id"]
    by_id = rows()
    for route in ("rl", "sft"):
        row = by_id["pilot-%s-damaged-spider" % route]
        command = row["command"][-1]
        assert "{work}/k2b/eval/q17-spider-a*" in command and "{work}/k2b/eval/%s-seed0-spider-a*" % route in command
        assert command.index("q17-spider") < command.index("%s-seed0-spider" % route), "--a is the untrained model, --b the trained one"
        bars = {bar["name"]: bar for bar in row["bars"]}
        assert bars["spider-fell-by-5"]["key"] == "delta" and bars["spider-fell-by-5"]["max"] == -5 and "min" not in bars["spider-fell-by-5"]
        assert bars["same-machine"]["key"] == "same_machine_flag" and bars["same-machine"]["min"] == 1
        gain = by_id["pilot-%s-gsm8k-gain" % route]
        assert not gain.get("pilot") and [b["name"] for b in gain["bars"]] == ["same-machine"], "the GSM8K gain is written, not gated"


def test_ten_repeats_per_route_each_on_its_own_launcher_from_the_untrained_model():
    by_id = rows()
    for route, launcher in (("rl", "run_grpo.sh"), ("sft", "run_sft.sh")):
        trains = [r for r in by_id.values() if re.fullmatch(r"%s-seed\d+" % route, r["id"])]
        assert [r["id"] for r in trains] == ["%s-seed%d" % (route, s) for s in range(10)]
        for seed, row in enumerate(trains):
            assert row["command"][-1] == 'MODEL_DIR="{work}/models/Qwen3-1.7B" bash "{kit}/%s"' % launcher
            assert row["env"]["SEED"] == str(seed) and row["env"]["STEPS"] == "40" and row["env"]["NGPU"] == "8"


def test_the_routes_training_rows_differ_only_in_the_launcher_and_the_route():
    """The same 1,280 questions reach both trainers, each in its own format (gsm8k-1280 and its SFT export), and
    each SFT repeat in its own order (gsm8k-1280-sft-seed<S>)."""
    by_id = rows()
    for seed in range(10):
        rl, sft = (json.dumps({k: v for k, v in by_id["%s-seed%d" % (r, seed)].items() if k not in ("needs", "_stated_needs")}, sort_keys=True) for r in ("rl", "sft"))
        swapped = (rl.replace("run_grpo.sh", "run_sft.sh").replace("rl-seed", "sft-seed")
                   .replace("/gsm8k-1280/train.parquet", "/gsm8k-1280-sft-seed%d/train.parquet" % seed)
                   .replace("/gsm8k-1280/heldout.parquet", "/gsm8k-1280-sft/heldout.parquet"))
        assert swapped == sft, seed
    prepare = " ".join(json.dumps(step) for step in by_id["q17-original"]["prepare"])
    assert "prepare --gsm8k-root \\\"$GSM8K_ROOT\\\" --out \\\"{work}/data/gsm8k-1280\\\" --limit 1280" in prepare, "K3's stage-B file, prepared as K3 prepares it"
    assert "sft-export --from \\\"{work}/data/gsm8k-1280/heldout.parquet\\\"" in prepare
    assert "sft-export --from \\\"{work}/data/gsm8k-1280/train.parquet\\\"" not in prepare, "no SFT repeat reads an unseeded train file"
    k3 = (KIT / "campaigns" / "k3-replay.yaml").read_text()
    assert 'python \\"{kit}/beds/gsm8k.py\\" prepare --gsm8k-root \\"$GSM8K_ROOT\\" --out \\"{work}/data/gsm8k-1280\\" --limit 1280' in k3


def test_each_sft_repeat_reads_its_own_seeds_order_and_the_rl_rows_are_unchanged():
    by_id = rows()
    ids = list(by_id)
    for seed in range(10):
        sft, data = by_id["sft-seed%d" % seed], by_id["sft-seed%d-data" % seed]
        own = "{work}/data/gsm8k-1280-sft-seed%d" % seed
        assert sft["env"]["TRAIN_FILE"] == own + "/train.parquet" and sft["env"]["VAL_FILE"] == "{work}/data/gsm8k-1280-sft/heldout.parquet"
        assert own + "/train.parquet" in sft["requires"] and "sft-seed%d-data" % seed in sft["needs"]
        assert ids.index("sft-seed%d-data" % seed) < ids.index("sft-seed%d" % seed)
        command = data["command"][-1]
        assert command.endswith('sft-export --from "{work}/data/gsm8k-1280/train.parquet" --gsm8k-root "$GSM8K_ROOT" --out "%s" --seed %d' % (own, seed))
        assert not data.get("pilot") and data["requires"] == ["{work}/data/gsm8k-1280/train.parquet"]
        bars = {bar["name"]: bar for bar in data["bars"]}
        assert {b["source"] for b in bars.values()} == {own + "/train.manifest.json"}
        assert bars["exported"]["min"] == bars["every-target-correct"]["min"] == 1280
        assert bars["its-own-order"]["key"] == "seed" and bars["its-own-order"]["min"] == bars["its-own-order"]["max"] == seed
        rl = by_id["rl-seed%d" % seed]
        assert rl["env"] == {"NAME": "rl-seed%d-a{attempt}" % seed, "WORK": "{work}", "TRAIN_FILE": "{work}/data/gsm8k-1280/train.parquet",
                             "VAL_FILE": "{work}/data/gsm8k-1280/heldout.parquet", "STEPS": "40", "SAVE_FREQ": "40", "SEED": str(seed),
                             "NGPU": "8", "FILE_LOG": "1"}
        assert rl["requires"] == ["{work}/data/gsm8k-1280/train.parquet", "{work}/data/gsm8k-1280/heldout.parquet", "{work}/models/Qwen3-1.7B/config.json"]
    assert [r for r in by_id if r.startswith("rl-") and "gsm8k-1280-sft" in json.dumps(by_id[r])] == [], "no RL row touches an SFT file"
    assert [r for r in by_id if r.endswith("-data")] == ["sft-seed%d-data" % s for s in range(10)]


def test_every_subject_is_repaired_measured_and_joined_to_the_original_and_one_control():
    by_id = rows()
    subjects = ["%s-seed%d" % (route, seed) for seed in range(10) for route in ("rl", "sft")]
    for sub in subjects:
        for part in ("spider", "gsm8k", "before", "damage-check", "repair", "after50", "after100", "after200", "after300", "drift-damaged"):
            assert "%s-%s" % (sub, part) in by_id, (sub, part)
        assert "--steps 300 --save-at 50,100,200,300 --lr 1e-4" in by_id["%s-repair" % sub]["command"][-1], "K2's repair, unchanged"
        assert "{work}/k2b/q17-targets.jsonl" in by_id["%s-repair" % sub]["command"][-1], "the ORIGINAL's answers are the targets"
        after = by_id["%s-after300" % sub]["command"][-1]
        assert "{work}/k2b/forgetting/%s-after300-a{attempt}" % sub in after and "{work}/k2b/eval/%s-after300-spider-a{attempt}" % sub in after
        assert '--reference "{work}/models/Qwen3-1.7B"' in after
    assert [r for r in by_id if "control" in r] == ["q17-control300"]
    assert len(by_id) == 242
    report_row = by_id["report"]
    assert {"%s-after300" % s for s in subjects} <= set(report_row["wants"]), "per-seed rows are wanted, not needed"
    assert report_row["_stated_needs"] == ["pilot-sft-damaged-spider", "q17-control300"]
    assert {b["key"]: b["min"] for b in report_row["bars"]} == {"subjects_reported": 20, "comparable": 1}


def test_every_bar_reads_a_key_some_tool_writes_as_a_number():
    numeric = {"total_correct", "identical_answers", "changed_verdicts", "same_machine_flag", "delta", "n", "returncode", "merged", "steps",
               "step", "relative_distance_all", "targets", "empty", "subjects_reported", "comparable", "rows", "scored_correct", "seed"}
    assert [(r["id"], b["key"]) for r in campaign()["rows"] for b in r["bars"] if b["key"] not in numeric] == []


# ------------------------------------------------------------------------------------ the SFT export
SOLUTIONS = [
    ("Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
     "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nNatalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n#### 72"),
    ("Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
     "Weng earns 12/60 = $<<12/60=0.2>>0.2 per minute.\nWorking 50 minutes, she earned 0.2 x 50 = $<<0.2*50=10>>10.\n#### 10"),
    ("A farm has 1,200 hens and sells 1,050 eggs a day at $0.50. What is the daily income?",
     "The answer: first, the income is 1,050 x 0.5 = $<<1050*0.5=525>>525.\n#### 525"),
    ("Tom had 5 apples and gave away 8. How many does he have now, as a signed number?",
     "5 - 8 = <<5-8=-3>>-3\n#### -3"),
]


@pytest.fixture()
def gsm8k_root(tmp_path):
    root = tmp_path / "gsm8k"
    root.mkdir()
    (root / "train.jsonl").write_text("".join(json.dumps({"question": q, "answer": a}) + "\n" for q, a in SOLUTIONS))
    tests = [("Test question %d: what is %d plus %d?" % (i, i, i), "%d+%d=<<%d+%d=%d>>%d\n#### %d" % (i, i, i, i, 2 * i, 2 * i, 2 * i)) for i in range(5)]
    (root / "test.jsonl").write_text("".join(json.dumps({"question": q, "answer": a}) + "\n" for q, a in tests))
    prepared = tmp_path / "prepared"
    assert gsm8k.main(["prepare", "--gsm8k-root", str(root), "--out", str(prepared), "--allow-subset", "--heldout-n", "3"]) == 0
    return root, prepared


def test_the_export_writes_the_rl_prompt_and_a_gold_solution_the_bed_marks_correct(gsm8k_root, tmp_path):
    pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq                                                    # noqa: PLC0415
    root, prepared = gsm8k_root
    out = tmp_path / "sft"
    for split in ("train", "heldout"):
        source = prepared / ("%s.parquet" % split)
        assert gsm8k.main(["sft-export", "--from", str(source), "--gsm8k-root", str(root), "--out", str(out), "--allow-subset"]) == 0
        rl = pq.read_table(source).to_pylist()
        sft = pq.read_table(out / ("%s.parquet" % split)).to_pylist()
        assert len(sft) == len(rl) and set(sft[0]) == {"id", "prompt", "response"}
        for rl_row, sft_row in zip(rl, sft):
            assert sft_row["prompt"] == rl_row["prompt"][0]["content"], "the prompt is the text the GRPO trainer sees"
            assert sft_row["response"].split("\n")[-1] == "Answer: %s" % rl_row["reward_model"]["ground_truth"]
            assert "<<" not in sft_row["response"] and "####" not in sft_row["response"]
            assert gsm8k.compute_score("gsm8k", sft_row["response"], rl_row["reward_model"]["ground_truth"])["acc"] == 1.0
        manifest = json.loads((out / ("%s.manifest.json" % split)).read_text())
        assert manifest["rows"] == manifest["scored_correct"] == len(rl)
        assert manifest["parquet_sha256"] == hashlib.sha256((out / ("%s.parquet" % split)).read_bytes()).hexdigest()
        assert manifest["jsonl_sha256"] == hashlib.sha256((out / ("%s.jsonl" % split)).read_bytes()).hexdigest()
    first = pq.read_table(out / "train.parquet").to_pylist()[0]["response"]
    assert first == "Natalia sold 48/2 = 24 clips in May.\nNatalia sold 48+24 = 72 clips altogether in April and May.\nAnswer: 72"
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        gsm8k.main(["sft-export", "--from", str(prepared / "train.parquet"), "--gsm8k-root", str(root), "--out", str(out), "--allow-subset"])


def test_the_ten_seeded_exports_hold_the_same_rows_each_in_its_own_order(tmp_path):
    """30 training questions, so ten seeds' orders are distinct with certainty; no seed is the source's order."""
    pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq                                                    # noqa: PLC0415
    root = tmp_path / "gsm8k"
    root.mkdir()
    train = [("Train question %d: what is %d times 3?" % (i, i), "%d*3=<<%d*3=%d>>%d\n#### %d" % (i, i, 3 * i, 3 * i, 3 * i)) for i in range(30)]
    tests = [("Test question %d: what is %d plus %d?" % (i, i, i), "%d+%d=<<%d+%d=%d>>%d\n#### %d" % (i, i, i, i, 2 * i, 2 * i, 2 * i)) for i in range(5)]
    (root / "train.jsonl").write_text("".join(json.dumps({"question": q, "answer": a}) + "\n" for q, a in train))
    (root / "test.jsonl").write_text("".join(json.dumps({"question": q, "answer": a}) + "\n" for q, a in tests))
    prepared = tmp_path / "prepared"
    assert gsm8k.main(["prepare", "--gsm8k-root", str(root), "--out", str(prepared), "--allow-subset", "--heldout-n", "3"]) == 0
    source = prepared / "train.parquet"
    export = ["sft-export", "--from", str(source), "--gsm8k-root", str(root), "--allow-subset"]

    assert gsm8k.main(export + ["--out", str(tmp_path / "sft")]) == 0
    unseeded = pq.read_table(tmp_path / "sft" / "train.parquet").to_pylist()
    source_ids = [row["extra_info"]["index"] for row in pq.read_table(source).to_pylist()]
    assert [r["id"] for r in unseeded] == source_ids, "no seed is today's order: the source's"
    assert not {"seed", "order_sha256"} & set(json.loads((tmp_path / "sft" / "train.manifest.json").read_text()))

    orders = []
    for seed in range(10):
        out = tmp_path / ("sft-seed%d" % seed)
        assert gsm8k.main(export + ["--out", str(out), "--seed", str(seed)]) == 0
        seeded = pq.read_table(out / "train.parquet").to_pylist()
        assert sorted(seeded, key=lambda r: r["id"]) == sorted(unseeded, key=lambda r: r["id"]), "the same rows, byte for byte"
        expected = list(unseeded)
        random.Random(seed).shuffle(expected)
        assert seeded == expected, "the order random.Random(seed).shuffle gives"
        assert [json.loads(line) for line in (out / "train.jsonl").read_text().splitlines()] == seeded
        manifest = json.loads((out / "train.manifest.json").read_text())
        ids = [r["id"] for r in seeded]
        assert manifest["seed"] == seed and manifest["rows"] == manifest["scored_correct"] == 30
        assert manifest["order_sha256"] == hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()
        orders.append(tuple(ids))
    assert len(set(orders)) == 10 and tuple(source_ids) not in orders, "ten repeats, ten orders"
    again = tmp_path / "sft-seed3-again"
    assert gsm8k.main(export + ["--out", str(again), "--seed", "3"]) == 0
    assert (again / "train.parquet").read_bytes() == (tmp_path / "sft-seed3" / "train.parquet").read_bytes(), "a seed's order is deterministic"


def test_the_export_refuses_a_dataset_copy_that_is_not_the_one_the_rows_came_from(gsm8k_root, tmp_path):
    pytest.importorskip("pyarrow")
    root, prepared = gsm8k_root
    changed = [(q, a.replace("#### 72", "#### 73")) for q, a in SOLUTIONS]
    (root / "train.jsonl").write_text("".join(json.dumps({"question": q, "answer": a}) + "\n" for q, a in changed))
    with pytest.raises(SystemExit, match="gold"):
        gsm8k.main(["sft-export", "--from", str(prepared / "train.jsonl"), "--gsm8k-root", str(root), "--out", str(tmp_path / "sft"), "--allow-subset"])
    assert not (tmp_path / "sft").exists(), "a refusal writes nothing"


def test_the_export_refuses_a_target_the_bed_would_mark_wrong(gsm8k_root, tmp_path, monkeypatch):
    pytest.importorskip("pyarrow")
    root, prepared = gsm8k_root
    monkeypatch.setattr(gsm8k, "sft_response", lambda solution, gold: "Answer: %s" % (gsm8k.to_number(gold) + 1))
    with pytest.raises(SystemExit, match="do not score correct"):
        gsm8k.main(["sft-export", "--from", str(prepared / "train.jsonl"), "--gsm8k-root", str(root), "--out", str(tmp_path / "sft"), "--allow-subset"])
    assert not (tmp_path / "sft").exists()


# ------------------------------------------------------------------------------------ run_sft.sh
EXPECTED = [
    "python", "-m", "torch.distributed.run", "--standalone", "--nnodes=1", "--nproc_per_node=8",
    str(KIT / "sft_entry.py"), "--config-name", "sft_trainer",
    "data.train_files=[/d/sft/train.parquet]",
    "data.val_files=[/d/sft/heldout.parquet]",
    "data.prompt_key=prompt",
    "data.response_key=response",
    "data.max_length=2048",
    "data.truncation=error",
    "data.train_batch_size=32",
    "data.micro_batch_size_per_gpu=4",
    "+data.apply_chat_template_kwargs.enable_thinking=False",
    "model.partial_pretrain=/m/Qwen3-1.7B",
    "model.strategy=fsdp2",
    "model.lora_rank=0",
    "optim.lr=1e-5",
    "optim.betas=[0.9,0.999]",
    "optim.weight_decay=0.01",
    "optim.clip_grad=1.0",
    "optim.lr_scheduler=wsd",
    "optim.lr_warmup_steps_ratio=0.125",
    "trainer.total_epochs=2",
    "trainer.total_training_steps=40",
    "trainer.save_freq=40",
    "trainer.test_freq=-1",
    "trainer.checkpoint.save_contents=[model]",
    "trainer.resume_mode=disable",
    "trainer.default_local_dir=/w/runs/sft-seed3-a1/train",
    "trainer.project_name=k2b-sft",
    "trainer.experiment_name=sft-seed3-a1",
    "trainer.logger=[console,file]",
    "trainer.nnodes=1",
    "trainer.n_gpus_per_node=8",
    "trainer.seed=3",
]


def dry(**extra):
    env = {k: v for k, v in shell().items() if k not in ("SEED", "STEPS", "SAVE_FREQ", "NGPU", "LR", "FILE_LOG")}
    env.update(SDPO_DIR="/sdpo", MODEL_DIR="/m/Qwen3-1.7B", NAME="sft-seed3-a1", TRAIN_FILE="/d/sft/train.parquet",
               VAL_FILE="/d/sft/heldout.parquet", WORK="/w", DRY_RUN="1", STEPS="40", SAVE_FREQ="40", SEED="3", NGPU="8", FILE_LOG="1")
    env.update(extra)
    return subprocess.run(["bash", str(SCRIPT)], env=env, capture_output=True, text=True)


def test_the_script_parses():
    assert subprocess.run(["bash", "-n", str(SCRIPT)], capture_output=True, text=True).returncode == 0


def test_the_dry_run_argv_is_the_expected_list():
    done = dry()
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip().split("\n") == EXPECTED
    unseeded = dry(SEED="", FILE_LOG="0").stdout.strip().split("\n")
    assert unseeded == [a.replace("[console,file]", "[console]") for a in EXPECTED[:-1]]


def test_the_launcher_refuses_a_dose_its_schedule_was_not_planned_for():
    assert dry(STEPS="41").returncode == 2, "more than one pass"
    assert dry(NGPU="3").returncode == 2, "a batch of 32 does not split over 3 GPUs"
    assert dry(LR="fast").returncode == 2
    small = dry(NGPU="16").stdout.split("\n")
    assert "data.micro_batch_size_per_gpu=2" in small and "--nproc_per_node=16" in small


@pytest.mark.skipif(PINNED is None, reason="the pinned lasgroup/SDPO checkout is not on this machine")
def test_every_key_exists_in_the_pinned_sft_config_and_the_one_added_key_does_not():
    yaml = pytest.importorskip("yaml")
    config = yaml.safe_load((PINNED / "verl/trainer/config/sft_trainer.yaml").read_text())
    optim_defaults = yaml.safe_load((PINNED / "verl/trainer/config/optim/fsdp.yaml").read_text())
    config["optim"] = {**optim_defaults, **config["optim"]}
    for token in EXPECTED[10:]:
        key = token.split("=", 1)[0]
        node, parts = config, key.lstrip("+").split(".")
        for part in parts[:-1]:
            node = node[part]
        if key.startswith("+"):
            assert parts[-1] not in node, "%s exists: Hydra refuses `+` on an existing key" % key
        else:
            assert parts[-1] in node, "%s is not in the pinned sft_trainer.yaml" % key
    trainer = (PINNED / "verl/trainer/fsdp_sft_trainer.py").read_text()
    assert "self.config.optim.lr_warmup_steps_ratio" in trainer and '"wsd"' in trainer
    assert "self.steps_per_epoch * self.config.trainer.total_epochs" in trainer, "the scheduler plans over epochs, which is why EPOCHS=2"
    assert 'f"global_step_{step}"' in trainer, "checkpoints land in global_step_<N>/ itself, not .../actor"


STUB = """#!/usr/bin/env bash
if [ "$1" = "-c" ]; then exec "$REAL_PYTHON" "$@"; fi
n=$(ls "$RECORD"/call-*.argv 2>/dev/null | wc -l | tr -d ' ')
printf '%s\\n' "$@" > "$RECORD/call-$n.argv"
env > "$RECORD/call-$n.env"
for arg in "$@"; do
  case "$arg" in
    trainer.default_local_dir=*) mkdir -p "${arg#*=}/global_step_$STEPS" ;;
    --target_dir) target=NEXT ;;
    *) if [ "${target:-}" = NEXT ]; then mkdir -p "$arg"; echo '{}' > "$arg/config.json"; target=; fi ;;
  esac
done
exit 0
"""


@pytest.fixture()
def stubbed(tmp_path):
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq                                                    # noqa: PLC0415
    work, sdpo, model, record, binaries = (tmp_path / n for n in ("work", "SDPO", "model", "record", "bin"))
    for directory in (work, sdpo, model, record, binaries):
        directory.mkdir()
    subprocess.run(["git", "init", "-q", str(sdpo)], check=True, capture_output=True)
    (model / "config.json").write_text("{}")
    train, val = tmp_path / "train.parquet", tmp_path / "heldout.parquet"
    pq.write_table(pa.Table.from_pylist([{"prompt": "p%d" % i, "response": "Answer: %d" % i} for i in range(1280)]), train)
    pq.write_table(pa.Table.from_pylist([{"prompt": "p", "response": "Answer: 1"}] * 8), val)
    (binaries / "python").write_text(STUB)
    (binaries / "python").chmod(0o755)
    env = {**shell(), "PATH": "%s:%s" % (binaries, os.environ["PATH"]), "RECORD": str(record), "REAL_PYTHON": sys.executable,
           "SDPO_DIR": str(sdpo), "MODEL_DIR": str(model), "NAME": "sft-seed0-a1", "TRAIN_FILE": str(train), "VAL_FILE": str(val),
           "WORK": str(work), "DRY_RUN": "0", "STEPS": "40", "SAVE_FREQ": "40", "SEED": "0", "NGPU": "8"}
    return {"env": env, "record": record, "out": work / "runs" / "sft-seed0-a1", "train": train, "tmp": tmp_path}


def grpo_summary_keys() -> list:
    text = (KIT / "run_grpo.sh").read_text()
    block = text.split('cat > "$OUT/train-summary.json" <<JSON', 1)[1].split("\nJSON\n", 1)[0]
    return re.findall(r'^ "([a-z_]+)":', block, re.M)


def test_the_launcher_trains_merges_and_writes_run_grpos_summary_plus_the_route(stubbed):
    done = subprocess.run(["bash", str(SCRIPT)], env=stubbed["env"], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr[-2000:]
    out = stubbed["out"]
    calls = sorted(stubbed["record"].glob("call-*.argv"))
    assert len(calls) == 2, "the trainer and the merger, once each"
    trainer = calls[0].read_text().strip().split("\n")
    assert trainer == (out / "env" / "argv.txt").read_text().strip().split("\n")[1:], "the recorded argv is the argv that ran"
    assert calls[1].read_text().strip().split("\n") == ["-m", "verl.model_merger", "merge", "--backend", "fsdp", "--local_dir",
                                                        "%s/train/global_step_40" % out, "--target_dir", "%s/hf-step40" % out]
    summary = json.loads((out / "train-summary.json").read_text())
    assert sorted(summary) == sorted(grpo_summary_keys() + ["route"])
    assert summary["route"] == "sft" and summary["returncode"] == 0 and summary["merged"] == 1 and summary["steps"] == 40
    child = dict(line.split("=", 1) for line in (stubbed["record"] / "call-0.env").read_text().split("\n") if "=" in line)
    assert child["PYTHONPATH"].startswith(stubbed["env"]["SDPO_DIR"]) and child["WANDB_MODE"] == "disabled"
    assert child["VERL_FILE_LOGGER_PATH"] == "%s/metrics.jsonl" % out
    again = subprocess.run(["bash", str(SCRIPT)], env=stubbed["env"], capture_output=True, text=True)
    assert again.returncode == 2 and "refusing to overwrite" in again.stderr


def test_the_launcher_refuses_a_file_that_is_not_one_pass_of_40_steps(stubbed):
    import pyarrow as pa                                                            # noqa: PLC0415
    import pyarrow.parquet as pq                                                    # noqa: PLC0415
    short = stubbed["tmp"] / "short.parquet"
    pq.write_table(pa.Table.from_pylist([{"prompt": "p", "response": "r"}] * 640), short)
    done = subprocess.run(["bash", str(SCRIPT)], env={**stubbed["env"], "TRAIN_FILE": str(short)}, capture_output=True, text=True)
    assert done.returncode == 2 and "640 rows" in done.stderr
    assert not list(stubbed["record"].glob("call-*.argv")), "the trainer never started"


# ------------------------------------------------------------------------------------ the report
def _panel(root: Path, name: str, scores: dict, machine: str = "m1"):
    directory = root / "forgetting" / ("%s-a1" % name)
    directory.mkdir(parents=True)
    panels = {p: {"correct": c, "n": 100, "median_output_chars": 400, "per_member": {}} for p, c in scores.items()}
    (directory / "forgetting.json").write_text(json.dumps({"panels": panels, "total_correct": sum(scores.values()), "machine": {"id": machine}}))


def _bed(root: Path, name: str, correct: int, n: int = 100, machine: str = "m1"):
    directory = root / "eval" / ("%s-a1" % name)
    directory.mkdir(parents=True)
    (directory / "bed-score.json").write_text(json.dumps({"correct": correct, "n": n, "machine": {"id": machine}, "output_tokens_mean": 80}))


def k2b_tree(root: Path, rl_share: float, sft_share: float) -> Path:
    """Spider: original 72, control 70; every trained model at 20, so 50 points to the control. After 300 repair steps
    a route's models stand at 20 + share x 50. The panel repeats the same shape on its three panels."""
    _bed(root, "q17-spider", 72)
    _bed(root, "q17-control300-spider", 70)
    _bed(root, "q17-gsm8k", 200, n=300)
    _panel(root, "q17-original", {"math": 60, "knowledge": 60, "ifeval": 60})
    _panel(root, "q17-control300", {"math": 60, "knowledge": 60, "ifeval": 60})
    for route, share in (("rl", rl_share), ("sft", sft_share)):
        for seed in range(10):
            sub = "%s-seed%d" % (route, seed)
            _bed(root, "%s-spider" % sub, 20)
            _bed(root, "%s-gsm8k" % sub, 230 if route == "rl" else 245, n=300)
            _panel(root, "%s-before" % sub, {"math": 40, "knowledge": 40, "ifeval": 40})
            for step in (50, 100, 200, 300):
                back = round(share * 50 * (0.5 if step == 50 else 1))
                _bed(root, "%s-after%d-spider" % (sub, step), 20 + back)
                _panel(root, "%s-after%d" % (sub, step), {"math": 40 + round(back * 20 / 50), "knowledge": 40 + round(back * 20 / 50), "ifeval": 40 + round(back * 20 / 50)})
                (root / "forgetting" / ("%s-drift-after%d.json" % (sub, step))).write_text(json.dumps({"relative_distance_all": 0.002}))
            (root / "forgetting" / ("%s-drift-damaged.json" % sub)).write_text(json.dumps({"relative_distance_all": 0.01 if route == "rl" else 0.03}))
    return root


def test_rl_96_and_sft_54_against_the_control_holds(tmp_path):
    root = k2b_tree(tmp_path / "k2b", 0.96, 0.54)
    assert k2b_report.main(["--root", str(root), "--out", str(tmp_path / "out")]) == 0
    report = json.loads((tmp_path / "out" / "k2b-report.json").read_text())
    assert report["subjects_reported"] == 20 and report["comparable"] == 1
    spider = report["verdicts"]["spider"]
    assert spider["verdict"] == report["verdict"] == "HOLDS" and spider["outcome"] == "RL hides, SFT overwrites"
    assert spider["rl"]["mean"] == 0.96 and spider["sft"]["mean"] == 0.54 and spider["gap"] == 0.42 and spider["rl"]["n"] == spider["sft"]["n"] == 10
    assert report["verdicts"]["panel"]["verdict"] == "HOLDS"
    rl = report["routes"]["rl"]
    assert rl["spider_damage"]["mean"] == 52 and rl["gsm8k_gain"]["mean"] == 30 and report["routes"]["sft"]["gsm8k_gain"]["mean"] == 45
    assert rl["spider_share_vs_control"]["after50"]["mean"] == 0.48 and rl["spider_share_vs_original"]["after300"]["mean"] == round(48 / 52, 4)
    assert rl["relative_distance"]["damaged"]["mean"] == 0.01 and report["learned_comparably"] is True
    # kit/repair/report.py joined every subject to q17's original and control, through the explicit aliases
    recovery = json.loads((tmp_path / "out" / "recovery" / "recovery-report.json").read_text())
    assert recovery["subjects_reported"] == 20 and set(recovery["alias"].values()) == {"q17"}
    assert recovery["subjects"]["rl-seed0"]["original_and_control_from"] == "q17"
    assert "**Verdict of record (Spider held-out): HOLDS**" in (tmp_path / "out" / "k2b-report.md").read_text()


def test_rl_70_and_sft_60_does_not_hold(tmp_path):
    root = k2b_tree(tmp_path / "k2b", 0.70, 0.60)
    assert k2b_report.main(["--root", str(root), "--out", str(tmp_path / "out")]) == 0
    spider = json.loads((tmp_path / "out" / "k2b-report.json").read_text())["verdicts"]["spider"]
    assert spider["verdict"] == "DOES_NOT_HOLD" and spider["outcome"] == "neither hides"


def test_the_rule_needs_all_three_conditions():
    assert k2b_report.decide([0.96] * 10, [0.54] * 10)["verdict"] == "HOLDS"
    assert k2b_report.decide([0.85] * 10, [0.75] * 10)["outcome"].startswith("RL hides, SFT overwrites, but by 10 points")
    assert k2b_report.decide([0.9] * 10, [0.9] * 10)["outcome"] == "both hide"
    assert k2b_report.decide([0.5] * 10, [0.9] * 10)["outcome"] == "SFT hides, RL does not"
    assert k2b_report.decide([0.9] * 10, [None] * 10)["verdict"] == "NOT_DECIDABLE"


def test_the_report_never_overwrites_and_refuses_a_tree_without_the_original(tmp_path):
    root = k2b_tree(tmp_path / "k2b", 0.96, 0.54)
    (tmp_path / "out").mkdir()
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        k2b_report.main(["--root", str(root), "--out", str(tmp_path / "out")])
    for path in (root / "eval" / "q17-spider-a1").iterdir():
        path.unlink()
    (root / "eval" / "q17-spider-a1").rmdir()
    with pytest.raises(SystemExit, match="q17"):
        k2b_report.main(["--root", str(root), "--out", str(tmp_path / "out2")])
    assert not (tmp_path / "out2").exists()


def test_the_sft_entry_turns_pinned_memory_off_before_it_imports_the_trainer():
    """The pinned trainer hard-codes pin_memory=True; under torchdata 0.11 + torch 2.9 its pin-memory thread crashes
    (the K2b smoke, 29 September). kit/sft_entry.py patches StatefulDataLoader first, then imports the trainer."""
    import ast
    tree = ast.parse((KIT / "sft_entry.py").read_text())
    order = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom, ast.Assign))]
    patch = next(i for i, n in enumerate(order) if isinstance(n, ast.Assign) and "StatefulDataLoader.__init__" in ast.unparse(n.targets[0]))
    trainer = next(i for i, n in enumerate(order) if isinstance(n, ast.ImportFrom) and (n.module or "").startswith("verl"))
    assert patch < trainer
    src = (KIT / "sft_entry.py").read_text()
    assert 'kwargs["pin_memory"] = False' in src and 'pop("pin_memory_device"' in src

