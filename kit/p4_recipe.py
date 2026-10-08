#!/usr/bin/env python3
"""Package 4's recipe-check: "the pilot's recipe otherwise unchanged", verified against the pilot's OWN records.

    python p4_recipe.py check --work WORK --campaign k8b-p4-<recipe>-<chain>.yaml --selection selection.json
                              --out WORK/k8b4/recipe-check --attempt N

Implements docs/phase2/plan-v3-package4-amendment3-20261004.md B6 (with registration 3 and supplement S2, S6). For EVERY
training row of the campaign (qualification, prevention and bridges) it:
  - freezes the comparison BASELINE: the pilot run of the corresponding recipe, task order and lineage at its attempt
    of record (kit/pilot_report.py: the highest attempt whose run-summary.json says merged 1). Prevention lineage rX
    rows and the qualification (r1): <recipe>-<chain>-rX; bridges: the r1 run of configuration a = g8, b = g8,
    c = g32, s = sema (b is the g8 baseline with the minibatch changed);
  - derives every launcher knob (NGPU, TP, OFFLOAD; LORA for GRPO; FEEDBACK, SOFT, TEMP for SDPO) from that baseline's
    recorded command and cross-checks it against its run-summary.json; the training rows then take the knobs from the
    frozen baseline.json (kit/p4_run.py), never from the shell;
  - runs the row's launcher with DRY_RUN=1 on exactly the row's env and those knobs, and compares the proposed command
    with the pilot's recorded env/argv.txt under the EXACT substitution table (round-3 ruling F3, `substitution_table`):
    a table, generated from the row's registered role, of FULL KEY PATHS with the exact old and new value of each --
    trainer.experiment_name and the enumerated output paths (trainer.default_local_dir, vars.log_dir, vars.ckpt_dir,
    trainer.rollout_data_dir, trainer.validation_data_dir: WORK/runs/<name>/...), actor_rollout_ref.model.path (the
    selected incoming checkpoint), the three seed keys, data.train_files, data.val_files and vars.task (the registered
    task data); b: actor_rollout_ref.actor.ppo_mini_batch_size 8 -> 32; interventions: data.max_response_length (absent
    -> 2048) and custom_reward_function.path (the authors' reward -> kit/beds/authors_gate.py); qualification:
    trainer.save_freq, total_training_steps 40 -> 2 and test_freq 20 -> 2. Any key outside the table that differs, any
    table key whose old or new value is not exactly the expected one, and any key present on one side only, fails. No
    substring is replaced anywhere;
  - resolves both commands with the trainer's own `python -m verl.trainer.main_ppo --cfg job --resolve` and compares
    them over the UNION of their keys with the same table, which for resolved configurations also names every key the
    trainer DERIVES by interpolation from a substituted one (MODEL_DERIVED, MB_DERIVED, RESPONSE_DERIVED, enumerated
    from the pinned trainer's configuration sources); the pilot's is labelled RECONSTRUCTED;
  - reports every command and resolved-configuration difference between c and s (S2: a report, not an equality test)
    and verifies the shared settings S2 lists over the UNION of the registered shared set (SHARED_ARGV,
    SHARED_RESOLVED): a key missing on either side fails, and so does a missing resolved configuration;
  - establishes the trainer's SOURCE IDENTITY (round-3 ruling F4): the pilot baseline run's env/sdpo-commit.txt is the
    checkout's commit now AND the pilot's env/sdpo-dirty.txt and the checkout's `git status --short` are both empty;
    anything else fails (there is no "not verifiable ... ok"). The sha256 of the trainer's entry point and configuration
    sources is recorded too;
  - establishes the TASK DATA identity: env/data-sha256.txt of a pilot run of record that trained the row's task (the
    baseline, else the lineage's stage-1 run) must equal the sha256 of SDPO_DIR/<task>/train.parquet and test.parquet
    now, compared by file name; missing, it fails closed. The incoming checkpoint is identified by the selection's
    recorded run-summary sha256 and its export file list;
  - FREEZES its comparison inputs at the FIRST check whatever its verdict (round-3 ruling F2): OUT/frozen.json holds the
    mapping, the pilot command and environment files with their sha256 (`inputs`, re-hashed by kit/p4_frozen.py before
    every dependent action), the proposed commands, the trainer and data identities and the verdict. A later check must
    find the same comparison inputs (else exit 2, "inputs changed after they were frozen"); a frozen pass must be
    reproduced; a frozen failure is final, unless every problem was a recorded TOOL ERROR (a dry run or a resolution
    that could not run) and the inputs are byte-identical, when the check is run again.

ROUND 4 (the manager's rulings G2 and G3, and the data-identity ruling):
  - the incoming checkpoint's identity is recorded by CONTENT in the frozen document: `comparison.incoming` holds the
    sha256 of every file of each incoming export (`export_identity`, streamed); kit/p4_run.py re-hashes it before every
    training and qualification launch;
  - frozen.json holds, per row, the sha256 of its reconstructed pilot configuration (`reconstructed`); `verify_run`
    (the reporter's comparison and the retry authorization of kit/p4_budget.py) verifies the baseline's seal, its link
    to frozen.json, the reconstruction's bytes and the pilot command's bytes against the frozen hashes before it
    compares anything; a mismatch is a refusal (None: cannot be verified);
  - task data by regeneration (`regenerated_data`): data/preprocess.py and every source JSON of the task must be
    TRACKED files of the pinned commit with no modification, and their sha256 are archived with the regenerated and
    in-place hashes. It establishes that the files in place equal what the pinned preprocessing writes from the
    pinned source; it does not establish what bytes the pilot read.

WRITTEN (all archived by kit/collect.py): OUT/a<N>/ with pilot/<key>-a<M>.argv.txt, proposed/<row>.argv.txt, both
resolved configurations, record.json; OUT/check-a<N>.json (`ok` 0/1 for the runner's bar); OUT/frozen.json at the first
check; and, when a check passes, OUT/baseline.json (sealed, bound to frozen.json), from which kit/p4_run.py takes the
knobs and kit/p4_report.py and kit/p4_budget.py the stored comparison spec of each row (`verify_run`).
Exit 0 when every row verifies, 1 otherwise, 2 for a refusal. PyYAML is needed only to read resolved configurations.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCHEMA = "kit-p4-recipe-check.v1"
GRPO, SDPO = "run_grpo_toolalpaca.sh", "run_sdpo_toolalpaca.sh"
CONFIG_RECIPE = {"a": "g8", "b": "g8", "c": "g32", "s": "sema"}
RECIPE_CONFIG = {"g8": "a", "sema": "s"}
SEED_KEYS = ("data.seed", "actor_rollout_ref.actor.data_loader_seed", "actor_rollout_ref.actor.fsdp_config.seed")
RUN_PATH_KEYS = ("trainer.default_local_dir", "vars.log_dir", "vars.ckpt_dir", "trainer.rollout_data_dir",
                 "trainer.validation_data_dir")
DATA_KEYS = ("data.train_files", "data.val_files", "vars.task")
STEP_KEYS = {"trainer.save_freq": ("40", "2"), "trainer.total_training_steps": ("40", "2"), "trainer.test_freq": ("20", "2")}
MB_KEY, RESPONSE_KEY, REWARD_KEY = "actor_rollout_ref.actor.ppo_mini_batch_size", "data.max_response_length", "custom_reward_function.path"
MODEL_KEY, NAME_KEY = "actor_rollout_ref.model.path", "trainer.experiment_name"
AUTHORS_REWARD = "verl/utils/reward_score/feedback/__init__.py"
#: supplement S2: "at equal learning rate, minibatch, warm-up, batch, attempts, steps, incoming weights, task and decoding"
SHARED_ARGV = ("actor_rollout_ref.actor.optim.lr", MB_KEY, "actor_rollout_ref.actor.optim.lr_warmup_steps", "data.train_batch_size",
               "actor_rollout_ref.rollout.n", "trainer.total_training_steps", MODEL_KEY, "data.train_files", "data.val_files",
               RESPONSE_KEY, "actor_rollout_ref.rollout.temperature", "actor_rollout_ref.rollout.val_kwargs.n")
DECODING_LEAVES = ("temperature", "top_p", "top_k", "response_length", "prompt_length", "do_sample", "max_response_length",
                   "max_prompt_length")
LAUNCHER_VARIABLES = ("SDPO_DIR", "MODEL_DIR", "NAME", "STEPS", "TEST_FREQ", "SEED", "NGPU", "TP", "OFFLOAD", "WORK", "DRY_RUN",
                      "LORA", "DATASET", "LR", "MINI_BATCH", "KEEP_TRAINER_CKPT", "MAX_RESPONSE", "FINISH_GATE", "FEEDBACK",
                      "SOFT", "TEMP", "TEACHER_RATE", "KIT_FINISH_GATE", "PYTHONPATH")
CONTENT_NOT_VERIFIABLE = "not verifiable from the pilot's records"


def _load(name: str):
    spec = importlib.util.spec_from_file_location("kit_p4_recipe_%s" % name, HERE / ("%s.py" % name))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256_bytes(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def content_sha256(content) -> str:
    return sha256_bytes(json.dumps(content, sort_keys=True, separators=(",", ":")).encode())


# ------------------------------------------------------------------------------------------------ commands
def parse_argv(lines: list) -> tuple:
    """(head tokens, {key: value}, duplicated keys) of a launcher argv, one token per line."""
    head, keys, dup = [], {}, []
    for line in lines:
        line = line.rstrip("\n")
        if not line:
            continue
        if "=" in line and not line.startswith("-"):
            key, _sep, value = line.partition("=")
            if key in keys:
                dup.append(key)
            keys[key] = value
        else:
            head.append(line)
    return head, keys, dup


def read_lines(path: Path):
    try:
        return Path(path).read_text(encoding="utf-8").splitlines()
    except OSError:
        return None


def _run_root(value: str, name: str):
    """The part of a run-derived path before /runs/<name>, and the suffix after it; None when it is not one."""
    match = re.match(r"^(.*)/runs/%s(/.*)?$" % re.escape(name), value or "")
    return (match.group(1), match.group(2) or "") if match else None


LAUNCHER_FOLDER = {GRPO: "tool-grpo", SDPO: "tool-sdpo"}
ABSENT = "<absent>"
#: the run name's enumerated output paths, under WORK/runs/<name>/ ({f}: the launcher's trainer folder)
RUN_PATHS = {"trainer.default_local_dir": "{f}", "vars.log_dir": "{f}/logs", "vars.ckpt_dir": "{f}",
             "trainer.rollout_data_dir": "rollouts", "trainer.validation_data_dir": "validation"}
#: the keys the trainer DERIVES by interpolation from a key the launchers override (round-3 ruling F3), enumerated from the
#: pinned trainer's configuration sources (verl/trainer/config: rollout/rollout.yaml, critic/critic.yaml,
#: reward_model/reward_model.yaml as composed by _generated_ppo_trainer.yaml, user.yaml and baseline_grpo.yaml / sdpo.yaml):
#:   actor_rollout_ref.model.path         -> rollout.prometheus.served_model_name, critic.model.tokenizer_path,
#:                                           reward_model.model.input_tokenizer
#:   actor.ppo_mini_batch_size            -> critic.ppo_mini_batch_size
#:   data.max_response_length            -> actor_rollout_ref.rollout.response_length
#: (trainer.experiment_name, default_local_dir, vars.ckpt_dir and vars.task are overridden by the launchers themselves, so
#: no interpolation reaches them; the stand-in trainer of scripts/sim derives these keys the same way)
MODEL_DERIVED = ("actor_rollout_ref.rollout.prometheus.served_model_name", "critic.model.tokenizer_path", "reward_model.model.input_tokenizer")
MB_DERIVED = ("critic.ppo_mini_batch_size",)
RESPONSE_DERIVED = ("actor_rollout_ref.rollout.response_length",)
#: the trainer's own seeds when a command sets none (the composed configuration: data.seed null, the actor's 42 and 42)
SEED_DEFAULTS = {"data.seed": None, "actor_rollout_ref.actor.data_loader_seed": 42, "actor_rollout_ref.actor.fsdp_config.seed": 42}
REGISTERED_LONG_RESPONSE, GATE_RESPONSE = 8192, 2048
SPEC_FIELDS = ("work_root", "sdpo", "kit", "folder", "baseline_model")


def spec_for(*, baseline_name: str, proposed_name: str, incoming: str, seed: str, dataset: str, b: bool = False,
             intervention: bool = False, qualification: bool = False, baseline_model=None, baseline_seed=None,
             baseline_dataset=None, work_root=None, sdpo=None, kit=None, folder=None) -> dict:
    """Everything the exact substitution table of one comparison needs: the two run names, the old and new incoming
    checkpoint, seed and task data, the WORK root and checkout the paths are under, the kit (the gate's reward path), the
    launcher's trainer folder, and the registered role (b, intervention, qualification)."""
    return {"baseline_name": baseline_name, "proposed_name": proposed_name, "incoming": incoming, "seed": str(seed),
            "dataset": dataset, "b": b, "intervention": intervention, "qualification": qualification,
            "baseline_model": baseline_model, "baseline_seed": None if baseline_seed is None else str(baseline_seed),
            "baseline_dataset": baseline_dataset if baseline_dataset is not None else dataset, "work_root": work_root,
            "sdpo": sdpo, "kit": kit, "folder": folder}


def substitution_table(spec: dict, resolved: bool = False) -> dict:
    """{FULL KEY PATH: (exact old value, exact new value, why)} of every permitted difference (amendment 3 B6; round-3
    ruling F3), for the commands (strings, ABSENT for a key a command does not pass) or for the resolved configurations
    (typed as the YAML gives them). A key whose old and new values are equal is not in the table: it must not differ."""
    missing = [f for f in SPEC_FIELDS if not spec.get(f)]
    if missing:
        raise ValueError("the substitution table needs %s" % ", ".join(missing))
    table = {}

    def add(key, old, new, why):
        if not (type(old) is type(new) and old == new):
            table[key] = (old, new, why)

    root, bn, pn, folder = spec["work_root"], spec["baseline_name"], spec["proposed_name"], spec["folder"]
    add(NAME_KEY, bn, pn, "run name")
    for key, tail in RUN_PATHS.items():
        tail = tail.format(f=folder)
        add(key, "%s/runs/%s/%s" % (root, bn, tail), "%s/runs/%s/%s" % (root, pn, tail), "output path of the run name")
    for key in (MODEL_KEY,) + (MODEL_DERIVED if resolved else ()):
        add(key, spec["baseline_model"], spec["incoming"], "the selected incoming checkpoint")
    for key in SEED_KEYS:
        old, new = spec["baseline_seed"], spec["seed"]
        if resolved:
            add(key, SEED_DEFAULTS[key] if old is None else int(old), int(new), "registered seed override")
        else:
            add(key, ABSENT if old is None else old, new, "registered seed override")
    sdpo, old_set, new_set = spec["sdpo"], spec["baseline_dataset"], spec["dataset"]
    for key, split in (("data.train_files", "train"), ("data.val_files", "test")):
        old, new = "%s/%s/%s.parquet" % (sdpo, old_set, split), "%s/%s/%s.parquet" % (sdpo, new_set, split)
        if resolved:
            add(key, json.dumps([old], sort_keys=True, default=str), json.dumps([new], sort_keys=True, default=str), "registered task data path")
        else:
            add(key, "[%s]" % old, "[%s]" % new, "registered task data path")
    add("vars.task", old_set, new_set, "registered task data path")
    if spec["b"]:
        for key in (MB_KEY,) + (MB_DERIVED if resolved else ()):
            add(key, 8 if resolved else "8", 32 if resolved else "32", "configuration b: minibatch 8 to 32")
    if spec["intervention"]:
        if resolved:
            for key in (RESPONSE_KEY,) + RESPONSE_DERIVED:
                add(key, REGISTERED_LONG_RESPONSE, GATE_RESPONSE, "intervention: response limit 2048")
        else:
            add(RESPONSE_KEY, ABSENT, str(GATE_RESPONSE), "intervention: response limit 2048")
        add(REWARD_KEY, "%s/%s" % (sdpo, AUTHORS_REWARD), "%s/beds/authors_gate.py" % spec["kit"], "intervention: the wrapper reward path")
    if spec["qualification"]:
        for key, (old, new) in STEP_KEYS.items():
            add(key, int(old) if resolved else old, int(new) if resolved else new, "qualification: two steps")
    return table


def _same(a, b) -> bool:
    return type(a) is type(b) and a == b


def _compare(base: dict, prop: dict, table: dict) -> tuple:
    diffs, bad = [], []
    for key in sorted(set(base) | set(prop) | set(table)):
        b, p = base.get(key, ABSENT), prop.get(key, ABSENT)
        if key in table:
            old, new, why = table[key]
            entry = {"key": key, "baseline": b, "proposed": p, "permitted": why if (_same(b, old) and _same(p, new)) else None,
                     "expected": [old, new]}
            diffs.append(entry)
            if entry["permitted"] is None:
                bad.append(entry)
        elif not _same(b, p):
            entry = {"key": key, "baseline": b, "proposed": p, "permitted": None}
            diffs.append(entry)
            bad.append(entry)
    return diffs, bad


def compare_argv(baseline_lines: list, proposed_lines: list, spec: dict) -> dict:
    """{ok, differences, not_permitted}: key by key over the UNION of both commands' keys, each permitted difference
    exactly the table's old and new value at its full key; the head (`python -m verl.trainer.main_ppo --config-name X`)
    must be equal and no key may be passed twice."""
    bhead, bkeys, bdup = parse_argv(baseline_lines)
    phead, pkeys, pdup = parse_argv(proposed_lines)
    bad = []
    if bhead != phead:
        bad.append({"key": "(command head)", "baseline": " ".join(bhead), "proposed": " ".join(phead), "permitted": None})
    for key in bdup + pdup:
        bad.append({"key": key, "baseline": "passed twice" if key in bdup else None, "proposed": "passed twice" if key in pdup else None,
                    "permitted": None})
    try:
        table = substitution_table(spec)
    except ValueError as error:
        return {"ok": False, "differences": [], "not_permitted": bad + [{"key": "(table)", "baseline": None, "proposed": None,
                                                                          "permitted": None, "why": str(error)}]}
    diffs, wrong = _compare(bkeys, pkeys, table)
    bad += wrong
    return {"ok": not bad, "differences": diffs, "not_permitted": bad}


# ------------------------------------------------------------------------------------------------ resolved configurations
def flatten(value, prefix="") -> dict:
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            out.update(flatten(item, "%s.%s" % (prefix, key) if prefix else str(key)))
        return out
    if isinstance(value, list):
        return {prefix: json.dumps(value, sort_keys=True, default=str)}
    return {prefix: value}


def read_resolved(path: Path):
    """The flattened resolved configuration in a YAML file, or None when absent, empty or unreadable."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        import yaml                                                     # noqa: PLC0415
    except ImportError:
        return None
    try:
        doc = yaml.safe_load(text)
    except yaml.YAMLError:
        return None
    return flatten(doc) if isinstance(doc, dict) and doc else None


def compare_resolved(baseline: dict, proposed: dict, spec: dict, baseline_argv=None, proposed_argv=None) -> dict:
    """Key by key over the UNION of both resolved configurations: every key equal (type and value) except the table's
    full key paths, each exactly its old and new value (round-3 ruling F3). No substring is replaced anywhere."""
    try:
        table = substitution_table(spec, resolved=True)
    except ValueError as error:
        return {"ok": False, "differences": [], "not_permitted": [{"key": "(table)", "baseline": None, "proposed": None,
                                                                   "permitted": None, "why": str(error)}]}
    diffs, bad = _compare(baseline, proposed, table)
    return {"ok": not bad, "differences": diffs, "not_permitted": bad}


def difference_report(first: dict, second: dict) -> list:
    """Every key whose value differs between two flattened configurations or two argv dicts (a report, no test)."""
    return [{"key": k, "c": first.get(k, "<absent>"), "s": second.get(k, "<absent>")}
            for k in sorted(set(first) | set(second)) if first.get(k, "<absent>") != second.get(k, "<absent>")]


#: supplement S2's shared settings of c and s, as full key paths of the RESOLVED configurations: learning rate, minibatch,
#: warm-up, batch, attempts, steps, incoming weights, task, and decoding (training and validation sampling, lengths)
SHARED_RESOLVED = ("actor_rollout_ref.actor.optim.lr", MB_KEY, "actor_rollout_ref.actor.optim.lr_warmup_steps", "data.train_batch_size",
                   "actor_rollout_ref.rollout.n", "trainer.total_training_steps", MODEL_KEY, "data.train_files", "data.val_files",
                   RESPONSE_KEY, "data.max_prompt_length", "actor_rollout_ref.rollout.response_length",
                   "actor_rollout_ref.rollout.prompt_length", "actor_rollout_ref.rollout.temperature", "actor_rollout_ref.rollout.top_p",
                   "actor_rollout_ref.rollout.top_k", "actor_rollout_ref.rollout.do_sample", "actor_rollout_ref.rollout.val_kwargs.n",
                   "actor_rollout_ref.rollout.val_kwargs.temperature", "actor_rollout_ref.rollout.val_kwargs.top_p",
                   "actor_rollout_ref.rollout.val_kwargs.top_k", "actor_rollout_ref.rollout.val_kwargs.do_sample")


def shared_settings(c_argv: list, s_argv: list, c_resolved, s_resolved) -> dict:
    """Supplement S2: c and s at equal learning rate, minibatch, warm-up, batch, attempts, steps, incoming weights, task
    and decoding. Over the UNION of the registered shared set (round-3 ruling F3): a command key passed by one and not
    the other fails; every registered resolved key must be PRESENT in both resolved configurations and equal (type and
    value); a resolved configuration that is missing fails the check (it cannot be shown equal)."""
    _h, ck, _d = parse_argv(c_argv)
    _h, sk, _d = parse_argv(s_argv)
    checked, unequal = [], []
    for key in SHARED_ARGV:
        checked.append("argv %s" % key)
        if ck.get(key, ABSENT) != sk.get(key, ABSENT):
            unequal.append({"where": "argv", "key": key, "c": ck.get(key, ABSENT), "s": sk.get(key, ABSENT)})
    if c_resolved is None or s_resolved is None:
        unequal.append({"where": "resolved", "key": "(the resolved configurations)", "c": "present" if c_resolved else "missing",
                        "s": "present" if s_resolved else "missing"})
    else:
        for key in SHARED_RESOLVED:
            checked.append("resolved %s" % key)
            c, s_ = c_resolved.get(key, ABSENT), s_resolved.get(key, ABSENT)
            if c == ABSENT or s_ == ABSENT or not _same(c, s_):
                unequal.append({"where": "resolved", "key": key, "c": c, "s": s_})
    return {"ok": not unequal, "checked": checked, "unequal": unequal}


# ------------------------------------------------------------------------------------------------ the rows
def role(row: dict) -> dict:
    """What the campaign's training row is: config, lineage, launcher, intervention, qualification."""
    env, text = row.get("env") or {}, " ".join(row.get("command") or [])
    launcher = GRPO if GRPO in text else SDPO if SDPO in text else None
    if launcher == SDPO:
        config = "s"
    elif launcher == GRPO:
        config = {("1e-6", "8"): "a", ("1e-6", "32"): "b", ("1e-5", "32"): "c"}.get((env.get("LR"), env.get("MINI_BATCH")))
    else:
        config = None
    return {"row": row["id"], "launcher": launcher, "config": config, "lineage": "r2" if row["id"].startswith("p4-r2-") else "r1",
            "intervention": env.get("FINISH_GATE") == "1", "qualification": env.get("P4_KIND") == "qualification",
            "seed": env.get("SEED"), "dataset": env.get("DATASET"), "block": env.get("P4_BLOCK")}


def training_rows(campaign: dict) -> list:
    return [row for row in campaign["rows"] if (row.get("env") or {}).get("P4_KIND") == "training"
            or ((row.get("env") or {}).get("P4_KIND") == "qualification" and str((row.get("env") or {}).get("P4_GPUS")) not in ("", "0", "1"))]


def baseline_key(r: dict, recipe: str, chain: str) -> str:
    """The pilot run a row is compared with (amendment 3 B6)."""
    base_recipe = recipe if r["block"] != "bridges" else CONFIG_RECIPE[r["config"]]
    return "%s-%s-%s" % (base_recipe, chain, r["lineage"] if r["block"] != "bridges" else "r1")


def knobs_from(argv_keys: dict, launcher: str) -> dict:
    """The launcher knobs the pilot's recorded command implies (each launcher writes them into its argv)."""
    knobs = {"NGPU": argv_keys.get("trainer.n_gpus_per_node"), "TP": argv_keys.get("actor_rollout_ref.rollout.tensor_model_parallel_size"),
             "OFFLOAD": "1" if argv_keys.get("actor_rollout_ref.actor.fsdp_config.param_offload") == "True" else "0"}
    if launcher == GRPO:
        knobs["LORA"] = "1" if "actor_rollout_ref.model.lora_rank" in argv_keys else "0"
    else:
        knobs["FEEDBACK"] = "1" if argv_keys.get("actor_rollout_ref.actor.self_distillation.include_environment_feedback") == "True" else "0"
        knobs["SOFT"] = "1" if (argv_keys.get(REWARD_KEY) or "").endswith("beds/tooluse_soft.py") else "0"
        knobs["TEMP"] = argv_keys.get("actor_rollout_ref.rollout.temperature") or ""
    return knobs


def knob_problems(knobs: dict, summary, launcher: str) -> list:
    if not isinstance(summary, dict):
        return ["the pilot run-summary.json is missing: the knobs cannot be cross-checked"]
    wrong = []
    if knobs.get("NGPU") is None or str(summary.get("n_gpus")) != str(knobs["NGPU"]):
        wrong.append("n_gpus %r in the pilot summary, %r in its command" % (summary.get("n_gpus"), knobs.get("NGPU")))
    if not knobs.get("TP"):
        wrong.append("the pilot command records no tensor_model_parallel_size")
    checks = (("lora", "LORA"),) if launcher == GRPO else (("feedback", "FEEDBACK"), ("soft", "SOFT"))
    for field, knob in checks:
        if field in summary and str(summary[field]) != knobs[knob]:
            wrong.append("%s %r in the pilot summary, %s=%s from its command" % (field, summary[field], knob, knobs[knob]))
    if launcher == SDPO and "temperature" in summary and summary["temperature"] != (knobs["TEMP"] or "default"):
        wrong.append("temperature %r in the pilot summary, TEMP=%r from its command" % (summary["temperature"], knobs["TEMP"]))
    return wrong


def clean_env() -> dict:
    return {k: v for k, v in os.environ.items() if k not in LAUNCHER_VARIABLES}


def dry_run(kit: Path, launcher: str, env: dict) -> tuple:
    done = subprocess.run(["bash", str(kit / launcher)], env={**clean_env(), **env, "DRY_RUN": "1"}, capture_output=True, text=True)
    return done.returncode, done.stdout.splitlines(), done.stderr


def resolve(argv: list, sdpo: str, env: dict) -> tuple:
    """(yaml text, None) from the trainer's own `--cfg job --resolve` on a recorded or proposed command, or (None, why)."""
    head, _keys, _dup = parse_argv(argv)
    if head[:3] != ["python", "-m", "verl.trainer.main_ppo"] or len(head) < 5:
        return None, "not a trainer command: %s" % " ".join(head)
    tokens = [line for line in argv if line]
    command = [shutil.which("python") or sys.executable] + tokens[1:5] + ["--cfg", "job", "--resolve"] + tokens[5:]
    inherited = os.environ.get("PYTHONPATH")                 # as the launchers do: $SDPO_DIR first, then what the shell had
    full = {**clean_env(), "PYTHONPATH": sdpo + (":" + inherited if inherited else ""), "VLLM_USE_V1": "1", "WANDB_MODE": "disabled", **env}
    try:
        done = subprocess.run(command, cwd=sdpo, env=full, capture_output=True, text=True, timeout=600)
    except (OSError, subprocess.SubprocessError) as error:
        return None, "the trainer could not resolve the configuration: %s" % error
    if done.returncode != 0 or not done.stdout.strip():
        return None, "the trainer's --cfg job --resolve exited %d: %s" % (done.returncode, (done.stderr or "no output")[-400:])
    return done.stdout, None


def git(sdpo: str, *args) -> tuple:
    done = subprocess.run(["git", "-C", sdpo] + list(args), capture_output=True, text=True)
    return done.returncode, done.stdout


def tracked_dirty(status_text: str) -> bool:
    """A tree has tracked modifications when `git status --short` lists any line that is not an untracked file."""
    return any(line.strip() and not line.startswith("??") for line in (status_text or "").splitlines())


def trainer_identity(sdpo: str) -> dict:
    code, head = git(sdpo, "rev-parse", "HEAD")
    scode, status = git(sdpo, "status", "--short")
    sources = {}
    root = Path(sdpo) / "verl" / "trainer"
    for path in sorted([root / "main_ppo.py"] + sorted((root / "config").rglob("*")) if (root / "config").is_dir() else [root / "main_ppo.py"]):
        if path.is_file() and "__pycache__" not in path.parts:
            sources[str(path.relative_to(sdpo))] = sha256_bytes(path.read_bytes())
    return {"commit": head.strip() if code == 0 and head.strip() else None, "status": status if scode == 0 else None,
            "status_empty": scode == 0 and not status.strip(), "tracked_modifications": tracked_dirty(status) if scode == 0 else None,
            "sources_sha256": sources, "sources_combined_sha256": content_sha256(sources)}


def source_identity(pilot_commit, pilot_status, trainer: dict, run: str) -> list:
    """Round-3 ruling F4: the trainer's source identity with the pilot's run is ESTABLISHED only when the pilot run's
    recorded commit (env/sdpo-commit.txt) is the checkout's commit now AND the pilot's recorded `git status --short`
    (env/sdpo-dirty.txt) and the checkout's now are both EMPTY: a clean tree at one commit has one content. Anything else
    is a problem (no "not verifiable ... ok")."""
    problems = []
    if not trainer.get("commit"):
        problems.append("source identity not established: the trainer checkout's commit cannot be read")
    if pilot_commit is None:
        problems.append("source identity not established: %s/env/sdpo-commit.txt is missing" % run)
    elif trainer.get("commit") and pilot_commit != trainer["commit"]:
        problems.append("source identity not established: trainer commit %s, the pilot's %s (%s)" % (trainer["commit"], pilot_commit, run))
    if pilot_status is None:
        problems.append("source identity not established: %s/env/sdpo-dirty.txt is missing" % run)
    elif pilot_status.strip():
        problems.append("source identity not established: the pilot's trainer tree was not clean (%s/env/sdpo-dirty.txt lists %s)"
                        % (run, "; ".join(pilot_status.strip().splitlines()[:3])))
    if not trainer.get("status_empty"):
        problems.append("source identity not established: the trainer checkout is not clean now (git status --short: %s)"
                        % ("; ".join((trainer.get("status") or "unreadable").strip().splitlines()[:3])))
    return problems


DATA_FILES = ("train.parquet", "test.parquet")


def data_hashes(sdpo: str, dataset: str) -> dict:
    """{file name: sha256 or None} of the task files the launchers read (SDPO_DIR/<dataset>/train.parquet, test.parquet)."""
    out = {}
    for name in DATA_FILES:
        try:
            out[name] = sha256_bytes((Path(sdpo) / dataset / name).read_bytes())
        except OSError:
            out[name] = None
    return out


def tracked_sources(sdpo: str, paths: list) -> tuple:
    """({path relative to the checkout: sha256}, problems): each path must be a TRACKED file of the pinned commit
    (`git ls-files --error-unmatch`) with no modification against HEAD (`git diff --quiet HEAD --`)."""
    root, problems, hashes = Path(sdpo), [], {}
    rels = [str(Path(p).relative_to(root)) if Path(p).is_absolute() else str(p) for p in paths]
    for rel in rels:
        try:
            hashes[rel] = sha256_bytes((root / rel).read_bytes())
        except OSError:
            hashes[rel] = None
            problems.append("%s cannot be read" % rel)
    if rels:
        code, _out = git(sdpo, "ls-files", "--error-unmatch", "--", *rels)
        if code != 0:
            untracked = [rel for rel in rels if git(sdpo, "ls-files", "--error-unmatch", "--", rel)[0] != 0]
            problems.append("not tracked files of the pinned commit: %s" % ", ".join(untracked or rels))
        code, _out = git(sdpo, "diff", "--quiet", "HEAD", "--", *rels)
        if code != 0:
            problems.append("modified against the pinned commit (git diff HEAD): %s" % ", ".join(
                rel for rel in rels if git(sdpo, "diff", "--quiet", "HEAD", "--", rel)[0] != 0))
    return hashes, problems


def regenerated_data(sdpo: str, dataset: str) -> tuple:
    """({file name: sha256}, None, sources) of the task files as the PINNED preprocessing writes them now, or (None, why,
    sources); `sources` = {path in the checkout: sha256} of data/preprocess.py and every source JSON of the task.

    The pilot's two launchers at tag kit-batch3-v1 recorded no hash of their task data (env/data-sha256.txt exists only
    from this tag on). Where that record is absent, identity is established another way: the trainer checkout is the
    pilot's commit with a clean tree (checked separately; a failure there fails the row anyway), data/preprocess.py and
    every source JSON of the task are TRACKED files of that commit with no modification (`git ls-files
    --error-unmatch`, `git diff --quiet HEAD --`; anything else is not established), and the authors' own
    `data/preprocess.py` is run again into a scratch folder on this machine. If the files in place are byte for byte
    what it writes, what is established is exactly this: the files in place equal what the pinned preprocessing writes
    from the pinned source. This does NOT establish what bytes the pilot read (the pilot campaign reuses parquet files
    that already existed; nothing recorded them). The sources' sha256 are archived with the regenerated and in-place
    hashes in the frozen recipe-check document.
    """
    import shutil, subprocess, sys, tempfile                                  # noqa: E401,PLC0415
    root, script = Path(sdpo), Path(sdpo) / "data" / "preprocess.py"
    sources = sorted((root / dataset).glob("*.json")) if (root / dataset).is_dir() else []
    if not script.is_file() or not sources:
        return None, "cannot regenerate %s: %s or its source JSON is missing" % (dataset, script), {}
    hashes, problems = tracked_sources(sdpo, [script] + sources)
    if problems:
        return None, "cannot regenerate %s from the pinned source: %s" % (dataset, "; ".join(problems)), hashes
    with tempfile.TemporaryDirectory() as scratch:
        top = Path(scratch)
        (top / "data").mkdir(); (top / dataset).mkdir(parents=True)
        shutil.copy(script, top / "data" / "preprocess.py")
        for source in sources:
            shutil.copy(source, top / dataset / source.name)
        env = {**os.environ, "PYTHONPATH": "%s%s%s" % (top, os.pathsep, root), "HF_DATASETS_CACHE": str(top / "hf-cache"),
               "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"}
        try:
            done = subprocess.run([sys.executable, "data/preprocess.py", "--data_source", dataset], cwd=top, env=env,
                                  capture_output=True, text=True, timeout=900)
        except (OSError, subprocess.SubprocessError) as error:
            return None, "cannot regenerate %s: %s" % (dataset, error), hashes
        if done.returncode != 0:
            return None, "cannot regenerate %s: the pinned preprocessing exited %d (%s)" % (dataset, done.returncode, done.stderr.strip()[-200:]), hashes
        out = {}
        for name in DATA_FILES:
            try:
                out[name] = sha256_bytes((top / dataset / name).read_bytes())
            except OSError:
                return None, "cannot regenerate %s: the pinned preprocessing wrote no %s" % (dataset, name), hashes
        return out, None, hashes


def recorded_data(folder: Path):
    """{file NAME: sha256} from a run's env/data-sha256.txt (`sha256sum` / `shasum -a 256` lines), or None when absent."""
    lines = read_lines(Path(folder) / "env" / "data-sha256.txt")
    if lines is None:
        return None
    out = {}
    for line in lines:
        parts = line.strip().split(None, 1)
        if len(parts) == 2 and re.fullmatch(r"[0-9a-f]{64}", parts[0]):
            out[Path(parts[1].lstrip("*")).name] = parts[0]
    return out


def export_files(folder: Path) -> dict:
    """{file name: size} of a merged-model folder (the incoming checkpoint's export file list)."""
    folder = Path(folder)
    return {p.name: p.stat().st_size for p in sorted(folder.iterdir()) if p.is_file()} if folder.is_dir() else {}


def sha256_stream(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def export_identity(folder: Path) -> dict:
    """{path relative to the folder: sha256} of EVERY file of a merged-model folder -- configuration, tokenizer files and
    every weight shard -- hashed by streaming (weights are many GB). The incoming checkpoint's identity by CONTENT (round-4
    ruling G2): a changed file list and a same-sized replacement of one file both change it."""
    folder = Path(folder)
    if not folder.is_dir():
        return {}
    return {str(p.relative_to(folder)): sha256_stream(p) for p in sorted(folder.rglob("*")) if p.is_file()}


def spec_of(frozen: dict, entry: dict, proposed_name: str) -> dict:
    """The comparison spec of one row from the frozen record, for the run named `proposed_name`."""
    stored = dict(entry.get("spec") or {})
    stored.pop("proposed_name", None)
    stored.update({"work_root": frozen.get("work_root"), "sdpo": frozen.get("sdpo_dir"), "kit": frozen.get("kit_dir")})
    return spec_for(proposed_name=proposed_name, **stored)


def reconstruction_problems(frozen: dict, entry: dict, row_id: str, recipe_dir: Path, work: Path) -> list:
    """Round-4 ruling G3: before a reconstructed pilot resolved configuration (or the pilot command) is read back, it is
    verified against the hash stored in the FROZEN record: the baseline document must still have its seal; the
    recipe-check frozen document (RECIPE_DIR/frozen.json) must have its seal and the baseline must be bound to it; the
    reconstruction's bytes must have the sha256 the frozen document recorded for this row (`reconstructed`, written at
    the first check; for a check passed after a recorded tool error, the sealed baseline's own `pilot_sha256`), and the
    pilot command's bytes the sha256 the frozen document recorded (`comparison.pilot_commands`). Empty when all hold."""
    frozen_tool = _load("p4_frozen")
    problems = []
    if frozen.get("content_sha256") != frozen_tool.content_sha256(frozen_tool.canonical(frozen)):
        problems.append("the recipe-check baseline was altered after it was sealed (its content no longer has its content_sha256)")
    document = frozen_tool._read(Path(recipe_dir) / "frozen.json")
    if document is None:
        return problems + ["the recipe-check frozen document %s is missing or unreadable" % (Path(recipe_dir) / "frozen.json")]
    if document.get("content_sha256") != frozen_tool.content_sha256(frozen_tool.canonical(document)):
        problems.append("the recipe-check frozen document was altered after it was frozen")
    elif frozen.get("frozen_sha256") != document.get("content_sha256"):
        problems.append("the recipe-check baseline is not bound to the frozen document")
    resolved = entry.get("resolved") or {}
    pilot_file = resolved.get("pilot_file")
    stored = (document.get("reconstructed") or {}).get(row_id) or {}
    expected = stored.get("pilot_sha256") or resolved.get("pilot_sha256")
    if stored and stored.get("pilot_sha256") != resolved.get("pilot_sha256"):
        problems.append("the baseline's reconstruction hash of %s is not the frozen document's" % row_id)
    if not pilot_file or not expected:
        problems.append("no hash of the reconstructed pilot configuration of %s is stored in the frozen record" % row_id)
    else:
        try:
            digest = sha256_bytes((Path(recipe_dir) / pilot_file).read_bytes())
        except OSError:
            digest = None
        if digest != expected:
            problems.append("the reconstructed pilot configuration %s has sha256 %s, the frozen record holds %s: refused"
                            % (pilot_file, str(digest)[:16], str(expected)[:16]))
    run = (entry.get("baseline") or {}).get("run")
    recorded = ((document.get("comparison") or {}).get("pilot_commands") or {}).get(run) or (entry.get("baseline") or {}).get("argv_sha256")
    if run:
        try:
            digest = sha256_bytes((Path(work) / run / "env" / "argv.txt").read_bytes())
        except OSError:
            digest = None
        if not recorded or digest != recorded:
            problems.append("the pilot command %s/env/argv.txt has sha256 %s, the frozen record holds %s: refused"
                            % (run, str(digest)[:16], str(recorded)[:16]))
    return problems


def verify_run(work: Path, frozen: dict, row_id: str, run_folder: Path, recipe_dir: Path) -> tuple:
    """(True, []) when a package-4 run's ACTUAL env/argv.txt and env/resolved-config.yaml are its frozen pilot baseline's
    under the exact table (round-3 rulings F3, F4); (False, problems) when they differ; (None, why) when the evidence --
    the run's files, the baseline's command, the reconstructed baseline configuration, the frozen entry -- is missing."""
    entry = ((frozen or {}).get("rows") or {}).get(row_id) or {}
    baseline = entry.get("baseline") or {}
    name = Path(run_folder).name
    if frozen is None or frozen.get("ok") is not True or entry.get("ok") is not True or not baseline.get("run") or not entry.get("spec"):
        return None, ["%s: the frozen recipe-check record does not verify %s" % (name, row_id)]
    base_lines = read_lines(Path(work) / baseline["run"] / "env" / "argv.txt")
    run_lines = read_lines(Path(run_folder) / "env" / "argv.txt")
    resolved = read_resolved(Path(run_folder) / "env" / "resolved-config.yaml")
    pilot_file = (entry.get("resolved") or {}).get("pilot_file")
    reconstructed = read_resolved(Path(recipe_dir) / pilot_file) if pilot_file else None
    integrity = reconstruction_problems(frozen, entry, row_id, recipe_dir, work)
    if integrity:                                              # G3: nothing unverified is compared (a refusal)
        return None, ["%s: %s" % (name, p) for p in integrity]
    missing = [what for what, value in (("the pilot baseline's command %s/env/argv.txt" % baseline["run"], base_lines),
                                        ("env/argv.txt", run_lines), ("env/resolved-config.yaml", resolved),
                                        ("the reconstructed pilot configuration %s" % pilot_file, reconstructed)) if value is None]
    if missing:
        return None, ["%s: %s missing or unreadable: the run's settings cannot be verified" % (name, ", ".join(missing))]
    spec = spec_of(frozen, entry, name)
    problems = ["%s: command %s is %r, the pilot baseline %s has %r: not a permitted substitution%s"
                % (name, d["key"], d["proposed"], Path(baseline["run"]).name, d["baseline"],
                   " (expected %r -> %r)" % tuple(d["expected"]) if d.get("expected") else "")
                for d in compare_argv(base_lines, run_lines, spec)["not_permitted"]]
    problems += ["%s: resolved %s is %r, the pilot baseline's (reconstructed) %r: not a permitted substitution%s"
                 % (name, d["key"], d["proposed"], d["baseline"], " (expected %r -> %r)" % tuple(d["expected"]) if d.get("expected") else "")
                 for d in compare_resolved(reconstructed, resolved, spec)["not_permitted"]]
    return (not problems) or False, problems


# ------------------------------------------------------------------------------------------------ the check
def check(work: Path, campaign_path: Path, selection_path: Path, out: Path, attempt: int, kit: Path = HERE) -> dict:
    """The comparison of every training row with its pilot baseline. Returns {content, comparison, inputs, tool_errors}:
    `comparison` holds the comparison INPUTS (frozen at the first check, whatever its verdict: round-3 ruling F2),
    `inputs` the WORK files among them with their sha256 (kit/p4_frozen.py re-hashes them)."""
    runner, pilot_report = _load("runner"), _load("pilot_report")
    campaign = runner.load_campaign(Path(campaign_path))
    selection = json.loads(Path(selection_path).read_text())
    recipe, chain = selection["group"]["recipe"], selection["group"]["chain"]
    sdpo = os.environ.get("SDPO_DIR")
    folder = out / ("a%d" % attempt)
    (folder / "pilot").mkdir(parents=True, exist_ok=True)
    (folder / "proposed").mkdir(parents=True, exist_ok=True)
    pilot = pilot_report.Work(Path(work))
    problems, tool_errors, rows, proposed_argv, proposed_resolved = [], [], {}, {}, {}
    if not sdpo:
        tool_errors.append("SDPO_DIR is not set")
    trainer = trainer_identity(sdpo) if sdpo else {}
    files, mapping, pilot_commands, data, incoming_ids = set(), {}, {}, {}, {}
    regenerated, data_identity = {}, {}
    for row in training_rows(campaign):
        r = role(row)
        entry = {"role": r, "ok": False, "problems": []}
        rows[row["id"]] = entry
        if r["config"] is None:
            entry["problems"].append("the row's launcher and LR/MINI_BATCH name no registered configuration")
            continue
        key = baseline_key(r, recipe, chain)
        merged = pilot.merged_run(key)
        if merged is None:
            entry["problems"].append("the pilot has no attempt of record of %s (no run-summary.json with merged 1)" % key)
            continue
        base_attempt, base_folder, base_summary = merged
        run_rel = "runs/%s" % base_folder.name
        mapping[row["id"]] = {"key": key, "attempt": base_attempt, "run": run_rel}
        for name in ("run-summary.json", "env/argv.txt", "env/sdpo-commit.txt", "env/sdpo-dirty.txt", "env/data-sha256.txt"):
            files.add("%s/%s" % (run_rel, name))
        base_argv = read_lines(base_folder / "env" / "argv.txt")
        if base_argv is None:
            entry["problems"].append("the pilot's recorded command %s/env/argv.txt is missing" % base_folder.name)
            continue
        pilot_commands[run_rel] = sha256_bytes(("\n".join(base_argv) + "\n").encode())
        (folder / "pilot" / ("%s.argv.txt" % base_folder.name)).write_text("\n".join(base_argv) + "\n")
        _h, bkeys, _d = parse_argv(base_argv)
        knobs = knobs_from(bkeys, r["launcher"])
        entry["problems"] += knob_problems(knobs, base_summary, r["launcher"])
        if bkeys.get(MODEL_KEY) != base_summary.get("model_dir"):
            entry["problems"].append("the pilot baseline's command trains %r, its run-summary.json says %r" % (bkeys.get(MODEL_KEY), base_summary.get("model_dir")))
        # F4: the trainer's source identity with this baseline
        commit_text = read_lines(base_folder / "env" / "sdpo-commit.txt")
        dirty_text = read_lines(base_folder / "env" / "sdpo-dirty.txt")
        entry["problems"] += source_identity((commit_text or [None])[0], None if dirty_text is None else "\n".join(dirty_text), trainer, run_rel)
        # F4: the identity of the task data, against a pilot run of record that trained the same task
        lineage = selection["lineages"][r["lineage"]]
        stage1 = lineage["stage1"]
        candidates = [(run_rel, base_folder, base_summary.get("dataset") or bkeys.get("vars.task"))]
        s1_merged = pilot.merged_run(stage1["key"])
        if s1_merged is not None:
            candidates.append(("runs/%s" % s1_merged[1].name, s1_merged[1], s1_merged[2].get("dataset")))
            files.update("runs/%s/%s" % (s1_merged[1].name, name) for name in ("run-summary.json", "env/data-sha256.txt"))
        source = next((c for c in candidates if c[2] == r["dataset"]), None)
        now = data_hashes(sdpo, r["dataset"]) if sdpo else {name: None for name in DATA_FILES}
        data[r["dataset"]] = now
        if source is None:
            entry["problems"].append("task data identity not established: no pilot run of record of this lineage trained %s" % r["dataset"])
        else:
            recorded = recorded_data(source[1])
            if recorded is None:
                # the pilot recorded no hash (its launchers did not, at tag kit-batch3-v1): regenerate from the pinned source
                if r["dataset"] not in regenerated:
                    regenerated[r["dataset"]] = regenerated_data(sdpo, r["dataset"]) if sdpo else (None, "no SDPO_DIR", {})
                fresh, why, source_hashes = regenerated[r["dataset"]]
                data_identity[r["dataset"]] = {"source": "regenerated from the pinned source (the pilot run %s recorded no data-sha256.txt)" % source[0],
                                               "regenerated": fresh, "in_place": now, "why_not": why,
                                               "pinned_sources_sha256": source_hashes,
                                               "establishes": "the files in place equal what the pinned preprocessing writes from the "
                                                              "pinned source; this does not establish what bytes the pilot read"}
                if fresh is None:
                    entry["problems"].append("task data identity not established: %s/env/data-sha256.txt is missing and %s" % (source[0], why))
                else:
                    for name in DATA_FILES:
                        if not now.get(name) or fresh.get(name) != now.get(name):
                            entry["problems"].append("task data identity not established: %s/%s has sha256 %s, the pinned preprocessing writes %s"
                                                     % (r["dataset"], name, str(now.get(name))[:16], str(fresh.get(name))[:16]))
            else:
                for name in DATA_FILES:
                    if not recorded.get(name) or recorded.get(name) != now.get(name):
                        entry["problems"].append("task data identity not established: %s/%s has sha256 %s, the pilot run %s read %s"
                                                 % (r["dataset"], name, str(now.get(name))[:16], source[0], str(recorded.get(name))[:16]))
        # the incoming checkpoint: the selection's recorded run-summary and the export's file list
        incoming_rel = "runs/%s-a%d" % (stage1["key"], stage1["attempt"])
        files.add("%s/run-summary.json" % incoming_rel)
        if incoming_rel not in incoming_ids:              # G2: the content of every file, hashed once per checkpoint
            incoming_ids[incoming_rel] = {"run_summary_sha256": stage1.get("run_summary_sha256"),
                                          "export_files": export_files(Path(work) / incoming_rel / "hf-step40"),
                                          "export_sha256": export_identity(Path(work) / incoming_rel / "hf-step40")}
        now_summary = sha256_bytes((Path(work) / incoming_rel / "run-summary.json").read_bytes()) if (Path(work) / incoming_rel / "run-summary.json").is_file() else None
        if not stage1.get("run_summary_sha256") or now_summary != stage1.get("run_summary_sha256"):
            entry["problems"].append("the incoming checkpoint %s: its run-summary.json has sha256 %s, the selection recorded %s"
                                     % (incoming_rel, str(now_summary)[:16], str(stage1.get("run_summary_sha256"))[:16]))
        if "config.json" not in incoming_ids[incoming_rel]["export_files"]:
            entry["problems"].append("the incoming checkpoint %s/hf-step40 holds no config.json" % incoming_rel)
        incoming = str(Path(work) / incoming_rel / "hf-step40")
        env = runner.resolve(row, campaign, 1)["env"]
        env = {k: v for k, v in env.items() if not k.startswith("P4_")}
        env.update({**knobs, "MODEL_DIR": incoming, "SDPO_DIR": sdpo or "", "WORK": str(work)})
        code, argv, err = dry_run(kit, r["launcher"], env)
        if code != 0:
            entry["problems"].append("the launcher's dry run exited %d: %s" % (code, err.strip()[-300:]))
            tool_errors.append("%s: the launcher's dry run exited %d" % (row["id"], code))
            continue
        (folder / "proposed" / ("%s.argv.txt" % row["id"])).write_text("\n".join(argv) + "\n")
        proposed_argv[row["id"]] = argv
        spec = spec_for(baseline_name=base_folder.name, proposed_name=env["NAME"], incoming=incoming, seed=r["seed"], dataset=r["dataset"],
                        b=r["config"] == "b", intervention=r["intervention"], qualification=r["qualification"],
                        baseline_model=bkeys.get(MODEL_KEY), baseline_seed=bkeys.get("data.seed"), baseline_dataset=bkeys.get("vars.task"),
                        work_root=str(work), sdpo=sdpo, kit=str(kit), folder=LAUNCHER_FOLDER[r["launcher"]])
        compared = compare_argv(base_argv, argv, spec)
        entry["problems"] += ["command: %s baseline %r, proposed %r is not a permitted substitution%s"
                              % (d["key"], d["baseline"], d["proposed"], " (expected %r -> %r)" % tuple(d["expected"]) if d.get("expected") else "")
                              for d in compared["not_permitted"]]
        base_yaml, why = resolve(base_argv, sdpo, {"TASK": bkeys.get("vars.task", ""), "EXPERIMENT": base_folder.name}) if sdpo else (None, "no SDPO_DIR")
        prop_yaml, why2 = resolve(argv, sdpo, {"TASK": env.get("DATASET", ""), "EXPERIMENT": env["NAME"]}) if sdpo else (None, "no SDPO_DIR")
        resolved = {"pilot_label": "reconstructed (re-resolved from the pilot's recorded command; the pilot wrote no resolved configuration)"}
        if base_yaml is None or prop_yaml is None:
            entry["problems"].append("resolved configuration: %s" % (why or why2))
            tool_errors.append("%s: the trainer could not resolve a configuration" % row["id"])
        else:
            base_path = folder / "pilot" / ("%s.resolved-reconstructed.yaml" % base_folder.name)
            prop_path = folder / "proposed" / ("%s.resolved.yaml" % row["id"])
            base_path.write_text(base_yaml)
            prop_path.write_text(prop_yaml)
            bflat, pflat = read_resolved(base_path), read_resolved(prop_path)
            proposed_resolved[row["id"]] = pflat
            if bflat is None or pflat is None:
                entry["problems"].append("a resolved configuration cannot be read as YAML (is PyYAML installed?)")
                tool_errors.append("%s: a resolved configuration cannot be read" % row["id"])
            else:
                rc = compare_resolved(bflat, pflat, spec)
                resolved.update({"differences": rc["differences"], "ok": rc["ok"], "pilot_sha256": sha256_bytes(base_yaml.encode()),
                                 "proposed_sha256": sha256_bytes(prop_yaml.encode()),
                                 "pilot_file": "a%d/pilot/%s" % (attempt, base_path.name), "proposed_file": "a%d/proposed/%s" % (attempt, prop_path.name)})
                entry["problems"] += ["resolved: %s baseline %r, proposed %r is not a permitted substitution%s"
                                      % (d["key"], d["baseline"], d["proposed"], " (expected %r -> %r)" % tuple(d["expected"]) if d.get("expected") else "")
                                      for d in rc["not_permitted"]]
        stored = {k: v for k, v in spec.items() if k not in ("proposed_name", "work_root", "sdpo", "kit")}
        entry.update({"baseline": {"key": key, "attempt": base_attempt, "run": run_rel,
                                   "argv_sha256": sha256_bytes(("\n".join(base_argv) + "\n").encode()),
                                   "run_summary_sha256": sha256_bytes((base_folder / "run-summary.json").read_bytes())},
                      "knobs": knobs, "incoming": "%s/hf-step40" % incoming_rel, "spec": stored,
                      "data_identity": {"dataset": r["dataset"], "source": source[0] if source else None, "now": now},
                      "proposed_argv_sha256": sha256_bytes(("\n".join(argv) + "\n").encode()),
                      "command_differences": compared["differences"], "resolved": resolved})
        entry["ok"] = not entry["problems"]
    # c against s, per seed: a REPORT of every difference, and the shared settings verified
    c_vs_s = {}
    for seed in ("101", "102"):
        c_row = "p4-br-c-s%s" % seed
        s_row = ("p4-br-s-s%s" % seed) if recipe == "g8" else ("p4-r1-ctl%d" % (1 if seed == "101" else 2))
        if c_row in proposed_argv and s_row in proposed_argv and (proposed_resolved.get(c_row) is None or proposed_resolved.get(s_row) is None) \
                and any(e.startswith((c_row + ":", s_row + ":")) for e in tool_errors):
            problems.append("tool error: no resolved configuration of c or s at seed %s (the trainer could not resolve one)" % seed)
        elif c_row in proposed_argv and s_row in proposed_argv:
            shared = shared_settings(proposed_argv[c_row], proposed_argv[s_row], proposed_resolved.get(c_row), proposed_resolved.get(s_row))
            c_vs_s[seed] = {"c": c_row, "s": s_row,
                            "command_differences": difference_report(parse_argv(proposed_argv[c_row])[1], parse_argv(proposed_argv[s_row])[1]),
                            "resolved_differences": difference_report(proposed_resolved.get(c_row) or {}, proposed_resolved.get(s_row) or {}),
                            "shared_settings": shared}
            if not shared["ok"]:
                problems.append("c and s (seed %s) differ in a shared setting of S2: %s" % (seed, shared["unequal"]))
        else:
            problems.append("no proposed commands for c and s at seed %s" % seed)
    bad_rows = [rid for rid, e in rows.items() if not e["ok"]]
    if bad_rows:
        problems.append("rows not verified: %s" % ", ".join(bad_rows))
    problems += ["tool error: %s" % e for e in tool_errors if e == "SDPO_DIR is not set"]
    sources = {"campaign": sha256_bytes(Path(campaign_path).read_bytes()),
               **{"kit/%s" % name: sha256_bytes((Path(kit) / name).read_bytes()) for name in (GRPO, SDPO, "beds/authors_gate.py")
                  if (Path(kit) / name).is_file()}}
    comparison = {"mapping": mapping, "pilot_commands": pilot_commands, "sources": sources,
                  "trainer": {k: trainer.get(k) for k in ("commit", "status", "sources_combined_sha256")}, "data": data,
                  "data_identity": data_identity,
                  "incoming": incoming_ids, "work_root": str(work), "sdpo_dir": sdpo, "kit_dir": str(kit),
                  "selection_content_sha256": selection.get("content_sha256")}
    inputs = [{"path": rel, "sha256": sha256_bytes((Path(work) / rel).read_bytes()) if (Path(work) / rel).is_file() else None}
              for rel in sorted(files)]
    content = {"schema": SCHEMA, "group": {"recipe": recipe, "chain": chain}, "selection_content_sha256": selection.get("content_sha256"),
               "work_root": str(work), "sdpo_dir": sdpo, "kit_dir": str(kit),
               "rows": rows, "trainer": {**trainer, "source_identity": "established only by the same commit and two empty `git status "
                                                                       "--short` (the pilot's recorded and the checkout's now)"},
               "c_vs_s": c_vs_s, "evidence_folder": "a%d" % attempt, "ok": not problems, "problems": problems}
    return {"content": content, "comparison": comparison, "inputs": inputs, "tool_errors": tool_errors,
            "proposed": {rid: sha256_bytes(("\n".join(a) + "\n").encode()) for rid, a in sorted(proposed_argv.items())}}


def comparison_digest(content: dict) -> str:
    """The canonical content of a check that a later check must reproduce: everything but the evidence folder and the
    attempt-specific file names of the resolved configurations."""
    return content_sha256({k: v for k, v in content.items() if k != "evidence_folder"} | {"rows": {
        rid: {k: v for k, v in e.items() if k != "resolved"} | {"resolved_ok": (e.get("resolved") or {}).get("ok"),
                                                                "resolved_sha256": [(e.get("resolved") or {}).get("pilot_sha256"),
                                                                                    (e.get("resolved") or {}).get("proposed_sha256")]}
        for rid, e in content["rows"].items()}})


def _diff_keys(a: dict, b: dict, prefix="") -> list:
    out = []
    for key in sorted(set(a) | set(b)):
        x, y = a.get(key), b.get(key)
        if isinstance(x, dict) and isinstance(y, dict):
            out += _diff_keys(x, y, "%s%s." % (prefix, key))
        elif x != y:
            out.append(prefix + str(key))
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("check")
    p.add_argument("--work", required=True)
    p.add_argument("--campaign", required=True)
    p.add_argument("--selection", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--attempt", type=int, required=True)
    args = parser.parse_args(argv)
    out, work = Path(args.out), Path(args.work)
    check_path = out / ("check-a%d.json" % args.attempt)
    if check_path.exists() or (out / ("a%d" % args.attempt)).exists():
        print("refused: attempt %d of the recipe-check exists under %s" % (args.attempt, out), file=sys.stderr)
        return 2
    frozen_inputs = _load("p4_frozen")

    def finish(result: dict, code: int) -> int:
        out.mkdir(parents=True, exist_ok=True)
        check_path.write_text(json.dumps(result, indent=1, sort_keys=True, default=str) + "\n")
        for problem in result.get("problems") or []:
            print("recipe-check: %s" % problem, file=sys.stderr)
        print("recipe-check: %s" % result.get("summary"))
        return code

    # round-3 ruling F2: the frozen decision inputs (and this check's own frozen inputs, once they exist) first
    # The recipe pilot precedes the rows that create containment receipts.
    stale = frozen_inputs.verify(work, need=("selection",) + (("reservation",) if (work / frozen_inputs.RESERVATION).exists() else ()), containment=False)
    if stale:
        return finish({"ok": 0, "problems": [frozen_inputs.refusal(stale)], "summary": "refused: %s" % frozen_inputs.CHANGED}, 2)
    made = check(work, Path(args.campaign), Path(args.selection), out, args.attempt)
    content, comparison = made["content"], made["comparison"]
    digest = comparison_digest(content)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    (out / ("a%d" % args.attempt) / "record.json").write_text(json.dumps({**content, "comparison": comparison, "inputs": made["inputs"],
                                                                         "comparison_digest": digest, "generated_at": stamp},
                                                                        indent=1, sort_keys=True, default=str) + "\n")
    tool_error_only = bool(content["problems"]) and bool(made["tool_errors"]) and all(
        e["problems"] == [] or all(p.startswith(("the launcher's dry run exited", "resolved configuration:", "a resolved configuration cannot"))
                                   for p in e["problems"]) for e in content["rows"].values()) and all(
        p.startswith(("rows not verified", "tool error", "no proposed commands")) for p in content["problems"])
    verdict = {"ok": content["ok"], "problems": content["problems"], "tool_errors": made["tool_errors"], "tool_error_only": tool_error_only}
    frozen_path, baseline = out / "frozen.json", out / "baseline.json"
    result = {"ok": int(content["ok"]), "rows": len(content["rows"]), "rows_verified": sum(e["ok"] for e in content["rows"].values()),
              "problems": list(content["problems"]), "comparison_digest": digest, "baseline": None, "frozen": None}
    if not frozen_path.exists():
        # the FIRST check is frozen whatever its verdict: its comparison inputs, the proposed commands, the verdict
        doc = frozen_inputs.seal({"schema": SCHEMA.replace("check", "frozen"), "group": content["group"],
                                  "selection_content_sha256": content["selection_content_sha256"], "work_root": str(work),
                                  "sdpo_dir": content["sdpo_dir"], "kit_dir": content["kit_dir"], "comparison": comparison,
                                  "inputs": made["inputs"], "proposed_commands": made["proposed"], "verdict": verdict,
                                  "reconstructed": {rid: {k: (e.get("resolved") or {}).get(k) for k in ("pilot_file", "pilot_sha256", "proposed_file", "proposed_sha256")}
                                                    for rid, e in sorted(content["rows"].items()) if (e.get("resolved") or {}).get("pilot_sha256")},
                                  "comparison_digest": digest, "evidence_folder": "a%d" % args.attempt})
        with frozen_path.open("x") as handle:
            json.dump(doc, handle, indent=1, sort_keys=True, default=str)
        result["frozen"] = "written"
        frozen = doc
    else:
        frozen = json.loads(frozen_path.read_text())
        changed = _diff_keys(frozen.get("comparison") or {}, comparison)
        if changed:
            result.update({"ok": 0, "frozen": "refused"})
            result["problems"] = ["%s: the recipe check's comparison inputs differ from the frozen ones in %s; nothing is overwritten"
                                  % (frozen_inputs.CHANGED, ", ".join(changed[:10]))]
            return finish({**result, "summary": "refused: %s" % frozen_inputs.CHANGED}, 2)
        before = frozen.get("verdict") or {}
        if before.get("ok") is True:
            if not content["ok"] or digest != frozen.get("comparison_digest") or made["proposed"] != frozen.get("proposed_commands"):
                result.update({"ok": 0, "frozen": "refused"})
                result["problems"] = result["problems"] + ["the frozen check passed and this one, from the same inputs, does not reproduce it "
                                                           "(comparison digest %s, frozen %s)" % (digest[:16], str(frozen.get("comparison_digest"))[:16])]
                return finish({**result, "summary": "refused: the frozen verdict is not reproduced"}, 2)
            result["frozen"] = "unchanged"
        elif not before.get("tool_error_only"):
            result.update({"ok": 0, "frozen": "failed and final"})
            result["problems"] = ["the frozen recipe check failed (%s) and is not repeated: a completed comparison that failed is final "
                                  "(only a recorded tool error with byte-identical inputs may be run again)" % "; ".join((before.get("problems") or [])[:3])]
            return finish({**result, "summary": "the frozen check failed"}, 1)
        else:
            result["frozen"] = "failed earlier by a recorded tool error; inputs byte-identical; run again"
    if content["ok"]:
        if baseline.exists():
            old = json.loads(baseline.read_text())
            if old.get("comparison_digest") != digest:
                result.update({"ok": 0, "baseline": "refused: the frozen baseline %s has comparison digest %s, this check %s; nothing is "
                               "overwritten" % (baseline, str(old.get("comparison_digest"))[:16], digest[:16])})
                result["problems"] = result["problems"] + [result["baseline"]]
            else:
                result["baseline"] = "unchanged"
        else:
            doc = frozen_inputs.seal({**content, "comparison_digest": digest, "frozen_sha256": frozen_inputs.content_sha256(frozen_inputs.canonical(frozen))})
            with baseline.open("x") as handle:
                json.dump({**doc, "generated_at": stamp}, handle, indent=1, sort_keys=True, default=str)
            result["baseline"] = "written"
    result["summary"] = ("%d of %d training rows verified against the pilot's records; frozen record %s; baseline %s"
                         % (result["rows_verified"], result["rows"], result["frozen"], result["baseline"]))
    return finish(result, 0 if result["ok"] else 1)


if __name__ == "__main__":
    sys.exit(main())
