"""CPU regressions for the four execution/retry defects; GPU work is substituted."""
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import pytest
from kit import runner, v4_campaign_ops as ops, v4_teacher as t

ROOT = Path(__file__).resolve().parents[1]


def scheduled_data(work, arm="F", *, allowed=True, synthetic=False, stand_in=False):
    import pyarrow as pa
    import pyarrow.parquet as pq
    target = work / "v4/report-main/scheduled/finqa-s101"
    target.mkdir(parents=True, exist_ok=True)
    rows = [dict(id=str(i), prompt=[dict(role="user", content="question")],
                 extra_info=dict(index=str(i), split="train", demonstration="demo", rewrite="rewrite"),
                 technical_synthetic=synthetic) for i in range(1280)]
    pq.write_table(pa.Table.from_pylist(rows), target / "train.parquet")
    pq.write_table(pa.Table.from_pylist(rows[:32]), target / "val.parquet")
    doc = dict(schema="kit-v4-sft-data.v1", arm="COMMON", seed=101, rows=1280, response_only=True,
               train_sha256=t.sha((target / "train.parquet").read_bytes()),
               val_sha256=t.sha((target / "val.parquet").read_bytes()),
               ids=[str(i) for i in range(1280)], technical_synthetic=synthetic,
               scientific_phase_allowed=allowed, stand_in_teacher=stand_in,
               teacher_models=[], common_training_manifest_sha256="a" * 64,
               schedule_sha256="b" * 64)
    t.atomic_json(target / "manifest.json", doc)
    model = work / "model"; model.mkdir(exist_ok=True)
    t.atomic_json(model / "config.json", {})
    return target, model


@pytest.mark.parametrize("arm", ["F", "R"])
@pytest.mark.parametrize("merge_status", [17, 0])
def test_failed_sft_merge_fails_runner_and_retry_only_recovers_export(tmp_path, monkeypatch, arm, merge_status):
    from kit import v4_retention
    from kit.sim.v4_merge_standin import export
    scheduled, model = scheduled_data(tmp_path, arm)
    ident = arm + "-finqa-alone-s101"
    sdpo = tmp_path / "sdpo"; sdpo.mkdir()
    helpers = tmp_path / "helpers"; helpers.mkdir()
    # Substitute only token counting and GPU/host commands, while executing the
    # real data checks, shell launcher, receipts, wrapper, runner and recovery.
    (helpers / "sitecustomize.py").write_text(
        "from kit import v4_contract\nv4_contract.prompt_length = lambda *a, **k: 1\n")
    python = helpers / "python"
    python.write_text("#!" + sys.executable + "\n" + textwrap.dedent(r'''
        import json, os, sys
        from pathlib import Path
        a = sys.argv[1:]
        if a[:2] == ['-m', 'torch.distributed.run']:
            from kit.v4_arms import checkpoint_path
            from kit.v4_retention import state_names
            out = Path(os.environ['WORK']) / 'runs' / os.environ['NAME']
            count = Path(os.environ['WORK']) / 'trainer-invocations'
            count.write_text(count.read_text() + 'dose\n' if count.exists() else 'dose\n')
            for step in (20, 40):
                root = checkpoint_path(out, os.environ['ARM'], step)
                for name in state_names(os.environ['ARM']):
                    path = root / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b'CPU state fixture')
            (out / 'metrics.jsonl').write_text(''.join(json.dumps({'step': i, 'data': {'train/loss': 1., 'train/lr': 1e-5 * min((i-1)/10, 1), 'v4/completed_optimizer_updates': i}}) + '\n' for i in range(1, 41)))
            sys.exit(0)
        if a[:2] == ['-m', 'verl.model_merger']:
            sys.exit(int(os.environ['TEST_MERGE_STATUS']))  # zero without weights is also incomplete
        if a and a[0].endswith('/sft_entry.py') and '--cfg' in a:
            print('CPU composed-config substitute: true'); sys.exit(0)
        os.execv(sys.executable, [sys.executable, *a])
    '''))
    python.chmod(0o755)
    for name, body in [("git", "[[ \"$*\" != *rev-parse* ]] || echo 7c457fc1b1f636ae794eb0362ba37d4743b06fbc"),
                       ("pip", "exit 0"), ("nvidia-smi", "exit 0")]:
        path = helpers / name; path.write_text("#!/bin/bash\n" + body + "\n"); path.chmod(0o755)
    env = dict(WORK=str(tmp_path), V4_FINQA_CAMPAIGN="1", V4_PHASE0_MODE="0",
               ARM=arm, NAME=ident, MODEL_DIR=str(model), SDPO_DIR=str(sdpo),
               TRAIN_FILE=str(scheduled / "train.parquet"), VAL_FILE=str(scheduled / "val.parquet"),
               DATA_MANIFEST=str(scheduled / "manifest.json"), SEED="101", STEPS="40",
               V4_PROFILE="scientific", NGPU="8", TEST_MERGE_STATUS=str(merge_status),
               PATH=str(helpers) + os.pathsep + os.environ["PATH"],
               PYTHONPATH=str(ROOT) + os.pathsep + str(helpers))
    for key, value in env.items(): monkeypatch.setenv(key, value)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    campaign_path = tmp_path / "campaign.json"
    campaign_path.write_text(json.dumps(dict(schema=runner.SCHEMA, name="merge-regression", workdir_env="WORK",
        rows=[dict(id="train-" + ident, command=["bash", str(ROOT / "kit/run_v4.sh")])])) )
    assert runner.main(["run", str(campaign_path), "--all"]) == 1
    campaign = runner.load_campaign(campaign_path)
    verdict = runner.latest_verdict(campaign, "train-" + ident)
    assert verdict["verdict"] == "FAIL"
    assert verdict["returncode"] == (merge_status or 4)
    out = tmp_path / "runs" / ident
    summary = json.loads((out / "run-summary.json").read_text())
    assert summary["dose_complete"] and summary["completed_optimizer_updates"] == 40
    assert summary["merged"] == 0 and summary["merge_returncode"] == merge_status
    before = {p.relative_to(out): p.read_bytes() for p in (out / "train").rglob("*.pt")}
    mergers = []
    def gpu_merge(argv, env=None):
        assert "verl.model_merger" in argv  # a second trainer invocation is forbidden
        destination = Path(argv[argv.index("--target_dir") + 1]); mergers.append(destination)
        export(destination)
    monkeypatch.setattr(ops, "run", gpu_merge)
    monkeypatch.setattr(v4_retention, "sweep", lambda *a, **k: None)
    monkeypatch.setattr(v4_retention, "reload_weights", lambda *a: dict(tensor_count=1, dtype="BF16"))
    ops.train(tmp_path, "main", ident)
    assert [p.name for p in mergers] == ["hf-step40", "hf-step20"]
    assert (tmp_path / "trainer-invocations").read_text() == "dose\n"
    assert before == {p.relative_to(out): p.read_bytes() for p in (out / "train").rglob("*.pt")}
    assert json.loads((out / "run-summary.json").read_text())["export_recovered"]


@pytest.mark.parametrize("arm", ["F", "R"])
@pytest.mark.parametrize("case", ["low_coverage", "changed_data", "stand_in", "synthetic"])
def test_real_sft_data_check_coverage_is_advisory_but_identity_and_synthetic_are_hard(tmp_path, monkeypatch, arm, case):
    from kit import v4_contract, v4_finqa_policy as policy
    target, model = scheduled_data(tmp_path, arm, allowed=False,
                                    synthetic=case == "synthetic", stand_in=case == "stand_in")
    if case == "changed_data":
        doc = json.loads((target / "manifest.json").read_text()); doc["train_sha256"] = "0" * 64
        t.atomic_json(target / "manifest.json", doc)
    monkeypatch.setenv("WORK", str(tmp_path)); monkeypatch.setenv("V4_FINQA_CAMPAIGN", "1")
    monkeypatch.setenv("V4_PHASE0_MODE", "0"); monkeypatch.setenv("V4_PROFILE", "scientific")
    monkeypatch.setattr(v4_contract, "prompt_length", lambda *a, **k: 1)
    # Execute the launcher's actual embedded admission block, before any trainer.
    source = (ROOT / "kit/run_sft_f.sh").read_text().split('python - "$TRAIN_FILE" "$VAL_FILE" "$DATA_MANIFEST"')[1]
    block = source.split("<<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
    monkeypatch.setattr(sys, "argv", ["-", str(target / "train.parquet"), str(target / "val.parquet"),
        str(target / "manifest.json"), "101", str(model), arm, "40", str(tmp_path / "labels.json")])
    if case == "low_coverage":
        exec(compile(block, "run_sft_f.sh:data-check", "exec"), {})
        assert any("coverage" in r["message"] for r in policy.warnings(tmp_path))
        assert json.loads((tmp_path / "labels.json").read_text())["target_field"] == "extra_info." + ("rewrite" if arm == "R" else "demonstration")
    else:
        with pytest.raises((SystemExit, ValueError)):
            exec(compile(block, "run_sft_f.sh:data-check", "exec"), {})
        assert not (tmp_path / "labels.json").exists()


@pytest.mark.parametrize("task", ["finqa", "general-v1"])
def test_score_retry_preserves_each_incomplete_attempt_and_reuses_complete_score(tmp_path, monkeypatch, task):
    from kit import eval_bed, score_forgetting, v4_readers
    scorer = score_forgetting if task == "general-v1" else eval_bed
    row = "score-baseline-" + task + "-2048"
    monkeypatch.setenv("WORK", str(tmp_path)); monkeypatch.setenv("MODEL_DIR", str(tmp_path / "model"))
    monkeypatch.setenv("V4_CURRENT_ROW", row)
    calls = []
    def generate(argv, env=None):
        # Real scorer fresh_dir behavior; an engine crash leaves this directory.
        out = scorer.fresh_dir(Path(argv[argv.index("--out") + 1]))
        number = os.environ["V4_RUNNER_ATTEMPT"]; calls.append(number)
        (out / "partial.jsonl").write_text("engine-attempt-" + number)
        if int(number) < 3: raise subprocess.CalledProcessError(17, argv)
        t.atomic_json(out / ("forgetting.json" if task == "general-v1" else "bed-score.json"), {"CPU_fixture": True})
    monkeypatch.setattr(ops, "run", generate)
    monkeypatch.setattr(t, "local_identity", lambda *a: {"CPU_fixture": True})
    for number in (1, 2):
        monkeypatch.setenv("V4_RUNNER_ATTEMPT", str(number))
        with pytest.raises(subprocess.CalledProcessError): ops.score(tmp_path, "main", "baseline", task, 2048)
    monkeypatch.setenv("V4_RUNNER_ATTEMPT", "3")
    ops.score(tmp_path, "main", "baseline", task, 2048)
    for number in (1, 2):
        preserved = tmp_path / f"campaign/v4-main/{row}/attempt-{number}/scoring-output/partial.jsonl"
        assert preserved.read_text() == "engine-attempt-" + str(number)
    out = ops.paths(tmp_path, "main") / "eval" / ("baseline-" + task + "-2048")
    snapshot = {p.name: p.read_bytes() for p in out.iterdir()}
    validated = []
    monkeypatch.setattr(v4_readers, "scorer_context", lambda *a: (None, None))
    monkeypatch.setattr(v4_readers, "scoring", lambda *a, **k: validated.append(a[3]))
    monkeypatch.setenv("V4_RUNNER_ATTEMPT", "4")
    ops.score(tmp_path, "main", "baseline", task, 2048)
    # Reuse checks parse and checkpoint identity only; the full reader runs in the report (Astra, confirmation round).
    assert calls == ["1", "2", "3"] and validated == []
    assert snapshot == {p.name: p.read_bytes() for p in out.iterdir()}


@pytest.mark.parametrize("failed_verdict", [False, True])
def test_restoration_retry_preserves_old_legs_and_cannot_use_stale_kill_marker(tmp_path, monkeypatch, failed_verdict):
    from kit import v4_restore_check as restore, v4_finqa_campaign as campaign
    from test_kit_finqa_retention import restoration_arm_fixture
    base = ops.paths(tmp_path, "qualification")
    (base / "scheduled/restoration-finqa").mkdir(parents=True)
    (tmp_path / "v4/report-inputs").mkdir(parents=True)
    t.atomic_json(tmp_path / "v4/report-inputs/phase0-binding.json", {})
    monkeypatch.setenv("WORK", str(tmp_path)); monkeypatch.setenv("V4_FINQA_CAMPAIGN", "1")
    monkeypatch.setenv("V4_PHASE0_MODE", "0")
    monkeypatch.setattr(campaign, "ALL_ARMS", ("S",))
    monkeypatch.setattr(ops, "launcher_env", lambda work, arm, name, *a, **k: {"TEST_NAME": name})
    launches = []; generation = [1]; waits = []
    out = base / "restoration/S"
    def launch(command, **kwargs):
        if command[0] == "ray": return {"returncode": 0}
        env = kwargs["env"]; leg = env["V4_RESTORE_LEG"]; launches.append((generation[0], leg))
        if leg == "uninterrupted":
            assert not out.exists() or not any(out.iterdir()), "old observations must be preserved elsewhere"
            assert not list((tmp_path / "runs").glob("restoration-S-*")), "old legs must be preserved elsewhere"
            root = restoration_arm_fixture(tmp_path, "S")
            for part in ("uninterrupted", "killed", "resumed"):
                run = tmp_path / "runs" / ("restoration-S-" + part)
                (run / "attempt-note").write_text(str(generation[0]))
                if part != "killed": (run / "metrics.jsonl").write_bytes((root / (part + "-metrics.jsonl")).read_bytes())
        if generation[0] == 1 and leg == "resumed": return {"returncode": 17}
        return {"returncode": 0}
    monkeypatch.setattr(runner, "bounded_command", launch)
    class Killed:
        pid = 999999; returncode = None
        def __init__(self, *a, **kwargs):
            assert not (out / "ready-to-kill.json").exists(), "stale marker must not satisfy a new wait"
            launches.append((generation[0], "killed"))
        def poll(self):
            waits.append(generation[0])
            t.atomic_json(out / "ready-to-kill.json", {"step": 2, "checkpoint_durable": True, "attempt": generation[0]})
            return None
        def wait(self, timeout): self.returncode = -9; return -9
    monkeypatch.setattr(restore.subprocess, "Popen", Killed)
    monkeypatch.setattr(restore.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0, stdout=""))
    monkeypatch.setattr(restore.os, "killpg", lambda *a: None)
    with pytest.raises(ValueError, match="real-stack resumed"): restore.run_check(tmp_path)
    if failed_verdict: t.atomic_json(out / "verdict.json", {"passed": False, "error": "failed comparison"})
    evidence = {p.relative_to(out): p.read_bytes() for p in out.rglob("*") if p.is_file()}
    runs = {p.relative_to(tmp_path / "runs"): p.read_bytes() for p in (tmp_path / "runs").rglob("*") if p.is_file()}
    generation[0] = 2
    result = restore.run_check(tmp_path)
    assert result["S"]["passed"]
    archived = base / "restoration/failed-attempts/S/attempt-1"
    assert evidence == {p.relative_to(archived / "observations"): p.read_bytes() for p in (archived / "observations").rglob("*") if p.is_file()}
    assert runs == {p.relative_to(archived / "runs"): p.read_bytes() for p in (archived / "runs").rglob("*") if p.is_file()}
    assert len(launches) == 6 and waits == [1, 2]
    assert restore.run_check(tmp_path) == result and len(launches) == 6


@pytest.mark.parametrize('task,final',[('general-v1','forgetting.json'),('finqa','bed-score.json')])
def test_completed_score_is_reused_without_the_mid_run_reader(tmp_path,monkeypatch,task,final):
    """Astra's confirmation round: reuse ran the full report reader, which needs scheduler evidence that does
    not exist mid-run, and failed before writing a missing checkpoint identity."""
    from kit import v4_campaign_ops as ops
    monkeypatch.setenv('MODEL_DIR',str(tmp_path/'model'))
    monkeypatch.setattr(ops.t,'local_identity',lambda model,name:{'model':name,'hash':'h1'})
    ran=[];monkeypatch.setattr(ops,'run',lambda argv,env:ran.append(argv))
    out=ops.paths(tmp_path,'main')/'eval'/f'baseline-{task}-2048';out.mkdir(parents=True)
    (out/final).write_text('{"done": true}')
    ops.score(tmp_path,'main','baseline',task,2048)
    assert ran==[] and json.loads((out/'checkpoint-identity.json').read_text())=={'model':'Qwen/Qwen3-8B','hash':'h1'}
    ops.score(tmp_path,'main','baseline',task,2048)
    assert ran==[]


def test_score_for_another_checkpoint_is_kept_aside_and_rescored(tmp_path,monkeypatch):
    from kit import v4_campaign_ops as ops
    monkeypatch.setenv('MODEL_DIR',str(tmp_path/'model'))
    monkeypatch.setenv('V4_RUNNER_ATTEMPT','2');monkeypatch.setenv('V4_CURRENT_ROW','baseline-general-v1-2048')
    monkeypatch.setattr(ops.t,'local_identity',lambda model,name:{'model':name,'hash':'new'})
    ran=[]
    def run(argv,env):ran.append(argv);Path(argv[argv.index('--out')+1]).mkdir(parents=True)
    monkeypatch.setattr(ops,'run',run)
    out=ops.paths(tmp_path,'main')/'eval'/'baseline-general-v1-2048';out.mkdir(parents=True)
    (out/'forgetting.json').write_text('{}');(out/'checkpoint-identity.json').write_text('{"hash": "old"}')
    ops.score(tmp_path,'main','baseline','general-v1',2048)
    assert len(ran)==1
    kept=tmp_path/'campaign/v4-main/baseline-general-v1-2048/attempt-1/scoring-output'
    assert json.loads((kept/'checkpoint-identity.json').read_text())=={'hash':'old'}
