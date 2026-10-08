"""kit/collect.py: one small archive back from the partner, with a manifest, and never the weights. Pure CPU."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("kit_collect", ROOT / "kit" / "collect.py")
collect = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(collect)


def write(path: Path, content="x", size: int | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if size is not None:
        with path.open("wb") as handle:
            handle.truncate(size)                                        # sparse: large on paper, free on disk
    else:
        path.write_text(content if isinstance(content, str) else json.dumps(content))
    return path


@pytest.fixture()
def work(tmp_path):
    """A WORK tree as a partner's cluster leaves it: results, a run with its checkpoint, a merged model, runner records."""
    w = tmp_path / "work"
    write(w / "k3" / "report-a1" / "k3-report.json", {"verdict": "PASS"})
    write(w / "k3" / "report-a1" / "k3-report.md", "# report")
    write(w / "k3" / "eval" / "a-seed0-a1" / "responses.jsonl", '{"id": 1}\n')
    write(w / "k3" / "forgetting" / "base-a1" / "forgetting.json", {"total_correct": 7})
    write(w / "k3" / "eval" / "a-seed0-a1" / "plot.png", "png")                  # not text: left out
    write(w / "runs" / "a-seed0" / "metrics.jsonl", '{"step": 1}\n')
    write(w / "runs" / "a-seed0" / "train-summary.json", {"steps": 20})
    write(w / "runs" / "a-seed0" / "fold.json", {"ok": True})
    write(w / "runs" / "a-seed0" / "merge.log", "merged")
    write(w / "runs" / "a-seed0" / "train" / "global_step_20" / "actor" / "model.pt", size=1024)
    write(w / "runs" / "a-seed0" / "hf-step20" / "config.json", {"arch": "qwen"})
    write(w / "runs" / "a-seed0" / "hf-step20" / "model.safetensors", size=1024)
    write(w / "runs" / "a-seed0" / "checkpoint-20" / "trainer_state.json", {})
    write(w / "models" / "Qwen3-1.7B" / "model-00001.safetensors", size=2048)
    write(w / "models" / "Qwen3-1.7B" / "pytorch_model.bin", size=2048)
    write(w / "data" / "spider" / "train.parquet", size=6 * collect.MB)
    write(w / "k3" / "report-a1" / "big.parquet", size=6 * collect.MB)
    write(w / "k3" / "report-a1" / "weights.bin", size=10)
    rows = w / "campaign" / "k3-replay"
    write(rows / "a-seed0" / "prepare-1.json", {"verdict": "PASS"})
    write(rows / "a-seed0" / "attempt-1" / "start.json", {"row": "a-seed0"})
    write(rows / "a-seed0" / "attempt-1" / "verdict.json", {"verdict": "PASS"})
    write(rows / "a-seed0" / "attempt-1" / "output.log", "long training log")
    write(rows / "b-none" / "attempt-1" / "start.json", {"row": "b-none"})
    write(rows / "b-none" / "attempt-1" / "verdict.json", {"verdict": "FAIL"})
    write(rows / "b-none" / "attempt-1" / "output.log", "Traceback: out of memory")
    return w


INCLUDED = {
    "k3/report-a1/k3-report.json", "k3/report-a1/k3-report.md", "k3/eval/a-seed0-a1/responses.jsonl",
    "k3/forgetting/base-a1/forgetting.json", "runs/a-seed0/metrics.jsonl", "runs/a-seed0/train-summary.json",
    "runs/a-seed0/fold.json", "runs/a-seed0/merge.log",
    "campaign/k3-replay/a-seed0/prepare-1.json", "campaign/k3-replay/a-seed0/attempt-1/start.json",
    "campaign/k3-replay/a-seed0/attempt-1/verdict.json", "campaign/k3-replay/b-none/attempt-1/start.json",
    "campaign/k3-replay/b-none/attempt-1/verdict.json", "campaign/k3-replay/b-none/attempt-1/output.log",
}


def unpack(archive: Path) -> tuple:
    with tarfile.open(archive) as tar:
        members = {m.name: tar.extractfile(m).read() for m in tar.getmembers() if m.isfile()}
    top = archive.name[:-len(".tar.gz")]
    manifest = json.loads(members.pop("%s/collect.manifest.json" % top))
    members = {name[len(top) + 1:]: data for name, data in members.items()}
    snapshot = {name: data for name, data in members.items() if name.startswith("kit-snapshot/")}
    assert set(snapshot) == set(manifest["kit_snapshot"]["files"]), "the analysis source is packed and listed (round 2, finding 9)"
    return {name: data for name, data in members.items() if not name.startswith("kit-snapshot/")}, manifest


def test_results_go_in_and_weights_checkpoints_and_big_files_stay_out(work, tmp_path, capsys):
    out = tmp_path / "k3-return.tar.gz"
    assert collect.main(["--work", str(work), "--out", str(out)]) == collect.EXIT_OK
    members, manifest = unpack(out)
    assert set(members) == INCLUDED
    assert not any(name.endswith((".safetensors", ".bin", ".pt", ".parquet", ".png")) or "hf-step" in name
                   or "checkpoint" in name or "global_step" in name for name in members)
    assert "campaign/k3-replay/a-seed0/attempt-1/output.log" not in members, "a PASS row's log is not needed"
    left_out = {entry["path"]: entry["reason"] for entry in manifest["excluded"]}
    assert left_out["runs/a-seed0/hf-step20/"] == left_out["runs/a-seed0/checkpoint-20/"] == "checkpoint or model folder"
    assert left_out["data/spider/train.parquet"] == "parquet over 5 MB"
    assert left_out["models/Qwen3-1.7B/pytorch_model.bin"].startswith("model weights")
    stdout = capsys.readouterr().out
    assert "ARCHIVE %s" % out.resolve() in stdout and "MB)" in stdout


def test_the_manifest_hashes_match_the_archived_files(work, tmp_path):
    out = tmp_path / "back.tar.gz"
    assert collect.main(["--work", str(work), "--out", str(out)]) == collect.EXIT_OK
    members, manifest = unpack(out)
    snapshot = len(manifest["kit_snapshot"]["files"])
    assert manifest["schema"] == collect.SCHEMA and manifest["file_count"] == len(members) + snapshot and len(members) == len(INCLUDED)
    for entry in manifest["files"]:
        if entry["path"].startswith("kit-snapshot/"):
            continue
        data = members[entry["path"]]
        assert entry["bytes"] == len(data) and entry["sha256"] == hashlib.sha256(data).hexdigest()
        assert entry["sha256"] == hashlib.sha256((work / entry["path"]).read_bytes()).hexdigest()
    assert manifest["total_bytes"] == sum(len(d) for d in members.values()) + sum(
        e["bytes"] for e in manifest["files"] if e["path"].startswith("kit-snapshot/"))


def test_an_existing_archive_is_never_overwritten(work, tmp_path, capsys):
    out = tmp_path / "back.tar.gz"
    out.write_bytes(b"the partner's earlier archive")
    assert collect.main(["--work", str(work), "--out", str(out)]) == collect.EXIT_REFUSED
    assert out.read_bytes() == b"the partner's earlier archive"
    assert "already exists; nothing is overwritten" in capsys.readouterr().err


def test_campaign_limits_the_archive_to_the_folders_it_writes(work, tmp_path):
    other = write(work / "k9" / "report-a1" / "k9-report.json", {"verdict": "PASS"})
    write(work / "campaign" / "k9" / "r" / "attempt-1" / "verdict.json", {"verdict": "PASS"})
    campaign = write(tmp_path / "k3.yaml", "schema: kit-campaign.v1\nname: k3-replay\nrows:\n"
                     "  - id: a\n    command: [\"x\", \"{work}/k3/eval/a\", \"{work}/runs/a-seed0\"]\n")
    out = tmp_path / "k3-only.tar.gz"
    assert collect.main(["--work", str(work), "--out", str(out), "--campaign", str(campaign)]) == collect.EXIT_OK
    members, manifest = unpack(out)
    assert set(members) == INCLUDED and other.relative_to(work).as_posix() not in members
    assert manifest["campaigns"][0]["name"] == "k3-replay"
    assert manifest["campaigns"][0]["sha256"] == hashlib.sha256(campaign.read_bytes()).hexdigest()


def test_symlinks_are_listed_not_followed(work, tmp_path):
    (work / "k2").mkdir()
    (work / "k2" / "q3-original-a1").mkdir()
    write(work / "k2" / "q3-original-a1" / "forgetting.json", {"total_correct": 1})
    (work / "k2" / "spider60-original-a1").symlink_to(work / "k2" / "q3-original-a1")
    out = tmp_path / "links.tar.gz"
    assert collect.main(["--work", str(work), "--out", str(out)]) == collect.EXIT_OK
    _, manifest = unpack(out)
    assert {"path": "k2/spider60-original-a1", "target": str(work / "k2" / "q3-original-a1")} in manifest["symlinks"]


def test_an_empty_or_missing_work_is_refused_and_writes_nothing(tmp_path, capsys):
    out = tmp_path / "none.tar.gz"
    assert collect.main(["--work", str(tmp_path / "absent"), "--out", str(out)]) == collect.EXIT_REFUSED
    (tmp_path / "empty").mkdir()
    assert collect.main(["--work", str(tmp_path / "empty"), "--out", str(out)]) == collect.EXIT_FAILED
    assert not out.exists()
    assert "NOTHING TO COLLECT" in capsys.readouterr().err


def test_a_runs_identity_files_are_collected_and_its_environment_dump_is_not():
    """The exact command and the trainer's commit tie a returned number to a recipe (plan v3: four recipes that differ
    only in a few command values). They live in runs/<run>/env/."""
    for name in ("argv.txt", "sdpo-commit.txt", "sdpo-dirty.txt", "started-at.txt", "finished-at.txt", "data-sha256.txt", "model-files.txt",
                 "trainer-started-at.txt", "stage.json"):          # round-6 ruling J3: the launcher's own stage record
        assert collect.wanted(Path("runs/g8-chem-r1-a1/env") / name, Path(".")), name
    for name in ("pip-freeze.txt", "nvidia-smi.txt"):
        assert not collect.wanted(Path("runs/g8-chem-r1-a1/env") / name, Path(".")), name
    assert not collect.wanted(Path("elsewhere/env/argv.txt"), Path("."))
    assert collect.wanted(Path("k8b/report-rollouts/g8-chem-r1-a1/rollout-stats.json"), Path("."))
    assert not collect.wanted(Path("runs/g8-chem-r1-a1/rollouts/3.jsonl"), Path(".")), "raw rollouts stay on his disk"
    assert not collect.wanted(Path("runs/g8-chem-r1-a1/validation/20.jsonl"), Path("."))


def test_a_wanted_file_over_the_limit_is_named_and_its_end_is_packed(tmp_path, capsys, monkeypatch):
    """A failed row's log is the one file we ask for when something breaks. Over 50 MB it used to be dropped while the
    collector still printed success (found by the review of send 4)."""
    monkeypatch.setattr(collect, "FILE_LIMIT", 1000)
    monkeypatch.setattr(collect, "TAIL_BYTES", 100)
    work = tmp_path / "work"
    attempt = work / "campaign" / "c" / "row" / "attempt-1"
    attempt.mkdir(parents=True)
    (attempt / "verdict.json").write_text(json.dumps({"verdict": "FAIL"}))
    (attempt / "start.json").write_text("{}")
    log = "".join("line %06d\n" % n for n in range(400))                         # 4,400 bytes
    (attempt / "output.log").write_text(log)
    out = tmp_path / "back.tar.gz"
    assert collect.main(["--work", str(work), "--out", str(out)]) == collect.EXIT_OK
    members, manifest = unpack(out)
    assert "campaign/c/row/attempt-1/output.log" not in members
    assert members["campaign/c/row/attempt-1/output.log.tail"] == log[-100:].encode()
    (entry,) = manifest["truncated"]
    assert entry["path"] == "campaign/c/row/attempt-1/output.log" and entry["bytes"] == len(log) and entry["packed_bytes"] == 100
    assert entry["sha256"] == collect.sha256(attempt / "output.log")
    assert "INCOMPLETE: campaign/c/row/attempt-1/output.log" in capsys.readouterr().out


def test_package_4s_selection_reservation_and_ledger_are_collected():
    """Found by the package-4 dry run (scripts/simulate_p4.py): kit/p4_select.py writes WORK/k8b4/selection/... and
    kit/p4_budget.py WORK/k8b4/budget/ (the reservation and the GPU-hour ledger); without them the report cannot be
    recomputed from the archive (registration section 5) and the ledger of every launch decision is lost."""
    for name in ("k8b4/selection/k8b-p4-sema-chemtool-a1/selection.json", "k8b4/selection/k8b-p4-sema-chemtool-a1/check.json",
                 "k8b4/budget/ledger.jsonl", "k8b4/budget/reservation-k8b-p4-sema-chemtool-a1.json"):
        assert collect.wanted(Path(name), Path(".")), name
    assert not collect.wanted(Path("k8b4/selection/model.safetensors"), Path("."))


def test_the_archive_carries_the_reduced_rollouts_the_resolved_configurations_and_the_analysis_source(tmp_path):
    """Round 2, finding 9 / amendment 3 B8: the archive lacked the rollout evidence, the resolved configurations and the
    exact analysis source, so the report could be recomputed only with the checkout's tools."""
    work = tmp_path / "w"
    write(work / "k8b4" / "report-rollouts" / "p4-r1-gate1-a1" / "rollout-rows.jsonl.gz", "gz")
    write(work / "k8b4" / "report-rollouts" / "p4-r1-gate1-a1" / "other.gz", "gz")
    write(work / "runs" / "p4-r1-gate1-a1" / "env" / "resolved-config.yaml", "a: 1\n")
    write(work / "runs" / "p4-r1-gate1-a1" / "rollouts" / "1.jsonl", "{}")
    write(work / "k8b4" / "recipe-check" / "a1" / "proposed" / "p4-r1-ctl1.resolved.yaml", "a: 1\n")
    write(work / "k8b4" / "recipe-check" / "baseline.json", {"ok": True})
    pilot = collect.HERE / "campaigns" / "k8b-pilot.yaml"
    digest = hashlib.sha256(pilot.read_text().encode()).hexdigest()
    write(work / "campaign" / "k8b-pilot" / "r" / "attempt-1" / "start.json", {"campaign_sha256": digest})
    write(work / "campaign" / "not-a-kit-campaign" / "r" / "attempt-1" / "start.json", {"campaign_sha256": "x"})
    out = tmp_path / "send5-k8b4.tar.gz"
    assert collect.main(["--work", str(work), "--out", str(out)]) == collect.EXIT_OK
    with tarfile.open(out) as tar:
        names = {m.name.split("/", 1)[1]: m for m in tar.getmembers() if m.isfile()}
    for name in ("k8b4/report-rollouts/p4-r1-gate1-a1/rollout-rows.jsonl.gz", "runs/p4-r1-gate1-a1/env/resolved-config.yaml",
                 "k8b4/recipe-check/a1/proposed/p4-r1-ctl1.resolved.yaml", "k8b4/recipe-check/baseline.json",
                 "kit-snapshot/p4_report.py", "kit-snapshot/p4_intervals.py", "kit-snapshot/pilot_report.py", "kit-snapshot/runner.py",
                 "kit-snapshot/p4_recipe.py", "kit-snapshot/budget_report.py", "kit-snapshot/campaigns/k8b-pilot.yaml"):
        assert name in names, name
    assert "k8b4/report-rollouts/p4-r1-gate1-a1/other.gz" not in names and "runs/p4-r1-gate1-a1/rollouts/1.jsonl" not in names
    with tarfile.open(out) as tar:
        manifest = json.loads(tar.extractfile("send5-k8b4/collect.manifest.json").read())
    listed = {e["path"]: e["sha256"] for e in manifest["files"]}
    assert listed["kit-snapshot/p4_report.py"] == hashlib.sha256((collect.HERE / "p4_report.py").read_bytes()).hexdigest()
    campaigns = {c["name"]: c for c in manifest["kit_snapshot"]["campaigns"]}
    assert campaigns["k8b-pilot"]["matches_every_run"] is True and campaigns["not-a-kit-campaign"]["file"] is None


def test_collector_keeps_successful_runner_dispatch_timing(work,tmp_path):
    path=work/'campaign/v4-main/cpu/attempt-1'
    write(path/'verdict.json',{'verdict':'PASS'})
    write(path/'runner-overhead.json',{'ok':True,'wall_seconds':.01,'command_seconds':1})
    out=tmp_path/'overhead.tar.gz'
    assert collect.main(['--work',str(work),'--out',str(out)])==collect.EXIT_OK
    members,manifest=unpack(out)
    assert json.loads(members['campaign/v4-main/cpu/attempt-1/runner-overhead.json'])['wall_seconds']==.01
