"""kit/stage_local.py: deterministic packing, verified unpacking, reuse, and the refusals."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("kit_stage_local", ROOT / "kit" / "stage_local.py")
stage = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(stage)


@pytest.fixture()
def images(tmp_path) -> Path:
    root = tmp_path / "volume"
    for index in range(40):
        path = root / ("shard%d" % (index % 4)) / ("img_%03d.bin" % index)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(hashlib.sha256(str(index).encode()).digest() * (index + 1))
    return root


def pack(root: Path, out: Path, extra=()):
    return stage.main(["pack", "--root", str(root), "--glob", "**/*.bin", "--out", str(out), *extra])


def test_the_same_inputs_always_pack_to_the_same_bytes(images, tmp_path):
    assert pack(images, tmp_path / "a.tar") == 0 and pack(images, tmp_path / "b.tar", ["--readers", "3"]) == 0
    assert (tmp_path / "a.tar").read_bytes() == (tmp_path / "b.tar").read_bytes()
    manifest = json.loads((tmp_path / "a.tar.manifest.json").read_text())
    assert manifest["file_count"] == 40 and not list(tmp_path.glob("*.partial"))


def test_unpack_verifies_and_a_second_run_reuses(images, tmp_path, capsys):
    pack(images, tmp_path / "a.tar")
    local = tmp_path / "local"
    assert stage.main(["unpack", "--archive", str(tmp_path / "a.tar"), "--to", str(local)]) == 0
    assert all((local / p.relative_to(images)).read_bytes() == p.read_bytes() for p in images.rglob("*.bin"))
    marker = json.loads((local / stage.MARKER).read_text())
    assert marker["file_count"] == 40 and {"copy_seconds", "extract_seconds", "verify_seconds"} <= set(marker)
    capsys.readouterr()
    assert stage.main(["unpack", "--archive", str(tmp_path / "a.tar"), "--to", str(local)]) == 0
    assert "already staged" in capsys.readouterr().out


def test_a_corrupted_archive_is_refused(images, tmp_path):
    pack(images, tmp_path / "a.tar")
    data = bytearray((tmp_path / "a.tar").read_bytes()); data[2000] ^= 0xFF
    (tmp_path / "a.tar").write_bytes(bytes(data))
    with pytest.raises(SystemExit, match="does not match its manifest"):
        stage.main(["unpack", "--archive", str(tmp_path / "a.tar"), "--to", str(tmp_path / "local")])


def test_a_listed_file_that_does_not_exist_stops_the_pack(images, tmp_path):
    listing = tmp_path / "files.txt"
    listing.write_text("shard0/img_000.bin\nshard9/never_written.bin\n")
    with pytest.raises(SystemExit, match="listed files do not exist"):
        stage.main(["pack", "--root", str(images), "--list", str(listing), "--out", str(tmp_path / "a.tar")])
    assert not (tmp_path / "a.tar").exists()


def test_unpack_refuses_a_directory_it_did_not_stage(images, tmp_path):
    pack(images, tmp_path / "a.tar")
    other = tmp_path / "other"; other.mkdir(); (other / "stranger.txt").write_text("x")
    with pytest.raises(SystemExit, match="non-empty directory"):
        stage.main(["unpack", "--archive", str(tmp_path / "a.tar"), "--to", str(other)])


def test_mirror_copies_large_files_with_hashes_and_reuses(tmp_path, capsys):
    source = tmp_path / "model"; source.mkdir()
    (source / "config.json").write_text("{}")
    (source / "model-00001.safetensors").write_bytes(b"w" * 3_000_000)
    local = tmp_path / "scratch"
    assert stage.main(["mirror", "--source", str(source), "--to", str(local)]) == 0
    marker = json.loads((local / stage.MARKER).read_text())
    assert marker["files"]["model-00001.safetensors"]["sha256"] == hashlib.sha256(b"w" * 3_000_000).hexdigest()
    capsys.readouterr()
    assert stage.main(["mirror", "--source", str(source), "--to", str(local)]) == 0
    assert "already mirrored" in capsys.readouterr().out
