"""Explicit operator-seal fixture shared by standalone exported legacy tests."""
import hashlib
from pathlib import Path

import pytest


@pytest.fixture
def sealed_v6_authority(monkeypatch,tmp_path):
    from kit import v6_infra
    import json
    root=Path(__file__).resolve().parents[1]
    doc=json.loads((root/v6_infra.AUTHORITY).read_text())
    for row in doc['files']:row['new_sha256']=hashlib.sha256((root/row['path']).read_bytes()).hexdigest()
    path=tmp_path/'cpu-authority.json';path.write_text(json.dumps(doc))
    monkeypatch.setattr(v6_infra,'AUTHORITY',str(path))
    seal = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setenv('KIT_V6A', '1')
    monkeypatch.setenv('KIT_V6_AUTHORITY_SHA', seal)
    v6_infra.activate_v6a()
    return seal
