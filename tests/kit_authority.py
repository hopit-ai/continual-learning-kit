"""Explicit operator-seal fixture shared by standalone exported legacy tests."""
import hashlib
from pathlib import Path

import pytest


@pytest.fixture
def sealed_v6_authority(monkeypatch):
    from kit import v6_infra
    path = Path(__file__).resolve().parents[1] / v6_infra.AUTHORITY
    seal = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setenv('KIT_V6A', '1')
    monkeypatch.setenv('KIT_V6_AUTHORITY_SHA', seal)
    v6_infra.activate_v6a()
    return seal
