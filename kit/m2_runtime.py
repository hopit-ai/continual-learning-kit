"""Prospective, exact-byte infrastructure updates to inherited checker entry points.

Historical science/data registrations are retained. Only the explicitly
listed infrastructure updates can satisfy their original implementation pins.
"""
import json
from pathlib import Path


def allowed_update(path, expected, observed, authority_sha256=None):
    root=Path(__file__).resolve().parents[1]
    try:name=str(Path(path).resolve().relative_to(root))
    except ValueError:return False
    updates=json.loads((root/'kit/resources/m2/runtime-updates.json').read_text())
    value=updates['files'].get(name,{})
    if expected==value.get('original_sha256') and observed==value.get('v2_sha256'):return True
    from kit.v6_infra import allowed
    return allowed(path,expected,observed,authority_sha256)
