"""Frozen runtime resources shipped with the kit; evidence copies are retained privately."""
import hashlib
from pathlib import Path
RESOURCE_HASHES = {'v4-heldout-prompt-identity.json': 'f8186f8ce484b7e103b05d5140120e84bc13bfb5d8b4034c6c5bda878650ccd6', 'k8b-chemistry-templates.json': 'ade0870f8facd139924ead94c293c5655698383637f088e0b805cdf89011d17d', 'frontier-yardstick-chemistry-openai/responses-chemistry.jsonl': '0312e2eafcd55ba5abba12c2145b1d7d0dc53c9ebff220d854f90f2fa28a23fc', 'v4-data-audit-TOKEN80.ids': '851651880b32c2b666e1e3c74713fdca9bb63130b7cc014416ea9c38382d23d0', 'v4-data-audit-CLEAN.ids': 'c33a937fde3db60fc758de9f21493957e2ca81b5e09e5c290b4e44adc21e2698', 'v4-data-audit-REACTION.ids': 'e858cbd1a945c84c9a3f080950695c5dd406f10bb466f8ba5fc7182d3c6814ca', 'v4-data-audit-NONREACTION.ids': '4cfd71d54b945eba5bf4b84566a399447d79b75e6335a49b5fb1e1e08b775ab7'}


def resource(name):
    """Return a kit-owned input only after checking its registered frozen byte hash."""
    if name not in RESOURCE_HASHES:raise ValueError('unknown frozen resource: '+name)
    path=Path(__file__).resolve().parent/'resources'/name
    if hashlib.sha256(path.read_bytes()).hexdigest()!=RESOURCE_HASHES[name]:
        raise ValueError('frozen resource hash changed: '+name)
    return path
