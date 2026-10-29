"""Materialize the approved PD preparation on CPU and enforce its send-time pin.

The frozen scientific builder remains unchanged: its bytes are part of the
owner-approved preparation identity. Large outputs stay in node-local storage.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kit import pd_scientific as scientific
from kit.pd_contract import digest, is_sha, require, sha256_file, verify_seal


def load_pin(root, approval_file):
    root = Path(root)
    package = scientific.load_package(root)
    pin = scientific.read(root/"finalization-pin.json")
    verify_seal(pin)
    approval = scientific.read(approval_file)
    require(pin.get("schema") == "pd-scientific-finalization-pin-v1"
            and pin.get("package_sha256") == package["identity_sha256"]
            and pin.get("builder_sha256") == package["builder_sha256"]
            and pin.get("materializer_sha256") == sha256_file(__file__)
            and pin.get("code_sha256") == package["code_sha256"]
            and pin.get("approval_sha256") == digest(approval)
            and sha256_file(root/"topic-approval.json") == pin.get("approval_file_sha256")
            and digest(scientific.read(root/"topic-approval.json")) == digest(approval)
            and is_sha(pin.get("preparation_sha256"))
            and is_sha(pin.get("preparation_file_sha256"))
            and type(pin.get("preparation_bytes")) is int and pin["preparation_bytes"] > 0,
            "DEFERRED: approved PD finalization pin or input binding changed")
    scientific.approval_exclusions(package, scientific.read(root/"rows.json.gz"),
                                  scientific.read(root/"topic-flags.json"), approval)
    return pin


def finalize_on_node(root, approval_file, output):
    """Run pd_scientific.py finalize; publish only a matching complete result."""
    root, approval_file, output = (Path(v).resolve() for v in (root, approval_file, output))
    require(output != root and root not in output.parents,
            "large PD preparation must stay outside the shipped scientific package")
    require(not output.exists(), "fresh node preparation destination required")
    pin = load_pin(root, approval_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Stage on the destination filesystem; failed or mismatched output is never
    # published at the path used by the plan. link() cannot overwrite a race.
    with tempfile.TemporaryDirectory(prefix=".pd-finalize-", dir=output.parent) as staging:
        temporary = Path(staging)/"preparation.json"
        child = subprocess.run([sys.executable, str(Path(scientific.__file__).resolve()),
            "finalize", "--package", str(root), "--approval", str(approval_file),
            "--out", str(temporary)], capture_output=True, text=True)
        require(child.returncode == 0, "DEFERRED: frozen PD finalizer failed")
        try:
            summary = json.loads(child.stdout)
        except (ValueError, TypeError):
            raise ValueError("DEFERRED: invalid frozen PD finalizer receipt") from None
        require(isinstance(summary, dict)
                and summary.get("identity_sha256") == pin["preparation_sha256"],
                "DEFERRED: node preparation identity does not match the approved pin")
        require(temporary.is_file() and temporary.stat().st_size == pin["preparation_bytes"]
                and sha256_file(temporary) == pin["preparation_file_sha256"],
                "DEFERRED: node preparation bytes do not match the approved pin")
        # Recheck small bound inputs after the child, before publishing anything.
        require(load_pin(root, approval_file) == pin, "DEFERRED: PD inputs changed during finalization")
        os.link(temporary, output)
    return dict(status="PIN_MATCH", identity_sha256=pin["preparation_sha256"],
                file_sha256=pin["preparation_file_sha256"], bytes=pin["preparation_bytes"],
                finalization_pin_sha256=pin["identity_sha256"])


def require_plan_pin(plan):
    """Pair with scientific.require_plan before GPU discovery.

    That gate checks the complete preparation and matches it to this small
    verification receipt. Comparing the receipt avoids parsing 509 MB again.
    """
    try:
        config = plan["scientific"]
        pin = load_pin(config["package"], config["approval"])
        verified = scientific.read(config["verification"])
        verify_seal(verified)
        require(verified.get("preparation_sha256") == pin["preparation_sha256"]
                and verified.get("package_sha256") == pin["package_sha256"]
                and verified.get("approval_sha256") == pin["approval_sha256"],
                "node preparation does not match pinned approved identity")
        return pin
    except (KeyError, TypeError, ValueError, OSError):
        raise ValueError("DEFERRED: PD requires the pinned approved node preparation match") from None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", required=True)
    parser.add_argument("--approval", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    try:
        result = finalize_on_node(args.package, args.approval, args.out)
    except (KeyError, TypeError, ValueError, OSError):
        print(json.dumps(dict(status="DEFERRED", reason="PD finalization inputs or output do not match the approved pin")))
        return 3
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
