"""Shared pieces of the three package smokes (K4, the coding bed, K5): the partner's image, a step runner, a record
writer. Kept small and boring on purpose."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import modal

# This helper is also imported directly from scripts/ by host launchers.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from kit.smoke_steps import Steps
IMAGE = "nvcr.io/nvidia/vllm:25.12.post1-py3"
SDPO_COMMIT = "7c457fc1b1f636ae794eb0362ba37d4743b06fbc"
GPU, USD_PER_HOUR = "H100", 3.95
MODEL = "Qwen/Qwen3-1.7B"


def partner_image(extra_dirs: dict):
    image = (modal.Image.from_registry(IMAGE, add_python=None)
             .run_commands("git clone https://github.com/lasgroup/SDPO.git /root/SDPO && git -C /root/SDPO checkout " + SDPO_COMMIT,
                           "pip install -e /root/SDPO",
                           # Fresh installs: pyarrow 26 requires NumPy 2; newer datasets needs a newer hub.
                           "pip install 'pyarrow==21.0.0' 'datasets==4.8.5'",
                           "pip install 'antlr4-python3-runtime==4.9.3' 'math-verify==0.8.0' 'ray==2.53.0' 'torchdata==0.11.0' 'transformers==4.57.1'",
                           "pip install -U 'typing_extensions>=4.14'",
                           "pip install rapidfuzz==3.14.6",
                           "pip install langdetect==1.0.9",
                           # the pinned trainer's SFT path imports it (verl/utils/dataset/vision_utils.py); setup.py lists it only
                           # under the optional "geo" extra, so a plain `pip install -e SDPO` lacks it (found by the K2b smoke)
                           "pip install qwen-vl-utils"))
    for local, remote in extra_dirs.items():
        image = image.add_local_dir(str(ROOT / local), remote, copy=True)
    # this helper itself, so the entry script can import it inside the container (Modal ships only the entry file)
    image = image.add_local_file(str(ROOT / "scripts/smoke_common_modal.py"), "/root/smoke_common_modal.py", copy=True)
    return image.add_local_dir(str(ROOT / "kit"), "/root/kit", copy=True)


def write_record(record: dict, stem: str, *, hourly_usd=USD_PER_HOUR, attempt=None) -> Path:
    record["cost_bound_usd"] = round(record["seconds_in_function"] / 3600 * hourly_usd * 1.25 + 0.05, 2)
    out = ROOT / "docs/phase2/evidence"
    n = attempt if attempt is not None else len(list(out.glob(stem + ".attempt*.json"))) + 1
    path = out / ("%s.attempt%d.json" % (stem, n))
    with path.open('x' if attempt is not None else 'w') as f:
        f.write(json.dumps(record, indent=1, sort_keys=True))
    for s in record["steps"]:
        print("%-28s -> %d (%6.1fs)" % (s["name"], s["returncode"], s["seconds"]))
        if s["returncode"]:
            for line in s["errors"] + s["tail"]: print("     ", line)
    return path
