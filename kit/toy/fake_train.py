#!/usr/bin/env python3
"""A stand-in for a training run: writes metrics in the reference trainer's file-logger shape.

Used only by the toy campaign and the tests, so the runner's gate can be exercised on a laptop in
under a second. BASE is the 'untrained score' it reports at step 0; CRASH=1 makes it exit non-zero;
it refuses to run unless the runner wrote start.json first, which is how the tests prove the order.
"""
import json
import os
import sys
from pathlib import Path

out = Path(os.environ["OUT"])
start = Path(os.environ["START_RECORD"])
assert start.is_file(), "the runner must write start.json before launching: %s" % start
out.mkdir(parents=True, exist_ok=False)
base, steps = float(os.environ.get("BASE", "0.58")), int(os.environ.get("STEPS", "2"))
rows = [{"step": 0, "data": {"val/acc": base}}]
rows += [{"step": s, "data": {"train/score": 0.5 + 0.01 * s, "train/signal": 0.4}} for s in range(1, steps + 1)]
rows.append({"step": steps, "data": {"val/acc": base + 0.02 * steps}})
(out / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
print("fake_train wrote %d rows to %s" % (len(rows), out))
sys.exit(1 if os.environ.get("CRASH") == "1" else 0)
