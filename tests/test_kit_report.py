"""kit/make_report.py on a synthetic run: the arithmetic, the flags and the crash path."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("kit_make_report", ROOT / "kit" / "make_report.py")
report = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(report)


def _run(tmp_path: Path, name: str, base: float, accs, steps: int, total: int, tokens: float = 90.0) -> Path:
    run = tmp_path / "runs" / name
    (run / "env").mkdir(parents=True)
    (run / "validation").mkdir()
    (run / "env" / "argv.txt").write_text("trainer.total_training_steps=%d\ndata.seed=7\n" % total)
    rows = [{"step": 0, "data": {report.VAL_KEY: base}}]
    rows += [{"step": s, "data": {"critic/score/mean": 0.5, "response_length/mean": tokens,
                                  "self_distillation/success_group_fraction": 0.4, "actor/grad_norm": 1.0}}
             for s in range(1, steps + 1)]
    if steps == total:
        rows.append({"step": total, "data": {report.VAL_KEY: sum(accs) / len(accs)}})
    (run / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))

    def dump(step, values):
        lines = [json.dumps({"input": "q%d" % i, "output": "x" * 10, "acc": a, "incorrect_format": 0})
                 for i, a in enumerate(values) for _ in range(2)]
        (run / "validation" / ("%d.jsonl" % step)).write_text("\n".join(lines) + "\n")
    dump(0, [base] * len(accs))
    if steps == total:
        dump(total, accs)
    return run


def test_paired_change_and_no_flags(tmp_path):
    _run(tmp_path, "good", base=0.575, accs=[0.575, 0.775, 0.575, 0.375], steps=3, total=3)
    built = report.build(tmp_path / "runs")
    run = built["runs"][0]
    assert run["complete"] and run["flags"] == []
    pc = run["paired_change"]
    assert (pc["better"], pc["worse"], pc["unchanged"]) == (1, 1, 2) and pc["mean_change"] == 0.0
    assert run["identity"]["seed"] == "7" and built["by_dose"]["3"]["runs"] == 1
    assert "| good | 7 | 3 of 3 |" in report.render(built)


def test_flags_calibration_length_and_incomplete(tmp_path):
    _run(tmp_path, "bad", base=0.40, accs=[0.4, 0.4], steps=2, total=5, tokens=8.0)
    (tmp_path / "runs" / "bad" / "console.log").write_text("Traceback: CUDA out of memory\n")
    run = report.build(tmp_path / "runs")["runs"][0]
    text = " ".join(run["flags"])
    assert not run["complete"]
    assert "CALIBRATION" in text and "LENGTH" in text and "INCOMPLETE: 2 of 5" in text
    assert "out of memory" in run["console_tail"]
