"""kit/density.py: tokens per correct answer, the bar, and the training-length trend (plan 4c, Q13)."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

KIT = Path(__file__).resolve().parents[1] / "kit"
_spec = importlib.util.spec_from_file_location("density", KIT / "density.py")
density = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(density)


def bed(tmp_path: Path, name: str, correct: int, *, total: int | None = None, rows: list | None = None) -> tuple:
    d = tmp_path / name; d.mkdir(parents=True, exist_ok=True)
    result = {"bed": "spider", "n": 100, "correct": correct}
    if total is not None:
        result["output_tokens_total"] = total
    path = d / "bed-score.json"; path.write_text(json.dumps(result))
    if rows is not None:
        (d / "responses.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return result, path


def test_tokens_per_correct_is_the_sets_output_tokens_over_its_correct_answers(tmp_path):
    result, path = bed(tmp_path, "a", 20, total=2000)
    cell = density.tokens_per_correct(result, path)
    assert cell["tokens_per_correct"] == 100 and cell["source"] == "result" and cell["n"] == 100


def test_the_result_wins_over_a_responses_file_beside_it(tmp_path):
    result, path = bed(tmp_path, "a", 20, total=2000, rows=[{"id": i, "output_tokens": 999} for i in range(100)])
    assert density.tokens_per_correct(result, path)["tokens_per_correct"] == 100


def test_a_responses_file_is_summed_when_the_result_carries_no_total(tmp_path):
    result, path = bed(tmp_path, "a", 20, rows=[{"id": i, "output_tokens": 30} for i in range(100)])
    cell = density.tokens_per_correct(result, path)
    assert cell["total"] == 3000 and cell["tokens_per_correct"] == 150 and cell["source"] == "responses.jsonl"


def test_no_counts_anywhere_is_unknown_not_an_error(tmp_path):
    result, path = bed(tmp_path, "a", 20)
    assert density.tokens_per_correct(result, path) is None
    assert density.compare(None, None)["verdict"] == "unknown"


def test_nothing_correct_is_an_infinite_cost_reported_as_none_and_unknown_ratio(tmp_path):
    result, path = bed(tmp_path, "a", 0, total=2000)
    cell = density.tokens_per_correct(result, path)
    assert cell["tokens_per_correct"] is None and density.ratio(cell, {"tokens_per_correct": 100}) is None


def test_a_panel_of_a_forgetting_file_is_read_by_name(tmp_path):
    d = tmp_path / "f"; d.mkdir()
    result = {"panels": {"ifeval": {"n": 100, "correct": 50, "output_tokens_total": 5000}, "math": {"n": 100, "correct": 40}}}
    path = d / "forgetting.json"; path.write_text(json.dumps(result))
    (d / "responses.jsonl").write_text("".join(json.dumps({"panel": "math", "id": i, "output_tokens": 20}) + "\n" for i in range(100)))
    assert density.tokens_per_correct(result, path, panel="ifeval")["tokens_per_correct"] == 100
    assert density.tokens_per_correct(result, path, panel="math")["tokens_per_correct"] == 50


@pytest.mark.parametrize("trained,untrained,verdict", [(150, 100, "within bar"), (151, 100, "over bar"), (50, 100, "within bar")])
def test_the_bar_is_one_and_a_half_times_the_untrained_cost(trained, untrained, verdict):
    assert density.BAR == 1.5
    assert density.compare({"tokens_per_correct": trained}, {"tokens_per_correct": untrained})["verdict"] == verdict


def test_the_training_trend_names_the_verbosity_shape_only_when_length_climbs_and_the_score_does_not():
    up = [{"step": s, "response_tokens": 100 + 10 * s} for s in range(1, 41)]
    flat = [{"step": s, "response_tokens": 100} for s in range(1, 41)]
    rose = [{"step": 0, "accuracy_from_dump": 0.5}, {"step": 40, "accuracy_from_dump": 0.6}]
    fell = [{"step": 0, "accuracy_from_dump": 0.5}, {"step": 40, "accuracy_from_dump": 0.45}]
    assert density.training_trend(up, fell)["shape"].startswith("length up, score not up")
    assert density.training_trend(up, rose)["shape"] == "length up with the score"
    assert density.training_trend(flat, fell)["shape"] == "length held"
    with pytest.raises(density.DensityError):
        density.training_trend(up[:3], rose)


def test_the_cli_reads_the_k0_report_on_record_and_finds_one_run_that_lengthened_with_its_score_held():
    report = Path(__file__).resolve().parents[1] / "docs/phase2/evidence/k0-partner/report.json"
    if not report.is_file():
        pytest.skip("the K0 partner report is not in this checkout")
    out = subprocess.run([sys.executable, str(KIT / "density.py"), "training", "--report", str(report)],
                         capture_output=True, text=True, check=True).stdout
    assert "| dose40-seed42 | 40 | 110.9 | 269.5 | 2.43 |" in out and "verbosity shape" not in out


def test_the_cli_scores_result_files_against_the_first(tmp_path):
    _, base = bed(tmp_path, "base", 20, total=2000)
    _, after = bed(tmp_path, "after", 40, total=8000)
    out = subprocess.run([sys.executable, str(KIT / "density.py"), "score", str(base), str(after)],
                         capture_output=True, text=True, check=True).stdout
    assert "| 100.0 | 1.00 | within bar |" in out and "| 200.0 | 2.00 | over bar |" in out
