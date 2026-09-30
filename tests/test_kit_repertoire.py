"""kit/repertoire.py: the style of a set of SQL answers, and how far training moved it.

Receipt 238 found GRPO on Spider collapses the model's SQL from JOINs to nested subqueries while the
accuracy count moves little, so the kit reads the answers themselves. Checked here: the feature set
and its regexes are scripts/analyse_spider_flips.py's (less quoting); the SQL is taken from an answer
exactly as kit/beds/spider.py takes it; the shares and the shift are right on crafted answers; a
scoring with no responses.jsonl is unknown and never a refusal; and the CLI prints the table.
Everything builds its own fixtures; nothing here needs a GPU, a network or a Spider copy.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
KIT = ROOT / "kit"


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


repertoire = load("kit_repertoire", KIT / "repertoire.py")
spider = load("kit_spider_for_repertoire", KIT / "beds" / "spider.py")


def fenced(sql: str) -> str:
    return "Some thinking first.\n```sql\n%s;\n```" % sql


JOIN_ANSWER = fenced("SELECT T1.name FROM singer AS T1 JOIN concert AS T2 ON T1.id = T2.singer_id")
SUB_ANSWER = fenced("SELECT name FROM singer WHERE id IN (SELECT singer_id FROM concert)")
PLAIN_ANSWER = fenced("SELECT count(*) FROM singer")
NO_SQL = "I cannot answer that."


def _scoring(path: Path, answers: list | None) -> Path:
    path.mkdir(parents=True)
    (path / "bed-score.json").write_text(json.dumps({"schema": "kit-bed-score.v1", "n": 100}), encoding="utf-8")
    if answers is not None:
        (path / "responses.jsonl").write_text("".join(
            json.dumps({"bed": "spider", "id": "q%d" % i, "response": text, "output_tokens": 10}) + "\n"
            for i, text in enumerate(answers)), encoding="utf-8")
    return path


@pytest.mark.skipif(not (Path(__file__).resolve().parents[1] / "scripts" / "analyse_spider_flips.py").is_file(), reason="scripts/ is not part of the exported kit")
def test_the_feature_set_is_the_flip_analysis_set_less_quoting():
    flips = (ROOT / "scripts" / "analyse_spider_flips.py").read_text()
    assert list(repertoire.FEATURES) == [
        "JOIN", "subquery", "WHERE", "GROUP BY", "HAVING", "ORDER BY", "LIMIT", "DISTINCT", "COUNT(",
        "AVG/SUM/MIN/MAX(", "LIKE", "UNION/INTERSECT/EXCEPT", "alias AS", "SELECT *"]
    for pattern in repertoire.FEATURES.values():
        assert 'r"%s"' % pattern in flips, pattern
    assert "double-quoted" not in " ".join(repertoire.FEATURES)


@pytest.mark.parametrize("text", [JOIN_ANSWER, SUB_ANSWER, NO_SQL, "SELECT a FROM b; SELECT c FROM d",
                                  "```\nselect 1\n```", "```sql\nexplain nothing\n``` then SELECT x FROM y",
                                  "", None])
def test_the_sql_is_taken_from_an_answer_exactly_as_the_spider_bed_takes_it(text):
    assert repertoire.extract_sql(text) == spider.extract_sql(text)


def test_join_detection_is_a_whole_word_case_insensitive():
    shares = repertoire.features([
        fenced("select a from t1 inner join t2 on t1.x = t2.x"),        # JOIN
        fenced("SELECT a FROM t1 LEFT JOIN t2 ON t1.x = t2.x"),          # JOIN
        fenced("SELECT joined_at FROM rejoin_log"),                       # not a JOIN: part of a word
        "We should join the tables, but here is the query:\n```sql\nSELECT a FROM t1\n```"])  # prose only
    assert shares["JOIN"] == 50.0


def test_features_are_shares_of_every_answer_including_those_with_no_sql():
    shares = repertoire.features([JOIN_ANSWER, SUB_ANSWER, PLAIN_ANSWER, NO_SQL])
    assert shares["JOIN"] == 25.0 and shares["subquery"] == 25.0
    assert shares["WHERE"] == 25.0 and shares["COUNT("] == 25.0 and shares["alias AS"] == 25.0
    assert shares["GROUP BY"] == 0.0 and shares["SELECT *"] == 0.0
    assert set(shares) == set(repertoire.FEATURES)
    # a responses.jsonl row is read by its `response`, exactly as the text alone
    assert repertoire.features([{"response": JOIN_ANSWER}, {"response": NO_SQL}])["JOIN"] == 50.0
    assert repertoire.features([]) is None and repertoire.features(None) is None


def test_shift_is_the_change_per_feature_and_the_mean_absolute_change():
    before = repertoire.features([JOIN_ANSWER, JOIN_ANSWER, PLAIN_ANSWER, PLAIN_ANSWER])
    after = repertoire.features([SUB_ANSWER, SUB_ANSWER, SUB_ANSWER, PLAIN_ANSWER])
    moved = repertoire.shift(before, after)
    assert moved["join_reference"] == 50.0 and moved["join_trained"] == 0.0
    assert moved["change"]["JOIN"] == -50.0 and moved["change"]["subquery"] == 75.0
    assert moved["change"]["WHERE"] == 75.0 and moved["change"]["alias AS"] == -50.0
    assert moved["change"]["COUNT("] == -25.0
    expected = sum(abs(v) for v in moved["change"].values()) / len(repertoire.FEATURES)
    assert moved["style_shift_points"] == pytest.approx(expected, abs=0.01)
    assert moved["style_shift_points"] == pytest.approx((50 + 75 + 75 + 50 + 25) / 14, abs=0.01)
    assert repertoire.shift(before, before)["style_shift_points"] == 0.0


def test_a_missing_side_is_unknown_not_a_refusal(tmp_path):
    known = repertoire.features([JOIN_ANSWER])
    moved = repertoire.shift(known, None)
    assert moved["style_shift_points"] is None and moved["join_reference"] == 100.0
    assert moved["join_trained"] is None and set(moved["change"].values()) == {None}
    assert repertoire.of_scoring(_scoring(tmp_path / "no-answers", None)) is None
    assert repertoire.read_responses(tmp_path / "nowhere") is None
    folder = _scoring(tmp_path / "answers", [JOIN_ANSWER, SUB_ANSWER])
    assert repertoire.of_scoring(folder)["JOIN"] == 50.0
    assert repertoire.of_scoring(folder / "bed-score.json")["JOIN"] == 50.0, "the result file names its folder"


def _cli(*args) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(KIT / "repertoire.py"), *map(str, args)],
                          capture_output=True, text=True)


def test_the_cli_prints_one_column_per_scoring_and_the_change(tmp_path):
    base = _scoring(tmp_path / "base-spider-a1", [JOIN_ANSWER, JOIN_ANSWER, PLAIN_ANSWER, PLAIN_ANSWER])
    trained = _scoring(tmp_path / "ref20-r1-spider-a1", [SUB_ANSWER, SUB_ANSWER, SUB_ANSWER, PLAIN_ANSWER])
    done = _cli("compare", "--reference", base, "--trained", trained)
    assert done.returncode == 0, done.stderr
    lines = done.stdout.split("\n")
    assert lines[0] == "| SQL feature (% of answers) | base-spider-a1 | ref20-r1-spider-a1 |"
    assert "| answers read | 4 | 4 |" in lines
    assert "| JOIN | 50 | 0 (-50) |" in lines
    assert "| subquery | 0 | 75 (+75) |" in lines
    assert "| HAVING | 0 | 0 (0) |" in lines
    assert "| **style shift** (mean absolute change, points) | - | **19.6** |" in lines


def test_the_cli_reports_a_scoring_without_answers_as_unknown_and_exits_zero(tmp_path):
    base = _scoring(tmp_path / "base-spider-a1", [JOIN_ANSWER, PLAIN_ANSWER])
    trained = _scoring(tmp_path / "ref20-spider-a1", [SUB_ANSWER, PLAIN_ANSWER])
    blind = _scoring(tmp_path / "steps60-spider-a1", None)
    done = _cli("compare", "--reference", base, "--trained", trained, "--trained", blind)
    assert done.returncode == 0, done.stderr
    assert "| answers read | 2 | 2 | unknown |" in done.stdout
    assert "| JOIN | 50 | 0 (-50) | unknown |" in done.stdout
    assert "| **style shift** (mean absolute change, points) | - | **" in done.stdout
    assert "- steps60-spider-a1: no responses.jsonl beside bed-score.json" in done.stdout
    # an unknown REFERENCE leaves every shift unknown and still exits 0
    done = _cli("compare", "--reference", blind, "--trained", trained)
    assert done.returncode == 0 and "| JOIN | unknown | 0 |" in done.stdout
    assert "| **style shift** (mean absolute change, points) | - | unknown |" in done.stdout


def test_the_cli_writes_json_once_and_never_overwrites(tmp_path):
    base = _scoring(tmp_path / "b", [JOIN_ANSWER])
    trained = _scoring(tmp_path / "t", [SUB_ANSWER])
    out = tmp_path / "compare.json"
    assert _cli("compare", "--reference", base, "--trained", trained, "--out", out).returncode == 0
    written = json.loads(out.read_text())
    assert written["schema"] == "kit-repertoire.v1"
    assert [c["role"] for c in written["columns"]] == ["reference", "trained"]
    assert written["columns"][1]["shift"]["join_trained"] == 0.0
    again = _cli("compare", "--reference", base, "--trained", trained, "--out", out)
    assert again.returncode == 2 and "refusing to overwrite" in again.stderr


EVIDENCE = ROOT / "docs" / "phase2" / "evidence" / "k3dose-partner" / "eval"


@pytest.mark.skipif(not (EVIDENCE / "base-spider-a1" / "responses.jsonl").is_file(),
                    reason="the partner's probe-1 scorings are not in this checkout")
def test_on_probe_1s_returned_answers_it_reads_the_collapse_receipt_238_found():
    """docs/phase2/analysis/spider-flips-probe1.md: JOIN 41 untrained, 7 after ref20, 0 after steps60."""
    base = repertoire.of_scoring(EVIDENCE / "base-spider-a1")
    assert base["JOIN"] == 41.0 and base["subquery"] == 26.0
    ref20 = repertoire.shift(base, repertoire.of_scoring(EVIDENCE / "ref20-spider-a1"))
    assert ref20["join_trained"] == 7.0 and ref20["change"]["subquery"] == 25.0
    assert repertoire.of_scoring(EVIDENCE / "steps60-spider-a1")["JOIN"] == 0.0
