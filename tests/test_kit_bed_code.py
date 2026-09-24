"""kit/beds/code.py and kit/sandbox.py: the sandbox's five outcomes, the refusals, and the pinned panel.

Four groups, and the first two must run anywhere:

1. The sandbox, on tiny hand-written problems: a program that passes, one that gives the wrong
   answer, one that never stops, one that crashes, and one that tries to open a socket. Both test
   forms. That the expected output never reaches the child, and that the temporary directory goes.
2. The bed, on a LiveCodeBench-shaped fixture built in `tmp_path`: code extraction, the reward
   function's shape, that its feedback carries neither the gold nor any digit, the date rule, the
   leakage filter, and that a changed prompt, changed tests, a missing panel problem, a missing
   tests file and an existing output are each refused.
3. The shipped split file read as data: 51 panel members in phase 1's order, the cut-off equal to
   the panel's earliest date, every pinned training problem before it, every functional problem
   carrying a function name. Built FROM the file, so a regenerated file is checked rather than
   assumed.
4. Parity against phase 1: the split file regenerates byte-identically from the local evidence, the
   51 prompts rebuild from the local release file, and phase 1's retained Opus 5 answers score the
   40 of 51 that receipt 205 published. These need trees that exist only on the machine that
   produced them, and skip when they are absent.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sandbox = _load("kit_sandbox", "kit/sandbox.py")
code = _load("kit_bed_code", "kit/beds/code.py")

SPLIT_FILE = ROOT / "kit" / "beds" / "code-split-v1.json"
GENERATOR = ROOT / "scripts" / "make_kit_code_split.py"
import subprocess as _sp
_COMMON = _sp.run(["git", "rev-parse", "--path-format=absolute", "--git-common-dir"], capture_output=True, text=True)
PHASE1 = (Path(_COMMON.stdout.strip()).parent if _COMMON.returncode == 0 else ROOT) / "work" / "outputs" / "livecodebench-one-update-readiness"
SOURCE = PHASE1 / "source" / "test6.jsonl"
PACKET = PHASE1 / "evaluation" / "prompt-only-code51.jsonl"
OPUS = PHASE1 / "opus-comparison" / "api-attempt1" / "volume" / "responses.jsonl"
RESCORE = ROOT / "docs" / "phase2" / "evidence" / "lcb-opus-stdin-rescore.json"
# Receipt 205 published 40 of 51 for Opus 5. This bed scores the same retained answers 41: it and the
# phase-1 scorer disagree on exactly one problem, `abc398_a`, where the released tests say input `3`
# gives `-=-` and the phase-1 scorer reported expecting `-=`. The model's answer is right and the
# pinned scorer was wrong, so this bed's panel baseline for the frontier is 41, not 40. That is
# recorded here as a number rather than left as a surprise.
PUBLISHED_OPUS = 40
THIS_BED_OPUS = 41
DISAGREEMENT = "abc398_a"

needs_phase1 = pytest.mark.skipif(not SOURCE.is_file() or not PACKET.is_file(),
                                  reason="phase 1's LiveCodeBench download is local only")
needs_answers = pytest.mark.skipif(not OPUS.is_file() or not SOURCE.is_file() or not RESCORE.is_file(),
                                   reason="phase 1's retained Opus answers are local only")
slow_parity = pytest.mark.skipif(os.environ.get("KIT_CODE_PARITY") != "1",
                                 reason="the full 51-problem parity run takes minutes; set KIT_CODE_PARITY=1")

# A program that reads one integer and prints its successor, and the test it must pass.
GOOD = "print(int(input()) + 1)"
ONE_TEST = [{"input": "1\n", "output": "2\n"}]


# ============================================================ 1. the sandbox, on hand-written problems
def test_sandbox_passes_a_correct_program():
    verdict = sandbox.run_tests(GOOD, ONE_TEST, testtype="stdin", timeout_s=5)
    assert verdict["kind"] == "passed"
    assert (verdict["passed"], verdict["n"], verdict["first_failure_index"]) == (1, 1, None)


def test_sandbox_reports_a_wrong_answer():
    verdict = sandbox.run_tests("print(int(input()) + 2)", ONE_TEST, testtype="stdin", timeout_s=5)
    assert verdict["kind"] == "wrong_answer"


def test_sandbox_stops_a_program_that_never_stops():
    verdict = sandbox.run_tests("while True:\n    pass", ONE_TEST, testtype="stdin", timeout_s=2)
    assert verdict["kind"] == "timeout"
    assert verdict["elapsed_s"] < 20                         # it was killed, not waited out


def test_sandbox_stops_a_program_that_sleeps():
    """A busy loop is also stopped by the CPU limit; a sleeping one is stopped only by the wall clock."""
    verdict = sandbox.run_tests("import time\ntime.sleep(120)\nprint(2)", ONE_TEST,
                                testtype="stdin", timeout_s=2)
    assert verdict["kind"] == "timeout"
    assert verdict["elapsed_s"] < 20


def test_sandbox_reports_a_crash():
    verdict = sandbox.run_tests("raise ValueError('boom')", ONE_TEST, testtype="stdin", timeout_s=5)
    assert verdict["kind"] == "error"


def test_sandbox_refuses_a_network_call():
    verdict = sandbox.run_tests("import socket\nsocket.socket()\nprint(2)", ONE_TEST,
                                testtype="stdin", timeout_s=5)
    assert verdict["kind"] == "network"


def test_sandbox_refuses_a_network_call_through_urllib():
    """The library path, not just the raw socket: `ssl` subclasses `socket.socket` at import time."""
    program = "import urllib.request\nurllib.request.urlopen('http://127.0.0.1:9/x')\nprint(2)"
    verdict = sandbox.run_tests(program, ONE_TEST, testtype="stdin", timeout_s=10)
    assert verdict["kind"] == "network"


def test_sandbox_bounds_a_program_that_prints_forever():
    verdict = sandbox.run_tests("while True:\n    print('x' * 1000)", ONE_TEST,
                                testtype="stdin", timeout_s=20, max_output_mib=1)
    assert verdict["kind"] == "output_limit"


def test_sandbox_gives_the_program_a_real_stdin_with_a_buffer():
    program = "import sys\nprint(int(sys.stdin.buffer.read().strip()) + 1)"
    assert sandbox.run_tests(program, ONE_TEST, testtype="stdin", timeout_s=5)["kind"] == "passed"


def test_sandbox_runs_a_stdin_program_as_main():
    program = "if __name__ == '__main__':\n    print(int(input()) + 1)"
    assert sandbox.run_tests(program, ONE_TEST, testtype="stdin", timeout_s=5)["kind"] == "passed"


def test_sandbox_runs_the_functional_form():
    tests = [{"input": "1\n2", "output": "3"}]
    solution = "class Solution:\n    def add(self, a, b):\n        return a + b"
    assert sandbox.run_tests(solution, tests, testtype="functional", fn_name="add",
                             timeout_s=5)["kind"] == "passed"
    wrong = "class Solution:\n    def add(self, a, b):\n        return a - b"
    assert sandbox.run_tests(wrong, tests, testtype="functional", fn_name="add",
                             timeout_s=5)["kind"] == "wrong_answer"


def test_sandbox_functional_accepts_a_module_level_function_and_a_tuple():
    assert sandbox.run_tests("def add(a, b):\n    return a + b", [{"input": "1\n2", "output": "3"}],
                             testtype="functional", fn_name="add", timeout_s=5)["kind"] == "passed"
    assert sandbox.run_tests("class Solution:\n    def pair(self, a, b):\n        return (a, b)",
                             [{"input": "1\n2", "output": "[1, 2]"}], testtype="functional",
                             fn_name="pair", timeout_s=5)["kind"] == "passed"


def test_sandbox_functional_does_not_run_the_main_block():
    """A functional solution is imported, so its `__main__` block -- which would read stdin -- must not run."""
    solution = ("class Solution:\n    def add(self, a, b):\n        return a + b\n"
                "if __name__ == '__main__':\n    raise SystemExit('main ran')")
    assert sandbox.run_tests(solution, [{"input": "1\n2", "output": "3"}], testtype="functional",
                             fn_name="add", timeout_s=5)["kind"] == "passed"


def test_a_program_that_hunts_for_the_expected_output_cannot_find_it():
    """The cheat that would work if the answer were written into the sandbox: read every file, print the answer.

    The expected output carries a marker. The program searches every file it can see (except itself,
    which of course contains the marker it is searching for) and prints whatever it finds. If the
    answer were anywhere in the sandbox this passes; it must not.
    """
    cheat = ("import pathlib\n"
             "for path in sorted(pathlib.Path('.').rglob('*')):\n"
             "    if path.is_file() and path.name != 'solution.py':\n"
             "        text = path.read_text(errors='ignore')\n"
             "        if 'zzq7' in text:\n"
             "            print(text.strip())\n"
             "            break\n"
             "else:\n"
             "    print('not found')")
    verdict = sandbox.run_tests(cheat, [{"input": "", "output": "zzq7-the-answer"}],
                                testtype="stdin", timeout_s=10)
    assert verdict["kind"] == "wrong_answer"                 # it printed `not found`


def test_the_temporary_directory_is_removed(tmp_path, monkeypatch):
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    import tempfile

    monkeypatch.setattr(tempfile, "tempdir", None)
    before = set(tmp_path.iterdir())
    sandbox.run_tests(GOOD, ONE_TEST, testtype="stdin", timeout_s=5)
    assert set(tmp_path.iterdir()) == before


def test_sandbox_stops_at_the_first_failure():
    tests = [{"input": "1\n", "output": "2\n"}, {"input": "5\n", "output": "999\n"},
             {"input": "7\n", "output": "8\n"}]
    verdict = sandbox.run_tests(GOOD, tests, testtype="stdin", timeout_s=5)
    assert (verdict["kind"], verdict["passed"], verdict["first_failure_index"]) == ("wrong_answer", 1, 1)


def test_sandbox_refuses_nonsense_rather_than_scoring_it():
    with pytest.raises(sandbox.SandboxError):
        sandbox.run_tests("", ONE_TEST, testtype="stdin")
    with pytest.raises(sandbox.SandboxError):
        sandbox.run_tests(GOOD, ONE_TEST, testtype="telepathy")
    with pytest.raises(sandbox.SandboxError):
        sandbox.run_tests(GOOD, [], testtype="stdin")
    with pytest.raises(sandbox.SandboxError):
        sandbox.run_tests("class Solution:\n    pass", ONE_TEST, testtype="functional", fn_name="")


def test_output_comparison_ignores_trailing_whitespace_only():
    assert sandbox.stdout_matches("2 \n\n\n", "2")
    assert not sandbox.stdout_matches(" 2", "2")             # leading whitespace is not ignored
    assert not sandbox.stdout_matches("2\n3", "2")


def test_limits_do_not_overclaim():
    limits = sandbox.limits()
    assert limits["is_a_security_boundary"] is False
    assert limits["expected_output_visible_to_the_program"] is False
    assert limits["not_prevented"]


# ================================================= 2. the bed, on a LiveCodeBench-shaped fixture
def lcb_row(problem_id, *, content, date, tests, starter="", fn_name="", difficulty="easy"):
    """One row in the shape the released dataset ships, with plain-JSON tests the loader also accepts."""
    return {"question_id": problem_id, "question_title": problem_id, "question_content": content,
            "starter_code": starter, "metadata": json.dumps({"func_name": fn_name}) if fn_name else "{}",
            "private_test_cases": json.dumps(tests), "public_test_cases": json.dumps(tests[:1]),
            "contest_date": date, "contest_id": "contest-" + problem_id, "platform": "atcoder",
            "difficulty": difficulty}


PANEL_ROW = lcb_row("p1", content="Print the number plus one.", date="2025-02-08T00:00:00",
                    tests=[{"input": "1\n", "output": "2\n", "testtype": "stdin"},
                           {"input": "4\n", "output": "5\n", "testtype": "stdin"}])
PANEL_ROW_2 = lcb_row("p2", content="Add two numbers.", date="2025-03-01T00:00:00",
                      starter="class Solution:\n    def add(self, a: int, b: int) -> int:\n        ",
                      fn_name="add",
                      tests=[{"input": "1\n2", "output": "3", "testtype": "functional"}])
TRAIN_ROW = lcb_row("t1", content="Print the number plus two.", date="2025-01-01T00:00:00",
                    tests=[{"input": "1\n", "output": "3\n", "testtype": "stdin"},
                           {"input": "4\n", "output": "6\n", "testtype": "stdin"}])
LATE_ROW = lcb_row("t2", content="Print the number plus three.", date="2025-04-01T00:00:00",
                   tests=[{"input": "1\n", "output": "4\n", "testtype": "stdin"}])


def make_root(tmp_path: Path, rows=None) -> Path:
    root = tmp_path / "lcb"
    root.mkdir(parents=True, exist_ok=True)
    rows = [PANEL_ROW, PANEL_ROW_2, TRAIN_ROW, LATE_ROW] if rows is None else rows
    (root / "test.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    return root


def pin(row, order=None) -> dict:
    prompt = code.render_prompt(row)
    tests = json.loads(row["private_test_cases"])
    metadata = json.loads(row["metadata"])
    entry = {"id": row["question_id"],
             "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
             "tests_sha256": hashlib.sha256(row["private_test_cases"].encode("utf-8")).hexdigest(),
             "test_count": len(tests), "testtype": tests[0]["testtype"],
             "fn_name": metadata.get("func_name", ""), "contest_date": row["contest_date"],
             "contest_id": row["contest_id"], "platform": row["platform"],
             "difficulty": row["difficulty"]}
    if order is not None:
        entry["order"] = order
    return entry


def make_split(tmp_path: Path, name="split.json", panel=(PANEL_ROW, PANEL_ROW_2), train=(TRAIN_ROW,)) -> Path:
    members = [pin(row, order) for order, row in enumerate(panel)]
    cutoff = min(entry["contest_date"] for entry in members)
    split = {
        "schema": code.SPLIT_SCHEMA,
        "prompt_template_sha256": hashlib.sha256(code.CODE_PROMPT.encode("utf-8")).hexdigest(),
        "dataset": {"name": "livecodebench/code_generation_lite", "resolved_commit": "fixture"},
        "panel": {"name": "fixture", "earliest_contest_date": cutoff, "members": members},
        "train_rule": {"released_before": cutoff, "excludes_panel_ids": True,
                       "excludes_panel_question_text": True},
        "train_pinned": {"source_file": "test.jsonl", "source_file_sha256": "fixture",
                         "members": [pin(row) for row in train]},
        "counts": {"panel": len(members), "train_pinned": len(train)},
    }
    path = tmp_path / name
    path.write_text(json.dumps(split, indent=1, sort_keys=True), encoding="utf-8")
    return path


def prepared(tmp_path: Path, out_name="out", **kwargs) -> Path:
    root = kwargs.pop("root", None) or make_root(tmp_path)
    split_file = kwargs.pop("split_file", None) or make_split(tmp_path)
    out = tmp_path / out_name
    assert code.main(["prepare", "--lcb-root", str(root), "--out", str(out),
                      "--split-file", str(split_file)] + list(kwargs.pop("extra", []))) == 0
    return out


def test_prepare_writes_both_splits_and_only_pre_cutoff_training(tmp_path):
    out = prepared(tmp_path)
    manifest = json.loads((out / "code.manifest.json").read_text())
    assert manifest["splits"]["heldout"]["rows"] == 2
    assert manifest["splits"]["train"]["ids"] == ["t1"]      # t2 post-dates the panel, p1/p2 are the panel
    assert manifest["splits"]["train"]["latest_contest_date"] < manifest["panel"]["earliest_contest_date"]
    rows = [json.loads(line) for line in (out / "train.jsonl").read_text().split("\n") if line.strip()]
    assert [row["data_source"] for row in rows] == [code.DATA_SOURCE]
    assert "ground_truth" in rows[0]["reward_model"]


def test_prepared_rows_carry_no_tests_and_no_answers(tmp_path):
    out = prepared(tmp_path)
    text = (out / "train.jsonl").read_text()
    assert "\\u0033\\n" not in text
    for row in [json.loads(line) for line in text.split("\n") if line.strip()]:
        reference = json.loads(row["reward_model"]["ground_truth"])
        assert set(reference) == {"problem_id", "testtype", "fn_name", "tests_sha256", "n_tests"}
        assert "output" not in json.dumps(row["extra_info"])


def test_prepare_refuses_to_overwrite_and_leaves_the_first_outputs_alone(tmp_path):
    out = prepared(tmp_path)
    before = {path.name: path.read_bytes() for path in out.iterdir()}
    root, split_file = tmp_path / "lcb", tmp_path / "split.json"
    with pytest.raises(SystemExit) as raised:
        code.main(["prepare", "--lcb-root", str(root), "--out", str(out), "--split-file", str(split_file)])
    assert "refusing to overwrite" in str(raised.value)
    assert {path.name: path.read_bytes() for path in out.iterdir()} == before


def test_write_new_refuses_an_existing_file(tmp_path):
    path = tmp_path / "kept.json"
    path.write_text("first", encoding="utf-8")
    with pytest.raises(code.CodeBedError):
        code._write_new(path, "second")
    assert path.read_text() == "first"


def test_a_refused_prepare_leaves_no_half_written_output(tmp_path):
    """The hash mismatch arrives part way through the panel, after one member is already on disk.

    Writing is streamed, so the refusal has to take the partly written files with it: a `heldout.jsonl`
    holding one of two problems would be worse than no file at all, and it would block the retry.
    """
    changed = dict(PANEL_ROW_2, private_test_cases=json.dumps(
        [{"input": "1\n2", "output": "99", "testtype": "functional"}]))
    root = make_root(tmp_path, rows=[PANEL_ROW, changed, TRAIN_ROW])
    out = tmp_path / "half"
    with pytest.raises(SystemExit) as raised:
        code.main(["prepare", "--lcb-root", str(root), "--out", str(out),
                   "--split-file", str(make_split(tmp_path))])
    assert "not the release phase 1 scored" in str(raised.value)
    left = sorted(path.name for path in out.iterdir())
    assert left == ["train.jsonl", "train.parquet", "train.tests.jsonl"]   # the split that finished
    assert not any(name.endswith(".part") for name in left)


def test_a_refusal_in_the_first_split_leaves_nothing_at_all(tmp_path):
    changed = dict(TRAIN_ROW, private_test_cases=json.dumps(
        [{"input": "1\n", "output": "99\n", "testtype": "stdin"}]))
    root = make_root(tmp_path, rows=[PANEL_ROW, PANEL_ROW_2, changed])
    out = tmp_path / "half"
    with pytest.raises(SystemExit):
        code.main(["prepare", "--lcb-root", str(root), "--out", str(out),
                   "--split-file", str(make_split(tmp_path))])
    assert list(out.iterdir()) == []


def test_prepare_refuses_a_changed_prompt(tmp_path):
    changed = dict(PANEL_ROW, question_content="Print the number plus one, but differently.")
    root = make_root(tmp_path, rows=[changed, PANEL_ROW_2, TRAIN_ROW])
    with pytest.raises(SystemExit) as raised:
        code.main(["prepare", "--lcb-root", str(root), "--out", str(tmp_path / "o"),
                   "--split-file", str(make_split(tmp_path))])
    assert "does not render the prompt phase 1 measured" in str(raised.value)


def test_prepare_refuses_changed_tests(tmp_path):
    changed = dict(PANEL_ROW, private_test_cases=json.dumps(
        [{"input": "1\n", "output": "99\n", "testtype": "stdin"}]))
    root = make_root(tmp_path, rows=[changed, PANEL_ROW_2, TRAIN_ROW])
    with pytest.raises(SystemExit) as raised:
        code.main(["prepare", "--lcb-root", str(root), "--out", str(tmp_path / "o"),
                   "--split-file", str(make_split(tmp_path))])
    assert "not the release phase 1 scored" in str(raised.value)


def test_prepare_refuses_a_missing_panel_problem(tmp_path):
    root = make_root(tmp_path, rows=[PANEL_ROW, TRAIN_ROW])   # p2 is gone
    with pytest.raises(SystemExit) as raised:
        code.main(["prepare", "--lcb-root", str(root), "--out", str(tmp_path / "o"),
                   "--split-file", str(make_split(tmp_path))])
    assert "not in this download" in str(raised.value)


def test_the_heldout_split_itself_refuses_a_short_panel(tmp_path):
    """Not only `prepare`: asking for the panel with a problem missing must refuse, not return 50 of 51."""
    root = make_root(tmp_path, rows=[PANEL_ROW, TRAIN_ROW])   # p2 is gone
    split = code.load_split(make_split(tmp_path))
    with pytest.raises(code.CodeBedError) as raised:
        code.load_members(root, "heldout", split)
    assert "A smaller panel is not the panel" in str(raised.value)
    with pytest.raises(code.CodeBedError):
        code.split_ids(root, "heldout", split)


def test_prepare_refuses_a_missing_pinned_training_problem(tmp_path):
    root = make_root(tmp_path, rows=[PANEL_ROW, PANEL_ROW_2, LATE_ROW])   # t1 is gone
    with pytest.raises(SystemExit) as raised:
        code.main(["prepare", "--lcb-root", str(root), "--out", str(tmp_path / "o"),
                   "--split-file", str(make_split(tmp_path))])
    assert "pinned training problems are not in this download" in str(raised.value)


def test_the_same_problem_under_another_id_is_kept_out_of_training(tmp_path):
    """A panel problem re-published with a different id and a different date must not become training data."""
    twin = lcb_row("t9", content=PANEL_ROW["question_content"], date="2025-01-02T00:00:00",
                   tests=json.loads(PANEL_ROW["private_test_cases"]))
    root = make_root(tmp_path, rows=[PANEL_ROW, PANEL_ROW_2, TRAIN_ROW, twin])
    out = prepared(tmp_path, root=root)
    assert json.loads((out / "code.manifest.json").read_text())["splits"]["train"]["ids"] == ["t1"]


def test_load_source_refuses_a_file_that_is_not_the_dataset(tmp_path):
    root = tmp_path / "lcb"
    root.mkdir()
    (root / "test.jsonl").write_text(json.dumps({"question_id": "x"}) + "\n", encoding="utf-8")
    with pytest.raises(code.CodeBedError):
        code.load_source(root)


# ---------------------------------------------------------------------------- extraction and reward
def test_extract_code_takes_the_last_fence():
    assert code.extract_code("```python\nfirst()\n```\nand then\n```\nsecond()\n```") == "second()"
    assert code.extract_code("no fence here") is None
    assert code.extract_code("```\n\n```") is None
    assert code.extract_code("```py\nx = 1\n```") == "x = 1"


def reward(tmp_path, response, problem="t1", **kwargs):
    out = kwargs.pop("out", None) or prepared(tmp_path)
    rows = [json.loads(line) for line in (out / "train.jsonl").read_text().split("\n") if line.strip()]
    row = next(row for row in rows if row["extra_info"]["problem_id"] == problem)
    return code.compute_score(code.DATA_SOURCE, response, row["reward_model"]["ground_truth"],
                              {"tests_path": str(out / "train.tests.jsonl")},
                              timeout_s=kwargs.pop("timeout_s", 5)), out


def test_reward_is_one_only_when_every_test_passes(tmp_path):
    out = prepared(tmp_path)
    good, _ = reward(tmp_path, "```python\nprint(int(input()) + 2)\n```", out=out)
    assert (good["score"], good["acc"], good["incorrect_format"], good["feedback"]) == (1.0, 1.0, 0, "")
    # Right on the first test, wrong on the second: the reward is still zero.
    half = "```python\nprint(3 if int(input()) == 1 else 0)\n```"
    partial, _ = reward(tmp_path, half, out=out)
    assert partial["score"] == 0.0


def test_reward_reports_each_failure_kind(tmp_path):
    out = prepared(tmp_path)
    cases = {
        "```python\nprint(int(input()) + 9)\n```": "wrong_answer",
        "```python\nwhile True:\n    pass\n```": "timeout",
        "```python\nraise ValueError('x')\n```": "error",
        "```python\nimport socket\nsocket.socket()\n```": "network",
    }
    for response, kind in cases.items():
        result, _ = reward(tmp_path, response, out=out, timeout_s=2)
        assert result["score"] == 0.0 and result["incorrect_format"] == 0
        assert result["feedback"] == code.FEEDBACK[kind], (response, result["feedback"])


def test_a_response_with_no_code_block_is_a_format_failure(tmp_path):
    result, _ = reward(tmp_path, "I think the answer is to add two.")
    assert (result["score"], result["incorrect_format"]) == (0.0, 1)
    assert result["feedback"] == code.FEEDBACK["no_code"]
    assert result["pred"] == ""


def test_feedback_carries_no_number_at_all():
    """Not the failing test's index, not how many passed, not how many there are: no digit anywhere."""
    for kind, text in code.FEEDBACK.items():
        assert not any(character.isdigit() for character in text), kind
    assert set(code.FEEDBACK) >= {"passed", "wrong_answer", "timeout", "error", "network", "no_code"}


def test_feedback_never_carries_the_expected_output(tmp_path):
    """The reward runs against a problem whose answers are distinctive words; none may reach the model."""
    row = lcb_row("t1", content="Print the secret.", date="2025-01-01T00:00:00",
                  tests=[{"input": "1\n", "output": "xyzzyplugh\n", "testtype": "stdin"},
                         {"input": "2\n", "output": "plovercrystal\n", "testtype": "stdin"}])
    root = make_root(tmp_path, rows=[PANEL_ROW, PANEL_ROW_2, row])
    out = prepared(tmp_path, root=root, split_file=make_split(tmp_path, train=(row,)),
                   extra=["--train-tests", "0"])
    result, _ = reward(tmp_path, "```python\nprint('wrong')\n```", out=out)
    assert result["score"] == 0.0
    blob = json.dumps(result)
    assert "xyzzyplugh" not in blob and "plovercrystal" not in blob


def test_reward_refuses_a_foreign_data_source_or_a_broken_ground_truth(tmp_path):
    out = prepared(tmp_path)
    tests = str(out / "train.tests.jsonl")
    with pytest.raises(code.CodeBedError):
        code.compute_score("gsm8k", "```\nx\n```", "{}", {"tests_path": tests})
    with pytest.raises(code.CodeBedError):
        code.compute_score(code.DATA_SOURCE, "```\nx\n```", "not json", {"tests_path": tests})
    with pytest.raises(code.CodeBedError):
        code.compute_score(code.DATA_SOURCE, "```\nx\n```",
                           json.dumps({"problem_id": "nope", "tests_sha256": "x"}), {"tests_path": tests})


def test_reward_refuses_tests_the_row_was_not_prepared_from(tmp_path):
    out = prepared(tmp_path)
    rows = [json.loads(line) for line in (out / "train.jsonl").read_text().split("\n") if line.strip()]
    reference = json.loads(rows[0]["reward_model"]["ground_truth"])
    reference["tests_sha256"] = "0" * 64
    with pytest.raises(code.CodeBedError) as raised:
        code.compute_score(code.DATA_SOURCE, "```\nprint(1)\n```", json.dumps(reference),
                           {"tests_path": str(out / "train.tests.jsonl")})
    assert "refusing to score against tests this row was not prepared from" in str(raised.value)


def test_reward_refuses_when_the_tests_file_is_not_named(tmp_path, monkeypatch):
    monkeypatch.delenv(code.TESTS_ENV, raising=False)
    with pytest.raises(code.CodeBedError):
        code.compute_score(code.DATA_SOURCE, "```\nx\n```", json.dumps({"problem_id": "t1", "tests_sha256": "x"}))


def test_eval_items_has_the_shape_eval_bed_asks_for(tmp_path):
    """`kit/eval_bed.py` wants {'id', 'prompt', 'ground_truth'} per item, and the panel on every test."""
    out = prepared(tmp_path)
    items = code.eval_items(tmp_path / "lcb", "heldout", tmp_path / "split.json")
    assert [item["id"] for item in items] == ["p1", "p2"]     # phase 1's order, not sorted
    assert all(set(item) == {"id", "prompt", "ground_truth"} for item in items)
    assert json.loads(items[0]["ground_truth"])["n_tests"] == 2
    result = code.compute_score(code.DATA_SOURCE, "```python\nprint(int(input()) + 1)\n```",
                                items[0]["ground_truth"],
                                {"tests_path": str(out / "heldout.tests.jsonl")}, timeout_s=5)
    assert result["score"] == 1.0


def test_prepare_accepts_the_code_root_spelling_too(tmp_path):
    """K5's package calls every bed's data directory `--code-root`; both spellings are the same flag."""
    root, split_file = make_root(tmp_path), make_split(tmp_path)
    assert code.main(["prepare", "--code-root", str(root), "--out", str(tmp_path / "o"),
                      "--split-file", str(split_file)]) == 0
    assert (tmp_path / "o" / "heldout.jsonl").is_file()


def test_the_functional_panel_problem_scores_through_the_bed(tmp_path):
    out = prepared(tmp_path)
    rows = [json.loads(line) for line in (out / "heldout.jsonl").read_text().split("\n") if line.strip()]
    row = next(row for row in rows if row["extra_info"]["problem_id"] == "p2")
    assert row["extra_info"]["testtype"] == "functional" and row["extra_info"]["fn_name"] == "add"
    assert "Your solution should have the following signature" in row["prompt"][0]["content"]
    result = code.compute_score(
        code.DATA_SOURCE, "```python\nclass Solution:\n    def add(self, a, b):\n        return a + b\n```",
        row["reward_model"]["ground_truth"], {"tests_path": str(out / "heldout.tests.jsonl")}, timeout_s=5)
    assert result["score"] == 1.0


# ------------------------------------------------------------------------------- the test subset rule
def test_the_training_subset_is_deterministic_and_bounded():
    tests = [{"input": str(index), "output": str(index)} for index in range(40)]
    first = code.select_tests("abc", tests, 8, 1 << 20)
    assert first == code.select_tests("abc", tests, 8, 1 << 20)
    assert len(first[0]) == 8 and first[0] == sorted(first[0])
    assert code.select_tests("xyz", tests, 8, 1 << 20)[0] != first[0]        # keyed on the problem id
    assert len(code.select_tests("abc", tests, 0, 0)[0]) == 40               # 0 keeps everything
    one, kept, capped = code.select_tests("abc", tests, 8, 1)               # a budget nothing fits in
    assert len(one) == 1 and capped and len(kept) == 1                       # never zero tests


def test_the_panel_is_scored_on_every_test_and_training_is_capped(tmp_path):
    out = prepared(tmp_path, extra=["--train-tests", "1"])
    manifest = json.loads((out / "code.manifest.json").read_text())
    assert manifest["splits"]["heldout"]["tests_run_per_problem"] == \
        manifest["splits"]["heldout"]["tests_available_per_problem"]
    assert manifest["splits"]["train"]["tests_run_per_problem"] == 1
    assert manifest["splits"]["train"]["tests_available_per_problem"] == 2
    assert manifest["train_test_cap"]["max_tests"] == 1


def test_the_tests_file_is_read_by_seeking_and_checks_its_own_hash(tmp_path):
    out = prepared(tmp_path)
    path = out / "train.tests.jsonl"
    handle = code.TestsFile(path)
    assert set(handle.offsets) == {"t1"}
    assert handle.get("t1")["testtype"] == "stdin"
    with pytest.raises(code.CodeBedError):
        handle.get("nope")
    damaged = json.loads(path.read_text().split("\n")[0])
    damaged["tests"][0]["output"] = "tampered"
    (out / "damaged.tests.jsonl").write_text(json.dumps(damaged, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(code.CodeBedError):
        code.TestsFile(out / "damaged.tests.jsonl").get("t1")


def test_the_tests_file_refuses_an_index_that_points_at_the_wrong_line(tmp_path):
    """The seek is only as good as the offset: a line that is not the problem asked for is refused."""
    out = prepared(tmp_path)
    handle = code.TestsFile(out / "heldout.tests.jsonl")
    assert set(handle.offsets) == {"p1", "p2"}
    handle.offsets["p1"] = handle.offsets["p2"]
    with pytest.raises(code.CodeBedError) as raised:
        handle.get("p1")
    assert "where its index says it does" in str(raised.value)


# ---------------------------------------------------------------------------------- scoring responses
def test_score_refuses_missing_and_foreign_responses(tmp_path):
    out = prepared(tmp_path)
    ids = code.split_ids(tmp_path / "lcb", "heldout", code.load_split(tmp_path / "split.json"))
    tests_file = code.TestsFile(out / "heldout.tests.jsonl")
    with pytest.raises(code.CodeBedError):
        code.score_responses(ids, [{"id": "p1", "response": "```\nx\n```"}], tests_file, timeout_s=2)
    with pytest.raises(code.CodeBedError):
        code.score_responses(ids, [{"id": "p1", "response": "x"}, {"id": "p1", "response": "y"}],
                             tests_file, timeout_s=2)
    with pytest.raises(code.CodeBedError):
        code.score_responses(ids, [{"id": "p1", "response": "x"}, {"id": "p2", "response": "y"},
                                   {"id": "ghost", "response": "z"}], tests_file, timeout_s=2)


def test_score_counts_the_panel(tmp_path):
    out = prepared(tmp_path)
    ids = code.split_ids(tmp_path / "lcb", "heldout", code.load_split(tmp_path / "split.json"))
    answers = [{"id": "p1", "response": "```python\nprint(int(input()) + 1)\n```"},
               {"id": "p2", "response": "```python\nclass Solution:\n    def add(self, a, b):\n        return a - b\n```"}]
    summary = code.score_responses(ids, answers, code.TestsFile(out / "heldout.tests.jsonl"), timeout_s=5)
    assert (summary["n"], summary["correct"]) == (2, 1)
    assert summary["per_problem"] == {"p1": 1, "p2": 0}


def test_the_score_subcommand_runs_end_to_end(tmp_path, monkeypatch, capsys):
    out = prepared(tmp_path)
    answers = tmp_path / "answers.jsonl"
    answers.write_text("".join(json.dumps(row) + "\n" for row in [
        {"id": "p1", "response": "```python\nprint(int(input()) + 1)\n```"},
        {"id": "p2", "response": "```python\nclass Solution:\n    def add(self, a, b):\n        return a + b\n```"},
    ]), encoding="utf-8")
    monkeypatch.setenv(code.TESTS_ENV, str(out / "heldout.tests.jsonl"))
    assert code.main(["score", "--responses", str(answers), "--split", "heldout",
                      "--lcb-root", str(tmp_path / "lcb"), "--split-file", str(tmp_path / "split.json"),
                      "--timeout", "5", "--out", str(tmp_path / "score.json")]) == 0
    summary = json.loads((tmp_path / "score.json").read_text())
    assert (summary["n"], summary["correct"], summary["verified_members"]) == (2, 2, 2)


# ==================================================== 3. the shipped split file, read as data
def test_the_shipped_split_file_loads():
    split = code.load_split(SPLIT_FILE)
    assert split["counts"]["panel"] == 51
    assert split["panel"]["name"] == "code51"


def test_the_shipped_panel_is_phase_ones_panel_in_phase_ones_order():
    panel = code.load_split(SPLIT_FILE)["panel"]["members"]
    assert len(panel) == 51 and len({row["id"] for row in panel}) == 51
    assert [row["order"] for row in panel] == list(range(51))
    assert {row["testtype"] for row in panel} == {"stdin", "functional"}
    for row in panel:
        assert (row["fn_name"] != "") == (row["testtype"] == "functional")
        assert row["test_count"] >= 1
        assert len(row["prompt_sha256"]) == 64 and len(row["tests_sha256"]) == 64


def test_nothing_pinned_for_training_post_dates_the_panel():
    split = code.load_split(SPLIT_FILE)
    cutoff = split["panel"]["earliest_contest_date"]
    assert cutoff == min(row["contest_date"] for row in split["panel"]["members"])
    assert split["train_rule"]["released_before"] == cutoff
    pinned = split["train_pinned"]["members"]
    assert pinned and all(row["contest_date"] < cutoff for row in pinned)
    assert not {row["id"] for row in pinned} & {row["id"] for row in split["panel"]["members"]}


def test_the_split_file_refuses_a_different_prompt_template(tmp_path, monkeypatch):
    monkeypatch.setattr(code, "CODE_PROMPT", code.CODE_PROMPT + " and be quick about it")
    with pytest.raises(code.CodeBedError):
        code.load_split(SPLIT_FILE)


def test_the_split_file_refuses_a_shuffled_panel(tmp_path):
    split = json.loads(SPLIT_FILE.read_text())
    split["panel"]["members"] = list(reversed(split["panel"]["members"]))
    path = tmp_path / "shuffled.json"
    path.write_text(json.dumps(split), encoding="utf-8")
    with pytest.raises(code.CodeBedError):
        code.load_split(path)


# ================================================================ 4. parity with phase 1 (local only)
@needs_phase1
def test_the_split_file_regenerates_byte_identically(tmp_path):
    out = tmp_path / "code-split-v1.json"
    result = subprocess.run([sys.executable, str(GENERATOR), "--out", str(out)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert out.read_bytes() == SPLIT_FILE.read_bytes()


@needs_phase1
def test_every_panel_prompt_rebuilds_from_the_local_release_file():
    problems = code.load_source(SOURCE)
    members = code.load_members(SOURCE, "heldout", code.load_split(SPLIT_FILE), problems)
    packet = [json.loads(line) for line in PACKET.read_text(encoding="utf-8").split("\n") if line.strip()]
    assert [member["id"] for member in members] == [row["id"] for row in packet]
    assert [member["prompt"] for member in members] == [row["prompt"] for row in packet]


@needs_phase1
def test_the_one_problem_this_bed_scores_differently_from_phase_one(tmp_path):
    """`abc398_a`: the released tests say input `3` gives `-=-`, and Opus wrote exactly that.

    Phase 1's pinned scorer marked it wrong, reporting an expected `-=`. This is fast enough to run
    everywhere the download exists, and it is the whole of the difference between 40 and 41.
    """
    problems = code.load_source(SOURCE)
    tests = code.decode_tests(problems[DISAGREEMENT]["private_test_cases"])
    threes = [test for test in tests if test["input"].strip() == "3"]
    assert threes and all(test["output"].strip() == "-=-" for test in threes)
    answer = ("```python\nn = int(input())\nif n % 2 == 1:\n    print('-'*(n//2) + '=' + '-'*(n//2))\n"
              "else:\n    print('-'*(n//2-1) + '==' + '-'*(n//2-1))\n```")
    stored = {"testtype": "stdin", "fn_name": "", "tests": tests}
    assert code.score_response(answer, stored, timeout_s=10)["correct"] is True


@needs_answers
@slow_parity
def test_phase_ones_opus_answers_score_forty_one_of_fifty_one(tmp_path):
    """Receipt 205's 40, reproduced by this bed's own sandbox, plus the one problem phase 1 got wrong."""
    out = tmp_path / "prepared"
    assert code.main(["prepare", "--lcb-root", str(SOURCE.parent), "--out", str(out)]) == 0
    tests_file = code.TestsFile(out / "heldout.tests.jsonl")
    answers = [json.loads(line) for line in OPUS.read_text(encoding="utf-8").split("\n") if line.strip()]
    published = json.loads(RESCORE.read_text())["per_problem"]
    mine = {row["member_id"]: int(code.score_response(row["response_text"], tests_file.get(row["member_id"]),
                                                      timeout_s=10)["correct"])
            for row in answers}
    assert sum(mine.values()) == THIS_BED_OPUS
    assert sum(entry["opus"] for entry in published.values()) == PUBLISHED_OPUS
    assert [key for key in mine if mine[key] != published[key]["opus"]] == [DISAGREEMENT]
