"""kit/beds/gsm8k.py: the contamination guard, the scoring rule, the reward function's shape, and no gold in feedback.

The test that matters here is the contamination one. 100 of the 300 members of the forgetting panel are
GSM8K questions, so a training set built without a guard -- or with a guard that quietly matches nothing
because the panel's wording moved -- trains on the measure and nothing downstream looks wrong. So the
guard is exercised on a planted question, the writer is exercised with the guard bypassed, and the panel
reader is exercised on a panel it cannot read exactly.

Everything here builds its own fixtures in tmp_path. The one test that needs a real GSM8K skips when
there is none on the machine.
"""
from __future__ import annotations

import ast
import importlib.util
import json
import os
import random
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BED = ROOT / "kit" / "beds" / "gsm8k.py"
_spec = importlib.util.spec_from_file_location("kit_bed_gsm8k", BED)
gsm8k = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gsm8k)

SUFFIX = gsm8k.PANEL_PROMPT_SUFFIX


# --------------------------------------------------------------------------------------- fixtures
def a_question(tag: str, i: int) -> str:
    return ("%s item %02d. A baker sells %d loaves on Monday and twice as many on Tuesday. "
            "How many loaves did the baker sell altogether?" % (tag, i, i + 3))


def a_row(question: str, gold) -> dict:
    return {"question": question, "answer": "Tuesday was 2 * x loaves.\n<<1+1=2>>\n#### %s" % gold}


def panel_file(tmp_path: Path, questions: list, name: str = "panel.jsonl") -> Path:
    """A forgetting panel of the shape kit/panels/general-v1.jsonl has: 100 maths members, and others."""
    members = [{"id": "gsm8k-%d" % i, "order": i, "panel": "math", "answer": str(i),
                "prompt": question + SUFFIX} for i, question in enumerate(questions)]
    members += [{"id": "mmlu-%d" % i, "order": i, "panel": "knowledge", "answer": "A", "prompt": "x"} for i in range(3)]
    path = tmp_path / name
    path.write_text("".join(json.dumps(m) + "\n" for m in members))
    return path


@pytest.fixture()
def panel_questions_100() -> list:
    return [a_question("Panel", i) for i in range(gsm8k.PANEL_MATH_MEMBERS)]


@pytest.fixture()
def panel(tmp_path, panel_questions_100) -> Path:
    return panel_file(tmp_path, panel_questions_100)


def write_split(root: Path, split: str, rows: list) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / ("%s.jsonl" % split)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


# ------------------------------------------------------------------------------- the scoring rule
@pytest.mark.parametrize("prediction, gold, want", [
    ("18", "18", True), ("18.0", "18", True), ("18", "18.0", True), ("18.00", "18.000", True),
    ("$18", "18", True), ("$ 18.", "18", True), ("1,800", "1800", True), ("-5", "-5", True), ("+7", "7", True),
    ("18", "19", False), ("180", "18", False), ("18.5", "18", False), ("", "18", False),
    ("eighteen", "18", False), ("18 eggs", "18", False), ("nan", "18", False), ("Infinity", "18", False),
])
def test_the_scoring_rule(prediction, gold, want):
    assert gsm8k.is_correct(gsm8k.extract_answer("Answer: " + prediction), gold) is want


def test_the_last_answer_line_wins_and_only_a_missing_one_is_a_format_failure():
    assert gsm8k.extract_answer("Answer: 3\non reflection that was wrong.\nAnswer: 18") == "18"
    assert gsm8k.extract_answer("the baker sold 18 loaves") is None
    missing = gsm8k.compute_score("gsm8k", "the baker sold 18 loaves", "18")
    assert missing["score"] == 0.0 and missing["incorrect_format"] == 1 and "Answer:" in missing["feedback"]
    # a line that is there but is not a number is wrong, NOT a format failure: the line was written
    wordy = gsm8k.compute_score("gsm8k", "Answer: eighteen loaves", "18")
    assert wordy["score"] == 0.0 and wordy["incorrect_format"] == 0 and "single number" in wordy["feedback"]


def test_the_reward_function_has_the_reference_shape():
    right = gsm8k.compute_score("gsm8k", "2 * 9 = 18\nAnswer: 18", "18")
    assert right == {"score": 1.0, "acc": 1.0, "pred": "18", "incorrect_format": 0, "feedback": ""}
    assert gsm8k.compute_score("gsm8k", "Answer: 18.0", "18", {"split": "train"})["score"] == 1.0
    wrong = gsm8k.compute_score("gsm8k", "Answer: 12", "18")
    assert set(wrong) == {"score", "acc", "pred", "incorrect_format", "feedback"}
    assert wrong["score"] == 0.0 and wrong["acc"] == 0.0


def test_feedback_never_carries_the_gold_in_any_form():
    """Stronger than 'the gold string is absent': the feedback must not depend on the gold at all.

    Two different wrong golds against the same answer must produce the identical sentence, so no
    wording of it -- the number, a near miss, a hint at the magnitude -- can leak what the answer was.
    """
    for solution in ("2 * 9 = 18\nAnswer: 18", "Answer: eighteen", "no final line at all"):
        feedbacks = {gsm8k.compute_score("gsm8k", solution, gold)["feedback"] for gold in ("7", "19", "1234", "-3")}
        assert len(feedbacks) == 1, (solution, feedbacks)
    for gold in ("7", "19", "1234", "18999"):
        for solution in ("Answer: 4", "Answer: 18999999", "Answer: nothing", "nothing at all"):
            feedback = gsm8k.compute_score("gsm8k", solution, gold)["feedback"]
            assert gold not in feedback, (gold, feedback)


# ------------------------------------------------------------------------- the contamination guard
def test_a_planted_panel_question_is_removed_from_the_training_set(panel, panel_questions_100):
    """The guard's whole purpose: a panel question in the training set is found and taken out."""
    questions = gsm8k.panel_questions(panel)
    assert len(questions) == gsm8k.PANEL_MATH_MEMBERS
    clean = [a_row(a_question("Train", i), i) for i in range(20)]
    planted = [a_row(panel_questions_100[3], 3),
               a_row("   " + panel_questions_100[7].upper() + "  \n ", 7),   # same question, different case and spacing
               a_row(panel_questions_100[99], 99)]
    items = [dict(row, id="x%d" % i) for i, row in enumerate(clean + planted)]
    kept, removed = gsm8k.remove_contaminated(items, questions)
    assert len(removed) == 3 and len(kept) == 20
    assert {row["question"] for row in kept}.isdisjoint({q for q in panel_questions_100})
    assert gsm8k.contaminated(kept, questions) == []
    assert len(gsm8k.contaminated(items, questions)) == 3


def test_the_writer_refuses_when_the_guard_is_bypassed(tmp_path, panel, panel_questions_100):
    """If the filter is skipped, forgotten, or silently returns everything, nothing reaches disk."""
    questions = gsm8k.panel_questions(panel)
    items = [dict(a_row(a_question("Train", i), i), id="t%d" % i) for i in range(3)]
    items.append(dict(a_row(panel_questions_100[5], 5), id="leak"))
    rows = gsm8k.rows_for_trainer(items, "train")
    out = tmp_path / "train.jsonl"
    with pytest.raises(SystemExit, match="refusing to write"):
        gsm8k.write_rows(out, rows, questions)
    assert not out.exists(), "nothing may be written when the guard refuses"
    # and a row whose recorded `problem` was scrubbed is still caught, because the prompt carries the question
    rows[-1]["extra_info"]["problem"] = "something else entirely"
    with pytest.raises(SystemExit, match="refusing to write"):
        gsm8k.write_rows(out, rows, questions)
    clean_hash = gsm8k.write_rows(out, rows[:3], questions)
    assert out.exists() and len(clean_hash) == 64


def test_the_writer_refuses_to_overwrite_an_output(tmp_path, panel):
    questions = gsm8k.panel_questions(panel)
    rows = gsm8k.rows_for_trainer([dict(a_row(a_question("Train", 1), 1), id="t1")], "train")
    out = tmp_path / "train.jsonl"
    gsm8k.write_rows(out, rows, questions)
    before = out.read_text()
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        gsm8k.write_rows(out, rows, questions)
    assert out.read_text() == before


def test_a_panel_the_guard_cannot_read_exactly_is_refused_rather_than_matching_nothing(tmp_path, panel_questions_100):
    """A guard that silently matches nothing is the failure this file exists to prevent."""
    with pytest.raises(SystemExit, match="missing"):
        gsm8k.panel_questions(tmp_path / "nothing-here.jsonl")
    short = panel_file(tmp_path, panel_questions_100[:99], "short.jsonl")
    with pytest.raises(SystemExit, match="maths members"):
        gsm8k.panel_questions(short)
    moved = tmp_path / "moved.jsonl"
    lines = panel_file(tmp_path, panel_questions_100, "ok.jsonl").read_text().splitlines()
    first = json.loads(lines[0])
    first["prompt"] = first["prompt"].replace(SUFFIX, "\n\nGive your answer as a number.")
    moved.write_text("".join([json.dumps(first) + "\n"] + [line + "\n" for line in lines[1:]]))
    with pytest.raises(SystemExit, match="cannot be recovered"):
        gsm8k.panel_questions(moved)


def test_the_shipped_panel_reads_and_its_questions_are_bare_questions():
    """No local dataset needed: the panel travels inside the kit."""
    questions = gsm8k.panel_questions()
    assert len(questions) == gsm8k.PANEL_MATH_MEMBERS
    assert all("solve step by step" not in question for question in questions)
    assert all(question == gsm8k.normalise(question) for question in questions)


# -------------------------------------------------------------------------------- the held-out set
def test_the_held_out_set_is_deterministic_and_disjoint_from_the_panel(panel, panel_questions_100):
    questions = gsm8k.panel_questions(panel)
    test_items = [dict(a_row(a_question("Test", i), i), id="e%d" % i) for i in range(40)]
    test_items += [dict(a_row(panel_questions_100[i], i), id="p%d" % i) for i in range(10)]
    chosen = gsm8k.held_out(test_items, questions, 20)
    assert len(chosen) == 20
    assert gsm8k.contaminated(chosen, questions) == []
    shuffled = list(test_items)
    random.Random(7).shuffle(shuffled)
    assert [item["id"] for item in gsm8k.held_out(shuffled, questions, 20)] == [item["id"] for item in chosen]
    with pytest.raises(SystemExit, match="held-out items were asked for"):
        gsm8k.held_out(test_items, questions, 45)


# ------------------------------------------------------------------------------------- the loader
def test_the_loader_reads_jsonl_refuses_a_wrong_size_and_refuses_a_row_with_no_gold(tmp_path):
    root = tmp_path / "gsm8k"
    write_split(root, "train", [a_row(a_question("Train", i), i * 3) for i in range(5)])
    items = gsm8k.load(root, "train", expect_size=False)
    assert len(items) == 5 and items[2]["id"] == "gsm8k-train-2" and items[2]["gold"] == "6"
    with pytest.raises(SystemExit, match="not the documented dataset"):
        gsm8k.load(root, "train")
    with pytest.raises(SystemExit, match="no GSM8K test split"):
        gsm8k.load(root, "test", expect_size=False)
    broken = tmp_path / "broken"
    write_split(broken, "train", [{"question": "how many?", "answer": "there is no gold line here"}])
    with pytest.raises(SystemExit, match="has no ####"):
        gsm8k.load(broken, "train", expect_size=False)
    empty = tmp_path / "empty"
    write_split(empty, "train", [{"question": "  ", "answer": "#### 3"}])
    with pytest.raises(SystemExit, match="has no question"):
        gsm8k.load(empty, "train", expect_size=False)


def test_the_loader_reads_the_hugging_face_parquet_layout(tmp_path):
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    rows = [a_row(a_question("Train", i), i) for i in range(4)]
    main = tmp_path / "snapshot" / "main"
    main.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(rows), main / "train-00000-of-00001.parquet")
    items = gsm8k.load(tmp_path / "snapshot", "train", expect_size=False)
    assert [item["gold"] for item in items] == ["0", "1", "2", "3"]


def test_the_prompt_carries_the_problem_and_the_format_it_will_be_judged_on():
    item = {"id": "gsm8k-train-0", "question": a_question("Train", 1), "answer": "#### 8", "gold": "8"}
    prompt = gsm8k.render_prompt(item)
    assert item["question"] in prompt and "Answer: <number>" in prompt
    row = gsm8k.rows_for_trainer([item], "train")[0]
    assert row["data_source"] == "gsm8k" and row["ability"] == "gsm8k"
    assert row["reward_model"] == {"style": "gsm8k", "ground_truth": "8"}
    assert row["prompt"][0]["role"] == "user" and row["prompt"][0]["content"] == prompt
    assert row["extra_info"]["split"] == "train" and row["extra_info"]["index"] == "gsm8k-train-0"
    assert set(row["extra_info"]) == {"split", "index", "problem", "description", "elo", "achievement_prior"}


# ------------------------------------------------------------------------------- end to end on CPU
def test_prepare_writes_a_clean_training_file_a_held_out_set_and_a_manifest(tmp_path, panel, panel_questions_100):
    root = tmp_path / "gsm8k"
    train = [a_row(a_question("Train", i), i) for i in range(12)] + [a_row(panel_questions_100[1], 1)]
    test = [a_row(a_question("Test", i), i) for i in range(15)] + [a_row(panel_questions_100[2], 2)]
    write_split(root, "train", train)
    write_split(root, "test", test)
    out = tmp_path / "out"
    argv = ["prepare", "--gsm8k-root", str(root), "--out", str(out), "--panel", str(panel),
            "--heldout-n", "5", "--allow-subset"]
    assert gsm8k.main(argv) == 0
    rows = [json.loads(line) for line in (out / "train.jsonl").read_text().splitlines()]
    assert len(rows) == 12
    questions = gsm8k.panel_questions(panel)
    assert gsm8k.rows_contaminated(rows, questions) == []
    heldout = [json.loads(line) for line in (out / "heldout.jsonl").read_text().splitlines()]
    assert len(heldout) == 5 and gsm8k.rows_contaminated(heldout, questions) == []
    assert {row["extra_info"]["problem"] for row in heldout} <= {a_question("Test", i) for i in range(15)}
    manifest = json.loads((out / "gsm8k.manifest.json").read_text())
    assert manifest["schema"] == "kit-bed-gsm8k.v1" and manifest["allow_subset"] is True
    assert manifest["panel"]["in_train_split"] == 1 and manifest["panel"]["in_test_split"] == 1
    assert manifest["splits"]["train"] == {"rows": 12, "jsonl_sha256": manifest["splits"]["train"]["jsonl_sha256"],
                                           "parquet": manifest["splits"]["train"]["parquet"],
                                           "source_split": "train", "contaminated_removed": 1}
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        gsm8k.main(argv)


def test_score_refuses_responses_that_do_not_belong_to_the_split(tmp_path, panel, capsys):
    root = tmp_path / "gsm8k"
    write_split(root, "test", [a_row(a_question("Test", i), i) for i in range(8)])
    items = gsm8k.held_out(gsm8k.load(root, "test", expect_size=False), gsm8k.panel_questions(panel), 4)
    responses = tmp_path / "responses.jsonl"
    argv = ["score", "--gsm8k-root", str(root), "--split", "heldout", "--responses", str(responses),
            "--panel", str(panel), "--heldout-n", "4", "--allow-subset"]

    responses.write_text("".join(json.dumps({"id": item["id"], "response": "Answer: %s" % item["gold"]}) + "\n"
                                 for item in items[:3]))
    with pytest.raises(SystemExit, match="have no response"):
        gsm8k.main(argv)

    responses.unlink()
    responses.write_text("".join(json.dumps({"id": item["id"], "response": "Answer: %s" % item["gold"]}) + "\n"
                                 for item in items) + json.dumps({"id": "gsm8k-test-999", "response": "Answer: 1"}) + "\n")
    with pytest.raises(SystemExit, match="do not belong to this split"):
        gsm8k.main(argv)

    responses.unlink()
    lines = [json.dumps({"id": item["id"], "response": "Answer: %s" % item["gold"]}) for item in items[:3]]
    lines.append(json.dumps({"id": items[3]["id"], "response": "no answer line here"}))
    responses.write_text("\n".join(lines) + "\n")
    assert gsm8k.main(argv) == 0
    summary = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert summary == {"split": "heldout", "n": 4, "correct": 3, "no_final_answer": 1, "accuracy": 0.75}


# ---------------------------------------------------------------------------------- the kit is standalone
def test_the_bed_imports_nothing_from_continual_or_sdft():
    """kit/ is published as a public repository of its own; it may not reach into ours."""
    tree = ast.parse(BED.read_text(), str(BED))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".")[0])
    assert not imported & {"continual", "sdft", "kit", "modal", "torch", "transformers"}, imported
    assert imported <= {"argparse", "hashlib", "json", "re", "sys", "decimal", "pathlib", "pyarrow", "__future__"}, imported


# ------------------------------------------------------- local data only: does the panel sit in the dataset?
def local_gsm8k():
    """A GSM8K on this machine, or None. Nothing is downloaded."""
    named = os.environ.get("GSM8K_ROOT")
    candidates = [Path(named)] if named else []
    candidates += [ROOT / "data" / "gsm8k", ROOT / "references" / "gsm8k"]
    candidates += sorted(Path.home().glob(".cache/huggingface/hub/datasets--openai--gsm8k/snapshots/*"))
    for candidate in candidates:
        try:
            gsm8k.split_files(candidate, "test")
            gsm8k.split_files(candidate, "train")
        except (SystemExit, OSError):
            continue
        return candidate
    return None


def test_where_the_panel_questions_sit_in_a_local_gsm8k():
    """Reports the overlap, and fails if any panel question is in the split we would train on."""
    root = local_gsm8k()
    if root is None:
        pytest.skip("no local GSM8K (set GSM8K_ROOT to a directory holding the openai/gsm8k `main` config)")
    questions = gsm8k.panel_questions()
    train, test = gsm8k.load(root, "train"), gsm8k.load(root, "test")
    in_train, in_test = gsm8k.contaminated(train, questions), gsm8k.contaminated(test, questions)
    print("\nlocal GSM8K at %s: %d train rows, %d test rows; of the %d panel maths questions, %d are in train "
          "and %d are in test" % (root, len(train), len(test), len(questions), len(in_train), len(in_test)))
    assert len(in_train) == 0, "the training split contains %d forgetting-panel questions" % len(in_train)
    assert len(in_test) == gsm8k.PANEL_MATH_MEMBERS, (
        "the panel was drawn from the GSM8K test split; %d of %d were found there"
        % (len(in_test), gsm8k.PANEL_MATH_MEMBERS))
    heldout = gsm8k.held_out(test, questions)
    assert len(heldout) == gsm8k.HELD_OUT_N and gsm8k.contaminated(heldout, questions) == []
