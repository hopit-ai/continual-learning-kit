"""The budget audit: kit/canonical.py, kit/cap_sweep.py and kit/budget_report.py, with no GPU, network or transformers.

The tokenizer is replaced by one whose every id is one character (`chr(id)`), so a budget of N tokens is the first N
characters of the answer and every expected score can be read off the test text. Scoring folders are built by hand in
the shape kit/eval_bed.py and kit/score_forgetting.py write them, with ids packed by kit/tokens_io.py.
"""
from __future__ import annotations

import importlib.util
import inspect
import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
KIT = ROOT / "kit"


def _load(name: str):
    spec = importlib.util.spec_from_file_location("kit_%s_budget_test" % name, KIT / ("%s.py" % name))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


canonical = _load("canonical")
cap_sweep = _load("cap_sweep")
budget_report = _load("budget_report")
eval_bed = _load("eval_bed")
tokens_io = _load("tokens_io")
score_forgetting = _load("score_forgetting")
GSM8K = eval_bed.load_bed("gsm8k")


class CharTokenizer:
    """One id per character, so prefixes of ids are prefixes of text."""

    def decode(self, ids, skip_special_tokens=True):
        assert skip_special_tokens is True
        return "".join(chr(i) for i in ids)


@pytest.fixture(autouse=True)
def fake_tokenizer(monkeypatch):
    monkeypatch.setattr(cap_sweep, "load_tokenizer", lambda model_dir: CharTokenizer())
    monkeypatch.setitem(sys.modules, "transformers", None)        # any import of transformers would now fail


# ------------------------------------------------------------------------------------- canonical
@pytest.mark.parametrize("text, expected", [
    ("Answer: 18", "18"),
    ("so the answer is $18.50", "18.50"),
    ("Final Answer = 18", "18"),
    ("the final answer would be 18%", "18"),
    ("The answer should be 7 apples.", "7"),
    ("The answers are 3 and 4.", "3"),
    ("**Answer**: 1,800", "1,800"),
    ("We get \\boxed{18 \\text{ eggs}}.", "18"),
    ("work\n#### 18", "18"),
    ("  ####  $-4.5", "-4.5"),
])
def test_every_marker_form(text, expected):
    assert canonical.extract("gsm8k", text) == expected


@pytest.mark.parametrize("text, expected", [
    ("Answer: 5\nchecking...\nSo the answer is 7.", "7"),
    ("\\boxed{5}\n#### 6", "6"),
    ("Answer: 5, written as \\boxed{9}", "9"),            # by position: the box comes after the marker
    ("#### 2\nAnswer: 3", "3"),
    ("Answer: 3\n#### Step 2: add 5", "3"),                 # a markdown heading is not a statement
])
def test_the_last_statement_wins(text, expected):
    assert canonical.extract("gsm8k", text) == expected


@pytest.mark.parametrize("text", [
    "", "I computed 18.", "3 + 15 = 18\n18", "To find the answer, add 5 and 13 to get 18.",
    "the answer isn't 5, it is 18", "So, the answer is", "Final Answer in the form `Answer: <number>`",
])
def test_no_candidate_and_never_the_last_number(text):
    assert canonical.extract("gsm8k", text) is None


@pytest.mark.parametrize("text, expected", [
    ("Answer: 5\n\nWait, let me re-check. So, the answer is", "5"),                   # cut right after a marker
    ("So the answer is 594 feet.\n\nFinal Answer in the form `Answer: <number>`", "594"),  # a quoted instruction states nothing
    ("Answer: 5\n\\boxed{x}", "5"), ("Answer: 5\nThe answer is eighteen.", "5"),   # a marker with no number is skipped
    ("the answer is:\n18", None),                                                    # the number is on a later line: no candidate
    ("Answer: 5\nBut wait, the answer is 7 or so.\nThe answer is", "7"),              # the last statement WITH a candidate
])
def test_a_marker_that_states_nothing_cannot_erase_an_answer_already_given(text, expected):
    """Rule 4 of kit-canonical.v2. Under v1 a text cut after "the answer is" lost the answer stated before it."""
    assert canonical.extract("gsm8k", text) == expected
    assert canonical.RULE == "kit-canonical.v3"          # v3 is v2 plus two beds; the numeric rules are unchanged


def test_the_reading_is_gold_blind_and_judged_by_the_bed():
    assert list(inspect.signature(canonical.extract).parameters) == ["bed", "text"]
    text = "She has 5 apples and buys 10 more, so the answer is 15."
    assert canonical.extract("gsm8k", text) == "15"
    assert canonical.is_correct("gsm8k", "15", "15") and not canonical.is_correct("gsm8k", canonical.extract("gsm8k", text), "5")
    assert not canonical.is_correct("gsm8k", None, "5") and not canonical.is_correct("finqa", None, "yes")
    assert canonical.is_correct("gsm8k", "1,800", "1800") and canonical.is_correct("gsm8k", "18.00", "18")
    # where both rules read the same line they agree
    for text in ("Answer: 18 eggs", "Answer: $18.00", "Answer: 3 out of 18", "\\boxed{440}"):
        assert canonical.extract("gsm8k", text) == GSM8K.extract_answer(text)
    with pytest.raises(ValueError, match="no canonical rule"):
        canonical.extract("spider", "Answer: 1")
    assert canonical.BEDS == ("gsm8k", "finqa", "chemistry", "toolalpaca")


# ------------------------------------------------------------------ canonical v3: multiple choice (rule 6)
@pytest.mark.parametrize("text, expected", [
    ("<reasoning>r</reasoning>\n<answer>\nC\n</answer>", "C"),
    ("<answer>A) acetone</answer>", "A"),
    ("<answer>(B)</answer>", "B"),
    ("<answer>**C**</answer>", "C"),
    ("<answer>D.</answer>", "D"),
    ("<answer>Option: (B)</answer>", "B"),
    ("<answer>choice D</answer>", "D"),
    ("<answer>`A`</answer>", "A"),
    ("<answer>[C]</answer>", "C"),
    ("The answer is B.", "B"),
    ("**Answer**: C", "C"),
    ("so we pick \\boxed{D}", "D"),
    ("<answer>Answer: C</answer>", "C"),                    # the tag's own content has no letter; the marker inside it does
    ("<reasoning>... so the answer is C", "C"),              # cut before any tag: the marker states it
])
def test_chemistry_every_statement_form(text, expected):
    assert canonical.extract("chemistry", text) == expected


@pytest.mark.parametrize("text, expected", [
    ("<answer>C</answer> wait, <answer>B</answer>", "B"),
    ("The answer is A.\n<answer>D</answer>", "D"),
    ("<answer>D</answer>\nOn reflection the answer is B.", "B"),
    ("<answer>B</answer> then <answer>Acetone</answer>", "B"),       # a letterless later tag cannot erase B
    ("<answer>B</answer>\nSo the answer is", "B"),
    ("<answer>A</answer> but <answer>", "A"),                         # an empty unclosed tag states nothing
])
def test_chemistry_the_last_statement_with_a_letter_wins(text, expected):
    assert canonical.extract("chemistry", text) == expected


@pytest.mark.parametrize("text, expected", [
    ("<reasoning>because</reasoning><answer>C", "C"),                  # unclosed: the content runs to the end
    ("<answer>\n  B and then some more reasoning that was cut", "B"),
    ("<answer>", None),
])
def test_chemistry_an_unclosed_answer_tag(text, expected):
    assert canonical.extract("chemistry", text) == expected


@pytest.mark.parametrize("text", [
    "", "<answer>Acetone</answer>", "<answer>a</answer>", "<answer>b)</answer>", "The answer is acetone.",
    "<answer>B12</answer>", "I think C is plausible.", "Option C looks good",  # no statement, so no candidate
    "<answer>Choice is hard</answer>", "<answer>E</answer>",
])
def test_chemistry_lowercase_words_and_bare_mentions_are_not_candidates(text):
    assert canonical.extract("chemistry", text) is None


def test_chemistry_is_gold_blind_and_compares_letters_exactly():
    assert canonical.is_correct("chemistry", "C", "C") and not canonical.is_correct("chemistry", "C", "D")
    assert not canonical.is_correct("chemistry", None, "C")
    assert [s[1] for s in canonical.statements("chemistry", "Answer: A <answer>B</answer> \\boxed{C}")] == ["answer", "<answer>", "boxed"]
    assert canonical.statements("chemistry", "#### 4") == []           # `####` is a numeric form only


# ---------------------------------------------------------------------- canonical v3: tool calls (rule 7)
def _tool_gold(*calls):
    return json.dumps([{"Action": a, "Action_Input": json.dumps(i) if as_string else i} for a, i, as_string in calls])


def test_toolalpaca_a_flat_call_reads_as_the_authors_read_it():
    text = 'Thought: t\nAction: search\nAction Input: {"q": "cats"}'
    candidate = canonical.extract("toolalpaca", text)
    assert json.loads(candidate) == {"actions": ["search"], "inputs": {"q": "cats"}}
    assert canonical.is_correct("toolalpaca", candidate, _tool_gold(("search", {"q": "cats"}, True)))
    assert canonical.is_correct("toolalpaca", candidate, _tool_gold(("search", {"q": "cats"}, False)))   # an object gold
    assert not canonical.is_correct("toolalpaca", candidate, _tool_gold(("search", {"q": "dogs"}, True)))
    assert not canonical.is_correct("toolalpaca", candidate, _tool_gold(("lookup", {"q": "cats"}, True)))


def test_toolalpaca_nested_objects_and_braces_in_strings_are_read_whole():
    """The authors' `Action Input:\\s*({.*?})` stops at the first `}`, so these correct calls score 0 there."""
    nested = 'Action: search\nAction Input: {"filter": {"kind": "cat"}, "n": 2}\nObservation: ...'
    gold = _tool_gold(("search", {"filter": {"kind": "cat"}, "n": 2}, True))
    assert canonical.is_correct("toolalpaca", canonical.extract("toolalpaca", nested), gold)
    brace = 'Action: note\nAction Input: {"text": "a } b"}'
    assert canonical.is_correct("toolalpaca", canonical.extract("toolalpaca", brace), _tool_gold(("note", {"text": "a } b"}, True)))


def test_toolalpaca_every_call_counts_and_inputs_merge_in_order():
    text = ('Action: a\nAction Input: {"x": 1, "y": 1}\nAction: b\nAction Input: {"y": 2}\n'
            'Action: a\nAction Input: not json\nAction: c\nAction Input: {broken')
    candidate = json.loads(canonical.extract("toolalpaca", text))
    assert candidate == {"actions": ["a", "a", "b", "c"], "inputs": {"x": 1, "y": 2}}   # malformed inputs skipped
    gold = _tool_gold(("b", {"y": 2}, True), ("a", {"x": 1}, True), ("a", {}, True), ("c", {}, True))
    assert canonical.is_correct("toolalpaca", canonical.extract("toolalpaca", text), gold)
    assert not canonical.is_correct("toolalpaca", canonical.extract("toolalpaca", text), _tool_gold(("a", {"x": 1, "y": 2}, True)))


@pytest.mark.parametrize("text", ["", "I would search for cats.", 'Action Input: {"q": "cats"}', "action: search"])
def test_toolalpaca_no_action_is_no_candidate(text):
    assert canonical.extract("toolalpaca", text) is None


def test_toolalpaca_an_action_with_no_readable_input_has_empty_inputs():
    assert json.loads(canonical.extract("toolalpaca", "Action: ping\nAction Input: none")) == {"actions": ["ping"], "inputs": {}}
    assert canonical.is_correct("toolalpaca", canonical.extract("toolalpaca", "Action: ping"), _tool_gold(("ping", "{}", False)))
    assert not canonical.is_correct("toolalpaca", canonical.extract("toolalpaca", "Action: ping"), "not json")
    assert not canonical.is_correct("toolalpaca", None, _tool_gold(("ping", {}, True)))


def test_the_new_beds_never_see_the_gold():
    assert list(inspect.signature(canonical.extract).parameters) == ["bed", "text"]
    for bed in ("chemistry", "toolalpaca"):                 # each candidate function takes one text and nothing else
        assert len(inspect.signature(canonical.EXTRACTORS[bed]).parameters) == 1
    # the same text gives the same candidate whatever the gold is; only is_correct differs
    text = "<answer>B</answer>"
    assert [canonical.is_correct("chemistry", canonical.extract("chemistry", text), g) for g in "ABCD"] == [False, True, False, False]


@pytest.mark.skipif(not (Path(__import__("os").environ.get("SDPO_DIR") or "/nonexistent") / "verl/utils/reward_score/feedback/tooluse.py").is_file(),
                    reason="needs SDPO_DIR set to the pinned lasgroup/SDPO checkout")
def test_the_new_rules_agree_with_the_authors_where_the_authors_read_the_answer():
    """Where the authors' rule can read an answer at all, the canonical reading gives the same verdict; where it
    cannot (a nested object), the canonical reading is the one that differs, and only there."""
    chemistry, toolalpaca = eval_bed.load_bed("chemistry"), eval_bed.load_bed("toolalpaca")
    for text, gold in (("<answer>C</answer>", "C"), ("<answer>\nB\n</answer>", "C"), ("<answer>C</answer> x <answer>D</answer>", "D"),
                       ("<answer>C", "C")):
        assert chemistry.compute_score("sciknoweval", text, gold)["acc"] == float(canonical.is_correct("chemistry", canonical.extract("chemistry", text), gold))
    flat_gold = _tool_gold(("search", {"q": "cats"}, True), ("send", {"to": "a"}, True))
    for text in ('Action: search\nAction Input: {"q": "cats"}\nAction: send\nAction Input: {"to": "a"}',
                 'Action: search\nAction Input: {"q": "dogs"}', 'Action: send\nAction Input: {"to": "a"}\nAction: search\nAction Input: {"q": "cats"}',
                 "no call at all"):
        assert toolalpaca.compute_score("tooluse", text, flat_gold)["acc"] == float(canonical.is_correct("toolalpaca", canonical.extract("toolalpaca", text), flat_gold))
    nested_gold = _tool_gold(("search", {"filter": {"kind": "cat"}}, True))
    nested = 'Action: search\nAction Input: {"filter": {"kind": "cat"}}'
    assert toolalpaca.compute_score("tooluse", nested, nested_gold)["acc"] == 0.0
    assert canonical.is_correct("toolalpaca", canonical.extract("toolalpaca", nested), nested_gold)


def test_finqa_forms_go_through_the_beds_own_comparison():
    finqa = eval_bed.load_bed("finqa")
    assert canonical.extract("finqa", "so the answer is 14.1%.") == "14.1%"
    assert canonical.is_correct("finqa", "14.1%", "0.141")
    assert canonical.extract("finqa", "Answer: (1,234.5)") == "(1,234.5)"
    assert canonical.is_correct("finqa", "(1,234.5)", "-1234.5") and not canonical.is_correct("finqa", "(1,234.5)", "1234.5")
    assert canonical.is_correct("finqa", canonical.extract("finqa", "the answer is 14%"), "0.14464")    # the rounding rule
    assert canonical.extract("finqa", "The answer is yes.") == "yes" and canonical.is_correct("finqa", "yes", "yes")
    assert canonical.extract("finqa", "\\boxed{-3.2}") == "-3.2" and canonical.is_correct("finqa", "-3.2", "-3.2")
    for text, gold in (("Answer: 14.1%", "0.141"), ("Answer: (5.0)", "-5.0"), ("Answer: no", "no")):
        strict = finqa.is_correct(finqa.extract_answer(text), gold)
        assert strict and canonical.is_correct("finqa", canonical.extract("finqa", text), gold)


# ------------------------------------------------------------------------------- scoring folders
@pytest.fixture()
def data(tmp_path):
    root = tmp_path / "gsm8k"
    root.mkdir()
    (root / "test.jsonl").write_text("".join(json.dumps({"question": "Question %d: what is %d plus zero, last digit only?" % (n, n),
                                                         "answer": "#### %d" % (n % 9)}) + "\n" for n in range(12)))   # no gold is 9
    return root


BED_ARGS = ["--bed", "gsm8k", "--allow-subset", "--heldout-n", "8"]


def items_for(root):
    ns = types.SimpleNamespace(bed="gsm8k", root=str(root), split=None, limit=None, heldout_n=8, allow_subset=True)
    return eval_bed.items_of(GSM8K, ns)


def write_bed_scoring(folder: Path, items: list, answers: dict, cap: int, *, machine="m1", max_model_len=12288,
                      tokens=True, per_item=None, items_sha256=None) -> Path:
    """answers: {id: (text, finish_reason)}; ids are the characters' code points."""
    folder.mkdir(parents=True)
    rows = [{"bed": "gsm8k", "split": "heldout", "id": i["id"], "response": answers[i["id"]][0],
             "output_tokens": len(answers[i["id"]][0]), "finish_reason": answers[i["id"]][1]} for i in items]
    (folder / "responses.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    if tokens:
        tokens_io.write(folder / "tokens.jsonl", [{"id": r["id"], "ids": [ord(c) for c in r["response"]], "prompt_sha256": "0" * 64,
                                                   "prompt_tokens": None} for r in rows])
    if per_item is None:
        per_item = {i["id"]: int(GSM8K.compute_score("gsm8k", answers[i["id"]][0], i["ground_truth"])["acc"]) for i in items}
    result = {"schema": "kit-bed-score.v1", "bed": "gsm8k", "split": "heldout", "n": len(items), "correct": sum(per_item.values()),
              "per_item": per_item, "items_sha256": items_sha256 or eval_bed.items_digest(items), "max_new_tokens": cap,
              "decoding": {"max_tokens": cap}, "tokens_file": "tokens.jsonl", "engine": {"max_model_len": max_model_len},
              "machine": {"id": machine}}
    (folder / "bed-score.json").write_text(json.dumps(result))
    return folder


def standard_answers(items):
    """Item 0: right at 9 tokens, WRONG at 19 (a later Answer: line). Item 1: canonical only. Item 2: cut at the cap."""
    answers = {}
    for index, item in enumerate(items):
        gold = item["ground_truth"]
        assert gold != "9"
        if index == 0:
            answers[item["id"]] = ("Answer: %s\nAnswer: 99" % gold, "stop")             # 20 tokens
        elif index == 1:
            answers[item["id"]] = ("So the answer is %s." % gold, "stop")               # 19 tokens
        elif index == 2:
            answers[item["id"]] = ("Working: " + "a" * 15, "length")                    # 24 tokens, the cap
        else:
            answers[item["id"]] = ("Answer: %s" % gold, "stop")                         # 9 tokens
    return answers


def sweep(tmp_path, scoring, data, budgets="9,19,20,24", out="sweep", extra=()):
    argv = ["bed", "--scoring", str(scoring), "--root", str(data), *BED_ARGS, "--model", "unused", "--budgets", budgets,
            "--out", str(tmp_path / out), *extra]
    assert cap_sweep.main(argv) == 0
    return json.loads((tmp_path / out / "sweep.json").read_text()), \
        [json.loads(line) for line in (tmp_path / out / "per_item.jsonl").read_text().splitlines()]


@pytest.fixture()
def standard(tmp_path, data):
    items = items_for(data)
    answers = standard_answers(items)
    return items, answers, write_bed_scoring(tmp_path / "long", items, answers, cap=24)


def test_a_non_monotone_curve_is_reported_and_cut_flags_follow_the_rule(tmp_path, data, standard):
    items, answers, scoring = standard
    report, rows = sweep(tmp_path, scoring, data)
    by_id = {r["id"]: r for r in rows}
    first, second, third = (by_id[i["id"]] for i in items[:3])
    assert report["budgets"] == [9, 19, 20, 24] and report["cap"] == 24 and report["n"] == 8
    assert first["strict"] == [1, 0, 0, 0] and first["canonical"] == [1, 0, 0, 0]          # right at 9, wrong at 19
    assert first["cut"] == [1, 1, 0, 0]                    # N < n; N == n with finish "stop" is not cut
    assert second["strict"] == [0, 0, 0, 0] and second["canonical"] == [0, 1, 1, 1]
    assert third["cut"] == [1, 1, 1, 1] and third["finish_reason"] == "length"    # N == n == cap with finish "length" is cut
    per = {e["budget"]: e for e in report["per_budget"]}
    assert per[9]["correct_strict"] == 6 and per[19]["correct_strict"] == 5           # the curve falls; nothing assumed it would not
    assert per[24]["correct_canonical"] == 6 and per[24]["cut"] == 1 and per[24]["finished"] == 7
    assert per[9]["tokens_total"] == 9 * 3 + 9 * 5 and per[24]["tokens_total"] == 20 + 19 + 24 + 9 * 5
    assert per[24]["tokens_per_correct_strict"] == round(per[24]["tokens_total"] / 5, 3)
    assert report["reproduces_scoring"] is True and report["decode_equals_text"] == 8
    assert report["max_model_len"] == 12288 and report["machine"] == {"id": "m1"} and report["kind"] == "bed"
    assert len(report["responses_sha256"]) == 64 and len(report["tokens_sha256"]) == 64
    assert "| 19 | 5 |" in (tmp_path / "sweep" / "sweep.md").read_text()


def test_zero_correct_gives_no_tokens_per_correct(tmp_path, data):
    items = items_for(data)
    scoring = write_bed_scoring(tmp_path / "s", items, {i["id"]: ("Answer: 9", "stop") for i in items}, cap=24)
    report, _ = sweep(tmp_path, scoring, data, budgets="4,24")
    assert all(e["correct_strict"] == 0 and e["tokens_per_correct_strict"] is None for e in report["per_budget"])


def test_the_cap_reproduces_the_scoring_or_says_where_not(tmp_path, data, standard):
    items, answers, _ = standard
    wrong = {i["id"]: int(GSM8K.compute_score("gsm8k", answers[i["id"]][0], i["ground_truth"])["acc"]) for i in items}
    wrong[items[4]["id"]] = 1 - wrong[items[4]["id"]]
    scoring = write_bed_scoring(tmp_path / "tampered", items, answers, cap=24, per_item=wrong)
    report, _ = sweep(tmp_path, scoring, data, budgets="9,19")          # the cap is read even when not asked for
    assert report["reproduces_scoring"] is False and report["reproduces_scoring_mismatched_ids"] == [items[4]["id"]]
    assert [e["budget"] for e in report["per_budget"]] == [9, 19]


def test_decode_equals_text_is_counted_not_refused(tmp_path, data, standard):
    items, answers, scoring = standard
    rows = [json.loads(line) for line in (scoring / "responses.jsonl").read_text().splitlines()]
    rows[3]["response"] = rows[3]["response"].replace("Answer", "answer")
    (scoring / "responses.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    report, _ = sweep(tmp_path, scoring, data)
    assert report["decode_equals_text"] == 7


def _refused(tmp_path, scoring, data, match, budgets="9,24", out="refused"):
    with pytest.raises(SystemExit, match=match):
        cap_sweep.main(["bed", "--scoring", str(scoring), "--root", str(data), *BED_ARGS, "--model", "unused",
                        "--budgets", budgets, "--out", str(tmp_path / out)])
    assert not (tmp_path / out).exists() or out == "exists"


def test_refusals(tmp_path, data, standard):
    items, answers, scoring = standard
    _refused(tmp_path, write_bed_scoring(tmp_path / "digest", items, answers, 24, items_sha256="0" * 64), data, "items_sha256")
    missing = write_bed_scoring(tmp_path / "missing", items, answers, 24)
    lines = (missing / "tokens.jsonl").read_text().splitlines()
    (missing / "tokens.jsonl").write_text("\n".join(lines[1:]) + "\n")
    _refused(tmp_path, missing, data, "no token row")
    no_answer = write_bed_scoring(tmp_path / "no-answer", items, answers, 24)
    lines = (no_answer / "responses.jsonl").read_text().splitlines()
    (no_answer / "responses.jsonl").write_text("\n".join(lines[1:]) + "\n")
    _refused(tmp_path, no_answer, data, "have no answer")
    mismatch = write_bed_scoring(tmp_path / "n", items, answers, 24)
    rows = [json.loads(line) for line in (mismatch / "responses.jsonl").read_text().splitlines()]
    rows[2]["output_tokens"] += 1
    (mismatch / "responses.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    _refused(tmp_path, mismatch, data, "disagree with their answer's length")
    _refused(tmp_path, scoring, data, "above the scoring's cap", budgets="9,30")
    _refused(tmp_path, write_bed_scoring(tmp_path / "old", items, answers, 24, tokens=False), data, "no tokens.jsonl")
    (tmp_path / "exists").mkdir()
    _refused(tmp_path, scoring, data, "refusing to overwrite", out="exists")


# ------------------------------------------------------------------------------------------ check
def short_scoring(tmp_path, items, answers, cap, name="short", *, tokens=True, edit=None, machine="m1", max_model_len=12288):
    short = {}
    for key, (text, finish) in answers.items():
        short[key] = (text[:cap], "length" if len(text) > cap else finish)
    if edit:
        key, text = edit
        short[key] = (text, short[key][1])
    return write_bed_scoring(tmp_path / name, items, short, cap, tokens=tokens, machine=machine, max_model_len=max_model_len)


def run_check(tmp_path, long, short, out="check.json"):
    assert cap_sweep.main(["check", "--long", str(long), "--short", str(short), "--model", "unused", "--out", str(tmp_path / out)]) == 0
    return json.loads((tmp_path / out).read_text())


def test_check_agrees_on_true_prefixes(tmp_path, standard):
    items, answers, long = standard
    report = run_check(tmp_path, long, short_scoring(tmp_path, items, answers, 9))
    assert report["compared"] == 8 and report["text_agree"] == 8 and report["text_agreement"] == 1.0
    assert report["by_short_finish"]["cut"]["n"] == 3 and report["text_agreement_among_cut"] == 1.0
    assert report["ids_agree"] == 8 and report["mode"] == "text and ids" and report["mismatches"] == []
    assert report["same_machine"] and report["same_max_model_len"] and report["short_cap"] == 9 and report["long_cap"] == 24


def test_check_reports_a_disagreement_with_its_offset(tmp_path, standard):
    items, answers, long = standard
    key = items[0]["id"]
    edited = answers[key][0][:9][:4] + "X" + answers[key][0][5:9]
    report = run_check(tmp_path, long, short_scoring(tmp_path, items, answers, 9, edit=(key, edited), machine="m2", max_model_len=4096))
    assert report["text_agree"] == 7 and report["text_agreement_among_cut"] == pytest.approx(2 / 3, abs=1e-6)
    (miss,) = report["mismatches"]
    assert miss["id"] == key and miss["first_difference_offset"] == 4 and miss["short_finish"] == "cut"
    assert "X" in miss["short_context"] and "X" not in miss["long_prefix_context"]
    assert report["same_machine"] is False and report["same_max_model_len"] is False
    assert report["max_model_len"] == {"long": 12288, "short": 4096}


def test_check_works_on_text_alone_and_refuses_what_it_cannot_compare(tmp_path, data, standard):
    items, answers, long = standard
    report = run_check(tmp_path, long, short_scoring(tmp_path, items, answers, 9, tokens=False))
    assert report["mode"] == "text only" and report["ids_agree"] is None and report["text_agree"] == 8
    with pytest.raises(SystemExit, match="other way round"):
        cap_sweep.main(["check", "--long", str(short_scoring(tmp_path, items, answers, 9, name="s2")), "--short", str(long),
                        "--model", "unused", "--out", str(tmp_path / "swapped.json")])
    other = write_bed_scoring(tmp_path / "other", items, answers, 9, items_sha256="1" * 64)
    with pytest.raises(SystemExit, match="same items"):
        cap_sweep.main(["check", "--long", str(long), "--short", str(other), "--model", "unused", "--out", str(tmp_path / "x.json")])
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        run_check(tmp_path, long, short_scoring(tmp_path, items, answers, 9, name="s3"), out="check.json")


# ------------------------------------------------------------------------------------------ panel
def test_the_panel_sweep_on_the_real_panel(tmp_path):
    members, digest = score_forgetting.load_panel()
    cap, rows, token_rows = 30, [], []
    for index, m in enumerate(members):
        if m["panel"] in ("math", "knowledge"):
            text, finish = ("Answer: %s" % m["answer"], "stop") if index % 2 == 0 else ("I do not know.", "stop")
        else:
            text, finish = "", "stop"
        if index == 1:                                                       # one math answer cut at the cap
            text, finish = ("Answer: %s" % m["answer"] + " " * cap)[:cap], "length"
        rows.append({"panel": m["panel"], "id": m["id"], "response": text, "output_tokens": len(text), "finish_reason": finish})
        token_rows.append({"panel": m["panel"], "id": m["id"], "ids": [ord(c) for c in text]})
    folder = tmp_path / "panel-scoring"
    folder.mkdir()
    (folder / "responses.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    tokens_io.write(folder / "tokens.jsonl", token_rows)
    panels = {}
    for m, r in zip(members, rows):
        slot = panels.setdefault(m["panel"], {"per_member": {}})
        slot["per_member"][m["id"]] = int(score_forgetting.scorers.score(m["panel"], r["response"], m)["correct"])
    (folder / "forgetting.json").write_text(json.dumps({"panel_file_sha256": digest, "panels": panels, "max_new_tokens": cap,
                                                        "engine": {"max_model_len": 4096 + cap}, "machine": {"id": "m1"}}))
    assert cap_sweep.main(["panel", "--scoring", str(folder), "--model", "unused", "--budgets", "5,30", "--out", str(tmp_path / "ps")]) == 0
    report = json.loads((tmp_path / "ps" / "sweep.json").read_text())
    expected = {name: sum(slot["per_member"].values()) for name, slot in panels.items()}
    at5, at30 = report["per_budget"]
    assert {name: p["correct"] for name, p in at30["panels"].items()} == expected and expected["math"] > 0
    assert at5["total"]["correct"] == 0 and at30["panels"]["math"]["cut"] == 1 and at30["total"]["n"] == 300
    assert report["reproduces_scoring"] is True and report["decode_equals_text"] == 300 and report["kind"] == "panel"
    per_item = [json.loads(line) for line in (tmp_path / "ps" / "per_item.jsonl").read_text().splitlines()]
    assert len(per_item) == 300 and per_item[1]["cut"] == [1, 1]
    # the bed subcommand refuses a panel folder
    with pytest.raises(SystemExit, match="panel scoring"):
        cap_sweep.main(["bed", "--scoring", str(folder), "--bed", "gsm8k", "--root", str(tmp_path), "--model", "unused",
                        "--budgets", "5", "--out", str(tmp_path / "nope")])


# --------------------------------------------------------------------------------- budget report
def write_sweep(folder: Path, points: dict, *, n=100, machine="m1", items="i" * 64, bed="gsm8k") -> Path:
    """points: {budget: (strict, canonical, cut, tokens_per_correct)}."""
    folder.mkdir(parents=True)
    per_budget = [{"budget": b, "n": n, "correct_strict": s, "correct_canonical": e, "cut": c, "finished": n - c,
                   "tokens_total": 0, "tokens_per_correct_strict": t} for b, (s, e, c, t) in sorted(points.items())]
    (folder / "sweep.json").write_text(json.dumps({"schema": cap_sweep.SCHEMA, "kind": "bed", "bed": bed, "n": n, "items_sha256": items,
                                                   "cap": max(points), "machine": {"id": machine}, "per_budget": per_budget,
                                                   "scoring": "/node/work/eval/%s-a1" % folder.name}))
    return folder


REF = {512: (50, 55, 30, 300.0), 2048: (80, 85, 5, 400.0), 8192: (82, 88, 0, 410.0)}


def after(strict_b, canon_b, strict_h, canon_h, cut_h, tokens_b=800.0):
    return {512: (10, 20, 60, 900.0), 2048: (strict_b, canon_b, 40, tokens_b), 8192: (strict_h, canon_h, cut_h, 900.0)}


def report_of(tmp_path, afters: dict, out="report", extra=()):
    ref = write_sweep(tmp_path / "ref", REF) if not (tmp_path / "ref").exists() else tmp_path / "ref"
    argv = ["--ref", str(ref), "--serving", "2048", "--diagnostic", "8192", "--seeds-label", "test seeds", "--out", str(tmp_path / out), *extra]
    for name, points in afters.items():
        folder = tmp_path / ("sweep-%s-%s" % (out, name))
        write_sweep(folder, points) if isinstance(points, dict) else None
        argv += ["--after", "%s=%s" % (name, folder if isinstance(points, dict) else points)]
    assert budget_report.main(argv) == 0
    return json.loads((tmp_path / out / "budget-report.json").read_text())


def test_the_accounting_identity_with_three_non_zero_terms(tmp_path):
    report = report_of(tmp_path, {"seed0": after(60, 70, 78, 84, 0)})
    (r,) = report["records"]
    assert (r["F"], r["residual"], r["budget"], r["extraction"]) == (20, 4, 11, 5)
    assert r["F"] == r["residual"] + r["budget"] + r["extraction"] and report["accounting_exact"] == 1
    assert r["tokens_per_correct_serving"] == {"ref": 400.0, "after": 800.0, "ratio": 2.0}
    assert r["label"] == "intact" and r["tolerance"] == 5 and report["scorings_reported"] == 1
    assert report["budgets_in_all_sweeps"] == [512, 2048, 8192] and report["curve"][0]["seed0"]["strict_minus_ref"] == -40
    md = (tmp_path / "report" / "budget-report.md").read_text()
    assert "not causes" in md and "| seed0 | +20 | +4 | +11 | +5 |" in md and "Neither proves" in md and "test seeds" in md


def test_seed_labels_at_their_boundaries(tmp_path):
    report = report_of(tmp_path, {
        "intact-edge": after(60, 70, 77, 80, 10),          # 10 of 100 cut is not MORE than 10 percent; 77 >= 82 - 5
        "degraded": after(60, 70, 76, 80, 10),
        "still-cut": after(60, 70, 82, 88, 11),            # more than 10 percent cut outranks a recovered score
    })
    assert report["seed_labels"] == {"intact-edge": "intact", "degraded": "degraded", "still-cut": "still cut"}
    assert report["bed_label"] == "mixed or inconclusive"
    assert budget_report.seed_label(n=300, cut_h=30, strict_h=100, ref_strict_h=115) == "intact"     # ceil(0.05 x 300) = 15
    assert budget_report.seed_label(n=301, cut_h=30, strict_h=100, ref_strict_h=116) == "intact"     # ceil(15.05) = 16
    assert budget_report.seed_label(n=301, cut_h=30, strict_h=99, ref_strict_h=116) == "degraded"
    assert budget_report.seed_label(n=301, cut_h=31, strict_h=116, ref_strict_h=116) == "still cut"  # 31 > 30.1


@pytest.mark.parametrize("labels, expected", [
    (["intact"] * 4 + ["degraded"], "intact"),
    (["intact"] * 3 + ["degraded"] * 2, "mixed or inconclusive"),
    (["degraded"] * 3 + ["intact"] * 2, "degraded"),
    (["degraded"] * 2 + ["still cut"] * 3, "mixed or inconclusive"),
    (["intact"], "intact"), (["still cut"], "mixed or inconclusive"),
    (["intact", "degraded"], "mixed or inconclusive"), (["degraded", "degraded"], "degraded"),
])
def test_the_per_bed_rule(labels, expected):
    assert budget_report.bed_label(labels) == expected


def test_the_per_bed_label_end_to_end(tmp_path):
    good, bad = after(60, 70, 80, 84, 0), after(60, 70, 70, 75, 0)
    report = report_of(tmp_path, {"s0": good, "s1": good, "s2": good, "s3": good, "s4": bad})
    assert report["bed_label"] == "intact" and report["label_counts"] == {"intact": 4, "degraded": 1, "still cut": 0}
    report = report_of(tmp_path, {"s0": bad, "s1": bad, "s2": bad, "s3": good, "s4": good}, out="report2")
    assert report["bed_label"] == "degraded"


def test_budget_report_refusals(tmp_path):
    write_sweep(tmp_path / "ref", REF)
    base = ["--ref", str(tmp_path / "ref"), "--serving", "2048", "--diagnostic", "8192"]
    cases = [
        ("other-machine", {"machine": "m2"}, "one machine"),
        ("other-items", {"items": "j" * 64}, "other items"),
        ("other-bed", {"bed": "finqa"}, "finqa bed"),
    ]
    for name, kwargs, match in cases:
        write_sweep(tmp_path / name, after(60, 70, 78, 84, 0), **kwargs)
        with pytest.raises(SystemExit, match=match):
            budget_report.main(base + ["--after", "a=%s" % (tmp_path / name), "--out", str(tmp_path / ("out-" + name))])
        assert not (tmp_path / ("out-" + name)).exists()
    write_sweep(tmp_path / "no-h", {512: (1, 1, 0, None), 2048: (60, 70, 0, 1.0)})
    with pytest.raises(SystemExit, match="did not read budget 8192"):
        budget_report.main(base + ["--after", "a=%s" % (tmp_path / "no-h"), "--out", str(tmp_path / "o1")])
    with pytest.raises(SystemExit, match="must be below"):
        budget_report.main(["--ref", str(tmp_path / "ref"), "--serving", "8192", "--diagnostic", "2048",
                            "--after", "a=%s" % (tmp_path / "other-machine"), "--out", str(tmp_path / "o2")])
    panel = tmp_path / "panel-sweep"; panel.mkdir()
    (panel / "sweep.json").write_text(json.dumps({"kind": "panel"}))
    with pytest.raises(SystemExit, match="not a bed sweep"):
        budget_report.main(base + ["--after", "a=%s" % panel, "--out", str(tmp_path / "o3")])
    (tmp_path / "taken").mkdir()
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        budget_report.main(base + ["--after", "a=%s" % (tmp_path / "other-machine"), "--out", str(tmp_path / "taken")])
    # different machines are compared only on request, and the request is recorded
    report = report_of(tmp_path, {"a": str(tmp_path / "other-machine")}, out="allowed", extra=["--allow-different-machines"])
    assert report["same_machine"] is False and report["different_machines_allowed"] is True
    assert "NOT all made on one machine" in (tmp_path / "allowed" / "budget-report.md").read_text()


# ------------------------------------------------------------------------- the sweep feeds the report
def test_two_real_sweeps_feed_the_report(tmp_path, data, standard):
    items, answers, ref_scoring = standard
    later = {key: ("Working: " + "b" * 15, "length") if index < 2 else (text, finish)
             for index, (key, (text, finish)) in enumerate(answers.items())}
    after_scoring = write_bed_scoring(tmp_path / "after", items, later, cap=24)
    sweep(tmp_path, ref_scoring, data, budgets="9,24", out="ref-sweep")
    sweep(tmp_path, after_scoring, data, budgets="9,24", out="after-sweep")
    assert budget_report.main(["--ref", str(tmp_path / "ref-sweep"), "--after", "s0=%s" % (tmp_path / "after-sweep"),
                               "--serving", "9", "--diagnostic", "24", "--out", str(tmp_path / "br")]) == 0
    report = json.loads((tmp_path / "br" / "budget-report.json").read_text())
    (r,) = report["records"]
    assert report["accounting_exact"] == 1 and r["F"] == r["residual"] + r["budget"] + r["extraction"]
    assert r["label"] == "still cut"                                 # 3 of 8 answers are cut at the diagnostic budget


def test_the_label_is_also_given_on_the_canonical_reading(tmp_path):
    """A checkpoint whose strict score is far below the reference but whose stated answers are right reads "degraded"
    by the label of record and "intact" by the canonical one; the report prints both and never merges them."""
    def sweep(name, strict, canon, cut=0, n=100):
        d = tmp_path / name; d.mkdir()
        per = [{"budget": b, "n": n, "correct_strict": strict, "correct_canonical": canon, "cut": cut, "finished": n - cut,
                "tokens_total": 1000, "tokens_per_correct_strict": (1000 / strict if strict else None)} for b in (2048, 8192)]
        (d / "sweep.json").write_text(json.dumps({"schema": "kit-cap-sweep.v1", "kind": "bed", "bed": "gsm8k", "n": n, "cap": 8192, "budgets": [2048, 8192],
                                                  "items_sha256": "x" * 64, "machine": {"id": "m1"}, "per_budget": per}))
        return d
    ref, after = sweep("ref", 80, 82), sweep("after", 40, 81)
    out = tmp_path / "report"
    assert budget_report.main(["--ref", str(ref), "--after", "seed0=%s" % after, "--serving", "2048", "--diagnostic", "8192", "--out", str(out)]) == 0
    report = json.loads((out / "budget-report.json").read_text())
    record = report["records"][0]
    assert record["label"] == "degraded" and record["label_canonical"] == "intact"
    assert report["bed_label"] == "degraded" and report["bed_label_canonical"] == "intact"
    assert "canonical reading gives: seed0 intact" in (out / "budget-report.md").read_text()


def _check_file(path: Path, agreement, *, of, compared=100, same_machine=True, same_context=True, short_cap=2048, long_cap=8192) -> Path:
    """`of`: the sweep folder name the check belongs to (report_of names them `sweep-<out>-<seed>`, the reference `ref`)."""
    path.write_text(json.dumps({"compared": compared, "text_agreement": agreement, "same_machine": same_machine, "same_max_model_len": same_context,
                                "long": "/moved/elsewhere/eval/%s-a1" % of, "short": "/x", "short_cap": short_cap, "long_cap": long_cap}))
    return path


def test_a_scoring_whose_prefix_check_failed_is_left_out_and_named(tmp_path):
    """The pre-registered rule (prefix reuse must agree on 99 percent of answers, or that checkpoint's curve is not
    used) was written down and not enforced: found by the review of the send. It is enforced here."""
    good = _check_file(tmp_path / "good.json", 0.995, of="sweep-report-seed0")
    bad = _check_file(tmp_path / "bad.json", 0.97, of="sweep-report-seed1")
    report = report_of(tmp_path, {"seed0": after(60, 70, 78, 84, 0), "seed1": after(70, 75, 80, 85, 0)},
                       extra=("--qualify", "seed0=%s" % good, "--qualify", "seed1=%s" % bad))
    assert [r["name"] for r in report["records"]] == ["seed0"] and report["scorings_reported"] == 1 and report["scorings_disqualified"] == 1
    assert (report["disqualified"][0]["name"], report["disqualified"][0]["text_agreement"], report["disqualified"][0]["why"]) == ("seed1", 0.97, "failed")
    assert report["prefix_checks"]["seed0"]["passed"] is True and report["prefix_checks"]["seed1"]["passed"] is False
    md = (tmp_path / "report" / "budget-report.md").read_text()
    assert "**Left out: seed1 (failed, agreement 0.97).**" in md and "| seed1 |" not in md
    # with no --qualify nothing changes: the report is the one it always was
    plain = report_of(tmp_path, {"seed0": after(60, 70, 78, 84, 0)}, out="plain")
    assert plain["disqualified"] == [] and plain["prefix_checks"] == {} and "Left out" not in (tmp_path / "plain" / "budget-report.md").read_text()


@pytest.mark.parametrize("kwargs", [{"same_machine": False}, {"same_context": False}, {"compared": 0}])
def test_a_prefix_check_passes_only_on_one_machine_one_context_and_some_answers(tmp_path, kwargs):
    check = _check_file(tmp_path / "c.json", 1.0, of="sweep-report-seed0", **kwargs)
    with pytest.raises(SystemExit, match="no --after scoring is qualified by its prefix check"):
        report_of(tmp_path, {"seed0": after(60, 70, 78, 84, 0)}, extra=("--qualify", "seed0=%s" % check))


def test_a_named_check_that_is_missing_or_of_another_scoring_does_not_qualify(tmp_path):
    """Round 2 of the review: a missing check file used to be skipped (the campaign passed no --qualify for it), and a
    check carried no identity, so an old passing check qualified a newly made scoring."""
    other = _check_file(tmp_path / "other.json", 1.0, of="sweep-report-seed9")              # passes, but of another scoring
    short = _check_file(tmp_path / "short.json", 1.0, of="sweep-report-seed2", short_cap=1024)
    good = _check_file(tmp_path / "good.json", 1.0, of="sweep-report-seed3")
    report = report_of(tmp_path, {"seed0": after(60, 70, 78, 84, 0), "seed1": after(61, 70, 78, 84, 0), "seed2": after(62, 70, 78, 84, 0),
                                  "seed3": after(63, 70, 78, 84, 0), "seed4": after(64, 70, 78, 84, 0)},
                       extra=("--qualify", "seed0=%s" % (tmp_path / "never-written.json"), "--qualify", "seed1=%s" % other,
                              "--qualify", "seed2=%s" % short, "--qualify", "seed3=%s" % good))
    assert [r["name"] for r in report["records"]] == ["seed3", "seed4"], "seed4 was given no check and is reported as before"
    why = {d["name"]: d["why"] for d in report["disqualified"]}
    assert why["seed0"] == "missing" and why["seed1"].startswith("of another scoring (sweep-report-seed9-a1") and why["seed2"].startswith("of another scoring")
    assert report["prefix_checks"]["seed3"]["bound"] is True and report["prefix_checks"]["seed3"]["long_scoring"] == "sweep-report-seed3-a1"


def test_prefix_check_refusals(tmp_path):
    bad = _check_file(tmp_path / "bad.json", None, of="ref")
    with pytest.raises(SystemExit, match="the reference's prefix check does not qualify it"):
        report_of(tmp_path, {"seed0": after(60, 70, 78, 84, 0)}, out="r1", extra=("--qualify", "ref=%s" % bad))
    with pytest.raises(SystemExit, match="names no scoring of this report"):
        report_of(tmp_path, {"seed0": after(60, 70, 78, 84, 0)}, out="r2", extra=("--qualify", "seed9=%s" % bad))
    with pytest.raises(SystemExit, match="the reference's prefix check does not qualify it .*missing"):
        report_of(tmp_path, {"seed0": after(60, 70, 78, 84, 0)}, out="r3", extra=("--qualify", "ref=%s" % (tmp_path / "absent.json")))
    assert not any((tmp_path / name).exists() for name in ("r1", "r2", "r3"))
    ok = _check_file(tmp_path / "ok.json", 1.0, of="ref")
    assert report_of(tmp_path, {"seed0": after(60, 70, 78, 84, 0)}, out="r4", extra=("--qualify", "ref=%s" % ok))["prefix_checks"]["ref"]["passed"] is True


def test_a_sweep_records_the_model_that_was_scored(tmp_path, data, standard):
    """A report refuses a sweep of an earlier attempt of a run that was later retried, so the sweep must say which
    model folder its scoring was made on: a path for a bed result, {"path": ...} for a panel result, None if unsaid."""
    items, answers, scoring = standard
    report, _ = sweep(tmp_path, scoring, data)
    assert "model" in report and report["model"] is None                    # this fixture's scoring does not say
    result = json.loads((scoring / "bed-score.json").read_text())
    result["model"] = "/work/runs/g8-chem-r1-a2/hf-step40"
    (scoring / "bed-score.json").write_text(json.dumps(result))
    assert sweep(tmp_path, scoring, data, out="s2")[0]["model"] == "/work/runs/g8-chem-r1-a2/hf-step40"
    result["model"] = {"path": "/work/models/Qwen3-8B", "files": {}}
    (scoring / "bed-score.json").write_text(json.dumps(result))
    assert sweep(tmp_path, scoring, data, out="s3")[0]["model"] == "/work/models/Qwen3-8B"
