"""kit/beds/finqa.py: the scoring rule, answer extraction, the reward function's shape, and no gold in feedback."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("kit_bed_finqa", ROOT / "kit" / "beds" / "finqa.py")
finqa = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(finqa)

ITEM = {"id": "ACME/2020/page_7.pdf-1", "pre_text": ["revenue rose in 2020 ."], "post_text": ["see note 4 ."],
        "table": [["", "2020", "2019"], ["net revenue", "5829", "5735"]],
        "qa": {"question": "what is the net change in net revenue during 2020?", "answer": "94", "exe_ans": 94.0, "program": "subtract(5829, 5735)"}}


@pytest.mark.parametrize("prediction, gold, want", [
    ("94", "94.0", True), ("95.5", "94.0", False),
    ("14.1%", "0.141", True), ("0.141", "0.141", True), ("14.1", "0.141", True),       # percent shown, decimal executed
    ("14%", "0.14464", True), ("7%", "0.06757", True), ("14.5", "14.464", True),       # written rounded, as analysts do
    ("15", "14.464", False), ("0", "0.4", False), ("1", "0.4", False),                  # rounding never rescues a wrong or tiny value
    ("(1,234.5)", "-1234.5", True), ("$ 1,234.5", "1234.5", True), ("-3%", "-0.02918", True),
    ("yes", "yes", True), ("no", "yes", False), ("94", "yes", False), ("yes", "94.0", False),
])
def test_scoring_rule(prediction, gold, want):
    assert finqa.is_correct(finqa.extract_answer("Answer: " + prediction), gold) is want


def test_the_last_answer_line_wins_and_a_missing_one_is_a_format_failure():
    assert float(finqa.extract_answer("first Answer: 3\nthen I reconsidered.\nAnswer: 94")) == 94.0
    assert finqa.extract_answer("the change is 94") is None
    result = finqa.compute_score("finqa", "the change is 94", "94.0")
    assert result["score"] == 0.0 and result["incorrect_format"] == 1 and "Answer:" in result["feedback"]


def test_reward_function_has_the_reference_shape_and_never_reveals_the_gold():
    right = finqa.compute_score("finqa", "5829 - 5735 = 94\nAnswer: 94", "94.0")
    assert right == {"score": 1.0, "acc": 1.0, "pred": "94.0", "incorrect_format": 0, "feedback": ""}
    for response, gold in (("Answer: 12", "94.0"), ("Answer: yes", "94.0"), ("no idea", "731.25")):
        wrong = finqa.compute_score("finqa", response, gold)
        assert set(wrong) == {"score", "acc", "pred", "incorrect_format", "feedback"} and wrong["score"] == 0.0
        assert gold.split(".")[0] not in wrong["feedback"], "feedback must never contain the answer"
    # a yes/no question: the feedback may say the answer is a yes or a no, never WHICH; it is identical for both golds
    assert finqa.compute_score("finqa", "Answer: 3", "yes")["feedback"] == finqa.compute_score("finqa", "Answer: 3", "no")["feedback"]
    wrong_binary = finqa.compute_score("finqa", "Answer: no", "yes")
    assert wrong_binary["score"] == 0.0 and wrong_binary["feedback"] == finqa.compute_score("finqa", "Answer: yes", "no")["feedback"]


def test_prompt_carries_the_table_the_question_and_the_format_it_will_be_judged_on():
    prompt = finqa.render_prompt(ITEM)
    assert "| net revenue | 5829 | 5735 |" in prompt and ITEM["qa"]["question"] in prompt and "Answer: <number>" in prompt
    row = finqa.rows_for_trainer([ITEM], "train")[0]
    assert row["data_source"] == "finqa" and row["reward_model"]["ground_truth"] == "94.0" and row["prompt"][0]["role"] == "user"
    assert row["extra_info"]["gold_consistent"] is True


def test_label_problems_are_flagged():
    broken = json.loads(json.dumps(ITEM)); broken["qa"]["answer"] = "11.9%"; broken["qa"]["exe_ans"] = -0.11825
    empty = json.loads(json.dumps(ITEM)); empty["qa"]["answer"] = ""
    assert finqa.gold_consistent(ITEM) and not finqa.gold_consistent(broken) and not finqa.gold_consistent(empty)


def test_a_dataset_of_the_wrong_size_is_refused(tmp_path):
    (tmp_path / "test.json").write_text(json.dumps([ITEM]))
    with pytest.raises(SystemExit, match="not the pinned dataset"):
        finqa.load(tmp_path, "test")
