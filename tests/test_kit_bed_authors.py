"""kit/beds/chemistry.py and kit/beds/toolalpaca.py: the SDPO authors' two tasks as scored beds (plan v3, package 3).

They read the trainer-format files the authors' preprocessing writes and score with the authors' own functions.
Loading and refusing are tested on small files written here; the scoring tests need the pinned checkout (named by
SDPO_DIR) and skip without it, because the scorer is the authors' and is not copied into the kit.
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import pytest

KIT = Path(__file__).resolve().parents[1] / "kit"


def load(name: str):
    spec = importlib.util.spec_from_file_location("kit_bed_test_%s" % name, KIT / "beds" / ("%s.py" % name))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


chemistry, toolalpaca = load("chemistry"), load("toolalpaca")
SDPO = os.environ.get("SDPO_DIR") or ""
has_checkout = pytest.mark.skipif(not (Path(SDPO) / "verl/utils/reward_score/feedback/mcq.py").is_file(),
                                  reason="needs SDPO_DIR set to the pinned lasgroup/SDPO checkout")


def row(source, index, user, gold, system=None):
    prompt = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
    return {"data_source": source, "prompt": prompt, "ability": "x", "reward_model": {"style": "x", "ground_truth": gold},
            "extra_info": {"split": "test", "index": str(index)}}


def write(root: Path, split: str, rows: list) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / ("%s.jsonl" % split)).write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return root


def test_importing_the_beds_needs_no_checkout(monkeypatch):
    monkeypatch.delenv("SDPO_DIR", raising=False)
    assert load("chemistry").DATA_SOURCE == "sciknoweval" and load("toolalpaca").DATA_SOURCE == "tooluse"
    with pytest.raises(ValueError, match="set SDPO_DIR"):
        load("chemistry").compute_score("sciknoweval", "<answer>A</answer>", "A")


def test_items_keep_the_authors_messages_and_ids_come_from_their_index(tmp_path):
    root = write(tmp_path / "chem", "test", [row("sciknoweval", 1522, "Which? A: x B: y", "A", system="Respond in <answer>"),
                                             row("sciknoweval", 7, "Which else?", "D", system="Respond in <answer>")])
    items = chemistry.load(root, "test")
    assert [i["id"] for i in items] == ["sciknoweval-test-1522", "sciknoweval-test-7"] and all(isinstance(i["id"], str) for i in items)
    assert chemistry.render_prompt(items[0]) == [{"role": "system", "content": "Respond in <answer>"}, {"role": "user", "content": "Which? A: x B: y"}]
    assert chemistry.gold_of(items[1]) == "D"
    tools = toolalpaca.load(write(tmp_path / "tool", "test", [row("tooluse", 4046, "Use the tools.", '[{"Action": "f", "Action_Input": "{}"}]')]), "test")
    assert [m["role"] for m in toolalpaca.render_prompt(tools[0])] == ["user"] and tools[0]["id"] == "tooluse-test-4046"


@pytest.mark.parametrize("rows, message", [
    ([row("tooluse", 1, "q", "A")], "another task's file"),
    ([row("sciknoweval", 1, "q", "E")], "not one of A/B/C/D"),
    ([row("sciknoweval", 1, "q", "A"), row("sciknoweval", 1, "q2", "B")], "repeats the id"),
    ([row("sciknoweval", 1, "  ", "A")], "no user message"),
    ([row("sciknoweval", 1, "q", "")], "no ground truth"),
    ([], "holds no rows"),
])
def test_the_chemistry_bed_refuses_what_it_cannot_score(tmp_path, rows, message):
    with pytest.raises(ValueError, match=message):
        chemistry.load(write(tmp_path / "chem", "test", rows), "test")


def test_missing_files_bad_splits_and_bad_tool_golds_are_refused(tmp_path):
    with pytest.raises(ValueError, match="data/preprocess.py"):
        chemistry.load(tmp_path, "test")
    with pytest.raises(ValueError, match="no 'heldout' split"):
        toolalpaca.load(tmp_path, "heldout")
    with pytest.raises(ValueError, match="not a JSON list of tool calls"):
        toolalpaca.load(write(tmp_path / "tool", "test", [row("tooluse", 1, "q", '{"Action": "f"}')]), "test")


@has_checkout
def test_chemistry_is_marked_by_the_authors_rule_and_reports_its_own_format_flag():
    right = chemistry.compute_score("sciknoweval", "<reasoning>because</reasoning>\n<answer>\nC\n</answer>", "C")
    assert right["acc"] == 1.0 and right["pred"] == "C" and right["incorrect_format"] == 0
    assert right["authors_incorrect_format"] == 1, "the authors' flag is inverted: 1 for a well-formed answer"
    cut = chemistry.compute_score("sciknoweval", "<reasoning>because ... the answer is C", "C")
    assert cut["acc"] == 0.0 and cut["incorrect_format"] == 1, "no <answer> tag: scored on the whole text, never a letter"
    assert chemistry.compute_score("sciknoweval", "<answer>C</answer> wait <answer>B</answer>", "C")["acc"] == 0.0, "the LAST tag decides"
    assert chemistry.compute_score("sciknoweval", "<answer>C", "C")["acc"] == 1.0, "an unclosed last tag still reads the letter"


@has_checkout
def test_toolalpaca_is_marked_by_the_authors_rule_including_its_nested_json_blind_spot():
    gold = '[{"Action": "search", "Action_Input": "{\\"q\\": \\"cats\\"}"}]'
    assert toolalpaca.compute_score("tooluse", 'Thought: t\nAction: search\nAction Input: {"q": "cats"}', gold)["acc"] == 1.0
    assert toolalpaca.compute_score("tooluse", 'Action: search\nAction Input: {"q": "dogs"}', gold)["acc"] == 0.0
    assert toolalpaca.compute_score("tooluse", "I would search for cats.", gold)["incorrect_format"] == 1
    nested = '[{"Action": "search", "Action_Input": "{\\"filter\\": {\\"kind\\": \\"cat\\"}}"}]'
    assert toolalpaca.compute_score("tooluse", 'Action: search\nAction Input: {"filter": {"kind": "cat"}}', nested)["acc"] == 0.0, \
        "the authors' non-greedy pattern cuts a nested object at the first brace: a correct nested call scores 0"


@has_checkout
def test_the_real_task_files_load_when_the_checkout_has_them():
    root = Path(SDPO) / "datasets" / "tooluse"
    if not (root / "test.parquet").is_file():
        pytest.skip("the checkout has no preprocessed tooluse files")
    items = toolalpaca.load(root, "test")
    assert len(items) == 68 and len({i["id"] for i in items}) == 68


# ---- through kit/eval_bed.py, with the engine faked: the message list reaches the template as the trainer sends it
def _eval_bed():
    spec = importlib.util.spec_from_file_location("kit_eval_bed_authors", KIT / "eval_bed.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_eval_bed_scores_the_chemistry_bed_from_its_messages_with_token_ids(tmp_path, monkeypatch):
    import sys
    import types
    seen = {}

    class LLM:
        def __init__(self, **kwargs):
            seen["engine"] = kwargs

        def generate(self, prompts, params):
            seen["prompts"] = prompts
            answers = ["<reasoning>r</reasoning>\n<answer>\nA\n</answer>", "<reasoning>r ... so the answer is B"]
            return [types.SimpleNamespace(outputs=[types.SimpleNamespace(text=answers[i % 2], token_ids=[7, 8, 9], finish_reason="stop" if i % 2 == 0 else "length")])
                    for i, _ in enumerate(prompts)]

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(path):
            def template(messages, **kwargs):
                seen.setdefault("template_kwargs", kwargs)
                return "".join("<%s>%s" % (m["role"], m["content"]) for m in messages) + "<assistant>"
            return types.SimpleNamespace(apply_chat_template=template)

    monkeypatch.setitem(sys.modules, "vllm", types.SimpleNamespace(LLM=LLM, SamplingParams=lambda **kw: types.SimpleNamespace(kwargs=kw), __version__="fake"))
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(AutoTokenizer=AutoTokenizer, __version__="fake"))
    checkout = tmp_path / "SDPO" / "verl" / "utils" / "reward_score" / "feedback"
    checkout.mkdir(parents=True)        # a stand-in for the authors' scorer with the same contract; the real one is tested above
    (checkout / "mcq.py").write_text("def compute_score(solution, ground_truth):\n"
                                     "    pred = solution.split('<answer>')[-1].split('</answer>')[0].strip()\n"
                                     "    return {'score': float(pred == ground_truth), 'acc': float(pred == ground_truth), 'pred': pred, 'incorrect_format': 1, 'feedback': ''}\n")
    monkeypatch.setenv("SDPO_DIR", str(tmp_path / "SDPO"))
    root = write(tmp_path / "chem", "test", [row("sciknoweval", 1, "Which? A: x B: y", "A", system="Respond in <answer>"),
                                             row("sciknoweval", 2, "Which else?", "B", system="Respond in <answer>")])
    model = tmp_path / "model"; model.mkdir(); (model / "config.json").write_text("{}")
    eb = _eval_bed()
    assert eb.main(["generate", "--bed", "chemistry", "--root", str(root), "--model", str(model), "--out", str(tmp_path / "out"), "--max-new-tokens", "8192"]) == 0
    assert seen["prompts"][0] == "<system>Respond in <answer><user>Which? A: x B: y<assistant>", "system then user, as the trainer renders them"
    assert seen["template_kwargs"] == {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}
    result = json.loads((tmp_path / "out" / "bed-score.json").read_text())
    assert result["bed"] == "chemistry" and result["split"] == "test" and result["n"] == 2 and result["correct"] == 1
    assert result["incorrect_format"] == 1, "the answer with no <answer> tag is the one format failure (our flag, not the authors' inverted one)"
    assert result["per_item"] == {"sciknoweval-test-1": 1, "sciknoweval-test-2": 0} and result["max_new_tokens"] == 8192
    assert (tmp_path / "out" / "tokens.jsonl").is_file() and result["truncated_at_max_tokens"] == 1
