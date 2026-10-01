"""`--max-new-tokens` on both scorers, end to end with the inference engine faked (receipt 241).

The K1c rescoring decodes at 8,192 new tokens instead of 2,048. No GPU test covers it, so this drives both `generate`
commands through a fake `vllm` and checks what reaches the engine and what the result records. What stays untested
here is only the real engine accepting the longer context.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location("kit_%s_longcap" % name, ROOT / "kit" / ("%s.py" % name))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def engine(monkeypatch):
    seen = {}

    class _Completion:
        def __init__(self, cut):
            self.text, self.token_ids, self.finish_reason = "Answer: 1", [0] * 5, "length" if cut else "stop"

    class LLM:
        def __init__(self, **kwargs):
            seen["engine"] = kwargs

        def generate(self, prompts, params):
            seen["params"] = params.kwargs
            return [types.SimpleNamespace(outputs=[_Completion(cut=index == 0)]) for index, _ in enumerate(prompts)]

    class SamplingParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(path):
            return types.SimpleNamespace(apply_chat_template=lambda messages, **kw: messages[0]["content"])

    monkeypatch.setitem(sys.modules, "vllm", types.SimpleNamespace(LLM=LLM, SamplingParams=SamplingParams, __version__="fake"))
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(AutoTokenizer=AutoTokenizer, __version__="fake"))
    return seen


def _model(tmp_path: Path) -> Path:
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    return model


@pytest.mark.parametrize("flag, cap, context", [([], 2048, 4096), (["--max-new-tokens", "8192"], 8192, 4096 + 8192)])
def test_the_panel_scorer_passes_the_cap_to_the_engine_and_records_it(tmp_path, engine, flag, cap, context):
    sf = _load("score_forgetting")
    assert sf.main(["generate", "--model", str(_model(tmp_path)), "--out", str(tmp_path / "out"), *flag]) == 0
    result = json.loads((tmp_path / "out" / "forgetting.json").read_text())
    assert engine["params"]["max_tokens"] == cap and engine["engine"]["max_model_len"] == context
    assert result["max_new_tokens"] == cap and result["decoding"]["max_tokens"] == cap
    assert result["engine"]["max_model_len"] == context and result["truncated_at_max_tokens"] == 1
    assert sf.DECODING["max_tokens"] == 2048 and sf.ENGINE["max_model_len"] == 4096       # the defaults are untouched


@pytest.mark.parametrize("flag, cap, context", [([], 2048, 4096), (["--max-new-tokens", "8192"], 8192, 4096 + 8192)])
def test_the_bed_scorer_passes_the_cap_to_the_engine_and_records_it(tmp_path, engine, flag, cap, context):
    eb = _load("eval_bed")
    data = tmp_path / "gsm8k"
    data.mkdir()
    (data / "test.jsonl").write_text("".join(json.dumps({"question": "What is %d minus %d plus one?" % (n, n), "answer": "#### 1"}) + "\n"
                                             for n in range(1000, 1012)))
    argv = ["generate", "--bed", "gsm8k", "--root", str(data), "--allow-subset", "--heldout-n", "8",
            "--model", str(_model(tmp_path)), "--out", str(tmp_path / "out"), *flag]
    assert eb.main(argv) == 0
    result = json.loads((tmp_path / "out" / "bed-score.json").read_text())
    assert engine["params"]["max_tokens"] == cap and engine["engine"]["max_model_len"] == context
    assert result["max_new_tokens"] == cap and result["decoding"]["max_tokens"] == cap
    assert result["engine"]["max_model_len"] == context and result["truncated_at_max_tokens"] == 1 and result["n"] == 8
    rows = [json.loads(line) for line in (tmp_path / "out" / "responses.jsonl").read_text().splitlines()]
    assert [row["finish_reason"] for row in rows].count("length") == 1 and result["correct"] == 8


def test_token_ids_round_trip_and_fit_the_collectors_limit(tmp_path):
    """The ids are what a prefix budget is decoded from. The worst case a campaign can produce (1,147 FinQA answers,
    every one 8,192 incompressible ids) must stay under the collector's 50 MB whole-file limit."""
    import random
    module = _load_tokens_io()
    rng = random.Random(0)
    ids = [rng.randrange(152_000) for _ in range(8192)]
    assert module.unpack(module.pack(ids)) == ids and module.unpack(module.pack([])) == []
    with pytest.raises(ValueError):
        module.pack([-1])
    path = tmp_path / "tokens.jsonl"
    assert module.write(path, ({"id": "q%d" % i, "ids": [rng.randrange(152_000) for _ in range(8192)]} for i in range(1147))) == 1147
    assert path.stat().st_size < 50 * 1024 * 1024
    rows = module.read(path)
    assert len(rows) == 1147 and rows[7]["n"] == 8192 and len(rows[7]["ids"]) == 8192
    broken = json.loads(path.read_text().split("\n")[0]); broken["n"] = 5
    (tmp_path / "bad.jsonl").write_text(json.dumps(broken) + "\n")
    with pytest.raises(ValueError, match="the row says 5"):
        module.read(tmp_path / "bad.jsonl")


def _load_tokens_io():
    spec = importlib.util.spec_from_file_location("kit_tokens_io_test", ROOT / "kit" / "tokens_io.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_both_scorers_write_the_generated_ids_beside_the_answers(tmp_path, engine):
    sf, tio = _load("score_forgetting"), _load_tokens_io()
    assert sf.main(["generate", "--model", str(_model(tmp_path)), "--out", str(tmp_path / "panel")]) == 0
    result = json.loads((tmp_path / "panel" / "forgetting.json").read_text())
    rows = tio.read(tmp_path / "panel" / result["tokens_file"])
    answers = [json.loads(line) for line in (tmp_path / "panel" / "responses.jsonl").read_text().splitlines()]
    assert len(rows) == len(answers) == 300 and result["token_encoding"] == tio.ENCODING
    assert [(r["panel"], r["id"], r["n"]) for r in rows] == [(a["panel"], a["id"], a["output_tokens"]) for a in answers]
    assert all(len(r["prompt_sha256"]) == 64 and r["prompt_tokens"] is None for r in rows)        # the fake engine reports no prompt ids
    assert len(result["prompts_sha256"]) == 64 and result["generation"]["n"] == 1
    eb = _load("eval_bed")
    data = tmp_path / "gsm8k"; data.mkdir()
    (data / "test.jsonl").write_text("".join(json.dumps({"question": "What is %d minus %d plus one?" % (n, n), "answer": "#### 1"}) + "\n" for n in range(1000, 1012)))
    assert eb.main(["generate", "--bed", "gsm8k", "--root", str(data), "--allow-subset", "--heldout-n", "8", "--model", str(tmp_path / "model"), "--out", str(tmp_path / "bed")]) == 0
    result = json.loads((tmp_path / "bed" / "bed-score.json").read_text())
    rows = tio.read(tmp_path / "bed" / result["tokens_file"])
    answers = [json.loads(line) for line in (tmp_path / "bed" / "responses.jsonl").read_text().splitlines()]
    assert [(r["id"], r["n"], r["ids"]) for r in rows] == [(a["id"], a["output_tokens"], [0] * 5) for a in answers]


@pytest.mark.parametrize("name", ["score_forgetting", "eval_bed"])
def test_one_context_for_two_caps(tmp_path, engine, name):
    """A budget audit scores 2,048 and 8,192 in the SAME 12,288-token context, so the cap is the only difference."""
    module = _load(name)
    extra = []
    if name == "eval_bed":
        data = tmp_path / "gsm8k"; data.mkdir()
        (data / "test.jsonl").write_text("".join(json.dumps({"question": "What is %d minus %d plus one?" % (n, n), "answer": "#### 1"}) + "\n" for n in range(1000, 1012)))
        extra = ["--bed", "gsm8k", "--root", str(data), "--allow-subset", "--heldout-n", "8"]
    model = str(_model(tmp_path))
    for cap in ("2048", "8192"):
        assert module.main(["generate", *extra, "--model", model, "--out", str(tmp_path / cap), "--max-new-tokens", cap, "--max-model-len", "12288"]) == 0
        assert engine["params"]["max_tokens"] == int(cap) and engine["engine"]["max_model_len"] == 12288
        result = json.loads(next((tmp_path / cap).glob("*.json")).read_text())
        assert result["engine"]["max_model_len"] == 12288 and result["max_new_tokens"] == int(cap)
