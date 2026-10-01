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
