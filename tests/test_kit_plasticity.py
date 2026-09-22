"""kit/plasticity.py: the plasticity probe and the plan 4b bar.

Five ways this file could be quietly wrong, one section each:

1. The 200 prompts are not the same 200 every time, or not the same across machines. Every number
   here is a comparison between checkpoints, so a probe set that drifted would move every one of them
   and look exactly like a model losing its room to learn.
2. The measurements do not measure what their names say: a dormant share that cannot reach 1 when
   every unit is dead, a saturated share that ignores the reference model's percentile, an effective
   rank that does not fall when a layer collapses, a weight distance that is not the distance. These
   run on a tiny real Llama on a CPU, so they exercise the actual tensor arithmetic and hooks, not a
   mock of them.
3. Steps-to-half-gain is read wrongly: a run that never improved reported as "learned instantly"
   (zero steps) rather than null, or a curve read against the wrong validation key.
4. THE ONE THAT MATTERS: the bar declares PRESENT on one half. plan 4b requires a 25 percent
   slowdown on at least 2 of 3 seeds AND an internal signal moving the same way; a tool that answers
   on either alone would hand the programme a finding it did not earn. The mutation pass drives each
   knob of the bar to both sides of its threshold.
5. A chain is compared across machines, prompt sets or architectures -- differences that are not
   differences -- or an output is overwritten.

Everything builds its own fixtures. Nothing needs a GPU, a trainer, or the network.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
KIT = ROOT / "kit"
README = KIT / "README-plasticity.md"


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pl = load("kit_plasticity", KIT / "plasticity.py")
VAL = "val-core/tooluse/acc/mean@16"


# ============================================================ 1. the fixed 200 prompts
def test_the_probe_set_is_the_frozen_panel_and_matches_its_manifest():
    members, digest = pl.load_panel()
    manifest = json.loads((KIT / "panels" / "general-v1.manifest.json").read_text())
    assert digest == manifest["sha256"] and len(members) == 300


def test_the_same_two_hundred_prompts_come_out_every_time_with_no_seed():
    members, _ = pl.load_panel()
    once, again = pl.select_prompts(members), pl.select_prompts(list(reversed(members)))
    assert [m["id"] for m in once] == [m["id"] for m in again], "the draw must not depend on input order"
    assert len(once) == pl.PROMPTS == 200
    assert len({m["id"] for m in once}) == 200
    assert pl.selection_digest(once) == pl.selection_digest(again)


def test_the_draw_keeps_all_three_panels_in_it():
    chosen = pl.select_prompts(pl.load_panel()[0])
    counts = {panel: sum(m["panel"] == panel for m in chosen) for panel in {m["panel"] for m in chosen}}
    assert sorted(counts) == ["ifeval", "knowledge", "math"]
    assert max(counts.values()) - min(counts.values()) <= 1, counts


def test_a_different_number_of_prompts_is_a_different_probe_set():
    members, _ = pl.load_panel()
    assert pl.selection_digest(pl.select_prompts(members, 200)) != pl.selection_digest(pl.select_prompts(members, 100))
    with pytest.raises(pl.PlasticityError, match="panel holds"):
        pl.select_prompts(members, 301)


def test_the_first_token_is_left_out_because_it_is_an_attention_sink():
    assert pl.select_positions(20, 4) == [1, 7, 13, 19]
    assert pl.select_positions(5, 8) == [1, 2, 3, 4], "every token but the sink, when there are few"
    assert pl.select_positions(1, 8) == [0], "a one-token prompt has only the sink"
    assert pl.select_positions(0, 8) == []


def test_which_activation_values_are_kept_is_fixed_by_the_seed_and_the_position():
    assert pl.sample_indices(10_000, 5, layer=0, prompt=0) == pl.sample_indices(10_000, 5, layer=0, prompt=0)
    assert pl.sample_indices(10_000, 5, layer=0, prompt=0) != pl.sample_indices(10_000, 5, layer=1, prompt=0)
    assert pl.sample_indices(10_000, 5, layer=0, prompt=0) != pl.sample_indices(10_000, 5, layer=0, prompt=1)
    assert sorted(pl.sample_indices(4, 500, layer=0, prompt=0)) == [0, 1, 2, 3], "fewer values than asked: keep all"


# ============================================================ 2. the measurements, on a real tiny model
@pytest.fixture(scope="module")
def torch_stack():
    try:
        import torch, transformers, safetensors                              # noqa: F401,PLC0415
    except ImportError:
        pytest.skip("torch/transformers/safetensors are not installed here")
    return True


def _tiny_model(path: Path, *, mlp_scale: float = 1.0, mlp_zero: bool = False, seed: int = 0) -> Path:
    """A real two-layer Llama and a real tokenizer, saved as a HuggingFace checkpoint."""
    import torch                                                             # noqa: PLC0415
    from transformers import AutoModelForCausalLM, LlamaConfig, PreTrainedTokenizerFast   # noqa: PLC0415
    from tokenizers import Tokenizer                                         # noqa: PLC0415
    from tokenizers.models import WordLevel                                  # noqa: PLC0415
    from tokenizers.pre_tokenizers import Whitespace                         # noqa: PLC0415

    torch.manual_seed(seed)
    cfg = LlamaConfig(vocab_size=256, hidden_size=8, intermediate_size=16, num_hidden_layers=2,
                      num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=512)
    model = AutoModelForCausalLM.from_config(cfg).to(torch.float32)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "mlp.gate_proj" in name or "mlp.up_proj" in name:
                parameter.mul_(0.0 if mlp_zero else mlp_scale)
    model.save_pretrained(str(path))
    # A vocabulary of the panel's own commonest words, so the prompts really do tokenise differently
    # from one another: a tokenizer that mapped every word to [UNK] would hand every layer the same
    # activations and quietly make the probe look far more collapsed than any model ever is.
    words = sorted({word for member in pl.select_prompts(pl.load_panel()[0], 64)
                    for word in member["prompt"].split()})[:255]
    obj = Tokenizer(WordLevel({"[UNK]": 0, **{word: i + 1 for i, word in enumerate(words)}}, unk_token="[UNK]"))
    obj.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=obj, unk_token="[UNK]").save_pretrained(str(path))
    return path


@pytest.fixture(scope="module")
def probed(torch_stack, tmp_path_factory):
    """One untrained tiny model, probed against itself: the reference every other probe uses."""
    root = tmp_path_factory.mktemp("plasticity")
    model = _tiny_model(root / "untrained")
    result = pl.probe(model, base=model, device="cpu", dtype="float32", count=8)
    out = root / "untrained-probe"
    out.mkdir()
    (out / "plasticity.json").write_text(json.dumps(result))
    return {"root": root, "model": model, "result": result, "file": out / "plasticity.json"}


def test_a_probe_measures_every_layer_and_reports_what_it_used(probed):
    result = probed["result"]
    assert result["schema"] == "kit-plasticity.v1" and result["mode"] == "probe"
    assert result["architecture"]["layers"] == 2 and result["architecture"]["mlp_hidden_size"] == 16
    assert len(result["layers"]) == 2 and result["prompts_measured"] == 8
    assert result["prompt_selection"]["n"] == 8 and result["prompt_selection"]["sha256"]
    assert result["settings"]["dormant_threshold"] == 1e-3 and result["settings"]["saturation_quantile"] == 0.999
    assert result["machine"]["id"] and result["machine"]["dtype"] == "float32"
    assert result["settings"]["chat_template"] is False, "this tokenizer has no template, and the result says so"
    assert result["tokens_measured"] > 0 and result["prompts_truncated"] == 0
    for key in ("dormant_share", "saturated_share", "effective_rank"):
        assert isinstance(result[key], (int, float)), "%s must be a number a bar could read" % key


def test_two_probes_of_one_model_on_one_machine_give_the_same_numbers(probed):
    again = pl.probe(probed["model"], base=probed["model"], device="cpu", dtype="float32", count=8)
    ignore = {"generated_at"}
    assert {k: v for k, v in again.items() if k not in ignore} == \
           {k: v for k, v in probed["result"].items() if k not in ignore}


def test_a_model_whose_hidden_units_are_all_dead_is_entirely_dormant(probed):
    dead = _tiny_model(probed["root"] / "dead", mlp_zero=True)
    result = pl.probe(dead, device="cpu", dtype="float32", count=8)
    assert result["dormant_share"] == 1.0, "every unit is silenced, so every unit is dormant"
    assert probed["result"]["dormant_share"] < 1.0, "a live model must not look dead"


def test_the_saturated_share_is_judged_against_the_reference_model_not_the_model_itself(probed):
    """The number that answers 'has this model grown out of the range it was calibrated in'. Measured
    against its own percentile it is 0.001 by construction; against the untrained model's it can rise."""
    hot = _tiny_model(probed["root"] / "hot", mlp_scale=25.0)
    reference = pl.reference_thresholds(probed["file"])
    against_reference = pl.probe(hot, reference=reference, device="cpu", dtype="float32", count=8)
    against_itself = pl.probe(hot, device="cpu", dtype="float32", count=8)
    assert against_reference["saturation_reference"]["source"] == "reference"
    assert against_itself["saturation_reference"]["source"] == "self"
    assert against_reference["saturated_share"] > 0.5, against_reference["saturated_share"]
    assert against_itself["saturated_share"] <= 1 - pl.SATURATION_QUANTILE, \
        "against its OWN percentile at most one value in a thousand can be above it, by construction"
    assert probed["result"]["saturated_share"] <= 1 - pl.SATURATION_QUANTILE


def test_a_reference_drawn_from_other_prompts_or_from_another_reference_is_refused(probed, tmp_path):
    reference = pl.reference_thresholds(probed["file"])
    with pytest.raises(pl.PlasticityError, match="different set of prompts"):
        pl.probe(probed["model"], reference=reference, device="cpu", dtype="float32", count=6)
    second_hand = dict(probed["result"], saturation_reference={"source": "reference", "path": "x", "model": None})
    path = tmp_path / "second-hand.json"
    path.write_text(json.dumps(second_hand))
    with pytest.raises(pl.PlasticityError, match="not the untrained reference"):
        pl.reference_thresholds(path)


def test_a_reference_of_a_different_depth_is_refused(probed, tmp_path):
    reference = pl.reference_thresholds(probed["file"])
    reference["_thresholds"] = reference["_thresholds"] + [0.5]
    with pytest.raises(pl.PlasticityError, match="layers and this model has"):
        pl.probe(probed["model"], reference=reference, device="cpu", dtype="float32", count=8)


def test_effective_rank_counts_the_directions_a_layer_is_really_using(torch_stack):
    import torch                                                             # noqa: PLC0415

    assert pl.effective_rank(torch.eye(4)) == pytest.approx(4.0, abs=1e-4)
    collapsed = torch.ones(6, 4) * torch.tensor([1.0, 2.0, 3.0, 4.0])
    assert pl.effective_rank(collapsed) == pytest.approx(1.0, abs=1e-4), "one direction, whatever its size"
    assert pl.effective_rank(torch.zeros(4, 4)) == 0.0
    mixed = torch.eye(4) * torch.tensor([10.0, 1.0, 1.0, 1.0])
    assert 1.0 < pl.effective_rank(mixed) < 4.0


def test_the_effective_rank_of_a_probe_lies_between_one_and_the_rows_it_read(probed):
    for layer in probed["result"]["layers"]:
        assert 1.0 <= layer["effective_rank"] <= layer["residual_rows"] + 1e-6, layer
        assert layer["residual_rows"] > 0 and layer["residual_width"] == 8


def test_the_residual_rows_are_each_layer_s_output_and_not_the_embedding(probed):
    """Recompute every layer's effective rank here, from the model's own hidden states, and require
    the probe's number. This pins WHICH activations were read: the embedding layer is not a layer."""
    import torch                                                             # noqa: PLC0415
    from transformers import AutoModelForCausalLM, AutoTokenizer             # noqa: PLC0415

    model = AutoModelForCausalLM.from_pretrained(str(probed["model"]), dtype=torch.float32).eval()
    tokenizer = AutoTokenizer.from_pretrained(str(probed["model"]))
    rows: list = [[], []]
    for member in pl.select_prompts(pl.load_panel()[0], 8):
        text, _ = pl.render_prompt(tokenizer, member["prompt"])
        ids = tokenizer(text, return_tensors="pt", add_special_tokens=True)["input_ids"]
        with torch.no_grad():
            out = model(input_ids=ids, output_hidden_states=True, use_cache=False)
        picks = torch.tensor(pl.select_positions(int(ids.shape[1]), pl.TOKENS_PER_PROMPT), dtype=torch.long)
        for index, state in enumerate(out.hidden_states[1:]):
            rows[index].append(state[0].index_select(0, picks))
    for index, layer in enumerate(probed["result"]["layers"]):
        assert round(pl.effective_rank(torch.cat(rows[index])), 6) == layer["effective_rank"], index


def test_the_weight_distance_is_the_distance_and_is_zero_against_itself(probed):
    import torch                                                             # noqa: PLC0415
    from safetensors.torch import load_file                                  # noqa: PLC0415

    assert probed["result"]["weights"]["relative_distance_from_base"] == 0.0
    assert probed["result"]["weights"]["frobenius_norm"] > 0
    moved = _tiny_model(probed["root"] / "moved", mlp_scale=1.5)
    result = pl.probe(moved, base=probed["model"], device="cpu", dtype="float32", count=8)
    here = load_file(str(moved / "model.safetensors"))
    there = load_file(str(probed["model"] / "model.safetensors"))
    expected = (sum(float((here[k] - there[k]).pow(2).sum()) for k in here) /
                sum(float(there[k].pow(2).sum()) for k in there)) ** 0.5
    assert result["weights"]["relative_distance_from_base"] == pytest.approx(expected, rel=1e-5)
    assert result["weights"]["tensors_compared"] > 0 and result["weights"]["tensors_missing_from_base"] == 0
    assert set(result["weights"]["relative_distance_per_layer"]) == {"0", "1", "other"}
    assert torch.is_tensor(here[list(here)[0]])


def test_a_model_this_file_cannot_read_is_refused_rather_than_measured_wrongly(torch_stack, tmp_path):
    import torch                                                             # noqa: PLC0415
    from torch import nn                                                     # noqa: PLC0415

    class NoMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = nn.Linear(2, 2)

    with pytest.raises(pl.PlasticityError, match="no MLP hidden layer"):
        pl.mlp_hidden_modules(NoMLP())
    with pytest.raises(pl.PlasticityError, match="not a HuggingFace model directory"):
        pl.probe(tmp_path, device="cpu", dtype="float32", count=4)
    assert torch.__version__


def test_a_refused_probe_writes_nothing_and_an_existing_output_is_refused_first(tmp_path):
    out = tmp_path / "out"
    with pytest.raises(SystemExit, match="not a HuggingFace model directory"):
        pl.main(["probe", "--model", str(tmp_path), "--out", str(out)])
    assert not out.exists(), "a refusal must leave no directory behind"
    out.mkdir()
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        pl.main(["probe", "--model", str(tmp_path / "nothing-here"), "--out", str(out)])
    assert list(out.iterdir()) == []


# ============================================================ 3. steps to half the final gain
def _metrics(run: Path, curve: list, key: str = VAL) -> Path:
    run.mkdir(parents=True, exist_ok=True)
    (run / "metrics.jsonl").write_text("".join(
        json.dumps({"step": step, "data": {key: value}}) + "\n" for step, value in curve))
    return run


def test_half_the_gain_is_measured_from_the_first_validation_to_the_last():
    out = pl.steps_to_half_gain([(0, 0.2), (5, 0.3), (10, 0.5), (15, 0.6), (20, 0.6)])
    assert out["first"] == 0.2 and out["final"] == 0.6 and out["gain"] == pytest.approx(0.4)
    assert out["steps"] == 10, "0.4 is halfway between 0.2 and 0.6, and step 10 is the first to reach it"
    assert out["steps_interpolated"] == pytest.approx(7.5), "0.4 sits halfway between the 5- and 10-step points"


def test_a_run_that_never_improved_has_no_steps_to_half_a_gain_it_did_not_make():
    flat = pl.steps_to_half_gain([(0, 0.4), (5, 0.4), (10, 0.35)])
    assert flat["steps"] is None and flat["gain"] <= 0
    assert "no gain to halve" in flat["note"], "null, never zero: zero would read as learned instantly"
    assert pl.steps_to_half_gain([(0, 0.4)])["steps"] is None


def test_a_dip_before_the_rise_does_not_stop_the_reading():
    out = pl.steps_to_half_gain([(0, 0.2), (5, 0.1), (10, 0.45), (15, 0.6)])
    assert out["steps"] == 10 and out["first"] == 0.2 and out["final"] == 0.6


def test_the_validation_key_is_found_when_there_is_one_and_asked_for_when_there_are_two(tmp_path):
    run = _metrics(tmp_path / "one", [(0, 0.2), (10, 0.6)])
    series, key = pl.validation_series(run)
    assert key == VAL and series == [(0, 0.2), (10, 0.6)]
    (run / "metrics.jsonl").write_text(json.dumps(
        {"step": 0, "data": {VAL: 0.2, "val-core/sql/acc/mean@8": 0.3}}) + "\n")
    with pytest.raises(pl.PlasticityError, match="name the one to use"):
        pl.validation_series(run)
    assert pl.validation_series(run, "val-core/sql/acc/mean@8")[0] == [(0, 0.3)]
    with pytest.raises(pl.PlasticityError, match="no metrics.jsonl"):
        pl.validation_series(tmp_path / "missing")


def test_runs_are_found_by_job_position_and_seed_with_the_highest_attempt_winning(tmp_path):
    _metrics(tmp_path / "sql-pos1-seed42-a1", [(0, 0.2), (10, 0.6)])
    _metrics(tmp_path / "sql-pos1-seed42-a2", [(0, 0.2), (5, 0.6)])
    _metrics(tmp_path / "sql-pos4-seed42", [(0, 0.2), (15, 0.6)])
    _metrics(tmp_path / "notes", [(0, 0.2), (10, 0.6)])
    found = pl.find_runs([str(tmp_path)])
    assert set(found) == {("sql", 1, 42), ("sql", 4, 42)}
    assert found[("sql", 1, 42)].name == "sql-pos1-seed42-a2"
    assert pl.find_runs([str(tmp_path / "sql-pos4-seed42")]) == {("sql", 4, 42): tmp_path / "sql-pos4-seed42"}


def test_runs_that_do_not_say_their_position_and_seed_are_refused_not_guessed(tmp_path):
    _metrics(tmp_path / "stage-four", [(0, 0.2), (10, 0.6)])
    with pytest.raises(pl.PlasticityError, match="pos<POSITION>-seed<SEED>"):
        pl.find_runs([str(tmp_path)])
    with pytest.raises(pl.PlasticityError, match="no such directory"):
        pl.find_runs([str(tmp_path / "nowhere")])


# ============================================================ 4. the bar, and the mutation pass
def _body(name: str, *, dormant=0.10, saturated=0.001, rank=50.0, machine="mach-1",
          layers=2, prompts="sha-200", moved=0.0) -> dict:
    return {"schema": "kit-plasticity.v1", "mode": "probe", "_name": name,
            "prompt_selection": {"n": 200, "sha256": prompts},
            "architecture": {"layers": layers, "mlp_hidden_size": 16},
            "saturation_reference": {"source": "self" if name == "untrained" else "reference"},
            "dormant_share": dormant, "saturated_share": saturated, "effective_rank": rank,
            "weights": {"relative_distance_from_base": moved}, "machine": {"id": machine}}


def _chain(tmp_path: Path, bodies: list) -> list:
    paths = []
    for body in bodies:
        directory = tmp_path / "chain" / body["_name"]
        directory.mkdir(parents=True)
        (directory / "plasticity.json").write_text(json.dumps(body))
        paths.append(str(directory / "plasticity.json"))
    return paths


def _scenario(tmp_path: Path, *, seeds=3, slower_seeds=3, early_half=12, late_half=20,
              dormant=(0.10, 0.20), saturated=(0.001, 0.02), rank=(50.0, 40.0),
              with_metrics=True, machines=("mach-1", "mach-1"), **build_kwargs) -> dict:
    """A whole chain plus its runs, with every knob of the bar exposed."""
    chain = _chain(tmp_path, [_body("untrained", dormant=dormant[0], saturated=saturated[0],
                                    rank=rank[0], machine=machines[0]),
                              _body("stage4", dormant=dormant[1], saturated=saturated[1],
                                    rank=rank[1], machine=machines[1], moved=0.04)])
    runs = tmp_path / "runs"
    for seed in range(seeds):
        for position, half in ((pl.POSITION_EARLY, early_half),
                               (pl.POSITION_LATE, late_half if seed < slower_seeds else early_half)):
            curve = [(0, 0.0)] + [(step, 0.5 if step == half else (1.0 if step > half else 0.1))
                                  for step in (5, 10, 12, 14, 15, 20, 25)]
            _metrics(runs / ("sql-pos%d-seed%d-a1" % (position, seed)), sorted(set(curve)))
    return pl.build(chain, [str(runs)] if with_metrics else None, **build_kwargs)


#: Every knob of plan 4b's bar, at a value that must NOT give PRESENT and one that must. A bar that
#: cannot fail is not a bar; a bar that cannot pass never says anything.
MUTATION_CASES = [
    ("slower-by-25-percent", {"early_half": 12, "late_half": 14}, {"early_half": 12, "late_half": 15}),
    ("at-least-two-seeds", {"slower_seeds": 1}, {"slower_seeds": 2}),
    ("out-of-three-seeds", {"seeds": 2, "slower_seeds": 2}, {"seeds": 3, "slower_seeds": 2}),
    ("an-internal-signal-moves", {"dormant": (0.2, 0.1), "saturated": (0.02, 0.001), "rank": (40.0, 50.0)},
                                 {"dormant": (0.2, 0.1), "saturated": (0.02, 0.001), "rank": (50.0, 40.0)}),
    ("dormant-share-alone", {"dormant": (0.2, 0.2), "saturated": (0.02, 0.001), "rank": (40.0, 50.0)},
                            {"dormant": (0.1, 0.2), "saturated": (0.02, 0.001), "rank": (40.0, 50.0)}),
    ("saturated-share-alone", {"dormant": (0.2, 0.2), "saturated": (0.02, 0.02), "rank": (40.0, 50.0)},
                              {"dormant": (0.2, 0.2), "saturated": (0.001, 0.02), "rank": (40.0, 50.0)}),
    ("the-learning-curves-at-all", {"with_metrics": False}, {"with_metrics": True}),
]


@pytest.mark.parametrize("name,bad,good", MUTATION_CASES)
def test_every_knob_of_the_bar_can_both_fail_and_pass(tmp_path, name, bad, good):
    for index, (knobs, want) in enumerate(((bad, False), (good, True))):
        report = _scenario(tmp_path / ("%s-%d" % (name, index)), **knobs)
        assert (report["verdict"] == "PRESENT") is want, (name, knobs, report["verdict"])


def test_the_tool_never_declares_plasticity_loss_on_one_half(tmp_path):
    """The reason this file exists. Each half alone is not the finding."""
    learning_only = _scenario(tmp_path / "learning", dormant=(0.2, 0.1), saturated=(0.02, 0.001),
                              rank=(40.0, 50.0))
    assert learning_only["learning_half_met"] == 1 and learning_only["internal_half_met"] == 0
    assert learning_only["verdict"] == "NOT_PRESENT" and learning_only["plasticity_loss_present"] == 0
    internal_only = _scenario(tmp_path / "internal", slower_seeds=0)
    assert internal_only["internal_half_met"] == 1 and internal_only["learning_half_met"] == 0
    assert internal_only["verdict"] == "NOT_PRESENT"
    both = _scenario(tmp_path / "both")
    assert both["verdict"] == "PRESENT" and both["plasticity_loss_present"] == 1


def test_too_few_seeds_is_incomplete_which_is_neither_a_pass_nor_a_fail(tmp_path):
    report = _scenario(tmp_path / "two-seeds", seeds=2, slower_seeds=2)
    assert report["verdict"] == "INCOMPLETE"
    assert report["learning_half"]["per_job"]["sql"]["seeds_compared"] == 2
    assert report["learning_half"]["per_job"]["sql"]["sufficient"] is False
    assert report["learning_half"]["per_job"]["sql"]["met"] is False, "2 of 2 is not 2 of 3"
    assert report["plasticity_loss_present"] == 0


def test_more_seeds_than_the_bar_was_written_for_are_judged_but_flagged(tmp_path):
    """plan 4b wrote "2 of 3". It never said what 2 of 5 means, so the tool applies the number it was
    given and says on its face that it is doing so."""
    report = _scenario(tmp_path / "five", seeds=5, slower_seeds=2)
    job = report["learning_half"]["per_job"]["sql"]
    assert job["seeds_compared"] == 5 and job["seeds_slower"] == 2 and job["met"] is True
    assert "never stated" in job["note"]
    assert _scenario(tmp_path / "three", seeds=3, slower_seeds=2)["learning_half"]["per_job"]["sql"]["note"] is None


def test_a_job_found_at_only_one_position_is_not_compared_with_itself(tmp_path):
    runs = tmp_path / "runs"
    for seed in range(3):
        _metrics(runs / ("sql-pos1-seed%d" % seed), [(0, 0.0), (10, 0.5), (20, 1.0)])
        _metrics(runs / ("maths-pos4-seed%d" % seed), [(0, 0.0), (20, 0.5), (25, 1.0)])
    report = pl.build(_chain(tmp_path, [_body("untrained"), _body("stage4", dormant=0.2)]), [str(runs)])
    assert report["learning_half"]["per_seed"] == []
    assert report["verdict"] == "INCOMPLETE"


def test_the_reported_ratio_is_the_one_the_bar_read(tmp_path):
    report = _scenario(tmp_path / "ratio", early_half=12, late_half=15)
    rows = report["learning_half"]["per_seed"]
    assert {row["ratio"] for row in rows} == {1.25}
    assert all(row["slower"] for row in rows), "exactly 25 percent slower is slower enough"
    assert report["learning_half"]["slower_by"] == 1.25 == pl.SLOWER_BY


def test_the_bar_divides_the_logged_steps_and_not_the_interpolated_ones(tmp_path):
    """Validations land every 5 steps, so a ratio of two step counts is coarse. The interpolated
    column is printed beside it to show how close a call was; the bar reads the logged step."""
    runs = tmp_path / "runs"
    for seed in range(3):
        _metrics(runs / ("sql-pos1-seed%d" % seed), [(0, 0.0), (5, 0.9), (10, 1.0)])
        _metrics(runs / ("sql-pos4-seed%d" % seed), [(0, 0.0), (5, 0.1), (10, 0.9), (20, 1.0)])
    report = pl.build(_chain(tmp_path, [_body("untrained"), _body("stage4", dormant=0.2)]), [str(runs)])
    row = report["learning_half"]["per_seed"][0]
    assert (row["early_steps"], row["late_steps"], row["ratio"]) == (5, 10, 2.0)
    assert row["early_steps_interpolated"] < 5 and row["late_steps_interpolated"] < 10


def test_the_internal_half_says_it_had_no_pre_set_size(tmp_path):
    report = _scenario(tmp_path / "size")
    assert report["internal_half"]["size_of_move_was_never_pre_set"] == 1
    assert sorted(report["internal_half"]["moving"]) == ["dormant_share", "effective_rank", "saturated_share"]
    directions = {s["signal"]: s["less_plastic_is"] for s in report["internal_half"]["signals"]}
    assert directions == {"dormant_share": "up", "saturated_share": "up", "effective_rank": "down"}


def test_the_readout_shows_both_halves_and_says_it_is_not_a_fix(tmp_path):
    text = pl.render(_scenario(tmp_path / "render"))
    assert "Half one: MET" in text and "Half two: MET" in text and "**PRESENT.**" in text
    assert "measurement, not a fix" in text
    assert "| sql | 0 |" in text
    assert "one half never answers" in text


# ============================================================ 5. comparisons that are not comparisons
def test_a_chain_across_two_machines_is_refused_unless_the_departure_is_declared(tmp_path):
    with pytest.raises(pl.PlasticityError, match="machine-and-mode fingerprints"):
        _scenario(tmp_path / "mixed", machines=("mach-1", "mach-2"))
    allowed = _scenario(tmp_path / "declared", machines=("mach-1", "mach-2"), allow_different_machines=True)
    assert allowed["comparable"] == 0 and allowed["different_machines_allowed"] == 1
    assert "not comparable" in pl.render(allowed)


def test_a_chain_of_different_prompt_sets_or_architectures_is_refused(tmp_path):
    with pytest.raises(pl.PlasticityError, match="different prompt sets"):
        pl.build(_chain(tmp_path / "a", [_body("untrained"), _body("stage4", prompts="sha-100")]), None)
    with pytest.raises(pl.PlasticityError, match="different architectures"):
        pl.build(_chain(tmp_path / "b", [_body("untrained"), _body("stage4", layers=4)]), None)
    with pytest.raises(pl.PlasticityError, match="at least two probes"):
        pl.build(_chain(tmp_path / "c", [_body("untrained")]), None)


def test_an_existing_output_is_refused_and_a_refusal_writes_nothing(tmp_path):
    chain = _chain(tmp_path, [_body("untrained"), _body("stage4", dormant=0.2)])
    out = tmp_path / "report"
    assert pl.main(["compare", "--chain", *chain, "--out", str(out)]) == 0
    assert (out / "plasticity-compare.json").is_file() and (out / "plasticity-compare.md").is_file()
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        pl.main(["compare", "--chain", *chain, "--out", str(out)])
    second = tmp_path / "second"
    with pytest.raises(SystemExit, match="different prompt sets"):
        pl.main(["compare", "--chain", *_chain(tmp_path / "d", [_body("untrained"), _body("s", prompts="x")]),
                 "--out", str(second)])
    assert not second.exists()


def test_compare_is_a_measurement_and_not_a_gate(tmp_path):
    """Whatever the verdict, the exit code is 0: nothing downstream refuses to run because of this."""
    for name, knobs in (("present", {}), ("absent", {"slower_seeds": 0}), ("incomplete", {"with_metrics": False})):
        report = _scenario(tmp_path / ("exit-" + name), **knobs)
        directory = tmp_path / ("out-" + name)
        (directory).mkdir()
        assert report["verdict"] in ("PRESENT", "NOT_PRESENT", "INCOMPLETE")
    chain = _chain(tmp_path / "exit", [_body("untrained"), _body("stage4", dormant=0.05)])
    assert pl.main(["compare", "--chain", *chain, "--out", str(tmp_path / "exit-out")]) == 0


# ============================================================ the README the partner reads
def test_the_readme_states_the_bar_this_file_actually_applies():
    text = README.read_text()
    assert "25 percent more steps to reach half its final gain in position 4 than in position 1" in text
    assert "at least 2 of 3 seeds" in text
    assert "<job>-pos<POSITION>-seed<SEED>" in text
    assert "0.001" in text and "99.9th percentile" in text
    assert "measurement, not a fix" in text or "not a fix" in text
    assert "exits 0 whatever the verdict" in text
    assert "--allow-different-machines" in text
