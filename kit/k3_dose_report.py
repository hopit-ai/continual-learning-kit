#!/usr/bin/env python3
"""The K3 stage-A dose probe's readout: which dose moved the Spider held-out score, and how.

    python k3_dose_report.py --root $WORK/k3dose --runs $WORK/runs --out $WORK/k3dose/report/a1

Reads the folders the campaign wrote:

    <root>/eval/base-spider-aN/bed-score.json     the untrained Qwen3-1.7B, 100 held-out questions
    <root>/eval/<arm>-spider-aN/bed-score.json    the same 100 questions after the arm
    <runs>/<arm>-seed0-aN/train-summary.json      what the launcher ran (steps, lr, training file)
    <runs>/<arm>-seed0-aN/metrics.jsonl           the per-step numbers (kit/run_grpo.sh FILE_LOG=1)

for the four arms `ref20`, `steps60`, `lr3x` and `hard20`, and the highest attempt of each. THE
QUESTION IT ANSWERS is the one the partner's K3 pilot raised: stage A moved the untrained model from
70 to 62 of 100 and the bar was +5, the training reward was flat, and only about 170 of the 640
training questions carry any gradient at all -- so is the dose too small? Each arm changes exactly
one thing about that dose, and this file puts the four side by side.

THREE THINGS IT COMPUTES, rather than reads:

1. THE PAIRED CHURN. The two scorings carry `per_item`, one verdict per question id, so the same 100
   questions can be followed one at a time: right -> wrong, wrong -> right, unchanged. A net delta of
   -8 made of 21 losses and 13 gains is a policy that moved a great deal; one made of 8 losses and no
   gains is not, and the net number alone cannot tell them apart.
2. McNEMAR, EXACTLY. The two-sided exact binomial test on the two discordant counts, from the
   standard library's `math.comb` -- no scipy, no normal approximation, no continuity correction. It
   asks one thing only: could a policy that changed nothing have produced this split by chance? It
   does not say the arm works.
3. WHAT A CORRECT ANSWER COSTS. kit/density.py, loaded by path from beside this file: output tokens
   over the whole held-out set divided by the correct answers, after against untrained, against that
   file's own bar. A scoring that carries no token counts is `-` here and changes nothing else.

Everything else is read from the files as it stands: steps, learning rate, the training file's rows
and how many passes over its distinct questions they are, the truncation count, and the per-step
reward, gradient norm, entropy and response length. A missing arm, run or file is reported as missing
and named; nothing is imputed, averaged in, or filled with a zero. THE WARM-UP IS 10 STEPS ON EVERY
ARM: kit/run_grpo.sh fixes it there whatever LR is, so a 20-step arm spends half its steps warming up
and a 60-step arm a sixth. That is part of what the probe measures.

ONE SEED. Every arm here is seed 0, so a difference between two arms is one run against one run, and
`ref20` is in the probe precisely to show what one run against one run is worth. Standard library
only. Nothing is overwritten: an existing --out is refused.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = "kit-k3-dose-report.v1"
HERE = Path(__file__).resolve().parent
ATTEMPT = re.compile(r"^(?P<stem>.+)-a(?P<attempt>\d+)$")
#: The four doses, in the order the campaign runs them. `ref20` is the reference dose rerun.
ARMS = ("ref20", "steps60", "lr3x", "hard20")
WHAT = {"ref20": "the reference dose, rerun on this machine (the pipeline's own run-to-run noise)",
        "steps60": "three passes over the same 640 questions: 60 steps instead of 20",
        "lr3x": "the reference dose at three times the learning rate",
        "hard20": "20 steps over the questions the untrained model does NOT already solve every time"}
BAR = 5                     # K3's own stage-A bar: Spider held-out correct must rise by at least 5
WARMUP_STEPS = 10           # kit/run_grpo.sh, on every arm, whatever LR is
BATCH = 32                  # data.train_batch_size
TRAIN_KEYS = {"reward": "critic/score/mean", "response_tokens": "response_length/mean",
              "entropy": "actor/entropy", "grad_norm": "actor/grad_norm"}


class K3DoseReportError(ValueError):
    """The evidence tree cannot support a report; nothing is written."""


# ------------------------------------------------------- intelligence density (plan 4c, row Q13)
def _load_density():
    """kit/density.py, loaded by path from this file's own directory, exactly as kit/k3_report.py
    loads it: nothing here needs the kit installed or on sys.path. A copy without the file still
    reports -- every density is then unknown, and an unknown density is never a refusal."""
    path = HERE / "density.py"
    if not path.is_file():
        return None
    spec = importlib.util.spec_from_file_location("kit_density_for_k3_dose", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


density = _load_density()
DENSITY_BAR = density.BAR if density else 1.5
NO_TOKENS = ("No scoring under this root carried token counts -- neither `output_tokens_total` in "
             "the result nor a `responses.jsonl` beside it -- so every figure in this table is `-`. "
             "Nothing else in this report depends on them.")


def cost(found, *, panel=None):
    """Tokens per correct answer for one (path, result) pair, or None when its tokens are absent."""
    if density is None or not found:
        return None
    path, result = found
    try:
        return density.tokens_per_correct(result, Path(path), panel=panel)
    except density.DensityError:
        return None


def compare_cost(trained, untrained) -> dict:
    if density is None:
        return {"untrained": None, "trained": None, "ratio": None, "bar": DENSITY_BAR, "verdict": "unknown"}
    return density.compare(trained, untrained, bar=DENSITY_BAR)


def show_cost(value, digits: int = 1) -> str:
    return density.show(value, digits) if density else "-"


# ------------------------------------------------------------------------------------ small helpers
def _finite(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _round(value, digits=4):
    return round(value, digits) if _finite(value) else None


def _jsonl(path: Path) -> list:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def _mean(values):
    clean = [v for v in values if _finite(v)]
    return _round(statistics.fmean(clean)) if clean else None


def latest_found(directory: Path, filename: str) -> dict:
    """{stem: (path, result)} for the highest attempt of every <stem>-aN folder holding `filename`.

    The PATH is carried beside the result because kit/density.py needs it: a scoring's token counts
    may live in the `responses.jsonl` written beside it rather than inside it.
    """
    found: dict = {}
    if not directory.is_dir():
        return found
    for child in sorted(directory.iterdir()):
        match = ATTEMPT.match(child.name) if child.is_dir() else None
        if not match or not (child / filename).is_file():
            continue
        stem, attempt = match["stem"], int(match["attempt"])
        if stem in found and attempt <= found[stem][0]:
            continue
        try:
            result = json.loads((child / filename).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        found[stem] = (attempt, child / filename, result)
    return {stem: (path, result) for stem, (_attempt, path, result) in found.items()}


def latest_dirs(directory: Path) -> dict:
    """{stem: path} for the highest attempt of every <stem>-aN directory."""
    found: dict = {}
    if not directory.is_dir():
        return found
    for child in sorted(directory.iterdir()):
        match = ATTEMPT.match(child.name) if child.is_dir() else None
        if not match:
            continue
        stem, attempt = match["stem"], int(match["attempt"])
        if stem not in found or attempt > found[stem][0]:
            found[stem] = (attempt, child)
    return {stem: path for stem, (_attempt, path) in found.items()}


def number(result: dict | None, key: str):
    if not isinstance(result, dict):
        return None
    value = result.get(key)
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


# ------------------------------------------------------------------------------ the paired churn
def churn(before: dict | None, after: dict | None) -> dict:
    """Question by question, on the ids both scorings carry: what moved, and in which direction."""
    before, after = before or {}, after or {}
    shared = sorted(set(before) & set(after))
    right_to_wrong = sum(1 for q in shared if float(before[q]) >= 1.0 > float(after[q]))
    wrong_to_right = sum(1 for q in shared if float(after[q]) >= 1.0 > float(before[q]))
    return {"items_compared": len(shared),
            "right_to_wrong": right_to_wrong, "wrong_to_right": wrong_to_right,
            "unchanged": len(shared) - right_to_wrong - wrong_to_right,
            "changed": right_to_wrong + wrong_to_right,
            "only_in_one_scoring": len(set(before) ^ set(after))}


def mcnemar_exact(b: int, c: int) -> float | None:
    """The exact two-sided McNemar p from the two discordant counts, from the standard library.

    Under the null hypothesis that the policy changed nothing, each of the b + c questions that moved
    was equally likely to move either way, so the smaller count is binomial(n, 0.5). The two-sided p
    is twice the lower tail, capped at 1 (and exactly 1 when the two counts are equal). No normal
    approximation and no continuity correction: with 34 discordant questions neither is needed, and
    an approximation would put a number in this report that the counts do not support.
    """
    if not isinstance(b, int) or not isinstance(c, int) or b < 0 or c < 0:
        return None
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2.0 ** n)
    return min(1.0, 2.0 * tail)


# ------------------------------------------------------------------------ what the trainer logged
def quarters(values: list) -> dict:
    """{mean, first_quarter, last_quarter} of a per-step signal, in step order."""
    clean = [v for v in values if _finite(v)]
    if not clean:
        return {"mean": None, "first_quarter": None, "last_quarter": None}
    quarter = max(1, len(clean) // 4)
    return {"mean": _round(statistics.fmean(clean)),
            "first_quarter": _round(statistics.fmean(clean[:quarter])),
            "last_quarter": _round(statistics.fmean(clean[-quarter:]))}


def read_metrics(run: Path | None) -> dict:
    """The per-step training signals of one run, or an empty reading with the reason it is empty."""
    out = {"steps_logged": 0, "missing": None,
           **{key: {"mean": None, "first_quarter": None, "last_quarter": None} for key in TRAIN_KEYS}}
    if run is None:
        out["missing"] = "no run directory, so there is no metrics.jsonl to read"
        return out
    path = run / "metrics.jsonl"
    if not path.is_file():
        out["missing"] = ("no metrics.jsonl in %s: the per-step numbers are missing. kit/run_grpo.sh "
                          "needs FILE_LOG=1 to write one" % run)
        return out
    steps = []
    for record in _jsonl(path):
        data = record.get("data") if isinstance(record.get("data"), dict) else record
        if TRAIN_KEYS["reward"] not in data:
            continue
        steps.append({"step": record.get("step"), **{key: data.get(name) for key, name in TRAIN_KEYS.items()}})
    steps.sort(key=lambda row: row["step"] if _finite(row["step"]) else 0)
    out["steps_logged"] = len(steps)
    for key in TRAIN_KEYS:
        out[key] = quarters([row[key] for row in steps])
    if not steps:
        out["missing"] = "metrics.jsonl in %s carries no step with %r" % (run, TRAIN_KEYS["reward"])
    return out


# ------------------------------------------------------------- the training file an arm consumed
def training_file_facts(train_file) -> dict:
    """{rows, distinct, passes, source} for the file an arm trained on, from its own manifest.

    `kit/sequence.py pool` and `kit/subset.py` both write a manifest beside the file they wrote, and
    the manifest is what says how many DISTINCT questions the rows are drawn from -- a 1,920-row pool
    of Spider's 640 is three passes, and reading the row count alone would call it one. A file with
    no manifest (a bed's own train.parquet) is counted from the jsonl beside it, and a file with
    neither is reported as unknown rather than guessed at.
    """
    out = {"rows": None, "distinct": None, "passes": None, "source": None}
    if not train_file:
        return out
    path = Path(train_file)
    folder = path.parent
    pool = folder / "pool.manifest.json"
    subset = folder / "subset.manifest.json"
    try:
        if pool.is_file():
            manifest = json.loads(pool.read_text(encoding="utf-8"))
            out["rows"] = number(manifest, "rows_total")
            distinct = manifest.get("distinct_per_input")
            if isinstance(distinct, list) and distinct and all(isinstance(v, int) for v in distinct):
                out["distinct"] = sum(distinct)
            out["source"] = "pool.manifest.json"
        elif subset.is_file():
            manifest = json.loads(subset.read_text(encoding="utf-8"))
            out["rows"] = out["distinct"] = number(manifest, "kept")
            out["source"] = "subset.manifest.json"
        elif (folder / (path.stem + ".jsonl")).is_file():
            text = (folder / (path.stem + ".jsonl")).read_text(encoding="utf-8")
            out["rows"] = out["distinct"] = len([line for line in text.split("\n") if line.strip()])
            out["source"] = path.stem + ".jsonl"
    except (OSError, json.JSONDecodeError, ValueError):
        return {"rows": None, "distinct": None, "passes": None, "source": None}
    if _finite(out["rows"]) and _finite(out["distinct"]) and out["distinct"]:
        out["passes"] = _round(out["rows"] / out["distinct"], 2)
    return out


# ------------------------------------------------------------------------------------ the build
def build(root: Path, runs: Path) -> dict:
    root, runs = Path(root), Path(runs)
    scorings = latest_found(root / "eval", "bed-score.json")
    if "base-spider" not in scorings:
        raise K3DoseReportError(
            "no untrained scoring under %s: expected eval/base-spider-aN/bed-score.json. Every number "
            "in this report is read against the untrained model's, so there is nothing to report "
            "without it." % root)
    base_path, base = scorings["base-spider"]
    base_correct, base_n = number(base, "correct"), number(base, "n")
    base_cost = cost((base_path, base))
    run_dirs = latest_dirs(runs)
    arms, missing, measured_density = {}, [], base_cost is not None
    for arm in ARMS:
        entry: dict = {"arm": arm, "what": WHAT[arm], "missing": [], "warmup_steps": WARMUP_STEPS,
                       "batch": BATCH, "seed": 0}
        found = scorings.get("%s-spider" % arm)
        run = run_dirs.get("%s-seed0" % arm)
        if found is None:
            entry["missing"].append("no held-out scoring (expected %s/eval/%s-spider-aN/bed-score.json)"
                                    % (root, arm))
        if run is None:
            entry["missing"].append("no run directory (expected %s/%s-seed0-aN/)" % (runs, arm))
        summary = {}
        if run is not None:
            path = run / "train-summary.json"
            if path.is_file():
                try:
                    summary = json.loads(path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    entry["missing"].append("unreadable train-summary.json in %s" % run)
            else:
                entry["missing"].append("no train-summary.json in %s, so what this arm ran is unrecorded"
                                        % run)
        after_path, after = found if found else (None, None)
        entry["run"] = run.name if run is not None else None
        entry["steps"] = number(summary, "steps")
        entry["lr"] = summary.get("lr")
        entry["merged"] = number(summary, "merged")
        entry["returncode"] = number(summary, "returncode")
        entry["train_file"] = summary.get("train_file")
        entry["training_file"] = training_file_facts(entry["train_file"])
        entry["untrained"] = base_correct
        entry["after"] = number(after, "correct")
        entry["n"] = number(after, "n")
        entry["delta"] = (entry["after"] - entry["untrained"]
                          if _finite(entry["after"]) and _finite(entry["untrained"]) else None)
        entry.update(churn((base or {}).get("per_item"), (after or {}).get("per_item")))
        entry["mcnemar_p"] = (mcnemar_exact(entry["right_to_wrong"], entry["wrong_to_right"])
                              if entry["items_compared"] else None)
        if entry["mcnemar_p"] is not None:
            entry["mcnemar_p"] = round(entry["mcnemar_p"], 6)
        entry["truncated_before"] = number(base, "truncated_at_max_tokens")
        entry["truncated_after"] = number(after, "truncated_at_max_tokens")
        entry["training"] = read_metrics(run)
        if entry["training"]["missing"]:
            entry["missing"].append(entry["training"]["missing"])
        after_cost = cost(found) if found else None
        measured_density = measured_density or after_cost is not None
        entry["density"] = compare_cost(after_cost, base_cost)
        entry["machine"] = (after or {}).get("machine", {}).get("id") if after else None
        entry["clears_bar"] = int(_finite(entry["delta"]) and entry["delta"] >= BAR)
        entry["verdict"] = verdict_of(entry)
        arms[arm] = entry
        missing += ["%s: %s" % (arm, reason) for reason in entry["missing"]]
    machines = sorted({m for m in [(base.get("machine") or {}).get("id")]
                       + [arms[a]["machine"] for a in ARMS] if m is not None})
    reported = [arm for arm in ARMS if _finite(arms[arm]["after"])]
    clears = [arm for arm in reported if arms[arm]["clears_bar"]]
    return {"schema": SCHEMA, "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "root": str(root.resolve()), "runs": str(runs.resolve()), "bar": BAR,
            "warmup_steps": WARMUP_STEPS, "batch": BATCH, "seeds": [0],
            "base": {"correct": base_correct, "n": base_n,
                     "truncated_at_max_tokens": number(base, "truncated_at_max_tokens"),
                     "tokens_per_correct": (base_cost or {}).get("tokens_per_correct"),
                     "machine": (base.get("machine") or {}).get("id"),
                     "source": str(Path(base_path).resolve())},
            "arms": arms, "arms_reported": len(reported), "arms_clearing_the_bar": clears,
            "density": {"bar": DENSITY_BAR, "measured": int(measured_density),
                        "note": None if measured_density else NO_TOKENS,
                        "definition": "output tokens over the whole held-out set, divided by the "
                                      "correct answers"},
            "machine_ids": machines, "comparable": int(len(machines) == 1 and bool(machines)),
            "missing": missing}


def verdict_of(entry: dict) -> str:
    """One line per arm: the bar, then how much of the policy moved to get there."""
    if not _finite(entry["delta"]):
        return "no held-out scoring, so this arm has no verdict"
    moved = "policy moved: %d of %d answers changed" % (entry["changed"], entry["items_compared"]) \
        if entry["items_compared"] else "policy movement unknown: the two scorings share no per-item verdicts"
    return "%s the +%d bar (%+d); %s" % ("clears" if entry["clears_bar"] else "does not clear",
                                         BAR, entry["delta"], moved)


# ------------------------------------------------------------------------------------ rendering
def _fmt(value, pattern="%.4f"):
    return pattern % value if _finite(value) else "-"


def _int(value):
    return "%d" % value if _finite(value) else "-"


def render(report: dict) -> str:
    base = report["base"]
    lines = ["# K3 stage A: which dose clears the +%d bar?" % report["bar"], "",
             "Generated %s by kit/k3_dose_report.py (%s). Every number is recomputed from the files "
             "the campaign wrote." % (report["generated_at"], SCHEMA), "",
             "Stage A of K3 was 20 GRPO steps over Spider's 640 training questions at lr 1e-5, and its "
             "pilot bar was +%d on the 100 held-out questions. Four arms change one thing each about "
             "that dose. Every arm here is ONE run at seed 0, and `ref20` is the reference dose rerun "
             "on this machine, so the gap between `ref20` and 0 is what one run against one run is "
             "worth before any arm is read." % report["bar"], "",
             "The untrained Qwen3-1.7B scored **%s of %s**." % (_int(base["correct"]), _int(base["n"])), ""]
    if not report["comparable"]:
        lines += ["**The scorings do not share one machine-and-mode fingerprint (%s), so these "
                  "differences are not comparable.**"
                  % (", ".join(str(m) for m in report["machine_ids"]) or "none recorded"), ""]

    lines += ["## The held-out set (Spider, 100 questions)", "",
              "`right -> wrong` and `wrong -> right` follow the SAME question id through both "
              "scorings, from the `per_item` verdicts each one carries. McNemar's p is the exact "
              "two-sided binomial test on those two counts: it asks whether a policy that changed "
              "nothing could have produced this split, and nothing else.", "",
              "| arm | what it changes | rows | passes | steps | lr | warm-up | untrained | after | "
              "delta | right -> wrong | wrong -> right | unchanged | McNemar p | truncated before | "
              "truncated after | verdict |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for arm in ARMS:
        row = report["arms"][arm]
        file_facts = row["training_file"]
        lines.append("| %s | %s | %s | %s | %s | %s | %d | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            arm, row["what"], _int(file_facts["rows"]), _fmt(file_facts["passes"], "%.2f"),
            _int(row["steps"]), row["lr"] or "-", row["warmup_steps"],
            _int(row["untrained"]), _int(row["after"]),
            "%+d" % row["delta"] if _finite(row["delta"]) else "-",
            _int(row["right_to_wrong"]), _int(row["wrong_to_right"]), _int(row["unchanged"]),
            _fmt(row["mcnemar_p"], "%.3f"), _int(row["truncated_before"]), _int(row["truncated_after"]),
            "clears" if row["clears_bar"] else ("-" if not _finite(row["delta"]) else "does not")))
    lines += [""] + ["- **%s**: %s" % (arm, report["arms"][arm]["verdict"]) for arm in ARMS]

    lines += ["", "## What the trainer logged", "",
              "Per step, from each run's `metrics.jsonl` (`kit/run_grpo.sh` with `FILE_LOG=1`). The "
              "reward is the training reward on the questions being trained on, not a held-out score. "
              "A flat reward with a moving gradient norm is a dose that is learning nothing from the "
              "questions it is being given.", "",
              "| arm | steps logged | reward first quarter | reward last quarter | reward mean | "
              "grad norm first quarter | grad norm last quarter | grad norm mean | entropy mean | "
              "response tokens mean |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for arm in ARMS:
        training = report["arms"][arm]["training"]
        lines.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            arm, _int(training["steps_logged"]),
            _fmt(training["reward"]["first_quarter"], "%.3f"), _fmt(training["reward"]["last_quarter"], "%.3f"),
            _fmt(training["reward"]["mean"], "%.3f"),
            _fmt(training["grad_norm"]["first_quarter"], "%.3f"), _fmt(training["grad_norm"]["last_quarter"], "%.3f"),
            _fmt(training["grad_norm"]["mean"], "%.3f"),
            _fmt(training["entropy"]["mean"], "%.3f"), _fmt(training["response_tokens"]["mean"], "%.1f")))

    block = report["density"]
    lines += ["", "## What a correct answer costs (tokens per correct answer)", "",
              "Plan section 4c, research row Q13. Output tokens over the whole held-out set divided by "
              "the correct answers, so it is a cost per task and not a length. A trained model may "
              "spend at most %.1f times the untrained model's; over the bar the gain is reported at "
              "cost." % block["bar"], ""]
    if not block["measured"]:
        lines += [block["note"] or NO_TOKENS, ""]
    lines += ["| arm | untrained | after | ratio | verdict |", "|---|---|---|---|---|"]
    for arm in ARMS:
        cell = report["arms"][arm]["density"]
        lines.append("| %s | %s | %s | %s | %s |" % (
            arm, show_cost(cell["untrained"]), show_cost(cell["trained"]),
            show_cost(cell["ratio"], 2), cell["verdict"]))

    if report["missing"]:
        lines += ["", "## Missing", "",
                  "Reported as missing, never filled in:", ""] + ["- %s" % item for item in report["missing"]]

    clears = report["arms_clearing_the_bar"]
    lines += ["", "## What to do next", ""]
    if clears:
        lines += ["%s %s the +%d bar (%s). The winner's dose is what stage A of K3 should be re-dosed "
                  "to, and K3 restarts from its gate; nothing beyond that is decided here. This probe "
                  "is ONE seed, so a winning dose is a candidate to confirm at K3's five seeds, not a "
                  "result, and `ref20` is in the table to say how much of any gap could be the "
                  "pipeline repeating itself."
                  % (", ".join("`%s`" % arm for arm in clears),
                     "clears" if len(clears) == 1 else "clear", report["bar"],
                     ", ".join("%s %+d" % (arm, report["arms"][arm]["delta"]) for arm in clears))]
    else:
        lines += ["No arm cleared the +%d bar. None of these three changes to the dose is the answer, "
                  "so stage A is not re-dosed from this probe and K3 stays at its gate; what the churn "
                  "and the training columns above say about WHY is the thing to read next. This probe "
                  "is ONE seed." % report["bar"]]
    lines += ["", "%d of %d arms reported, seed 0, warm-up %d steps on every arm."
              % (report["arms_reported"], len(ARMS), report["warmup_steps"]), ""]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", type=Path, required=True, help="the campaign's k3dose directory")
    parser.add_argument("--runs", type=Path, required=True, help="the directory holding the arms' runs")
    parser.add_argument("--out", type=Path, required=True, help="a new directory; never overwritten")
    args = parser.parse_args(argv)
    if args.out.exists():
        raise SystemExit("refusing to overwrite %s: an output is never replaced" % args.out)
    try:
        report = build(args.root, args.runs)
    except K3DoseReportError as exc:
        raise SystemExit(str(exc))
    args.out.mkdir(parents=True)
    (args.out / "report.json").write_text(json.dumps(report, indent=1, sort_keys=True) + "\n",
                                          encoding="utf-8")
    text = render(report)
    (args.out / "report.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
