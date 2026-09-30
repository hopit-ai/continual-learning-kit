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

WITH --campaign <yaml> the arm list comes from the campaign file instead of the four names above:

    python k3_dose_report.py --root $WORK/k3dose2 --runs $WORK/runs \\
        --campaign kit/campaigns/k3-dose-2.yaml --out $WORK/k3dose2/report/a1

Every row whose command runs kit/run_grpo.sh is an arm, in the file's order; its `description` is
what the report says it changes, its NAME (less `-a{attempt}`) is the run directory looked for, its
SEED is the seed reported, and its scoring is the `-spider` row that needs it. Steps, lr and
`length_budget_chars` are read from each run's own train-summary.json, and the rows from the file it
trained on, as without --campaign. The report then adds a seed and a length-budget column, and a run
clears only if it reaches the +5 bar AND its tokens per correct answer stay within the density bar:
probe 1 showed a dose that reaches the bar by writing longer answers is not a dose to keep.

RUNS ARE GROUPED INTO ARMS by the row id before `-r<N>` (`budget20-r1`, `-r2`, `-r3` are the arm
`budget20`; a row with no such suffix is an arm of one run), because probe 1's reference arm read +5
where the same recipe had read -8 a day earlier and one run of 100 questions cannot decide a dose.
Each arm gets a bold mean row, and AN ARM CLEARS only if its MEAN delta is at least +5 AND the mean
of its runs' tokens-per-correct ratios is within the density bar AND at least RUNS_AT_BAR (2) of its
runs are individually at +5. An arm with a run unscored is `incomplete` and cannot clear. The
over-budget share of answers is not reported, because verl does not log the reward's `over_budget`
per step. Reading a .yaml campaign needs PyYAML, which the runner already needs. Without --campaign
the output is the probe-1 report exactly as before.

THE ARM RULE IS THE CAMPAIGN'S when it carries a top-level `decision:` block (the anchor test,
kit/campaigns/k3-anchor.yaml, asks "does it stop falling and keep its joins", not "does it gain 5"):

    decision: {min_mean_delta: 0, min_runs_passing: 2, max_density_ratio: 1.5, min_join_share: 30}

`min_mean_delta` is the bar on the arm's mean delta AND on each run (a run passes if its own delta is
at least that), `min_runs_passing` how many runs must pass, `max_density_ratio` the bar on the mean
tokens-per-correct ratio, and the optional `min_join_share` a floor, in percent, on the arm's mean
share of answers that write a JOIN. A campaign with no block (probe 2's) keeps the rule above: 5, 2,
1.5 and no JOIN floor, and renders exactly as it did.

WHAT THE ANSWERS LOOK LIKE. With --campaign the report adds kit/repertoire.py's readout, loaded by
path like density.py: per run and per arm mean, the share of answers writing a JOIN, the share nesting
a subquery, and the style shift in points against the untrained scoring. Receipt 238 found that the
accuracy count hides a collapse from joins to subqueries; this table shows it. A scoring with no
`responses.jsonl` is `-` and is never a refusal; under a JOIN floor it leaves the arm `join share
unknown`, which cannot clear.

MORE THAN ONE BED. An arm's bed is the `--bed` of the eval_bed.py row that scores it (spider when no
scoring row names one), and each bed is read against its own untrained scoring, `base-<bed>` (a
campaign whose arms train on a bed with no such scoring is a refusal, as a missing `base-spider` always
was). The dose-and-anchor test (kit/campaigns/k3-anchor.yaml) trains on Spider and on GSM8K and gives
each bed its own rule:

    decision:
      spider: {min_mean_delta: 0, min_runs_passing: 2, max_density_ratio: 1.5, min_join_share: 30}
      gsm8k: {min_mean_delta: 0, min_runs_passing: 2, max_density_ratio: 1.5, max_format_failures_pp: 5}

A flat block (no bed names) is the rule of every bed, as before. `min_join_share` reads SQL, so it is a
refusal on any bed but Spider. `max_format_failures_pp` is how far, in percentage points of the
held-out set, the arm's MEAN share of `incorrect_format` answers (bed-score.json: the answers the bed
could not parse) may sit above the untrained model's; a run passes it alone on its own share, and a
scoring with no `incorrect_format` leaves the arm `format unknown`, which cannot clear. K1c Part B found
40 steps at lr 1e-5 break GSM8K's answer format (207 of 300 malformed on one seed), which a count of
correct answers reports only as a fall. With two beds or more the report prints one held-out table and
one "what the answers look like" table per bed -- on Spider kit/repertoire.py's SQL constructs, on any
other bed the format failures, the median answer length in tokens and the truncation count -- and
"What to do next" names, per bed, the arms that clear and the best-clearing arm by mean delta. A
campaign on one bed renders exactly as it did.
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


# ------------------------------------------------------------- what the answers look like (receipt 238)
def _load_repertoire():
    """kit/repertoire.py, loaded by path exactly as density.py is; without it every share is unknown."""
    path = HERE / "repertoire.py"
    if not path.is_file():
        return None
    spec = importlib.util.spec_from_file_location("kit_repertoire_for_k3_dose", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


repertoire = _load_repertoire()
NO_RESPONSES = ("No scoring under this root carried a `responses.jsonl` beside its `bed-score.json`, so "
                "every figure in this table is `-`.")


def style_of(found) -> dict | None:
    """kit/repertoire.py's feature shares for one (path, result) pair, or None when unknown."""
    if repertoire is None or not found:
        return None
    return repertoire.of_scoring(Path(found[0]).parent)


def style_cell(shares: dict | None, base_shares: dict | None) -> dict:
    """{join_share, subquery_share, style_shift_points} for one scoring; None where unknown."""
    moved = repertoire.shift(base_shares, shares) if repertoire is not None and shares else {}
    return {"join_share": (shares or {}).get("JOIN"), "subquery_share": (shares or {}).get("subquery"),
            "style_shift_points": moved.get("style_shift_points")}


def show_share(value) -> str:
    return "-" if not _finite(value) else ("%.1f" % value).rstrip("0").rstrip(".")


# ------------------------------------------------------- the answer's form on a bed that is not SQL (K1c)
def answer_tokens(found) -> float | None:
    """The median output tokens of one scoring's answers, from the `responses.jsonl` beside it; None
    when that file is absent or any answer carries no count."""
    if not found:
        return None
    path = Path(found[0]).parent / "responses.jsonl"
    if not path.is_file():
        return None
    counts = [row.get("output_tokens") for row in _jsonl(path)]
    if not counts or not all(_finite(count) for count in counts):
        return None
    return _round(statistics.median(counts), 1)


def form_of(result: dict | None, found=None) -> dict:
    """{format_failures, format_failures_pct, median_answer_tokens, truncated} for one scoring: the
    answers the bed could not parse (`incorrect_format`), as a count and as a percent of `n`."""
    failures, n = number(result, "incorrect_format"), number(result, "n")
    return {"format_failures": failures,
            "format_failures_pct": _round(100.0 * failures / n, 2) if _finite(failures) and n else None,
            "median_answer_tokens": answer_tokens(found),
            "truncated": number(result, "truncated_at_max_tokens")}


def form_cell(after: dict, base: dict) -> dict:
    """form_of() of a trained scoring, with `format_excess_points`: its share of unparsed answers minus
    the untrained model's, in percentage points; None when either is unknown."""
    cell = dict(after)
    mine, theirs = after["format_failures_pct"], base["format_failures_pct"]
    cell["format_excess_points"] = _round(mine - theirs, 2) if _finite(mine) and _finite(theirs) else None
    return cell


# ------------------------------------------------------------------------------------ the arm rule
#: Probe 2's rule, which a campaign with no `decision:` block keeps.
DECISION_KEYS = ("min_mean_delta", "min_runs_passing", "max_density_ratio", "min_join_share",
                 "max_format_failures_pp")
#: The keys a rule may leave unset (None): the floors that apply only where a campaign asks for them.
OPTIONAL_KEYS = ("min_join_share", "max_format_failures_pp")
#: The beds kit/eval_bed.py scores, as the report names them.
BED_TITLE = {"spider": "Spider", "gsm8k": "GSM8K", "finqa": "FinQA", "code": "code"}
BED_OF = re.compile(r"--bed\s+[\"']?(?P<bed>\w+)")


def default_decision() -> dict:
    return {"min_mean_delta": BAR, "min_runs_passing": RUNS_AT_BAR, "max_density_ratio": DENSITY_BAR,
            "min_join_share": None}


def _read_rule(block, path, where: str = "decision") -> dict:
    """One rule over the default; an unknown key or a non-number is a refusal."""
    rule = default_decision()
    if block is None:
        return rule
    if not isinstance(block, dict):
        raise K3DoseReportError("campaign %s: `%s` must be a mapping of %s" % (path, where, ", ".join(DECISION_KEYS)))
    unknown = sorted(set(block) - set(DECISION_KEYS))
    if unknown:
        raise K3DoseReportError("campaign %s: unknown `%s` keys %s (known: %s)"
                                % (path, where, unknown, ", ".join(DECISION_KEYS)))
    for key, value in block.items():
        if not _finite(value) and not (key in OPTIONAL_KEYS and value is None):
            raise K3DoseReportError("campaign %s: `%s.%s` must be a number, not %r" % (path, where, key, value))
        rule[key] = value
    return rule


def read_decision(block, path, beds=("spider",)) -> dict:
    """{bed: rule} for every bed the campaign's arms train on, from its `decision:` block.

    A block whose every value is a mapping is PER BED: it must name exactly the beds the arms train on.
    Any other block (or none) is the rule of every bed. A JOIN floor on a bed that is not Spider is a
    refusal: it reads SQL."""
    beds = list(beds)
    if isinstance(block, dict) and block and all(isinstance(value, dict) for value in block.values()):
        unknown, missing = sorted(set(block) - set(beds)), [bed for bed in beds if bed not in block]
        if unknown:
            raise K3DoseReportError("campaign %s: `decision` names beds %s that no arm trains on (arms train on %s)"
                                    % (path, unknown, ", ".join(beds)))
        if missing:
            raise K3DoseReportError("campaign %s: `decision` has no rule for %s, which arms train on"
                                    % (path, ", ".join(missing)))
        rules = {bed: _read_rule(block[bed], path, "decision.%s" % bed) for bed in beds}
    else:
        rule = _read_rule(block, path)
        rules = {bed: dict(rule) for bed in beds}
    for bed, rule in rules.items():
        if bed != "spider" and rule.get("min_join_share") is not None:
            raise K3DoseReportError("campaign %s: `min_join_share` reads SQL and applies only to Spider, not to %s"
                                    % (path, bed))
    return rules


def _signed(value) -> str:
    """+5, +0, +2.5: a bar as the report prints it."""
    return ("%+d" % value) if float(value).is_integer() else ("%+g" % value)


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


# ------------------------------------------------------------------------------ the arm list
NAME_ATTEMPT = re.compile(r"-a\{attempt\}$")
SCORING_OUT = re.compile(r"/eval/(?P<stem>[^/]+)-a\{attempt\}$")
RUN_OF_ARM = re.compile(r"^(?P<arm>.+)-r(?P<run>\d+)$")
RUNS_AT_BAR = 2             # probe 2's arm rule: at least 2 of an arm's 3 runs individually at +5


def default_arms() -> list:
    """Probe 1's four arms, exactly as this report has always read them."""
    return [{"arm": arm, "what": WHAT[arm], "seed": 0, "run": "%s-seed0" % arm,
             "scoring": "%s-spider" % arm, "group": arm, "bed": "spider"} for arm in ARMS]


def arms_from_campaign(path: Path) -> dict:
    """{name, path, arms} from a kit-campaign file: every row that runs kit/run_grpo.sh is an arm."""
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
        if path.suffix == ".json":
            campaign = json.loads(text)
        else:
            import yaml                                                    # noqa: PLC0415
            campaign = yaml.safe_load(text)
    except (OSError, ValueError) as exc:
        raise K3DoseReportError("cannot read the campaign %s: %s" % (path, exc)) from exc
    rows = (campaign or {}).get("rows") or []
    arms = []
    for row in rows:
        if "run_grpo.sh" not in " ".join(str(part) for part in row.get("command") or []):
            continue
        env = row.get("env") or {}
        name = env.get("NAME")
        if not name or not NAME_ATTEMPT.search(name):
            raise K3DoseReportError("campaign row %s runs kit/run_grpo.sh with no NAME ending in "
                                    "-a{attempt}, so its run directory cannot be found" % row.get("id"))
        seed = env.get("SEED")
        scoring, bed = "%s-spider" % row["id"], "spider"
        for other in rows:
            match = SCORING_OUT.search((other.get("env") or {}).get("OUT", ""))
            command = " ".join(other["command"])
            if match and row["id"] in (other.get("needs") or []) and "eval_bed" in command:
                scoring = match["stem"]
                named = BED_OF.search(command)
                bed = named["bed"] if named else bed
                break
        grouped = RUN_OF_ARM.match(row["id"])
        arms.append({"arm": row["id"], "what": row.get("description") or "-",
                     "seed": int(seed) if str(seed).isdigit() else None,
                     "run": NAME_ATTEMPT.sub("", name), "scoring": scoring,
                     "group": grouped["arm"] if grouped else row["id"], "bed": bed})
    if not arms:
        raise K3DoseReportError("campaign %s has no row that runs kit/run_grpo.sh, so it has no arm" % path)
    beds = list(dict.fromkeys(arm["bed"] for arm in arms))
    rules = read_decision(campaign.get("decision"), path, beds)
    # `decision` is the one rule of a campaign on one bed, exactly as before; with more beds it is {bed: rule}.
    return {"name": campaign.get("name"), "path": str(path.resolve()), "arms": arms, "beds": beds,
            "decisions": rules, "decision": rules[beds[0]] if len(beds) == 1 else rules}


# ------------------------------------------------------------------------------------ the build
def build(root: Path, runs: Path, campaign: dict | None = None) -> dict:
    """The report. `campaign` is arms_from_campaign()'s reading, or None for probe 1's four arms."""
    root, runs = Path(root), Path(runs)
    specs = campaign["arms"] if campaign else default_arms()
    order = [spec["arm"] for spec in specs]
    beds = list(dict.fromkeys(spec.get("bed", "spider") for spec in specs))
    scorings = latest_found(root / "eval", "bed-score.json")
    bases = {}
    for bed in beds:
        if "base-%s" % bed not in scorings:
            raise K3DoseReportError(
                "no untrained scoring under %s: expected eval/base-%s-aN/bed-score.json. Every number "
                "in this report is read against the untrained model's, so there is nothing to report "
                "without it." % (root, bed))
        found = scorings["base-%s" % bed]
        bases[bed] = {"found": found, "result": found[1], "cost": cost(found),
                      "style": style_of(found) if campaign and bed == "spider" else None,
                      "form": form_of(found[1], found) if campaign else None}
    first = bases[beds[0]]
    base_path, base = first["found"]
    base_correct, base_n = number(base, "correct"), number(base, "n")
    base_cost = first["cost"]
    run_dirs = latest_dirs(runs)
    arms, missing = {}, []
    measured_density = any(b["cost"] is not None for b in bases.values())
    rules = (campaign.get("decisions") or {bed: default_decision() for bed in beds}) if campaign else None
    base_style = (bases.get("spider") or {}).get("style")
    styles = {}
    for spec in specs:
        arm = spec["arm"]
        bed = spec.get("bed", "spider")
        rule = rules[bed] if campaign else None
        bar = rule["min_mean_delta"] if campaign else BAR
        untrained = bases[bed]["result"]
        entry: dict = {"arm": arm, "what": spec["what"], "missing": [], "warmup_steps": WARMUP_STEPS,
                       "batch": BATCH, "seed": spec["seed"]}
        found = scorings.get(spec["scoring"])
        run = run_dirs.get(spec["run"])
        if found is None:
            entry["missing"].append("no held-out scoring (expected %s/eval/%s-aN/bed-score.json)"
                                    % (root, spec["scoring"]))
        if run is None:
            entry["missing"].append("no run directory (expected %s/%s-aN/)" % (runs, spec["run"]))
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
        entry["untrained"] = number(untrained, "correct")
        entry["after"] = number(after, "correct")
        entry["n"] = number(after, "n")
        entry["delta"] = (entry["after"] - entry["untrained"]
                          if _finite(entry["after"]) and _finite(entry["untrained"]) else None)
        entry.update(churn((untrained or {}).get("per_item"), (after or {}).get("per_item")))
        entry["mcnemar_p"] = (mcnemar_exact(entry["right_to_wrong"], entry["wrong_to_right"])
                              if entry["items_compared"] else None)
        if entry["mcnemar_p"] is not None:
            entry["mcnemar_p"] = round(entry["mcnemar_p"], 6)
        entry["truncated_before"] = number(untrained, "truncated_at_max_tokens")
        entry["truncated_after"] = number(after, "truncated_at_max_tokens")
        entry["training"] = read_metrics(run)
        if entry["training"]["missing"]:
            entry["missing"].append(entry["training"]["missing"])
        after_cost = cost(found) if found else None
        measured_density = measured_density or after_cost is not None
        entry["density"] = compare_cost(after_cost, bases[bed]["cost"])
        entry["machine"] = (after or {}).get("machine", {}).get("id") if after else None
        entry["clears_bar"] = int(_finite(entry["delta"]) and entry["delta"] >= bar)
        entry["verdict"] = verdict_of(entry, bar)
        if campaign:
            budget = summary.get("length_budget_chars")
            entry["length_budget_chars"] = budget if _finite(budget) else None
            if bed == "spider":
                styles[arm] = style_cell(style_of(found), base_style) if found else style_cell(None, None)
            entry["form"] = form_cell(form_of(after, found), bases[bed]["form"])
            entry["decision"] = decision_of(entry, rule, (styles.get(arm) or {}).get("join_share"),
                                            entry["form"]["format_excess_points"])
            entry["group"] = spec["group"]
            entry["bed"] = bed
        arms[arm] = entry
        missing += ["%s: %s" % (arm, reason) for reason in entry["missing"]]
    machines = sorted({m for m in [(b["result"].get("machine") or {}).get("id") for b in bases.values()]
                       + [arms[a]["machine"] for a in order] if m is not None})
    reported = [arm for arm in order if _finite(arms[arm]["after"])]
    clears = [arm for arm in reported if arms[arm]["clears_bar"]]
    extra, arms_reported = {}, len(reported)
    if campaign:
        groups = group_runs(specs, arms, rules, styles)
        arms_reported = sum(1 for group in groups.values() if group["runs_reported"] == len(group["runs"]))
        by_bed = {bed: clearing(groups, bed) for bed in beds}
        extra = {"campaign": {"name": campaign["name"], "path": campaign["path"]}, "arm_order": order,
                 "decision": campaign.get("decision") or rules[beds[0]],
                 "repertoire": repertoire_block(base_style, styles),
                 "groups": groups, "group_order": list(groups), "runs_reported": len(reported),
                 "arms_clearing_the_decision": [name for bed in beds for name in by_bed[bed]],
                 "beds": beds, "decisions": rules, "clearing_by_bed": by_bed,
                 "best_clearing_by_bed": {bed: (by_bed[bed] or [None])[0] for bed in beds},
                 "bases": {bed: base_block(bases[bed]) for bed in beds}}
    return {"schema": SCHEMA, "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "root": str(root.resolve()), "runs": str(runs.resolve()), "bar": BAR,
            "warmup_steps": WARMUP_STEPS, "batch": BATCH,
            "seeds": sorted({spec["seed"] for spec in specs if spec["seed"] is not None}), **extra,
            "base": {"correct": base_correct, "n": base_n,
                     "truncated_at_max_tokens": number(base, "truncated_at_max_tokens"),
                     "tokens_per_correct": (base_cost or {}).get("tokens_per_correct"),
                     "machine": (base.get("machine") or {}).get("id"),
                     "source": str(Path(base_path).resolve())},
            "arms": arms, "arms_reported": arms_reported, "arms_clearing_the_bar": clears,
            "density": {"bar": DENSITY_BAR, "measured": int(measured_density),
                        "note": None if measured_density else NO_TOKENS,
                        "definition": "output tokens over the whole held-out set, divided by the "
                                      "correct answers"},
            "machine_ids": machines, "comparable": int(len(machines) == 1 and bool(machines)),
            "missing": missing}


def verdict_of(entry: dict, bar=BAR) -> str:
    """One line per arm: the bar, then how much of the policy moved to get there."""
    if not _finite(entry["delta"]):
        return "no held-out scoring, so this arm has no verdict"
    moved = "policy moved: %d of %d answers changed" % (entry["changed"], entry["items_compared"]) \
        if entry["items_compared"] else "policy movement unknown: the two scorings share no per-item verdicts"
    return "%s the %s bar (%+d); %s" % ("clears" if entry["clears_bar"] else "does not clear",
                                        _signed(bar), entry["delta"], moved)


def _join_verdict(share, rule: dict) -> str | None:
    """None when the rule has no JOIN floor or the share clears it; else 'does not' or 'join share unknown'."""
    floor = rule.get("min_join_share")
    if floor is None or (_finite(share) and share >= floor):
        return None
    return "does not" if _finite(share) else "join share unknown"


def _format_verdict(excess, rule: dict) -> str | None:
    """None when the rule has no format limit or `excess` (points above the untrained model's share of
    unparsed answers) is within it; else 'does not' or 'format unknown'."""
    limit = rule.get("max_format_failures_pp")
    if limit is None or (_finite(excess) and excess <= limit):
        return None
    return "does not" if _finite(excess) else "format unknown"


def decision_of(entry: dict, rule: dict | None = None, join_share=None, format_excess=None) -> str:
    """One run against the campaign's rule (probe 2's by default): its delta at the bar AND its tokens
    per correct answer within the density bar AND, under a JOIN floor, its JOIN share at the floor
    AND, under a format limit, its share of unparsed answers within that many points of the untrained's.

    'clears', 'does not', 'density unknown' (the bar is reached but its cost cannot be read, so the
    run cannot clear), 'join share unknown' and 'format unknown' (likewise for its answers), or '-'
    when there is no held-out scoring at all."""
    rule = rule or default_decision()
    if not _finite(entry["delta"]):
        return "-"
    if not entry["clears_bar"]:
        return "does not"
    join = _join_verdict(join_share, rule)
    form = _format_verdict(format_excess, rule)
    if "does not" in (join, form):
        return "does not"
    ratio = entry["density"]["ratio"]
    if ratio is None:
        return "density unknown"
    if ratio > rule["max_density_ratio"]:
        return "does not"
    return join or form or "clears"


FORM_KEYS = ("format_failures", "format_failures_pct", "format_excess_points", "median_answer_tokens", "truncated")


def group_runs(specs: list, entries: dict, rules: dict | None = None, styles: dict | None = None) -> dict:
    """{arm: its runs and their mean} for a campaign, arms in the file's order.

    A run belongs to the arm its id names before `-r<N>`; a row with no such suffix is an arm of one
    run. `rules` is {bed: rule} (a single rule is every bed's). The means are over the runs that were
    scored, and are None when none was; the mean density ratio, and each mean share or figure of the
    answers, is None unless every scored run's figure could be read."""
    rules, styles = rules or {}, styles or {}
    if "min_mean_delta" in rules:
        rules = {spec.get("bed", "spider"): rules for spec in specs}
    groups: dict = {}
    for spec in specs:
        groups.setdefault(spec["group"], {"what": spec["what"], "bed": spec.get("bed", "spider"),
                                          "runs": []})["runs"].append(spec["arm"])
    for group in groups.values():
        rule = rules.get(group["bed"]) or default_decision()
        names = [run for run in group["runs"] if _finite(entries[run]["delta"])]
        scored = [entries[run] for run in names]
        ratios = [run["density"]["ratio"] for run in scored]
        group["runs_reported"] = len(scored)
        group["deltas"] = [run["delta"] for run in scored]
        group["mean_after"] = _mean([run["after"] for run in scored])
        group["mean_delta"] = _mean(group["deltas"])
        group["mean_density_ratio"] = _mean(ratios) if scored and all(_finite(r) for r in ratios) else None
        for key in ("join_share", "subquery_share", "style_shift_points"):
            values = [(styles.get(run) or {}).get(key) for run in names]
            group["mean_" + key] = _mean(values) if names and all(_finite(v) for v in values) else None
        if any("form" in run for run in scored) or rule.get("max_format_failures_pp") is not None:
            for key in FORM_KEYS:
                values = [(run.get("form") or {}).get(key) for run in scored]
                group["mean_" + key] = _mean(values) if scored and all(_finite(v) for v in values) else None
        group["runs_at_bar"] = sum(1 for delta in group["deltas"] if delta >= rule["min_mean_delta"])
        group["decision"] = group_decision_of(group, rule)
    return groups


def group_decision_of(group: dict, rule: dict | None = None) -> str:
    """The arm rule (probe 2's by default): the MEAN delta at least `min_mean_delta` AND at least
    `min_runs_passing` runs individually there AND the mean tokens per correct answer within
    `max_density_ratio` AND, when the campaign sets `min_join_share`, the mean JOIN share at least that
    AND, when it sets `max_format_failures_pp`, the mean share of unparsed answers at most that many
    points above the untrained model's.

    'clears', 'does not', 'density unknown' (the rest holds but the cost cannot be read), 'join share
    unknown' and 'format unknown' (likewise for the answers), 'incomplete' (a run of the arm has no
    held-out scoring, so the rule cannot be applied), or '-' when no run of the arm was scored."""
    rule = rule or default_decision()
    if not group["runs_reported"]:
        return "-"
    if group["runs_reported"] < len(group["runs"]):
        return "incomplete"
    if group["mean_delta"] < rule["min_mean_delta"] or group["runs_at_bar"] < rule["min_runs_passing"]:
        return "does not"
    join = _join_verdict(group.get("mean_join_share"), rule)
    form = _format_verdict(group.get("mean_format_excess_points"), rule)
    if "does not" in (join, form):
        return "does not"
    if group["mean_density_ratio"] is None:
        return "density unknown"
    if group["mean_density_ratio"] > rule["max_density_ratio"]:
        return "does not"
    return join or form or "clears"


def clearing(groups: dict, bed: str) -> list:
    """The arms of one bed that clear its rule, the largest mean delta first (ties keep the file's order)."""
    names = list(groups)
    return sorted((name for name, group in groups.items()
                   if group.get("bed", "spider") == bed and group["decision"] == "clears"),
                  key=lambda name: (-groups[name]["mean_delta"], names.index(name)))


def base_block(base: dict) -> dict:
    """One bed's untrained scoring, as the report's `bases` block carries it."""
    path, result = base["found"]
    return {"correct": number(result, "correct"), "n": number(result, "n"),
            "truncated_at_max_tokens": number(result, "truncated_at_max_tokens"),
            "tokens_per_correct": (base["cost"] or {}).get("tokens_per_correct"),
            "machine": (result.get("machine") or {}).get("id"), "form": base["form"],
            "source": str(Path(path).resolve())}


def repertoire_block(base_style: dict | None, styles: dict) -> dict:
    """The report's `repertoire` block: the untrained model's shares and each run's cell."""
    measured = base_style is not None or any(cell["join_share"] is not None for cell in styles.values())
    return {"measured": int(measured), "note": None if measured else NO_RESPONSES,
            "features": list(repertoire.FEATURES) if repertoire else [],
            "base": {"shares": base_style, "join_share": (base_style or {}).get("JOIN"),
                     "subquery_share": (base_style or {}).get("subquery")},
            "runs": styles}


# ------------------------------------------------------------------------------------ rendering
def _fmt(value, pattern="%.4f"):
    return pattern % value if _finite(value) else "-"


def _int(value):
    return "%d" % value if _finite(value) else "-"


def _budget(value) -> str:
    return "%d chars" % value if _finite(value) else "none"


def _points(value) -> str:
    """+17.3, -2, +0: points above (or below) the untrained model's share, as the report prints them."""
    if not _finite(value):
        return "-"
    text = ("%+.1f" % value).rstrip("0").rstrip(".")
    return "+0" if text in ("+0", "-0") else text


def group_line(group: dict, bar, join_floor=None, format_limit=None) -> str:
    """An arm's mean row, after its verdict: the runs, how many reached the bar, the cost, and --
    under a JOIN floor -- the mean share of answers writing a JOIN, and -- under a format limit --
    how far the mean share of unparsed answers sits above the untrained model's."""
    ratio = group["mean_density_ratio"]
    line = "runs %s; %d of %d at %s or more; mean tokens per correct %s the untrained model's" % (
        ", ".join("%+d" % delta for delta in group["deltas"]) or "-", group["runs_at_bar"],
        len(group["runs"]), _signed(bar), show_cost(ratio, 2) + "x" if ratio is not None else "-")
    if join_floor is not None:
        line += "; JOIN in %s%% of answers (floor %s%%)" % (show_share(group.get("mean_join_share")),
                                                           show_share(join_floor))
    if format_limit is not None:
        line += "; format failures %s points against the untrained model's (limit %s)" % (
            _points(group.get("mean_format_excess_points")), _signed(format_limit))
    return line


def rule_text(rule: dict) -> str:
    """The arm rule in one clause, as the header and 'What to do next' print it."""
    text = ("mean delta %s or more, at least %d runs at %s, and mean tokens per correct answer within "
            "%.1fx" % (_signed(rule["min_mean_delta"]), rule["min_runs_passing"],
                       _signed(rule["min_mean_delta"]), rule["max_density_ratio"]))
    if rule.get("min_join_share") is not None:
        text += ", with a JOIN in at least %s%% of answers (mean over runs)" % show_share(rule["min_join_share"])
    if rule.get("max_format_failures_pp") is not None:
        text += (", with answers the bed cannot parse at most %s points above the untrained model's share "
                 "(mean over runs)" % show_share(rule["max_format_failures_pp"]))
    return text


def render(report: dict) -> str:
    """Markdown. A report built with --campaign adds the seed and length-budget columns and the
    +5-and-density decision; one built without it is probe 1's report, unchanged."""
    base = report["base"]
    camp = report.get("campaign")
    order = report.get("arm_order") or list(ARMS)
    if camp and report.get("beds", ["spider"]) != ["spider"]:
        return render_beds(report)
    if camp:
        groups = report["groups"]
        decision = report.get("decision") or default_decision()
        floor, limit = decision.get("min_join_share"), decision.get("max_format_failures_pp")
        lines = ["# K3 stage A, `%s`: which dose clears the %s bar at a cost within %.1fx%s?"
                 % (camp["name"], _signed(decision["min_mean_delta"]), decision["max_density_ratio"],
                    "" if floor is None else ", writing a JOIN in at least %s%% of its answers"
                    % show_share(floor)), "",
                 "Generated %s by kit/k3_dose_report.py (%s) from the arms of `%s`. Every number is "
                 "recomputed from the files the campaign wrote." % (report["generated_at"], SCHEMA,
                                                                    Path(camp["path"]).name), "",
                 "Stage A of K3 was 20 GRPO steps over Spider's 640 training questions at lr 1e-5, and "
                 "its pilot bar was +%d on the 100 held-out questions. %d arms, %d runs; each run's "
                 "steps, learning rate and length budget are read from its own `train-summary.json`, "
                 "and its seed is only a label. An arm CLEARS only if its MEAN delta is at least %s "
                 "AND its mean tokens per correct answer is at most %.1f times the untrained model's "
                 "AND at least %d of its runs are individually at %s%s: one run of 100 questions "
                 "cannot decide a dose." % (
                     report["bar"], len(groups), len(order), _signed(decision["min_mean_delta"]),
                     decision["max_density_ratio"], decision["min_runs_passing"], _signed(decision["min_mean_delta"]),
                     "" if floor is None else " AND, averaged over its runs, at least %s%% of its "
                     "answers write a JOIN (the untrained model's share is in \"What the answers look "
                     "like\" below)" % show_share(floor)), "",
                 "The untrained Qwen3-1.7B scored **%s of %s**." % (_int(base["correct"]), _int(base["n"])), ""]
    else:
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

    dose_head = "| steps | lr | seed | length budget | warm-up |" if camp else "| steps | lr | warm-up |"
    lines += ["## The held-out set (Spider, 100 questions)", "",
              "`right -> wrong` and `wrong -> right` follow the SAME question id through both "
              "scorings, from the `per_item` verdicts each one carries. McNemar's p is the exact "
              "two-sided binomial test on those two counts: it asks whether a policy that changed "
              "nothing could have produced this split, and nothing else.", "",
              "| arm | what it changes | rows | passes %s untrained | after | "
              "delta | right -> wrong | wrong -> right | unchanged | McNemar p | truncated before | "
              "truncated after | verdict |" % dose_head,
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"
              + ("---|---|" if camp else "")]
    def run_row(arm):
        row = report["arms"][arm]
        file_facts = row["training_file"]
        dose = [_int(row["steps"]), row["lr"] or "-"]
        if camp:
            dose += [_int(row["seed"]), _budget(row["length_budget_chars"])]
        dose.append("%d" % row["warmup_steps"])
        verdict = row["decision"] if camp else \
            ("clears" if row["clears_bar"] else ("-" if not _finite(row["delta"]) else "does not"))
        lines.append("| " + " | ".join([
            arm, row["what"], _int(file_facts["rows"]), _fmt(file_facts["passes"], "%.2f")] + dose + [
            _int(row["untrained"]), _int(row["after"]),
            "%+d" % row["delta"] if _finite(row["delta"]) else "-",
            _int(row["right_to_wrong"]), _int(row["wrong_to_right"]), _int(row["unchanged"]),
            _fmt(row["mcnemar_p"], "%.3f"), _int(row["truncated_before"]), _int(row["truncated_after"]),
            verdict]) + " |")

    if camp:
        for name in report["group_order"]:
            group = groups[name]
            for arm in group["runs"]:
                run_row(arm)
            lines.append("| " + " | ".join(
                ["**%s: mean of %d runs**" % (name, len(group["runs"])), group["what"]] + [""] * 7
                + [_int(base["correct"]), "**%s**" % _fmt(group["mean_after"], "%.1f"),
                   "**%s**" % _fmt(group["mean_delta"], "%+.1f")] + [""] * 6
                + ["**%s**: %s" % (group["decision"], group_line(group, decision["min_mean_delta"], floor, limit))]) + " |")
        lines.append("")
        for arm in order:
            row = report["arms"][arm]
            ratio = row["density"]["ratio"]
            lines.append("- **%s**: %s; tokens per correct answer %s the untrained model's (%s): %s" % (
                arm, row["verdict"], show_cost(ratio, 2) + "x" if ratio is not None else "-",
                row["density"]["verdict"], row["decision"]))
    else:
        for arm in order:
            run_row(arm)
        lines += [""] + ["- **%s**: %s" % (arm, report["arms"][arm]["verdict"]) for arm in order]

    lines += training_lines(report, order, camp) + density_lines(report, order)

    if camp:
        lines += render_repertoire(report)

    if report["missing"]:
        lines += missing_lines(report)

    lines += ["", "## What to do next", ""]
    if camp:
        winners = report["arms_clearing_the_decision"]
        names = report["group_order"]
        rule = rule_text(decision)
        if winners:
            top = [name for name in winners if groups[name]["mean_delta"] == groups[winners[0]]["mean_delta"]]
            lines += ["%s %s the arm rule, %s (%s). %s" % (
                ", ".join("`%s`" % name for name in winners), "clears" if len(winners) == 1 else "clear",
                rule, ", ".join("%s mean %+.1f" % (name, groups[name]["mean_delta"]) for name in winners),
                ("`%s` has the largest mean gain, so its dose becomes K3's stage-A dose and K3 restarts "
                 "from its gate; nothing beyond that is decided here." % top[0]) if len(top) == 1 else
                ("%s share the largest mean gain, so this probe does not choose between them; K3's "
                 "stage-A dose is one of them, and nothing beyond that is decided here."
                 % ", ".join("`%s`" % name for name in top)))]
        else:
            scored = [name for name in names if groups[name]["mean_delta"] is not None]
            best = max(scored, key=lambda name: (groups[name]["mean_delta"], -names.index(name))) \
                if scored else None
            lines += ["No arm cleared the arm rule, %s. Stage A is not re-dosed from this probe and K3 "
                      "stays at its gate.%s" % (rule, (
                          " The best mean is `%s` at %+.1f (%s)." % (
                              best, groups[best]["mean_delta"],
                              group_line(groups[best], decision["min_mean_delta"], floor, limit)))
                          if best else "")]
        lines += unclear_lines(names, groups)
        lines += [reported_line(report), ""]
        return "\n".join(lines)
    clears = report["arms_clearing_the_bar"]
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


def training_lines(report: dict, order: list, camp) -> list:
    """'What the trainer logged', one row per run."""
    lines = ["", "## What the trainer logged", "",
             "Per step, from each run's `metrics.jsonl` (`kit/run_grpo.sh` with `FILE_LOG=1`). The "
             "reward is the training reward on the questions being trained on, not a held-out score. "
             "A flat reward with a moving gradient norm is a dose that is learning nothing from the "
             "questions it is being given.", ""]
    if camp:
        lines += ["With a length budget the reward is 0 for any answer over it, so a falling reward "
                  "beside a rising response length is the budget biting. The share of answers over the "
                  "budget is not in this table: verl does not log the reward's `over_budget` per step.", ""]
    lines += ["| arm |%s steps logged | reward first quarter | reward last quarter | reward mean | "
              "grad norm first quarter | grad norm last quarter | grad norm mean | entropy mean | "
              "response tokens mean |" % (" length budget |" if camp else ""),
              "|---|---|---|---|---|---|---|---|---|---|" + ("---|" if camp else "")]
    for arm in order:
        training = report["arms"][arm]["training"]
        cells = [arm] + ([_budget(report["arms"][arm]["length_budget_chars"])] if camp else []) + [
            _int(training["steps_logged"]),
            _fmt(training["reward"]["first_quarter"], "%.3f"), _fmt(training["reward"]["last_quarter"], "%.3f"),
            _fmt(training["reward"]["mean"], "%.3f"),
            _fmt(training["grad_norm"]["first_quarter"], "%.3f"), _fmt(training["grad_norm"]["last_quarter"], "%.3f"),
            _fmt(training["grad_norm"]["mean"], "%.3f"),
            _fmt(training["entropy"]["mean"], "%.3f"), _fmt(training["response_tokens"]["mean"], "%.1f")]
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def density_lines(report: dict, order: list) -> list:
    """'What a correct answer costs', one row per run, each against its own bed's untrained model."""
    block = report["density"]
    lines = ["", "## What a correct answer costs (tokens per correct answer)", "",
             "Plan section 4c, research row Q13. Output tokens over the whole held-out set divided by "
             "the correct answers, so it is a cost per task and not a length. A trained model may "
             "spend at most %.1f times the untrained model's; over the bar the gain is reported at "
             "cost." % block["bar"], ""]
    if not block["measured"]:
        lines += [block["note"] or NO_TOKENS, ""]
    lines += ["| arm | untrained | after | ratio | verdict |", "|---|---|---|---|---|"]
    for arm in order:
        cell = report["arms"][arm]["density"]
        lines.append("| %s | %s | %s | %s | %s |" % (
            arm, show_cost(cell["untrained"]), show_cost(cell["trained"]),
            show_cost(cell["ratio"], 2), cell["verdict"]))
    return lines


def missing_lines(report: dict) -> list:
    return ["", "## Missing", "", "Reported as missing, never filled in:", ""] + \
        ["- %s" % item for item in report["missing"]]


def unclear_lines(names: list, groups: dict) -> list:
    """The arms the rule could not be applied to, and why: each a paragraph, none when there are none."""
    def named(verdict):
        return ", ".join("`%s`" % name for name in names if groups[name]["decision"] == verdict)
    lines = []
    if named("density unknown"):
        lines += ["", "%s met the delta half of the rule but carried no token counts, so its cost "
                  "cannot be read and it cannot clear until it is re-scored with them." % named("density unknown")]
    if named("join share unknown"):
        lines += ["", "%s met the rest of the rule but its answers could not be read (no "
                  "`responses.jsonl` beside a scoring), so its JOIN share is unknown and it cannot "
                  "clear until it is re-scored with them." % named("join share unknown")]
    if named("format unknown"):
        lines += ["", "%s met the rest of the rule but a scoring carried no `incorrect_format` count, so its "
                  "format failures are unknown and it cannot clear until it is re-scored." % named("format unknown")]
    if named("incomplete"):
        lines += ["", "%s is missing a run's held-out scoring, so the arm rule cannot be applied to "
                  "it; run the missing rows before reading it." % named("incomplete")]
    return lines


def reported_line(report: dict) -> str:
    return ("\n%d of %d arms reported (%d of %d runs), seeds %s (labels only), warm-up %d steps on every arm."
            % (report["arms_reported"], len(report["group_order"]), report["runs_reported"],
               len(report["arm_order"]), ", ".join(str(s) for s in report["seeds"]) or "-",
               report["warmup_steps"]))


# ------------------------------------------------------------------------------ more than one bed
#: What a bed's best-clearing arm decides, as "What to do next" says it.
NEXT = {"spider": "its dose is the Spider dose to carry forward (K3's stage A and K5's SQL job), and nothing "
                  "beyond that is decided here",
        "gsm8k": "its dose replaces K3's stage-B dose (40 steps at lr 1e-5, which K1c Part B found breaks the "
                 "answer format), and nothing beyond that is decided here"}


def _title(bed: str) -> str:
    return BED_TITLE.get(bed, bed)


def held_out_head() -> list:
    """The held-out table's header, as a campaign prints it."""
    return ["| arm | what it changes | rows | passes | steps | lr | seed | length budget | warm-up | untrained | "
            "after | delta | right -> wrong | wrong -> right | unchanged | McNemar p | truncated before | "
            "truncated after | verdict |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]


def campaign_run_row(row: dict, arm: str) -> str:
    """One run's row of a campaign's held-out table."""
    facts = row["training_file"]
    return "| " + " | ".join([
        arm, row["what"], _int(facts["rows"]), _fmt(facts["passes"], "%.2f"), _int(row["steps"]), row["lr"] or "-",
        _int(row["seed"]), _budget(row["length_budget_chars"]), "%d" % row["warmup_steps"],
        _int(row["untrained"]), _int(row["after"]), "%+d" % row["delta"] if _finite(row["delta"]) else "-",
        _int(row["right_to_wrong"]), _int(row["wrong_to_right"]), _int(row["unchanged"]),
        _fmt(row["mcnemar_p"], "%.3f"), _int(row["truncated_before"]), _int(row["truncated_after"]),
        row["decision"]]) + " |"


def mean_row(name: str, group: dict, untrained, rule: dict) -> str:
    """An arm's bold mean row, with its verdict and the line that explains it."""
    return "| " + " | ".join(
        ["**%s: mean of %d runs**" % (name, len(group["runs"])), group["what"]] + [""] * 7
        + [_int(untrained), "**%s**" % _fmt(group["mean_after"], "%.1f"), "**%s**" % _fmt(group["mean_delta"], "%+.1f")]
        + [""] * 6 + ["**%s**: %s" % (group["decision"], group_line(
            group, rule["min_mean_delta"], rule.get("min_join_share"), rule.get("max_format_failures_pp")))]) + " |"


def render_beds(report: dict) -> str:
    """Markdown for a campaign on more than one bed (or on one that is not Spider): per bed, a held-out
    table, what the answers look like, and what to do next."""
    camp, groups, beds = report["campaign"], report["groups"], report["beds"]
    rules, bases, order = report["decisions"], report["bases"], report["arm_order"]
    arms_of = {bed: [name for name in report["group_order"] if groups[name]["bed"] == bed] for bed in beds}
    titles = [_title(bed) for bed in beds]
    lines = ["# `%s`: which dose does not break the model on %s, and does an anchor help at that dose?"
             % (camp["name"], " or ".join(titles)), "",
             "Generated %s by kit/k3_dose_report.py (%s) from the arms of `%s`. Every number is recomputed "
             "from the files the campaign wrote." % (report["generated_at"], SCHEMA, Path(camp["path"]).name), "",
             "%d arms, %d runs on %d bed%s. Each run's steps, learning rate and length budget are read from its "
             "own `train-summary.json`, and its seed is only a label. Each bed is read against the untrained "
             "model's own scoring on it, under its own arm rule from the campaign's `decision:` block; one run "
             "of a held-out set cannot decide a dose, so every rule is on an arm's mean and on how many of its "
             "runs pass alone." % (len(groups), len(order), len(beds), "" if len(beds) == 1 else "s"), ""]
    lines += ["- **%s**: an arm CLEARS only with %s." % (_title(bed), rule_text(rules[bed])) for bed in beds]
    lines += ["", "The untrained Qwen3-1.7B scored %s." % ", ".join(
        "**%s of %s** on %s" % (_int(bases[bed]["correct"]), _int(bases[bed]["n"]), _title(bed)) for bed in beds), ""]
    if not report["comparable"]:
        lines += ["**The scorings do not share one machine-and-mode fingerprint (%s), so these "
                  "differences are not comparable.**"
                  % (", ".join(str(m) for m in report["machine_ids"]) or "none recorded"), ""]
    for bed in beds:
        lines += ["## The held-out set (%s, %s questions)" % (_title(bed), _int(bases[bed]["n"])), "",
                  "`right -> wrong` and `wrong -> right` follow the SAME question id through both "
                  "scorings, from the `per_item` verdicts each one carries. McNemar's p is the exact "
                  "two-sided binomial test on those two counts: it asks whether a policy that changed "
                  "nothing could have produced this split, and nothing else.", ""] + held_out_head()
        for name in arms_of[bed]:
            group = groups[name]
            lines += [campaign_run_row(report["arms"][arm], arm) for arm in group["runs"]]
            lines.append(mean_row(name, group, bases[bed]["correct"], rules[bed]))
        lines.append("")
        for name in arms_of[bed]:
            for arm in groups[name]["runs"]:
                row = report["arms"][arm]
                ratio = row["density"]["ratio"]
                lines.append("- **%s**: %s; tokens per correct answer %s the untrained model's (%s): %s" % (
                    arm, row["verdict"], show_cost(ratio, 2) + "x" if ratio is not None else "-",
                    row["density"]["verdict"], row["decision"]))
        lines.append("")
    lines.pop()
    lines += training_lines(report, order, camp) + density_lines(report, order)
    for bed in beds:
        if bed == "spider":
            lines += render_repertoire(report, arms_of[bed], "## What the answers look like (Spider)")
        else:
            lines += render_form(report, bed, arms_of[bed])
    if report["missing"]:
        lines += missing_lines(report)
    lines += ["", "## What to do next", ""]
    for bed in beds:
        names, rule = arms_of[bed], rules[bed]
        winners = report["clearing_by_bed"][bed]
        head = "**%s** (the rule: %s)." % (_title(bed), rule_text(rule))
        if winners:
            top = [name for name in winners if groups[name]["mean_delta"] == groups[winners[0]]["mean_delta"]]
            lines.append("%s %s %s (%s). %s" % (
                head, ", ".join("`%s`" % name for name in winners), "clears" if len(winners) == 1 else "clear",
                ", ".join("%s mean %+.1f" % (name, groups[name]["mean_delta"]) for name in winners),
                ("The best-clearing arm by mean delta is `%s`: %s." % (top[0], NEXT.get(bed, "nothing beyond "
                 "that is decided here"))) if len(top) == 1 else
                ("%s share the largest mean delta, so this test does not choose between them; the %s dose is "
                 "one of them, and nothing beyond that is decided here."
                 % (", ".join("`%s`" % name for name in top), _title(bed)))))
        else:
            scored = [name for name in names if groups[name]["mean_delta"] is not None]
            best = max(scored, key=lambda name: (groups[name]["mean_delta"], -names.index(name))) if scored else None
            lines.append("%s No %s arm cleared it, so no %s dose is chosen from this test.%s" % (
                head, _title(bed), _title(bed), (" The best mean is `%s` at %+.1f (%s)." % (
                    best, groups[best]["mean_delta"], group_line(
                        groups[best], rule["min_mean_delta"], rule.get("min_join_share"),
                        rule.get("max_format_failures_pp")))) if best else ""))
        lines += unclear_lines(names, groups) + [""]
    lines.pop()
    lines += [reported_line(report), ""]
    return "\n".join(lines)


def render_form(report: dict, bed: str, names: list) -> list:
    """'What the answers look like' on a bed that is not SQL: format failures, answer length, truncation."""
    groups, base = report["groups"], report["bases"][bed]["form"] or {}
    lines = ["", "## What the answers look like (%s)" % _title(bed), "",
             "K1c Part B (30 September): 40 GRPO steps at lr 1e-5 broke the untrained 1.7B's answers on %s before "
             "they changed its arithmetic -- 245 of 300 right before, 137 after, with as many as 207 of 300 "
             "answers the bed could no longer parse on one seed -- and a count of correct answers reports that "
             "only as a fall. `format failures` is bed-score.json's `incorrect_format` (answers the bed could not "
             "parse), as a count and as a share of the held-out set; `against untrained` is that share minus the "
             "untrained model's, in percentage points. The median answer length is the median `output_tokens` "
             "of the `responses.jsonl` beside each scoring, and `truncated` counts answers cut off at the token "
             "limit." % _title(bed), ""]
    lines += ["| arm | format failures | format failures, % | against untrained (points) | median answer tokens | "
              "truncated |", "|---|---|---|---|---|---|",
              "| untrained | %s | %s | - | %s | %s |" % (
                  _int(base.get("format_failures")), show_share(base.get("format_failures_pct")),
                  show_share(base.get("median_answer_tokens")), _int(base.get("truncated")))]
    for name in names:
        group = groups[name]
        for run in group["runs"]:
            cell = report["arms"][run].get("form") or {}
            lines.append("| %s | %s | %s | %s | %s | %s |" % (
                run, _int(cell.get("format_failures")), show_share(cell.get("format_failures_pct")),
                _points(cell.get("format_excess_points")), show_share(cell.get("median_answer_tokens")),
                _int(cell.get("truncated"))))
        lines.append("| **%s: mean of %d runs** | **%s** | **%s** | **%s** | **%s** | **%s** |" % (
            name, len(group["runs"]), show_share(group.get("mean_format_failures")),
            show_share(group.get("mean_format_failures_pct")), _points(group.get("mean_format_excess_points")),
            show_share(group.get("mean_median_answer_tokens")), show_share(group.get("mean_truncated"))))
    return lines


def render_repertoire(report: dict, names: list | None = None, heading: str = "## What the answers look like") -> list:
    """'What the answers look like': per run and per arm mean, from kit/repertoire.py."""
    block, groups = report["repertoire"], report["groups"]
    base = block["base"]
    lines = ["", heading, "",
             "Receipt 238: twenty GRPO steps on Spider change HOW the model writes SQL before they change "
             "how often it is right. In every run of the reference dose the untrained model's JOINs "
             "(41 of 100 answers) gave way to nested subqueries (0 to 9 JOINs after), and the questions "
             "it lost were the ones whose gold answer joins tables; one run read +5 on the count with the "
             "same collapse. Each figure is the share of the 100 held-out answers whose SQL uses the "
             "construct (kit/repertoire.py, the SQL taken from each answer as the bed takes it). The style "
             "shift is the mean absolute change, in percentage points, over %d constructs against the "
             "untrained scoring: 0 is the untrained repertoire unchanged." % len(block["features"] or []), ""]
    if not block["measured"]:
        lines += [block["note"] or NO_RESPONSES, ""]
    lines += ["| arm | JOIN share | subquery share | style shift (points) |", "|---|---|---|---|",
              "| untrained | %s | %s | - |" % (show_share(base["join_share"]), show_share(base["subquery_share"]))]
    for name in report["group_order"] if names is None else names:
        group = groups[name]
        for run in group["runs"]:
            cell = block["runs"].get(run) or {}
            lines.append("| %s | %s | %s | %s |" % (run, show_share(cell.get("join_share")),
                                                    show_share(cell.get("subquery_share")),
                                                    show_share(cell.get("style_shift_points"))))
        lines.append("| **%s: mean of %d runs** | **%s** | **%s** | **%s** |" % (
            name, len(group["runs"]), show_share(group.get("mean_join_share")),
            show_share(group.get("mean_subquery_share")), show_share(group.get("mean_style_shift_points"))))
    return lines


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", type=Path, required=True, help="the campaign's k3dose directory")
    parser.add_argument("--runs", type=Path, required=True, help="the directory holding the arms' runs")
    parser.add_argument("--out", type=Path, required=True, help="a new directory; never overwritten")
    parser.add_argument("--campaign", type=Path, default=None,
                        help="the campaign file to read the arms from; without it, probe 1's four arms")
    args = parser.parse_args(argv)
    if args.out.exists():
        raise SystemExit("refusing to overwrite %s: an output is never replaced" % args.out)
    try:
        campaign = arms_from_campaign(args.campaign) if args.campaign else None
        report = build(args.root, args.runs, campaign)
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
