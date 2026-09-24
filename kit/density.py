"""Intelligence density: what a correct answer costs in output tokens, on every scoring the kit writes.

Goal 1 is "within 5 to 15 percent of a frontier model on a task". A task has a price, and a model that
reaches the band by writing five times the tokens has not reached it. This file turns the token counts
every scoring already records into one number per scoring, TOKENS PER CORRECT ANSWER, and compares a
trained model's figure with the untrained model's. Nothing here trains, generates or scores anything.

    tokens per correct answer = (sum of output tokens over the whole held-out set) / (correct answers)

so it is a cost per task, not a length: a model that writes short answers and gets none right has an
infinite cost, and a model that writes long answers and gets them all right may still be cheap.

Where the tokens come from, in this order:
  1. `output_tokens_total` in the scoring's own result (bed-score.json, or a panel block of
     forgetting.json) -- written by kit/eval_bed.py and kit/score_forgetting.py from 25 September 2026;
  2. `responses.jsonl` beside the result, which both scorers have always written, one row per answer
     with `output_tokens` (and `panel` for the forgetting panels);
  3. otherwise the figure is unknown and is reported as such. A missing density is NEVER a refusal:
     the counts a report is built on do not depend on it.

The bar (plan section 4c, written 25 September 2026 before any number was read): a trained model spends
at most BAR times the untrained model's tokens per correct answer on the bed it learned and on the
general panel. Over the bar, the gain is reported "at cost" and the arm is not a recipe candidate.

`python density.py training --report report.json` reads a K0-style training log (per-step
`response_tokens` and per-validation accuracy) and says whether the response length climbed while the
held-out score did not: the shape of reward hacking through verbosity.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCHEMA = "kit-density.v1"
BAR = 1.5                       # plan section 4c


class DensityError(Exception):
    """A density could not be computed from what the file names; the message says what is missing."""


def _rows(path: Path) -> list:
    text = path.read_text(encoding="utf-8")
    return [json.loads(line) for line in text.split("\n") if line.strip()]


def output_tokens(result: dict, path: Path, *, panel: str | None = None) -> dict | None:
    """{total, mean, n, source} for one scoring, or None when neither the result nor a responses.jsonl
    beside it carries token counts. `panel` selects one panel of a forgetting.json."""
    block = result
    if panel is not None:
        block = (result.get("panels") or {}).get(panel) or {}
    total = block.get("output_tokens_total")
    if isinstance(total, (int, float)) and not isinstance(total, bool):
        n = block.get("n")
        return {"total": int(total), "n": n, "mean": (total / n) if n else None, "source": "result"}
    responses = Path(path).parent / "responses.jsonl"
    if not responses.is_file():
        return None
    rows = [r for r in _rows(responses) if panel is None or r.get("panel") == panel]
    if not rows or any(not isinstance(r.get("output_tokens"), (int, float)) for r in rows):
        return None
    total = sum(int(r["output_tokens"]) for r in rows)
    return {"total": total, "n": len(rows), "mean": total / len(rows), "source": "responses.jsonl"}


def tokens_per_correct(result: dict, path: Path, *, panel: str | None = None) -> dict | None:
    """{tokens_per_correct, correct, total, n, source}; tokens_per_correct is None when nothing was
    correct (an infinite cost, reported as such), and the whole value is None when tokens are unknown."""
    tokens = output_tokens(result, path, panel=panel)
    if tokens is None:
        return None
    block = result if panel is None else (result.get("panels") or {}).get(panel) or {}
    correct = block.get("correct")
    if not isinstance(correct, (int, float)) or isinstance(correct, bool):
        raise DensityError("%s has no numeric `correct`%s" % (path, "" if panel is None else " for panel %s" % panel))
    return {"tokens_per_correct": (tokens["total"] / correct) if correct else None, "correct": float(correct),
            "total": tokens["total"], "n": tokens["n"], "mean_output_tokens": tokens["mean"], "source": tokens["source"]}


def ratio(trained: dict | None, untrained: dict | None) -> float | None:
    """trained / untrained tokens per correct answer, or None when either is unknown or infinite."""
    if not trained or not untrained:
        return None
    a, b = trained.get("tokens_per_correct"), untrained.get("tokens_per_correct")
    if a is None or b is None or b == 0:
        return None
    return a / b


def verdict(r: float | None, *, bar: float = BAR) -> str:
    """'within bar', 'over bar', or 'unknown'."""
    if r is None:
        return "unknown"
    return "over bar" if r > bar else "within bar"


def compare(trained: dict | None, untrained: dict | None, *, bar: float = BAR) -> dict:
    """One cell of a density table."""
    r = ratio(trained, untrained)
    return {"untrained": (untrained or {}).get("tokens_per_correct"), "trained": (trained or {}).get("tokens_per_correct"),
            "ratio": r, "bar": bar, "verdict": verdict(r, bar=bar)}


def show(value, digits: int = 0) -> str:
    if value is None:
        return "-"
    return ("%%.%df" % digits) % value


# --------------------------------------------------------------- training length against held-out score
def training_trend(training: list, validations: list, *, length_key: str = "response_tokens",
                   step_key: str = "step", accuracy_key: str = "accuracy_from_dump") -> dict:
    """Did the response length climb while the held-out score did not? Compares the mean length over
    the last quarter of training with the first quarter, and the last validation with the first."""
    steps = sorted((row for row in training if isinstance(row.get(length_key), (int, float))), key=lambda r: r[step_key])
    if len(steps) < 4:
        raise DensityError("fewer than 4 training steps with %r" % length_key)
    quarter = max(1, len(steps) // 4)
    first = sum(r[length_key] for r in steps[:quarter]) / quarter
    last = sum(r[length_key] for r in steps[-quarter:]) / quarter
    vals = sorted((v for v in validations if isinstance(v.get(accuracy_key), (int, float))), key=lambda v: v[step_key])
    if len(vals) < 2:
        raise DensityError("fewer than 2 validations with %r" % accuracy_key)
    length_ratio = last / first if first else None
    score_change = vals[-1][accuracy_key] - vals[0][accuracy_key]
    climbed = length_ratio is not None and length_ratio > 1.25
    return {"steps": len(steps), "length_first_quarter": first, "length_last_quarter": last, "length_ratio": length_ratio,
            "score_first": vals[0][accuracy_key], "score_last": vals[-1][accuracy_key], "score_change": score_change,
            "length_climbed": climbed, "score_rose": score_change > 0,
            "shape": ("length up, score not up: the verbosity shape" if climbed and score_change <= 0
                      else "length up with the score" if climbed else "length held")}


def cmd_training(args) -> int:
    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    runs = report.get("runs") or []
    if not runs:
        raise DensityError("%s has no `runs`" % args.report)
    out = {"schema": SCHEMA, "report": str(Path(args.report).resolve()), "runs": {}}
    print("| run | steps | length first quarter | last quarter | ratio | score first | last | shape |")
    print("|---|---|---|---|---|---|---|---|")
    for run in runs:
        trend = training_trend(run.get("training") or [], run.get("validations") or [])
        out["runs"][run.get("name", "?")] = trend
        print("| %s | %d | %s | %s | %s | %s | %s | %s |" % (run.get("name", "?"), trend["steps"], show(trend["length_first_quarter"], 1),
              show(trend["length_last_quarter"], 1), show(trend["length_ratio"], 2), show(trend["score_first"], 4), show(trend["score_last"], 4), trend["shape"]))
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        print("wrote", args.out)
    return 0


def cmd_score(args) -> int:
    """Tokens per correct answer for one or more result files, the first one being the untrained model's."""
    cells = []
    for name in args.results:
        path = Path(name)
        result = json.loads(path.read_text(encoding="utf-8"))
        if "panels" in result:
            for panel in sorted(result["panels"]):
                cells.append((str(path), panel, tokens_per_correct(result, path, panel=panel)))
        else:
            cells.append((str(path), result.get("bed", "-"), tokens_per_correct(result, path)))
    base = {panel: cell for (_, panel, cell) in cells[: len(set(p for _, p, _ in cells))]}
    print("| scoring | bed or panel | correct | output tokens | tokens per correct | ratio to first | verdict |")
    print("|---|---|---|---|---|---|---|")
    for path, panel, cell in cells:
        c = compare(cell, base.get(panel), bar=args.bar)
        print("| %s | %s | %s | %s | %s | %s | %s |" % (path, panel, show((cell or {}).get("correct")), show((cell or {}).get("total")),
              show((cell or {}).get("tokens_per_correct"), 1), show(c["ratio"], 2), c["verdict"]))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    t = sub.add_parser("training", help="response length against held-out score, from a K0-style report.json")
    t.add_argument("--report", required=True)
    t.add_argument("--out")
    t.set_defaults(func=cmd_training)
    s = sub.add_parser("score", help="tokens per correct answer for result files; the first is the untrained model")
    s.add_argument("results", nargs="+")
    s.add_argument("--bar", type=float, default=BAR)
    s.set_defaults(func=cmd_score)
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except DensityError as exc:
        print("refused:", exc, file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
