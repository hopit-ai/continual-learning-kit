#!/usr/bin/env python3
"""FinQA as a training and evaluation bed: prompts, a fixed scoring rule, and a reward function.

FinQA (Chen et al., 2021; MIT licence; github.com/czyssrs/FinQA) asks numerical questions about a page
of a company's annual report: some text, a table, a question whose answer takes one to three steps of
arithmetic. It was admitted as a bed by receipt 207: a system with no reasoning scores 5.3 percent, and
no report page appears in both the training and the test split (99 of 100 test companies do appear in
training, with other pages: the same customers, new documents).

THE SCORING RULE, fixed before any model was scored (receipt 207). In 590 of the 1,147 test questions
the dataset shows the answer as a percentage ("14.1%") while the value it executes to is a decimal
(0.141). A prediction is therefore correct if it equals the executed value, or 100 times it, or one
hundredth of it, within a relative tolerance of 1 percent; yes/no answers match exactly. Anything
stricter would turn half the panel into a formatting test.

AMENDED the same day, still before any model was scored (receipt 208). Judged by that rule, the
dataset's OWN displayed answers disagreed with its executed values in 18 percent of test items,
because analysts write rounded figures: "14%" for 0.14464, "7%" for 0.06757. A prediction is therefore
ALSO correct if it equals the target rounded to the number of decimal places the prediction was given
with, provided the target is at least 1 in magnitude at that scale (so "0" never matches 0.4).

    python finqa.py prepare --finqa-root /path/to/FinQA/dataset --out /work/data/finqa     # CPU, before any GPU is held
    python finqa.py score   --finqa-root ... --split test --responses responses.jsonl

`compute_score` has the signature the SDPO reference expects of a custom reward function, and its
feedback never contains the gold answer: a teacher may see what went wrong, never the solution.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

DATA_SOURCE = "finqa"
DATASET_COMMIT = "0f16e2867befa6840783e58be38c9efb9229d742"
SPLIT_SIZES = {"train": 6251, "dev": 883, "test": 1147}
TOLERANCE = 0.01
INSTRUCTION = ("Answer the question using the report excerpt. Work through the arithmetic, then give the final answer alone on the last "
               "line in the form `Answer: <number>` (or `Answer: yes` / `Answer: no`).")
ANSWER_LINE = re.compile(r"answer\s*[:=]\s*(.+)", re.I)
NUMBER = re.compile(r"[-+]?\(?\$?\s*\d[\d,]*\.?\d*\)?\s*%?|[-+]?\.\d+\s*%?")


def load(root: Path, split: str) -> list:
    items = json.loads((Path(root) / ("%s.json" % split)).read_text())
    if len(items) != SPLIT_SIZES[split]:
        raise SystemExit("FinQA %s has %d items, expected %d: not the pinned dataset (%s)" % (split, len(items), SPLIT_SIZES[split], DATASET_COMMIT[:7]))
    return items


def render_table(table: list) -> str:
    return "\n".join("| " + " | ".join(str(cell).strip() for cell in row) + " |" for row in table)


def render_prompt(item: dict) -> str:
    parts = [INSTRUCTION, "", "Report excerpt:", " ".join(item["pre_text"]).strip(), "", render_table(item["table"]), "",
             " ".join(item["post_text"]).strip(), "", "Question: " + item["qa"]["question"].strip()]
    return "\n".join(parts)


def gold_of(item: dict) -> str:
    return str(item["qa"]["exe_ans"]).strip()


class Number(float):
    """A parsed number that remembers how many decimal places it was written with."""
    decimals = 0

    def __new__(cls, value: float, decimals: int = 0):
        self = super().__new__(cls, value)
        self.decimals = decimals
        return self


def parse_number(text: str):
    """'$ (1,234.5)' -> -1234.5 (1 decimal) ; '14.1%' -> 14.1 ; None when there is no number."""
    match = NUMBER.search(text)
    if not match:
        return None
    token = match.group(0)
    negative = "(" in token and ")" in token or token.strip().startswith("-")
    digits = re.sub(r"[^\d.]", "", token)
    if digits in ("", "."):
        return None
    try:
        value = float(digits)
    except ValueError:
        return None
    decimals = len(digits.split(".", 1)[1]) if "." in digits else 0
    return Number(-value if negative else value, decimals)


def extract_answer(response: str):
    """The model's final answer: the LAST `Answer:` line. None means the requested format was not followed."""
    lines = [m.group(1).strip() for m in ANSWER_LINE.finditer(response or "")]
    if not lines:
        return None
    last = lines[-1].strip().strip("`*. ")
    low = last.lower()
    if low.startswith("yes"):
        return "yes"
    if low.startswith("no") and not low.startswith(("not", "none")):
        return "no"
    return parse_number(last)


def is_correct(prediction, gold: str) -> bool:
    gold = str(gold).strip().lower()
    if gold in ("yes", "no"):
        return prediction == gold
    if prediction is None or isinstance(prediction, str):
        return False
    try:
        executed = float(gold)
    except ValueError:
        return False
    decimals = getattr(prediction, "decimals", None)
    for target in (executed, executed * 100, executed / 100):
        if abs(prediction - target) <= TOLERANCE * max(abs(target), 1e-9):
            return True
        if decimals is not None and abs(target) >= 1 and abs(round(target, decimals) - float(prediction)) < 0.5 * 10 ** (-decimals) * 1e-6 + 1e-9:
            return True
    return False


def gold_consistent(item: dict) -> bool:
    """Does the dataset's own displayed answer agree with the value its program executes to?

    Under the rule above it does for 91.7 percent of test items. The rest are label problems in the
    dataset (a wrong sign, a program that contradicts the written answer, prose where a number should
    be), and no model can be fairly judged on them: PROCESS 3c excludes items whose gold does not score.
    """
    shown = str(item["qa"].get("answer", "")).strip()
    return bool(shown) and is_correct(extract_answer("Answer: " + shown), gold_of(item))


def compute_score(data_source: str, solution_str: str, ground_truth: str, extra_info: dict | None = None) -> dict:
    """Reward function in the SDPO reference's shape. The feedback never reveals `ground_truth`."""
    prediction = extract_answer(solution_str)
    if prediction is None:
        return {"score": 0.0, "acc": 0.0, "pred": "", "incorrect_format": 1,
                "feedback": "No final answer was found. End with a last line of the form `Answer: <number>`, or `Answer: yes` / `Answer: no`."}
    correct = is_correct(prediction, ground_truth)
    wants_yes_no = str(ground_truth).strip().lower() in ("yes", "no")
    if correct:
        feedback = ""
    elif wants_yes_no != isinstance(prediction, str):
        feedback = "The question asks for %s, and the final answer given was %s." % ("yes or no" if wants_yes_no else "a number", "yes or no" if isinstance(prediction, str) else "a number")
    elif wants_yes_no:
        # On a two-way question "not correct" gives the answer away by elimination, exactly as the zero reward does. 20 of 1,147 test items.
        feedback = "The final answer is not correct. Re-read what the question compares."
    else:
        feedback = "The final answer %s is not correct. Check which rows and years the question refers to, and the order of the operations." % float(prediction)
    return {"score": float(correct), "acc": float(correct), "pred": (prediction if isinstance(prediction, str) else repr(float(prediction))), "incorrect_format": 0, "feedback": feedback}


def rows_for_trainer(items: list, split: str) -> list:
    return [{"data_source": DATA_SOURCE, "prompt": [{"role": "user", "content": render_prompt(item)}], "ability": DATA_SOURCE,
             "reward_model": {"style": DATA_SOURCE, "ground_truth": gold_of(item)},
             "extra_info": {"split": split, "index": item["id"], "problem": item["qa"]["question"], "description": "", "elo": 0, "achievement_prior": 0,
                            "gold_consistent": gold_consistent(item)}}
            for item in items]


def cmd_prepare(args) -> int:
    root, out = Path(args.finqa_root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"schema": "kit-bed-finqa.v1", "dataset_commit": DATASET_COMMIT, "instruction_sha256": hashlib.sha256(INSTRUCTION.encode()).hexdigest(), "splits": {}}
    try:
        import pyarrow as pa                                                 # noqa: PLC0415
        import pyarrow.parquet as pq                                         # noqa: PLC0415
    except ImportError:
        pa = pq = None
    for split, name in (("train", "train"), ("dev", "dev"), ("test", "test")):
        items = load(root, split)
        rows = rows_for_trainer(items[: args.limit] if args.limit else items, split)
        text = "".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in rows)
        (out / ("%s.jsonl" % name)).write_text(text)
        if pq is not None:
            pq.write_table(pa.Table.from_pylist(rows), out / ("%s.parquet" % name))
        manifest["splits"][split] = {"rows": len(rows), "gold_consistent": sum(r["extra_info"]["gold_consistent"] for r in rows), "jsonl_sha256": hashlib.sha256(text.encode()).hexdigest(), "parquet": pq is not None,
                                     "source_sha256": hashlib.sha256((root / ("%s.json" % split)).read_bytes()).hexdigest()}
    (out / "finqa.manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True))
    print("prepared FinQA in %s: %s%s" % (out, {k: v["rows"] for k, v in manifest["splits"].items()}, "" if pq else " (jsonl only: pyarrow not installed)"))
    return 0


def cmd_score(args) -> int:
    items = {item["id"]: item for item in load(Path(args.finqa_root), args.split)}
    rows = [json.loads(line) for line in Path(args.responses).read_text().splitlines() if line.strip()]
    missing = set(items) - {r["id"] for r in rows}
    if missing:
        raise SystemExit("%d items have no response, e.g. %s" % (len(missing), sorted(missing)[:3]))
    scored = [(items[r["id"]], compute_score(DATA_SOURCE, r["response"], gold_of(items[r["id"]]))) for r in rows if r["id"] in items]
    results = [result for _, result in scored]
    clean = [result for item, result in scored if gold_consistent(item)]
    summary = {"split": args.split, "n": len(results), "correct": int(sum(r["acc"] for r in results)), "no_final_answer": sum(r["incorrect_format"] for r in results),
               "gold_consistent_n": len(clean), "gold_consistent_correct": int(sum(r["acc"] for r in clean))}
    summary["accuracy"] = round(summary["correct"] / summary["n"], 4)
    summary["accuracy_on_gold_consistent"] = round(summary["gold_consistent_correct"] / max(1, summary["gold_consistent_n"]), 4)
    print(json.dumps(summary))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="FinQA bed: prepare trainer files, or score saved responses.")
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("prepare"); p.add_argument("--finqa-root", required=True); p.add_argument("--out", required=True); p.add_argument("--limit", type=int)
    s = sub.add_parser("score"); s.add_argument("--finqa-root", required=True); s.add_argument("--split", default="test"); s.add_argument("--responses", required=True)
    args = parser.parse_args(argv)
    return {"prepare": cmd_prepare, "score": cmd_score}[args.action](args)


if __name__ == "__main__":
    sys.exit(main())
