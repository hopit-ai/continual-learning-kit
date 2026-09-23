#!/usr/bin/env python3
"""GSM8K as a training and evaluation bed: prompts, a fixed scoring rule, a reward function, a contamination guard.

GSM8K (Cobbe et al., 2021; MIT licence; the `openai/gsm8k` dataset, config `main`) is 8,792 grade-school
word problems, each with a worked solution whose last line is `#### <number>`. The splits are train
(7,473) and test (1,319). THE KIT SHIPS NO GSM8K TEXT: the partner downloads the dataset himself and
points `--gsm8k-root` at the directory that holds it.

THE CONTAMINATION GUARD, and why it is the load-bearing part of this file. 100 of the 300 members of
`kit/panels/general-v1.jsonl` -- the forgetting panel every package is measured on -- are GSM8K
questions. Training on any of them would turn the forgetting measure into a training-set measure, and
nothing downstream would look wrong: the panel score would simply be high. So every question whose
normalised text (lower-case, whitespace collapsed) matches a `math` panel member is removed from the
training set before it is written, the count removed is printed and recorded in the manifest, and the
writer REFUSES to write a file in which any remains -- checking both the question it was built from and
the rendered prompt it would train on, so a renderer that wrapped the question differently is still
caught. On the dataset as published the guard removes nothing: the 100 panel questions are all in the
TEST split (they carry its row positions in their ids, `gsm8k-<test index>`), and no train question
equals a test question. The guard exists for the copy that is not the dataset as published.

THE HELD-OUT SET. The bed needs a measure of the new skill that is not the forgetting panel, so
`prepare` also writes 300 test items chosen deterministically (by the hash of the normalised question)
from the test items that are NOT panel members. It is disjoint from the panel by construction and does
not move when the file order does.

THE SCORING RULE, fixed before any model was scored, and CHANGED on 20 September 2026 before the K3
campaign ran. The instruction still asks for the final answer alone on the last line as
`Answer: <number>`, but the mark is now the FIRST NUMBER in the LAST `Answer:` line, whatever else the
line carries: `Answer: 18 eggs` and `Answer: $18.00` are both 18, and `Answer: 1,800` against a gold of
`1800` still compares as the number it is. The comparison remains numeric and exact.

Why it changed. The old rule required the line to be a bare number after cleaning and scored
`Answer: 18 eggs` zero, which measured obedience to the format as much as arithmetic -- and this bed is
now one half of a forgetting experiment, where a model that learns SQL and loses the habit of writing
the unit-free line would read as a model that has forgotten how to add. Marking the first number
measures the arithmetic. The cost is stated rather than hidden: `Answer: 3 out of 18` is marked as 3,
so a model that writes its answer second is marked wrong, and a model that writes a stray number first
is marked on that number.

CHANGED AGAIN on 23 September 2026 (receipt 222), before any GSM8K training run had reported: a response
with no `Answer:` line but a `\boxed{...}` is marked on the FIRST NUMBER in the LAST `\boxed{}`. Asked for
`Answer: <number>`, the untrained Qwen3-8B wrote `\boxed{440}` instead in 69 of 300 held-out answers, 68 of
them right (receipt 221); under the old rule they were format failures, and a model trained on that
reward would learn to write `Answer:` before it learned any arithmetic. The `Answer:` line, when present,
still decides, because it is the form the instruction asks for.

An `Answer:` line with no number in it at all (`Answer: eighteen`) is a FORMAT FAILURE, and so is a
response with neither an `Answer:` line nor a `\boxed{}` with a number in it: those are answers this rule
cannot mark, and they are counted as `incorrect_format` rather than as wrong arithmetic.

    python gsm8k.py prepare --gsm8k-root /path/to/gsm8k --out /work/data/gsm8k   # CPU, before any GPU is held
    python gsm8k.py score   --gsm8k-root ... --split heldout --responses responses.jsonl

`compute_score` has the signature the SDPO reference expects of a custom reward function. Its feedback
is a function of the model's own answer alone: it never sees, and can never carry, the gold.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path

DATA_SOURCE = "gsm8k"
DATASET = "openai/gsm8k"
DATASET_CONFIG = "main"
SPLIT_SIZES = {"train": 7473, "test": 1319}
HELD_OUT_N = 300
GOLD_MARK = "####"
CURRENCY = "$£€¥"

PANEL_PATH = Path(__file__).resolve().parents[1] / "panels" / "general-v1.jsonl"
PANEL_MATH_MEMBERS = 100
# Every `math` member's prompt is a GSM8K question followed by exactly this. The guard strips it to
# recover the question; a member that does not end with it is refused rather than passed through,
# because a guard that silently matches nothing is worse than no guard at all.
PANEL_PROMPT_SUFFIX = '\n\nSolve step by step, then end your response with "Answer: <number>".'

INSTRUCTION = ("Solve the problem. Work through it step by step, then give the final answer alone on the last line "
               "in the form `Answer: <number>`.")
ANSWER_LINE = re.compile(r"answer\s*[:=]\s*(.+)", re.I)
BOXED = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}")   # the LaTeX final-answer form models write unasked; one nested brace level, as in \boxed{18 \text{ eggs}}
NUMERIC = re.compile(r"[-+]?(?:\d+(?:\.\d+)?|\.\d+)")
# The number to mark, found inside whatever else the answer line says. Thousands separators are part of
# the token -- `1,800` is one number, not `1` followed by `800` -- and are removed before it is parsed.
FIRST_NUMBER = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?|\.\d+)")


# ------------------------------------------------------------------------------------ text and numbers
def normalise(text: str) -> str:
    """The comparison key for contamination: lower-case, whitespace collapsed."""
    return re.sub(r"\s+", " ", str(text)).strip().lower()


def to_number(text):
    """'$ 1,234.50.' -> Decimal('1234.50'); None when the text is not a bare number.

    Decimal, not float, so `18` and `18.0` compare equal exactly and no answer is lost to binary
    rounding. The regex gate is what keeps `nan` and `Infinity` -- which Decimal would happily
    accept -- from parsing as numbers.
    """
    if text is None:
        return None
    token = str(text).strip()
    for sign in CURRENCY:
        token = token.replace(sign, "")
    token = token.replace(",", "").replace(" ", "").rstrip(".").lstrip("+")
    if not NUMERIC.fullmatch(token):
        return None
    try:
        return Decimal(token)
    except InvalidOperation:
        return None


def extract_answer(response: str):
    """The model's final answer: the FIRST number in the LAST `Answer:` line, or, when that line carries no
    number or is absent, the FIRST number in the LAST `\\boxed{...}`.

    `Answer: 18 eggs` -> '18'; `Answer: $18.00` -> '18.00'; `Answer: 3 out of 18` -> '3'; `\\boxed{440}` with
    no `Answer:` line -> '440'. None means neither form was there, or the one found carried no number.
    """
    lines = [m.group(1) for m in ANSWER_LINE.finditer(response or "")]
    found = FIRST_NUMBER.search(lines[-1]) if lines else None
    if found:
        return found.group(0)
    # No `Answer:` line, or one with no number in it (models write `### Final Answer:` and then a boxed
    # number on the next line): the last `\\boxed{}` with a number decides instead.
    boxed = [m.group(1) for m in BOXED.finditer(response or "")]
    found = FIRST_NUMBER.search(boxed[-1]) if boxed else None
    return found.group(0) if found else None


def is_correct(prediction, gold) -> bool:
    """Numeric equality of the cleaned prediction and the gold. Anything unparsed is wrong, never correct."""
    predicted, target = to_number(prediction), to_number(gold)
    return predicted is not None and target is not None and predicted == target


def gold_of(item: dict) -> str:
    """The number after `####` in the dataset's worked solution, commas removed."""
    solution = str(item.get("answer", ""))
    if GOLD_MARK not in solution:
        raise ValueError("the solution has no %s line, so it has no gold answer" % GOLD_MARK)
    shown = solution.split(GOLD_MARK)[-1].strip().replace(",", "")
    if to_number(shown) is None:
        raise ValueError("the %s line is not a number: %r" % (GOLD_MARK, shown[:40]))
    return shown


def render_prompt(item: dict) -> str:
    return "\n".join([INSTRUCTION, "", "Problem: " + str(item["question"]).strip()])


# --------------------------------------------------------------------------------------- loading
def split_files(root, split: str):
    """The files holding one split, and how to read them: the Hugging Face parquet export or a jsonl fallback."""
    base_dir = Path(root)
    for base in (base_dir, base_dir / DATASET_CONFIG):
        for pattern, kind in (("%s-*.parquet" % split, "parquet"), ("%s.parquet" % split, "parquet"),
                              ("%s.jsonl" % split, "jsonl"), ("%s-*.jsonl" % split, "jsonl")):
            hits = sorted(base.glob(pattern))
            if hits:
                return hits, kind
    raise SystemExit("no GSM8K %s split under %s: expected %s-00000-of-00001.parquet (the `%s` config of %s) "
                     "or %s.jsonl with one {'question', 'answer'} object per line"
                     % (split, base_dir, split, DATASET_CONFIG, DATASET, split))


def _read(path: Path, kind: str) -> list:
    if kind == "jsonl":
        return [json.loads(line) for line in path.read_text().split("\n") if line.strip()]
    try:
        import pyarrow.parquet as pq                                          # noqa: PLC0415
    except ImportError:
        raise SystemExit("%s is a parquet file and pyarrow is not installed; convert the split to jsonl "
                         "({'question', 'answer'} per line) or install pyarrow" % path)
    return pq.read_table(path).to_pylist()


def load(root, split: str, expect_size: bool = True) -> list:
    """GSM8K one split, as [{'id', 'question', 'answer', 'gold'}]. Ids carry the row position, as the panel's do."""
    files, kind = split_files(root, split)
    items = []
    for path in files:
        for row in _read(path, kind):
            question = str(row.get("question", "")).strip()
            index = len(items)
            if not question:
                raise SystemExit("GSM8K %s row %d has no question: %s is not the %s dataset" % (split, index, path, DATASET))
            try:
                gold = gold_of(row)
            except ValueError as bad:
                raise SystemExit("GSM8K %s row %d: %s" % (split, index, bad))
            items.append({"id": "%s-%s-%d" % (DATA_SOURCE, split, index), "question": question,
                          "answer": str(row["answer"]), "gold": gold})
    if expect_size and split in SPLIT_SIZES and len(items) != SPLIT_SIZES[split]:
        raise SystemExit("GSM8K %s has %d rows, expected %d: not the documented dataset (%s, config %s). "
                         "Pass --allow-subset to prepare a truncated copy on purpose."
                         % (split, len(items), SPLIT_SIZES[split], DATASET, DATASET_CONFIG))
    return items


# ---------------------------------------------------------------------------- the contamination guard
def panel_questions(panel_path=PANEL_PATH) -> frozenset:
    """The 100 forgetting-panel maths questions, normalised. Refuses anything it cannot read exactly."""
    path = Path(panel_path)
    if not path.is_file():
        raise SystemExit("the forgetting panel is missing: %s. The training set cannot be checked for "
                         "contamination without it, so nothing will be written." % path)
    members = [json.loads(line) for line in path.read_text().split("\n") if line.strip()]
    maths = [m for m in members if m.get("panel") == "math"]
    if len(maths) != PANEL_MATH_MEMBERS:
        raise SystemExit("%s has %d maths members, expected %d: not the panel this guard was written for"
                         % (path, len(maths), PANEL_MATH_MEMBERS))
    unexpected = [m.get("id") for m in maths if not str(m.get("prompt", "")).endswith(PANEL_PROMPT_SUFFIX)]
    if unexpected:
        raise SystemExit("%d panel members do not end with the expected instruction (e.g. %s), so their questions "
                         "cannot be recovered and the guard would match nothing" % (len(unexpected), unexpected[:3]))
    return frozenset(normalise(m["prompt"][: -len(PANEL_PROMPT_SUFFIX)]) for m in maths)


def contaminated(items: list, panel: frozenset) -> list:
    """The items whose question is a forgetting-panel question."""
    return [item for item in items if normalise(item["question"]) in panel]


def remove_contaminated(items: list, panel: frozenset):
    """(kept, removed). Removed items are returned, not counted, so the caller can name them."""
    kept, removed = [], []
    for item in items:
        (removed if normalise(item["question"]) in panel else kept).append(item)
    return kept, removed


def rows_contaminated(rows: list, panel: frozenset) -> list:
    """Trainer rows that still carry a panel question: by the question they were built from, and by the
    rendered prompt containing one. The second check is what a wrapped or reformatted question fails."""
    hits = []
    for row in rows:
        problem = normalise(row["extra_info"]["problem"])
        rendered = normalise(" ".join(turn["content"] for turn in row["prompt"]))
        if problem in panel or any(question in rendered for question in panel):
            hits.append(row["extra_info"]["index"])
    return hits


def held_out(test_items: list, panel: frozenset, n: int = HELD_OUT_N) -> list:
    """A deterministic held-out set from the TEST split, disjoint from the forgetting panel.

    Ordered by the hash of the normalised question, so the same dataset gives the same 300 items
    whatever order the rows arrived in.
    """
    candidates = [item for item in test_items if normalise(item["question"]) not in panel]
    if len(candidates) < n:
        raise SystemExit("%d held-out items were asked for and only %d of %d test items are outside the "
                         "forgetting panel" % (n, len(candidates), len(test_items)))
    ordered = sorted(candidates, key=lambda item: (hashlib.sha256(normalise(item["question"]).encode("utf-8")).hexdigest(), item["id"]))
    return ordered[:n]


# ------------------------------------------------------------------------------------ reward and rows
def compute_score(data_source: str, solution_str: str, ground_truth: str, extra_info: dict | None = None) -> dict:
    """Reward function in the SDPO reference's shape.

    The feedback is chosen from the model's own answer and from `correct` alone; no branch of it reads
    `ground_truth`, so no wording of it can carry the gold -- not the number, and not the elimination
    that naming a near miss would give.
    """
    prediction = extract_answer(solution_str)
    if prediction is None:
        return {"score": 0.0, "acc": 0.0, "pred": "", "incorrect_format": 1,
                "feedback": "No final number was found. End with a last line of the form `Answer: <number>`."}
    correct = is_correct(prediction, ground_truth)
    if correct:
        feedback = ""
    else:
        feedback = ("The final answer is not correct. Re-read the problem, check each arithmetic step, and make sure "
                    "the last line is the quantity the problem asks for.")
    return {"score": float(correct), "acc": float(correct), "pred": prediction, "incorrect_format": 0,
            "feedback": feedback}


def rows_for_trainer(items: list, split: str) -> list:
    return [{"data_source": DATA_SOURCE, "prompt": [{"role": "user", "content": render_prompt(item)}], "ability": DATA_SOURCE,
             "reward_model": {"style": DATA_SOURCE, "ground_truth": item.get("gold") or gold_of(item)},
             "extra_info": {"split": split, "index": item["id"], "problem": item["question"], "description": "", "elo": 0,
                            "achievement_prior": 0}}
            for item in items]


# ------------------------------------------------------------------------------------------- writing
def write_rows(path, rows: list, panel: frozenset) -> str:
    """Write trainer rows as jsonl. Refuses to overwrite, and refuses outright if any row carries a panel question."""
    path = Path(path)
    if path.exists():
        raise SystemExit("refusing to overwrite %s: an output is never replaced, write to a new directory" % path)
    left = rows_contaminated(rows, panel)
    if left:
        raise SystemExit("refusing to write %s: %d of %d rows carry a forgetting-panel question (e.g. %s)"
                         % (path, len(left), len(rows), left[:3]))
    text = "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows)
    path.write_text(text)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_parquet(path, rows: list) -> bool:
    """The same rows as parquet, when pyarrow is installed. Refuses to overwrite."""
    path = Path(path)
    if path.exists():
        raise SystemExit("refusing to overwrite %s: an output is never replaced, write to a new directory" % path)
    try:
        import pyarrow as pa                                                  # noqa: PLC0415
        import pyarrow.parquet as pq                                          # noqa: PLC0415
    except ImportError:
        return False
    pq.write_table(pa.Table.from_pylist(rows), path)
    return True


# ---------------------------------------------------------------------------------------- subcommands
def cmd_prepare(args) -> int:
    root, out = Path(args.gsm8k_root), Path(args.out)
    panel_path = Path(args.panel) if args.panel else PANEL_PATH
    panel = panel_questions(panel_path)
    train = load(root, "train", expect_size=not args.allow_subset)
    test = load(root, "test", expect_size=not args.allow_subset)
    kept, removed = remove_contaminated(train, panel)
    in_test = contaminated(test, panel)
    heldout = held_out(test, panel, args.heldout_n)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "gsm8k.manifest.json"
    if manifest_path.exists():
        raise SystemExit("refusing to overwrite %s: an output is never replaced, write to a new directory" % manifest_path)
    manifest = {"schema": "kit-bed-gsm8k.v1", "dataset": DATASET, "config": DATASET_CONFIG,
                "instruction_sha256": hashlib.sha256(INSTRUCTION.encode("utf-8")).hexdigest(),
                "panel": {"path": str(panel_path), "sha256": hashlib.sha256(panel_path.read_bytes()).hexdigest(),
                          "maths_members": len(panel), "in_train_split": len(removed), "in_test_split": len(in_test)},
                "allow_subset": bool(args.allow_subset), "limit": args.limit, "splits": {}}
    for name, items in (("train", kept[: args.limit] if args.limit else kept), ("heldout", heldout)):
        rows = rows_for_trainer(items, name)
        digest = write_rows(out / ("%s.jsonl" % name), rows, panel)
        wrote_parquet = write_parquet(out / ("%s.parquet" % name), rows)
        manifest["splits"][name] = {"rows": len(rows), "jsonl_sha256": digest, "parquet": wrote_parquet,
                                    "source_split": "train" if name == "train" else "test",
                                    "contaminated_removed": len(removed) if name == "train" else 0}
    manifest_path.write_text(json.dumps(manifest, indent=1, sort_keys=True))
    print("prepared GSM8K in %s: train %d rows (%d removed as forgetting-panel questions), heldout %d rows; "
          "%d of the %d panel questions are in the train split and %d in the test split%s"
          % (out, manifest["splits"]["train"]["rows"], len(removed), manifest["splits"]["heldout"]["rows"],
             len(removed), len(panel), len(in_test),
             "" if manifest["splits"]["train"]["parquet"] else " (jsonl only: pyarrow not installed)"))
    return 0


def cmd_score(args) -> int:
    panel = panel_questions(Path(args.panel) if args.panel else PANEL_PATH)
    root = Path(args.gsm8k_root)
    if args.split == "heldout":
        items = held_out(load(root, "test", expect_size=not args.allow_subset), panel, args.heldout_n)
    else:
        items = load(root, args.split, expect_size=not args.allow_subset)
    by_id = {item["id"]: item for item in items}
    responses = [json.loads(line) for line in Path(args.responses).read_text().split("\n") if line.strip()]
    answered = {row["id"] for row in responses}
    missing, unknown = set(by_id) - answered, answered - set(by_id)
    if missing:
        raise SystemExit("%d items have no response, e.g. %s" % (len(missing), sorted(missing)[:3]))
    if unknown:
        raise SystemExit("%d responses are for items that are not in %s, e.g. %s: the responses do not belong to this "
                         "split" % (len(unknown), args.split, sorted(unknown)[:3]))
    results = [compute_score(DATA_SOURCE, row.get("response", ""), by_id[row["id"]]["gold"]) for row in responses]
    summary = {"split": args.split, "n": len(results), "correct": int(sum(r["acc"] for r in results)),
               "no_final_answer": sum(r["incorrect_format"] for r in results)}
    summary["accuracy"] = round(summary["correct"] / max(1, summary["n"]), 4)
    print(json.dumps(summary))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="GSM8K bed: prepare trainer files, or score saved responses.")
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--gsm8k-root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int)
    p.add_argument("--heldout-n", type=int, default=HELD_OUT_N)
    p.add_argument("--panel")
    p.add_argument("--allow-subset", action="store_true",
                   help="the copy is not the published dataset (a smoke, or a truncated export); recorded in the manifest")
    s = sub.add_parser("score")
    s.add_argument("--gsm8k-root", required=True)
    s.add_argument("--split", default="heldout", choices=["train", "test", "heldout"])
    s.add_argument("--responses", required=True)
    s.add_argument("--heldout-n", type=int, default=HELD_OUT_N)
    s.add_argument("--panel")
    s.add_argument("--allow-subset", action="store_true")
    args = parser.parse_args(argv)
    return {"prepare": cmd_prepare, "score": cmd_score}[args.action](args)


if __name__ == "__main__":
    sys.exit(main())
