#!/usr/bin/env python3
"""Scores at every answer budget, read from prefixes of ONE long generation.

    python cap_sweep.py bed   --scoring $WORK/eval/seed0-gsm8k-8k-a1 --bed gsm8k --root /data/gsm8k \\
                              --model /work/models/qwen3-1.7b --out $WORK/sweeps/seed0-gsm8k-a1
    python cap_sweep.py panel --scoring $WORK/forgetting/seed0-8k-a1 --model ... --out $WORK/sweeps/seed0-panel-a1
    python cap_sweep.py check --long $WORK/eval/base-gsm8k-8k-a1 --short $WORK/eval/base-gsm8k-2k-a1 \\
                              --model ... --out $WORK/sweeps/prefix-check-a1.json

Why. Models trained by RL learn to write much longer answers, and the scorers stop generating at a token cap: an
answer cut at the cap is scored as it stands. So a trained model can lose points at 2,048 tokens that it would keep
at 8,192, and the two readings mean different things (it no longer answers within the serving budget; it no longer
answers at all). Rather than generate once per cap, the campaign generates ONCE at a long cap (8,192 new tokens),
keeps the generated token ids (kit/tokens_io.py), and reads the score at every shorter budget N from the first N ids
of each answer. Greedy decoding makes that prefix what a generation capped at N would have produced, IF the engine
is deterministic; `check` measures whether it is, against a scoring generated directly at the shorter cap.

WHAT EACH BUDGET RECORDS, per answer, with n its generated length and N the budget:
    text      = tokenizer.decode(ids[:N], skip_special_tokens=True)
    cut       = n > N, or n == N and the generation stopped at the cap (finish_reason "length")
    strict    = the bed's own reward function on that text (kit/eval_bed.py score_one), 0 or 1
    canonical = kit/canonical.py's gold-blind reading of that text, judged by the bed's own comparison, 0 or 1
    tokens    = min(n, N), what the answer cost at that budget
Curves are recorded, never assumed: an answer right at 512 tokens may be wrong at 1,024 because a later `Answer:`
line replaced the first, and the sweep reports that as it is.

INTEGRITY, refused or recorded. REFUSED, so no number is produced: items that do not hash to the scoring's
`items_sha256` (the items are rebuilt exactly as eval_bed builds them), an item with no answer or no token row, a
token row whose length differs from the answer's `output_tokens`, and any budget above the scoring's cap (an answer
cut at the cap says nothing about a longer budget). RECORDED, for the reader to judge: `decode_equals_text`, how many
answers decode from their ids to exactly the stored text (a tokenizer other than the model's shows here first), and
`reproduces_scoring`, whether the strict score at the cap itself equals the scoring's own per-item verdicts.

`check` compares two scorings of one model on the same items, one at the long cap and one generated directly at a
shorter cap c: does the long answer's first c ids decode to the short answer's text? A low agreement is a recorded
fact and the exit code is still 0: the campaign gates on the number, not on the exit status.

Only `load_tokenizer` imports transformers; everything else is the standard library (and whatever the bed's data
loader needs). Siblings are imported by file path, so `kit/` works when exported alone. Nothing is overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
SCHEMA = "kit-cap-sweep.v1"
CHECK_SCHEMA = "kit-cap-check.v1"
DEFAULT_BUDGETS = "256,512,1024,2048,4096,8192"
MISMATCHES_SHOWN = 20
CONTEXT_CHARS = 60


def _load(name: str):
    spec = importlib.util.spec_from_file_location("kit_cap_sweep_%s" % name, HERE / ("%s.py" % name))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tokens_io = _load("tokens_io")
canonical = _load("canonical")
eval_bed = _load("eval_bed")


class SweepError(ValueError):
    """The scoring folder cannot support a sweep; nothing is written."""


def load_tokenizer(model_dir):
    """The model's own tokenizer. The ONLY place this kit's budget audit imports transformers (tests replace it)."""
    from transformers import AutoTokenizer                                  # noqa: PLC0415
    return AutoTokenizer.from_pretrained(str(model_dir))


def decode(tokenizer, ids) -> str:
    return tokenizer.decode(list(ids), skip_special_tokens=True)


# ---------------------------------------------------------------------------------------- reading
def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict:
    if not path.is_file():
        raise SweepError("%s is missing: is %s a scoring folder?" % (path.name, path.parent))
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise SweepError("%s is not JSON: %s" % (path, exc)) from exc
    if not isinstance(value, dict):
        raise SweepError("%s is not a JSON object" % path)
    return value


def _jsonl(path: Path) -> list:
    if not path.is_file():
        raise SweepError("%s is missing" % path)
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").split("\n"), start=1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except ValueError as exc:
                raise SweepError("%s line %d is not JSON: %s" % (path, number, exc)) from exc
    return rows


def _key(row: dict, panel: bool) -> str:
    """One answer's identity: the id for a bed, `panel/id` for the general panel (ids repeat across panels)."""
    ident = str(row.get("id") or row.get("member_id") or "")
    return "%s/%s" % (row.get("panel"), ident) if panel else ident


def _keyed(rows: list, panel: bool, what: str, path: Path) -> dict:
    keyed: dict = {}
    for row in rows:
        key = _key(row, panel)
        if not key or key.endswith("/"):
            raise SweepError("%s has a %s row with no id" % (path, what))
        if key in keyed:
            raise SweepError("%s has two %s rows for %s" % (path, what, key))
        keyed[key] = row
    return keyed


def scoring_kind(folder: Path) -> str:
    if (folder / "bed-score.json").is_file():
        return "bed"
    if (folder / "forgetting.json").is_file():
        return "panel"
    raise SweepError("%s holds neither bed-score.json nor forgetting.json: not a scoring folder" % folder)


def read_scoring(folder: Path, *, need_tokens: bool) -> dict:
    """The result, the answers and (when present) the token rows of one scoring folder, keyed alike."""
    if not folder.is_dir():
        raise SweepError("no such scoring folder: %s" % folder)
    kind = scoring_kind(folder)
    panel = kind == "panel"
    result = _json(folder / ("forgetting.json" if panel else "bed-score.json"))
    responses_path = folder / "responses.jsonl"
    responses = _keyed(_jsonl(responses_path), panel, "response", responses_path)
    tokens_path = folder / (result.get("tokens_file") or tokens_io.FILE_NAME)
    tokens = None
    if tokens_path.is_file():
        try:
            token_rows = tokens_io.read(tokens_path)
        except (ValueError, KeyError) as exc:
            raise SweepError("%s cannot be read: %s" % (tokens_path, exc)) from exc
        tokens = _keyed(token_rows, panel, "token", tokens_path)
    elif need_tokens:
        raise SweepError("%s has no %s: it was scored before the generated ids were kept, so no prefix can be decoded. "
                         "Generate it again with the current kit/eval_bed.py or kit/score_forgetting.py"
                         % (folder, tokens_path.name))
    cap = result.get("max_new_tokens", (result.get("decoding") or {}).get("max_tokens"))
    if not isinstance(cap, int) or isinstance(cap, bool) or cap <= 0:
        raise SweepError("%s does not say its answer cap (`max_new_tokens`), so no budget can be checked against it" % folder)
    return {"folder": folder, "kind": kind, "result": result, "responses": responses, "tokens": tokens, "cap": cap,
            "responses_path": responses_path, "tokens_path": tokens_path if tokens is not None else None}


def parse_budgets(text: str, cap: int) -> list:
    try:
        budgets = [int(part) for part in str(text).split(",") if part.strip()]
    except ValueError as exc:
        raise SweepError("--budgets must be comma-separated whole numbers, e.g. %s" % DEFAULT_BUDGETS) from exc
    if not budgets or any(b <= 0 for b in budgets) or len(set(budgets)) != len(budgets):
        raise SweepError("--budgets must be distinct positive whole numbers, got %s" % text)
    above = [b for b in budgets if b > cap]
    if above:
        raise SweepError("budget %s is above the scoring's cap of %d new tokens: an answer cut at the cap says nothing "
                         "about a longer budget. Drop it, or generate at a longer cap" % (", ".join(map(str, above)), cap))
    return sorted(budgets)


def check_rows(keys: list, scoring: dict) -> None:
    """Every answer present, with a token row whose length is the answer's `output_tokens`."""
    responses, tokens, folder = scoring["responses"], scoring["tokens"], scoring["folder"]
    missing = [k for k in keys if k not in responses]
    if missing:
        raise SweepError("%d items have no answer in %s (e.g. %s)" % (len(missing), scoring["responses_path"], ", ".join(missing[:3])))
    missing = [k for k in keys if k not in tokens]
    if missing:
        raise SweepError("%d items have no token row in %s (e.g. %s)" % (len(missing), scoring["tokens_path"], ", ".join(missing[:3])))
    extra = sorted((set(responses) | set(tokens)) - set(keys))
    if extra:
        raise SweepError("%s holds %d answers for items not being swept (e.g. %s): the folder is not this scoring"
                         % (folder, len(extra), ", ".join(extra[:3])))
    wrong = [k for k in keys if tokens[k]["n"] != responses[k].get("output_tokens")]
    if wrong:
        k = wrong[0]
        raise SweepError("%d token rows disagree with their answer's length (e.g. %s: %s ids, output_tokens %s): the ids "
                         "are not the ones that answer was generated from" % (len(wrong), k, tokens[k]["n"], responses[k].get("output_tokens")))


def is_cut(n: int, budget: int, finish_reason) -> bool:
    return n > budget or (n >= budget and finish_reason == "length")


# ----------------------------------------------------------------------------------------- sweeps
def sweep_answer(tokenizer, ids: list, finish_reason, budgets: list, score) -> dict:
    """strict, canonical, cut and tokens at each budget for one answer; `score(text) -> (strict, canonical)`."""
    n, full, out = len(ids), None, {"strict": [], "canonical": [], "cut": [], "tokens": []}
    for budget in budgets:
        if budget >= n:
            full = decode(tokenizer, ids) if full is None else full
            text = full
        else:
            text = decode(tokenizer, ids[:budget])
        strict, canon = score(text)
        out["strict"].append(int(strict)); out["canonical"].append(int(canon))
        out["cut"].append(int(is_cut(n, budget, finish_reason))); out["tokens"].append(min(n, budget))
    return out


def _per_budget(budgets: list, rows: list, *, canonical_too: bool) -> list:
    table = []
    for index, budget in enumerate(budgets):
        strict = sum(r["strict"][index] for r in rows)
        cut = sum(r["cut"][index] for r in rows)
        tokens = sum(r["tokens"][index] for r in rows)
        entry = {"budget": budget, "n": len(rows), "correct_strict": strict, "cut": cut, "finished": len(rows) - cut,
                 "tokens_total": tokens, "tokens_per_correct_strict": round(tokens / strict, 3) if strict else None}
        if canonical_too:
            entry["correct_canonical"] = sum(r["canonical"][index] for r in rows)
        table.append(entry)
    return table


def _common(scoring: dict, budgets: list, decode_equal: int, mismatches: list) -> dict:
    result = scoring["result"]
    return {"schema": SCHEMA, "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "scoring": str(scoring["folder"].resolve()), "cap": scoring["cap"], "budgets": budgets,
            # which model folder was scored (bed results record a path, panel results {"path": ...}): a report uses it
            # to refuse a sweep of an earlier attempt of a run that was later retried
            "model": (result.get("model") or {}).get("path") if isinstance(result.get("model"), dict) else result.get("model"),
            "machine": result.get("machine"), "max_model_len": (result.get("engine") or {}).get("max_model_len"),
            "decode_equals_text": decode_equal, "reproduces_scoring": not mismatches,
            "reproduces_scoring_mismatches": len(mismatches), "reproduces_scoring_mismatched_ids": mismatches[:MISMATCHES_SHOWN],
            "responses_sha256": _sha256(scoring["responses_path"]), "tokens_sha256": _sha256(scoring["tokens_path"]),
            "canonical_rule": canonical.RULE}


def _fresh(out: Path) -> Path:
    if out.exists():
        raise SweepError("refusing to overwrite %s: choose a new --out" % out)
    return out


def sweep_bed(args) -> dict:
    out = _fresh(Path(args.out))
    scoring = read_scoring(Path(args.scoring), need_tokens=True)
    result = scoring["result"]
    if scoring["kind"] != "bed":
        raise SweepError("%s is a panel scoring; use the `panel` subcommand" % scoring["folder"])
    if result.get("bed") not in (None, args.bed):
        raise SweepError("%s scored the %s bed, not %s" % (scoring["folder"], result.get("bed"), args.bed))
    budgets = parse_budgets(args.budgets, scoring["cap"])
    module = eval_bed.load_bed(args.bed)
    namespace = SimpleNamespace(bed=args.bed, root=args.root, split=args.split, limit=args.limit,
                                heldout_n=args.heldout_n, allow_subset=args.allow_subset)
    items = eval_bed.items_of(module, namespace)
    digest = eval_bed.items_digest(items)
    if not result.get("items_sha256"):
        raise SweepError("%s records no `items_sha256`, so the rebuilt items cannot be shown to be the ones it scored" % scoring["folder"])
    if digest != result["items_sha256"]:
        raise SweepError("the rebuilt items do not hash to the scoring's items_sha256 (%s vs %s): pass the same --root, "
                         "--split, --heldout-n, --allow-subset and --limit the scoring was made with"
                         % (digest[:12], result["items_sha256"][:12]))
    keys = [item["id"] for item in items]
    check_rows(keys, scoring)
    tokenizer = load_tokenizer(args.model)
    per_item_scored = result.get("per_item") or {}
    sweep_at = budgets if scoring["cap"] in budgets else budgets + [scoring["cap"]]   # the cap is always read, for `reproduces_scoring`
    rows, decode_equal, mismatches = [], 0, []
    for item in items:
        key, response = item["id"], scoring["responses"][item["id"]]
        ids = scoring["tokens"][key]["ids"]
        gold = item["ground_truth"]

        def score(text, gold=gold):
            return (eval_bed.score_one(module, namespace, text, gold)["acc"],
                    canonical.is_correct(args.bed, canonical.extract(args.bed, text), gold))

        swept = sweep_answer(tokenizer, ids, response.get("finish_reason"), sweep_at, score)
        decode_equal += int(decode(tokenizer, ids) == str(response.get("response") or ""))
        at_cap = swept["strict"][sweep_at.index(scoring["cap"])]
        if per_item_scored.get(key) is None or int(per_item_scored[key]) != at_cap:
            mismatches.append(key)
        rows.append({"id": key, "n": len(ids), "finish_reason": response.get("finish_reason"),
                     **{field: values[: len(budgets)] for field, values in swept.items()}})
    report = {**_common(scoring, budgets, decode_equal, mismatches), "kind": "bed", "bed": args.bed,
              "split": eval_bed.split_of(namespace), "n": len(items), "items_sha256": digest,
              "per_budget": _per_budget(budgets, rows, canonical_too=True)}
    out.mkdir(parents=True)
    (out / "sweep.json").write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    (out / "per_item.jsonl").write_text("".join(json.dumps({k: r[k] for k in ("id", "n", "finish_reason", "strict", "canonical", "cut")},
                                                           separators=(",", ":")) + "\n" for r in rows), encoding="utf-8")
    (out / "sweep.md").write_text(render_bed(report), encoding="utf-8")
    return report


def sweep_panel(args) -> dict:
    out = _fresh(Path(args.out))
    scoring = read_scoring(Path(args.scoring), need_tokens=True)
    result = scoring["result"]
    if scoring["kind"] != "panel":
        raise SweepError("%s is a bed scoring; use the `bed` subcommand" % scoring["folder"])
    budgets = parse_budgets(args.budgets, scoring["cap"])
    score_forgetting = _load("score_forgetting")
    members, panel_sha = score_forgetting.load_panel()
    if result.get("panel_file_sha256") != panel_sha:
        raise SweepError("%s was scored on another panel file (%s) than this kit's %s"
                         % (scoring["folder"], str(result.get("panel_file_sha256"))[:12], panel_sha[:12]))
    keys = ["%s/%s" % (m["panel"], m["id"]) for m in members]
    check_rows(keys, scoring)
    tokenizer = load_tokenizer(args.model)
    scored = {name: (slot or {}).get("per_member") or {} for name, slot in (result.get("panels") or {}).items()}
    sweep_at = budgets if scoring["cap"] in budgets else budgets + [scoring["cap"]]
    rows, decode_equal, mismatches = [], 0, []
    for member, key in zip(members, keys):
        response, ids = scoring["responses"][key], scoring["tokens"][key]["ids"]

        def score(text, member=member):
            return score_forgetting.scorers.score(member["panel"], text, member)["correct"], 0

        swept = sweep_answer(tokenizer, ids, response.get("finish_reason"), sweep_at, score)
        decode_equal += int(decode(tokenizer, ids) == str(response.get("response") or ""))
        verdict = scored.get(member["panel"], {}).get(member["id"])
        if verdict is None or int(verdict) != swept["strict"][sweep_at.index(scoring["cap"])]:
            mismatches.append(key)
        rows.append({"panel": member["panel"], "id": member["id"], "n": len(ids), "finish_reason": response.get("finish_reason"),
                     **{field: values[: len(budgets)] for field, values in swept.items() if field != "canonical"}})
    names = list(dict.fromkeys(m["panel"] for m in members))
    per_budget = []
    for index, budget in enumerate(budgets):
        panels = {}
        for name in names:
            mine = [r for r in rows if r["panel"] == name]
            panels[name] = {"n": len(mine), "correct": sum(r["strict"][index] for r in mine),
                            "cut": sum(r["cut"][index] for r in mine), "tokens_total": sum(r["tokens"][index] for r in mine)}
        per_budget.append({"budget": budget, "panels": panels,
                           "total": {key: sum(p[key] for p in panels.values()) for key in ("n", "correct", "cut", "tokens_total")}})
    report = {**_common(scoring, budgets, decode_equal, mismatches), "kind": "panel", "n": len(members),
              "panel_file_sha256": panel_sha, "panels": names, "per_budget": per_budget}
    out.mkdir(parents=True)
    (out / "sweep.json").write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    (out / "per_item.jsonl").write_text("".join(json.dumps({k: r[k] for k in ("panel", "id", "n", "finish_reason", "strict", "cut")},
                                                           separators=(",", ":")) + "\n" for r in rows), encoding="utf-8")
    (out / "sweep.md").write_text(render_panel(report), encoding="utf-8")
    return report


# ------------------------------------------------------------------------------------------ check
def _first_difference(a: str, b: str) -> int:
    for index, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return index
    return min(len(a), len(b))


def _context(text: str, offset: int) -> str:
    start = max(0, offset - CONTEXT_CHARS // 3)
    return text[start: start + CONTEXT_CHARS]


def check(args) -> dict:
    out = _fresh(Path(args.out))
    long = read_scoring(Path(args.long), need_tokens=True)
    short = read_scoring(Path(args.short), need_tokens=False)
    if long["kind"] != short["kind"]:
        raise SweepError("%s is a %s scoring and %s a %s scoring: they do not cover the same items"
                         % (long["folder"], long["kind"], short["folder"], short["kind"]))
    field = "items_sha256" if long["kind"] == "bed" else "panel_file_sha256"
    pair = (long["result"].get(field), short["result"].get(field))
    if not pair[0] or pair[0] != pair[1]:
        raise SweepError("the two scorings do not record the same items (%s %s vs %s): they cannot be compared answer by answer"
                         % (field, str(pair[0])[:12], str(pair[1])[:12]))
    c = short["cap"]
    if c > long["cap"]:
        raise SweepError("the short scoring's cap (%d) is above the long scoring's (%d): pass them the other way round" % (c, long["cap"]))
    tokenizer = load_tokenizer(args.model)
    common = [k for k in long["responses"] if k in short["responses"]]
    missing = [k for k in common if k not in long["tokens"]]
    if missing:
        raise SweepError("%d answers of %s have no token row (e.g. %s)" % (len(missing), long["folder"], ", ".join(missing[:3])))
    with_ids = short["tokens"] is not None
    groups = {status: {"n": 0, "text_agree": 0, "ids_agree": 0 if with_ids else None} for status in ("finished", "cut")}
    shown, mismatched = [], 0
    for key in common:
        ids = long["tokens"][key]["ids"][:c]
        mine = decode(tokenizer, ids)
        theirs = str(short["responses"][key].get("response") or "")
        status = "cut" if short["responses"][key].get("finish_reason") == "length" else "finished"
        group = groups[status]
        group["n"] += 1
        text_ok = mine == theirs
        group["text_agree"] += int(text_ok)
        ids_ok = None
        if with_ids:
            short_ids = (short["tokens"].get(key) or {}).get("ids")
            ids_ok = short_ids is not None and list(short_ids) == list(ids)
            group["ids_agree"] += int(ids_ok)
        if not text_ok or ids_ok is False:
            mismatched += 1
            if len(shown) < MISMATCHES_SHOWN:
                offset = _first_difference(mine, theirs)
                shown.append({"id": key, "short_finish": status, "text_agree": text_ok, "ids_agree": ids_ok,
                              "first_difference_offset": offset if not text_ok else None,
                              "long_prefix_context": _context(mine, offset) if not text_ok else None,
                              "short_context": _context(theirs, offset) if not text_ok else None})
    n = sum(g["n"] for g in groups.values())
    agree = sum(g["text_agree"] for g in groups.values())
    engines = [(s["result"].get("engine") or {}).get("max_model_len") for s in (long, short)]
    machines = [(s["result"].get("machine") or {}).get("id") for s in (long, short)]
    report = {"schema": CHECK_SCHEMA, "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
              "kind": long["kind"], "long": str(long["folder"].resolve()), "short": str(short["folder"].resolve()),
              "long_cap": long["cap"], "short_cap": c, "mode": "text and ids" if with_ids else "text only",
              "compared": n, "only_in_long": len(set(long["responses"]) - set(short["responses"])),
              "only_in_short": len(set(short["responses"]) - set(long["responses"])),
              "text_agree": agree, "text_agreement": round(agree / n, 6) if n else None,
              "text_agreement_among_cut": round(groups["cut"]["text_agree"] / groups["cut"]["n"], 6) if groups["cut"]["n"] else None,
              "ids_agree": sum(g["ids_agree"] for g in groups.values()) if with_ids else None,
              "by_short_finish": groups, "mismatched": mismatched, "mismatches": shown,
              "max_model_len": {"long": engines[0], "short": engines[1]}, "same_max_model_len": engines[0] is not None and engines[0] == engines[1],
              "machine_ids": {"long": machines[0], "short": machines[1]}, "same_machine": machines[0] is not None and machines[0] == machines[1]}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return report


# ---------------------------------------------------------------------------------------- reports
def _per(tokens):
    return "-" if tokens is None else "%.1f" % tokens


def render_bed(report: dict) -> str:
    lines = ["# Cap sweep: %s %s, %d items" % (report["bed"], report["split"], report["n"]), "",
             "Scoring `%s`, generated at most %d new tokens an answer; every budget is read from a prefix of that one generation. "
             "Strict is the bed's own rule, canonical the gold-blind reading of kit/canonical.py (%s). Cut: the answer had not "
             "finished by the budget. %d of %d answers decode from their ids to exactly the stored text; the strict score at the "
             "cap %s the scoring's own verdicts%s." % (report["scoring"], report["cap"], report["canonical_rule"],
                                                       report["decode_equals_text"], report["n"],
                                                       "reproduces" if report["reproduces_scoring"] else "does NOT reproduce",
                                                       "" if report["reproduces_scoring"] else
                                                       " (%d items differ)" % report["reproduces_scoring_mismatches"]), "",
             "| budget | strict correct | canonical correct | cut | finished | tokens | tokens per strict correct |",
             "|---:|---:|---:|---:|---:|---:|---:|"]
    lines += ["| %d | %d | %d | %d | %d | %d | %s |" % (r["budget"], r["correct_strict"], r["correct_canonical"], r["cut"],
                                                     r["finished"], r["tokens_total"], _per(r["tokens_per_correct_strict"]))
              for r in report["per_budget"]]
    return "\n".join(lines) + "\n"


def render_panel(report: dict) -> str:
    names = report["panels"]
    lines = ["# Cap sweep: the general panel, %d questions" % report["n"], "",
             "Scoring `%s`, generated at most %d new tokens an answer; every budget is read from a prefix of that one generation. "
             "Each cell is correct / cut. %d of %d answers decode from their ids to exactly the stored text; the score at the cap "
             "%s the scoring's own verdicts." % (report["scoring"], report["cap"], report["decode_equals_text"], report["n"],
                                                "reproduces" if report["reproduces_scoring"] else "does NOT reproduce"), "",
             "| budget | " + " | ".join(names) + " | total | tokens |", "|---:|" + "---:|" * (len(names) + 2)]
    lines += ["| %d | %s | %d / %d | %d |" % (r["budget"], " | ".join("%d / %d" % (r["panels"][p]["correct"], r["panels"][p]["cut"]) for p in names),
                                             r["total"]["correct"], r["total"]["cut"], r["total"]["tokens_total"]) for r in report["per_budget"]]
    return "\n".join(lines) + "\n"


# -------------------------------------------------------------------------------------------- main
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Scores at every answer budget, from prefixes of one long generation.")
    sub = parser.add_subparsers(dest="action", required=True)
    b = sub.add_parser("bed", help="one bed scoring (bed-score.json, responses.jsonl, tokens.jsonl)")
    b.add_argument("--scoring", required=True, help="the scoring folder kit/eval_bed.py generate wrote")
    b.add_argument("--bed", required=True, choices=list(canonical.BEDS))
    b.add_argument("--root", required=True, help="the bed's data, as given to eval_bed (the gold answers are rebuilt from it)")
    b.add_argument("--split", default=None)
    b.add_argument("--heldout-n", type=int, default=300)
    b.add_argument("--allow-subset", action="store_true")
    b.add_argument("--limit", type=int, default=None)
    p = sub.add_parser("panel", help="one general-panel scoring (forgetting.json, responses.jsonl, tokens.jsonl)")
    p.add_argument("--scoring", required=True)
    for q in (b, p):
        q.add_argument("--model", required=True, help="a directory holding the scored model's tokenizer")
        q.add_argument("--budgets", default=DEFAULT_BUDGETS, help="comma-separated, none above the scoring's cap (default %s)" % DEFAULT_BUDGETS)
        q.add_argument("--out", required=True, help="a new directory")
    c = sub.add_parser("check", help="does a long generation's prefix equal a short-cap generation of the same model?")
    c.add_argument("--long", required=True, help="the scoring at the long cap; must hold tokens.jsonl")
    c.add_argument("--short", required=True, help="the scoring generated directly at the shorter cap; tokens.jsonl optional")
    c.add_argument("--model", required=True)
    c.add_argument("--out", required=True, help="a new JSON file")
    args = parser.parse_args(argv)
    try:
        if args.action == "check":
            r = check(args)
            print("%s scorings, cap %d against %d, %s: %d answers compared" % (r["kind"], r["short_cap"], r["long_cap"], r["mode"], r["compared"]))
            print("prefix text agrees on %d (%s); among answers cut at the short cap %s" % (r["text_agree"], r["text_agreement"], r["text_agreement_among_cut"]))
            print("wrote", args.out)
            return 0
        r = sweep_bed(args) if args.action == "bed" else sweep_panel(args)
        first, last = r["per_budget"][0], r["per_budget"][-1]
        correct = (lambda e: e["correct_strict"]) if r["kind"] == "bed" else (lambda e: e["total"]["correct"])
        print("%s sweep of %d answers at budgets %s (cap %d)" % (r["kind"], r["n"], ",".join(map(str, r["budgets"])), r["cap"]))
        print("strict correct %d at %d, %d at %d; decode equals text %d of %d; reproduces scoring: %s"
              % (correct(first), first["budget"], correct(last), last["budget"], r["decode_equals_text"], r["n"], r["reproduces_scoring"]))
        print("wrote", Path(args.out) / "sweep.json")
        return 0
    except (SweepError, eval_bed.EvalBedError) as exc:
        raise SystemExit(str(exc))


if __name__ == "__main__":
    sys.exit(main())
