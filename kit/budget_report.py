#!/usr/bin/env python3
"""The four-score accounting of a bed score lost after training, at a serving budget and a diagnostic one.

    python budget_report.py --ref $WORK/sweeps/base-gsm8k-a1 \\
                            --after seed0=$WORK/sweeps/seed0-gsm8k-a1 --after seed1=$WORK/sweeps/seed1-gsm8k-a1 \\
                            --serving 2048 --diagnostic 8192 --seeds-label "GRPO gsm8k, seeds 0-4" \\
                            --out $WORK/budget/gsm8k-a1

Every input is a `sweep.json` from kit/cap_sweep.py `bed`: one scoring read at several answer budgets, strictly (the
bed's own rule) and canonically (kit/canonical.py's gold-blind reading). `--ref` is the reference model (usually the
untrained one); each `--after NAME=DIR` is one later scoring, usually one training seed.

THE FOUR SCORES. For a scoring x, S_x(N) is its strict correct count at budget N and E_x(N) its canonical correct
count. With B the serving budget and H the diagnostic one, the strict loss at the serving budget,

    F = S_r(B) - S_a(B),

is written as the exact sum of three terms:

    residual   = E_r(H) - E_a(H)                              what is still lost when answers may run to H and any
                                                              stated final answer counts
    budget     = [E_a(H) - E_a(B)] - [E_r(H) - E_r(B)]        how much more the later model gains from the extra
                                                              tokens than the reference does
    extraction = [E_a(B) - S_a(B)] - [E_r(B) - S_r(B)]        how much more of the later model's answers at B only
                                                              the canonical reading finds

    F == residual + budget + extraction, in integers, always; the report asserts it and writes `accounting_exact`.

These are SIGNED ACCOUNTING TERMS, NOT CAUSES. A positive budget term says the later model's canonical score rises
more between B and H than the reference's does; it does not say length is why the strict score fell.

THE LABELS, fixed in advance. Per `--after`, with n questions:
    "still cut"  more than 10 percent of its answers are cut at H: the diagnostic budget is itself too short to read;
    "intact"     otherwise, if S_a(H) >= S_r(H) - ceil(0.05 n)  (five per 100 questions);
    "degraded"   otherwise.
Per bed, over all --after entries (k of them): "intact" if at least ceil(4k/5) are intact, "degraded" if at least
ceil(3k/5) are degraded, otherwise "mixed or inconclusive". Intact means the strict score recovers within that
tolerance at the long cap; degraded means a measured failure remains at it. Neither proves the ability was
preserved or destroyed.

REFUSED: a sweep that is not of kind "bed", sweeps of different beds or items, a budget (B or H) a sweep did not
read, B not below H, and scorings from different machines (a bed count moves by several points across machines
before any training) unless --allow-different-machines, which is recorded. Standard library only. Nothing is
overwritten: an existing --out is refused.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = "kit-budget-report.v1"
STILL_CUT_SHARE = (1, 10)          # more than 1/10 of answers cut at H
TOLERANCE_PER_100 = 5              # S_a(H) may trail S_r(H) by ceil(5n/100)
QUALIFY_AGREEMENT = 0.99           # a prefix check (--qualify) passes at this share of agreeing answers, cut ones included
BED_INTACT = (4, 5)                # at least ceil(4k/5) seeds intact
BED_DEGRADED = (3, 5)              # at least ceil(3k/5) seeds degraded


class BudgetReportError(ValueError):
    """The sweeps cannot support the accounting; nothing is written."""


def ceil_frac(value: int, numerator: int, denominator: int) -> int:
    """ceil(value * numerator / denominator) in integers, so no rounding of a float decides a label."""
    return -((-value * numerator) // denominator)


def read_sweep(where) -> dict:
    path = Path(where)
    path = path / "sweep.json" if path.is_dir() else path
    if not path.is_file():
        raise BudgetReportError("no sweep.json at %s: run kit/cap_sweep.py bed first" % where)
    try:
        sweep = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise BudgetReportError("%s is not JSON: %s" % (path, exc)) from exc
    if not isinstance(sweep, dict) or sweep.get("kind") != "bed":
        raise BudgetReportError("%s is not a bed sweep (kind %r): the accounting needs strict and canonical counts"
                                % (path, sweep.get("kind") if isinstance(sweep, dict) else None))
    sweep["_path"] = str(path.resolve())
    sweep["_by_budget"] = {entry["budget"]: entry for entry in sweep.get("per_budget") or []}
    return sweep


def at(sweep: dict, budget: int, name: str) -> dict:
    if budget not in sweep["_by_budget"]:
        raise BudgetReportError("%s did not read budget %d (it read %s): sweep that scoring again with %d among --budgets"
                                % (name, budget, ", ".join(map(str, sorted(sweep["_by_budget"]))), budget))
    return sweep["_by_budget"][budget]


def machine_id(sweep: dict):
    return (sweep.get("machine") or {}).get("id")


def seed_label(n: int, cut_h: int, strict_h: int, ref_strict_h: int) -> str:
    if cut_h * STILL_CUT_SHARE[1] > n * STILL_CUT_SHARE[0]:
        return "still cut"
    if strict_h >= ref_strict_h - ceil_frac(n, TOLERANCE_PER_100, 100):
        return "intact"
    return "degraded"


def bed_label(labels: list) -> str:
    k = len(labels)
    if labels.count("intact") >= ceil_frac(k, *BED_INTACT):
        return "intact"
    if labels.count("degraded") >= ceil_frac(k, *BED_DEGRADED):
        return "degraded"
    return "mixed or inconclusive"


def account(ref: dict, after: dict, serving: int, diagnostic: int, name: str) -> dict:
    """The four scores of `after` against `ref`, the three terms, and the seed's label."""
    rb, rh = at(ref, serving, "--ref"), at(ref, diagnostic, "--ref")
    ab, ah = at(after, serving, name), at(after, diagnostic, name)
    S_rB, S_aB, S_rH, S_aH = rb["correct_strict"], ab["correct_strict"], rh["correct_strict"], ah["correct_strict"]
    E_rB, E_aB, E_rH, E_aH = rb["correct_canonical"], ab["correct_canonical"], rh["correct_canonical"], ah["correct_canonical"]
    loss = S_rB - S_aB
    residual = E_rH - E_aH
    budget = (E_aH - E_aB) - (E_rH - E_rB)
    extraction = (E_aB - S_aB) - (E_rB - S_rB)
    exact = loss == residual + budget + extraction
    assert exact, "the accounting identity failed: %d != %d + %d + %d" % (loss, residual, budget, extraction)
    n = after["n"]
    ref_per, after_per = rb.get("tokens_per_correct_strict"), ab.get("tokens_per_correct_strict")
    return {"name": name, "sweep": after["_path"], "n": n, "machine_id": machine_id(after),
            "scores": {"strict_serving": S_aB, "canonical_serving": E_aB, "strict_diagnostic": S_aH, "canonical_diagnostic": E_aH},
            "F": loss, "residual": residual, "budget": budget, "extraction": extraction, "accounting_exact": int(exact),
            "cut_diagnostic": ah["cut"], "cut_share_diagnostic": round(ah["cut"] / n, 6) if n else None,
            "tokens_per_correct_serving": {"ref": ref_per, "after": after_per,
                                           "ratio": round(after_per / ref_per, 4) if ref_per and after_per is not None else None},
            "tolerance": ceil_frac(n, TOLERANCE_PER_100, 100),
            "label": seed_label(n, ah["cut"], S_aH, S_rH),
            # the same rule applied to the canonical (gold-blind) reading, printed beside the label of record
            "label_canonical": seed_label(n, ah["cut"], E_aH, E_rH)}


def read_checks(specs: list, sweeps: dict, serving=None) -> dict:
    """{name: {file, compared, text_agreement, same_machine, same_max_model_len, long_scoring, short_cap, long_cap,
    bound, passed, why}} from `--qualify NAME=FILE`; `sweeps` is {name: sweep} with "ref" for the reference.

    FILE is what `kit/cap_sweep.py check` wrote for that scoring: a direct generation at the serving budget against the
    same-length prefix of the long one, on one GPU in one context. The scoring QUALIFIES (`passed`) when
      - the file is there and readable (a named check that is absent does not qualify: `why` "missing");
      - at least QUALIFY_AGREEMENT of the answers agree, cut ones included, on one machine and one context length;
      - the check is of the scoring this sweep read: the long scoring's folder NAME equals the name of the folder the
        sweep read (names, not paths, so the rule survives the tree being packed and moved), its long cap is the
        sweep's cap, and its short cap is the serving budget. An old passing check does not qualify a new scoring.
    A scoring that does not qualify is left out of the report and listed under `disqualified`; a reference that does
    not qualify refuses the report."""
    checks = {}
    for spec in specs:
        name, sep, where = spec.partition("=")
        if not sep or not name or not where:
            raise BudgetReportError("--qualify %r is not NAME=CHECK_FILE" % spec)
        if name not in sweeps:
            raise BudgetReportError("--qualify %s names no scoring of this report (it has: %s)" % (name, ", ".join(sorted(sweeps))))
        if name in checks:
            raise BudgetReportError("--qualify %s is given twice" % name)
        entry = {"file": str(Path(where)), "compared": None, "text_agreement": None, "same_machine": None, "same_max_model_len": None,
                 "long_scoring": None, "short_cap": None, "long_cap": None, "bound": None, "passed": False, "why": None}
        try:
            check = json.loads(Path(where).read_text(encoding="utf-8"))
            if not isinstance(check, dict):
                raise ValueError("not a JSON object")
        except (OSError, ValueError):
            entry["why"] = "missing"
            checks[name] = entry
            continue
        agreement, compared, long = check.get("text_agreement"), check.get("compared"), check.get("long")
        number = isinstance(agreement, (int, float)) and not isinstance(agreement, bool)
        agrees = bool(compared) and number and agreement >= QUALIFY_AGREEMENT and check.get("same_machine") is True and check.get("same_max_model_len") is True
        sweep = sweeps[name]
        scoring = Path(sweep["scoring"]).name if isinstance(sweep.get("scoring"), str) and sweep.get("scoring") else None
        long_name = Path(long).name if isinstance(long, str) and long else None
        bound = bool(long_name is not None and long_name == scoring and check.get("long_cap") == sweep.get("cap")
                     and (serving is None or check.get("short_cap") == serving))
        entry.update({"file": str(Path(where).resolve()), "compared": compared, "text_agreement": agreement, "same_machine": check.get("same_machine"),
                      "same_max_model_len": check.get("same_max_model_len"), "long_scoring": long_name, "short_cap": check.get("short_cap"),
                      "long_cap": check.get("long_cap"), "bound": bound, "passed": bool(agrees and bound),
                      "why": None if agrees and bound else ("failed" if not agrees else "of another scoring (%s; the sweep read %s)" % (long_name, scoring))})
        checks[name] = entry
    return checks


def build(args) -> dict:
    serving, diagnostic = int(args.serving), int(args.diagnostic)
    if serving >= diagnostic:
        raise BudgetReportError("--serving (%d) must be below --diagnostic (%d)" % (serving, diagnostic))
    if not args.after:
        raise BudgetReportError("give at least one --after NAME=SWEEP_DIR")
    ref = read_sweep(args.ref)
    afters, seen = [], set()
    for spec in args.after:
        name, sep, where = spec.partition("=")
        if not sep or not name or not where:
            raise BudgetReportError("--after %r is not NAME=SWEEP_DIR" % spec)
        if name in seen:
            raise BudgetReportError("--after %s is given twice" % name)
        seen.add(name)
        afters.append((name, read_sweep(where)))
    disqualified, checks = [], read_checks(getattr(args, "qualify", None) or [], {"ref": ref, **dict(afters)}, serving)
    if "ref" in checks and not checks["ref"]["passed"]:
        raise BudgetReportError("the reference's prefix check does not qualify it (%s: %s, agreement %s): no curve read against it can be used"
                                % (checks["ref"]["file"], checks["ref"]["why"], checks["ref"]["text_agreement"]))
    for name in [n for n, _ in afters if n in checks and not checks[n]["passed"]]:
        disqualified.append({"name": name, **checks[name]})
    afters = [(name, sweep) for name, sweep in afters if not (name in checks and not checks[name]["passed"])]
    if not afters:
        raise BudgetReportError("no --after scoring is qualified by its prefix check (%s): there is nothing to report"
                                % ", ".join(d["name"] for d in disqualified))
    for name, sweep in afters:
        if sweep.get("bed") != ref.get("bed"):
            raise BudgetReportError("%s swept the %s bed and --ref the %s bed" % (name, sweep.get("bed"), ref.get("bed")))
        if not sweep.get("items_sha256") or sweep.get("items_sha256") != ref.get("items_sha256") or sweep.get("n") != ref.get("n"):
            raise BudgetReportError("%s was scored on other items than --ref (items_sha256 %s vs %s)"
                                    % (name, str(sweep.get("items_sha256"))[:12], str(ref.get("items_sha256"))[:12]))
    machines = {"ref": machine_id(ref), **{name: machine_id(sweep) for name, sweep in afters}}
    same_machine = machines["ref"] is not None and len(set(machines.values())) == 1
    if not same_machine and not args.allow_different_machines:
        raise BudgetReportError("the sweeps were not all scored on one machine (%s): bed counts move by several points across "
                                "machines before any training. Score the reference on the same machine, or pass "
                                "--allow-different-machines, which is recorded" % ", ".join("%s=%s" % kv for kv in machines.items()))
    records = [account(ref, sweep, serving, diagnostic, name) for name, sweep in afters]
    rb, rh = at(ref, serving, "--ref"), at(ref, diagnostic, "--ref")
    budgets = sorted(set.intersection(*(set(s["_by_budget"]) for s in [ref] + [sw for _, sw in afters])))
    curve = []
    for budget in budgets:
        row = {"budget": budget, "ref": {"strict": ref["_by_budget"][budget]["correct_strict"],
                                         "canonical": ref["_by_budget"][budget]["correct_canonical"],
                                         "cut": ref["_by_budget"][budget]["cut"]}}
        for name, sweep in afters:
            entry = sweep["_by_budget"][budget]
            row[name] = {"strict": entry["correct_strict"], "canonical": entry["correct_canonical"], "cut": entry["cut"],
                         "strict_minus_ref": entry["correct_strict"] - row["ref"]["strict"]}
        curve.append(row)
    labels = [r["label"] for r in records]
    return {"schema": SCHEMA, "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "bed": ref.get("bed"), "n": ref.get("n"), "items_sha256": ref.get("items_sha256"),
            "serving": serving, "diagnostic": diagnostic, "seeds_label": args.seeds_label,
            "ref": {"sweep": ref["_path"], "machine_id": machines["ref"], "cap": ref.get("cap"),
                    "scores": {"strict_serving": rb["correct_strict"], "canonical_serving": rb["correct_canonical"],
                               "strict_diagnostic": rh["correct_strict"], "canonical_diagnostic": rh["correct_canonical"]},
                    "cut_diagnostic": rh["cut"], "cut_share_diagnostic": round(rh["cut"] / ref["n"], 6) if ref.get("n") else None,
                    "tokens_per_correct_serving": rb.get("tokens_per_correct_strict")},
            "scorings_reported": len(records), "accounting_exact": int(all(r["accounting_exact"] for r in records)),
            "same_machine": same_machine, "different_machines_allowed": bool(args.allow_different_machines), "machine_ids": machines,
            "prefix_checks": checks, "disqualified": disqualified, "scorings_disqualified": len(disqualified),
            "records": records, "seed_labels": {r["name"]: r["label"] for r in records}, "bed_label": bed_label(labels),
            "label_counts": {label: labels.count(label) for label in ("intact", "degraded", "still cut")},
            "seed_labels_canonical": {r["name"]: r["label_canonical"] for r in records},
            "bed_label_canonical": bed_label([r["label_canonical"] for r in records]),
            "budgets_in_all_sweeps": budgets, "curve": curve}


# ------------------------------------------------------------------------------------------ render
def _ratio(value) -> str:
    return "-" if value is None else "%.2f" % value


def _tokens(value) -> str:
    return "-" if value is None else "%.1f" % value


def render(report: dict) -> str:
    B, H, n = report["serving"], report["diagnostic"], report["n"]
    names = [r["name"] for r in report["records"]]
    ref = report["ref"]
    lines = ["# Budget report: %s, %d questions" % (report["bed"], n), "",
             "%d later scoring%s%s against the reference `%s`, read at a serving budget of %d and a diagnostic budget of %d "
             "new tokens. Every number comes from kit/cap_sweep.py reading one long generation per scoring at shorter prefixes. "
             "Strict (S) is the bed's own scoring rule; canonical (E) is the gold-blind reading of kit/canonical.py, judged by "
             "the same comparison with the gold. Per-bed label: **%s**."
             % (len(names), "" if len(names) == 1 else "s", " (%s)" % report["seeds_label"] if report["seeds_label"] else "",
                ref["sweep"], B, H, report["bed_label"]), ""]
    if report.get("disqualified"):
        lines += ["**Left out: %s.** Their prefix check does not qualify them (a direct generation at the serving budget must agree "
                  "with the prefix of the long one on at least %d percent of answers, on one machine and one context, and the check "
                  "must be there and be of the scoring that was swept), so their curves are not read here."
                  % (", ".join("%s (%s, agreement %s)" % (d["name"], d["why"], d["text_agreement"]) for d in report["disqualified"]),
                     round(100 * QUALIFY_AGREEMENT)), ""]
    if not report["same_machine"]:
        lines += ["The scorings were NOT all made on one machine (%s); they are compared anyway because "
                  "--allow-different-machines was given." % ", ".join("%s=%s" % kv for kv in report["machine_ids"].items()), ""]
    lines += ["## The four scores", "",
              "| scoring | S(%d) | E(%d) | S(%d) | E(%d) | cut at %d | tokens per strict correct at %d | ratio to reference | label |" % (B, B, H, H, H, B),
              "|---|---:|---:|---:|---:|---:|---:|---:|---|",
              "| reference | %d | %d | %d | %d | %d (%.1f%%) | %s | - | - |"
              % (ref["scores"]["strict_serving"], ref["scores"]["canonical_serving"], ref["scores"]["strict_diagnostic"],
                 ref["scores"]["canonical_diagnostic"], ref["cut_diagnostic"], 100 * (ref["cut_share_diagnostic"] or 0),
                 _tokens(ref["tokens_per_correct_serving"]))]
    for r in report["records"]:
        s = r["scores"]
        lines.append("| %s | %d | %d | %d | %d | %d (%.1f%%) | %s | %s | %s |"
                     % (r["name"], s["strict_serving"], s["canonical_serving"], s["strict_diagnostic"], s["canonical_diagnostic"],
                        r["cut_diagnostic"], 100 * (r["cut_share_diagnostic"] or 0), _tokens(r["tokens_per_correct_serving"]["after"]),
                        _ratio(r["tokens_per_correct_serving"]["ratio"]), r["label"]))
    lines += ["", "## The accounting", "",
              "F = S_ref(%d) - S(%d), the strict loss at the serving budget, is the exact sum of three signed terms: residual = "
              "E_ref(%d) - E(%d); budget = [E(%d) - E(%d)] - [E_ref(%d) - E_ref(%d)]; extraction = [E(%d) - S(%d)] - "
              "[E_ref(%d) - S_ref(%d)]. They are accounting terms, not causes: each says where in the four scores the loss "
              "sits, not why it happened. Identity exact for every scoring: %s."
              % (B, B, H, H, H, B, H, B, B, B, B, B, "yes" if report["accounting_exact"] else "NO"), "",
              "| scoring | F | residual | budget | extraction |", "|---|---:|---:|---:|---:|"]
    lines += ["| %s | %+d | %+d | %+d | %+d |" % (r["name"], r["F"], r["residual"], r["budget"], r["extraction"]) for r in report["records"]]
    tolerance = report["records"][0]["tolerance"] if report["records"] else 0
    lines += ["", "## The labels", "",
              "Per scoring: \"still cut\" if more than 10 percent of its answers are cut at %d; otherwise \"intact\" if "
              "S(%d) >= S_ref(%d) - %d (five per 100 of %d questions, rounded up); otherwise \"degraded\". Per bed: \"intact\" "
              "if at least four fifths of the scorings (rounded up) are intact, \"degraded\" if at least three fifths (rounded "
              "up) are degraded, otherwise \"mixed or inconclusive\". Intact means the score recovers within tolerance at the "
              "long cap; degraded means a measured failure remains at it. Neither proves the ability was preserved or destroyed."
              % (H, H, H, tolerance, n), "",
              "Counts: %s. Per-bed label: %s." % (", ".join("%s %d" % kv for kv in report["label_counts"].items()), report["bed_label"]),
              "", "The label of record is read from the strict score. The same rule applied to the canonical reading gives: %s; per bed: %s. "
              "The canonical reading takes the last answer the model had stated, provisional or not, so it is the more generous of the two."
              % (", ".join("%s %s" % kv for kv in report["seed_labels_canonical"].items()), report["bed_label_canonical"]),
              "", "## Score against budget", "",
              "Strict correct (canonical correct) at every budget all the sweeps read.", "",
              "| budget | reference | " + " | ".join(names) + " |", "|---:|---:|" + "---:|" * len(names)]
    for row in report["curve"]:
        lines.append("| %d | %d (%d) | %s |" % (row["budget"], row["ref"]["strict"], row["ref"]["canonical"],
                                               " | ".join("%d (%d)" % (row[name]["strict"], row[name]["canonical"]) for name in names)))
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="The four-score accounting of a bed score at a serving and a diagnostic budget.")
    parser.add_argument("--ref", required=True, help="the reference scoring's sweep directory (kit/cap_sweep.py bed)")
    parser.add_argument("--after", action="append", default=[], metavar="NAME=SWEEP_DIR", help="a later scoring; repeat per seed")
    parser.add_argument("--serving", type=int, required=True, help="the serving budget B, e.g. 2048")
    parser.add_argument("--diagnostic", type=int, required=True, help="the diagnostic budget H, e.g. 8192")
    parser.add_argument("--seeds-label", default=None, help="free text naming what the --after scorings are, printed in the lead")
    parser.add_argument("--allow-different-machines", action="store_true", help="compare scorings from different machines; recorded")
    parser.add_argument("--qualify", action="append", default=[], metavar="NAME=CHECK_FILE",
                        help="a prefix check (kit/cap_sweep.py check) of the scoring NAME, or of `ref`: a scoring whose check is "
                             "absent, does not reach %.2f agreement on one machine and one context, or is of another scoring is "
                             "left out and listed" % QUALIFY_AGREEMENT)
    parser.add_argument("--out", required=True, help="a new directory")
    args = parser.parse_args(argv)
    out = Path(args.out)
    if out.exists():
        raise SystemExit("refusing to overwrite %s: choose a new --out" % out)
    try:
        report = build(args)
    except BudgetReportError as exc:
        raise SystemExit(str(exc))
    out.mkdir(parents=True)
    (out / "budget-report.json").write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    (out / "budget-report.md").write_text(render(report), encoding="utf-8")
    print("%s, %d questions: %d later scorings at serving %d and diagnostic %d; accounting exact: %s"
          % (report["bed"], report["n"], report["scorings_reported"], report["serving"], report["diagnostic"], bool(report["accounting_exact"])))
    print("labels: %s; per bed: %s" % (", ".join("%s=%s" % kv for kv in report["seed_labels"].items()), report["bed_label"]))
    print("wrote", out / "budget-report.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
