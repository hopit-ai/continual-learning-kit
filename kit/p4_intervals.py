#!/usr/bin/env python3
"""The one interval method of package 4, as code: conservative finite-sample bounds from a multinomial region.

Implements docs/phase2/plan-v3-package4-preregistration-20261002.md section 3.1 and the contrasts of
docs/phase2/plan-v3-package4-supplement1-20261003.md section S4. It implements the text; it does not improve it.

    samples = q_retention(stage1, control, intervention, task="chemistry")
    bounds = contrast_bounds(samples, alpha=0.05)
    decide(bounds, "lower_at_least_zero")             # "pass" | "fail" | "inconclusive"

A CONTRAST is a mean over questions of a fixed linear combination of paired binary verdicts (strict correct = 1,
else 0) of several checkpoints on ONE task sample, possibly summed over two task samples (separate question sets),
plus a fixed constant.

THE METHOD (registration 3.1, steps 1 to 3):
    1. Per task sample: the full mathematically possible support z_1 < ... < z_m of the per-question value (every
       combination of 0/1 verdicts of the checkpoints with a non-zero coefficient), unobserved values included;
       k_j questions at z_j, n questions in all.
    2. eta = alpha / (2 M), M the total number of support cells across the task samples entering the contrast.
       l_j = 0 if k_j = 0 else BetaQuantile(eta; k_j, n - k_j + 1);
       u_j = 1 if k_j = n else BetaQuantile(1 - eta; k_j + 1, n - k_j).
    3. Lower bound of a sample's mean: start from p = l, give the remaining probability 1 - sum(l), within the
       capacities u - l, to the support values in INCREASING order; upper bound: DECREASING order. Two samples:
       separate probability vectors, contributions added. The constant is added afterwards.
No resampling. The finite-panel mean (`estimate`) is exact; the bounds assume i.i.d. question draws for fixed
checkpoints and are marginal item-sampling bounds (registration 3.1, last paragraphs).

FAILS CLOSED with IntervalInputError (a ValueError) on a missing checkpoint, differing question-id sets, a verdict
that is not 0/1, an empty sample, ids that differ from `expected_ids`, a duplicate id in rows, or two samples of one
contrast naming the same task.

Standard library only. Pure functions: no file reading, no CLI, no printing.
"""
from __future__ import annotations

import math
from fractions import Fraction
from functools import lru_cache

IDS_SHOWN = 5                               # question ids named in an error message
X_TOLERANCE = 1e-14                         # bisection stops when the bracket is this narrow
CF_EPS = 1e-15                              # continued-fraction convergence (relative change of one step)
CF_MAXIT = 100000
CF_TINY = 1e-300
KINDS = ("lower_at_least_zero", "non_inferiority", "lower_above_zero")


class IntervalInputError(ValueError):
    """An input to the interval engine is missing, mismatched or malformed; no bound is produced."""


# ---------------------------------------------------------------------------------------------------- Beta quantile

def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction of the incomplete beta function (modified Lentz), valid for x < (a + 1) / (a + b + 2)."""
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < CF_TINY:
        d = CF_TINY
    d = 1.0 / d
    h = d
    for m in range(1, CF_MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < CF_TINY:
            d = CF_TINY
        c = 1.0 + aa / c
        if abs(c) < CF_TINY:
            c = CF_TINY
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < CF_TINY:
            d = CF_TINY
        c = 1.0 + aa / c
        if abs(c) < CF_TINY:
            c = CF_TINY
        d = 1.0 / d
        step = d * c
        h *= step
        if abs(step - 1.0) < CF_EPS:
            return h
    raise ArithmeticError("incomplete beta continued fraction did not converge (a=%r, b=%r, x=%r)" % (a, b, x))


def beta_cdf(x: float, a: float, b: float) -> float:
    """I_x(a, b), the regularised incomplete beta function (the Beta(a, b) distribution function)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_front = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                 + a * math.log(x) + b * math.log1p(-x))
    front = math.exp(log_front)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def beta_quantile(q: float, a: float, b: float) -> float:
    """The q-quantile of Beta(a, b): the x in [0, 1] with I_x(a, b) = q.

    I_x(a, b) is computed as exp(lgamma terms + a ln x + b ln(1 - x)) times a continued fraction (modified Lentz,
    stopped at a relative step change below 1e-15), using the symmetry I_x(a, b) = 1 - I_{1-x}(b, a) on the slow
    side. x is then found by plain bisection on [0, 1] (I is increasing in x) until the bracket is narrower than
    1e-14, so x is within 5e-15 of the root of the computed I. For the parameters this package uses (a, b up to a
    few hundred), the computed I carries a relative error of the order of 1e-13 (lgamma), and the result agrees with
    scipy.stats.beta.ppf to better than 1e-9 (tested on a grid with n = 68 and n = 210 when scipy is present; the
    worst difference seen on that grid was 7e-15). Far in the upper tail (q within about 1e-4 of 1) I is computed as
    1 minus a small number, so x is only as precise as about 1e-16 divided by the density at x; at the registered
    allowances (1 - q >= 0.05 / 32) that is far below 1e-9.
    q = 0 gives 0 and q = 1 gives 1.
    """
    q, a, b = float(q), float(a), float(b)
    if not (a > 0.0 and b > 0.0) or math.isnan(q) or not 0.0 <= q <= 1.0:
        raise ValueError("beta_quantile needs 0 <= q <= 1 and a, b > 0 (got q=%r, a=%r, b=%r)" % (q, a, b))
    if q == 0.0:
        return 0.0
    if q == 1.0:
        return 1.0
    lo, hi = 0.0, 1.0
    while hi - lo > X_TOLERANCE:
        mid = 0.5 * (lo + hi)
        if mid <= lo or mid >= hi:                                          # no float strictly between: done
            break
        if beta_cdf(mid, a, b) < q:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


@lru_cache(maxsize=None)
def cell_bounds(k: int, n: int, eta: float) -> tuple:
    """(l_j, u_j) of registration 3.1 step 2 for a cell holding k of n questions (one-sided Clopper-Pearson at eta)."""
    if not (isinstance(k, int) and isinstance(n, int)) or n < 1 or not 0 <= k <= n:
        raise ValueError("cell_bounds needs integers 0 <= k <= n, n >= 1 (got k=%r, n=%r)" % (k, n))
    if not 0.0 < eta < 0.5:
        raise ValueError("cell_bounds needs 0 < eta < 0.5 (got %r)" % (eta,))
    lower = 0.0 if k == 0 else beta_quantile(eta, k, n - k + 1)
    upper = 1.0 if k == n else beta_quantile(1.0 - eta, k + 1, n - k)
    return lower, upper


# ---------------------------------------------------------------------------------------------- Coefficients, support

def _fraction(value, what: str = "coefficient") -> Fraction:
    """Exact Fraction from an int, a Fraction, a string such as "1/2" or "-0.5", or a float.

    A float is converted with Fraction(x).limit_denominator(1000): the registered coefficients (0, +-1/2, +-1) and
    constants (0.05, 0.025) come out exact, and a float such as 0.1 becomes 1/10 rather than its binary expansion.
    """
    if isinstance(value, bool):
        raise IntervalInputError("%s must be a number, not a bool (%r)" % (what, value))
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int):
        return Fraction(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise IntervalInputError("%s must be finite (%r)" % (what, value))
        return Fraction(value).limit_denominator(1000)
    if isinstance(value, str):
        try:
            return Fraction(value.strip())
        except (ValueError, ZeroDivisionError) as error:
            raise IntervalInputError("%s %r is not a number: %s" % (what, value, error)) from None
    raise IntervalInputError("%s must be an int, Fraction, float or string (%r)" % (what, value))


def support(coefficients) -> list:
    """Sorted distinct values of sum(c_i v_i) over every v in {0, 1}^m, exactly (registration 3.1 step 1).

    `coefficients` is a sequence of coefficients or a mapping whose values are the coefficients.
    """
    values = coefficients.values() if hasattr(coefficients, "values") else coefficients
    sums = {Fraction(0)}
    for raw in values:
        c = _fraction(raw)
        sums |= {s + c for s in sums}
    return sorted(sums)


# ------------------------------------------------------------------------------------------------------- Samples

def _show(ids) -> str:
    shown = sorted(ids, key=lambda i: (str(type(i)), str(i)))[:IDS_SHOWN]
    more = len(ids) - len(shown)
    return ", ".join(repr(i) for i in shown) + (" and %d more" % more if more > 0 else "")


def _verdict(value, task: str, checkpoint: str, question) -> int:
    """0 or 1 from an int 0/1 or a bool; anything else (1.0, "1", None) is refused."""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int) and value in (0, 1):
        return value
    raise IntervalInputError("task %r, checkpoint %r: verdict for question %r is %r, not 0/1"
                             % (task, checkpoint, question, value))


def verdicts_from_rows(rows, id_key: str = "id", verdict_key: str = "correct", task: str = "",
                       checkpoint: str = "") -> dict:
    """{question_id: 0/1} from a list of row dicts; a duplicate id, a missing key or a non-binary verdict raises."""
    out = {}
    duplicates = []
    for position, row in enumerate(rows):
        if not isinstance(row, dict) or id_key not in row or verdict_key not in row:
            raise IntervalInputError("task %r, checkpoint %r: row %d lacks %r or %r"
                                     % (task, checkpoint, position, id_key, verdict_key))
        qid = row[id_key]
        if qid in out:
            duplicates.append(qid)
            continue
        out[qid] = _verdict(row[verdict_key], task, checkpoint, qid)
    if duplicates:
        raise IntervalInputError("task %r, checkpoint %r: duplicate question ids %s"
                                 % (task, checkpoint, _show(set(duplicates))))
    return out


def _sample(task: str, coefficients: dict, verdicts: dict, expected_ids=None) -> dict:
    sample = {"task": task, "coefficients": coefficients, "verdicts": verdicts}
    if expected_ids is not None:
        sample["expected_ids"] = expected_ids
    return sample


def sample_counts(sample: dict, expected_ids=None) -> dict:
    """n, the complete support, the count at each support value and the exact mean of one task sample.

    `expected_ids` (or the sample's own "expected_ids" key) is the question-id set the sample must have exactly.
    Checkpoints with a zero coefficient are not read. Fails closed with IntervalInputError (see the module docstring).
    """
    if not isinstance(sample, dict):
        raise IntervalInputError("a sample must be a dict with task, coefficients and verdicts (%r)" % (sample,))
    task = sample.get("task")
    if not isinstance(task, str) or not task:
        raise IntervalInputError("sample has no task name")
    coefficients = sample.get("coefficients")
    verdicts = sample.get("verdicts")
    if not isinstance(coefficients, dict) or not isinstance(verdicts, dict):
        raise IntervalInputError("task %r: coefficients and verdicts must be dicts" % task)
    active = {}
    for name, raw in coefficients.items():
        c = _fraction(raw, "task %r, checkpoint %r coefficient" % (task, name))
        if c != 0:
            active[name] = c
    if not active:
        raise IntervalInputError("task %r: no checkpoint has a non-zero coefficient" % task)
    if expected_ids is None:
        expected_ids = sample.get("expected_ids")

    reference_name, reference_ids = None, None
    clean = {}
    for name in active:
        rows = verdicts.get(name)
        if not isinstance(rows, dict):
            raise IntervalInputError("task %r, checkpoint %r: no verdicts (questions none)" % (task, name))
        if not rows:
            raise IntervalInputError("task %r, checkpoint %r: empty sample, n = 0 (questions none)" % (task, name))
        ids = set(rows)
        if reference_ids is None:
            reference_name, reference_ids = name, ids
        elif ids != reference_ids:
            missing, extra = reference_ids - ids, ids - reference_ids
            raise IntervalInputError(
                "task %r, checkpoint %r: question ids differ from checkpoint %r (missing %s; extra %s)"
                % (task, name, reference_name, _show(missing) or "none", _show(extra) or "none"))
        clean[name] = {qid: _verdict(v, task, name, qid) for qid, v in rows.items()}
    if expected_ids is not None:
        expected = set(expected_ids)
        if expected != reference_ids:
            missing, extra = expected - reference_ids, reference_ids - expected
            raise IntervalInputError(
                "task %r, checkpoint %r: question ids differ from the expected set (missing %s; extra %s)"
                % (task, reference_name, _show(missing) or "none", _show(extra) or "none"))
    n = len(reference_ids)
    values = support(active)
    index = {z: j for j, z in enumerate(values)}
    counts = [0] * len(values)
    total = Fraction(0)
    for qid in reference_ids:
        z = sum((c * clean[name][qid] for name, c in active.items()), Fraction(0))
        counts[index[z]] += 1
        total += z
    return {"task": task, "n": n, "support": values, "counts": counts, "mean": total / n}


def _fill(values: list, lower: list, upper: list, order: list) -> float:
    """Registration 3.1 step 3: p = l, then the rest of the probability to `order` within the capacities u - l."""
    p = list(lower)
    remaining = 1.0 - math.fsum(lower)
    for j in order:
        if remaining <= 0.0:
            break
        add = min(upper[j] - lower[j], remaining)
        p[j] += add
        remaining -= add
    if remaining > 1e-9:
        raise ArithmeticError("the cell bounds cannot carry total probability 1 (left %r)" % remaining)
    return math.fsum(pj * float(z) for pj, z in zip(p, values))


def sample_bounds(sample: dict, eta: float, expected_ids=None) -> dict:
    """One task sample's cell bounds and its lower and upper contributions at a given eta (steps 2 and 3)."""
    counted = sample_counts(sample, expected_ids)
    n, values, counts = counted["n"], counted["support"], counted["counts"]
    cells = [cell_bounds(k, n, float(eta)) for k in counts]
    lower = [c[0] for c in cells]
    upper = [c[1] for c in cells]
    m = len(values)
    return {"task": counted["task"], "n": n,
            "support": [str(z) for z in values], "counts": counts, "mean": counted["mean"],
            "cell_lower": lower, "cell_upper": upper,
            "lower": _fill(values, lower, upper, list(range(m))),
            "upper": _fill(values, lower, upper, list(range(m - 1, -1, -1)))}


def contrast_bounds(samples, alpha=0.05, constant=0) -> dict:
    """Bounds of one contrast over one or more task samples (separate question sets), plus a fixed constant.

    eta = alpha / (2 M), M the total number of support cells across the samples; each sample's contribution is
    bounded with its own probability vector and the contributions are added; the constant is added afterwards.
    Returns estimate, lower, upper (floats), alpha, eta, cells (M), constant and the per-sample detail.
    """
    if isinstance(samples, dict):
        samples = [samples]
    samples = list(samples)
    if not samples:
        raise IntervalInputError("a contrast needs at least one task sample")
    alpha_f = float(_fraction(alpha, "alpha")) if not isinstance(alpha, float) else alpha
    if not 0.0 < alpha_f < 1.0:
        raise ValueError("alpha must lie in (0, 1) (got %r)" % (alpha,))
    const = _fraction(constant, "constant")
    tasks = [s.get("task") if isinstance(s, dict) else None for s in samples]
    seen = set()
    for task in tasks:
        if task in seen:
            raise IntervalInputError("task %r appears in two samples of one contrast: terms on one task must be "
                                     "formed per question in one sample" % (task,))
        seen.add(task)

    counted = [sample_counts(s) for s in samples]
    cells = sum(len(c["support"]) for c in counted)
    eta = alpha_f / (2 * cells)
    detail = [sample_bounds(s, eta) for s in samples]
    exact = sum((d["mean"] for d in detail), Fraction(0)) + const
    estimate = float(exact)
    lower = math.fsum(d["lower"] for d in detail) + float(const)
    upper = math.fsum(d["upper"] for d in detail) + float(const)
    if not (lower <= estimate + 1e-12 and estimate <= upper + 1e-12):
        raise ArithmeticError("bounds do not contain the estimate (%r, %r, %r)" % (lower, estimate, upper))
    for d in detail:
        d["mean"] = str(d["mean"])
    return {"estimate": estimate, "estimate_exact": str(exact), "lower": lower, "upper": upper,
            "alpha": alpha_f, "eta": eta, "cells": cells, "constant": str(const), "samples": detail}


# ---------------------------------------------------------------------------------------- The registered contrasts

def delta(first: dict, second: dict, task: str, expected_ids=None) -> list:
    """Mean of S_first - S_second on one task, per question (support {-1, 0, 1}).

    Registration 3.1 step 1, "The difference Δ of two checkpoints has support {−1, 0, 1}"; used for condition 5
    (Δ = S_intervention(B) − S_control(B)) and for supplement S4 contrasts 1 to 5 (D_ab = S_a(A) − S_b(A),
    D_bc = S_b(A) − S_c(A), D_sc = S_s(A) − S_c(A), G = S_a(A) − S_c(A), Δ_B = S_s(B) − S_c(B)) and the
    B-preservation contrasts S_a(B) − S_b(B) and S_b(B) − S_c(B).
    """
    return [_sample(task, {"first": Fraction(1), "second": Fraction(-1)},
                    {"first": first, "second": second}, expected_ids)]


def q_retention(stage1: dict, control: dict, intervention: dict, task: str, expected_ids=None) -> list:
    """Q = ½ F_control − F_intervention with F_x = S_stage1(A) − S_x(A), i.e. S_intervention − ½ S_stage1 − ½ S_control.

    Registration 3.2 condition 3, "Q = ½ F_control − F_intervention", and 3.1 step 1, "Q, written T − ½R − ½C for
    intervention, reference and control, has support {−1, −½, 0, ½, 1}". The two agree with no constant:
    ½(S1 − Sc) − (S1 − Si) = Si − ½S1 − ½Sc.
    """
    return [_sample(task, {"stage1": Fraction(-1, 2), "control": Fraction(-1, 2), "intervention": Fraction(1)},
                    {"stage1": stage1, "control": control, "intervention": intervention}, expected_ids)]


def half_gap_minibatch(a: dict, b: dict, c: dict, task: str, expected_ids=None) -> list:
    """D_ab − ½G = ½ S_a(A) − S_b(A) + ½ S_c(A), formed per question from the three verdicts.

    Supplement S4, attribution: "D_ab − ½G = ½S_a(A) − S_b(A) + ½S_c(A) for the minibatch step" (with D_ab = S_a −
    S_b and G = S_a − S_c; support {−1, −½, 0, ½, 1}).
    """
    return [_sample(task, {"a": Fraction(1, 2), "b": Fraction(-1), "c": Fraction(1, 2)},
                    {"a": a, "b": b, "c": c}, expected_ids)]


def half_gap_learning_rate(a: dict, b: dict, c: dict, task: str, expected_ids=None) -> list:
    """D_bc − ½G = −½ S_a(A) + S_b(A) − ½ S_c(A), formed per question from the three verdicts.

    Supplement S4, attribution: "D_bc − ½G = −½S_a(A) + S_b(A) − ½S_c(A) for the learning-rate step" (with D_bc =
    S_b − S_c and G = S_a − S_c; support {−1, −½, 0, ½, 1}).
    """
    return [_sample(task, {"a": Fraction(-1, 2), "b": Fraction(1), "c": Fraction(-1, 2)},
                    {"a": a, "b": b, "c": c}, expected_ids)]


# -------------------------------------------------------------------------------------------------------- Decisions

def decide(bounds: dict, kind: str, margin=0) -> str:
    """The three-valued decision of a registered condition: "pass", "fail" or "inconclusive".

        lower_at_least_zero  pass if lower >= 0;        fail if upper < 0;        registration 3.2 condition 3
        non_inferiority      pass if lower > -margin;   fail if upper < -margin;  registration 3.2 condition 5
                                                                                  (margin 0.025) and the S4
                                                                                  B-preservation contrasts
        lower_above_zero     pass if lower > 0;         fail if upper < 0;        supplement S4 labels
    Anything else is "inconclusive". `margin` is used only by non_inferiority.
    """
    if kind not in KINDS:
        raise ValueError("unknown decision kind %r (known: %s)" % (kind, ", ".join(KINDS)))
    lower, upper = float(bounds["lower"]), float(bounds["upper"])
    if math.isnan(lower) or math.isnan(upper) or lower > upper:
        raise ValueError("bounds must satisfy lower <= upper (got %r, %r)" % (lower, upper))
    if kind == "lower_at_least_zero":
        if lower >= 0.0:
            return "pass"
        return "fail" if upper < 0.0 else "inconclusive"
    if kind == "non_inferiority":
        m = float(_fraction(margin, "margin")) if not isinstance(margin, float) else margin
        if lower > -m:
            return "pass"
        return "fail" if upper < -m else "inconclusive"
    if lower > 0.0:
        return "pass"
    return "fail" if upper < 0.0 else "inconclusive"
