#!/usr/bin/env python3
"""Package 4A-0: fresh SEC filings -> FinQA-style questions, first slice (CPU only, no model, no paid call).

Synthetic questions over real dated 10-K/10-Q filings filed after 1 May 2025 (after Qwen3's training data),
written in exactly the item format kit/beds/finqa.py reads (pre_text, table, post_text, qa.question,
qa.exe_ans, qa.answer, qa.program), so the existing FinQA prompt, scorer and trainer-row code run unchanged.

Method, in one paragraph. For each month, filings are drawn from EDGAR's quarterly form index (10-K and
10-Q originals only; no 8-K, no amendments as question sources), in a fixed hash order. For each filing the
filing index page gives the form, filing date, acceptance time and period of report; FilingSummary.xml
names the income statement; the statement's R page (EDGAR's rendering of THAT accession's XBRL, so every
number is bound to the original accession, never to a later restatement as the frames API would be) gives
the table. The period column is checked against the period of report, the comparison column must be the
same-length period about a year earlier, units come from the statement heading, and only unambiguous,
positive, whole-number money lines are used. Two questions per filing are drawn from five templates
(percentage change, change, share of revenue, change in share of revenue in percentage points, average),
and the gold answer is computed by the program recorded with it.

    python scripts/p4_sec_slice.py build --sample                  # offline, on the hand-saved sample
    SEC_USER_AGENT="Org Name contact@org" python scripts/p4_sec_slice.py build   # the 120-question slice
    python scripts/p4_sec_slice.py dryrun --slice data/p4_sec/sample/out       # FinQA scorer, stand-in answers

SEC access: EDGAR requires a declared User-Agent with a contact and at most 10 requests per second. This
script reads the User-Agent ONLY from $SEC_USER_AGENT, refuses to fetch without it, keeps to 2 requests per
second, and caches every response under --cache (data/cache/ is gitignored). Nothing is fetched twice.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import html as htmllib
import importlib.util
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
import zlib
from collections import Counter, deque
from datetime import date, datetime, timedelta
from html.parser import HTMLParser
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SEC = "https://www.sec.gov/"
MIN_INTERVAL = 0.5                      # seconds between requests: 2 per second, a fifth of SEC's ceiling
CUTOFF = date(2025, 5, 1)               # filings on or after this date only (Qwen3's data predates it)
FORMS = ("10-K", "10-Q")
DEFAULT_MONTHS = ("2025-06", "2025-07", "2025-08")
SAMPLE_DIR = REPO / "data" / "p4_sec" / "sample"
MAX_CANDIDATES = 80                     # per month: bounds the fetches if many filings are unusable
SCHEMA = "p4-sec-finqa-slice.v1"


def sha256(data) -> str:
    return hashlib.sha256(data if isinstance(data, bytes) else data.encode("utf-8")).hexdigest()


def load_kit(name: str):
    spec = importlib.util.spec_from_file_location("p4sec_" + name.replace("/", "_"), REPO / "kit" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------------------------------------- SEC access
class NotCached(LookupError):
    pass


class SecSource:
    """Cache-first EDGAR reader. Paths are relative to https://www.sec.gov/ and mirror the cache layout.

    A cached file may be stored plain (`root/path`) or gzip-compressed (`root/path + ".gz"`, as written by
    scripts/sec_cache_compress.py or by a fetch with SEC_CACHE_GZIP=1). Both read as the same text, and
    `self.read[path]` is always the sha256 of the DECOMPRESSED bytes, so manifests do not depend on storage.
    Within a root the plain file wins when both exist. New fetches are written plain unless SEC_CACHE_GZIP=1."""

    def __init__(self, cache: Path, extra_roots=(), allow_fetch: bool = True):
        self.cache, self.roots, self.allow_fetch = Path(cache), [Path(cache), *map(Path, extra_roots)], allow_fetch
        self.requests, self.read, self._last = 0, {}, 0.0

    @staticmethod
    def read_cached(root: Path, path: str):
        """The decompressed bytes of `path` under `root` (plain first, then `.gz`), or None if neither is a file."""
        file = Path(root) / path
        if file.is_file():
            return file.read_bytes()
        packed = file.parent / (file.name + ".gz")
        if packed.is_file():
            return gzip.decompress(packed.read_bytes())
        return None

    def get(self, path: str) -> str:
        for root in self.roots:
            data = self.read_cached(root, path)
            if data is not None:
                self.read[path] = sha256(data)
                return data.decode("utf-8", errors="replace")
        if not self.allow_fetch:
            raise NotCached(path)
        agent = os.environ.get("SEC_USER_AGENT", "").strip()
        if not agent:
            raise SystemExit("SEC_USER_AGENT is not set: EDGAR requires a declared User-Agent with a contact "
                             "(e.g. 'Organisation Name contact@organisation'). The owner must supply it; "
                             "nothing was fetched. Missing: %s" % path)
        data = self._fetch(SEC + path, agent)
        file = self.cache / path
        file.parent.mkdir(parents=True, exist_ok=True)
        if os.environ.get("SEC_CACHE_GZIP", "").strip() == "1":
            file, stored = file.parent / (file.name + ".gz"), gzip.compress(data, compresslevel=6, mtime=0)
        else:
            stored = data
        tmp = file.with_suffix(file.suffix + ".part")
        tmp.write_bytes(stored)
        tmp.replace(file)
        self.read[path] = sha256(data)
        return data.decode("utf-8", errors="replace")

    def _fetch(self, url: str, agent: str) -> bytes:
        for attempt in range(4):
            wait = self._last + MIN_INTERVAL - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            self.requests += 1
            request = urllib.request.Request(url, headers={"User-Agent": agent, "Accept-Encoding": "gzip, deflate"})
            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    data, encoding = response.read(), response.headers.get("Content-Encoding", "")
                if encoding == "gzip":
                    data = gzip.decompress(data)
                elif encoding == "deflate":
                    data = zlib.decompress(data)
                return data
            except urllib.error.HTTPError as error:
                if error.code == 404:
                    raise NotCached(url) from error
                if error.code not in (403, 429, 500, 502, 503, 504) or attempt == 3:
                    raise
                time.sleep(10 * 2 ** attempt)           # SEC throttles by pausing the client; back off, never hammer
        raise RuntimeError("unreachable")


# ------------------------------------------------------------------------------------- selection
INDEX_LINE = re.compile(r"^(?P<form>\S+(?: \S+)*?)\s{2,}(?P<company>.+?)\s{2,}(?P<cik>\d+)\s+"
                        r"(?P<date>\d{4}-?\d{2}-?\d{2})\s+(?P<file>edgar/data/\S+)\s*$")


def parse_form_index(text: str) -> list:
    """EDGAR form.idx (full-index or daily-index) -> [{form, company, cik, date, accession}]."""
    rows = []
    for line in text.splitlines():
        match = INDEX_LINE.match(line)
        if not match:
            continue
        raw = match.group("date").replace("-", "")
        rows.append({"form": match.group("form"), "company": match.group("company").strip(), "cik": str(int(match.group("cik"))),
                     "date": "%s-%s-%s" % (raw[:4], raw[4:6], raw[6:]),
                     "accession": match.group("file").rsplit("/", 1)[1].removesuffix(".txt")})
    return rows


def candidates_for_month(source: SecSource, month: str, seed: str) -> tuple:
    """Original 10-K/10-Q filings with a filing date in `month`, one per company, in a fixed hash order;
    plus every amendment (10-K/A, 10-Q/A) by the same companies in the same index, recorded as provenance."""
    year, mon = map(int, month.split("-"))
    rows = parse_form_index(source.get("Archives/edgar/full-index/%d/QTR%d/form.idx" % (year, (mon - 1) // 3 + 1)))
    in_month = [r for r in rows if r["date"].startswith(month)]
    originals = [r for r in in_month if r["form"] in FORMS]
    amendments = {}
    for r in rows:
        if r["form"] in tuple(f + "/A" for f in FORMS):
            amendments.setdefault(r["cik"], []).append({"form": r["form"], "accession": r["accession"], "date": r["date"]})
    originals.sort(key=lambda r: sha256("%s:%s" % (seed, r["accession"])))
    # One filing can be listed under several CIKs (co-registrants: a REIT and its operating partnership, an
    # airline group and its airline). It is one document: keep its first listing, and record every filer on it.
    filers = {}
    for r in originals:
        filers.setdefault(r["accession"], []).append(r["cik"])
    seen, ordered = set(), []
    for r in originals:
        if r["cik"] in seen or r["accession"] in seen:
            continue
        seen.update([r["accession"], *filers[r["accession"]]])
        ordered.append({**r, "co_filers": sorted(set(filers[r["accession"]]) - {r["cik"]}, key=int)})
    return ordered, amendments


# ------------------------------------------------------------------------------------- one filing
def folder(cik: str, accession: str) -> str:
    return "Archives/edgar/data/%s/%s/" % (int(cik), accession.replace("-", ""))


def parse_filing_index(html: str) -> dict:
    form = re.search(r'id="formName">\s*<strong>\s*Form\s+([^<]+?)\s*</strong>', html)
    pairs = dict((k.strip(), v.strip()) for k, v in re.findall(r'class="infoHead">([^<]+)</div>\s*<div class="info">([^<]*)</div>', html))
    company = re.search(r'class="companyName">\s*([^<]+?)\s*\((?:Filer|Filed by|Subject)\)', html)
    if not form or "Filing Date" not in pairs:
        raise ValueError("filing index page not recognised")
    return {"form": form.group(1), "filing_date": pairs["Filing Date"], "accepted": pairs.get("Accepted"),
            "period_of_report": pairs.get("Period of Report"), "company": htmllib.unescape(company.group(1).strip()) if company else None}


def statements(summary_xml: str) -> list:
    out = []
    for block in re.findall(r"<Report\b[^>]*>(.*?)</Report>", summary_xml, re.S):
        get = lambda tag: (re.search(r"<%s>(.*?)</%s>" % (tag, tag), block, re.S) or [None, ""])[1].strip()   # noqa: E731
        if get("MenuCategory") == "Statements" and get("HtmlFileName"):
            out.append({"file": get("HtmlFileName"), "name": get("ShortName")})
    return out


def income_statement(reports: list):
    """The income statement's R page: a pure statement of operations/income/loss first; then a combined one with
    comprehensive income; then a single statement of comprehensive income (loss), which many filers use as their
    only income statement (one-statement presentation: net income, then other comprehensive income)."""
    usable = [r for r in reports if re.search(r"OPERATIONS|INCOME|EARNINGS|LOSS|PROFIT", r["name"], re.I)
              and not re.search(r"PARENTHETICAL|EQUITY|DEFICIT|CASH FLOW|BALANCE|CHANGES IN|SEGMENT|FINANCIAL CONDITION", r["name"], re.I)]
    pure = [r for r in usable if not re.search(r"COMPREHENSIVE", r["name"], re.I)]
    combined = [r for r in usable if re.search(r"(OPERATIONS|INCOME|EARNINGS|LOSS)\b.*\bAND\b.*COMPREHENSIVE", r["name"], re.I)]
    single = [r for r in usable if re.search(r"COMPREHENSIVE", r["name"], re.I)] if len([r for r in usable if re.search(r"COMPREHENSIVE", r["name"], re.I)]) == 1 else []
    return (pure or combined or single or [None])[0]


class _ReportTable(HTMLParser):
    """Rows of the first <table class="report"> in an R page: [{'class', 'cells': [{tag, class, colspan, rowspan, text}]}]."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.rows, self.depth, self.done, self.row, self.cell = [], 0, False, None, None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "table":
            if self.depth:
                self.depth += 1
            elif not self.done and "report" in (attrs.get("class") or "").split():
                self.depth = 1
            return
        if self.depth != 1:
            return
        if tag == "tr":
            self.row = {"class": (attrs.get("class") or "").strip(), "cells": []}
        elif tag in ("td", "th") and self.row is not None:
            self.cell = {"tag": tag, "class": (attrs.get("class") or "").strip(), "colspan": int(attrs.get("colspan") or 1),
                         "rowspan": int(attrs.get("rowspan") or 1), "text": []}
        elif tag == "br" and self.cell is not None:
            self.cell["text"].append("\n")

    def handle_endtag(self, tag):
        if not self.depth:
            return
        if tag == "table":
            self.depth -= 1
            self.done = self.depth == 0
        elif self.depth == 1 and tag in ("td", "th") and self.cell is not None:
            text = "".join(self.cell["text"]).replace("\xa0", " ")
            self.cell["text"] = " | ".join(" ".join(part.split()) for part in text.split("\n") if part.strip())
            self.row["cells"].append(self.cell)
            self.cell = None
        elif self.depth == 1 and tag == "tr" and self.row is not None:
            self.rows.append(self.row)
            self.row = None

    def handle_data(self, data):
        if self.cell is not None:
            self.cell["text"].append(data)


VALUE = re.compile(r"^\$?\s*(\()?\s*\$?\s*([\d,]+(?:\.\d+)?)\s*(\))?$")
DURATION = re.compile(r"(\d+)\s+Months?\s+Ended", re.I)
SCALE = re.compile(r"\$ in (Thousands|Millions|Billions)", re.I)


def parse_value(text: str):
    """'$ 94,036' -> 94036.0 ; '(171)' -> -171.0 ; '' -> None. Footnote markers like '[1]' are dropped first."""
    text = re.sub(r"\[\d+\]", "", text or "").strip()
    match = VALUE.match(text)
    if not match:
        return None
    value = float(match.group(2).replace(",", ""))
    return -value if match.group(1) and match.group(3) else value


def parse_date(text: str):
    text = text.strip().replace("Sept.", "Sep.")
    for fmt in ("%b. %d, %Y", "%b %d, %Y", "%B %d, %Y"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            pass
    return None


def parse_statement(html: str, instants: bool = False) -> dict:
    """A statement's R page. Duration statements (income, cash flow) have two header rows: "3 Months Ended" groups
    over dates. With `instants=True` a one-row header of dates is also read (a balance sheet: each column is a point
    in time, recorded as months = 0)."""
    parser = _ReportTable()
    parser.feed(html)
    rows = parser.rows
    if len(rows) < 3 or not rows[0]["cells"] or rows[0]["cells"][0]["tag"] != "th":
        raise ValueError("no report table")
    head = rows[0]["cells"]
    title, label_cols = head[0]["text"], head[0]["colspan"]
    if head[0]["rowspan"] != 2:
        if not instants:
            raise ValueError("single header row (no duration groups)")
        columns = [{"months": 0, "date": parse_date(c["text"]), "header": c["text"]} for c in head[1:] for _ in range(c["colspan"])]
        if not columns or any(c["date"] is None for c in columns):
            raise ValueError("instant header is not a row of dates")
        first = 1
    else:
        durations = []
        for cell in head[1:]:
            durations += [cell["text"]] * cell["colspan"]
        dates = [cell["text"] for cell in rows[1]["cells"]]
        if len(dates) != len(durations):
            raise ValueError("header columns do not line up")
        columns = []
        for duration, raw_date in zip(durations, dates):
            months = DURATION.search(duration)
            columns.append({"months": int(months.group(1)) if months else None, "date": parse_date(raw_date), "header": "%s %s" % (duration, raw_date)})
        first = 2
    # The main statement, then EDGAR's dimension sections: an 'rh' row names a member (e.g. "Products") and the
    # rows under it repeat lines for that member. Members are shown in the excerpt as "Products - Net sales", the
    # breakdown the filing document prints inside the statement; questions are asked about main lines only.
    body, segments, footnotes, member, section = [], [], [], None, ""
    for row in rows[first:]:
        if row["class"] == "rh":
            member = row["cells"][0]["text"] if row["cells"] else ""
            continue
        if row["class"] == "outerFootnote" or row["class"] not in ("re", "ro", "reu", "rou") or len(row["cells"]) < len(columns) + 1:
            continue
        cells = [c["text"] for c in row["cells"][-len(columns):]]
        marks = " ".join(c["text"] for c in row["cells"][1:-len(columns)] if c["text"])      # a footnote column, e.g. '[1]'
        line = {"label": row["cells"][0]["text"], "shown": (row["cells"][0]["text"] + " " + marks).strip(),
                "cells": cells, "values": [parse_value(c) for c in cells], "section": section}
        if all(v is None for v in line["values"]):            # a heading row ("Earnings per share:") opens a section
            section = line["label"]
        if member is None:
            body.append(line)
        elif any(v is not None for v in line["values"]):
            segments.append({**line, "shown": "%s - %s" % (member, line["shown"])})
    for row in rows[first:]:                         # footnotes sit after the segment rows too
        if row["class"] == "outerFootnote" and len(row["cells"]) >= 2:
            note = "%s %s" % (row["cells"][0]["text"], row["cells"][1]["text"])
            if note not in footnotes:
                footnotes.append(note)
    scale = SCALE.search(title)
    if "USD" not in title:
        raise ValueError("not in US dollars: %s" % title)
    unit = {"thousands": "thousands of US dollars", "millions": "millions of US dollars", "billions": "billions of US dollars"}[scale.group(1).lower()] if scale else "US dollars"
    shares = re.search(r"shares in (Thousands|Millions|Billions)", title, re.I)
    return {"title": title, "statement": title.split(" - ")[0].strip(), "unit": unit, "shares_scale": shares.group(1).lower() if shares else None,
            "label_cols": label_cols, "columns": columns, "body": body, "segments": segments, "footnotes": footnotes}


# ------------------------------------------------------------------------------------- questions
SKIP_LABEL = re.compile(r"per share|per unit|per common|in shares|\bshares?\b|weighted|\beps\b|\[abstract\]|dividends? declared", re.I)
SKIP_SECTION = re.compile(r"\bper\b.*\b(share|unit)|\bshares?\b|weighted|\beps\b", re.I)    # a heading over per-share or share lines
REVENUE = re.compile(r"^(total\s+)?(net\s+)?(operating\s+)?(revenues?|sales)(,\s*net)?$", re.I)
# A percent answer below 1 in magnitude is not asked: given to one decimal place, a correct "0.4" misses 0.44 by
# more than the FinQA scorer's 1 percent tolerance, and its rounding rule only applies at magnitude 1 or more.
SMALLEST_PERCENT = 1.0
NUMBER_WORD = {3: "three", 6: "six", 9: "nine", 12: "twelve"}


def period_phrase(column: dict, form: str) -> str:
    when = column["date"].strftime("%B %-d, %Y")
    if column["months"] == 12 and form.startswith("10-K"):
        return "the fiscal year ended %s" % when
    return "the %s months ended %s" % (NUMBER_WORD.get(column["months"], str(column["months"])), when)


def comparison(statement: dict, period_of_report: date):
    """(group columns, current index, prior index) for the shortest duration that has the reported period and
    the same-length period about a year earlier; None (with a reason) otherwise. This is the period check."""
    columns = statement["columns"]
    months = sorted({c["months"] for c in columns if c["months"]})
    for m in months:
        group = [i for i, c in enumerate(columns) if c["months"] == m and c["date"]]
        current = [i for i in group if columns[i]["date"] == period_of_report]
        if not current:
            continue
        cur = current[0]
        prior = [i for i in group if 350 <= (columns[cur]["date"] - columns[i]["date"]).days <= 380]
        if prior:
            return group, cur, prior[0]
    return None


def eligible_lines(statement: dict, cols: list) -> list:
    """Money lines that are unambiguous (label unique among valued lines) and positive in `cols`. Per-share and
    share lines are excluded by their label or by their section heading ("Earnings per share:" over "Basic"); a
    decimal value is accepted only in a scaled statement ($ in thousands/millions/billions), since in an unscaled
    one it is a per-share amount."""
    valued = [r for r in statement["body"] if any(v is not None for v in r["values"])]
    counts = {}
    for r in valued:
        counts[r["label"].casefold()] = counts.get(r["label"].casefold(), 0) + 1
    out = []
    for r in valued:
        label = r["label"]
        if counts[label.casefold()] > 1 or label.endswith(":") or SKIP_LABEL.search(label) or SKIP_SECTION.search(r["section"]) \
                or re.match(r"^(basic|diluted)\b", label, re.I):
            continue
        scaled = statement["unit"] != "US dollars"
        if any(r["values"][i] is None or r["values"][i] <= 0 or ("." in r["cells"][i] and not scaled) for i in cols):
            continue
        out.append(r)
    return out


def decimals(row: dict, *cols) -> int:
    """How many decimal places the table shows for these cells ('775.1' -> 1)."""
    return max((len(row["cells"][i].split(".", 1)[1].rstrip(")").strip()) if "." in row["cells"][i] else 0) for i in cols)


def fmt(value: float) -> str:
    return ("%d" % value) if float(value).is_integer() else repr(value)


def make_questions(statement: dict, meta: dict, rng: random.Random, n: int) -> tuple:
    """Up to `n` (template, question, exe_ans, answer, program, operands) for one filing, or ([], reason)."""
    found = comparison(statement, meta["period"])
    if not found:
        return [], "no column for the period of report with a same-length prior-year column"
    group, cur, pri = found
    cols = statement["columns"]
    lines = eligible_lines(statement, [cur, pri])
    if len(lines) < 2:
        return [], "fewer than two unambiguous positive money lines"
    revenue = [r for r in lines if REVENUE.match(r["label"].strip())]
    revenue = sorted(revenue, key=lambda r: not r["label"].lower().startswith("total"))[:1]
    full = [r for r in eligible_lines(statement, group)] if len(group) >= 3 else []
    p_cur, p_pri = period_phrase(cols[cur], meta["form"]), period_phrase(cols[pri], meta["form"])
    unit = statement["unit"]
    q = lambda label: '"%s"' % label                                                           # noqa: E731
    op = lambda r, i: {"label": r["label"], "period": cols[i]["header"], "cell": r["cells"][i], "value": r["values"][i]}   # noqa: E731

    def pct_change(r):
        c, p = r["values"][cur], r["values"][pri]
        if c == p:
            return None
        v = (c - p) / p
        if abs(100 * v) < SMALLEST_PERCENT:
            return None
        return ("By what percentage did %s change from %s to %s? Answer in percent (negative for a decrease), rounded to one decimal place."
                % (q(r["label"]), p_pri, p_cur), v, "%.1f%%" % (100 * v), "subtract(%s, %s), divide(#0, %s)" % (fmt(c), fmt(p), fmt(p)), [op(r, cur), op(r, pri)])

    def change(r):
        c, p = r["values"][cur], r["values"][pri]
        if c == p:
            return None
        return ("By how much did %s change from %s to %s, in %s? Answer as a number (negative for a decrease)."
                % (q(r["label"]), p_pri, p_cur, unit), c - p, "{:,.{d}f}".format(c - p, d=decimals(r, cur, pri)), "subtract(%s, %s)" % (fmt(c), fmt(p)), [op(r, cur), op(r, pri)])

    def share(r):
        if not revenue or r is revenue[0] or r["values"][cur] >= revenue[0]["values"][cur]:
            return None
        a, b = r["values"][cur], revenue[0]["values"][cur]
        if 100 * a / b < SMALLEST_PERCENT:
            return None
        return ("What was %s as a percentage of %s for %s? Answer in percent, rounded to one decimal place."
                % (q(r["label"]), q(revenue[0]["label"]), p_cur), a / b, "%.1f%%" % (100 * a / b), "divide(%s, %s)" % (fmt(a), fmt(b)), [op(r, cur), op(revenue[0], cur)])

    def share_pp(r):
        if not revenue or r is revenue[0] or r["values"][cur] >= revenue[0]["values"][cur]:
            return None
        ac, bc, ap, bp = r["values"][cur], revenue[0]["values"][cur], r["values"][pri], revenue[0]["values"][pri]
        v = ac / bc - ap / bp
        if abs(100 * v) < SMALLEST_PERCENT:
            return None
        return ("By how many percentage points did %s as a percentage of %s change from %s to %s? Answer in percentage points "
                "(negative for a decrease), rounded to one decimal place." % (q(r["label"]), q(revenue[0]["label"]), p_pri, p_cur),
                v, "%.1f" % (100 * v), "divide(%s, %s), divide(%s, %s), subtract(#0, #1)" % (fmt(ac), fmt(bc), fmt(ap), fmt(bp)),
                [op(r, cur), op(revenue[0], cur), op(r, pri), op(revenue[0], pri)])

    def average(r):
        if r not in full:
            return None
        idx = sorted(group, key=lambda i: cols[i]["date"], reverse=True)
        values = [r["values"][i] for i in idx]
        v = sum(values) / len(values)
        names = ", ".join(period_phrase(cols[i], meta["form"]) for i in idx[:-1]) + " and " + period_phrase(cols[idx[-1]], meta["form"])
        prog = "add(%s, %s)" % (fmt(values[0]), fmt(values[1]))
        for k, value in enumerate(values[2:]):
            prog += ", add(#%d, %s)" % (k, fmt(value))
        prog += ", divide(#%d, const_%d)" % (len(values) - 2, len(values))
        return ("What was the average annual %s over %s, in %s? Round to one decimal place." % (q(r["label"]), names, unit),
                v, "{:,.1f}".format(v), prog, [op(r, i) for i in idx])

    templates = {"pct_change": pct_change, "change": change, "share_of_revenue": share, "share_pp_change": share_pp, "average": average}
    order = list(templates)
    rng.shuffle(order)
    out, used = [], set()
    for name in order:
        pool = [r for r in lines if r["label"] not in used]
        rng.shuffle(pool)
        for r in pool:
            made = templates[name](r)
            if made:
                out.append((name, *made))
                used.add(r["label"])
                break
        if len(out) == n:
            break
    if len(out) < n:
        return [], "fewer than %d distinct questions could be made" % n
    return out, None


def shown_rows(statement: dict) -> list:
    """The excerpt's rows: XBRL '[Abstract]' rows and an empty heading directly above a line of the same name
    (EDGAR's "Revenues" over "Revenues") are dropped; everything else is shown as rendered."""
    body, out = statement["body"], []
    key = lambda r: r["label"].rstrip(":").strip().casefold()                                 # noqa: E731
    for k, r in enumerate(body):
        empty = all(v is None for v in r["values"])
        if empty and ("[abstract]" in r["label"].lower() or (k + 1 < len(body) and key(body[k + 1]) == key(r))):
            continue
        out.append(r)
    return out + statement["segments"]


def build_filing(source: SecSource, cik: str, accession: str, n: int) -> tuple:
    """(items, None) or ([], exclusion reason) for one filing."""
    base = folder(cik, accession)
    try:
        meta = parse_filing_index(source.get(base + accession + "-index.htm"))
    except (NotCached, ValueError) as error:
        return [], "filing index unreadable: %s" % error, {}
    record = {"cik": str(int(cik)), "accession": accession, **meta}
    if meta["form"] not in FORMS:
        return [], "form %s is not an original 10-K/10-Q" % meta["form"], record
    filed = date.fromisoformat(meta["filing_date"])
    if filed < CUTOFF:
        return [], "filed before %s" % CUTOFF, record
    try:
        report = income_statement(statements(source.get(base + "FilingSummary.xml")))
    except NotCached:
        return [], "no FilingSummary.xml (no XBRL rendering)", record
    if not report:
        return [], "no income statement among the XBRL statements", record
    try:
        statement = parse_statement(source.get(base + report["file"]))
    except (NotCached, ValueError) as error:
        return [], "income statement unreadable: %s" % error, record
    if not meta.get("period_of_report"):
        return [], "no period of report on the index page", record
    meta["period"] = date.fromisoformat(meta["period_of_report"])
    rng = random.Random(sha256("questions:" + accession))
    questions, reason = make_questions(statement, meta, rng, n)
    if reason:
        return [], reason, record
    table = [[""] + [c["header"] for c in statement["columns"]]] + [[r["shown"]] + r["cells"] for r in shown_rows(statement)]
    company = meta["company"] or "The company"
    shares = "; share counts in %s" % statement["shares_scale"] if statement["shares_scale"] else ""
    pre = ["%s (CIK %s) filed this Form %s with the SEC on %s (accession %s); the period of report ends %s."
           % (company, int(cik), meta["form"], meta["filing_date"], accession, meta["period_of_report"]),
           "The table below is the %s from that filing." % statement["statement"],
           "Amounts are in %s, except per-share amounts%s." % (statement["unit"], shares)]
    items = []
    for k, (template, question, value, answer, program, operands) in enumerate(questions, 1):
        items.append({"id": "sec-%s-q%d" % (accession, k), "filename": "%s%s" % (base, report["file"]),
                      "pre_text": pre, "table": table, "post_text": statement["footnotes"],
                      "qa": {"question": question, "answer": answer, "exe_ans": round(value, 6), "program": program},
                      "sec": {"cik": str(int(cik)), "company": company, "form": meta["form"], "accession": accession,
                              "filing_date": meta["filing_date"], "accepted": meta["accepted"], "period_of_report": meta["period_of_report"],
                              "table_location": {"r_page": base + report["file"], "statement": report["name"]},
                              "template": template, "unit": statement["unit"], "operands": operands,
                              "rounding": "one decimal place" if template != "change" else "exact (table units)"}})
    return items, None, record


# ------------------------------------------------------------------------------------- slice v2: harder questions
# v1's frozen readout (Qwen3-8B 36/40, 34/40, 35/40) showed single-step lookups on the income statement are too easy
# for an updating-versus-memory study. v2 keeps every rule of v1 (original accession, filing-month batches, the same
# candidate order) and asks multi-step questions: composed income-statement questions, and balance-sheet and
# cash-flow questions from the same filing's XBRL rendering, each on its own statement with its own period logic
# (a balance sheet column is an instant, "as of"; a cash-flow column is a year-to-date duration).
V2_SCHEMA = "p4-sec-finqa-slice.v2"
V2_USAGE = __import__("collections").Counter()          # template counts so far in this build: keeps the mix balanced
OPERATING = re.compile(r"^(total\s+)?(operating income|income from operations|operating profit|income \(loss\) from operations|operating income \(loss\))$", re.I)
TCA = re.compile(r"^total current assets$", re.I)
TCL = re.compile(r"^total current liabilities$", re.I)
TOTAL_LIABILITIES = re.compile(r"^total liabilities$", re.I)
PARENT_EQUITY = re.compile(r"^total (stockholders|shareholders)['’]? ?equity( \(deficit\))?(,? attributable to .*)?$", re.I)
TOTAL_EQUITY = re.compile(r"^total equity( \(deficit\))?$", re.I)
# a total line ("Net cash provided by ...", "Cash used in ..."), never a section heading ("CASH FLOWS FROM ...")
OCF = re.compile(r"^(net cash|cash (provided|used|generated))\b.*\boperating activities$", re.I)
ICF = re.compile(r"^(net cash|cash (provided|used|generated))\b.*\binvesting activities$", re.I)
FINCF = re.compile(r"^(net cash|cash (provided|used|generated))\b.*\bfinancing activities$", re.I)
CAPEX = re.compile(r"^(capital expenditures?|((payments? for( the)?|cash paid for) )?(purchases?|acquisitions?|additions?)( of| to)? (property|premises|plant|fixed assets|capital assets))", re.I)
NOT_A_COST = re.compile(r"\b(total|income|profit|margin|earnings|loss|losses|gains?)\b|^(net )?(revenues?|sales)$", re.I)   # subtotals and results, not costs
CF_NET_INCOME = re.compile(r"^net (income|earnings)( \(loss\))?$", re.I)


def one_line(statement: dict, pattern, cols: list):
    """The single main-body line whose label matches `pattern` and which has a value in every column of `cols`."""
    found = [r for r in statement["body"] if pattern.match(r["label"].strip()) and all(r["values"][i] is not None for i in cols)]
    return found[0] if len(found) == 1 else None


def statement_of(reports: list, kind: str):
    if kind == "balance":
        pick = [r for r in reports if re.search(r"BALANCE SHEET|FINANCIAL CONDITION|FINANCIAL POSITION|STATEMENTS? OF CONDITION", r["name"], re.I)]
    else:
        pick = [r for r in reports if re.search(r"CASH FLOW", r["name"], re.I)]
    pick = [r for r in pick if not re.search(r"PARENTHETICAL|SUPPLEMENT", r["name"], re.I)]
    return pick[0] if pick else None


def instant_pair(statement: dict, period_of_report: date):
    cols = statement["columns"]
    cur = [i for i, c in enumerate(cols) if c["months"] == 0 and c["date"] == period_of_report]
    if not cur:
        return None
    earlier = [i for i, c in enumerate(cols) if c["months"] == 0 and c["date"] < period_of_report]
    return (cur[0], max(earlier, key=lambda i: cols[i]["date"])) if earlier else None


def as_of(column: dict) -> str:
    return "as of " + column["date"].strftime("%B %-d, %Y")


def v2_questions(kind: str, statement: dict, meta: dict, rng: random.Random) -> list:
    """Every v2 question this statement supports: [(template, question, value, answer, program, operands)]."""
    cols, unit, out = statement["columns"], statement["unit"], []
    q = lambda label: '"%s"' % label                                                                    # noqa: E731
    op = lambda r, i: {"label": r["label"], "period": cols[i]["header"], "cell": r["cells"][i], "value": r["values"][i]}   # noqa: E731
    dec = lambda pairs: max(decimals(r, i) for r, i in pairs)                                           # noqa: E731
    amount = lambda v, d: "{:,.{d}f}".format(v, d=d)                                                   # noqa: E731
    pct_ok = lambda v: abs(100 * v) >= SMALLEST_PERCENT                                                # noqa: E731
    if kind == "income":
        found = comparison(statement, meta["period"])
        if not found:
            return []
        _, cur, pri = found
        p_cur, p_pri = period_phrase(cols[cur], meta["form"]), period_phrase(cols[pri], meta["form"])
        lines = eligible_lines(statement, [cur, pri])
        revenue = sorted([r for r in lines if REVENUE.match(r["label"].strip())], key=lambda r: not r["label"].lower().startswith("total"))[:1]
        rev = revenue[0] if revenue else None
        operating = one_line(statement, OPERATING, [cur, pri])
        if rev and operating and operating is not rev:
            dr = rev["values"][cur] - rev["values"][pri]
            do = operating["values"][cur] - operating["values"][pri]
            if abs(dr) >= 0.02 * rev["values"][pri] and pct_ok(do / dr) and abs(do / dr) <= 10:
                out.append(("incremental_margin",
                            "What was the change in %s divided by the change in %s, from %s to %s? Answer in percent (negative if the two "
                            "changes have opposite signs), rounded to one decimal place." % (q(operating["label"]), q(rev["label"]), p_pri, p_cur),
                            do / dr, "%.1f%%" % (100 * do / dr),
                            "subtract(%s, %s), subtract(%s, %s), divide(#0, #1)" % (fmt(operating["values"][cur]), fmt(operating["values"][pri]), fmt(rev["values"][cur]), fmt(rev["values"][pri])),
                            [op(operating, cur), op(operating, pri), op(rev, cur), op(rev, pri)]))
        others = [r for r in lines if r is not rev]
        rng.shuffle(others)
        if rev:
            for r in others:                                                     # margin change in percentage points
                if r["values"][cur] < rev["values"][cur]:
                    v = r["values"][cur] / rev["values"][cur] - r["values"][pri] / rev["values"][pri]
                    if pct_ok(v):
                        out.append(("margin_pp_change",
                                    "By how many percentage points did %s as a percentage of %s change from %s to %s? Answer in percentage points "
                                    "(negative for a decrease), rounded to one decimal place." % (q(r["label"]), q(rev["label"]), p_pri, p_cur),
                                    v, "%.1f" % (100 * v), "divide(%s, %s), divide(%s, %s), subtract(#0, #1)" % (fmt(r["values"][cur]), fmt(rev["values"][cur]), fmt(r["values"][pri]), fmt(rev["values"][pri])),
                                    [op(r, cur), op(rev, cur), op(r, pri), op(rev, pri)]))
                        break
            # costs: lines below the revenue line and above the first result line (operating income, income before tax)
            body = [r for r in statement["body"] if any(v is not None for v in r["values"])]
            start = body.index(rev) + 1
            stop = next((k for k in range(start, len(body)) if NOT_A_COST.search(body[k]["label"])), len(body))
            window = body[start:stop]
            costs = [r for r in others if r in window and not NOT_A_COST.search(r["label"])]
            for a_, b_ in ((a, b) for a in costs for b in costs if a is not b):                # two cost lines combined, share of revenue
                v = (a_["values"][cur] + b_["values"][cur]) / rev["values"][cur]
                if pct_ok(v) and v < 1:
                    out.append(("combined_share",
                                "What were %s and %s combined, as a percentage of %s, for %s? Answer in percent, rounded to one decimal place."
                                % (q(a_["label"]), q(b_["label"]), q(rev["label"]), p_cur),
                                v, "%.1f%%" % (100 * v), "add(%s, %s), divide(#0, %s)" % (fmt(a_["values"][cur]), fmt(b_["values"][cur]), fmt(rev["values"][cur])),
                                [op(a_, cur), op(b_, cur), op(rev, cur)]))
                    break
        for a_, b_ in ((a, rev) for a in others if rev):                                         # growth of a line relative to revenue
            v = (a_["values"][cur] / a_["values"][pri] - 1) - (b_["values"][cur] / b_["values"][pri] - 1)
            if pct_ok(v) and abs(v) <= 5:
                out.append(("relative_growth",
                            "By how many percentage points did the growth rate of %s exceed the growth rate of %s from %s to %s? (Growth rate = "
                            "change divided by the earlier value.) Answer in percentage points (negative if %s grew more slowly), rounded to one "
                            "decimal place." % (q(a_["label"]), q(b_["label"]), p_pri, p_cur, q(a_["label"])),
                            v, "%.1f" % (100 * v), "divide(%s, %s), divide(%s, %s), subtract(#0, #1)" % (fmt(a_["values"][cur]), fmt(a_["values"][pri]), fmt(b_["values"][cur]), fmt(b_["values"][pri])),
                            [op(a_, cur), op(a_, pri), op(b_, cur), op(b_, pri)]))
                break
    elif kind == "balance":
        pair = instant_pair(statement, meta["period"])
        if not pair:
            return []
        cur, pri = pair
        d_cur, d_pri = as_of(cols[cur]), as_of(cols[pri])
        tca, tcl = one_line(statement, TCA, [cur, pri]), one_line(statement, TCL, [cur, pri])
        if tca and tcl and tcl["values"][cur] > 0 and tcl["values"][pri] > 0:
            d = dec([(tca, cur), (tcl, cur), (tca, pri), (tcl, pri)])
            wc_cur, wc_pri = tca["values"][cur] - tcl["values"][cur], tca["values"][pri] - tcl["values"][pri]
            if wc_cur != 0:
                out.append(("working_capital",
                            "What was working capital (%s minus %s) %s, in %s? Answer as a number (negative if current liabilities exceed current assets)."
                            % (q(tca["label"]), q(tcl["label"]), d_cur, unit), wc_cur, amount(wc_cur, d),
                            "subtract(%s, %s)" % (fmt(tca["values"][cur]), fmt(tcl["values"][cur])), [op(tca, cur), op(tcl, cur)]))
            if wc_cur != wc_pri:
                out.append(("working_capital_change",
                            "By how much did working capital (%s minus %s) change from %s to %s, in %s? Answer as a number (negative for a decrease)."
                            % (q(tca["label"]), q(tcl["label"]), d_pri.replace("as of ", ""), d_cur.replace("as of ", ""), unit),
                            wc_cur - wc_pri, amount(wc_cur - wc_pri, d),
                            "subtract(%s, %s), subtract(%s, %s), subtract(#0, #1)" % (fmt(tca["values"][cur]), fmt(tcl["values"][cur]), fmt(tca["values"][pri]), fmt(tcl["values"][pri])),
                            [op(tca, cur), op(tcl, cur), op(tca, pri), op(tcl, pri)]))
            ratio = tca["values"][cur] / tcl["values"][cur]
            if ratio >= 1:
                out.append(("current_ratio",
                            "What was the current ratio (%s divided by %s) %s? Round to two decimal places." % (q(tca["label"]), q(tcl["label"]), d_cur),
                            ratio, "%.2f" % ratio, "divide(%s, %s)" % (fmt(tca["values"][cur]), fmt(tcl["values"][cur])), [op(tca, cur), op(tcl, cur)]))
        liabilities = one_line(statement, TOTAL_LIABILITIES, [cur])
        equity = one_line(statement, PARENT_EQUITY, [cur]) or one_line(statement, TOTAL_EQUITY, [cur])
        if liabilities and equity and equity["values"][cur] > 0 and liabilities["values"][cur] > 0:
            v = liabilities["values"][cur] / equity["values"][cur]
            if pct_ok(v):
                out.append(("liabilities_to_equity",
                            "What was %s as a percentage of %s %s? Answer in percent, rounded to one decimal place." % (q(liabilities["label"]), q(equity["label"]), d_cur),
                            v, "%.1f%%" % (100 * v), "divide(%s, %s)" % (fmt(liabilities["values"][cur]), fmt(equity["values"][cur])), [op(liabilities, cur), op(equity, cur)]))
    elif kind == "cash_flow":
        found = comparison(statement, meta["period"])
        if not found:
            return []
        _, cur, pri = found
        p_cur, p_pri = period_phrase(cols[cur], meta["form"]), period_phrase(cols[pri], meta["form"])
        ocf = one_line(statement, OCF, [cur, pri])
        capex = one_line(statement, CAPEX, [cur, pri])
        if ocf and capex and capex["values"][cur] < 0 and capex["values"][pri] < 0:          # an outflow, shown in parentheses
            d = dec([(ocf, cur), (capex, cur), (ocf, pri), (capex, pri)])
            fcf_cur, fcf_pri = ocf["values"][cur] + capex["values"][cur], ocf["values"][pri] + capex["values"][pri]
            definition = "%s minus the cash paid in %s" % (q(ocf["label"]), q(capex["label"]))
            if fcf_cur != 0:
                out.append(("free_cash_flow",
                            "What was free cash flow for %s, defined as %s, in %s? Answer as a number (negative if the result is negative)."
                            % (p_cur, definition, unit), fcf_cur, amount(fcf_cur, d),
                            "subtract(%s, %s)" % (fmt(ocf["values"][cur]), fmt(-capex["values"][cur])), [op(ocf, cur), op(capex, cur)]))
            if fcf_cur != fcf_pri:
                out.append(("free_cash_flow_change",
                            "By how much did free cash flow, defined as %s, change from %s to %s, in %s? Answer as a number (negative for a decrease)."
                            % (definition, p_pri, p_cur, unit), fcf_cur - fcf_pri, amount(fcf_cur - fcf_pri, d),
                            "subtract(%s, %s), subtract(%s, %s), subtract(#0, #1)" % (fmt(ocf["values"][cur]), fmt(-capex["values"][cur]), fmt(ocf["values"][pri]), fmt(-capex["values"][pri])),
                            [op(ocf, cur), op(capex, cur), op(ocf, pri), op(capex, pri)]))
        icf, fincf = one_line(statement, ICF, [cur]), one_line(statement, FINCF, [cur])
        if ocf and icf and fincf:
            total = ocf["values"][cur] + icf["values"][cur] + fincf["values"][cur]
            if total != 0:
                d = dec([(ocf, cur), (icf, cur), (fincf, cur)])
                out.append(("cash_flow_sum",
                            "What was the sum of %s, %s and %s for %s, in %s? Answer as a number (negative if the total is negative)."
                            % (q(ocf["label"]), q(icf["label"]), q(fincf["label"]), p_cur, unit), total, amount(total, d),
                            "add(%s, %s), add(#0, %s)" % (fmt(ocf["values"][cur]), fmt(icf["values"][cur]), fmt(fincf["values"][cur])),
                            [op(ocf, cur), op(icf, cur), op(fincf, cur)]))
        income = one_line(statement, CF_NET_INCOME, [cur])
        if ocf and income and income["values"][cur] > 0:
            v = ocf["values"][cur] / income["values"][cur]
            if pct_ok(v) and abs(v) <= 20:
                out.append(("cash_conversion",
                            "What was %s as a percentage of %s for %s? Answer in percent (negative if operating cash flow was negative), rounded to one decimal place."
                            % (q(ocf["label"]), q(income["label"]), p_cur), v, "%.1f%%" % (100 * v),
                            "divide(%s, %s)" % (fmt(ocf["values"][cur]), fmt(income["values"][cur])), [op(ocf, cur), op(income, cur)]))
    return out


V2_ROUNDING = {"working_capital": "exact (table units)", "working_capital_change": "exact (table units)", "free_cash_flow": "exact (table units)",
               "free_cash_flow_change": "exact (table units)", "cash_flow_sum": "exact (table units)", "current_ratio": "two decimal places"}


def build_filing_v2(source: SecSource, cik: str, accession: str, n: int) -> tuple:
    """(items, None, record) or ([], reason, record): `n` questions with distinct templates, at least one from the
    balance sheet or the cash-flow statement when the filing supports it, the rest composed income-statement ones."""
    base = folder(cik, accession)
    try:
        meta = parse_filing_index(source.get(base + accession + "-index.htm"))
    except (NotCached, ValueError) as error:
        return [], "filing index unreadable: %s" % error, {}
    record = {"cik": str(int(cik)), "accession": accession, **meta}
    if meta["form"] not in FORMS:
        return [], "form %s is not an original 10-K/10-Q" % meta["form"], record
    if date.fromisoformat(meta["filing_date"]) < CUTOFF:
        return [], "filed before %s" % CUTOFF, record
    if not meta.get("period_of_report"):
        return [], "no period of report on the index page", record
    meta["period"] = date.fromisoformat(meta["period_of_report"])
    try:
        reports = statements(source.get(base + "FilingSummary.xml"))
    except NotCached:
        return [], "no FilingSummary.xml (no XBRL rendering)", record
    pools, tables = {}, {}
    rng = random.Random(sha256("v2-questions:" + accession))
    for kind, report in (("income", income_statement(reports)), ("balance", statement_of(reports, "balance")), ("cash_flow", statement_of(reports, "cash_flow"))):
        if not report:
            continue
        try:
            statement = parse_statement(source.get(base + report["file"]), instants=(kind == "balance"))
        except (NotCached, ValueError):
            continue
        made = v2_questions(kind, statement, meta, rng)
        if made:
            pools[kind], tables[kind] = made, (report, statement)
    chosen, used = [], set()
    order = [k for k in ("balance", "cash_flow") if k in pools]
    rng.shuffle(order)
    order = order[:1] + (["income"] if "income" in pools else []) + order[1:]
    while len(chosen) < n and any(pools.get(k) for k in order):
        for kind in order:
            options = [t for t in pools.get(kind, []) if t[0] not in used]
            if options and len(chosen) < n:
                # the template asked least often so far in this build (filings are built in a fixed order), ties at random
                rng.shuffle(options)
                pick = min(options, key=lambda t: V2_USAGE[t[0]])
                chosen.append((kind, pick))
                used.add(pick[0])
            pools[kind] = [t for t in pools.get(kind, []) if t[0] not in used]
    if len(chosen) < n:
        return [], "fewer than %d distinct v2 questions (balance sheet, cash flow, composed income)" % n, record
    V2_USAGE.update(t[0] for _, t in chosen)
    company = meta["company"] or "The company"
    items = []
    for k, (kind, (template, question, value, answer, program, operands)) in enumerate(chosen, 1):
        report, statement = tables[kind]
        shares = "; share counts in %s" % statement["shares_scale"] if statement["shares_scale"] else ""
        pre = ["%s (CIK %s) filed this Form %s with the SEC on %s (accession %s); the period of report ends %s."
               % (company, int(cik), meta["form"], meta["filing_date"], accession, meta["period_of_report"]),
               "The table below is the %s from that filing." % statement["statement"],
               "Amounts are in %s, except per-share amounts%s." % (statement["unit"], shares)]
        table = [[""] + [c["header"] for c in statement["columns"]]] + [[r["shown"]] + r["cells"] for r in shown_rows(statement)]
        items.append({"id": "sec2-%s-q%d" % (accession, k), "filename": base + report["file"],
                      "pre_text": pre, "table": table, "post_text": statement["footnotes"],
                      "qa": {"question": question, "answer": answer, "exe_ans": round(value, 6), "program": program},
                      "sec": {"cik": str(int(cik)), "company": company, "form": meta["form"], "accession": accession,
                              "filing_date": meta["filing_date"], "accepted": meta["accepted"], "period_of_report": meta["period_of_report"],
                              "table_location": {"r_page": base + report["file"], "statement": report["name"]}, "statement_kind": kind,
                              "template": template, "unit": statement["unit"], "operands": operands,
                              "rounding": V2_ROUNDING.get(template, "one decimal place")}})
    return items, None, record


# ------------------------------------------------------------------------------------- build
V3_SCHEMA = "p4-sec-finqa-slice.v3"
V3_MONTHS = ("2026-07", "2026-08", "2026-09")
V3_FAMILIES = ("free_cash_flow_change", "margin_pp_change", "relative_growth", "combined_share", "incremental_margin")
V3_HOLDOUT_SALT = "4a0-sec-issuer-holdout-v3"  # independent of selection seed, month and model
V3_MAX_CANDIDATES = 240


def issuer_holdout(cik: str) -> bool:
    """A permanent 25% hash partition, not a forced 25% of this small sample."""
    return int(sha256(V3_HOLDOUT_SALT + ":" + str(int(cik))), 16) % 4 == 0


def v3_questions(kind: str, statement: dict, meta: dict, rng: random.Random) -> list:
    """v2 procedures, recurring by construction; margin means operating profit / revenue.

    Relative growth compares two growth rates over the same current/prior-year periods (four cells).
    It does not require three annual columns, which would restrict supply to annual filers.
    """
    made = [q for q in v2_questions(kind, statement, meta, rng)
            if q[0] in V3_FAMILIES and q[0] != "margin_pp_change"]
    pair = comparison(statement, meta["period"])
    if kind != "income" or not pair:
        return made
    _, cur, pri = pair
    revs = [r for r in eligible_lines(statement, [cur, pri]) if REVENUE.match(r["label"].strip())]
    revs.sort(key=lambda r: not r["label"].lower().startswith("total"))
    operating = one_line(statement, OPERATING, [cur, pri])
    if revs and operating:
        rev = revs[0]
        value = operating["values"][cur] / rev["values"][cur] - operating["values"][pri] / rev["values"][pri]
        if abs(100 * value) >= SMALLEST_PERCENT:
            operands = [{"label": r["label"], "period": statement["columns"][i]["header"],
                         "cell": r["cells"][i], "value": r["values"][i]}
                        for r, i in ((operating, cur), (rev, cur), (operating, pri), (rev, pri))]
            made.append(("margin_pp_change",
                         'By how many percentage points did the operating margin ("%s" divided by "%s") change from %s to %s? '
                         'Answer in percentage points (negative for a decrease), rounded to one decimal place.'
                         % (operating["label"], rev["label"], period_phrase(statement["columns"][pri], meta["form"]),
                            period_phrase(statement["columns"][cur], meta["form"])),
                         value, "%.1f" % (100 * value),
                         "divide(%s, %s), divide(%s, %s), subtract(#0, #1)" % tuple(fmt(o["value"]) for o in operands), operands))
    return made


def check_v3_item(item: dict, finqa) -> None:
    """Fail closed on source, column, unit and gold discrepancies, before writing a batch."""
    s = item["sec"]
    assert s["template"] in V3_FAMILIES
    assert s["unit"] in ("US dollars", "thousands of US dollars", "millions of US dollars", "billions of US dollars")
    for o in s["operands"]:
        assert o["unit"] == s["unit"] and o["r_page"] == item["filename"] == s["table_location"]["r_page"]
        assert item["table"][o["table_row"]][0].split(" [")[0] == o["label"]
        assert item["table"][0][o["table_column"]] == o["period"]
        assert item["table"][o["table_row"]][o["table_column"]] == o["cell"]
        assert parse_value(o["cell"]) == o["value"]
    assert finqa.gold_consistent(item)
    # The scorer intentionally accepts percent/fraction scales. Independently execute the recorded program.
    values = []
    for operation, a, b in re.findall(r"(add|subtract|divide)\(([^,]+),\s*([^\)]+)\)", item["qa"]["program"]):
        def val(token):
            return values[int(token[1:])] if token.startswith("#") else float(token)
        x, y = val(a), val(b)
        values.append({"add": lambda: x + y, "subtract": lambda: x - y, "divide": lambda: x / y}[operation]())
    assert values and abs(values[-1] - item["qa"]["exe_ans"]) <= 1e-6


def build_filing_v3(source: SecSource, cik: str, accession: str) -> tuple:
    """All supported family candidates; final selection is balanced across the whole month."""
    base = folder(cik, accession)
    try:
        meta = parse_filing_index(source.get(base + accession + "-index.htm"))
    except (NotCached, ValueError) as error:
        return [], "filing index unreadable: %s" % error, {}
    record = {"cik": str(int(cik)), "accession": accession, **meta}
    if meta["form"] not in FORMS or date.fromisoformat(meta["filing_date"]) < CUTOFF or not meta.get("period_of_report"):
        return [], "not an eligible original dated 10-K/10-Q with a period of report", record
    meta["period"] = date.fromisoformat(meta["period_of_report"])
    try:
        reports = statements(source.get(base + "FilingSummary.xml"))
    except NotCached:
        return [], "no FilingSummary.xml (no XBRL rendering)", record
    items, rng = [], random.Random(sha256("v3-questions:" + accession))
    for kind, report in (("income", income_statement(reports)), ("cash_flow", statement_of(reports, "cash_flow"))):
        if not report:
            continue
        try:
            statement = parse_statement(source.get(base + report["file"]))
        except (NotCached, ValueError):
            continue
        table = [[""] + [c["header"] for c in statement["columns"]]] + [[r["shown"]] + r["cells"] for r in shown_rows(statement)]
        company = meta["company"] or "The company"
        for template, question, value, answer, program, operands in v3_questions(kind, statement, meta, rng):
            # Unique source rows and period columns are required even for signed lines outside eligible_lines().
            if any(sum(r[0].split(" [")[0] == o["label"] for r in table[1:]) != 1 or
                   table[0].count(o["period"]) != 1 for o in operands):
                continue
            for o in operands:
                o.update(unit=statement["unit"], r_page=base + report["file"],
                         table_row=next(i for i, r in enumerate(table[1:], 1) if r[0].split(" [")[0] == o["label"]),
                         table_column=table[0].index(o["period"]))
            answer_unit = (statement["unit"] if template == "free_cash_flow_change" else
                           "percentage points" if template in ("margin_pp_change", "relative_growth") else "percent")
            items.append({"id": "sec3-%s-%s" % (accession, template), "filename": base + report["file"],
                          "pre_text": ["%s (CIK %s), Form %s filed %s, accession %s; period of report ends %s."
                                       % (company, int(cik), meta["form"], meta["filing_date"], accession, meta["period_of_report"]),
                                       "The table is the %s from that filing. Amounts are in %s, except per-share amounts."
                                       % (statement["statement"], statement["unit"])],
                          "table": table, "post_text": statement["footnotes"],
                          "qa": {"question": question, "answer": answer, "exe_ans": round(value, 6), "program": program},
                          "sec": {**record, "table_location": {"r_page": base + report["file"], "statement": report["name"]},
                                  "statement_kind": kind, "template": template, "unit": statement["unit"], "answer_unit": answer_unit,
                                  "rounding": V2_ROUNDING.get(template, "one decimal place"), "operands": operands}})
    return items, None if items else "no qualifying v3 families", record


def balanced_v3(candidates: list, per_family: int, families=V3_FAMILIES) -> list:
    """Deterministic integral max flow: family quotas, one family per filing, <=2 items per filing.

    Residual edges let a later scarce family displace an earlier choice, avoiding greedy false shortages.
    """
    capacity, neighbours = {}, {}
    def edge(a, b, n):
        capacity[a, b], capacity[b, a] = n, 0
        neighbours.setdefault(a, []).append(b)
        neighbours.setdefault(b, []).append(a)
    for family in families:
        edge("source", "family:" + family, per_family)
    by_filing = {}
    for item in candidates:
        accession = item["sec"]["accession"]
        by_filing.setdefault(accession, {})[item["sec"]["template"]] = item
    for accession, supported in by_filing.items():
        edge("filing:" + accession, "sink", 2)
        for family in families:
            if family in supported:
                edge("family:" + family, "filing:" + accession, 1)
    while True:
        previous, queue = {"source": None}, deque(["source"])
        while queue and "sink" not in previous:
            a = queue.popleft()
            for b in neighbours.get(a, []):
                if b not in previous and capacity[a, b] > 0:
                    previous[b] = a
                    queue.append(b)
        if "sink" not in previous:
            break
        b = "sink"
        while previous[b] is not None:
            a = previous[b]
            capacity[a, b] -= 1
            capacity[b, a] += 1
            b = a
    return [item for accession, families in by_filing.items() for family, item in families.items()
            if capacity["filing:" + accession, "family:" + family] == 1]


def cmd_build_v3(args) -> int:
    months = args.months.split(",")
    if args.sample:
        raise SystemExit("v3 uses --cache <fixture cache> --offline for fixtures; --sample remains v1/v2 only")
    if (not months or months != sorted(set(months)) or
            any(not re.fullmatch(r"\d{4}-(0[1-9]|1[0-2])", m) for m in months)):
        raise SystemExit("--months must be unique, valid YYYY-MM values in chronological order")
    if any(m >= date.today().strftime("%Y-%m") for m in months):
        raise SystemExit("v3 requires complete months")
    target = args.per_month * args.questions_per_filing
    if args.questions_per_filing != 2 or args.per_month <= 0 or target % len(V3_FAMILIES):
        raise SystemExit("v3 requires --questions-per-filing 2 and a positive question target divisible by five")
    quota = target // len(V3_FAMILIES)
    if args.dry_run:
        print(json.dumps({"schema": V3_SCHEMA, "months": months, "out": args.out, "questions_per_month": target,
                          "family_targets_per_month": dict.fromkeys(V3_FAMILIES, quota), "max_questions_per_filing": 2,
                          "max_candidates_per_month": V3_MAX_CANDIDATES, "holdout_salt": V3_HOLDOUT_SALT,
                          "fetch": False}, indent=2))
        return 0
    out, finqa = Path(args.out), load_kit("beds/finqa")
    if list(out.glob("batch-*.json")) or (out / "manifest.json").exists():
        raise SystemExit("v3 build requires a fresh output directory (README is allowed)")
    source = SecSource(Path(args.cache), allow_fetch=not args.offline)
    batches, exclusions, records, selected_months, shortages = {}, [], [], {}, {}
    # Preserve co-registrant identities from the entire indexes, not just whichever listing was selected first.
    indexes = {}
    peers = {}
    for m in months:
        year, mon = map(int, m.split("-"))
        path = "Archives/edgar/full-index/%d/QTR%d/form.idx" % (year, (mon - 1) // 3 + 1)
        indexes.setdefault(path, parse_form_index(source.get(path)))
    for rows in indexes.values():
        accessions = {}
        for row in rows:
            if row["form"] in (*FORMS, "10-K/A", "10-Q/A"):
                accessions.setdefault(row["accession"], set()).add(row["cik"])
        for ciks in accessions.values():
            for cik in ciks:
                peers.setdefault(cik, set()).update(ciks)
    def issuer_group(cik):
        group, queue = set(), [cik]
        while queue:
            c = queue.pop()
            if c not in group:
                group.add(c)
                queue.extend(peers.get(c, set()) - group)
        return sorted(group, key=int)
    used_groups = set()
    for m in months:
        ordered, amendments = candidates_for_month(source, m, args.seed)
        pool, filing_records, selected = [], {}, []
        seen_issuers = set()
        for f in ordered[:V3_MAX_CANDIDATES]:
            made, reason, record = build_filing_v3(source, f["cik"], f["accession"])
            group_ciks = issuer_group(f["cik"])
            group = "%s:%s" % (group_ciks[0], record.get("period_of_report"))
            if record and not record.get("filing_date", "").startswith(m):
                reason = "filing date is outside requested month"
            if group in used_groups or set(group_ciks) & seen_issuers:
                reason = "issuer or filing group already used in this month, or filing group in an earlier month"
            if reason:
                exclusions.append({"month": m, "cik": f["cik"], "accession": f["accession"], "reason": reason})
                continue
            seen_issuers.update(group_ciks)
            heldout = any(issuer_holdout(c) for c in group_ciks)
            split = {"issuer_ciks": group_ciks, "issuer_group": group_ciks[0], "filing_group": group,
                     "issuer_holdout": heldout, "filing_group_holdout": heldout,
                     "split": "holdout" if heldout else "stream", "training_allowed": not heldout}
            for item in made:
                item["sec"].update(split)
                check_v3_item(item, finqa)
            pool.extend(made)
            filing_records[f["accession"]] = {**record, **split, "month": m, "group": group,
                                              "co_filers": f["co_filers"],
                                              "amendments_seen": [a for c in group_ciks for a in amendments.get(c, [])
                                                                  if a["date"] >= record["filing_date"]]}
            selected = balanced_v3(pool, quota)
            if len(selected) == target:
                break
        counts = Counter(i["sec"]["template"] for i in selected)
        if len(selected) != target:
            shortages[m] = {family: quota - counts[family] for family in V3_FAMILIES}
        selected_months[m] = selected
        chosen = {i["sec"]["accession"] for i in selected}
        records.extend(r for a, r in filing_records.items() if a in chosen)
        used_groups.update(r["group"] for a, r in filing_records.items() if a in chosen)
    out.mkdir(parents=True, exist_ok=True)
    # An incomplete slice has diagnostics only. No unbalanced batch can accidentally reach a paid readout.
    if not shortages:
        for m, items in selected_months.items():
            trainable = [i for i in items if i["sec"]["training_allowed"]]
            batches[m] = write_batch(out, "batch-" + m, items, finqa, trainer_items=trainable)
            batches[m].update(trainer_questions=len(trainable),
                              holdout_questions=len(items) - len(trainable),
                              source_unit_checks=len(items))
    manifest = {"schema": V3_SCHEMA, "status": "incomplete" if shortages else "complete", "mode": "slice", "months": months,
                "seed": args.seed, "questions_per_month_target": target, "family_targets_per_month": dict.fromkeys(V3_FAMILIES, quota),
                "max_questions_per_filing": 2, "max_candidates_per_month": V3_MAX_CANDIDATES,
                "cutoff": str(CUTOFF), "forms": list(FORMS), "shortages": shortages,
                "holdout": {"salt": V3_HOLDOUT_SALT, "rule": "sha256(salt:canonical_cik) modulo 4 == 0",
                            "nominal_issuer_fraction": 0.25, "co_registrants": "entire connected issuer group held out if any member hashes to holdout",
                            "filing_groups": "issuer:period_of_report; inherit issuer holdout; never reused across months",
                            "trainer_export": "stream issuers only; holdouts remain in evaluation JSON"},
                "batches": batches, "filings": records, "exclusions": exclusions,
                "script_sha256": sha256(Path(__file__).read_bytes()),
                "finqa_bed_sha256": sha256((REPO / "kit/beds/finqa.py").read_bytes()),
                "finqa_instruction_sha256": sha256(finqa.INSTRUCTION),
                "sources": [{"path": p, "sha256": h} for p, h in sorted(source.read.items())],
                "fetch": {"requests": source.requests, "user_agent_set": bool(os.environ.get("SEC_USER_AGENT", "").strip()),
                          "min_interval_seconds": MIN_INTERVAL}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print("wrote", out / "manifest.json", manifest["status"], "shortages:", shortages)
    return 2 if shortages else 0


# ------------------------------------------------------------------------------------- slice v4
# Isolated from legacy generation/serialization. Conservative source ambiguity is reported, never difficulty-mined.
V4_SCHEMA = "p4-sec-finqa-slice.v4"
V4_FAMILIES = ("fcf_margin_change", "cash_conversion_change", "quarter_incremental_margin",
               "growth_acceleration", "segment_margin_divergence")
V4_SCALE = {"US dollars": 1, "thousands of US dollars": 1000,
            "millions of US dollars": 1000000, "billions of US dollars": 1000000000}
V4_MAX_CANDIDATES = 240
# These are operand labels, not a document-wide keyword screen. An adoption note or a
# policy discussion does not establish that any particular comparative amount changed.
V4_BASIS_MARK = re.compile(r"\b(?:as\s+(?:restated|adjusted|revised)|restated|recast|revised)\b", re.I)
V4_XBRL_BASIS = re.compile(r"restat|recast|asadjusted|asrevised|revisionofprior|changeinaccountingprinciple", re.I)
V4_POLICY_MENTION = re.compile(r"accounting\s+polic\w*|accounting\s+(?:standards?|principles?)|"
                               r"adopt\w*.{0,60}(?:standard|ASU\b)|restat\w*|recast\w*|revis\w*.{0,30}(?:prior|comparative)", re.I)
V4_CAPEX = re.compile(r"^(capital expenditures?|(?:payments? for |cash paid for |purchases? of |additions? to )"
                     r"(?:property,? plant(?:,? and| and)? equipment|property and equipment|fixed assets))$", re.I)
# Owner-approved, 20 Oct, before any January–June model output. Used only by the
# opt-in amendment screen; rendered-label/custom concepts are added from the original.
V4_AMENDMENT_CONCEPTS = frozenset('''
NetCashProvidedByUsedInOperatingActivities
NetCashProvidedByUsedInOperatingActivitiesContinuingOperations
PaymentsToAcquirePropertyPlantAndEquipment PaymentsToAcquireProductiveAssets
PaymentsForProceedsFromProductiveAssets PaymentsToDevelopRealEstateAssets
PaymentsToAcquireOtherPropertyPlantAndEquipment
Revenues SalesRevenueNet SalesRevenueGoodsNet SalesRevenueServicesNet
RevenueFromContractWithCustomerExcludingAssessedTax
RevenueFromContractWithCustomerIncludingAssessedTax OperatingIncomeLoss
'''.split())
V4_PERIODIC_AMENDMENTS = frozenset({'10-K/A', '10-Q/A', '10-KT/A', '10-QT/A'})
V4_RESTATEMENT = re.compile(r'\b(?:restate|restatement|restated)\b', re.I)


class V4AmendmentDocument(HTMLParser):
    """Visible paragraph boundaries plus inline-XBRL concept/context ends (no amounts needed)."""
    breaks = {'p', 'div', 'br', 'tr', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'}

    def __init__(self, raw):
        super().__init__(convert_charrefs=True)
        self.blocks, self.pending, self.contexts, self.facts = [], [], {}, []
        self.context, self.end_date, self.end_text, self.ignored = None, False, [], 0
        self.feed(raw); self.close(); self.flush()

    def flush(self):
        text = ' '.join(''.join(self.pending).split())
        if text:
            self.blocks.append(text)
        self.pending = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        local = tag.rsplit(':', 1)[-1]
        if tag in ('script', 'style'):
            self.ignored += 1
        if tag in self.breaks:
            self.flush()
        if local == 'context':
            self.context = attrs['id']
        if local == 'enddate':
            self.end_date, self.end_text = True, []
        if tag in ('ix:nonfraction', 'ix:nonnumeric', 'ix:fraction'):
            self.facts.append((attrs['name'].rsplit(':', 1)[-1], attrs['contextref']))

    def handle_endtag(self, tag):
        local = tag.rsplit(':', 1)[-1]
        if local == 'enddate':
            self.contexts[self.context] = ''.join(self.end_text).strip()
            self.end_date = False
        if local == 'context':
            # Instant contexts (cover facts) have no duration end.
            self.contexts.setdefault(self.context, None)
            self.context = None
        if tag in self.breaks:
            self.flush()
        if tag in ('script', 'style'):
            self.ignored = max(0, self.ignored - 1)

    def handle_data(self, text):
        if self.end_date:
            self.end_text.append(text)
        if not self.ignored:
            self.pending.append(text)


def v4_amendment_text_restates(blocks):
    """Screen section contents, not unrelated Part III/exhibit prose."""
    section = re.compile(r'^(?:(?:item\s+(?:1|8)\s*[.:-]?\s*)?'
                         r'(?:(?:notes?\s+to\s+)?(?:(?:the|our|condensed|consolidated|combined|audited|unaudited)\s+)*'
                         r'(?:financial statements(?:\s+and\s+supplementary data)?|'
                         r'statements?\s+of\s+(?:cash\s+flows?|operations|income|earnings|loss|comprehensive\s+income)|'
                         r'balance\s+sheets?)|explanatory notes?))\s*[.:]*$', re.I)
    boundary = re.compile(r'^(?:part\s+[IVX]+\b|item\s+\d+[A-Z]?\b|signatures?\b|exhibit\s+(?:index|\d))', re.I)
    active = False
    for block in blocks:
        heading = block.split(':', 1)[0].strip()
        if section.fullmatch(heading) or re.match(r'^explanatory notes?\b', block, re.I):
            active = True
            if V4_RESTATEMENT.search(block):
                return True
        elif boundary.match(block):
            active = False
        elif active and V4_RESTATEMENT.search(block):
            return True
    return False


def v4_read_amendment(source, row):
    """Read an amendment without requiring financial statements, units or a linkbase.

    No instance is normal for Part III/exhibit-only amendments. An advertised but
    missing/malformed instance, or an unbound inline fact, is an unreadable source.
    """
    import xml.etree.ElementTree as ET
    base, accession = folder(row['cik'], row['accession']), row['accession']
    index = source.get(base + accession + '-index.htm')
    meta = parse_filing_index(index)
    if meta['form'] != row['form'] or meta['filing_date'] != row['date']:
        raise ValueError('amendment index identity mismatch')
    primary = set()
    for tr in re.findall(r'<tr\b[^>]*>(.*?)</tr>', index, re.S | re.I):
        cells = re.findall(r'<td\b[^>]*>(.*?)</td>', tr, re.S | re.I)
        if any(re.sub('<[^>]+>', '', c).strip() == meta['form'] for c in cells):
            primary.update(re.findall(r'href=[\"\']([^\"\']+\.(?:htm|html))[\"\']', tr, re.I))
    if len(primary) != 1:
        raise ValueError('amendment primary document not uniquely identified')
    raw = source.get(base + next(iter(primary)).rsplit('/', 1)[-1])
    document = V4AmendmentDocument(raw)
    if not document.blocks:
        raise ValueError('empty amendment document')
    facts = []
    for concept, context in document.facts:
        facts.append((concept, document.contexts[context]))
    links = re.findall(r'href=[\"\']([^\"\']+)[\"\']', index, re.I)
    names = sorted({p.rsplit('/', 1)[-1] for p in links if p.lower().endswith('.xml')
                    and not re.search(r'_(cal|def|lab|pre)\.xml$|FilingSummary', p, re.I)})
    instance_found = False
    for name in names:
        root = ET.fromstring(source.get(base + name))
        if root.tag.rsplit('}', 1)[-1] != 'xbrl':
            continue
        instance_found = True
        contexts = {}
        for e in root:
            if e.tag.rsplit('}', 1)[-1] == 'context':
                contexts[e.attrib['id']] = next((x.text for x in e.iter()
                                                if x.tag.rsplit('}', 1)[-1] == 'endDate'), None)
        for e in root:
            if 'contextRef' in e.attrib:
                facts.append((e.tag.rsplit('}', 1)[-1], contexts[e.attrib['contextRef']]))
    if names and not instance_found:
        raise ValueError('no readable amendment XBRL instance')
    return {'facts':facts, 'text_restates':v4_amendment_text_restates(document.blocks)}


def v4_amendment_restates(evidence, filing):
    period = date.fromisoformat(filing['record']['period_of_report'])
    # Use the actual comparative fiscal end(s), including 52/53-week calendars.
    ends = {period.isoformat()}
    concepts = set(V4_AMENDMENT_CONCEPTS)
    for table in filing['tables'].values():
        for col in table['statement']['columns']:
            if col['date'] and 350 <= (period - col['date']).days <= 380:
                ends.add(col['date'].isoformat())
        for line in table['statement']['body'] + table['statement']['segments']:
            if line.get('concept') and any(p.match(line['label']) for p in (OCF, V4_CAPEX, REVENUE, OPERATING)):
                concepts.add(line['concept'].split('_', 1)[-1])
    return evidence['text_restates'] or any(concept in concepts and end in ends
                                           for concept, end in evidence['facts'])


class V4ReportTable(_ReportTable):
    def handle_starttag(self, tag, attrs):
        super().handle_starttag(tag, attrs)
        if self.depth == 1 and self.row is not None:
            concepts = re.findall(r"defref_([\w.-]+)", " ".join(v or "" for _, v in attrs))
            if concepts:
                self.row.setdefault("concepts", set()).update(concepts)


def v4_policy_warnings(raw, path):
    text = ' '.join(htmllib.unescape(re.sub('<[^>]+>', ' ', raw)).split())
    return [{'document': path, 'mention': text[max(0, m.start()-80):m.end()+160]}
            for m in V4_POLICY_MENTION.finditer(text)]


def v4_unmarked_label(text):
    return re.sub(r'\s+', ' ', V4_BASIS_MARK.sub('', text)).strip(' ()[]|,;:-')


def v4_parse_statement(raw):
    st = parse_statement(raw)
    parser = V4ReportTable(); parser.feed(raw)
    # Retain the original printed headers, but recognize dates with basis annotations
    # so the operand screen can give the real exclusion reason instead of a date error.
    dates = [c['text'] for c in parser.rows[1]['cells']]
    for col, text in zip(st['columns'], dates):
        col['date'] = parse_date(v4_unmarked_label(text))
    for row in st['body'] + st['segments']:
        row['source_label'] = row['label']
        row['label'] = v4_unmarked_label(row['label'])
    return st


def v4_xml(source, base, index):
    """Use the accession's instance, not companyfacts/frames. No guessed contexts or dimensions."""
    import io
    import xml.etree.ElementTree as ET
    links = re.findall(r'href=[\"\']([^\"\']+)[\"\']', index, re.I)
    names = sorted({p.rsplit("/", 1)[-1] for p in links
                    if p.endswith(".xml") and not re.search(r"_(cal|def|lab|pre)\.xml$|FilingSummary", p)})
    for name in names:
        raw = source.get(base + name)
        try:
            namespaces = {url: prefix for _, (prefix, url) in ET.iterparse(io.StringIO(raw), events=("start-ns",))}
            root = ET.fromstring(raw)
        except ET.ParseError:
            continue
        if root.tag.rsplit("}", 1)[-1] != "xbrl":
            continue
        def qname(tag):
            if tag.startswith("{"):
                url, local = tag[1:].split("}", 1)
                return namespaces.get(url, url) + "_" + local
            return tag.replace(":", "_")
        contexts, units, facts = {}, {}, []
        for e in root:
            kind = e.tag.rsplit("}", 1)[-1]
            if kind == "context":
                get = lambda k: next((x.text for x in e.iter() if x.tag.rsplit("}", 1)[-1] == k), None)
                dims = {qname(x.attrib["dimension"]): qname(x.text or "") for x in e.iter()
                        if x.tag.rsplit("}", 1)[-1] == "explicitMember"}
                contexts[e.attrib["id"]] = {"start": get("startDate"), "end": get("endDate"),
                                              "entity": get("identifier"), "dimensions": dims,
                                              "typed": any(x.tag.rsplit("}", 1)[-1] == "typedMember" for x in e.iter())}
            elif kind == "unit":
                measures = [x.text for x in e.iter() if x.tag.rsplit("}", 1)[-1] == "measure"]
                units[e.attrib["id"]] = measures == ["iso4217:USD"] and not any(x.tag.rsplit("}", 1)[-1] == "divide" for x in e.iter())
            elif "contextRef" in e.attrib and e.text:
                try:
                    from decimal import Decimal
                    value = Decimal(e.text.strip())
                    if value.is_finite():
                        facts.append({"concept": qname(e.tag), "context_id": e.attrib["contextRef"],
                                      "unit_id": e.attrib.get("unitRef"), "value": value,
                                      "decimals": e.attrib.get("decimals")})
                except Exception:
                    continue
        negated = set()
        for link in links:
            if link.endswith('_pre.xml'):
                try:
                    pre = ET.fromstring(source.get(base + link.rsplit('/', 1)[-1]))
                except ET.ParseError as error:
                    raise ValueError('unreadable XBRL presentation linkbase') from error
                for presentation in pre:
                    locators = {x.attrib.get('{http://www.w3.org/1999/xlink}label'):x.attrib.get('{http://www.w3.org/1999/xlink}href','').split('#')[-1]
                                for x in presentation if x.tag.rsplit('}',1)[-1]=='loc'}
                    for arc in presentation:
                        if 'negated' in arc.attrib.get('preferredLabel','').lower():
                            negated.add(locators.get(arc.attrib.get('{http://www.w3.org/1999/xlink}to')))
        return {"contexts": contexts, "units": units, "facts": facts, "instance": base + name, "negated_concepts":negated}
    raise ValueError("no readable accession XBRL instance")


def v4_load_filing(source, cik, accession):
    base = folder(cik, accession)
    index = source.get(base + accession + "-index.htm")
    meta = parse_filing_index(index)
    record = {"cik": str(int(cik)), "accession": accession, **meta}
    if meta["form"] not in FORMS or not meta.get("period_of_report") or date.fromisoformat(meta["filing_date"]) < CUTOFF:
        raise ValueError("not an eligible original dated filing")
    # Narrative policy mentions are audit warnings. Only evidence bound to used
    # statement operands can exclude a question for a changed accounting basis.
    links = re.findall(r'<tr\b[^>]*>(.*?)</tr>', index, re.S | re.I)
    primary = []
    for row in links:
        cells = re.findall(r'<td\b[^>]*>(.*?)</td>', row, re.S | re.I)
        if any(re.sub('<[^>]+>', '', c).strip() == meta['form'] for c in cells):
            primary += re.findall(r'href=[\"\']([^\"\']+\.(?:htm|html))[\"\']', row, re.I)
    if len(set(primary)) != 1:
        raise ValueError("primary document not uniquely identified for accounting-basis screen")
    primary_path = base + primary[0].rsplit('/', 1)[-1]
    warnings = v4_policy_warnings(source.get(primary_path), primary_path)
    xml = v4_xml(source, base, index)
    reports = []
    summary = source.get(base + 'FilingSummary.xml')
    for block in re.findall(r'<Report\b[^>]*>(.*?)</Report>', summary, re.S):
        get = lambda k: (re.search(r'<%s>(.*?)</%s>' % (k, k), block, re.S) or [None, ''])[1].strip()
        if get('HtmlFileName'):
            reports.append({'file': get('HtmlFileName'), 'name': get('ShortName'), 'category': get('MenuCategory')})
    picks = [('income', income_statement([r for r in reports if r['category'] == 'Statements'])),
             ('cash_flow', statement_of([r for r in reports if r['category'] == 'Statements'], 'cash_flow'))]
    segment = [r for r in reports if re.search(r'segment', r['name'], re.I)
               and not re.search(r'parenthetical|polic|text', r['name'], re.I)]
    picks.extend(('segment', report) for report in segment)
    tables, problems, table_problems, ambiguous_segment = {}, [], {}, False
    for kind, report in picks:
        if not report:
            continue
        try:
            raw = source.get(base + report['file'])
            if kind == 'segment' and re.search(r'eliminat|unallocated|corporate|reconcil', raw, re.I):
                raise ValueError('segment eliminations, unallocated amounts or reconciliation')
            st = v4_parse_statement(raw)
            # USD without a scale is allowed only if the title explicitly says USD ($).
            if '$ in' in st['title'] and not SCALE.search(st['title']):
                raise ValueError('unsupported visible monetary scale')
            parser = V4ReportTable(); parser.feed(raw)
            member, mapping = None, {}
            for row in parser.rows:
                if row['class'] == 'rh':
                    cs = row.get('concepts', set())
                    member = next(iter(cs)) if len(cs) == 1 else '?'  # never infer a dimension from a label
                elif row['class'] in ('re', 'ro', 'reu', 'rou') and row['cells']:
                    label = row['cells'][0]['text']
                    shown = (member, label)
                    mapping.setdefault(shown, []).append(row.get('concepts', set()))
            for row in st['body'] + st['segments']:
                if row in st['segments']:
                    label_member = row['shown'].removesuffix(' - ' + row['shown'].split(' - ', 1)[-1])
                    # Resolve the rendered member heading, retaining its exact XBRL member concept.
                    matches = [r for r in parser.rows if r['class'] == 'rh' and r['cells'] and r['cells'][0]['text'] == label_member]
                    cs = matches[0].get('concepts', set()) if len(matches) == 1 else set()
                    m = next(iter(cs)) if len(cs) == 1 else '?'
                else:
                    m = None
                choices = mapping.get((m, row['source_label']), [])
                concepts = choices[0] if len(choices) == 1 else set()
                row['concept'] = next(iter(concepts)) if len(concepts) == 1 else None
                row['member'] = m
            if kind == 'segment':
                if not (any(REVENUE.match(r['label']) for r in st['segments']) and
                        any(OPERATING.match(r['label']) for r in st['segments'])):
                    raise ValueError('segment report has no jointly rendered revenue and operating-income members')
                if ambiguous_segment or 'segment' in tables:
                    ambiguous_segment = True
                    tables.pop('segment', None)
                    raise ValueError('multiple candidate numeric segment reports; no unique source')
            table = [[''] + [c['header'] for c in st['columns']]] + [[r['shown']] + r['cells'] for r in shown_rows(st)]
            tables[kind] = {'statement': st, 'table': table, 'r_page': base + report['file'], 'kind': kind,
                            'accession': accession, 'filing_date': meta['filing_date'], 'xml': xml, 'cik': str(int(cik)),
                            'accounting_policy_warnings': warnings + v4_policy_warnings(raw, base + report['file'])}
        except (NotCached, ValueError) as e:
            problems.append(kind + ': ' + str(e))
            table_problems[kind] = str(e)
    return {'record': record, 'tables': tables, 'screened_document': primary_path, 'problems': problems,
            'table_problems': table_problems, 'accounting_policy_warnings': warnings}


def v4_operand(t, row, i):
    from decimal import Decimal
    st, col = t['statement'], t['statement']['columns'][i]
    for label in (st['title'], col['header'], row.get('source_label', row['label']),
                  row['shown'], row.get('section', '')):
        if V4_BASIS_MARK.search(label):
            raise ValueError('used statement operand marked restated/adjusted/recast/revised: ' + label)
    if V4_XBRL_BASIS.search(row.get('concept') or ''):
        raise ValueError('used XBRL operand marked revised: ' + row['concept'])
    if not row.get('concept') or row.get('member') == '?' or row['values'][i] is None or not col['date'] or not col['months']:
        raise ValueError('unbound rendered concept, dimension, value or period')
    match = VALUE.match(re.sub(r'\[\d+\]', '', row['cells'][i]).strip())
    value = Decimal(match.group(2).replace(',', '')) * (-1 if match.group(1) and match.group(3) else 1) * V4_SCALE[st['unit']]
    matches = []
    for fact in t['xml']['facts']:
        ctx = t['xml']['contexts'].get(fact['context_id'], {})
        dims = ctx.get('dimensions', {})
        if (fact['concept'] != row['concept'] or not t['xml']['units'].get(fact['unit_id']) or
                ctx.get('end') != col['date'].isoformat() or not ctx.get('start') or ctx.get('typed') or
                str(ctx.get('entity', '')).lstrip('0') != t['cik']):
            continue
        days = (col['date'] - date.fromisoformat(ctx['start'])).days + 1
        if abs(days - col['months'] * 365.25 / 12) > 10:
            continue
        # Display rounding is allowed only within half the last visible decimal unit.
        precision = decimals(row, i)
        tolerance = Decimal(V4_SCALE[st['unit']]) * Decimal(10) ** -precision / 2
        basis_dims = {k: v for k, v in dims.items() if V4_XBRL_BASIS.search(k) or V4_XBRL_BASIS.search(v)}
        other_dims = {k: v for k, v in dims.items() if k not in basis_dims}
        same_member = (len(other_dims) == 1 and row['member'] in other_dims.values()) if row['member'] else not other_dims
        if basis_dims and same_member:
            raise ValueError('used XBRL operand has revision/restatement dimension')
        if ((dims and (len(dims) != 1 or row['member'] not in dims.values())) or
                (not dims and row['member'] is not None)):
            continue
        if abs(abs(fact['value']) - abs(value)) <= tolerance and ((fact['value'] >= 0) == (value >= 0) or
                (row['concept'] in t['xml']['negated_concepts'] and fact['value'] > 0 and value < 0)):
            matches.append((fact, ctx))
    # Duplicate facts/contexts and possible presentation sign flips are ambiguous, not silently resolved.
    if len(matches) != 1:
        raise ValueError('source cell does not bind exactly one signed USD fact/context')
    fact, ctx = matches[0]
    shown = row['shown']
    rowids = [n for n, r in enumerate(t['table']) if n and r[0] == shown]
    if len(rowids) != 1 or t['table'][0].count(col['header']) != 1:
        raise ValueError('ambiguous rendered row/column')
    return {'label': row['label'], 'shown_label': shown, 'period': col['header'], 'cell': row['cells'][i],
            'value': row['values'][i], 'unit': st['unit'], 'normalized_usd': str(value),
            'r_page': t['r_page'], 'table_row': rowids[0], 'table_column': i + 1, 'accession': t['accession'],
            'concept': fact['concept'], 'context_id': fact['context_id'], 'unit_id': fact['unit_id'],
            'context': ctx, 'instance': t['xml']['instance'], 'fact_value_usd': str(fact['value']),
            'presentation_negated': (fact['value'] > 0 and value < 0)}


def v4_compute(formula, operands):
    """Decimal expression and separate linear program execution, no intermediate rounding."""
    import ast
    from decimal import Decimal, localcontext
    names = {chr(97 + k): Decimal(o['normalized_usd']) for k, o in enumerate(operands)}
    steps, results = [], []
    operations = {ast.Add: ('add', lambda a, b: a + b), ast.Sub: ('subtract', lambda a, b: a - b),
                  ast.Mult: ('multiply', lambda a, b: a * b), ast.Div: ('divide', lambda a, b: a / b)}
    def visit(node):
        if isinstance(node, ast.Name):
            return names[node.id], str(names[node.id])
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return Decimal(node.value), str(node.value)
        if not isinstance(node, ast.BinOp) or type(node.op) not in operations:
            raise ValueError('unsupported formula')
        a, at = visit(node.left); b, bt = visit(node.right)
        op, fn = operations[type(node.op)]
        value = fn(a, b)
        steps.append('%s(%s, %s)' % (op, at, bt))
        return value, '#' + str(len(steps) - 1)
    with localcontext() as ctx:
        ctx.prec = 50
        value, _ = visit(ast.parse(formula, mode='eval').body)
        for op, a, b in re.findall(r'(add|subtract|multiply|divide)\(([^,]+),\s*([^\)]+)\)', ', '.join(steps)):
            get = lambda x: results[int(x[1:])] if x.startswith('#') else Decimal(x)
            x, y = get(a), get(b)
            results.append({'add': lambda: x + y, 'subtract': lambda: x - y,
                            'multiply': lambda: x * y, 'divide': lambda: x / y}[op]())
        assert results[-1] == value
        from decimal import ROUND_HALF_UP
        rounded = value.quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)
    return value, format(rounded, '.1f'), ', '.join(steps)


def v4_make_item(filing, family, question, formula, refs):
    operands = [v4_operand(t, r, i) for t, r, i in refs]
    # For a cross-filing subtraction, a later required instance can supply direct
    # evidence that the exact earlier operand was revised. Compare only the same
    # concept, USD unit, entity, fiscal boundaries and dimensions, never prose.
    sources = {t['accession']: t for t, _, _ in refs}
    def context_key(ctx):
        return (ctx.get('start'), ctx.get('end'), str(ctx.get('entity', '')).lstrip('0'),
                ctx.get('dimensions'), ctx.get('typed'))
    from decimal import Decimal
    for operand in operands:
        original = sources[operand['accession']]
        for t in sources.values():
            if t['filing_date'] <= original['filing_date']:
                continue
            for fact in t['xml']['facts']:
                if (fact['concept'] == operand['concept'] and t['xml']['units'].get(fact['unit_id']) and
                        context_key(t['xml']['contexts'].get(fact['context_id'], {})) == context_key(operand['context']) and
                        fact['value'] != Decimal(operand['fact_value_usd'])):
                    raise ValueError('used prior-period operand revised in later required XBRL instance')
    value, answer, program = v4_compute(formula, operands)
    # Avoid unresolved near-zero/rounding cases, using source arithmetic only, never model scores.
    if abs(value) < 1:
        raise ValueError('answer below one percent/pp')
    excerpts = []
    for t, _, _ in refs:
        if not any(e['r_page'] == t['r_page'] for e in excerpts):
            excerpts.append({k: t[k] for k in ('kind', 'r_page', 'accession', 'filing_date', 'table')} |
                            {'unit': t['statement']['unit'], 'footnotes': t['statement']['footnotes']})
    # Exact boundaries, not merely approximately year-apart printed dates.
    record = filing['record']
    required = sorted({e['accession'] for e in excerpts})
    latest = max(e['filing_date'] for e in excerpts)
    if latest != record['filing_date']:
        raise ValueError('join must be dated by latest required filing')
    unit = 'percent' if family == 'quarter_incremental_margin' else 'percentage points'
    sec = {**record, 'template': family, 'version': 4, 'operands': operands, 'excerpts': excerpts,
           'unit': 'visible units per excerpt', 'answer_unit': unit, 'formula': formula,
           'unrounded_gold': str(value), 'rounding': 'one decimal place, half up, once at the end',
           'required_accessions': required, 'screened_document': filing['screened_document']}
    sec['accounting_policy_warnings'] = []
    for t, _, _ in refs:
        for warning in t.get('accounting_policy_warnings', []):
            if warning not in sec['accounting_policy_warnings']:
                sec['accounting_policy_warnings'].append(warning)
    first = excerpts[0]
    pre = ['%s (CIK %s); latest required filing date %s.' % (record['company'], record['cik'], latest),
           v4_excerpt_label(first)]
    post = list(first['footnotes'])
    finqa = load_kit('beds/finqa')
    for e in excerpts[1:]:
        post += [v4_excerpt_label(e), finqa.render_table(e['table']), *e['footnotes']]
    item = {'id': 'sec4-%s-%s' % (record['accession'], family), 'filename': first['r_page'],
            'pre_text': pre, 'table': first['table'], 'post_text': post,
            'qa': {'question': question + ' Answer in %s (signed), rounded to one decimal place; round only the final result.' % unit,
                   'exe_ans': float(value), 'answer': answer, 'program': program}, 'sec': sec}
    if not finqa.gold_consistent(item):
        raise ValueError('final rounding incompatible with inherited FinQA gold-consistency check')
    return item


def v4_excerpt_label(e):
    return 'Statement: %s; accession %s, filed %s; source %s. Amounts in %s.' % (
        e['kind'], e['accession'], e['filing_date'], e['r_page'], e['unit'])


def v4_aligned(operands):
    return len({(o['context']['start'], o['context']['end']) for o in operands}) == 1


def v4_questions(filing, earlier=()):
    """Return candidates plus per-family exclusion diagnostics; no source guessing on failure."""
    from decimal import Decimal
    items, reasons = [], {}
    tables, meta = filing['tables'], filing['record']
    def pair(t, months=None):
        cols = t['statement']['columns']
        for i, c in enumerate(cols):
            if c['date'] != date.fromisoformat(meta['period_of_report']) or not c['months'] or (months and c['months'] != months):
                continue
            for j, p in enumerate(cols):
                if p['date'] and p['months'] == c['months'] and 350 <= (c['date'] - p['date']).days <= 380:
                    return i, j
        raise ValueError('no aligned duration/prior-year columns')
    def line(t, pattern, ids):
        r = one_line(t['statement'], pattern, ids)
        if not r:
            raise ValueError('missing or ambiguous line: ' + pattern.pattern)
        return r
    def need(kind):
        if kind not in tables:
            reason = {'income': 'no source-bound income statement',
                      'cash_flow': 'no cash-flow statement found',
                      'segment': 'no segment note with operating profit'}[kind]
            detail = filing.get('table_problems', {}).get(kind)
            raise ValueError(reason + (': ' + detail if detail else ''))
        return tables[kind]
    for family in V4_FAMILIES:
        try:
            inc = need('income') if family != 'segment_margin_divergence' else None
            if family in ('fcf_margin_change', 'cash_conversion_change'):
                cf = need('cash_flow'); c, p = pair(cf)
                icols = inc['statement']['columns']
                ids = []
                for n in (c, p):
                    col = cf['statement']['columns'][n]
                    found = [j for j, x in enumerate(icols) if x['date'] == col['date'] and x['months'] == col['months']]
                    if len(found) != 1:
                        raise ValueError('income/cash-flow duration mismatch')
                    ids.append(found[0])
                ci, pi = ids
                cash = line(cf, OCF, [c, p]); rev = line(inc, REVENUE, ids)
                if any(rev['values'][j] <= 0 for j in ids):
                    raise ValueError('nonpositive revenue denominator')
                if family == 'fcf_margin_change':
                    cap = line(cf, V4_CAPEX, [c, p])
                    if any(cap['values'][j] >= 0 for j in (c, p)) or len([r for r in cf['statement']['body'] if CAPEX.match(r['label'])]) != 1:
                        raise ValueError('ambiguous capex definition or outflow sign')
                    refs = [(cf, cash, c), (cf, cap, c), (inc, rev, ci), (cf, cash, p), (cf, cap, p), (inc, rev, pi)]
                    formula = '100*((a+b)/c-(d+e)/f)'
                    question = 'How many percentage points did FCF/revenue change, current minus prior year? FCF is signed operating cash flow minus positive cash paid for "%s"; use "%s" and "%s" for the matched durations.' % (cap['label'], cash['label'], rev['label'])
                else:
                    op = line(inc, OPERATING, ids)
                    if any(op['values'][j] <= 0 or op['values'][j] < .01 * rev['values'][j] for j in ids):
                        raise ValueError('unstable operating-income denominator (<1% of revenue or nonpositive)')
                    refs = [(cf, cash, c), (inc, op, ci), (cf, cash, p), (inc, op, pi)]
                    formula = '100*(a/b-c/d)'
                    question = 'How many percentage points did cash conversion change, current minus prior year? Cash conversion is "%s" divided by "%s", expressed as a percentage.' % (cash['label'], op['label'])
                ops = [v4_operand(*r) for r in refs]
                cut = 3 if family == 'fcf_margin_change' else 2
                if not v4_aligned(ops[:cut]) or not v4_aligned(ops[cut:]):
                    raise ValueError('cross-statement exact fiscal boundaries differ')
            elif family == 'growth_acceleration':
                c, p = pair(inc, 12)
                cols = inc['statement']['columns']
                oldest = [j for j, x in enumerate(cols) if x['months'] == 12 and x['date'] and 350 <= (cols[p]['date'] - x['date']).days <= 380]
                if len(oldest) != 1:
                    raise ValueError('requires three comparable annual columns')
                ids = [c, p, oldest[0]]
                rev = line(inc, REVENUE, ids)
                if any(rev['values'][j] <= 0 for j in ids):
                    raise ValueError('nonpositive growth base')
                refs = [(inc, rev, j) for j in ids]
                bound = [v4_operand(*r) for r in refs]
                if any(date.fromisoformat(bound[j]['context']['start']) != date.fromisoformat(bound[j+1]['context']['end']) + timedelta(days=1) for j in (0,1)):
                    raise ValueError('annual periods are not consecutive fiscal years')
                formula = '100*((a/b-1)-(b/c-1))'
                question = 'By how many percentage points did annual growth of "%s" accelerate from the oldest-to-middle year to the middle-to-latest year?' % rev['label']
            elif family == 'quarter_incremental_margin':
                c, p = pair(inc, 9)
                matches = []
                for prev in earlier:
                    if prev['record']['cik'] != meta['cik'] or prev['record']['filing_date'] >= meta['filing_date'] or 'income' not in prev['tables']:
                        continue
                    t = prev['tables']['income']; cols = t['statement']['columns']
                    found = [j for j, x in enumerate(cols) if x['months'] == 6 and x['date'] and 75 <= (inc['statement']['columns'][c]['date'] - x['date']).days <= 105]
                    if len(found) != 1:
                        continue
                    pc = found[0]
                    pp = [j for j, x in enumerate(cols) if x['months'] == 6 and x['date'] and 350 <= (cols[pc]['date'] - x['date']).days <= 380]
                    if len(pp) == 1:
                        matches.append((prev, t, pc, pp[0]))
                if len(matches) != 1:
                    raise ValueError('no unique earlier six-month filing')
                prev, six, sc, sp = matches[0]
                r9, o9 = line(inc, REVENUE, [c,p]), line(inc, OPERATING, [c,p])
                r6, o6 = line(six, REVENUE, [sc,sp]), line(six, OPERATING, [sc,sp])
                refs = [(inc,o9,c),(six,o6,sc),(inc,o9,p),(six,o6,sp),
                        (inc,r9,c),(six,r6,sc),(inc,r9,p),(six,r6,sp)]
                ops = [v4_operand(*r) for r in refs]
                for a,b in ((0,1),(2,3),(4,5),(6,7)):
                    if ops[a]['context']['start'] != ops[b]['context']['start'] or ops[a]['concept'] != ops[b]['concept']:
                        raise ValueError('six/nine-month basis or fiscal starts mismatch')
                for a,b in ((0,4),(1,5),(2,6),(3,7)):
                    if not v4_aligned([ops[a],ops[b]]):
                        raise ValueError('income/revenue fiscal boundaries differ')
                vals = [Decimal(o['normalized_usd']) for o in ops]
                rq1, rq0 = vals[4]-vals[5], vals[6]-vals[7]
                if min(rq1,rq0) <= 0 or abs(rq1-rq0) < Decimal('.02')*rq0 or abs((vals[0]-vals[1]-vals[2]+vals[3])/(rq1-rq0)) > 10:
                    raise ValueError('unstable standalone-quarter revenue delta (<2%) or margin >1000%')
                formula = '100*((a-b)-(c-d))/((e-f)-(g-h))'
                question = 'Derive each year\'s standalone third quarter as nine-month minus six-month totals. What is the current-minus-prior-year change in quarter operating income divided by the corresponding change in quarter revenue, expressed as a percentage?'
            else:
                seg = need('segment'); c,p = pair(seg)
                members = {}
                for r in seg['statement']['segments']:
                    if r.get('member') and r['member'] != '?':
                        members.setdefault(r['member'], []).append(r)
                pairs = []
                for member, rows in sorted(members.items()):
                    rev = [r for r in rows if REVENUE.match(r['label'])]
                    op = [r for r in rows if OPERATING.match(r['label'])]
                    if len(rev) == len(op) == 1 and all(rev[0]['values'][j] is not None and rev[0]['values'][j] > 0 and op[0]['values'][j] is not None for j in (c,p)):
                        pairs.append((rev[0], op[0]))
                if len(pairs) != 2:
                    raise ValueError('requires exactly two disjoint explicit segment members with stable operating-profit measure')
                (ra,oa),(rb,ob) = pairs
                refs = [(seg,oa,c),(seg,ra,c),(seg,oa,p),(seg,ra,p),(seg,ob,c),(seg,rb,c),(seg,ob,p),(seg,rb,p)]
                ops = [v4_operand(*r) for r in refs]
                if not v4_aligned([ops[j] for j in (0,1,4,5)]) or not v4_aligned([ops[j] for j in (2,3,6,7)]):
                    raise ValueError('segment period mismatch')
                dims = [o['context']['dimensions'] for o in ops]
                axis = next(iter(dims[0]))
                if ('segment' not in axis.lower() or
                        any(re.search(r'total|consolidat|eliminat|corporate|other|unallocat', r['shown'], re.I) for r in (oa,ob)) or
                        len({next(iter(d)) for d in dims}) != 1 or oa['concept'] != ob['concept'] or ra['concept'] != rb['concept']):
                    raise ValueError('segment axes/profit measures differ')
                formula = '100*((a/b-c/d)-(e/f-g/h))'
                question = 'By how many percentage points did the current-minus-prior-year operating-margin change of "%s" exceed that of "%s"? Margin is segment operating income divided by segment revenue.' % (oa['shown'].rsplit(' - ',1)[0],ob['shown'].rsplit(' - ',1)[0])
            item = v4_make_item(filing, family, question, formula, refs)
            items.append(item)
        except (KeyError, TypeError, ValueError, ZeroDivisionError) as e:
            reasons[family] = str(e)
    return items, reasons


def check_v4_item(item, finqa):
    s = item['sec']
    assert s['template'] in V4_FAMILIES and s['version'] == 4
    for o in s['operands']:
        e = next(e for e in s['excerpts'] if e['r_page'] == o['r_page'])
        assert e['table'][o['table_row']][0] == o['shown_label']
        assert e['table'][0][o['table_column']] == o['period']
        assert e['table'][o['table_row']][o['table_column']] == o['cell']
        assert o['unit'] == e['unit'] and parse_value(o['cell']) == o['value']
        from decimal import Decimal
        assert Decimal(str(o['value'])) * V4_SCALE[o['unit']] == Decimal(o['normalized_usd'])
        assert o['context']['start'] and o['context']['end'] and o['context_id'] and o['concept']
    value, answer, program = v4_compute(s['formula'], s['operands'])
    assert str(value) == s['unrounded_gold'] and float(value) == item['qa']['exe_ans']
    assert answer == item['qa']['answer'] and program == item['qa']['program']
    assert finqa.gold_consistent(item)


def v4_explicit_correct(item, response):
    """Preregistered: requested percent/pp number within 0.05 (one final decimal); no x100 tolerance.

    Numeric final answers need no unit suffix because the prompt specifies it. A conflicting explicit suffix fails.
    """
    from decimal import Decimal
    f = load_kit('beds/finqa')
    pred = f.extract_answer(response)
    if pred is None or isinstance(pred, str):
        return False
    last = f.ANSWER_LINE.findall(response)[-1].strip().lower()
    unit = item['sec']['answer_unit']
    if re.search(r'\$|dollars?|thousands?|millions?|billions?', last):
        return False
    if unit == 'percentage points' and ('%' in last or re.search(r'\bpercent\b(?!age points)', last)):
        return False
    if unit == 'percent' and re.search(r'\bpp\b|percentage points', last):
        return False
    return abs(Decimal(str(float(pred))) - Decimal(item['sec']['unrounded_gold'])) <= Decimal('.0500000001')


def v4_case_text(case, response, finqa):
    parts = []
    for e in case['sec']['excerpts']:
        rowids = {o['table_row'] for o in case['sec']['operands'] if o['r_page'] == e['r_page']}
        parts += [v4_excerpt_label(e), finqa.render_table([e['table'][0]] + [e['table'][r] for r in sorted(rowids)])]
    return '\n'.join([*parts, 'Question: ' + case['qa']['question'], 'Solution:', response.strip()])


def v4_audit(out, items):
    import csv
    path = out / 'gold-audit.csv'
    fields = ['id','family','question','formula','gold_unrounded','gold_rounded','answer_unit','operand',
              'label','r_page','accession','table_row','table_column','period','cell','unit','normalized_usd',
              'concept','context_id','context','instance','fact_value_usd','presentation_negated']
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fields); writer.writeheader()
        for item in items:
            s = item['sec']
            for k,o in enumerate(s['operands']):
                writer.writerow({'id':item['id'], 'family':s['template'], 'question':item['qa']['question'],
                                 'formula':s['formula'],'gold_unrounded':s['unrounded_gold'],
                                 'gold_rounded':item['qa']['answer'], 'answer_unit':s['answer_unit'], 'operand':chr(97+k),
                                 **{key: json.dumps(o[key],sort_keys=True) if key == 'context' else o[key]
                                    for key in fields[8:]}})
    return {'path':path.name, 'sha256':sha256(path.read_bytes()), 'questions':len(items),
            'instruction':'Human-check every question against original sources before model inference; export is not audit approval.'}


def v4_index_quarters(months):
    """Full lookup window: four predecessor quarters through the last arrival quarter."""
    quarters = [int(m[:4]) * 4 + (int(m[5:]) - 1) // 3 for m in months]
    return [(q // 4, q % 4 + 1) for q in range(min(quarters) - 4, max(quarters) + 1)]


def v4_probe_exclusions():
    """Bind exclusions to both frozen probes, including required joins and co-registrants."""
    accessions, ciks, groups, hashes, components = set(), set(), set(), {}, []
    for name in ('slice-v4-cash', 'slice-v4-scale'):
        root = REPO / 'data/p4_sec' / name
        manifest_path = root / 'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        if (manifest.get('schema') != V4_SCHEMA or
                manifest.get('status') not in ('complete', 'complete-with-declared-shortfall')):
            raise SystemExit('feasibility requires frozen v4 cash/scale probe manifests and batches')
        hashes[name + '/manifest.json'] = sha256(manifest_path.read_bytes())
        records = [*manifest['filings'], *manifest.get('required_filings', [])]
        for month in manifest['months']:
            path = root / ('batch-' + month + '.json')
            digest = sha256(path.read_bytes())
            assert digest == manifest['batches'][month]['finqa_json_sha256'], 'probe exclusion batch changed'
            hashes[name + '/' + path.name] = digest
            records.extend(i['sec'] for i in json.loads(path.read_text()))
        for rec in records:
            accessions.update(rec.get('required_accessions', []))
            if rec.get('accession'):
                accessions.add(rec['accession'])
            members = {str(int(c)) for c in [*rec.get('issuer_ciks', []), *rec.get('co_filers', [])]}
            if rec.get('cik'):
                members.add(str(int(rec['cik'])))
            ciks.update(members)
            components.append(members)
            groups.update(rec.get('required_filing_groups', []))
            if rec.get('filing_group'):
                groups.add(rec['filing_group'])
    return {'hashes':hashes, 'accessions':sorted(accessions), 'issuer_ciks':sorted(ciks, key=int),
            'filing_groups':sorted(groups)}, components


def cmd_build_v4(args):
    # Manager amendment, 18 Oct (owner-approved reduced probe): --families builds only the families that have
    # supply in July-September filings; --allow-shortfall writes the batches found and records each shortage.
    families = tuple(args.families.split(',')) if getattr(args, 'families', None) else V4_FAMILIES
    if not families or any(f not in V4_FAMILIES for f in families):
        raise SystemExit('--families must name v4 families: ' + ','.join(V4_FAMILIES))
    # Manager amendment, 18 Oct (owner: "Go ahead with option 1"): scan more filings for a larger slice.
    quota = getattr(args, 'per_family', None) or 4
    max_candidates = getattr(args, 'max_candidates', None) or V4_MAX_CANDIDATES
    per_month = quota * len(families)
    """Fixed 60-item unseen probe; structural completeness is not an accuracy/yield stopping rule."""
    feasibility = args.feasibility
    amendment_rule = args.amendment_rule
    months = args.months.split(',')
    if (not months or months != sorted(set(months)) or
            any(not re.fullmatch(r'\d{4}-(0[1-9]|1[0-2])', m) for m in months)):
        raise SystemExit('--months must be unique, valid YYYY-MM values in chronological order')
    if args.sample or (not feasibility and months != list(V3_MONTHS)) or args.per_month != 10 or args.questions_per_filing != 2:
        raise SystemExit('v4 requires July–September 2026, --per-month 10, --questions-per-filing 2 (20 questions/month)')
    if any(m >= date.today().strftime('%Y-%m') for m in months):
        raise SystemExit('v4 requires complete months')
    if args.dry_run:
        plan = {'schema':V4_SCHEMA, 'months':months, 'questions_per_month':20,
                          'family_targets_per_month':dict.fromkeys(families,quota), 'out':args.out,
                          'holdout_salt':V3_HOLDOUT_SALT,'exclude_v3':args.exclude_v3,'fetch':False}
        if feasibility:
            plan.update(purpose='4a-feasibility-supply-scan', status='feasibility-only',
                        questions_per_month=per_month, index_quarters=v4_index_quarters(months),
                        max_candidates_per_month=max_candidates,
                        exclude_probes=['slice-v4-cash', 'slice-v4-scale'])
        if amendment_rule == 'restating':
            plan['amendment_rule'] = amendment_rule
        print(json.dumps(plan,indent=2))
        return 0
    out = Path(args.out)
    if (out/'manifest.json').exists() or list(out.glob('batch-*')) or (out/'gold-audit.csv').exists():
        raise SystemExit('v4 requires a fresh output directory')
    v3 = Path(args.exclude_v3)
    oldmanifest = json.loads((v3/'manifest.json').read_text())
    if oldmanifest.get('schema') != V3_SCHEMA or oldmanifest.get('status') != 'complete':
        raise SystemExit('v4 requires the complete frozen v3 manifest and batches for exclusion')
    oldids, oldaccessions = set(), {f['accession'] for f in oldmanifest['filings']}
    exclusion_hashes = {'manifest.json':sha256((v3/'manifest.json').read_bytes())}
    for m in oldmanifest['months']:
        p = v3/('batch-%s.json'%m)
        digest = sha256(p.read_bytes())
        assert digest == oldmanifest['batches'][m]['finqa_json_sha256'], 'v3 exclusion batch changed'
        exclusion_hashes[p.name] = digest
        for i in json.loads(p.read_text()):
            oldids.add(i['id']); oldaccessions.add(i['sec']['accession'])
    probe, probe_components = v4_probe_exclusions() if feasibility else ({}, [])
    probe_accessions, probe_ciks = set(probe.get('accessions', [])), set(probe.get('issuer_ciks', []))
    source, finqa = SecSource(Path(args.cache),allow_fetch=not args.offline), load_kit('beds/finqa')
    # Include four earlier quarters for fiscal-quarter joins and their complete co-registrant components.
    indexes, allrows = {}, []
    for year, q in v4_index_quarters(months):
        path = 'Archives/edgar/full-index/%d/QTR%d/form.idx'%(year,q)
        indexes[path] = parse_form_index(source.get(path)); allrows.extend(indexes[path])
    peers, byacc = {}, {}
    for r in allrows:
        if r['form'] in (*FORMS,'10-K/A','10-Q/A'):
            byacc.setdefault(r['accession'],set()).add(r['cik'])
    for ciks in [*byacc.values(), *probe_components]:
        for c in ciks:
            peers.setdefault(c,set()).update(ciks)
    def group(cik):
        seen, queue = set(), [cik]
        while queue:
            c = queue.pop()
            if c not in seen:
                seen.add(c); queue.extend(peers.get(c,set())-seen)
        return sorted(seen,key=int)
    loaded, exclusions, records, selected_months, shortages = {}, [], {}, {}, {}
    amendment_evidence, non_restating = {}, {}
    amendment_errors = {}  # why each unreadable amendment failed (diagnostic, recorded in the manifest)
    # Unique amendment accessions examined per scan month (including join sources).
    # If one amendment is checked against multiple periods, the strongest verdict wins.
    amendment_verdicts = {m:{} for m in months}
    verdict_rank = {'non_restating':0, 'restating':1, 'unreadable':2}

    def amendment_counts(month):
        counts = Counter(amendment_verdicts[month].values())
        return {'examined':sum(counts.values()), **{s:counts[s] for s in verdict_rank}}

    def load(row):
        a = row['accession']
        if a in oldaccessions:
            raise ValueError('v3 filing excluded, including join operands')
        if a in probe_accessions or set(group(row['cik'])) & probe_ciks:
            raise ValueError('probe accession or issuer/co-registrant group excluded, including join operands')
        if a not in loaded:
            loaded[a] = v4_load_filing(source,row['cik'],a)
        f = loaded[a]
        # The registered default retains the exact old screen and serialization.
        if amendment_rule == 'any':
            if any(r['form'].endswith('/A') and r['cik'] in group(row['cik'])
                   and r['date'] >= f['record']['filing_date'] for r in allrows):
                raise ValueError('known amendment: unresolved accounting basis')
        else:
            later = {}
            # Only amended periodic reports can restate the statements used; ownership
            # (SC 13D/G/A), current-report (8-K/A) and registration amendments cannot.
            for r in allrows:
                if (r['form'] in V4_PERIODIC_AMENDMENTS and r['cik'] in group(row['cik'])
                        and r['date'] >= f['record']['filing_date']):
                    later.setdefault(r['accession'], r)
            statuses = {}
            for acc, amendment in sorted(later.items()):
                if acc not in amendment_evidence:
                    try:
                        amendment_evidence[acc] = v4_read_amendment(source, amendment)
                    except (Exception, SystemExit) as error:
                        # Includes cache misses, failed fetches, absent User-Agent and parse errors.
                        amendment_evidence[acc] = None
                        amendment_errors[acc] = '%s: %s' % (type(error).__name__, str(error)[:160])
                evidence = amendment_evidence[acc]
                status = ('unreadable' if evidence is None else
                          'restating' if v4_amendment_restates(evidence, f) else 'non_restating')
                statuses[acc] = status
                previous = amendment_verdicts[month].get(acc, 'non_restating')
                amendment_verdicts[month][acc] = max((previous, status), key=verdict_rank.get)
            non_restating[a] = sorted(acc for acc, status in statuses.items() if status == 'non_restating')
            if 'unreadable' in statuses.values():
                raise ValueError('amendment unreadable: unresolved accounting basis')
            if 'restating' in statuses.values():
                raise ValueError('known restating amendment: unresolved accounting basis')
        return f
    used_groups, used_accession_month, month_supply = set(), {}, {}
    for month in months:
        ordered,_ = candidates_for_month(source,month,args.seed)
        pool, selected, seen_issuers, scanned = [], [], set(), 0
        for row in ordered[:max_candidates]:
            scanned += 1
            a, ciks = row['accession'],group(row['cik'])
            try:
                filing = load(row)
                rec = filing['record']; fg = ciks[0]+':'+rec['period_of_report']
                if rec['filing_date'] != row['date'] or not rec['filing_date'].startswith(month):
                    raise ValueError('filing date differs from index/month')
                if fg in used_groups or set(ciks)&seen_issuers:
                    raise ValueError('issuer/filing group already used')
                earlier = []
                inc = filing['tables'].get('income')
                # Only six-month predecessor filings are needed; bound by the reported fiscal end and public date.
                if inc and any(c['months']==9 and c['date']==date.fromisoformat(rec['period_of_report']) for c in inc['statement']['columns']):
                    previous = {r['accession']:r for r in allrows if r['cik']==row['cik'] and r['form']=='10-Q'
                                and 30 <= (date.fromisoformat(rec['filing_date'])-date.fromisoformat(r['date'])).days <= 160}
                    for prev in sorted(previous.values(),key=lambda r:(r['date'],r['accession'])):
                        try:
                            earlier.append(load(prev))
                        except (NotCached,ValueError) as e:
                            exclusions.append({'month':month,'accession':a,'dependency':prev['accession'],'reason':str(e)})
                made, reasons = v4_questions(filing,earlier)
                for f, reason in reasons.items():
                    exclusions.append({'month':month,'accession':a,'family':f,'reason':reason})
                held = any(issuer_holdout(c) for c in ciks)
                split = {'issuer_ciks':ciks,'issuer_group':ciks[0],'filing_group':fg,
                         'issuer_holdout':held,'filing_group_holdout':held,
                         'training_allowed':not held,'split':'holdout' if held else 'stream'}
                accepted = []
                for item in made:
                    required = item['sec']['required_accessions']
                    if (item['id'] in oldids or set(required)&oldaccessions or
                            any(x in used_accession_month and used_accession_month[x] != month for x in required)):
                        exclusions.append({'month':month,'accession':a,'family':item['sec']['template'],'reason':'v3 or earlier-month filing reuse'})
                        continue
                    # All operands must belong to the current permanent issuer component, even a predecessor.
                    if any(loaded[x]['record']['cik'] not in ciks for x in required):
                        exclusions.append({'month':month,'accession':a,'family':item['sec']['template'],
                                           'reason':'join crosses permanent issuer components'})
                        continue
                    item['sec'].update(split)
                    if amendment_rule == 'restating':
                        item['sec']['non_restating_amendments'] = sorted({amendment for acc in required
                                                                        for amendment in non_restating[acc]})
                    item['sec']['required_filing_groups'] = sorted({ciks[0]+':'+loaded[x]['record']['period_of_report'] for x in required})
                    if set(item['sec']['required_filing_groups']) & used_groups:
                        exclusions.append({'month':month,'accession':a,'family':item['sec']['template'],'reason':'earlier-month filing group reuse'})
                        continue
                    check_v4_item(item,finqa); accepted.append(item)
                if not accepted:
                    continue
                seen_issuers.update(ciks)
                pool.extend(accepted)
                records[a] = {**rec,**split,'month':month,'source_problems':filing['problems']}
                selected = balanced_v3([x for x in pool if x['sec']['template'] in families],quota,families=families)
                if len(selected)==per_month:
                    break
            except (NotCached,ValueError) as e:
                exclusions.append({'month':month,'accession':a,'reason':str(e)})
        counts = Counter(i['sec']['template'] for i in selected)
        if len(selected)!=per_month:
            shortages[month] = {f:quota-counts[f] for f in families}
        if feasibility:
            reasons = Counter(e['reason'] for e in exclusions if e['month'] == month
                              and ('family' not in e or e['family'] in families))
            month_supply[month] = {'candidates_scanned':scanned,
                                   'questions_per_family':{f:counts[f] for f in families},
                                   'shortfall':{f:quota-counts[f] for f in families},
                                   'rejection_reasons':dict(reasons),
                                   'top_10_rejection_reasons':[{'reason':r, 'count':n} for r,n in reasons.most_common(10)]}
            if amendment_rule == 'restating':
                month_supply[month]['amendment_counts'] = amendment_counts(month)
                errs = Counter(amendment_errors[a] for a, v in amendment_verdicts[month].items() if v == 'unreadable' and a in amendment_errors)
                month_supply[month]['amendment_unreadable_reasons'] = dict(errs.most_common(10))
            print(json.dumps({'month':month, **month_supply[month]},sort_keys=True))
        selected_months[month] = selected
        for i in selected:
            used_groups.update(i['sec']['required_filing_groups'])
            for acc in i['sec']['required_accessions']:
                used_accession_month[acc] = month
    out.mkdir(parents=True,exist_ok=True)
    batches = {}
    allitems = [i for items in selected_months.values() for i in items]
    write = not shortages or getattr(args, 'allow_shortfall', False)
    if write:
        for month, items in selected_months.items():
            trainable = [i for i in items if i['sec']['training_allowed']]
            batches[month] = write_batch(out,'batch-'+month,items,finqa,trainer_items=trainable)
            batches[month].update(trainer_questions=len(trainable),holdout_questions=len(items)-len(trainable),source_unit_checks=len(items))
    audit = v4_audit(out,allitems) if write else None
    required = {a for i in allitems for a in i['sec']['required_accessions']}
    manifest = {'schema':V4_SCHEMA,'status':('complete' if not shortages else ('complete-with-declared-shortfall' if write else 'incomplete')),'months':months,'seed':args.seed,
                'questions_per_month_target':per_month,'families':list(families),'allow_shortfall':bool(getattr(args,'allow_shortfall',False)),'family_targets_per_month':dict.fromkeys(families,quota),
                'max_questions_per_filing':2,'max_candidates_per_month':max_candidates,
                'shortages':shortages,'batches':batches,'gold_audit':audit,'exclusions':exclusions,
                'source_ambiguity':{'by_family':dict(Counter(e.get('family','filing_or_dependency') for e in exclusions)),
                                    'by_reason':dict(Counter(e['reason'] for e in exclusions)),
                                    'note':'Exclusions include unsupported templates and intentional reuse exclusions; inspect reasons to distinguish source ambiguity.'},
                'holdout':{'salt':V3_HOLDOUT_SALT,'co_registrants':'connected components across full lookup indexes; any member held out holds out group'},
                'filings':[records[a] for a in records if a in required],
                'required_filings':[loaded[a]['record'] for a in sorted(required)],
                'v3_exclusion_hashes':exclusion_hashes,'selection':'fixed hash order; no model scores',
                'explicit_unit_score':'absolute error <=0.05 in requested percent/pp; conflicting unit suffix rejected; no x100/div100 tolerance',
                'memory_admission':'earlier frozen own-correct under BOTH historical and explicit-unit scores; k=3',
                'decision_policy':'accuracy, mixed-outcome families, yield and ambiguity guide redesign; no automatic kill bands',
                'sources':[{'path':p,'sha256':h} for p,h in sorted(source.read.items())],
                'script_sha256':sha256(Path(__file__).read_bytes()),'finqa_bed_sha256':sha256((REPO/'kit/beds/finqa.py').read_bytes()),
                'fetch':{'requests':source.requests,'user_agent_set':bool(os.environ.get('SEC_USER_AGENT','').strip()),'min_interval_seconds':MIN_INTERVAL}}
    if amendment_rule == 'restating':
        manifest.update(amendment_rule=amendment_rule,
                        amendment_counts={m:amendment_counts(m) for m in months},
                        amendment_counting='Unique amendment accessions per scan month, including join sources; unreadable then restating takes precedence if examined against multiple original periods. Default any omits these fields to preserve registered bytes.')
    if feasibility:
        manifest.update(purpose='4a-feasibility-supply-scan', status='feasibility-only',
                        month_supply=month_supply, probe_exclusion=probe, lookup_indexes=list(indexes),
                        annual_report_policy={f: '10-K filings are supported using matched current/prior-year durations, including 12-month annual columns; income and cash-flow operands must have identical exact fiscal boundaries. Missing/ambiguous bindings, mismatched durations and all existing v4 accounting/sign/denominator guards exclude the affected family. Formulas are unchanged.'
                                              for f in ('fcf_margin_change', 'cash_conversion_change')})
    (out/'manifest.json').write_text(json.dumps(manifest,indent=1,sort_keys=True)+'\n')
    print('wrote',out/'manifest.json',manifest['status'],'shortages:',shortages)
    return 2 if shortages else 0


def write_batch(out: Path, name: str, items: list, finqa, trainer_items=None) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    text = json.dumps(items, indent=1, ensure_ascii=False) + "\n"
    (out / ("%s.json" % name)).write_text(text, encoding="utf-8")
    rows = finqa.rows_for_trainer(items, name)
    export_rows = rows if trainer_items is None else finqa.rows_for_trainer(trainer_items, name)
    jsonl = "".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in export_rows)
    (out / ("%s.trainer.jsonl" % name)).write_text(jsonl, encoding="utf-8")
    return {"questions": len(items), "filings": len({i["sec"]["accession"] for i in items}),
            "gold_consistent": sum(r["extra_info"]["gold_consistent"] for r in rows),
            "templates": {t: sum(i["sec"]["template"] == t for i in items) for t in sorted({i["sec"]["template"] for i in items})},
            "finqa_json_sha256": sha256(text), "trainer_jsonl_sha256": sha256(jsonl),
            "ids_sha256": sha256("".join(i["id"] + "\n" for i in items))}


def cmd_build(args) -> int:
    if args.version == 4:
        return cmd_build_v4(args)
    if args.version == 3:
        return cmd_build_v3(args)
    finqa = load_kit("beds/finqa")
    n = args.questions_per_filing
    if args.sample:
        plan = json.loads((SAMPLE_DIR / "filings.json").read_text())
        source = SecSource(SAMPLE_DIR / "cache", allow_fetch=False)
        out = Path(args.out or SAMPLE_DIR / "out")
        months = sorted({f["month"] for f in plan["filings"]})
        per_month = {m: [f for f in plan["filings"] if f["month"] == m] for m in months}
        amendments = {}
    else:
        source = SecSource(Path(args.cache), allow_fetch=not args.offline)
        out = Path(args.out)
        months = args.months.split(",")
        per_month, amendments = {}, {}
        for m in months:
            ordered, amended = candidates_for_month(source, m, args.seed)
            per_month[m] = [{"cik": r["cik"], "accession": r["accession"], "month": m, "co_filers": r["co_filers"]} for r in ordered[:MAX_CANDIDATES]]
            amendments.update(amended)
    batches, exclusions, filings = {}, [], []
    V2_USAGE.clear()
    for m in months:
        items, accepted = [], 0
        for f in per_month[m]:
            if accepted == (len(per_month[m]) if args.sample else args.per_month):
                break
            made, reason, record = (build_filing_v2 if args.version == 2 else build_filing)(source, f["cik"], f["accession"], n)
            if record and not record.get("filing_date", "").startswith(m):
                reason = reason or "filing date %s is outside %s" % (record.get("filing_date"), m)
                made = []
            if reason:
                exclusions.append({"month": m, "cik": f["cik"], "accession": f["accession"], "reason": reason,
                                   **{k: record.get(k) for k in ("company", "form", "filing_date")}})
                continue
            accepted += 1
            items += made
            later = [a for a in amendments.get(str(int(f["cik"])), []) if a["date"] >= record["filing_date"]]
            filings.append({"month": m, **{k: record[k] for k in ("cik", "accession", "company", "form", "filing_date", "accepted", "period_of_report")},
                            "group": "%s:%s" % (record["cik"], record["period_of_report"]), "co_filers": f.get("co_filers", []),
                            "amendments_seen": later})
        batches[m] = write_batch(out, "batch-" + m, items, finqa)
        print("%s: %d filings, %d questions" % (m, batches[m]["filings"], batches[m]["questions"]))
    manifest = {"schema": V2_SCHEMA if args.version == 2 else SCHEMA, "mode": "sample" if args.sample else "slice", "months": months, "seed": args.seed,
                "per_month_target": None if args.sample else args.per_month, "questions_per_filing": n, "cutoff": str(CUTOFF),
                "forms": list(FORMS), "script_sha256": sha256(Path(__file__).read_bytes()),
                "finqa_bed_sha256": sha256((REPO / "kit" / "beds" / "finqa.py").read_bytes()),
                "finqa_instruction_sha256": sha256(finqa.INSTRUCTION), "batches": batches, "filings": filings, "exclusions": exclusions,
                "sources": [{"path": p, "sha256": h} for p, h in sorted(source.read.items())],
                "fetch": {"requests": source.requests, "user_agent_set": bool(os.environ.get("SEC_USER_AGENT", "").strip()),
                          "min_interval_seconds": MIN_INTERVAL}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print("wrote %s (%d questions, %d exclusions, %d requests)" % (out / "manifest.json", sum(b["questions"] for b in batches.values()),
                                                                  len(exclusions), source.requests))
    return 0


# ------------------------------------------------------------------------------------- dry run
def cmd_dryrun(args) -> int:
    """The existing FinQA scorer on the slice with trivial stand-in answers: proves the format, measures nothing."""
    eval_bed = load_kit("eval_bed")
    finqa = eval_bed.load_bed("finqa")
    slice_dir = Path(args.slice)
    bed_args = argparse.Namespace(bed="finqa")
    report = {"schema": SCHEMA + ".dryrun", "scorer": eval_bed.scorer_identity("finqa"), "batches": {}}
    for path in sorted(slice_dir.glob("batch-*.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        # the same three fields eval_bed.items_of builds for FinQA, from the same bed functions
        items = [{"id": item["id"], "prompt": finqa.render_prompt(item), "ground_truth": finqa.gold_of(item)} for item in raw]
        standins = {"empty": {i["id"]: "" for i in raw},
                    "answer_zero": {i["id"]: "Answer: 0" for i in raw},
                    "echo_executed_gold": {i["id"]: "Answer: %s" % finqa.gold_of(i) for i in raw},
                    "echo_displayed_answer": {i["id"]: "Some working.\nAnswer: %s" % i["qa"]["answer"] for i in raw},
                    # two near misses the scorer must refuse: the wrong sign, and the right digits in the wrong unit
                    "negated_gold": {i["id"]: "Answer: %r" % -float(finqa.gold_of(i)) for i in raw},
                    "gold_times_1000": {i["id"]: "Answer: %r" % (1000 * float(finqa.gold_of(i))) for i in raw}}
        result = {"n": len(items), "gold_consistent": sum(finqa.gold_consistent(i) for i in raw),
                  "items_sha256": eval_bed.items_digest(items),
                  "prompt_chars_max": max(len(i["prompt"]) for i in items)}
        for name, answers in standins.items():
            graded = eval_bed.grade(finqa, bed_args, items, answers)
            result[name] = {k: graded[k] for k in ("n", "correct", "incorrect_format", "accuracy")}
        print(path.stem, json.dumps({k: (v["correct"] if isinstance(v, dict) else v) for k, v in result.items()}))
        responses = slice_dir / "dryrun-cli" / ("%s.responses.jsonl" % path.stem)
        responses.parent.mkdir(parents=True, exist_ok=True)
        responses.write_text("".join(json.dumps({"id": i["id"], "response": "Answer: %s" % i["qa"]["answer"]}) + "\n" for i in raw), encoding="utf-8")
        cli_out = slice_dir / "dryrun-cli" / path.stem
        if cli_out.exists():                      # a regenerated dry run; eval_bed itself never overwrites
            import shutil                                                                     # noqa: PLC0415
            shutil.rmtree(cli_out)
        run_eval_bed(slice_dir, path.stem[len("batch-"):], "score", ["--responses", str(responses), "--out", str(cli_out)])
        cli = json.loads((cli_out / "bed-score.json").read_text())
        result["eval_bed_cli_score"] = {k: cli[k] for k in ("n", "correct", "incorrect_format", "accuracy", "scorer_sha256", "items_sha256")}
        report["batches"][path.stem] = result
    (slice_dir / "dryrun.json").write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print("wrote", slice_dir / "dryrun.json")
    return 0


# ------------------------------------------------------------------------------------- eval_bed, unchanged
def run_eval_bed(slice_dir: str, batch: str, mode: str, rest: list) -> int:
    """kit/eval_bed.py `generate` or `score` on one monthly batch, with no kit file changed (a change to finqa.py or
    eval_bed.py would move `scorer_sha256` for every FinQA scoring). The FinQA bed's loader is pinned to the
    published dataset's split sizes, so only its `load` is pointed at the batch file; prompts, gold, scoring and the
    result file are eval_bed's own. eval_bed records the split as `test`: the batch is named by the --out path."""
    if str(REPO) not in sys.path:                 # eval_bed's generate imports kit.v4_teacher
        sys.path.insert(0, str(REPO))
    eval_bed = load_kit("eval_bed")
    original = eval_bed.load_bed

    def load_bed(name):
        module = original(name)
        if name == "finqa":
            module.load = lambda root, split: json.loads((Path(root) / ("batch-%s.json" % batch)).read_text(encoding="utf-8"))
        return module

    eval_bed.load_bed = load_bed
    try:
        result = eval_bed.main([mode, "--bed", "finqa", "--root", str(slice_dir), "--split", "test", *rest])
        batch_path = Path(slice_dir) / ("batch-%s.json" % batch)
        raw = json.loads(batch_path.read_text())
        if result == 0 and raw and raw[0]["sec"].get("version") == 4:
            score_path = Path(rest[rest.index("--out") + 1]) / "bed-score.json"
            score = json.loads(score_path.read_text())
            score["sec_v4_input_batch_sha256"] = sha256(batch_path.read_bytes())
            score_path.write_text(json.dumps(score, indent=1, sort_keys=True) + "\n")
        return result
    finally:
        eval_bed.load_bed = original


# ------------------------------------------------------------------------------------- frozen plus memory
# The memory arm (4A's required baseline, here at feasibility scale). For an item of batch m the model sees the SAME
# prompt as the frozen arm, with k solved cases inserted between the instruction and the report excerpt. A case is an
# item from an EARLIER batch that the frozen model answered correctly, stored as the deployed system would have stored
# it: the earlier question, the table rows it used, and the model's own verified solution (not the gold program, which
# would turn memory into demonstrations). Retrieval is deterministic: same template first, then word overlap of the
# operand labels, then same form, ties by id. The first batch has no earlier batch, so its memory arm IS the frozen arm.
MEMORY_K = 3
MEMORY_CASE_CHARS = 2500        # a stored solution longer than this is not kept: bounded, comparable context
MEMORY_HEADER = ("Solved examples from earlier filings (questions this assistant answered before, each checked "
                 "correct). Use them only as worked examples; the report excerpt below is a different filing.")


def label_words(item: dict) -> set:
    return {w for o in item["sec"]["operands"] for w in re.findall(r"[a-z]+", o["label"].lower()) if len(w) > 2}


def similarity(item: dict, case: dict) -> float:
    a, b = label_words(item), label_words(case)
    overlap = len(a & b) / len(a | b) if a | b else 0.0
    return 2.0 * (item["sec"]["template"] == case["sec"]["template"]) + overlap + 0.25 * (item["sec"]["form"] == case["sec"]["form"])


def case_text(case: dict, response: str, finqa) -> str:
    if case["sec"].get("version") == 4:
        return v4_case_text(case, response, finqa)
    labels = {o["label"] for o in case["sec"]["operands"]}
    rows = [case["table"][0]] + [r for r in case["table"][1:] if r[0].split(" [")[0] in labels]
    return "\n".join(["%s, Form %s filed %s; amounts in %s." % (case["sec"]["company"], case["sec"]["form"], case["sec"]["filing_date"], case["sec"]["unit"]),
                      finqa.render_table(rows), "Question: " + case["qa"]["question"], "Solution:", response.strip()])


def memory_pool(slice_dir: Path, runs: Path, batches: list) -> list:
    """Correct frozen answers of `batches`: [(item, response)]. The frozen result must be for exactly these items."""
    eval_bed = load_kit("eval_bed")
    pool = []
    for b in batches:
        items = json.loads((slice_dir / ("batch-%s.json" % b)).read_text(encoding="utf-8"))
        score = json.loads((runs / ("frozen-%s" % b) / "bed-score.json").read_text())
        if score["items_sha256"] != eval_bed.items_digest(items):
            raise SystemExit("frozen-%s was not scored on this slice's batch %s" % (b, b))
        response_path = runs / ("frozen-%s" % b) / "responses.jsonl"
        if items and items[0]["sec"].get("version") == 4:
            assert score["sec_v4_input_batch_sha256"] == sha256((slice_dir / ("batch-%s.json" % b)).read_bytes()), "frozen input batch changed"
            assert score["responses_sha256"] == sha256(response_path.read_bytes()), "frozen responses changed"
        responses = {json.loads(l)["id"]: json.loads(l)["response"] for l in response_path.read_text().split('\n') if l.strip()}
        pool += [(i, responses[i["id"]]) for i in items
                 if score["per_item"][i["id"]] == 1 and len(responses[i["id"]].strip()) <= MEMORY_CASE_CHARS
                 and not i["sec"].get("issuer_holdout", False) and i["sec"].get("training_allowed", True)
                 and (i["sec"].get("version") != 4 or v4_explicit_correct(i, responses[i["id"]]))]
    return pool


def build_memory_batch(slice_dir: Path, runs: Path, batch: str, out: Path, k: int = MEMORY_K) -> dict:
    finqa = load_kit("beds/finqa")
    batches = sorted(p.stem[len("batch-"):] for p in slice_dir.glob("batch-*.json"))
    earlier = [b for b in batches if b < batch]
    if not earlier:
        raise SystemExit("batch %s has no earlier batch: its memory arm is the frozen arm" % batch)
    pool = memory_pool(slice_dir, runs, earlier)
    items = json.loads((slice_dir / ("batch-%s.json" % batch)).read_text(encoding="utf-8"))
    if items and items[0]["sec"].get("version") == 4 and k != MEMORY_K:
        raise SystemExit("v4 memory is fixed at k=3")
    out.mkdir(parents=True, exist_ok=True)
    built, lengths = [], []
    for item in items:
        frozen = finqa.render_prompt(item)
        assert frozen.startswith(finqa.INSTRUCTION)
        eligible = pool
        if "issuer_ciks" in item["sec"]:
            eligible = [(c, r) for c, r in pool if
                        not (set(c["sec"]["issuer_ciks"]) & set(item["sec"]["issuer_ciks"]))
                        and c["sec"]["filing_group"] != item["sec"]["filing_group"]
                        and c["sec"]["accession"] != item["sec"]["accession"]
                        and not (set(c["sec"].get("required_accessions", [])) & set(item["sec"].get("required_accessions", [])))
                        and not (set(c["sec"].get("required_filing_groups", [])) & set(item["sec"].get("required_filing_groups", [])))]
        cases = sorted(eligible, key=lambda c: (-similarity(item, c[0]), c[0]["id"]))[:k]
        block = "\n\n".join("Example %d\n%s" % (n, case_text(c, r, finqa)) for n, (c, r) in enumerate(cases, 1))
        prompt = frozen if not cases else finqa.INSTRUCTION + "\n\n" + MEMORY_HEADER + "\n\n" + block + frozen[len(finqa.INSTRUCTION):]
        built.append({**item, "messages": [{"role": "user", "content": prompt}],
                      "memory": {"k": k, "source": "frozen arm's correct answers, earlier batches only", "earlier_batches": earlier,
                                 "case_ids": [c["id"] for c, _ in cases]}})
        lengths.append(len(prompt) - len(frozen))
    text = json.dumps(built, indent=1, ensure_ascii=False) + "\n"
    (out / ("batch-%s.json" % batch)).write_text(text, encoding="utf-8")
    info = {"batch": batch, "earlier_batches": earlier, "pool": len(pool), "k": k, "items": len(built),
            "items_with_k_cases": sum(len(i["memory"]["case_ids"]) == k for i in built),
            "added_chars_mean": round(sum(lengths) / len(lengths), 1), "added_chars_max": max(lengths),
            "batch_sha256": sha256(text), "case_chars_cap": MEMORY_CASE_CHARS}
    (out / ("memory-%s.json" % batch)).write_text(json.dumps(info, indent=1, sort_keys=True) + "\n")
    print("memory %s: pool %d correct earlier answers, %d of %d items got %d cases, +%.0f chars per prompt on average"
          % (batch, len(pool), info["items_with_k_cases"], len(built), k, info["added_chars_mean"]))
    return info


# ------------------------------------------------------------------------------------- readout
def v3_arm_metrics(items: list, score: dict, rows: list) -> dict:
    """Numerical signatures are diagnostic hypotheses, not claims about the model's reasoning."""
    finqa = load_kit("beds/finqa")
    assert set(score["per_item"]) == {i["id"] for i in items}
    assert len(rows) == len(items) and {r["id"] for r in rows} == set(score["per_item"])
    responses = {r["id"]: r["response"] for r in rows}
    families = {f: {"n": 0, "correct": 0, "accuracy": None, "eligible_traces": 0} for f in V3_FAMILIES}
    errors, examples, holdout = Counter(), [], {"n": 0, "correct": 0}
    for item in items:
        s, response = item["sec"], responses[item["id"]]
        correct = int(finqa.compute_score(finqa.DATA_SOURCE, response, finqa.gold_of(item))["acc"])
        assert correct == score["per_item"][item["id"]], "response disagrees with recorded score"
        f = families[s["template"]]
        f["n"] += 1
        f["correct"] += correct
        f["eligible_traces"] += int(correct and not s["issuer_holdout"] and s["training_allowed"]
                                    and len(response.strip()) <= MEMORY_CASE_CHARS)
        if s["issuer_holdout"]:
            holdout["n"] += 1
            holdout["correct"] += correct
        if correct:
            continue
        pred, gold = finqa.extract_answer(response), finqa.gold_of(item)
        if pred is None:
            error = "no_final_answer"
        elif isinstance(pred, str):
            error = "nonnumeric_answer"
        elif finqa.is_correct(-pred, gold):
            error = ("reversed_percentage_point_subtraction" if s["template"] in ("margin_pp_change", "relative_growth")
                     else "cash_flow_sign_or_change_direction" if s["template"] == "free_cash_flow_change" else "sign_or_change_direction")
        elif any(finqa.is_correct(float(pred) / scale, gold) for scale in (1000, 0.001, 1000000, 0.000001)):
            error = "possible_unit_scale_error"
        else:
            error = "other_arithmetic_or_source_reading"
        errors[error] += 1
        examples.append({"id": item["id"], "family": s["template"], "type": error})
    for f in families.values():
        f["accuracy"] = f["correct"] / f["n"] if f["n"] else None
    return {"per_family": families,
            "trace_supply": {"verified_correct": sum(f["correct"] for f in families.values()),
                             "eligible_own_correct": sum(f["eligible_traces"] for f in families.values()),
                             "per_family": {k: {"verified_correct": v["correct"], "eligible_own_correct": v["eligible_traces"]}
                                            for k, v in families.items()}},
            "holdout": {**holdout, "accuracy": holdout["correct"] / holdout["n"] if holdout["n"] else None},
            "error_types": dict(sorted(errors.items())), "error_items": examples,
            "error_type_note": "numerical signatures; manually review source/period/unit errors in responses; scorer accepts x100 and /100"}


def v4_metrics(items, rows, prompt_rows):
    finqa = load_kit('beds/finqa')
    byid = {r['id']:r for r in rows}
    assert len(byid) == len(rows) == len(items) and set(byid) == {i['id'] for i in items}
    prompts = {r['id']:r.get('prompt_tokens') for r in prompt_rows}
    families, predictions = {}, {}
    for family in V4_FAMILIES:
        subset = [i for i in items if i['sec']['template']==family]
        errors, error_items, c, explicit, eligible = Counter(), [], 0, 0, 0
        for i in subset:
            r, s = byid[i['id']],i['sec']; response = r['response']
            old = int(finqa.compute_score(finqa.DATA_SOURCE,response,finqa.gold_of(i))['acc'])
            strict = int(v4_explicit_correct(i,response)); c += old; explicit += strict
            eligible += int(old and strict and s['training_allowed'] and not s['issuer_holdout'] and len(response.strip())<=MEMORY_CASE_CHARS)
            predictions[i['id']] = {'historical':old,'explicit_unit':strict}
            if not strict:
                pred = finqa.extract_answer(response)
                if pred is None:
                    error = 'no_final_answer'
                elif isinstance(pred,str):
                    error = 'nonnumeric_answer'
                elif old:
                    error = 'historical_only_unit_or_precision_error'
                elif v4_explicit_correct(i,'Answer: '+str(-float(pred))):
                    error = 'sign_or_change_direction'
                elif any(v4_explicit_correct(i,'Answer: '+str(float(pred)*scale)) for scale in (100,.01,1000,.001,1e6,1e-6)):
                    error = 'possible_unit_scale_error'
                else:
                    error = 'other_arithmetic_or_source_reading'
                errors[error] += 1; error_items.append({'id':i['id'],'type':error})
        n = len(subset)
        output = sum(byid[i['id']]['output_tokens'] for i in subset) if n and all(byid[i['id']].get('output_tokens') is not None for i in subset) else None
        prompt = sum(prompts[i['id']] for i in subset) if n and all(prompts.get(i['id']) is not None for i in subset) else None
        families[family] = {'n':n,'correct':c,'accuracy':c/n if n else None,
                            'historical_finqa':{'correct':c,'accuracy':c/n if n else None},
                            'explicit_unit':{'correct':explicit,'accuracy':explicit/n if n else None},
                            'eligible_own_correct_traces':eligible,'error_types':dict(errors),'error_items':error_items,
                            'output_tokens':output,'prompt_tokens':prompt,
                            'output_tokens_per_correct':output/c if output is not None and c else None,
                            'all_tokens_per_correct':(output+prompt)/c if output is not None and prompt is not None and c else None,
                            'explicit_unit_output_tokens_per_correct':output/explicit if output is not None and explicit else None,
                            'explicit_unit_tokens_per_correct':(output+prompt)/explicit if output is not None and prompt is not None and explicit else None}
    n,c,strict = len(items),sum(f['correct'] for f in families.values()),sum(f['explicit_unit']['correct'] for f in families.values())
    outputs = sum(f['output_tokens'] for f in families.values()) if all(f['output_tokens'] is not None for f in families.values()) else None
    prompts_total = sum(f['prompt_tokens'] for f in families.values()) if all(f['prompt_tokens'] is not None for f in families.values()) else None
    return {'n':n,'correct':c,'accuracy':c/n if n else None,'no_final_answer':sum(f['error_types'].get('no_final_answer',0) for f in families.values()),
            'explicit_unit':{'correct':strict,'accuracy':strict/n if n else None},'per_family':families,
            'per_template':{f:'%d/%d'%(v['correct'],v['n']) for f,v in families.items()},
            'output_tokens_per_correct':outputs/c if outputs is not None and c else None,
            'all_tokens_per_correct':(outputs+prompts_total)/c if outputs is not None and prompts_total is not None and c else None,
            'trace_supply':{'verified_correct':c,'eligible_own_correct':sum(f['eligible_own_correct_traces'] for f in families.values())},
            'mixed_outcome_families':{sc:[f for f,v in families.items() if 0<(v['correct'] if sc=='historical' else v['explicit_unit']['correct'])<v['n']]
                                      for sc in ('historical','explicit_unit')},
            'error_types':dict(sum((Counter(v['error_types']) for v in families.values()),Counter())),
            'error_type_note':'Numerical signatures only; manually classify reasoning, period and source errors.',
            'predictions':predictions}


def v4_paired(items, frozen, memory):
    result = {}
    for family in V4_FAMILIES:
        ids = [i['id'] for i in items if i['sec']['template']==family and i['id'] in frozen['predictions'] and i['id'] in memory['predictions']]
        result[family] = {score:{'n':len(ids),
                                'gains':sum(not frozen['predictions'][i][score] and memory['predictions'][i][score] for i in ids),
                                'losses':sum(frozen['predictions'][i][score] and not memory['predictions'][i][score] for i in ids)}
                          for score in ('historical','explicit_unit')}
    return result


def readout_v4(slice_dir, runs):
    manifest = json.loads((slice_dir/'manifest.json').read_text())
    assert manifest['status'] in ('complete','complete-with-declared-shortfall'), 'v4 build incomplete'
    result, allitems, pooled, pooled_prompts = {}, [], {'frozen':[],'memory':[]}, {'frozen':[],'memory':[]}
    memoryitems, frozenitems = [], []
    for month in manifest['months']:
        items = json.loads((slice_dir/('batch-'+month+'.json')).read_text()); allitems.extend(items)
        result[month] = {}
        for arm in ('frozen','memory'):
            path = runs/(arm+'-'+month)
            if not (path/'bed-score.json').is_file():
                result[month][arm] = '= frozen (no earlier batch)' if arm=='memory' and month==manifest['months'][0] else None
                continue
            score = json.loads((path/'bed-score.json').read_text())
            rows = [json.loads(l) for l in (path/'responses.jsonl').read_text().split('\n') if l.strip()]
            prompts = [json.loads(l) for l in (path/'tokens.jsonl').read_text().split('\n') if l.strip()] if (path/'tokens.jsonl').is_file() else []
            assert score['responses_sha256']==sha256((path/'responses.jsonl').read_bytes()), 'responses changed since scoring'
            if arm=='frozen':
                assert score['sec_v4_input_batch_sha256']==sha256((slice_dir/('batch-'+month+'.json')).read_bytes()), 'frozen input changed'
                frozenitems.extend(items)
                assert score['items_sha256']==load_kit('eval_bed').items_digest(items), 'frozen result belongs to another slice'
            else:
                # Verify memory score provenance against its actual prompted batch retained by the launcher.
                memory_file = runs.parent/'memory'/('batch-'+month+'.json')
                assert memory_file.is_file(), 'memory prompted batch is required for provenance'
                prompted = json.loads(memory_file.read_text())
                assert score['sec_v4_input_batch_sha256']==sha256(memory_file.read_bytes()), 'memory input changed'
                assert score['items_sha256']==load_kit('eval_bed').items_digest(prompted), 'memory result belongs to another prompted batch'
                assert {i['id'] for i in prompted}=={i['id'] for i in items}
                for p,i in zip(sorted(prompted,key=lambda i:i['id']),sorted(items,key=lambda i:i['id'])):
                    assert all(p[k]==i[k] for k in ('qa','sec','pre_text','table','post_text')), 'memory current documents changed'
            metrics = v4_metrics(items,rows,prompts)
            assert {i: p['historical'] for i,p in metrics['predictions'].items()}==score['per_item']
            result[month][arm] = metrics; pooled[arm].extend(rows); pooled_prompts[arm].extend(prompts)
            if arm=='memory':
                memoryitems.extend(items)
        if all(isinstance(result[month][a],dict) for a in ('frozen','memory')):
            result[month]['paired'] = v4_paired(items,result[month]['frozen'],result[month]['memory'])
    result['overall'] = {arm:v4_metrics(frozenitems if arm=='frozen' else memoryitems,pooled[arm],pooled_prompts[arm])
                         for arm in ('frozen','memory') if pooled[arm]}
    for arm in ('frozen','memory'):
        result['overall'].setdefault(arm,None)
    if all(isinstance(result['overall'][a],dict) for a in ('frozen','memory')):
        paired = v4_paired(memoryitems,result['overall']['frozen'],result['overall']['memory'])
        result['overall']['paired'] = paired
        result['overall']['frozen']['paired_later_months'] = paired
    result['overall']['completed_responses'] = {a:len(pooled[a]) for a in ('frozen','memory')}
    result['overall']['source_ambiguity'] = manifest.get('source_ambiguity',{})
    result['overall']['decision_policy'] = 'Diagnostics guide redesign; no automatic kill bands.'
    return result


def readout(slice_dir: Path, runs: Path) -> dict:
    """Per batch: frozen against memory -- correct, output tokens per correct answer, prompt tokens, per-template accuracy."""
    if (slice_dir / 'manifest.json').is_file() and json.loads((slice_dir / 'manifest.json').read_text()).get('schema') == V4_SCHEMA:
        return readout_v4(slice_dir, runs)
    def arm(path: Path, items: list):
        if not (path / "bed-score.json").is_file():
            return None
        score = json.loads((path / "bed-score.json").read_text())
        rows = [json.loads(l) for l in (path / "responses.jsonl").read_text().split('\n') if l.strip()] if (path / "responses.jsonl").is_file() else []
        out_tokens = sum(r.get("output_tokens") or 0 for r in rows) if rows and all("output_tokens" in r for r in rows) else None
        prompt_tokens = None
        if (path / "tokens.jsonl").is_file():
            toks = [json.loads(l) for l in (path / "tokens.jsonl").read_text().split('\n') if l.strip()]
            prompt_tokens = sum(t.get("prompt_tokens") or 0 for t in toks) if toks and all(t.get("prompt_tokens") is not None for t in toks) else None
        per_template = {}
        for i in items:
            t = per_template.setdefault(i["sec"]["template"], [0, 0])
            t[0] += score["per_item"][i["id"]]
            t[1] += 1
        c = score["correct"]
        result = {"n": score["n"], "correct": c, "accuracy": score["accuracy"], "no_final_answer": score["incorrect_format"],
                "truncated": score.get("truncated_at_max_tokens"), "output_tokens": out_tokens, "prompt_tokens": prompt_tokens,
                "output_tokens_per_correct": round(out_tokens / c, 1) if out_tokens is not None and c else None,
                "all_tokens_per_correct": round((out_tokens + prompt_tokens) / c, 1) if out_tokens is not None and prompt_tokens is not None and c else None,
                "per_template": {t: "%d/%d" % tuple(v) for t, v in sorted(per_template.items())}}
        if items and "issuer_holdout" in items[0]["sec"]:
            if path.name.startswith("frozen-"):
                assert score["items_sha256"] == load_kit("eval_bed").items_digest(items), "frozen result is for another slice"
            result.update(v3_arm_metrics(items, score, rows))
        return result
    result = {}
    for path in sorted(slice_dir.glob("batch-*.json")):
        b = path.stem[len("batch-"):]
        items = json.loads(path.read_text(encoding="utf-8"))
        frozen = arm(runs / ("frozen-%s" % b), items)
        memory = arm(runs / ("memory-%s" % b), items)
        result[b] = {"frozen": frozen, "memory": memory if memory else ("= frozen (no earlier batch)" if b == min(p.stem[6:] for p in slice_dir.glob("batch-*.json")) else None)}
    if any(isinstance(a["frozen"], dict) and "trace_supply" in a["frozen"] for a in result.values()):
        overall = {}
        for name in ("frozen", "memory"):
            arms = [a[name] for a in result.values() if isinstance(a[name], dict)]
            families = {f: {k: sum(a["per_family"][f][k] for a in arms) for k in ("n", "correct", "eligible_traces")}
                        for f in V3_FAMILIES}
            for f in families.values():
                f["accuracy"] = f["correct"] / f["n"] if f["n"] else None
            n, c = sum(a["n"] for a in arms), sum(a["correct"] for a in arms)
            output = sum(a["output_tokens"] for a in arms) if arms and all(a["output_tokens"] is not None for a in arms) else None
            prompt = sum(a["prompt_tokens"] for a in arms) if arms and all(a["prompt_tokens"] is not None for a in arms) else None
            overall[name] = {"n": n, "correct": c, "accuracy": c / n if n else None, "per_family": families,
                             "trace_supply": {"verified_correct": c, "eligible_own_correct": sum(f["eligible_traces"] for f in families.values()),
                                              "per_family": {k: {"verified_correct": v["correct"], "eligible_own_correct": v["eligible_traces"]}
                                                             for k, v in families.items()}},
                             "error_types": dict(sum((Counter(a["error_types"]) for a in arms), Counter())),
                             "output_tokens_per_correct": round(output / c, 1) if output is not None and c else None,
                             "all_tokens_per_correct": round((output + prompt) / c, 1) if output is not None and prompt is not None and c else None}
        result["overall"] = overall
    return result


def cmd_readout(args) -> int:
    result = readout(Path(args.slice), Path(args.runs))
    (Path(args.runs) / "readout.json").write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    print("| batch | arm | correct | accuracy | output tokens per correct | prompt+output tokens per correct | per template |")
    print("|---|---|---|---|---|---|---|")
    for b, arms in result.items():
        for name in ("frozen", "memory"):
            r = arms[name]
            if isinstance(r, dict):
                if b == "overall":
                    print("overall %s: %s" % (name, json.dumps(r, sort_keys=True)))
                    continue
                print("| %s | %s | %d/%d | %.3f | %s | %s | %s |" % (b, name, r["correct"], r["n"], r["accuracy"], r["output_tokens_per_correct"],
                                                                 r["all_tokens_per_correct"], ", ".join("%s %s" % kv for kv in r["per_template"].items())))
                if "trace_supply" in r:
                    print("trace supply %s %s: %s; errors: %s" % (b, name, json.dumps(r["trace_supply"], sort_keys=True), json.dumps(r["error_types"], sort_keys=True)))
            else:
                print("| %s | %s | %s |  |  |  |  |" % (b, name, r or "not run"))
    print("wrote", Path(args.runs) / "readout.json")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="action", required=True)
    b = sub.add_parser("build")
    b.add_argument("--sample", action="store_true", help="offline, on the hand-saved sample in data/p4_sec/sample")
    b.add_argument("--months", default=None, help="comma-separated filing months; v3/v4 default to 2026-07,2026-08,2026-09")
    b.add_argument("--dry-run", action="store_true", help="v3/v4 build plan only: no fetching or writing")
    b.add_argument("--per-month", type=int, default=None, help="filings per monthly batch")
    b.add_argument("--questions-per-filing", type=int, default=2)
    b.add_argument("--seed", default="4a0-slice-v1")
    b.add_argument("--cache", default=str(REPO / "data" / "cache" / "sec"))
    b.add_argument("--offline", action="store_true", help="never fetch; fail on anything not cached")
    b.add_argument("--out", default=None)
    b.add_argument("--version", type=int, choices=(1, 2, 3, 4), default=1, help="1: single-step income questions; 2: composed statements; "
                   "3: fresh balanced recurring families; 4: chained 60-question feasibility probe")
    b.add_argument("--exclude-v3", default=str(REPO / "data/p4_sec/slice-v3"), help="complete frozen v3 slice to exclude, including join filings")
    b.add_argument("--families", help="v4 only: comma-separated subset of families to build")
    b.add_argument("--feasibility", action="store_true", help="v4 only: allow other complete months for a supply scan; exclude cash/scale probes and mark feasibility-only")
    b.add_argument("--amendment-rule", choices=('any', 'restating'), default='any',
                   help="v4 only: reject any later amendment (default, registered bytes), or only relevant restating/unreadable amendments")
    b.add_argument("--allow-shortfall", action="store_true", help="v4 only: write the batches found and record shortages")
    b.add_argument("--per-family", type=int, help="v4 only: questions per family per month (default 4)")
    b.add_argument("--max-candidates", type=int, help="v4 only: hash-ordered candidate filings scanned per month (default 240)")
    d = sub.add_parser("dryrun")
    d.add_argument("--slice", required=True)
    e = sub.add_parser("eval", help="kit/eval_bed.py generate|score on one batch; other arguments pass through "
                                    "(generate: --model --out; score: --responses --out)")
    e.add_argument("mode", choices=("generate", "score"))
    e.add_argument("--slice", required=True)
    e.add_argument("--batch", required=True, help="e.g. 2025-06")
    mm = sub.add_parser("memory", help="frozen-plus-memory batch for one month, from the frozen run's correct answers")
    mm.add_argument("--slice", required=True)
    mm.add_argument("--runs", required=True, help="directory holding frozen-<batch>/ results of the earlier batches")
    mm.add_argument("--batch", required=True)
    mm.add_argument("--out", required=True)
    mm.add_argument("--k", type=int, default=MEMORY_K)
    r = sub.add_parser("readout", help="frozen against memory, per batch")
    r.add_argument("--slice", required=True)
    r.add_argument("--runs", required=True)
    args, rest = parser.parse_known_args(argv)
    if args.action == "eval":
        return run_eval_bed(args.slice, args.batch, args.mode, rest)
    if rest:
        parser.error("unrecognized arguments: %s" % " ".join(rest))
    if args.action == "build":
        if args.feasibility and args.version != 4:
            parser.error('build --feasibility is for --version 4')
        if args.amendment_rule != 'any' and args.version != 4:
            parser.error('build --amendment-rule restating is for --version 4')
        args.per_month = args.per_month if args.per_month is not None else (10 if args.version == 4 else 20)
        args.months = args.months or ",".join(V3_MONTHS if args.version >= 3 else DEFAULT_MONTHS)
        if args.dry_run and args.version < 3:
            parser.error("build --dry-run is for --version 3/4")
        if not args.sample and not args.out:
            args.out = str(REPO / "data" / "p4_sec" / ('4a-feasibility-supply-scan' if args.feasibility else "slice-v%d" % args.version))
    if args.action == "memory":
        build_memory_batch(Path(args.slice), Path(args.runs), args.batch, Path(args.out), args.k)
        return 0
    return {"build": cmd_build, "dryrun": cmd_dryrun, "readout": cmd_readout}[args.action](args)


if __name__ == "__main__":
    sys.exit(main())
