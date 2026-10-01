#!/usr/bin/env python3
"""A GOLD-BLIND second reading of a model's answer: one unambiguous final answer, or none.

    from canonical import extract, is_correct
    candidate = extract("gsm8k", response)          # never sees the gold
    right = is_correct("gsm8k", candidate, gold)     # the bed's own comparison

Why it exists. Each bed's strict rule (kit/beds/gsm8k.py, kit/beds/finqa.py) reads the answer from the form its
instruction asks for, `Answer: <number>`. A model whose answer was cut at a token budget, or which ended with "So the
answer is 18." or `#### 18`, may have stated a final answer in another form, and the strict rule then scores it as
no answer at all. The budget audit (kit/cap_sweep.py, kit/budget_report.py) needs to tell "the model stopped stating
answers the way it was asked" apart from "the model got the arithmetic wrong", so it reads every answer twice: once
by the strict rule, once by this one. The two differ ONLY in extraction; the comparison with the gold is the bed's own
`is_correct`, unchanged.

THE RULES, AND THEIR HISTORY. This is rule set `kit-canonical.v2`, fixed on 1 October 2026 and written into the plan-v3
pre-registration before any plan-v3 scoring exists. It may not be tuned after those scores are seen; a change is a
new rule with a new name, never an edit to this one. Its one predecessor, v1, was read once, on development data: the
partner's K1c GSM8K answers at the 2,048-token cap (six scorings, 1,800 answers). That reading showed one defect, and
v2 differs from v1 in that one place (rule 4): under v1 a marker with no number after it, such as a text cut right
after "So, the answer is" or a quoted instruction "Final Answer in the form `Answer: <number>`", erased an answer the
model had already stated (67 strict-right answers, every one of them cut at the cap, read as no answer). Because the
rule was shaped on those ten checkpoints' answers, canonical scores ON THOSE CHECKPOINTS are development readings
and are labelled so; the rule is confirmatory only for models it has not seen.

1. GOLD-BLIND. `extract` receives the bed name and the response text and nothing else. No rule searches the text
   for the gold number, and no rule may: an extractor that looks for the right answer anywhere in the text would
   score every answer that mentions it in passing as right.

2. FINAL-ANSWER STATEMENTS, for the numeric beds ("gsm8k", "finqa"). Every one of these in the text is a statement,
   taken in the order its marker begins in the text:
     a. the word `answer` (or `answers`; case-insensitive; a whole word), then optionally spaces, tabs, `*` or `_`
        on the same line (markdown emphasis, as in `**Answer**:`), then one of `:`, `=`, or the whole words `is`,
        `are`, `would be`, `should be`. Examples: "Answer: 18", "the answer is $18.50", "Final Answer = 18",
        "the final answer would be 18%". "To find the answer, ..." and "the answer isn't" are NOT statements.
     b. a `\\boxed{...}`, with one nested brace level, exactly as gsm8k.BOXED matches it.
     c. a `####` line: `####` at the start of a line (after optional spaces) whose remaining text BEGINS with a
        number (after optional spaces, and an optional sign and an optional currency sign or opening parenthesis,
        in either order). A
        `####` followed by anything else is a markdown heading ("#### Step 2: ..."), not a statement.

3. THE CANDIDATE OF A STATEMENT is the FIRST number after the marker on the same line (for a box: inside the box),
   found and written by THAT BED'S own number rule:
     gsm8k: gsm8k.FIRST_NUMBER, returned as the matched text (`$1,800.50` gives "1,800.50"), which gsm8k.is_correct
            cleans with gsm8k.to_number exactly as it does a strict prediction.
     finqa: finqa.NUMBER, returned as the matched text with its parentheses, `$` and `%` kept ("(1,234.5)",
            "14.1%"), which `is_correct` below turns into a number with finqa.parse_number, exactly as
            finqa.extract_answer does: parentheses mean negative, and the decimals written are remembered for the
            rounding rule. Before looking for a number, the text after the marker is read the way
            finqa.extract_answer reads an `Answer:` line: stripped of spaces, backticks, `*` and `.`, a text that
            begins with "yes" is the candidate "yes", and one that begins with "no" (but not "not" or "none") is
            "no", because FinQA has yes/no questions and its strict rule accepts those two words.
   A statement after which the line (or box) holds no number has NO candidate.

4. THE LAST STATEMENT THAT STATES SOMETHING DECIDES. Among the statements that have a candidate, the last one in
   the text is the answer, whatever its form; a later one supersedes an earlier one. A marker with no candidate
   (nothing, or no number, after it on its line) states nothing and is skipped, so it cannot erase an answer already
   given. If no statement has a candidate, there is no candidate. There is NEVER a fallback to "the last number in
   the text".

   What this reading is, so nobody over-reads it: it is THE LAST ANSWER THE MODEL HAD STATED BY THE END OF THE TEXT,
   whether or not the model would have gone on to revise it. In an answer cut mid-reasoning ("So the answer is 594
   feet.\n\nBut wait...") that is a provisional answer. The canonical score therefore counts provisional answers,
   right or wrong, exactly as a reader who stopped at the cut would have to. It is gold-blind either way.

5. `is_correct(bed, candidate, gold)` delegates to the bed's own `is_correct`. None is never correct.

How it differs from the strict rules, so a reader can predict every disagreement. gsm8k's strict rule takes the
last `Answer:`/`Answer=` line, and only when that line has no number does it look at the last `\\boxed{}`; finqa's
strict rule reads `Answer:`/`Answer=` lines only. The canonical rule also accepts `answer is/are/would be/should
be`, `**Answer**:`, and `#### 18`, and it decides by POSITION: a box written after the last `Answer:` line wins
here and loses there. An answer cut before any statement scores zero by both rules.

VERSION 3, THE TWO AUTHORS' BEDS. This module now carries rule set `kit-canonical.v3`. It is v2 plus two beds,
"chemistry" (four-option multiple choice) and "toolalpaca" (tool calls); rules 1-5 above, for "gsm8k" and "finqa",
are unchanged byte for byte in behaviour. The two rules below were written BEFORE any answer of either task was seen
by anyone writing them, so, unlike v2 on the K1c checkpoints, they are confirmatory from their first use. Rule 1
(gold-blind) holds for both: `extract` receives the bed name and the text, never the gold.

6. CHEMISTRY (multiple choice; kit/beds/chemistry.py, whose strict rule is the authors' mcq.py: the text after the
   LAST `<answer>` up to `</answer>`, stripped, must equal the gold letter exactly).
     a. Statements, in text order, by where each begins:
          - every `<answer>` tag; its content runs to the next `</answer>` or, if there is none, to the end of the
            text (an answer cut inside its tag still states what it had written);
          - the `answer` marker of rule 2a, with the rest of its line;
          - every `\\boxed{...}`, matched as rule 2b matches it.
     b. The candidate of a statement is a single capital letter A, B, C or D that begins its content after
        skipping whitespace, `(`, `[`, `*`, `_`, `"`, `'` and backticks, and an optional word `option` or `choice`
        (case-insensitive) with the whitespace or colon after it (and the same punctuation again, so "Option (B)"
        reads B). The letter must be followed by a character that is not a letter or digit, or by the end. So
        "A) acetone", "(B)", "**C**", "D." and "Option: D" have a candidate; "Acetone", "a)", "B12" do not.
     c. Rule 4: the LAST statement that has a candidate decides. `is_correct` is candidate == gold.
   It differs from the strict rule in three ways a reader can predict: the letter may carry punctuation or a word
   "option" around it, a statement outside the tags counts ("The answer is C."), and a later empty or letterless
   statement cannot erase an earlier letter.

7. TOOLALPACA (tool calls; kit/beds/toolalpaca.py, whose strict rule is the authors' tooluse.py). The candidate is
   ONE canonical JSON string, `{"actions": [...sorted...], "inputs": {...}}` (keys sorted), built from the WHOLE
   answer, as the authors' rule reads the whole answer:
     actions = every match of `Action:\\s*(\\w+)`, the authors' own pattern;
     inputs  = for every `Action Input:` in text order, the JSON OBJECT that begins at the first `{` after it
               (only whitespace may come between), decoded with `json.JSONDecoder().raw_decode`, so a nested
               object, or a brace inside a string, is read to its real end; the objects are merged with
               dict.update in order. An input that does not decode is skipped, exactly as the authors skip it.
   There is NO candidate when the answer has no `Action:` at all. `is_correct(bed, candidate, gold)` parses the
   gold as the authors do (a JSON list of {"Action", "Action_Input"}, where Action_Input is a JSON string or an
   object; an input that does not parse counts as {}; the dictionaries are merged in order) and requires the
   multiset of action names, and the merged inputs, to be equal.
   So this reading differs from the strict one ONLY in how an input object is DELIMITED: the authors' non-greedy
   `Action Input:\\s*({.*?})` ends an object at its first `}`, so a correct call with a nested object (or a `}`
   inside a string value) is cut, fails to parse and is dropped; here it is read whole. Every other property of
   the authors' rule is kept on purpose, including that every `Action:` line counts, so an answer that revises
   itself is judged on every action it wrote. Rule 4 has no separate role here: the candidate is the whole
   answer's calls, not one statement.

`EXTRACTORS` below is where a further bed would be added, with its own pre-registered rule written into this
docstring, under a new rule name, before it is used. `extract` refuses any bed not in `BEDS`.

Standard library only; the beds are imported by file path, so `kit/` works when exported alone.
"""
from __future__ import annotations

import importlib.util
import json
import re
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
BEDS = ("gsm8k", "finqa", "chemistry", "toolalpaca")
RULE = "kit-canonical.v3"          # the name of the rules above; a changed rule gets a new name

# Rule 2a. `[ \t*_]*` keeps the marker on one line and lets markdown emphasis sit between the word and the colon.
ANSWER_MARKER = re.compile(r"\banswers?\b[ \t*_]*(?::|=|(?:is|are|would[ \t]+be|should[ \t]+be)\b)", re.I)
# Rule 2c. The look-ahead is what tells a statement from a markdown heading.
HASH_MARKER = re.compile(r"^[ \t]*####(?=[ \t]*[-+]?[ \t]*[$£€¥(]?[ \t]*[-+]?[ \t]*(?:\d|\.\d))", re.M)
# Rule 6a. The tag is matched as the authors' mcq.py splits on it: the literal text, case-sensitive.
ANSWER_TAG, ANSWER_CLOSE = "<answer>", "</answer>"
# Rule 6b. The punctuation set is skipped before and after the optional word; the letter must stand alone.
_SKIP = r"""[\s(\[*_"'`]*"""
LETTER = re.compile(r"%s(?:(?i:option|choice)[\s:]*%s)?([ABCD])(?![A-Za-z0-9])" % (_SKIP, _SKIP))
# Rule 7. `ACTION` is the authors' own pattern; an input object begins at the first `{` after its marker.
ACTION = re.compile(r"Action:\s*(\w+)")
ACTION_INPUT = re.compile(r"Action Input:\s*(?=\{)")
_DECODER = json.JSONDecoder()

_BEDS: dict = {}


def _bed(name: str):
    """The bed's own module, loaded once by file path (kit/beds/<name>.py)."""
    if name not in _BEDS:
        spec = importlib.util.spec_from_file_location("kit_canonical_bed_%s" % name, HERE / "beds" / ("%s.py" % name))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _BEDS[name] = module
    return _BEDS[name]


def _line_after(text: str, end: int) -> str:
    """The rest of the line from `end`."""
    stop = text.find("\n", end)
    return text[end:] if stop < 0 else text[end:stop]


def _tag_contents(text: str) -> list:
    """[(position, content)] for every `<answer>` tag: to the next `</answer>`, or to the end (rule 6a)."""
    found, start = [], text.find(ANSWER_TAG)
    while start >= 0:
        begin = start + len(ANSWER_TAG)
        stop = text.find(ANSWER_CLOSE, begin)
        found.append((start, text[begin:] if stop < 0 else text[begin:stop]))
        start = text.find(ANSWER_TAG, begin)
    return found


def _action_inputs(text: str) -> list:
    """[(position, decoded object or None)] for every `Action Input:` followed by a `{` (rule 7)."""
    found = []
    for m in ACTION_INPUT.finditer(text):
        try:
            value, _end = _DECODER.raw_decode(text, m.end())
        except ValueError:
            value = None
        found.append((m.start(), value if isinstance(value, dict) else None))
    return found


def statements(bed: str, text: str) -> list:
    """[(position, form, content)] for every final-answer statement, in text order.

    Numeric beds (rules 2a-2c): the content is the text after the marker. chemistry (rule 6a): the same `answer`
    marker and box, and every `<answer>` tag with its content. toolalpaca (rule 7): every `Action:` with its name
    and every `Action Input:` with its decoded object (None when it does not decode); the candidate is built from
    all of them together, not from the last one."""
    text = text or ""
    if bed == "toolalpaca":
        found = [(m.start(), "action", m.group(1)) for m in ACTION.finditer(text)]
        found += [(position, "action input", value) for position, value in _action_inputs(text)]
        return sorted(found, key=lambda s: s[0])
    found = [(m.start(), "answer", _line_after(text, m.end())) for m in ANSWER_MARKER.finditer(text)]
    found += [(m.start(), "boxed", m.group(1)) for m in _bed("gsm8k").BOXED.finditer(text)]
    if bed == "chemistry":
        found += [(position, "<answer>", content) for position, content in _tag_contents(text)]
    else:
        found += [(m.start(), "####", _line_after(text, m.end())) for m in HASH_MARKER.finditer(text)]
    return sorted(found, key=lambda s: s[0])


def _gsm8k_candidate(after: str):
    found = _bed("gsm8k").FIRST_NUMBER.search(after)
    return found.group(0) if found else None


def _finqa_candidate(after: str):
    finqa = _bed("finqa")
    low = after.strip().strip("`*. ").lower()
    if low.startswith("yes"):
        return "yes"
    if low.startswith("no") and not low.startswith(("not", "none")):
        return "no"
    found = finqa.NUMBER.search(after)
    return found.group(0).strip() if found else None


def _chemistry_candidate(content: str):
    found = LETTER.match(content)
    return found.group(1) if found else None


def _toolalpaca_candidate(text: str):
    """Rule 7: the whole answer's calls as one canonical JSON string, or None when it has no `Action:`."""
    actions = ACTION.findall(text)
    if not actions:
        return None
    inputs: dict = {}
    for _position, value in _action_inputs(text):
        if value is not None:                     # an input that does not decode is skipped, as the authors skip it
            inputs.update(value)
    return json.dumps({"actions": sorted(actions), "inputs": inputs}, sort_keys=True, ensure_ascii=False)


# The extension point: bed -> the function giving one statement's candidate (rule 4 then picks the last one). A bed
# whose candidate is built from the whole text instead is listed in WHOLE_TEXT. A new bed is added here only after its
# rule is written into the module docstring under a new rule name.
EXTRACTORS = {"gsm8k": _gsm8k_candidate, "finqa": _finqa_candidate, "chemistry": _chemistry_candidate,
              "toolalpaca": _toolalpaca_candidate}
WHOLE_TEXT = ("toolalpaca",)


def extract(bed: str, text: str):
    """The candidate of the last final-answer statement in `text` that has one, or None. Never sees the gold.

    For toolalpaca the candidate is the whole answer's tool calls (rule 7)."""
    if bed not in EXTRACTORS:
        raise ValueError("no canonical rule for the %r bed (there is one for %s); a new bed's rule must be written "
                         "into kit/canonical.py before it is used" % (bed, ", ".join(BEDS)))
    candidate_of = EXTRACTORS[bed]
    if bed in WHOLE_TEXT:
        return candidate_of(text or "")
    for _position, _form, after in reversed(statements(bed, text)):      # rule 4: the last statement WITH a candidate
        candidate = candidate_of(after)
        if candidate is not None:
            return candidate
    return None


def gold_calls(gold) -> tuple:
    """The authors' parse of a toolalpaca ground truth: (action names, merged input dictionary).

    A JSON list of {"Action", "Action_Input"}; Action_Input is a JSON string or an object, and one that does not
    parse counts as {}. A gold that is not such a list gives ([], {}), which no candidate equals."""
    try:
        calls = json.loads(gold) if isinstance(gold, str) else gold
    except ValueError:
        return [], {}
    if not isinstance(calls, list):
        return [], {}
    actions, merged = [], {}
    for call in calls:
        if not isinstance(call, dict):
            continue
        actions.append(call.get("Action"))
        raw = call.get("Action_Input")
        try:
            value = json.loads(raw) if isinstance(raw, str) else raw
        except ValueError:
            value = {}
        if isinstance(value, dict):
            merged.update(value)
    return actions, merged


def is_correct(bed: str, candidate, gold) -> bool:
    """The bed's own comparison of a canonical candidate with the gold. None is never correct."""
    if candidate is None:
        return False
    if bed == "gsm8k":
        return bool(_bed("gsm8k").is_correct(candidate, gold))
    if bed == "finqa":
        finqa = _bed("finqa")
        prediction = candidate if candidate in ("yes", "no") else finqa.parse_number(candidate)
        return bool(finqa.is_correct(prediction, gold))
    if bed == "chemistry":                                                  # rule 6c
        return candidate == gold
    if bed == "toolalpaca":                                                 # rule 7
        mine = json.loads(candidate)
        actions, inputs = gold_calls(gold)
        return Counter(mine["actions"]) == Counter(actions) and mine["inputs"] == inputs
    raise ValueError("no canonical rule for the %r bed (there is one for %s)" % (bed, ", ".join(BEDS)))
