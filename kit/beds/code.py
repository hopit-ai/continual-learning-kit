#!/usr/bin/env python3
"""LiveCodeBench as a training and evaluation bed: prompts, a held-out panel, a reward that runs the code.

LiveCodeBench (Jain et al., 2024; `livecodebench/code_generation_lite` on Hugging Face) is a stream of
competitive-programming problems dated by the contest they came from, which is what makes it usable
here: a training set can be cut off before the evaluation set starts, so nothing the model trains on
post-dates what it is measured on. This is K5's coding job, and the job K1c could not include because
there was no bed for it.

WHAT IS FROZEN, and why the loader refuses rather than assumes.

The held-out set is the SAME 51 problems phase 1 measured (receipt 205): Opus 5 answered 40 of them
and the untrained Qwen3-8B answered 20. K5's coding numbers are only worth reading against those two,
so the panel is pinned problem by problem and in phase 1's own order. LiveCodeBench is not ours to
redistribute, so `code-split-v1.json` carries no problem statement and no test: it carries the problem
ids, the sha256 of every prompt as phase 1 rendered it, and the sha256 of every problem's encoded
private-test payload as the release ships it. Point this module at your own download and it rebuilds
each prompt and re-hashes each payload; a different release, an edited problem or a changed prompt is
a refusal, not a quietly different number.

THE TRAINING SPLIT is every problem in your copy of the release released BEFORE the panel's earliest
contest date (2025-02-08), minus the panel itself and minus anything whose problem text matches a
panel problem. The cut-off is the whole point: a problem that post-dates the panel cannot be in the
training set, so the held-out score cannot be a training score. The split file additionally pins the
56 of those problems that this laboratory was able to hash (they live in the release file phase 1
verified); if any of them is missing from your copy or its bytes differ, the loader refuses, because
that copy is not this release. The rest of the training split is selected by the date rule from your
download and recorded in the manifest -- ids, count and a hash of the id list -- rather than pinned.

THE SCORING RULE. A response is expected to carry the complete program in a ``` fence, as the prompt
asks; the LAST fenced block is taken. That program is run against the problem's tests by
`kit/sandbox.py`, one subprocess per test, and the reward is 1 only if EVERY test passes. Nothing is
partial: a program that passes thirty-nine of forty tests scores zero, which is the reference's rule
and the only rule under which "solved" means solved. Scoring stops at the first failure.

    python3 code.py prepare --lcb-root /path/to/code_generation_lite --out /work/data/code   # CPU, before any GPU is held
    CODE_TESTS=/work/data/code/train.tests.jsonl python3 code.py score --split heldout --responses responses.jsonl

`compute_score` has the signature the SDPO reference expects of a custom reward function. Its feedback
names the KIND of the first failure -- wrong answer, too slow, an error, a network attempt -- and
nothing else: not the failing test's input, not its expected output, not which test it was, not how
many passed. A teacher may see that the attempt was wrong, never what right would have been.

TESTS IN TRAINING ARE A SUBSET, and this is declared rather than hidden. The held-out panel is scored
on ALL of a problem's tests, exactly as phase 1 scored it. Training rows carry a deterministic subset
(by default at most 8 tests and 128 KiB of them, chosen by the hash of the problem id and the test
index, so the choice does not move when the file order does), because a training reward that runs
forty tests of up to ten seconds each per rollout is not affordable. The cap, the selected indices and
the number of problems it bound are all in the manifest.

Standard library only, plus pyarrow when it happens to be installed, plus `kit/sandbox.py` beside it.
Nothing here imports anything from this programme's private packages.
"""
from __future__ import annotations

import argparse
import base64
import functools
import hashlib
import importlib.util
import io
import json
import os
import pickle
import re
import sys
import zlib
from pathlib import Path

DATA_SOURCE = "livecodebench"
DATASET_URL = "https://huggingface.co/datasets/livecodebench/code_generation_lite"
SPLIT_FILE = Path(__file__).resolve().with_name("code-split-v1.json")
SPLIT_SCHEMA = "kit-bed-code-split.v1"
SPLITS = ("train", "heldout")
TESTS_ENV = "CODE_TESTS"
LCB_ROOT_ENV = "LCB_ROOT"

# The prompt phase 1 rendered, character for character. It is in the split file as a hash, and the
# loader refuses a split file built against a different one.
CODE_PROMPT = """You are a coding expert. You will be given a coding problem, and you need to write a correct Python program that matches the specification and passes all tests. The time limit is 1 second. You may start by outlining your thought process. In the end, please provide the complete code in a code block enclosed with ``` ```.

{problem}"""

# Training-set caps. Both are declared departures from "every test", recorded in the manifest.
TRAIN_TESTS = 8
TRAIN_TEST_BYTES = 128 * 1024
MAX_PRED_CHARS = 4000

_FENCE = re.compile(r"```[ \t]*([A-Za-z0-9+#_-]*)[ \t]*\r?\n(.*?)```", re.S)
_PROBLEM_ID = re.compile(r"^[A-Za-z0-9_.-]+$")

# The only feedback this bed ever returns. Fixed strings, one per outcome kind: no test input, no
# expected output, no index, no count, and no digits at all, so nothing can leak by arithmetic.
FEEDBACK = {
    "passed": "",
    "wrong_answer": "The program ran but gave the wrong answer on one of the tests. Re-read the statement, "
                    "check the algorithm on the sample the statement gives, and check the output format.",
    "timeout": "The program was still running when the time limit was reached on one of the tests. A faster "
               "algorithm is needed, not a faster line.",
    "error": "The program stopped with an error on one of the tests. Check the edge cases and check that the "
             "input is read exactly as the statement describes.",
    "network": "The program tried to open a network connection. Solve the problem with the standard library "
               "and the input alone.",
    "output_limit": "The program printed far more than the problem can expect. Check the loop that prints.",
    "no_code": "Return the complete program in a Python code block enclosed with ``` ```.",
}
INCORRECT_FORMAT = {"no_code"}


class CodeBedError(Exception):
    """A source, a hash or a request does not match what the split file pins."""


def _load_sandbox():
    """`kit/sandbox.py`, loaded by path so this file works both as a module and as a script."""
    path = Path(__file__).resolve().parents[1] / "sandbox.py"
    if not path.is_file():
        raise CodeBedError("kit/sandbox.py is missing: %s. The bed cannot score code without it." % path)
    spec = importlib.util.spec_from_file_location("kit_sandbox", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@functools.lru_cache(maxsize=1)
def sandbox():
    return _load_sandbox()


# ----------------------------------------------------------------- the frozen split and its sources
def load_split(path=SPLIT_FILE) -> dict:
    """Read `code-split-v1.json` and check it is the shape the loader is allowed to trust."""
    path = Path(path)
    try:
        split = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise CodeBedError("cannot read the split file %s: %s" % (path, exc)) from exc
    except ValueError as exc:
        raise CodeBedError("the split file %s is not JSON: %s" % (path, exc)) from exc
    if not isinstance(split, dict) or split.get("schema") != SPLIT_SCHEMA:
        raise CodeBedError("%s is not a %s file" % (path, SPLIT_SCHEMA))
    if split.get("prompt_template_sha256") != hashlib.sha256(CODE_PROMPT.encode("utf-8")).hexdigest():
        raise CodeBedError("the split file was built against a different prompt template than this module renders")
    panel = split.get("panel")
    if not isinstance(panel, dict) or not isinstance(panel.get("members"), list) or not panel["members"]:
        raise CodeBedError("the split file pins no panel members")
    pinned = split.get("train_pinned")
    if not isinstance(pinned, dict) or not isinstance(pinned.get("members"), list) or not pinned["members"]:
        raise CodeBedError("the split file pins no training problems")
    cutoff = split.get("train_rule", {}).get("released_before")
    if cutoff != panel.get("earliest_contest_date"):
        raise CodeBedError("the split file's training cut-off is not the panel's earliest contest date")
    for group, members in (("panel", panel["members"]), ("train_pinned", pinned["members"])):
        for row in members:
            missing = {"id", "prompt_sha256", "tests_sha256", "test_count", "testtype", "fn_name",
                       "contest_date"} - set(row)
            if missing:
                raise CodeBedError("a %s member is missing %s" % (group, ", ".join(sorted(missing))))
            if not _PROBLEM_ID.match(str(row["id"])):
                raise CodeBedError("unsafe problem id in %s: %r" % (group, row["id"]))
    if [row.get("order") for row in panel["members"]] != list(range(len(panel["members"]))):
        raise CodeBedError("the split file's panel is not in the order phase 1 generated it")
    overlap = {row["id"] for row in panel["members"]} & {row["id"] for row in pinned["members"]}
    if overlap:
        raise CodeBedError("the split file puts %d problems in both the panel and the training set" % len(overlap))
    return split


class _NoGlobals(pickle.Unpickler):
    """An unpickler that refuses every global. The payload is a pickled string and needs none."""

    def find_class(self, module, name):
        raise CodeBedError("the encoded test payload unpickles a global (%s.%s); the released payload is a "
                           "pickled string, so this is not the released payload" % (module, name))


def decode_tests(encoded: str) -> list:
    """The released `private_test_cases` payload, decoded as the dataset's own loader decodes it.

    The payload is either plain JSON or base64 of zlib of a pickled JSON string. `pickle.loads` on a
    downloaded byte string is arbitrary code execution, so this uses an unpickler that raises on any
    global: a pickled `str` needs none, and anything that does need one is refused instead of run.
    """
    if not isinstance(encoded, str) or not encoded:
        raise CodeBedError("a problem has no private-test payload")
    try:
        tests = json.loads(encoded)
    except ValueError:
        try:
            serialised = _NoGlobals(io.BytesIO(zlib.decompress(base64.b64decode(encoded)))).load()
        except CodeBedError:
            raise
        except Exception as exc:
            raise CodeBedError("a private-test payload could not be decoded: %s" % exc) from exc
        if not isinstance(serialised, str):
            raise CodeBedError("a private-test payload is not a pickled string")
        try:
            tests = json.loads(serialised)
        except ValueError as exc:
            raise CodeBedError("a decoded private-test payload is not JSON: %s" % exc) from exc
    if not isinstance(tests, list) or not tests:
        raise CodeBedError("a private-test payload is not a non-empty list")
    if any(not isinstance(test, dict) or "input" not in test or "output" not in test
           or test.get("testtype") not in ("stdin", "functional") for test in tests):
        raise CodeBedError("a private-test payload is not a list of {input, output, testtype}")
    if len({test["testtype"] for test in tests}) != 1:
        raise CodeBedError("a private-test payload mixes test forms")
    return tests


def render_prompt(row) -> str:
    """Phase 1's rendering: the statement, the required signature for a functional problem, the wrapper."""
    problem = str(row["question_content"])
    starter = str(row.get("starter_code") or "")
    if starter.strip():
        if "def " not in starter:
            raise CodeBedError("problem %s has starter code with no function in it" % row.get("question_id"))
        signature = "def " + starter.split("def ", 1)[1].split("Input\n", 1)[0].strip()
        problem += "\n\nYour solution should have the following signature: ```python\n%s\n```" % signature
    return CODE_PROMPT.format(problem=problem).replace("(self, ", "(")


def normalise(text) -> str:
    """The comparison key for leakage: lower-case, whitespace collapsed."""
    return re.sub(r"\s+", " ", str(text)).strip().lower()


def source_files(lcb_root) -> list:
    """The release's problem files under a local copy: `test.jsonl`, `test2.jsonl`, ... in that order."""
    root = Path(lcb_root)
    if root.is_file():
        return [root]
    if not root.is_dir():
        raise CodeBedError("LCB_ROOT=%s is neither a file nor a directory" % root)
    for base in (root, root / "default", root / "code_generation_lite"):
        hits = sorted(base.glob("test*.jsonl"), key=lambda path: (len(path.name), path.name))
        if hits:
            return hits
    raise CodeBedError("no LiveCodeBench problem files under %s: expected test.jsonl, test2.jsonl ... test6.jsonl "
                       "from %s. Download the dataset and point --lcb-root at the directory holding them." % (root, DATASET_URL))


REQUIRED_FIELDS = ("question_id", "question_content", "starter_code", "metadata",
                   "private_test_cases", "contest_date", "contest_id", "platform", "difficulty")


def load_source(lcb_root) -> dict:
    """Every problem in the local copy, by id. A repeated id with different bytes is a refusal."""
    problems = {}
    for path in source_files(lcb_root):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise CodeBedError("cannot read %s: %s" % (path, exc)) from exc
        for number, line in enumerate(text.split("\n"), start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError as exc:
                raise CodeBedError("%s line %d is not JSON: %s" % (path, number, exc)) from exc
            missing = [field for field in REQUIRED_FIELDS if field not in row]
            if missing:
                raise CodeBedError("%s line %d has no %s: this is not %s"
                                   % (path, number, ", ".join(missing), DATASET_URL))
            problem_id = str(row["question_id"])
            previous = problems.get(problem_id)
            if previous is not None:
                if previous["private_test_cases"] != row["private_test_cases"] \
                        or previous["question_content"] != row["question_content"]:
                    raise CodeBedError("problem %s appears twice with different content (%s)" % (problem_id, path))
                continue                                     # the release repeats problems across its files
            problems[problem_id] = row
    if not problems:
        raise CodeBedError("the local copy under %s holds no problems" % lcb_root)
    return problems


def _member(row, pinned) -> dict:
    """One problem, rebuilt from the local copy and checked against what the split file pins."""
    problem_id = str(row["question_id"])
    prompt = render_prompt(row)
    prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    tests_sha256 = hashlib.sha256(str(row["private_test_cases"]).encode("utf-8")).hexdigest()
    if pinned is not None:
        if prompt_sha256 != pinned["prompt_sha256"]:
            raise CodeBedError("the prompt rebuilt for %s has sha256 %s, the split file pins %s: this download does "
                               "not render the prompt phase 1 measured"
                               % (problem_id, prompt_sha256[:12], pinned["prompt_sha256"][:12]))
        if tests_sha256 != pinned["tests_sha256"]:
            raise CodeBedError("the tests shipped for %s have sha256 %s, the split file pins %s: this is not the "
                               "release phase 1 scored" % (problem_id, tests_sha256[:12], pinned["tests_sha256"][:12]))
        if str(row["contest_date"]) != pinned["contest_date"]:
            raise CodeBedError("problem %s is dated %s here and %s in the split file"
                               % (problem_id, row["contest_date"], pinned["contest_date"]))
    tests = decode_tests(row["private_test_cases"])
    metadata = json.loads(row["metadata"]) if str(row["metadata"]).strip() else {}
    fn_name = metadata.get("func_name", "") or ""
    testtype = tests[0]["testtype"]
    if testtype == "functional" and not fn_name:
        raise CodeBedError("functional problem %s has no func_name in its metadata" % problem_id)
    if pinned is not None:
        if len(tests) != pinned["test_count"] or testtype != pinned["testtype"] or fn_name != pinned["fn_name"]:
            raise CodeBedError("problem %s does not have the tests the split file pins" % problem_id)
    return {"id": problem_id, "prompt": prompt, "prompt_sha256": prompt_sha256,
            "tests": [{"input": test["input"], "output": test["output"]} for test in tests],
            "tests_sha256": tests_sha256, "testtype": testtype, "fn_name": fn_name,
            "question": str(row["question_content"]), "contest_date": str(row["contest_date"]),
            "contest_id": str(row["contest_id"]), "platform": str(row["platform"]),
            "difficulty": str(row["difficulty"])}


def split_ids(lcb_root, split_name: str, split=None, problems=None) -> list:
    """Which problems are in one split, and in what order, without decoding a single test.

    The membership rule reads only ids, dates and problem text, so it is cheap; decoding the tests is
    the expensive part, and that happens one problem at a time in `iter_members`. Every refusal that
    is about membership rather than content is raised here.
    """
    if split_name not in SPLITS:
        raise CodeBedError("unknown split %r: expected one of %s" % (split_name, ", ".join(SPLITS)))
    split = load_split() if split is None else split
    problems = load_source(lcb_root) if problems is None else problems
    panel_pins = {row["id"]: row for row in split["panel"]["members"]}
    absent = [problem_id for problem_id in panel_pins if problem_id not in problems]
    if absent:
        raise CodeBedError("%d panel problems are not in this download (e.g. %s): the held-out set cannot be "
                           "assembled and the training set cannot be checked against a panel it cannot see, so "
                           "nothing is built. A smaller panel is not the panel."
                           % (len(absent), ", ".join(sorted(absent)[:3])))
    # Unconditional: the refusal above has already established that every pinned panel problem is in
    # the download, so the panel is returned whole or not at all. Filtering here would be dead code.
    if split_name == "heldout":
        return [row["id"] for row in split["panel"]["members"]]       # phase 1's order

    cutoff = split["train_rule"]["released_before"]
    panel_text = {normalise(problems[pid]["question_content"]) for pid in panel_pins}
    train_pins = {row["id"]: row for row in split["train_pinned"]["members"]}
    missing = [problem_id for problem_id in train_pins if problem_id not in problems]
    if missing:
        raise CodeBedError("%d of the %d pinned training problems are not in this download (e.g. %s): this copy is "
                           "not the pinned release" % (len(missing), len(train_pins), ", ".join(sorted(missing)[:3])))
    chosen = []
    for problem_id in sorted(problems):
        row = problems[problem_id]
        if problem_id in panel_pins or str(row["contest_date"]) >= cutoff:
            continue
        if normalise(row["question_content"]) in panel_text:
            continue                                         # the same problem under another id
        chosen.append(problem_id)
    if len(chosen) < len(train_pins):
        raise CodeBedError("the training split has %d problems and the split file pins %d"
                           % (len(chosen), len(train_pins)))
    return chosen


def iter_members(lcb_root, split_name: str, split=None, problems=None):
    """Yield one split's members one at a time, refusing every hash that disagrees.

    One problem's decoded tests run to tens of megabytes, so a whole split does not fit comfortably in
    memory: the caller writes each member out and lets it go. Membership refusals happen before the
    first member is yielded; content refusals as each problem is reached.
    """
    split = load_split() if split is None else split
    problems = load_source(lcb_root) if problems is None else problems
    chosen = split_ids(lcb_root, split_name, split, problems)
    pins = {row["id"]: row for row in
            (split["panel"] if split_name == "heldout" else split["train_pinned"])["members"]}
    cutoff = split["train_rule"]["released_before"]

    def generate():
        for problem_id in chosen:
            member = _member(problems[problem_id], pins.get(problem_id))
            # An invariant, not a filter: `split_ids` has already applied the cut-off, so this cannot
            # fire today and no test can make it. It is here because the date rule is the one thing in
            # this bed that must never quietly stop being true, and the two functions could drift.
            if split_name == "train" and member["contest_date"] >= cutoff:
                raise CodeBedError("training problem %s does not pre-date the panel: refusing to build a training "
                                   "set that post-dates what it is measured on" % problem_id)
            yield member

    return generate()


def load_members(lcb_root, split_name: str, split=None, problems=None) -> list:
    """A whole split at once. Fine for the 51-problem panel; `iter_members` is what `prepare` uses."""
    return list(iter_members(lcb_root, split_name, split, problems))


# ------------------------------------------------------------------------------ tests and their file
def canonical_tests(tests) -> str:
    return json.dumps(tests, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def tests_digest(tests) -> str:
    return hashlib.sha256(canonical_tests(tests).encode("utf-8")).hexdigest()


def select_tests(problem_id: str, tests, max_tests: int = TRAIN_TESTS, max_bytes: int = TRAIN_TEST_BYTES):
    """A deterministic subset of one problem's tests: (indices, tests, capped_by_bytes).

    Ordered by the hash of `<problem id>:<index>`, so the choice does not move when the file order
    does and does not favour the small early tests. At least one test is always kept, even when that
    one test is larger than the byte budget -- a problem with no test cannot be scored at all.
    """
    if max_tests is None or max_tests <= 0:
        return list(range(len(tests))), list(tests), False
    order = sorted(range(len(tests)),
                   key=lambda index: hashlib.sha256(("%s:%d" % (problem_id, index)).encode("utf-8")).hexdigest())
    chosen, used, capped = [], 0, False
    for index in order:
        if len(chosen) >= max_tests:
            break
        size = len(canonical_tests([tests[index]]).encode("utf-8"))
        if chosen and max_bytes and used + size > max_bytes:
            capped = True                                    # this one does not fit; a smaller later one may
            continue
        chosen.append(index)
        used += size
    chosen.sort()
    return chosen, [tests[index] for index in chosen], capped


def tests_row(member: dict, indices, tests) -> dict:
    return {"problem_id": member["id"], "testtype": member["testtype"], "fn_name": member["fn_name"],
            "test_indices": indices, "tests_sha256": tests_digest(tests), "tests": tests}


class TestsFile:
    """One problem's tests, read by seeking to its line. The whole file is never held in memory."""

    def __init__(self, path):
        self.path = Path(path)
        if not self.path.is_file():
            raise CodeBedError("no tests file at %s: prepare writes train.tests.jsonl and heldout.tests.jsonl, "
                               "and %s must point at the one this split was trained from" % (self.path, TESTS_ENV))
        # The index is built without parsing any line: a single problem's tests can be tens of
        # megabytes, and the reward function needs one of them, not all of them. `prepare` writes each
        # line with sorted keys, so `fn_name` is the only key before `problem_id` and the FIRST
        # occurrence of the marker is always the key rather than something inside a test.
        self.offsets = {}
        offset = 0
        with self.path.open("rb") as handle:
            for line in handle:
                stripped = line.strip()
                if stripped:
                    marker = b'"problem_id":'
                    start = stripped.find(marker)
                    if start < 0:
                        raise CodeBedError("%s has a line with no problem_id" % self.path)
                    problem_id = json.loads(stripped[start + len(marker):].split(b",", 1)[0].strip())
                    self.offsets[str(problem_id)] = (offset, len(line))
                offset += len(line)
        if not self.offsets:
            raise CodeBedError("%s holds no problems" % self.path)

    def get(self, problem_id: str) -> dict:
        entry = self.offsets.get(str(problem_id))
        if entry is None:
            raise CodeBedError("problem %s is not in %s: the tests file does not belong to these rows"
                               % (problem_id, self.path))
        with self.path.open("rb") as handle:
            handle.seek(entry[0])
            row = json.loads(handle.read(entry[1]).decode("utf-8"))
        if str(row.get("problem_id")) != str(problem_id):
            raise CodeBedError("%s does not hold %s where its index says it does" % (self.path, problem_id))
        if tests_digest(row["tests"]) != row["tests_sha256"]:
            raise CodeBedError("the tests stored for %s do not hash to their recorded sha256" % problem_id)
        return row


@functools.lru_cache(maxsize=8)
def _tests_file(path: str) -> TestsFile:
    return TestsFile(path)


def tests_path_from_env(tests_path=None, extra_info=None) -> str:
    """The tests file at scoring time: the argument, else `extra_info['tests_path']`, else `CODE_TESTS`."""
    if tests_path:
        return str(tests_path)
    if isinstance(extra_info, dict) and extra_info.get("tests_path"):
        return str(extra_info["tests_path"])
    value = os.environ.get(TESTS_ENV)
    if not value:
        raise CodeBedError("%s is not set: the code reward needs the tests file `prepare` wrote next to the "
                           "trainer rows (train.tests.jsonl)" % TESTS_ENV)
    return value


# ------------------------------------------------------------------------------- answers and scoring
def extract_code(text) -> str | None:
    """The program in the LAST ``` fence. None when the response carries no fenced block with code in it."""
    blocks = [match.group(2) for match in _FENCE.finditer(str(text or ""))]
    for block in reversed(blocks):
        if block.strip():
            return block.strip("\n")
    return None


def score_response(response, member_tests: dict, *, timeout_s=None, memory_mib=None) -> dict:
    """One response against one problem's tests: the outcome kind, and whether every test passed."""
    box = sandbox()
    program = extract_code(response)
    if program is None:
        return {"kind": "no_code", "correct": False, "pred": "", "passed": 0,
                "n": len(member_tests["tests"])}
    verdict = box.run_tests(
        program, member_tests["tests"], testtype=member_tests["testtype"], fn_name=member_tests["fn_name"],
        timeout_s=box.DEFAULT_TIMEOUT_S if timeout_s is None else float(timeout_s),
        memory_mib=box.DEFAULT_MEMORY_MIB if memory_mib is None else int(memory_mib))
    return {"kind": verdict["kind"], "correct": verdict["kind"] == "passed", "pred": program[:MAX_PRED_CHARS],
            "passed": verdict["passed"], "n": verdict["n"]}


def ground_truth_for(member: dict, indices, tests) -> str:
    """What a trainer row carries: an identity, never the tests themselves and never an answer."""
    return json.dumps({"problem_id": member["id"], "testtype": member["testtype"],
                       "fn_name": member["fn_name"], "tests_sha256": tests_digest(tests),
                       "n_tests": len(indices)}, sort_keys=True)


def compute_score(data_source: str, solution_str: str, ground_truth: str, extra_info=None,
                  *, tests_path=None, timeout_s=None, memory_mib=None) -> dict:
    """Reward function in the SDPO reference's shape. The feedback never reveals `ground_truth`.

    `ground_truth` is the JSON string `{"problem_id", "testtype", "fn_name", "tests_sha256", "n_tests"}`;
    the tests themselves come from the file `prepare` wrote, named by `CODE_TESTS`. Identity failures
    raise: a reward that silently scores against another problem's tests is worse than a crash.
    """
    if data_source != DATA_SOURCE:
        raise CodeBedError("unsupported reward data source %r, expected %r" % (data_source, DATA_SOURCE))
    if not isinstance(solution_str, str):
        raise CodeBedError("the response must be text")
    try:
        reference = json.loads(ground_truth)
    except (TypeError, ValueError) as exc:
        raise CodeBedError("ground_truth is not a JSON object: %s" % exc) from exc
    if not isinstance(reference, dict) or not {"problem_id", "tests_sha256"} <= set(reference):
        raise CodeBedError("ground_truth must carry at least a problem_id and a tests_sha256")
    stored = _tests_file(tests_path_from_env(tests_path, extra_info)).get(reference["problem_id"])
    if stored["tests_sha256"] != reference["tests_sha256"]:
        raise CodeBedError("the tests file holds a different set of tests for %s than the row was built with: "
                           "refusing to score against tests this row was not prepared from" % reference["problem_id"])
    outcome = score_response(solution_str, stored, timeout_s=timeout_s, memory_mib=memory_mib)
    return {"score": float(outcome["correct"]), "acc": float(outcome["correct"]), "pred": outcome["pred"],
            "incorrect_format": int(outcome["kind"] in INCORRECT_FORMAT), "feedback": FEEDBACK[outcome["kind"]]}


def eval_items(lcb_root, split_name: str = "heldout", split_file=SPLIT_FILE) -> list:
    """`[{'id', 'prompt', 'ground_truth'}]` for one split, in the shape `kit/eval_bed.py` asks each bed for.

    The panel is scored on every test, as phase 1 scored it, so the ground truth names every test the
    problem has. The tests themselves still come from the tests file at scoring time.
    """
    items = []
    for member in iter_members(lcb_root, split_name, load_split(split_file)):
        indices = list(range(len(member["tests"]))) if split_name == "heldout" else \
            select_tests(member["id"], member["tests"])[0]
        tests = [member["tests"][index] for index in indices]
        items.append({"id": member["id"], "prompt": member["prompt"],
                      "ground_truth": ground_truth_for(member, indices, tests)})
    return items


def trainer_row(member: dict, split_name: str, indices, tests) -> dict:
    """One row in the shape the pinned trainer consumes. It carries the prompt and an identity, never a test."""
    return {
        "data_source": DATA_SOURCE,
        "prompt": [{"role": "user", "content": member["prompt"]}],
        "ability": "code",
        "reward_model": {"style": "rule", "ground_truth": ground_truth_for(member, indices, tests)},
        "extra_info": {"split": split_name, "index": member["id"], "problem": member["prompt"],
                       "description": "", "elo": 0, "achievement_prior": 0,
                       "problem_id": member["id"], "testtype": member["testtype"],
                       "fn_name": member["fn_name"], "test_indices": indices,
                       "test_count": len(member["tests"]), "contest_date": member["contest_date"],
                       "prompt_sha256": member["prompt_sha256"]},
    }


# --------------------------------------------------------------------------------------- subcommands
def _write_new(path: Path, text: str) -> None:
    if path.exists():
        raise CodeBedError("refusing to overwrite an existing output: %s" % path)
    path.write_text(text, encoding="utf-8")


def _write_parquet(path: Path, rows) -> bool:
    if path.exists():
        raise CodeBedError("refusing to overwrite an existing output: %s" % path)
    try:
        import pyarrow as pa                                                   # noqa: PLC0415
        import pyarrow.parquet as pq                                           # noqa: PLC0415
    except ImportError:
        return False
    pq.write_table(pa.Table.from_pylist(rows), path)
    return True


def cmd_prepare(args) -> int:
    root, out = Path(args.lcb_root), Path(args.out)
    split = load_split(args.split_file)
    problems = load_source(root)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": "kit-bed-code.v1", "data_source": DATA_SOURCE,
        "dataset": split["dataset"], "lcb_root": str(root),
        "source_files": [str(path) for path in source_files(root)],
        "split_file": str(Path(args.split_file)),
        "split_file_sha256": hashlib.sha256(Path(args.split_file).read_bytes()).hexdigest(),
        "problems_in_local_copy": len(problems),
        "panel": {"name": split["panel"]["name"], "members": len(split["panel"]["members"]),
                  "earliest_contest_date": split["panel"]["earliest_contest_date"]},
        "train_rule": dict(split["train_rule"]),
        "train_test_cap": {"max_tests": args.train_tests, "max_bytes": args.train_test_bytes},
        "sandbox": sandbox().limits(args.timeout, args.memory_mib),
        "splits": {},
    }
    for split_name in SPLITS:
        # Written as it goes: a member's decoded tests are large, and the panel's whole tests file is
        # about 50 MB, so nothing holds a split's tests in memory. The trainer rows are text and small,
        # and they are kept only because parquet has to be handed the table whole.
        rows_path, tests_path = out / ("%s.jsonl" % split_name), out / ("%s.tests.jsonl" % split_name)
        parquet_path = out / ("%s.parquet" % split_name)
        for path in (rows_path, tests_path, parquet_path):
            if path.exists():
                raise CodeBedError("refusing to overwrite an existing output: %s" % path)
        rows, ids, capped, kept, available, forms = [], [], 0, 0, 0, {"stdin": 0, "functional": 0}
        rows_digest, tests_digest_stream = hashlib.sha256(), hashlib.sha256()
        dates = []
        # Written under `.part` names and renamed at the end, so a refusal half way through -- a hash
        # that does not match on the five hundredth problem -- leaves no half-output behind to block
        # the next attempt.
        rows_part, tests_part = Path("%s.part" % rows_path), Path("%s.part" % tests_path)
        try:
            with rows_part.open("w", encoding="utf-8") as rows_handle, \
                    tests_part.open("w", encoding="utf-8") as tests_handle:
                for member in iter_members(root, split_name, split, problems):
                    if args.limit and split_name == "train" and len(ids) >= args.limit:
                        break
                    if split_name == "heldout":              # the panel is scored on every test
                        indices, tests, was_capped = list(range(len(member["tests"]))), member["tests"], False
                    else:
                        indices, tests, was_capped = select_tests(member["id"], member["tests"],
                                                                  args.train_tests, args.train_test_bytes)
                    row = trainer_row(member, split_name, indices, tests)
                    rows.append(row)
                    for handle, digest, payload in (
                            (rows_handle, rows_digest, row),
                            (tests_handle, tests_digest_stream, tests_row(member, indices, tests))):
                        line = json.dumps(payload, sort_keys=True, ensure_ascii=False) + "\n"
                        handle.write(line)
                        digest.update(line.encode("utf-8"))
                    ids.append(member["id"])
                    dates.append(member["contest_date"])
                    capped += int(was_capped)
                    kept += len(indices)
                    available += len(member["tests"])
                    forms[member["testtype"]] += 1
        except BaseException:
            rows_part.unlink(missing_ok=True)
            tests_part.unlink(missing_ok=True)
            raise
        for part, final in ((rows_part, rows_path), (tests_part, tests_path)):
            if final.exists():                               # another process got there while we wrote
                part.unlink(missing_ok=True)
                raise CodeBedError("refusing to overwrite an existing output: %s" % final)
            os.replace(part, final)
        wrote_parquet = _write_parquet(parquet_path, rows)
        manifest["splits"][split_name] = {
            "rows": len(rows), "jsonl_sha256": rows_digest.hexdigest(),
            "parquet": wrote_parquet, "tests_file": tests_path.name,
            "tests_file_bytes": tests_path.stat().st_size,
            "tests_file_sha256": tests_digest_stream.hexdigest(),
            "tests_run_per_problem": kept, "tests_available_per_problem": available,
            "problems_capped_by_test_bytes": capped,
            "stdin_problems": forms["stdin"], "functional_problems": forms["functional"],
            "ids_sha256": hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest(),
            "earliest_contest_date": min(dates, default=""), "latest_contest_date": max(dates, default=""),
        }
        if split_name == "train":
            manifest["splits"][split_name]["pinned_of_which_present"] = len(split["train_pinned"]["members"])
            manifest["splits"][split_name]["ids"] = ids
    _write_new(out / "code.manifest.json", json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    train, heldout = manifest["splits"]["train"], manifest["splits"]["heldout"]
    print("prepared LiveCodeBench in %s: train %d problems (%s to %s, %d tests kept of %d available, %d capped by "
          "bytes), heldout %d problems (%s to %s, every test)%s"
          % (out, train["rows"], train["earliest_contest_date"][:10], train["latest_contest_date"][:10],
             train["tests_run_per_problem"], train["tests_available_per_problem"],
             train["problems_capped_by_test_bytes"], heldout["rows"], heldout["earliest_contest_date"][:10],
             heldout["latest_contest_date"][:10], "" if train["parquet"] else " (jsonl only: pyarrow not installed)"))
    return 0


def score_responses(member_ids, responses, tests_file: TestsFile, *, timeout_s=None, memory_mib=None) -> dict:
    """Score saved responses for one split. Missing, duplicate or foreign ids are refused, never zero.

    The tests come from the tests file, which re-hashes what it returns; `member_ids` is what the
    answers are checked against, in the order the split defines.
    """
    member_ids = [str(member_id) for member_id in member_ids]
    by_id = {}
    for row in responses:
        member_id = str(row.get("id") or row.get("problem_id") or row.get("member_id") or "")
        if member_id in by_id:
            raise CodeBedError("response file has a duplicate answer for %s" % member_id)
        by_id[member_id] = str(row.get("response") or "")
    missing = [member_id for member_id in member_ids if member_id not in by_id]
    extra = sorted(set(by_id) - set(member_ids))
    if missing or extra:
        raise CodeBedError("response membership failed: %d missing (e.g. %s), %d unexpected (e.g. %s)"
                           % (len(missing), ", ".join(missing[:3]) or "-", len(extra), ", ".join(extra[:3]) or "-"))
    outcomes, correct, per_problem = {}, 0, {}
    for member_id in member_ids:
        stored = tests_file.get(member_id)               # `get` re-hashes the tests it returns
        result = score_response(by_id[member_id], stored, timeout_s=timeout_s, memory_mib=memory_mib)
        outcomes[result["kind"]] = outcomes.get(result["kind"], 0) + 1
        correct += int(result["correct"])
        per_problem[member_id] = int(result["correct"])
    return {"n": len(member_ids), "correct": correct,
            "accuracy": round(correct / max(1, len(member_ids)), 6),
            "outcomes": dict(sorted(outcomes.items())), "per_problem": per_problem}


def cmd_score(args) -> int:
    split = load_split(args.split_file)
    root = args.lcb_root or os.environ.get(LCB_ROOT_ENV)
    if not root:
        raise CodeBedError("%s is not set: scoring rebuilds the split from your copy of the release" % LCB_ROOT_ENV)
    problems = load_source(root)
    # Rebuild and hash-check every member before anything is scored, one at a time, then throw them
    # away: what the answers are actually run against is the tests file, which hashes itself.
    verified = sum(1 for _ in iter_members(root, args.split, split, problems))
    member_ids = split_ids(root, args.split, split, problems)
    tests_file = _tests_file(tests_path_from_env(args.tests))
    responses = [json.loads(line) for line in Path(args.responses).read_text(encoding="utf-8").split("\n")
                 if line.strip()]
    if args.panel:
        responses = [row for row in responses if row.get("panel") in (None, args.panel)]
    summary = {"split": args.split, "responses": str(args.responses), "verified_members": verified,
               "tests_file": str(tests_file.path), "timeout_s": args.timeout,
               **score_responses(member_ids, responses, tests_file, timeout_s=args.timeout,
                                 memory_mib=args.memory_mib)}
    text = json.dumps(summary, indent=1, sort_keys=True) + "\n"
    if args.out:
        _write_new(Path(args.out), text)
    print(json.dumps({key: value for key, value in summary.items() if key != "per_problem"}, sort_keys=True))
    return 0


def main(argv=None) -> int:
    box = _load_sandbox()
    parser = argparse.ArgumentParser(description="LiveCodeBench bed: prepare trainer files, or score saved responses.")
    sub = parser.add_subparsers(dest="action", required=True)
    prepare = sub.add_parser("prepare")
    # `--code-root` is the name the K5 package's README uses for every bed's data directory; both
    # spellings mean the same directory, and neither is preferred over the other.
    prepare.add_argument("--lcb-root", "--code-root", required=True, dest="lcb_root",
                         help="the directory holding test.jsonl ... test6.jsonl")
    prepare.add_argument("--out", required=True)
    prepare.add_argument("--split-file", default=str(SPLIT_FILE))
    prepare.add_argument("--train-tests", type=int, default=TRAIN_TESTS,
                         help="at most this many tests per training problem; 0 keeps every test")
    prepare.add_argument("--train-test-bytes", type=int, default=TRAIN_TEST_BYTES)
    prepare.add_argument("--timeout", type=float, default=box.DEFAULT_TIMEOUT_S)
    prepare.add_argument("--memory-mib", type=int, default=box.DEFAULT_MEMORY_MIB)
    prepare.add_argument("--limit", type=int, help="keep only the first N training problems (a smoke, not a run)")
    score = sub.add_parser("score")
    score.add_argument("--responses", required=True)
    score.add_argument("--split", default="heldout", choices=list(SPLITS))
    score.add_argument("--lcb-root", "--code-root", default=None, dest="lcb_root",
                       help="default: the LCB_ROOT environment variable")
    score.add_argument("--tests", default=None, help="default: the %s environment variable" % TESTS_ENV)
    score.add_argument("--split-file", default=str(SPLIT_FILE))
    score.add_argument("--panel", default=None, help="keep only response rows carrying this panel name")
    score.add_argument("--timeout", type=float, default=box.DEFAULT_TIMEOUT_S)
    score.add_argument("--memory-mib", type=int, default=box.DEFAULT_MEMORY_MIB)
    score.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    try:
        return {"prepare": cmd_prepare, "score": cmd_score}[args.action](args)
    except CodeBedError as exc:
        raise SystemExit(str(exc))


if __name__ == "__main__":
    sys.exit(main())
