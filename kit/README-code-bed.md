# The coding bed: LiveCodeBench, run for real

**What this is.** A fourth task bed, next to `beds/spider.py` (SQL), `beds/gsm8k.py` (maths) and
`beds/finqa.py` (finance). It turns LiveCodeBench problems into training rows and into a held-out
panel, and it scores an answer by **running the program against the problem's tests**. There is no
string comparison anywhere in it: the only way to know whether a program is right is to run it.

K5 — four jobs in a row — needs a coding job, and K1c had none because this bed did not exist.

**Three files, and nothing else.**

| file | what it does |
|---|---|
| `beds/code.py` | prompts, the two splits, the reward function |
| `sandbox.py` | runs one program against one test, under limits, and deletes everything after |
| `beds/code-split-v1.json` | the ids and hashes that say whether your download is the right one |

They import nothing from our private code, use the standard library (`pyarrow` only if you happen to
have it), and need no root, no Docker socket and no network.

---

## 1. The data, and what you have to download

The dataset is **`livecodebench/code_generation_lite`** on Hugging Face
(<https://huggingface.co/datasets/livecodebench/code_generation_lite>), at the commit the split file
pins (`eef2046…`, the revision phase 1 used). Its problem files are `test.jsonl`, `test2.jsonl` …
`test6.jsonl`. Download them yourself; **we ship no problem text and no tests**, because the dataset
is not ours to redistribute. Point `--lcb-root` at the directory that holds those files.

    python3 kit/beds/code.py prepare \
        --lcb-root /data/code_generation_lite \
        --out /work/data/code

That runs on the CPU and finishes before any GPU is held, like every other `prepare` in this kit. It
writes, and refuses to overwrite anything that already exists:

| file | what it is |
|---|---|
| `train.jsonl`, `train.parquet` | the training rows for the trainer |
| `heldout.jsonl`, `heldout.parquet` | the 51-problem panel |
| `train.tests.jsonl` | the tests the training reward runs |
| `heldout.tests.jsonl` | the tests the panel is scored on — this one is large, about 50 MB |
| `code.manifest.json` | every count, every hash, the date range of each split, and what the sandbox enforced |

Two practical notes. `prepare` reads the release's problem files into memory, so expect a couple of
gigabytes of RSS for the full download; it then writes each split **as it goes**, under `.part` names
that are renamed only when the split finishes, so nothing holds a split's decoded tests and a refusal
half way through leaves no partial file to block the retry. And it never replaces an existing file:
if you want to prepare again, write to a new directory.

---

## 2. The two splits, and the one rule that matters

**The held-out set is the same 51 problems phase 1 measured.** On those 51, Opus 5 answered **40**
and the untrained Qwen3-8B answered **20** (receipt 205). Every coding number K5 produces is read
against those two, so the panel cannot drift: `code-split-v1.json` pins all 51 by id, in phase 1's
own order, with the sha256 of each rendered prompt and of each problem's shipped test payload. The
loader rebuilds each prompt from your download and re-hashes each payload. A different release, an
edited problem, a changed prompt: each is a **refusal**, not a quietly different number. A panel
problem missing from your copy is also a refusal — a smaller panel is not the panel.

**One number moved, and here is why.** We re-scored phase 1's own retained Opus 5 answers with this
bed's sandbox and got **41**, not 40. The single difference is `abc398_a`: the released tests say the
input `3` must produce `-=-`, Opus wrote exactly that, and phase 1's pinned scorer marked it wrong
while reporting that it expected `-=`. The model was right and the old scorer was wrong, so **the
frontier baseline on this bed is 41 of 51**, and that is the number K5's coding results should be
read against. The other fifty agree problem for problem. The untrained student's 20 of 51 we could
**not** re-check — its raw answers were not kept on this side — so treat 20 as phase 1's number under
phase 1's scorer, and re-measure the untrained model once on this bed before using it as a floor.

**The training set is everything released before the panel starts.** The panel's earliest contest
date is **2025-02-08**. The training split is every problem in your copy dated before that, minus the
51 themselves, minus anything whose problem text matches a panel problem under another id. So nothing
the model trains on post-dates what it is measured on, and the held-out score cannot be a training
score wearing a hat.

**What is pinned and what is not, said plainly.** This laboratory only ever downloaded one of the
release's files (`test6.jsonl`, the one holding the 2025 contests). So the split file pins the **56**
pre-cut-off problems that live in that file, and the loader refuses if any of them is missing from
your copy or its bytes differ — that is the check that says your download is this release. The rest
of the training split, which lives in the release's earlier files, is selected by the date rule from
your download and **recorded** in the manifest (ids, count, a hash of the id list) rather than pinned
in advance. We are not going to claim to have hashed files we never had.

---

## 3. How an answer is scored

The prompt asks for the complete program in a ``` fence. The scorer takes the **last** fenced block;
a response with no fenced block is a format failure (`incorrect_format`), not a wrong answer.

The program is then run against the problem's tests, one subprocess per test, and

> **the reward is 1 only if every test passes.**

Thirty-nine of forty is zero. That is the reference's rule, and it is the only rule under which
"solved" means solved. Scoring stops at the first failure, which is what makes it affordable.

LiveCodeBench has two test forms and this bed runs both, as the benchmark defines them:

- **stdin** — the test input goes to the program's standard input; its standard output is compared
  line by line, each line right-stripped and trailing blank lines dropped on both sides. The input
  is a real file on the child's stdin, so `sys.stdin.buffer` works. (Phase 1 lost a scoring run to a
  `StringIO` stand-in that had no `.buffer`; this bed does not repeat that.)
- **functional** — the test input is one JSON value per line, the arguments. The program is imported,
  not run as `__main__`; `Solution().<name>` is called with those arguments, or a module-level
  `<name>` if there is no `Solution` class; the returned value is JSON round-tripped and compared
  with the expected value. Tuples become lists in that round trip, as in the reference harness, and
  the reference's `expected == [result]` fallback is kept.

Of the 51 panel problems, 33 are stdin and 18 are functional.

### The feedback never contains the answer

`compute_score` returns `{score, acc, pred, incorrect_format, feedback}`, the shape the SDPO
reference expects. The feedback names **the kind of the first failure and nothing else**:

| what happened | what the model is told |
|---|---|
| wrong answer | "The program ran but gave the wrong answer on one of the tests…" |
| too slow | "The program was still running when the time limit was reached on one of the tests…" |
| an error | "The program stopped with an error on one of the tests…" |
| a network attempt | "The program tried to open a network connection…" |
| too much output | "The program printed far more than the problem can expect…" |
| no code block | "Return the complete program in a Python code block…" |

Not the failing test's input. Not its expected output. Not which test it was, not how many passed,
not how many there are. There is not a single digit in any of those strings, so nothing can leak by
arithmetic either. A teacher may see that the attempt was wrong; never what right would have been.

### Training runs a subset of the tests — declared, not hidden

The **panel is scored on every test**, exactly as phase 1 scored it. **Training rows carry at most 8
tests and at most 128 KiB of them** (`--train-tests`, `--train-test-bytes`; `--train-tests 0` keeps
everything). The subset is chosen by the hash of `<problem id>:<test index>`, so it does not move
when the file order does and does not just take the easy early tests. The cap, the chosen indices and
the number of problems the byte budget bound are all in `code.manifest.json` and in each row's
`extra_info`.

The reason is arithmetic, and you should check it against your own batch size before you run: forty
tests at up to ten seconds each is up to four hundred seconds **per rollout**, and a GRPO step with
eight rollouts over sixteen prompts is a hundred and twenty-eight of those. Eight tests, with scoring
stopping at the first failure, is what makes a coding reward fit inside a training loop at all. If
your reward workers are serial, this is the number to look at first.

---

## 4. Wiring it into a run

The trainer row carries only an **identity** — problem id, test form, function name, the sha256 of
the tests it was prepared from — never the tests and never an answer. The tests come from the file
`prepare` wrote:

    export CODE_TESTS=/work/data/code/train.tests.jsonl      # staged on local disk, not a network mount
    # custom_reward_function.path=<kit>/beds/code.py, name=compute_score

If the tests file holds a different set of tests for a problem than the row was built with, the
reward **raises** rather than scoring: a reward that silently scores against the wrong tests is worse
than a crash. The same applies to an unknown problem id.

To score saved answers for the panel:

    export LCB_ROOT=/data/code_generation_lite
    export CODE_TESTS=/work/data/code/heldout.tests.jsonl
    python3 kit/beds/code.py score --split heldout --responses answers.jsonl --out heldout-score.json

`answers.jsonl` is one `{"id": "<problem id>", "response": "<what the model wrote>"}` per line.
Missing, duplicate or unexpected ids are refused rather than scored as zero. Before it scores
anything, `score` rebuilds and re-hashes every member of the split from your download — the count it
checked is `verified_members` in the output — so a panel that has drifted is caught before an hour of
subprocesses, not after. The written summary also carries `per_problem`, one `0` or `1` per problem,
which is what a paired comparison against the frontier's 41 needs.

---

## 5. Wiring it into `kit/eval_bed.py`

`eval_bed.py` is the common generate-and-score front end for the beds, and it carries one table entry
per bed. That file is not part of this bed, so it is not changed here; the entry it needs is:

| table | entry |
|---|---|
| `BED_FILES` | `"code": HERE / "beds" / "code.py"` |
| `DEFAULT_SPLIT` | `"code": "heldout"` |
| `SPLITS` | `"code": ("train", "heldout")` |
| `MAX_NEW_TOKENS` | `"code": 8192` — phase 1 generated these 51 answers at 8,192 tokens, and a shorter budget would truncate the fence and read as a format failure |
| `items_of` | `elif args.bed == "code": items = module.eval_items(root, split)` |

`eval_items` is in this bed for exactly that purpose: it returns `{"id", "prompt", "ground_truth"}`
per item, the panel on every test, in phase 1's order. `--root` for this bed is the directory holding
the release's `test*.jsonl` files, and `CODE_TESTS` must point at the matching tests file. Both
`--lcb-root` and `--code-root` are accepted on `prepare` and `score`, because the K5 package spells
it the second way.

## 6. What the sandbox does, and what it does not

`kit/sandbox.py` runs each test in a **fresh temporary directory**, in its own process session, with
a minimal environment, and deletes the directory afterwards whatever happened. Before a line of the
model's code runs, the child:

- sets `RLIMIT_AS` (address space, 4 GiB by default), `RLIMIT_CPU` (the wall-clock limit plus a
  second) and `RLIMIT_FSIZE` (32 MiB, which is what stops a program that prints forever);
- replaces `socket.socket` and its neighbours with things that raise, so an incidental `urllib` call
  fails instead of reaching the internet;

and the parent gives it an environment with **every proxy variable removed** and `no_proxy=*`, waits
at most ten seconds of wall clock, and then kills the whole process group, so nothing is orphaned.
The expected output is **never** written anywhere the program could read it and never enters the
child process at all.

**It is not a security boundary.** The child runs as your own user, on your own filesystem. A program
that wants to read your files, write outside its temporary directory, spawn a subprocess that does
not inherit the socket patch, or call a socket through `ctypes`, can. What the limits stop is the
accidents that wrong programs actually produce: infinite loops, runaway memory, endless output, a
stray network call. Run this on model output, in a container you would not mind losing. Do not run it
on code from someone who means you harm.

One honest caveat about memory: `RLIMIT_AS` is enforced by Linux and **is not enforced by macOS**, so
on a Mac the address-space limit silently does nothing (the wall clock and the CPU limit still bite).
Your container is Linux, so it bites there — but check rather than believe us:

    python3 kit/sandbox.py

prints what the sandbox enforces on that machine, runs a trivial program end to end, and deliberately
allocates past the limit to report whether the kernel actually stopped it.

---

## 7. If it refuses

Every refusal names what disagreed. The common ones:

| message | what it means |
|---|---|
| "the prompt rebuilt for X … this download does not render the prompt phase 1 measured" | your copy is a different release of the dataset |
| "N of the M pinned training problems are not in this download" | you have some of the release's files but not the one the panel came from |
| "panel problem X is not in this download" | same, and the panel cannot be assembled at all |
| "refusing to overwrite an existing output" | write to a new directory; outputs are never replaced |
| "CODE_TESTS is not set" | the reward needs the tests file `prepare` wrote |

None of them has a flag to make it pass. A bed that quietly measured something else would cost more
than a bed that stops.
