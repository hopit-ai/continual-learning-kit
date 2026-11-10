"""Export the public kit: kit/, standalone tests, pinned runtime sources, licence and README.

    python scripts/publish_kit.py /path/to/empty/dir        # builds the tree and checks it; publishes nothing

Before a tag, the manager must also run (outside this offline export gate):
    python3 scripts/check_pinned_inputs.py --contract-source "$CONTRACT_SOURCE" --receipt pinned-inputs.json
Require exit 0 / all inputs ok and retain the receipt. See kit/README-message.md,
"Stage before job 1". Stand-in/export tests alone do not prove real input pins.

The private repository stays the source of truth. The export is a plain copy with two changes: the
partner instructions clone the PUBLIC repository, and compiled files are dropped. A denylist then
scans every exported text file, and the exported tests run inside the export, so what is published is
proven to stand on its own and to carry nothing from outside the declared export scope.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import unicodedata
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PUBLIC_URL = "https://github.com/hopit-ai/continual-learning-kit"
PUBLIC_TAG = "kit-b0-v3"
K1A_TAG = "kit-k1a-v1"
TESTS = ("test_kit_runner.py", "test_kit_report.py", "test_kit_stage_local.py", "test_kit_bed_finqa.py", "test_kit_bed_gsm8k.py",
         "test_kit_k4a.py", "test_kit_k1c.py", "test_kit_scorecard.py", "test_kit_plasticity.py", "test_kit_bed_code.py", "test_kit_k5.py", "test_kit_k4.py", "test_kit_density.py", "test_kit_k3_dose.py", "test_kit_collect.py", "test_kit_repertoire.py", "test_kit_k2b.py", "test_kit_long_cap.py", "test_kit_budget.py", "test_kit_k1c_audit.py", "test_kit_pilot_report.py", "test_kit_control_report.py", "test_kit_pilot_campaign.py", "test_kit_k1c_control.py", "test_kit_bed_authors.py", "test_kit_pilot_launchers.py")
TESTS += ("test_kit_phase0_public.py", "test_kit_phase0_g7_public.py", "test_kit_finqa_phase0_carry.py", "test_kit_phase0_host.py", "test_kit_phase0_host_review.py", "test_kit_phase0_host_round2.py", "test_kit_phase0_storage.py", "test_kit_phase0_lustre.py", "test_kit_teacher_config_shape.py")
TESTS += ("test_kit_phase0_timing.py",)
TESTS += ("test_kit_phase0_closed_r3.py", "test_kit_finqa_host.py", "test_kit_finqa_retention.py")
TESTS += ("test_kit_finqa_fix_round.py", "test_kit_finqa_lean.py", "test_kit_finqa_self_monitor.py")
TESTS += ("test_kit_finqa_round3.py", "test_p3a_partner.py", "test_p3b_plain.py", "test_p3b_mixed.py")
TESTS += ("test_c_kit.py",)
TESTS += ("test_kit_phase0_v3.py", "test_kit_phase0_v4.py", "test_kit_phase0_v5.py", "test_kit_phase0_v7.py")
TESTS += ("test_next_kit.py",)
TESTS += ("test_math_hard_teacher.py",)
TESTS += ("test_a4_kit.py", "test_a4_v6b.py")
TESTS += ("test_sp.py", "test_combined_send.py", "test_x_smoke_attempt5.py", "test_text_return.py", "test_m2_public_expect.py", "test_likelihood_near_tie.py")
TESTS += ("test_ref_score_job.py",)
TESTS += ("test_util_watch.py", "test_lane_pool.py", "test_util70_tier.py", "test_m3_protocol.py", "test_message2_v6_resources.py", "test_util70_integration.py")
TESTS += ("test_v6_infra_authority.py",)
TESTS += ("test_v6a.py", "test_v6b.py",)
TESTS += ("test_v7_operator_fixes.py",)
TESTS += ("test_nm_actor.py", "test_nm_binding.py", "test_nm_budget.py", "test_nm_data.py", "test_nm_jobs.py", "test_nm_math.py", "test_nm_pipeline.py", "test_nm_prepare.py", "test_nm_read.py", "test_nm_return.py", "test_nm_text.py",)
TESTS += ("test_cr_confirmation.py", "test_cr_finqa.py", "test_v7_operator_fixes.py", "test_v7b_fixes.py",)
TESTS += ("test_cr_v7c_closure.py",)
# A2 public checks use shipped kit files and explicit synthetic CPU identities.
# The separate private registration parity test runs when its document exists.
TESTS += ("test_a2_review.py", "test_a2_sec_prompts.py", "test_a2_rewrite.py", "test_a2_arms.py",)
TESTS += ("kit_authority.py", "util70_fixtures.py",)
# Runtime hashes and the S reader require these exact source bytes on the node.
# Copying them grants no launch authority; Modal smokes remain manager-run.
RUNTIME_FILES = ("scripts/check_pinned_inputs.py", "scripts/screen_s_analysis.py",
                 "scripts/smoke_rung_p2_modal.py", "scripts/smoke_x_integration_modal.py",
                 "scripts/smoke_common_modal.py", "docs/send/final-review-redacted.md",
                 "scripts/read_text_return.py", "tests/text_return_fixtures.py")
RUNTIME_FILES += ("docs/phase3/text-return-utilisation-20261102.md", "scripts/v6_infra_authority.py")
RUNTIME_FILES += ("scripts/v6a_dryrun.py", "scripts/v6b_dryrun.py",)
RUNTIME_FILES += ("scripts/cr_confirmation_dryrun.py", "scripts/cr_confirmation_fixture.py",)
RUNTIME_FILES += ("docs/phase2/v4-frozen-gate-commands.md",
                  "docs/phase3/evidence/screen-s/finqa-arc-feasibility-20261025.json",)
# Standalone registration comparisons need the original source documents even
# when unrelated phase-3 runtime documents also create docs/phase3 in an export.
RUNTIME_FILES += ("docs/phase3/hard-math-screen-preregistration-20261020.md",
                  "docs/phase3/hard-math-screen-amendment-1-20261020.md",
                  "docs/phase3/hard-math-screen-result-20261020.md")
RUNTIME_FILES += ("scripts/apache-2.0.txt",
                  "docs/phase3/next-message-20261102/registration-cr-confirmation.md")
K2_TAG = "kit-k2-v2"
K3_TAG = "kit-k3-v1"
K1C_TAG = "kit-k1c-v1"
# Owner, 23 Oct 2026: the partner's name must never appear in shipped material.
# Identifiers and digests are private build inputs, never exported source.
from functools import lru_cache


def normalise(text):
    text = unicodedata.normalize('NFKD', text).casefold().replace('ı', 'i').replace('ſ', 's')
    return ''.join(c for c in text if not unicodedata.combining(c) and unicodedata.category(c) != 'Cf')


def load_private_names():
    path = ROOT/'management/private-names.json'
    try:
        doc = json.loads(path.read_text())
        rows = doc['names']
        names = tuple(re.sub('[^a-z0-9]', '', normalise(row['name'])) for row in rows)
        if not names or any(not name for name in names):
            raise ValueError('empty names')
        if any(hashlib.sha256(name.encode()).hexdigest() != row['sha256']
               for name, row in zip(names, rows)):
            raise ValueError('digest mismatch')
        repository = re.sub('[^a-z0-9]', '', normalise(doc['repository']))
        if repository not in names:
            raise ValueError('repository absent from names')
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise SystemExit(f'publisher requires valid private build input {path}: {exc}') from exc
    return names, repository


@lru_cache(maxsize=32768)
def _private_identifier(token, names):
    normal = re.sub('[^a-z0-9]', '', normalise(token))
    return any(name in normal for name in names)


class _Deny:
    structural = re.compile(r"/[U]sers/|@g(?:mail)\.|sk-[a]nt-|hf_[A-Za-z0-9]{12,}|BEGIN [A-Z ]*PRIVATE KEY")

    def search(self, text, names=None):
        if names is None:
            names, _ = load_private_names()
        match = self.structural.search(''.join(c for c in text if unicodedata.category(c) != 'Cf'))
        if match:
            return match
        text = normalise(text)
        return next((m for m in re.finditer(r'[\w]+(?:[-_][\w]+)*', text)
                     if _private_identifier(m.group(), names)), None)


DENY = _Deny()


def public_repository_names(text):
    _, repository = load_private_names()
    def replace(match):
        normal = re.sub('[^a-z0-9]', '', normalise(match.group()))
        return 'continual-learning-kit' if normal == repository else match.group()
    return re.sub(r'[\w]+(?:[-_][\w]+)*', replace, text)


def separator_name_hits(text, names):
    """Report all candidates; embedded runs in one ordinary word are advisory.

    Whole tokens and spans across multiple words/punctuation are blocking.
    An ordinary word is a maximal Unicode alphanumeric run (no underscore).
    Normalised offsets and context are used consistently in diagnostics.
    """
    text = normalise(text)
    separator = r'[\s._\-/\\|:,]{0,3}'
    for name in names:
        pattern = separator.join(map(re.escape, name))
        for match in re.finditer(pattern, text):
            start, end = match.span()
            whole = (not start or not text[start-1].isalnum()) and (end == len(text) or not text[end].isalnum())
            inside_word = all(c.isalnum() for c in match.group()) and not whole
            context = text[max(0, start-45):min(len(text), end+45)].replace('\n', r'\n')
            yield dict(line=text.count('\n', 0, start)+1, context=context,
                       blocking=whole or not inside_word)


PUBLIC_TEST_ARGS = ('-m', 'pytest', '-q', 'tests', '--ignore=tests/test_x_smoke_stage_audit.py')


README = """# continual-learning-kit

> **CR safety for newer tags:** CR runs only from a `kit-message-v7c` clone,
> checked against the owner's commit and receipt SHA256
> `04861c64970d3fb8e04936ec2042852120cc94cd04fb063a5084bb71066df380`.
> Clone any newer tag into a **NEW directory**, never into the v7c clone.
> On a newer tag, the CR refusal "confirmation code changed since diff review"
> is expected, not a defect. A newer tag's launch JSON carries no `cr_confirmation`.

The current combined message 2+3 entry point is [COMBINED SEND](kit/README-combined.md).
Message-3 joins only on Opus SEND by 2 November; tier failures never block core.

Small, dependency-light tools for running continual-learning experiments on any GPU machine and sending
back a report. Open source under Apache 2.0. Maintained by Hopit AI as the public half of a research
programme on small models that keep learning new tasks without forgetting old ones.

| File | What it does |
|---|---|
| `kit/README-phase0.md`, `kit/read_v4_phase0.py`, `kit/simulate_v4_phase0.py`, `kit/sim/bin/` | Bounded technical phase 0: confirmed host-venv payload, zero-GPU download job and Slurm header, archive reader, nine-case CPU rehearsal and process-only Slurm stand-ins. All entry points ship under kit/. |
| `kit/runner.py` | Runs a campaign file row by row **with a pilot gate**: a row marked `pilot` is judged against bars written in advance, and every later row is refused unless the pilot passed. A `prepare` stage does all IO on a CPU machine before any GPU is held. Records a start file before launch and a verdict after; never overwrites. |
| `kit/README-send4.md` | **Send 4, archived pilot runbook** (tag `kit-batch3-v1`): the budget audit of the ten K1c checkpoints, the 1.7B control on GSM8K (learning rate by training limit with a finish-gated reward), and the Qwen3-8B pilot on the SDPO authors' Chemistry and ToolAlpaca tasks under four recipes. Three commands, two archives. The runner now loses only the rows that need a failed run, and runs a row again by itself when the run it read was retried. |
| `kit/campaigns/k1c-control.yaml`, `kit/run_grpo.sh` (`MAX_RESPONSE`, `FINISH_GATE`), `kit/beds/rewards.py` | The small-model control: K1c trained with an 8,192-token response limit and was served at 2,048, and the trainer paid for whatever answer a cut rollout happened to contain. `MAX_RESPONSE` sets the training limit; `FINISH_GATE=1` scores 0 for a rollout the trainer marks as cut (it reads the trainer's own mark and refuses to run without it). Three cells, three runs each, against K1c's own five. |
| `kit/campaigns/k8b-pilot.yaml`, `kit/beds/chemistry.py`, `kit/beds/toolalpaca.py`, `kit/pilot_report.py`, `kit/rollout_stats.py`, the two ToolAlpaca launchers (`DATASET`, `LR`, `MINI_BATCH`, `TEACHER_RATE`) | The 8B discovery pilot: the SDPO authors' two tasks as scored beds (their data, their messages, their scorer), each learned alone and on top of the other under tuned GRPO (1e-6, minibatch 8), K1c's GRPO (1e-5, 32), SDPO, and SDPO with a frozen teacher. The report gives acquisition, retention of an acquired task with its signed accounting across answer budgets, and the training-side statistics. A discovery run: four recipes differ in several settings at once. |
| `kit/README-send3.md` | RETIRED, never sent. Send 3 as one runbook: the dose-and-anchor test, the K1c rescoring, then K2b; the runner now refuses a campaign whose row order is implicit, writes pilot dependencies into `needs`, continues past a failed run that nothing later needs, and runs reduced-seed sets with `--seeds`. |
| `kit/campaigns/k1c-audit.yaml`, `kit/cap_sweep.py`, `kit/canonical.py`, `kit/budget_report.py`, `kit/tokens_io.py` | The budget audit (plan v3, package 1): K1c's first archive showed that in eight of ten runs the trained 1.7B's answers had grown during training and were cut at the 2,048-token scoring cap, so the count there cannot separate a wrong answer from one that did not end. Each of the ten checkpoints and the untrained model is generated once at 8,192 new tokens on one GPU with its token ids kept; the score at every shorter budget is read from prefixes of that one generation, by the bed's strict rule and by a gold-blind rule that takes the last answer the model had stated; and each checkpoint's loss is written as a signed accounting of budget sensitivity, extraction sensitivity and what remains at the long cap. `eval_bed.py` and `score_forgetting.py` take `--max-new-tokens` and `--max-model-len`; `compare`, `summarize` and `delta.py` refuse two caps. |
| `kit/README-send2.md`, `kit/collect.py` | Send 2 as one runbook: four campaigns in order through `runner.py batch` (prepare then run for each, stop at the first failure; rows whose inputs come from earlier GPU rows are prepared right before they run), and `collect.py`, which packs every report, scoring, delta, metrics log and summary into one hash-manifested archive with no weights. |
| `kit/score_forgetting.py` | Scores a saved model on 300 fixed questions (GSM8K, MMLU, an IFEval-style subset), one greedy answer each, **deterministically by default**: under vLLM's ordinary compiled, batched engine two scorings of one model on one machine agreed on as few as 98 of 300 answers; with compilation off, a fixed seed and batch-invariant kernels they agree on 300 of 300. Compares two results only if they carry the same machine-and-mode fingerprint. |
| `kit/beds/spider.py`, `kit/beds/gsm8k.py` | Spider 1.0 text-to-SQL and GSM8K maths as beds. Spider ships as ids and hashes only and refuses any prompt or database that does not match; its checker is read-only and time-limited. GSM8K carries a contamination guard against the 100 maths questions of the forgetting panel. |
| `kit/beds/finqa.py` | FinQA as a training and evaluation bed: prompts, a scoring rule that accepts percent-or-decimal and analyst-style rounding, label-problem flags, and a reward function in the SDPO reference's shape whose feedback never carries the answer. |
| `kit/README-k3.md`, `kit/campaigns/k3-replay.yaml`, `kit/run_grpo.sh`, `kit/mix.py`, `kit/beds/rewards.py`, `kit/eval_bed.py`, `kit/delta.py`, `kit/k3_report.py` | Package K3, the rehearsal test: a 1.7B model learns SQL, then maths, and loses some SQL. Does asking the old job's QUESTIONS in 10 or 30 percent of each step protect it, compared with nothing and with a KL penalty? GRPO with the SDPO authors' trainer, unmodified; rehearsal rows are spread evenly and answered afresh, never replayed; every arm trains on the same number of rows. Opens with a pilot that stops the package unless the old job was first learned and then damaged. |
| `kit/README-k3-dose.md`, `kit/campaigns/k3-dose-probe.yaml`, `kit/subset.py`, `kit/k3_dose_report.py` | The K3 stage-A dose probe: stage A moved the Spider held-out score from 70 to 62 against a +5 bar, with the training reward flat and 300 of the 640 training questions solved in all 8 attempts. Four arms, one seed, one change each: the reference dose rerun, three passes at 60 steps, three times the learning rate, and 20 steps over only the questions that carry gradient. The report gives the paired churn, an exact McNemar p, the trainer's reward and gradient norm, and the density column per arm. No gate: the per-arm +5 rows are readouts. |
| `kit/README-k2.md`, `kit/campaigns/k2-recovery-test.yaml`, `kit/repair/` | Package K2, the recovery test: when a model seems to forget, is the ability gone or only hidden behind a bad answering habit? A short adapter repair on the original model's own answers to everyday prompts, scored at 50, 100, 200 and 300 steps, with a healthy-model control and a weight-drift measure. Opens with a pilot on a 0.6B model damaged on purpose. |
| `kit/README-k1a.md`, `kit/campaigns/k1a-forgetting-of-k0.yaml` | Package K1a: did the K0 models forget? 25 minutes on one GPU, with a pilot that scores the untrained model twice and refuses to continue unless the two agree. |
| `kit/README-k4a.md`, `kit/campaigns/k4a-stuck-problems.yaml`, `kit/beds/tooluse_soft.py`, `kit/k4a_report.py` | Package K4a, the stuck-question test: on SDPO's tool-use task about 63 percent of every training batch has no successful attempt among 8, so self-distillation has nothing to learn from there. Three arms, each ONE declared override of the K0 command — environment feedback to the teacher, a partial-credit TRAINING reward, and hotter training sampling — over three seeds, against the finished K0 runs as the control. The launcher refuses to run two arms at once; the validation score stays the authors' strict all-or-nothing one in every arm, because it is computed from a field the reward never touches. Opens with a two-step pilot per arm that stops the package on exit code, a merged model, response length and the untrained calibration band. |
| `kit/beds/code.py`, `kit/sandbox.py`, `kit/README-code-bed.md` | The coding bed: LiveCodeBench-lite problems rebuilt from your own download (ids and hashes only), the 51-problem held-out panel phase 1 used, and a local sandbox that runs a solution's tests with time and memory limits and no network. Not a security boundary against hostile code. |
| `kit/README-k4.md`, `kit/campaigns/k4-hints.yaml`, `kit/hints.py`, `kit/run_sdpo_bed.sh` | Package K4, hints for the stuck questions on Spider and FinQA: a large open model you serve writes a short plan for each question the 1.7B never solves, a leakage filter drops any plan carrying the answer, and four arms test whether the hint transfers to unaided answers. |
| `kit/README-k5.md`, `kit/campaigns/k5-sequence.yaml`, `kit/sequence.py`, `kit/k5_report.py` | Package K5, the headline: four jobs learned one after another in two orders, every bed and the general panel scored after every stage, the scorecard and the plasticity probe at the end. Arms are fixed after K3 and K4 report. |
| `kit/README-k1c.md`, `kit/campaigns/k1c-grpo-baselines.yaml`, `kit/run_grpo_toolalpaca.sh`, `kit/run_grpo.sh`, `kit/fold_lora.py`, `kit/k1c_report.py` | Package K1c, the learned-alone baselines: every job trained by itself, from the untrained model, with the authors' own reward-only `--config-name baseline_grpo` — no teacher, no self-distillation. Qwen3-8B on ToolAlpaca (full training against a LoRA arm, rank 64 on every linear layer, alpha 32, ten times the learning rate), paired seed for seed with the finished K0 SDPO runs, plus Qwen3-1.7B on GSM8K and FinQA alone at K3's stage-B dose. `verl.model_merger merge` does not fold a LoRA adapter — it leaves the untrained base model on disk with an adapter whose `lora_alpha` is hard-coded to 0 — so `kit/fold_lora.py` recomputes every adapted weight's delta itself and refuses a fold that changed nothing or changed something it should not have. Opens with the K1a machine check, then a two-step pilot per arm and per job. |
| `kit/density.py` | What a correct answer costs: tokens per correct answer (a held-out set's output tokens over its correct answers) from every scoring's own counts, against the untrained model's and a bar of 1.5 written before any number was read. The scorecard and the K3, K4, K5 and K1c reports carry the column; a scoring without counts shows `-`, never a refusal. `training` reads a report's per-step response length against its held-out score and names the verbosity shape (length up, score not up). |
| `kit/README-k3-anchor.md`, `kit/campaigns/k3-anchor.yaml`, `kit/repertoire.py` | The dose-and-anchor test: plain GRPO at the reference dose (learning rate 1e-5) degrades the 1.7B on every bed, collapsing its SQL style on Spider and its answer format on GSM8K and FinQA. Eleven arms at three runs each on Spider (20 steps) and GSM8K (40 steps): the reference, learning rates 1e-6 and 3e-6, a KL pull back towards the untrained model at two strengths, and an entropy bonus. An arm passes if its score does not fall, its answers stay within the cost bar, and, on Spider, it keeps joins in at least 30 of 100 answers or, on GSM8K, its format failures stay within 5 points of the untrained model's. `repertoire.py` reports how far a model's SQL style moved. |
| `kit/README-k2b.md`, `kit/campaigns/k2b-route.yaml`, `kit/run_sft.sh`, `kit/sft_entry.py`, `kit/k2b_report.py` | Package K2b: does the route of damage decide whether an ability comes back? The untrained Qwen3-1.7B learns GSM8K either by GRPO or by supervised fine-tuning, matched in questions, steps and rate, ten repeats each; the damage to its SQL and general ability, and how much a short repair restores against its own control. `run_sft.sh` runs the pinned fork's SFT trainer through `sft_entry.py`, which only turns pinned host memory off. |
| `kit/stage_local.py` | Packs many small files into one deterministic archive on CPU, then stages it to local disk at GPU start with hash verification and reuse. `mirror` does the same for a few large files such as model weights. |
| `kit/make_report.py` | Turns run directories into `report.md` and `report.json`. |
| `kit/run_sdpo_toolalpaca.sh`, `kit/README-partner.md`, `kit/campaigns/` | Package K0: a replication of SDPO (Huebotter et al., arXiv 2601.20802) on its tool-use task with the authors' code at a pinned commit. |

Standard library plus PyYAML for everything except `score_forgetting.py generate`, which needs vLLM and one GPU.

**Getting the data without a laptop in between.** Spider 1.0 (CC BY-SA 4.0) and FinQA (MIT) are on the `data-v1` branch of this repository, split under GitHub's file limit and listed in a hash manifest; its README has the three-line fetch. GSM8K and the LiveCodeBench file come from Hugging Face inside each package's `prepare`.

```bash
pip install pyyaml pytest pyarrow pandas
python -m pytest -q tests --ignore=tests/test_x_smoke_stage_audit.py
WORK=/tmp/toy python kit/runner.py prepare kit/campaigns/toy.yaml --all
WORK=/tmp/toy python kit/runner.py run kit/campaigns/toy.yaml --all
```

A `test_util_watch` failure under load is a timing flake; rerun that file alone: `python -m pytest -q tests/test_util_watch.py`.

Docstrings mention numbered receipts and a run record. Those live in our research repository; every
number quoted here was checked there against raw files before it was written down.

Data notices are in `NOTICE`.
"""

NOTICE = """continual-learning-kit
Copyright 2026 Hopit AI. Licensed under the Apache License, Version 2.0.

kit/panels/general-v1.jsonl contains 300 evaluation prompts selected from three public datasets:
  - GSM8K (Cobbe et al., 2021), MIT License.
  - MMLU (Hendrycks et al., 2021), MIT License.
  - IFEval (Zhou et al., 2023), Apache License 2.0.
They are evaluation prompts. Do not train on them.

kit/beds/spider.py reads, and does not include, Spider 1.0 (Yu et al., 2018), CC BY-SA 4.0; kit/beds/spider-split-v1.json holds\nonly question ids and hashes. kit/beds/gsm8k.py reads, and does not include, GSM8K, MIT License.\n\nkit/beds/finqa.py reads, and does not include, FinQA (Chen et al., 2021; github.com/czyssrs/FinQA), MIT License.

kit/resources/c/contract_nli/ contains pairs selected from ContractNLI (Koreeda and Manning, 2021;
Hitachi America, Ltd.; https://stanfordnlp.github.io/contract-nli/), CC BY 4.0
(https://creativecommons.org/licenses/by/4.0/). Changes: pairs selected and grouped, prompts and
metadata added; original text reproduced unchanged.

kit/resources/math-hard-teacher/ contains Level-5 problems from the MATH dataset (Hendrycks et al., 2021),
MIT License, and the frozen Qwen3-8B's responses to them.

kit/resources/next/sec/ contains questions, answers and the filing excerpts they are asked on, from
public SEC EDGAR filings (U.S. Securities and Exchange Commission, https://www.sec.gov/edgar).
kit/resources/a4/ holds only identifiers and hashes of a further SEC slice, not its text.

kit/run_sdpo_toolalpaca.sh launches, and does not include, lasgroup/SDPO
(https://github.com/lasgroup/SDPO), which carries its own licence.
"""


def licence_text() -> str:
    source = ROOT/'scripts/apache-2.0.txt'
    text=(source if source.is_file() else ROOT/'LICENSE').read_text()
    assert "Apache License" in text and "Version 2.0" in text
    return text


def build(target: Path, tag=None) -> Path:
    load_private_names()  # Fail before writing anything if private build inputs are absent.
    if target.exists() and any(target.iterdir()):
        raise SystemExit("export directory must be empty: %s" % target)
    (target / "tests").mkdir(parents=True)
    shutil.copytree(ROOT / "kit", target / "kit", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    partner = target / "kit" / "README-partner.md"
    text = public_repository_names(partner.read_text())
    text = text.replace("You have access to our private repository. Clone it", "Clone the kit")
    text = text.replace("Cloning on the host keeps your GitHub credentials out of the container; the clone appears inside it at", "The clone appears inside the container at")
    text = re.sub(r"checkout kit-b0-v\d+", "checkout " + PUBLIC_TAG, text)
    partner.write_text(text)
    campaign = target / "kit" / "campaigns" / "k0-sdpo-toolalpaca.yaml"
    campaign.write_text(public_repository_names(campaign.read_text()))
    for name in TESTS:
        shutil.copy(ROOT / "tests" / name, target / "tests" / name)
    for name in RUNTIME_FILES:
        destination=target/name;destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/name,destination)
    # The confirmation seal includes conservative runtime and review tooling.
    # Copy its external files verbatim too; never trim a sealed public closure.
    from kit.cr_dependencies import FROZEN_DEPENDENCIES
    for name in FROZEN_DEPENDENCIES:
        if name.startswith('../'):
            destination = target/name[3:]
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT/name[3:], destination)
    (target / "README.md").write_text(README)
    (target / "NOTICE").write_text(NOTICE)
    (target / "LICENSE").write_text(licence_text())
    (target / ".gitignore").write_text("__pycache__/\n*.pyc\n.pytest_cache/\n")
    if tag is None:
        try:
            tag = subprocess.check_output(["git", "-C", str(ROOT), "describe", "--tags",
                "--exact-match", "--match", "kit-*", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            tag = "untagged"
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", tag):
        raise ValueError("invalid kit identity tag")
    shipped_identity = ROOT/'kit/KIT-IDENTITY.json'
    try:
        toplevel = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "--show-toplevel"],
            text=True, stderr=subprocess.DEVNULL).strip()
        if Path(toplevel).resolve() != ROOT.resolve():
            raise ValueError('git belongs to an enclosing checkout')
        commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, ValueError, subprocess.CalledProcessError):
        commit = json.loads(shipped_identity.read_text())['commit']
    identity = {"tag": tag, "commit": commit}
    (target / "kit/KIT-IDENTITY.json").write_text(json.dumps(identity, sort_keys=True)+"\n")
    # Exact inventory for every shipped tier resource, including frozen bytes. No local
    # runtime maps, source docs/smoke inputs or generated operational manifests are copied.
    files = {str(p.relative_to(target)): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(target.rglob('*')) if p.is_file()}
    (target / 'EXPORT-INVENTORY.json').write_text(json.dumps(files, indent=2, sort_keys=True)+'\n')
    return target


def check_markdown_links(target: Path) -> None:
    """Require shipped file targets for public operator runbook links.

    Frozen registration/resource copies retain their original research-tree
    citations byte-for-byte. This gate covers the root and kit/*.md entry points.
    """
    target=target.resolve();missing=[]
    inline_destination=re.compile(r'\(\s*(?:<([^>\n]+)>|([^\s)]+))(?:\s+["\'][^\n]*?["\'])?\s*\)')
    definitions=re.compile(r'^ {0,3}\[(?!\^)[^\]\n]+\]:\s*(?:<([^>\n]+)>|(\S+))',re.M)
    for path in sorted([*target.glob('*.md'),*(target/'kit').glob('*.md')]):
        # Examples in fenced/inline code are not navigable Markdown links.
        text=re.sub(r'(?ms)^ {0,3}(`{3,}|~{3,})[^\n]*\n.*?^ {0,3}\1[ \t]*(?:\n|$)',
            lambda match:'\n'*match.group().count('\n'),path.read_text())
        text=re.sub(r'(`+)[^\n]*?\1',lambda match:' '*len(match.group()),text)
        links=[(match.start(),next(g for g in match.groups() if g is not None))
               for match in definitions.finditer(text)]
        # Balance label brackets so nested labels and soft line breaks work.
        # Escaped brackets are literal label text, not delimiters.
        brackets=[]
        for token in re.finditer(r'\\.|[\[\]]',text):
            if token.group()=='[':brackets.append(token.start())
            elif token.group()==']' and brackets:
                start=brackets.pop()
                match=inline_destination.match(text,token.end())
                if match:links.append((start,next(g for g in match.groups() if g is not None)))
        for start,link in sorted(links):
            url=urlsplit(link)
            if url.scheme or url.netloc or not url.path or url.path.startswith('/'):continue
            destination=(path.parent/unquote(url.path)).resolve()
            if not destination.is_relative_to(target) or not destination.is_file():
                number=text.count('\n',0,start)+1
                missing.append(f'{path.relative_to(target)}:{number}: {link}')
    if missing:raise SystemExit('export refused, dangling Markdown links:\n  '+'\n  '.join(missing))


def check(target: Path) -> None:
    names, _ = load_private_names()
    check_markdown_links(target)
    hits = []
    for path in sorted(p for p in target.rglob("*") if p.is_file()):
        if path.suffix in (".pyc",) or "__pycache__" in path.parts:
            hits.append("%s: compiled file" % path.relative_to(target))
            continue
        for number, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
            if DENY.search(line, names):
                hits.append("%s:%d: %s" % (path.relative_to(target), number, line.strip()[:120]))
        for hit in separator_name_hits(path.read_text(errors='replace'), names):
            diagnostic = '%s:%d: %s' % (path.relative_to(target), hit['line'], hit['context'])
            if hit['blocking']:
                hits.append(diagnostic)
            else:
                print('privacy advisory (inside one ordinary word): '+diagnostic)
    if hits:
        raise SystemExit("export refused, %d denylist hits:\n  %s" % (len(hits), "\n  ".join(hits[:20])))
    # Match MESSAGE 2's privacy rules against every byte actually shipped,
    # including the selected standalone tests and generated root files.
    from kit.m2_privacy import scan
    private=scan(target, paths=(p for p in target.rglob('*') if p.is_file()))
    if private:raise SystemExit('export refused by public privacy rules: '+', '.join(private))
    # test_x_smoke_stage_audit.py ships as a sealed reference-tooling input;
    # its native-reference tests require the separate SDPO checkout and are
    # outside the standalone TESTS gate. All selected standalone tests run.
    # The real exported-test gate owns and deletes its basetemp, even on failure.
    # Supply the copied build's seal explicitly to CPU test launches. Runtime
    # activation itself never discovers/approves the authority's own digest.
    import hashlib,os
    test_env=dict(os.environ,KIT_V6_AUTHORITY_SHA=hashlib.sha256(
        (target/'kit/resources/v6/infrastructure-update.json').read_bytes()).hexdigest())
    with tempfile.TemporaryDirectory(prefix='.publish-tests-',dir=target) as scratch:
        done = subprocess.run([sys.executable, "-I", *PUBLIC_TEST_ARGS, "-o", "pythonpath=.", "-p", "no:cacheprovider", "--basetemp",str(Path(scratch)/'bt')], cwd=target, env=test_env,capture_output=True, text=True)
    print(done.stdout.strip().splitlines()[-1])
    if done.returncode != 0:
        raise SystemExit("the exported tests do not pass on their own:\n" + done.stdout[-2000:])
    shutil.rmtree(target / ".pytest_cache", ignore_errors=True)
    for cache in list(target.rglob("__pycache__")):
        shutil.rmtree(cache, ignore_errors=True)


def main(argv=None) -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output",type=Path,help="empty directory for the checked public export; publishes nothing")
    parser.add_argument("--tag",help="actual intended release tag; otherwise use the exact checked-out kit tag or untagged")
    args=parser.parse_args(argv)
    target = build(args.output.resolve(),tag=args.tag)
    check(target)
    files = sorted(str(p.relative_to(target)) for p in target.rglob("*") if p.is_file())
    print("%d files ready in %s:\n  %s" % (len(files), target, "\n  ".join(files)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
