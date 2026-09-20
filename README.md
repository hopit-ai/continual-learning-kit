# continual-learning-kit

Small, dependency-light tools for running continual-learning experiments on any GPU machine and sending
back a report. Open source under Apache 2.0. Maintained by Hopit AI as the public half of a research
programme on small models that keep learning new tasks without forgetting old ones.

| File | What it does |
|---|---|
| `kit/runner.py` | Runs a campaign file row by row **with a pilot gate**: a row marked `pilot` is judged against bars written in advance, and every later row is refused unless the pilot passed. A `prepare` stage does all IO on a CPU machine before any GPU is held. Records a start file before launch and a verdict after; never overwrites. |
| `kit/score_forgetting.py` | Scores a saved model on 300 fixed questions (GSM8K, MMLU, an IFEval-style subset), one greedy answer each, **deterministically by default**: under vLLM's ordinary compiled, batched engine two scorings of one model on one machine agreed on as few as 98 of 300 answers; with compilation off, a fixed seed and batch-invariant kernels they agree on 300 of 300. Compares two results only if they carry the same machine-and-mode fingerprint. |
| `kit/beds/spider.py`, `kit/beds/gsm8k.py` | Spider 1.0 text-to-SQL and GSM8K maths as beds. Spider ships as ids and hashes only and refuses any prompt or database that does not match; its checker is read-only and time-limited. GSM8K carries a contamination guard against the 100 maths questions of the forgetting panel. |
| `kit/beds/finqa.py` | FinQA as a training and evaluation bed: prompts, a scoring rule that accepts percent-or-decimal and analyst-style rounding, label-problem flags, and a reward function in the SDPO reference's shape whose feedback never carries the answer. |
| `kit/README-k3.md`, `kit/campaigns/k3-replay.yaml`, `kit/run_grpo.sh`, `kit/mix.py`, `kit/beds/rewards.py`, `kit/eval_bed.py`, `kit/delta.py`, `kit/k3_report.py` | Package K3, the rehearsal test: a 1.7B model learns SQL, then maths, and loses some SQL. Does asking the old job's QUESTIONS in 10 or 30 percent of each step protect it, compared with nothing and with a KL penalty? GRPO with the SDPO authors' trainer, unmodified; rehearsal rows are spread evenly and answered afresh, never replayed; every arm trains on the same number of rows. Opens with a pilot that stops the package unless the old job was first learned and then damaged. |
| `kit/README-k2.md`, `kit/campaigns/k2-recovery-test.yaml`, `kit/repair/` | Package K2, the recovery test: when a model seems to forget, is the ability gone or only hidden behind a bad answering habit? A short adapter repair on the original model's own answers to everyday prompts, scored at 50, 100, 200 and 300 steps, with a healthy-model control and a weight-drift measure. Opens with a pilot on a 0.6B model damaged on purpose. |
| `kit/README-k1a.md`, `kit/campaigns/k1a-forgetting-of-k0.yaml` | Package K1a: did the K0 models forget? 25 minutes on one GPU, with a pilot that scores the untrained model twice and refuses to continue unless the two agree. |
| `kit/stage_local.py` | Packs many small files into one deterministic archive on CPU, then stages it to local disk at GPU start with hash verification and reuse. `mirror` does the same for a few large files such as model weights. |
| `kit/make_report.py` | Turns run directories into `report.md` and `report.json`. |
| `kit/run_sdpo_toolalpaca.sh`, `kit/README-partner.md`, `kit/campaigns/` | Package K0: a replication of SDPO (Huebotter et al., arXiv 2601.20802) on its tool-use task with the authors' code at a pinned commit. |

Standard library plus PyYAML for everything except `score_forgetting.py generate`, which needs vLLM and one GPU.

```bash
pip install pyyaml pytest
python -m pytest -q tests
WORK=/tmp/toy python kit/runner.py prepare kit/campaigns/toy.yaml --all
WORK=/tmp/toy python kit/runner.py run kit/campaigns/toy.yaml --all
```

Docstrings mention numbered receipts and a run record. Those live in our research repository; every
number quoted here was checked there against raw files before it was written down.

Data notices are in `NOTICE`.
