# continual-learning-kit

Small, dependency-light tools for running continual-learning experiments on any GPU machine and sending
back a report. Open source under Apache 2.0. Maintained by Hopit AI as the public half of a research
programme on small models that keep learning new tasks without forgetting old ones.

| File | What it does |
|---|---|
| `kit/runner.py` | Runs a campaign file row by row **with a pilot gate**: a row marked `pilot` is judged against bars written in advance, and every later row is refused unless the pilot passed. A `prepare` stage does all IO on a CPU machine before any GPU is held. Records a start file before launch and a verdict after; never overwrites. |
| `kit/score_forgetting.py` | Scores a saved model on 300 fixed questions (GSM8K, MMLU, an IFEval-style subset), one greedy answer each, **deterministically by default**: under vLLM's ordinary compiled, batched engine two scorings of one model on one machine agreed on as few as 98 of 300 answers; with compilation off, a fixed seed and batch-invariant kernels they agree on 300 of 300. Compares two results only if they carry the same machine-and-mode fingerprint. |
| `kit/beds/finqa.py` | FinQA as a training and evaluation bed: prompts, a scoring rule that accepts percent-or-decimal and analyst-style rounding, label-problem flags, and a reward function in the SDPO reference's shape whose feedback never carries the answer. |
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
