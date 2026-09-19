# Partner instructions: SDPO on ToolAlpaca, Qwen3-8B (batch 0, first package)

**What this is.** Three small jobs that repeat, on your GPUs, an experiment we ran once on 12 September 2026: the SDPO paper's tool-use task with the authors' own code. We measured 57.9 → 66.1 (avg@16 on 68 held-out questions) in 17 training steps. One run cannot tell us how big the gain really is. You are running it again with different seeds, and once at the paper's full one-hour dose.

**What you run is not our code.** It is `lasgroup/SDPO` at a pinned commit, unmodified, launched by one shell script from us (`kit/run_sdpo_toolalpaca.sh`) and summarised by one Python file (`kit/make_report.py`), both in the repository you clone in step 0. No Modal, no API keys, no uploads, no network at training time.

**Hardware.** One node. We ran on 4 × H200 (141 GB) with no offload and measured a peak of **113 GB allocated per GPU** (121 GB reserved), so 4 × 80 GB cards will not hold the same configuration.

| Your node | Add to every command | Notes |
|---|---|---|
| 4 × H200 or other 141 GB cards | nothing | what we ran |
| **8 × H100 or A100 80 GB (recommended for 80 GB cards)** | `NGPU=8` | halves the training state held on each GPU (about 33 GB to about 16 GB); expected peak about 60 GB. Untested by us. |
| 4 × H100 or A100 80 GB | `OFFLOAD=1` | moves training state to host memory; needs at least 256 GB of host RAM and runs slower. Untested by us, and tight. |

Using 8 GPUs instead of 4 does **not** change the experiment: 32 questions and 8 attempts per step are global settings, and the generation engine still uses 2 GPUs per copy. Because neither 80 GB path has been run by us, step 5 is a two-step smoke; after it, check the `max GB` column of the report stays under about 72. Roughly 23 GPU-hours on 4 × H200; expect 30 to 40 GPU-hours, and 5 to 6 hours of wall clock, on 8 × H100.

## 0. Get the kit (on the host, before starting the container)

Clone the kit onto the disk you will mount into the container, and pin it to the tag for this package so nothing shifts under you:

```bash
git clone https://github.com/hopit-ai/continual-learning-kit.git /your/fast/disk/continual-learning-kit
```

```bash
git -C /your/fast/disk/continual-learning-kit checkout kit-b0-v3
```

You need exactly three files from it, all in `kit/`: this README, `run_sdpo_toolalpaca.sh` and `make_report.py`. Nothing else in the repository is used. The clone appears inside the container at `/work/continual-learning-kit`.

## 1. Container

Use the image the reference documents and we used:

```bash
docker pull nvcr.io/nvidia/vllm:25.12.post1-py3
```

Start it with all four GPUs, a large shared-memory segment and a work directory mounted, for example:

```bash
docker run --gpus all --ipc=host --shm-size=64g -it -v /your/fast/disk:/work -w /work nvcr.io/nvidia/vllm:25.12.post1-py3 bash
```

Mount the same disk you cloned onto. Everything below runs inside that container. (On Slurm with enroot/pyxis or apptainer, use the same image.)

Set one variable so every later command is copy-paste:

```bash
export KIT=/work/continual-learning-kit/kit
```

Check it:

```bash
ls $KIT/run_sdpo_toolalpaca.sh $KIT/make_report.py
```

## 2. The reference code, pinned

```bash
git clone https://github.com/lasgroup/SDPO.git /work/SDPO
```

```bash
git -C /work/SDPO checkout 7c457fc1b1f636ae794eb0362ba37d4743b06fbc
```

```bash
pip install -e /work/SDPO
```

```bash
pip install 'antlr4-python3-runtime==4.9.3' 'math-verify==0.8.0' 'ray==2.53.0' 'torchdata==0.11.0' 'transformers==4.57.1'
```

Check the one import that cost us a paid run:

```bash
python -c "from math_verify import parse, verify; import verl, vllm, torch; print(torch.__version__, vllm.__version__)"
```

## 3. Data (already in the repository; only a format conversion)

```bash
cd /work/SDPO && PYTHONPATH=/work/SDPO python data/preprocess.py --data_source datasets/tooluse
```

This writes `datasets/tooluse/train.parquet` (4,046 rows) and `test.parquet` (68 rows). Check:

```bash
python -c "import pandas as pd; print(len(pd.read_parquet('/work/SDPO/datasets/tooluse/train.parquet')), len(pd.read_parquet('/work/SDPO/datasets/tooluse/test.parquet')))"
```

Expected output: `4046 68`.

## 4. Model, pinned revision

```bash
huggingface-cli download Qwen/Qwen3-8B --revision b968826d9c46dd6066d109eabc6255188de91218 --local-dir /work/models/Qwen3-8B
```

## 5. Smoke and calibration (about 20 minutes; do not skip)

Run two steps, straight from the clone:

```bash
SDPO_DIR=/work/SDPO MODEL_DIR=/work/models/Qwen3-8B WORK=/work/sdpo-work NAME=smoke-01 STEPS=2 bash $KIT/run_sdpo_toolalpaca.sh
```

Add `NGPU=8` (or `OFFLOAD=1` on a 4-card 80 GB node) exactly as in the hardware table, here and on every later command. Two things must be true before anything else runs:

- **Calibration.** The first validation, before any training, prints `val-core/tooluse/acc/mean@16`. Ours read 0.574 and 0.579 on two occasions. **Yours must be between 0.555 and 0.600.** If it is not, stop and send us `console.log`; the environments differ and no later number would be comparable.
- **Signal.** In `metrics.jsonl`, `self_distillation/success_group_fraction` should be above zero on both steps (over our 17 steps it ranged from 0.22 to 0.63, mean 0.39). Zero on both means the scorer or the prompt template is broken.

Every run gets a fresh `NAME`; the script refuses to overwrite.

## 6. The runs

**A. Two more seeds of our run 3** (17 steps each, about 35 minutes each):

```bash
SDPO_DIR=/work/SDPO MODEL_DIR=/work/models/Qwen3-8B WORK=/work/sdpo-work NAME=run3-seed43 SEED=43 STEPS=17 bash $KIT/run_sdpo_toolalpaca.sh
```

```bash
SDPO_DIR=/work/SDPO MODEL_DIR=/work/models/Qwen3-8B WORK=/work/sdpo-work NAME=run3-seed44 SEED=44 STEPS=17 bash $KIT/run_sdpo_toolalpaca.sh
```

**B. The paper's one-hour dose, three seeds** (40 steps, validation every 5 steps, about 80 minutes each):

```bash
SDPO_DIR=/work/SDPO MODEL_DIR=/work/models/Qwen3-8B WORK=/work/sdpo-work NAME=dose40-seed42 SEED=42 STEPS=40 TEST_FREQ=5 bash $KIT/run_sdpo_toolalpaca.sh
```

Repeat with `NAME=dose40-seed43 SEED=43` and `NAME=dose40-seed44 SEED=44`.

Do not change any other setting. Batch size, rollouts, learning rate, warm-up and step counts are what make the numbers comparable with the paper and with our run; if something does not fit in memory, use the hardware table's options and tell us, rather than shrinking a batch.

**When to stop a run early.** If `response_length/mean` in `metrics.jsonl` falls below 16 tokens and stays there, or a GPU has under 2 GB free, stop it and keep the directory. We lost $67 once to a collapse nobody was watching.

## 7. What to send back: one report

You do not need to upload anything large. After the runs (and after the smoke, if you want us to check it), generate the report:

```bash
python $KIT/make_report.py /work/sdpo-work/runs --out /work/sdpo-work/report
```

It writes two small text files, `report.md` and `report.json`, a few hundred KB together. **Send us both.** They contain, for every run: what was run (reference commit, library versions, GPUs, seed, the hash of the exact command), the held-out accuracy at every validation, the per-question results behind it, the paired change with its interval, the per-step training signals (reward, response length, sibling-success fraction, gradient norm, step time, peak memory), and automatic flags for a base accuracy outside the calibration band, a response-length collapse, steps with no teacher signal, or an incomplete run together with the end of its log.

The report is recomputed from the raw files every time, needs only the Python standard library, and never writes inside a run directory. We verified it against our own run: it reproduces our published 0.5790 → 0.6608, +0.0818, 12 better / 5 worse / 51 unchanged exactly.

If a run crashes, run the report anyway and send it; the failure is a result too. Anything you noticed that the files would not show (a restart, a node change, a setting you had to touch) belongs in the email, in your own words.

**Please keep the six final checkpoints on disk** (`runs/<NAME>/hf-step17/` and `hf-step40/`, 16 GB each, about 100 GB in total) until we send a second, small package that scores them for forgetting on your machine. You can delete every `tool-sdpo/` directory (the raw sharded checkpoints) once `merge.log` shows the merge finished.

## 8. What we will do with it

Three seeds at 17 steps and three at 40 steps turn "+8 points, interval from 0 to +16" into a number with an error bar. The checkpoints you keep give the first multi-seed forgetting measurement for SDPO at the paper's geometry once the scoring package arrives. Both go into the paper with your cluster acknowledged.
