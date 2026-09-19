# Package K1a: did the K0 models forget?

**What this is.** A 25-minute job on **one GPU**. It scores the untrained Qwen3-8B twice, then each of the five models your K0 runs produced, on 300 fixed questions (100 maths, 100 general knowledge, 100 instruction following), and writes one small report. It answers: did learning the tool-use task cost the model any of its general ability?

**Why the untrained model is scored twice.** We found that ordinary "greedy" scoring is not repeatable: two scorings of one model on one machine agreed on as few as 98 of 300 answers. The scorer therefore runs in a deterministic mode by default (no compilation, fixed seed, batch-invariant kernels), in which our two scorings agreed on 300 of 300. The first three rows prove the same holds on your machine; if they do not, the campaign refuses to go on and nothing is wasted.

## Run it

Inside the same container and environment as K0 (the authors' code installed, as in `README-partner.md` sections 1 and 2):

```bash
export KIT=/work/continual-learning-kit/kit WORK=/work/sdpo-work MODEL_DIR=/work/models/Qwen3-8B
```

```bash
python $KIT/runner.py plan $KIT/campaigns/k1a-forgetting-of-k0.yaml
```

```bash
python $KIT/runner.py prepare $KIT/campaigns/k1a-forgetting-of-k0.yaml --all
```

`prepare` needs no GPU. It checks that the untrained model and the five K0 models are on disk where the campaign expects them (`$WORK/runs/<name>/hf-step17` or `hf-step40`). If your run directories have other names, edit the five `requires` and `--model` paths in the campaign file, then run `prepare` again.

```bash
python $KIT/runner.py run $KIT/campaigns/k1a-forgetting-of-k0.yaml --all
```

It uses GPU 0 only, on purpose: every model has to be scored on the same GPU.

## What to send back

One folder of small text files: `$WORK/forgetting/report-a1/` (`forgetting-report.md` and `.json`), plus `$WORK/forgetting/repeatable-a1/agreement.json`. If a row is refused or fails, send the output of:

```bash
python $KIT/runner.py status $KIT/campaigns/k1a-forgetting-of-k0.yaml
```

## What we expect, so you can tell if something is off

| Row | On our machines |
|---|---|
| `base-1` | 252 of 300 correct (maths 91, knowledge 74, instruction following 87). The bar is 245 to 257 |
| `repeatable` | 300 of 300 identical answers, no changed verdict. The bar is at least 295, and none changed |
| a healthy trained model | within 3 of the untrained model on every panel. Our own run 3 scored +2, +1, -1 |
