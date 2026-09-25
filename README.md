# Data for the continual-learning kit (branch `data-v1`)

Two datasets the kit's beds read, hosted here so a cluster that can reach github.com can fetch them
without a laptop in between. The kit itself is on `main` and the `kit-*` tags. This branch holds no code.

| folder | what | licence | upstream |
|---|---|---|---|
| `spider/` | Spider 1.0, `spider_data.zip` (the September 2024 re-release, 205,800,266 bytes) split into three parts under GitHub's 100 MB file limit | CC BY-SA 4.0 (Yu et al., 2018) | https://yale-lily.github.io/spider |
| `finqa/` | FinQA `dataset/train.json`, `dev.json`, `test.json` at commit `0f16e2867befa6840783e58be38c9efb9229d742` | MIT (Chen et al., 2021) | https://github.com/czyssrs/FinQA |

`MANIFEST.sha256` lists the sha256 of every file here and of the reassembled `spider_data.zip`. The kit's
`prepare` steps verify the databases and prompts against their own pinned hashes as well, so a wrong or
damaged copy is refused before any GPU is held.

## Fetch on the cluster

```bash
git clone --depth 1 --branch data-v1 https://github.com/hopit-ai/continual-learning-kit.git kit-data
cd kit-data
cat spider/spider_data.zip.part-* > spider_data.zip
shasum -a 256 -c <(grep 'spider_data.zip (reassembled' MANIFEST.sha256 | sed 's/ (reassembled.*//')
unzip -q spider_data.zip            # gives spider_data/ with database/, train_spider.json, train_others.json
export SPIDER_ROOT=$PWD/spider_data
export FINQA_ROOT=$PWD/finqa        # the folder holding train.json, dev.json, test.json
```

GSM8K is fetched from Hugging Face by the kit's own `prepare`; so is the LiveCodeBench file for the coding bed.

## Attribution

Spider: Tao Yu, Rui Zhang, Kai Yang, Michihiro Yasunaga, Dongxu Wang, Zifan Li, James Ma, Irene Li, Qingning Yao,
Shanelle Roman, Zilin Zhang, Dragomir Radev. "Spider: A Large-Scale Human-Labeled Dataset for Complex and
Cross-Domain Semantic Parsing and Text-to-SQL Task." EMNLP 2018. Redistributed unchanged under CC BY-SA 4.0.

FinQA: Zhiyu Chen, Wenhu Chen, Charese Smiley, Sameena Shah, Iana Borova, Dylan Langdon, Reema Moussa, Matt Beane,
Ting-Hao Huang, Bryan Routledge, William Yang Wang. "FinQA: A Dataset of Numerical Reasoning over Financial Data."
EMNLP 2021. Redistributed unchanged under the MIT licence in `finqa/LICENSE`.
