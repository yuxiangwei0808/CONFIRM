---
pretty_name: NeuroClaimBench
language:
- en
license: cc-by-4.0
size_categories:
- n<1K
tags:
- neuroimaging
- scientific-reporting
- benchmark
- claim-validation
configs:
- config_name: cases
  default: true
  data_files:
  - split: test
    path: cases.jsonl
- config_name: references
  data_files:
  - split: test
    path: references.jsonl
- config_name: tasks
  data_files:
  - split: test
    path: tasks.jsonl
- config_name: outcomes
  data_files:
  - split: test
    path: outcomes.jsonl
---

# NeuroClaimBench v2.1

NeuroClaimBench evaluates whether a system supports or withholds a
neuroimaging claim.

The package contains 489 executable tasks: 279 scientific claims, 150
constructed stress controls, and 60 external-transfer claims and controls.
The inventory also includes 55 cases excluded from execution because of
ambiguous question-plan alignment, unavailable analysis inputs, or invalid designs.
Questions, analysis plans, reference labels, and recorded CONFIRM outcomes
are provided as JSONL files.

**Code:** [CONFIRM](https://github.com/yuxiangwei0808/CONFIRM) · **Dataset:** [Hugging Face](https://huggingface.co/datasets/stanjsx/NeuroClaimBench)

## Load from Hugging Face

Install the dataset library:

```bash
python -m pip install datasets
```

Load an individual record collection:

```python
from datasets import load_dataset

tasks = load_dataset("stanjsx/NeuroClaimBench", "tasks", split="test", revision="v2.1.1")
print(tasks[0]["task_id"])
print(tasks[0]["contract"])
```

The configurations are `cases` (544 rows), `references` (544), `tasks` (489),
and `outcomes` (489). Each uses a single `test` split. The default configuration
is `cases`; these are record collections for evaluation, with links described
in the [data dictionary](DATA_DICTIONARY.md).

To download the complete benchmark for the command-line tools:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="stanjsx/NeuroClaimBench",
    repo_type="dataset",
    revision="v2.1.1",
    local_dir="neuroclaimbench-v2.1",
)
```

## Install the tools

Use Python 3.10 or newer. CONFIRM supplies the `neuroclaimbench` command.
Download `confirm-0.1.1-py3-none-any.whl` from the
[CONFIRM v0.1.1 release](https://github.com/yuxiangwei0808/CONFIRM/releases/tag/v0.1.1)
into the directory containing the extracted `neuroclaimbench-v2.1` folder.
Run from that parent directory:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install ./confirm-0.1.1-py3-none-any.whl
```

To install from source,
extract `confirm-0.1.1.zip` and run `python -m pip install ./confirm-0.1.1`
from its parent directory. An existing CONFIRM installation provides the
same benchmark commands.

## Verify and score

Run from the directory containing the extracted benchmark folder:

```bash
neuroclaimbench verify --benchmark neuroclaimbench-v2.1
neuroclaimbench predictions --benchmark neuroclaimbench-v2.1 \
  --standard full --out outputs/confirm-full.jsonl
neuroclaimbench score --benchmark neuroclaimbench-v2.1 \
  --predictions outputs/confirm-full.jsonl
```

These commands verify all files, export 489 decisions from the recorded
gate results, and score agreement with the reference labels. They require
no cohort data or model calls. Choose a new output filename for each export.

## Guides

- [Python loading, metrics, and evaluation](USAGE.md)
- [File and field definitions](DATA_DICTIONARY.md)
- [Cohort data and exact-input requirements](DATA_ACCESS.md)

Executing a task requires its corresponding prepared cohort files. The
metadata package contains no participant tables, random-control assignment
tables, or PubMed abstracts.
