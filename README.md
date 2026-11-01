# CONFIRM

CONFIRM is an agentic framework for scientific reporting in neuroimaging.
It drafts analysis plans, executes analyses, checks the results against fixed
methodological requirements, and uses failed-gate diagnoses to guide follow-up
analyses. It also evaluates plans supplied by other agents.

The framework operates on tables of imaging-derived measurements. Its five
gates cover search history and multiplicity, confounds, power, analytic
stability, and replication. Screen, Replicate, and Full select which gates a
result must pass before it is reported as supported.

**Paper:** [CONFIRM: Governing Scientific Reporting for Trustworthy Agentic Neuroimaging](https://openreview.net/forum?id=KYE1DzsxWc)

**Dataset:** [NeuroClaimBench on Hugging Face](https://huggingface.co/datasets/stanjsx/NeuroClaimBench)

## Install

Use Python 3.10 or newer. Clone the repository:

```bash
git clone https://github.com/yuxiangwei0808/CONFIRM.git
cd CONFIRM
```

From the repository or extracted source directory:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install .
```

On Windows, activate with `.venv\Scripts\activate`.

| Installation | Use |
|---|---|
| `python -m pip install .` | Prepared analysis plans, synthetic demo, benchmark scoring |
| `python -m pip install '.[llm]'` | LLM drafting and follow-up analysis |
| `python -m pip install '.[imaging]'` | Dataset fetching and image/table imports |
| `python -m pip install '.[dev]'` | Development and tests |

The [release assets](https://github.com/yuxiangwei0808/CONFIRM/releases/tag/v0.1.0) include an installable
wheel with the same commands: install it with
`python -m pip install ./confirm-0.1.0-py3-none-any.whl`. The source ZIP also
contains the guides, examples, tests, and benchmark metadata.

## Try a complete local example

```bash
confirm demo --out outputs/demo
```

This creates two synthetic cohorts, three analysis plans, and a result receipt
for each plan. It requires no API key and downloads no data. Use a new output
directory for each demo run.

| Example | Screen | Replicate | Full |
|---|---|---|---|
| Planted association in both cohorts | Supported | Supported | Supported |
| Association explained by an omitted site effect | Withheld | Withheld | Withheld |
| Association present only in discovery | Supported | Withheld | Withheld |

Inspect `outputs/demo/summary.json`, the YAML plans in
`outputs/demo/contracts/`, and the receipts in `outputs/demo/results/`.
These synthetic examples illustrate how each reporting standard responds
to a planted association, a confound, and a replication failure.

Run a plan yourself:

```bash
confirm run --contract outputs/demo/contracts/supported.yaml \
  --data-dir outputs/demo/cohorts --standard full --out outputs/my-claim
```

The command prints the gate results and reporting decision. Exit code 0 means
the selected standard supports the claim; 1 means it withholds the claim.

## Use an LLM agent

Install the `llm` extra, set your provider's API key in your environment, and
set `CONFIRM_LLM` to `provider:model`. Supported providers are `openai`,
`anthropic`, `google`, and `openrouter`. Then run:

```bash
confirm ask "Does cortical thickness decrease with age?" \
  --data-dir path/to/cohorts --standard full --out outputs/age-question
```

CONFIRM displays the drafted plan for approval before execution. Add
`--claim-search on` to enable bounded follow-up search after a failed claim.
These commands call the selected model provider. See the
[framework guide](docs/FRAMEWORK.md) for configuration and output details.

## NeuroClaimBench

The included [NeuroClaimBench v2.1](benchmark/neuroclaimbench-v2.1/README.md)
contains 489 executable tasks, frozen reference decisions, and compact
CONFIRM results. Validate the files and inspect the inventory:

```bash
neuroclaimbench verify --benchmark benchmark/neuroclaimbench-v2.1
```

Export the frozen CONFIRM-Full decisions and score them without cohort data
or model calls:

```bash
neuroclaimbench predictions --benchmark benchmark/neuroclaimbench-v2.1 \
  --standard full --out outputs/confirm-full.jsonl
neuroclaimbench score --benchmark benchmark/neuroclaimbench-v2.1 \
  --predictions outputs/confirm-full.jsonl
```

The [benchmark guide](docs/NEUROCLAIMBENCH.md) explains how to score another
system and rerun tasks with the corresponding prepared data.

## Guides

- [Framework, reporting standards, and agent integration](docs/FRAMEWORK.md)
- [Preparing your own data](docs/DATA.md)
- [Benchmark format, evaluation, and scoring](docs/NEUROCLAIMBENCH.md)
- [Release notes](RELEASE_NOTES.md)

## Development

```bash
python -m pip install '.[dev]'
python -m pytest
```

The release test suite uses synthetic inputs and requires no model credentials
or cohort downloads. `python scripts/package_public_release.py` creates the
source and benchmark ZIPs using an explicit file list.

## Citation and licenses

Use [CITATION.cff](CITATION.cff) to cite the software and paper. Code and
framework documentation are licensed under [Apache-2.0](LICENSE).
NeuroClaimBench metadata and its accompanying documentation are licensed under
[CC BY 4.0](benchmark/neuroclaimbench-v2.1/LICENSE). The benchmark license
covers the authors' contributions; source datasets and third-party works
retain their own terms. Raw cohort data and PubMed abstracts are not included.
