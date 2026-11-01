# Using CONFIRM

## The workflow

A claim contract is a machine-readable analysis plan. It fixes the question,
target association or group comparison, cohorts, covariates, search history,
and pass criteria. The LLM can draft the plan and propose follow-up plans;
executed code computes the statistics and gate decisions.

`confirm ask` drafts a plan from a question and the available data columns.
It asks you to approve that plan, executes it, and writes an interpretation.
`confirm run` starts from an existing YAML or JSON plan. Both commands accept
`--claim-search on` for follow-up analysis after a failed claim.

## Reporting standards

| CLI option | Required checks | Python API value |
|---|---|---|
| `--standard screen` | Search history, multiplicity, declared and checked confounds | `discovery` |
| `--standard replicate` | Screen checks plus replication | `replicated` |
| `--standard full` | Replicate checks plus power and analytic stability | `confirmed` |

All gates are computed. The selected standard determines the reporting
decision from those results. `support_decision.supported` is the boolean to
use for that decision. `label` gives the Full-standard diagnosis; for
example, a claim labeled `non_replicated` passes Screen if its search-history
and confound checks pass. Use the values in the final column when calling
the Python API.

The CLI exits with 0 for support and 1 for withholding. `--accept-label`
overrides the exit-status rule with a list of accepted diagnostic labels;
the reporting decision still follows `--standard`. Follow-up search uses
Full-standard support as its stopping criterion for all three standards.

## Analysis plans

Run `confirm demo --out outputs/demo` to obtain complete example plans.
Each plan specifies:

- `estimand`: the association or group difference being estimated, its outcome,
  predictor, expected direction, and scalar or brain-wide analysis unit;
- `covariates` and `inclusion`: adjustment variables and an optional subgroup;
- `discovery_cohort` and `replication_cohorts`: exact input filenames without
  the `.parquet` extension;
- `search_provenance`: how the hypothesis was selected and the search family
  size;
- `gates`: correction, confound, power, stability, and replication criteria.

Freeze these choices before executing the analysis. Record earlier
hypothesis searches and their family sizes in `search_provenance`.
For the power gate, `ref_effect`
lets you supply the reference effect used by the implementation.

`harmonize: combat` attempts ComBat and logs a fallback to additive site-effect
removal if the data or library cannot support it. `harmonize: none` disables
that transformation. The synthetic quickstart uses `none`.

## Outputs

Each run writes `receipt.json` with the plan and cohort hashes, package
versions, computed results, gate decisions, and the selected reporting
standard. Undefined numerical diagnostics are stored as JSON `null`.
`confirm ask` saves the analysis receipt before requesting narration and
also writes `drafted_contract.yaml` and
`narrative.txt`. Follow-up search adds its proposals, validation decisions,
and execution records to the output directory. Use a separate output
directory for each analysis; existing output directories are rejected.

Support means the specified reporting requirements passed. Literature
reference agreement is measured separately by NeuroClaimBench.

## Integrate another agent

Have the agent produce a `ClaimContract` before analysis, then evaluate that
contract against the declared cohorts:

```python
from pathlib import Path
from confirm import load_contract
from confirm.execution import evaluate_contract

contract = load_contract("outputs/demo/contracts/supported.yaml")
verdict, results, inputs = evaluate_contract(
    contract,
    Path("outputs/demo/cohorts"),
    ref_effect=contract.gates.power.ref_effect,
    minimum_evidence_tier="confirmed",
)
decision = results["support_decision"]
print(decision.supported)
print(decision.failed_required_gates)
print(verdict.rationale)
```

Supply the plan, declared search history, and cohort tables from the
external agent. `evaluate_contract` executes the analysis and returns its
gate results and reporting decision. The complete command-line example is
[examples/audit_contract.py](../examples/audit_contract.py).

## Configure an LLM

Install `.[llm]` and set `CONFIRM_LLM=provider:model` using a model available
to your account. Provider variables are:

| Provider | Credential variable |
|---|---|
| `openai` | `OPENAI_API_KEY` |
| `anthropic` | `ANTHROPIC_API_KEY` |
| `google` or `gemini` | `GEMINI_API_KEY` or `GOOGLE_API_KEY` |
| `openrouter` | `OPENROUTER_API_KEY` |

Supply credentials through the environment or a local `.env` file, which
is excluded by `.gitignore`. The package loads `.env` when an LLM client
is initialized. `CONFIRM_LLM_TIMEOUT` sets the request timeout in seconds
(default: 60). Follow-up defaults are 3 rounds and 5 candidates per round; use
`--claim-search-max-rounds` and `--claim-search-max-candidates` to change them.

Inspect and approve the drafted plan before execution. Each model call uses
the selected provider account and its billing terms.
