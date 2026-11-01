# NeuroClaimBench evaluation guide

Install CONFIRM using the [quickstart](README.md). Run the commands below
from the benchmark directory containing `cases.jsonl`. In the CONFIRM source
package, that directory is `benchmark/neuroclaimbench-v2.1/`.

## Verify the benchmark

```bash
neuroclaimbench verify --benchmark .
```

Verification checks SHA-256 checksums, record schemas, unique IDs, links
between records, and inventory counts. The `summary` command performs the
same verification and prints the inventory.

| Track | Inventory cases | Executable tasks |
|---|---:|---:|
| Scientific claims | 332 | 279 |
| Constructed stress controls | 150 | 150 |
| External-transfer claims and controls | 62 | 60 |
| Total | 544 | 489 |

Each executable case has one task. The 55 excluded cases comprise 40 with
ambiguous question-plan alignment and 15 with unavailable analysis inputs
or invalid designs. The benchmark has 268 score-eligible references:
96 literature references and 172 constructed controls. The 221 executable
tasks with unresolved references contribute to coverage counts and are
excluded from reference-agreement metrics.

## Load records in Python

```python
from bench.benchmark import BenchmarkDataset

benchmark = BenchmarkDataset(".")
cases = {case.benchmark_case_id: case for case in benchmark.cases}
references = {ref.reference_id: ref for ref in benchmark.references}

task = benchmark.tasks[0]
case = cases[task.benchmark_case_id]
reference = references[case.reference_id]
print(case.question)
print(task.contract.model_dump())
print(reference.disposition, reference.score_eligible)
```

The loader validates record schemas. Use `neuroclaimbench verify` for
checksum and cross-record checks. Join tasks and outcomes on `task_id`;
join a task to its case through `benchmark_case_id`, then to its reference
through `reference_id`. The [data dictionary](DATA_DICTIONARY.md) defines
the files and fields.

## Score recorded CONFIRM decisions

```bash
neuroclaimbench predictions --benchmark . \
  --standard full --out ../outputs/confirm-full.jsonl
neuroclaimbench score --benchmark . \
  --predictions ../outputs/confirm-full.jsonl
```

Choose `screen`, `replicate`, or `full`. The predictions command applies the
selected standard to the recorded gate vectors. It writes a new file and
exports an error row for any outcome lacking a completed gate vector.

## Score another system

Give your system the question, frozen analysis plan, and required data.
Reserve the reference labels and recorded CONFIRM outcomes for evaluation.
Write one JSON object per task to a JSONL file:

```json
{"task_id": "<exact task_id from tasks.jsonl>", "status": "completed", "supported": false}
```

Use a JSON boolean for `supported`. For an execution failure or a missing
verdict, use an error row:

```json
{"task_id": "<exact task_id from tasks.jsonl>", "status": "error", "error": "Input cohort unavailable"}
```

Pass the file to `neuroclaimbench score --benchmark . --predictions FILE`.
The scorer requires unique, known task IDs and complete coverage by default.
`--allow-partial` scores a subset and reports omitted tasks. Use complete
predictions when comparing systems.

### Metrics and error handling

| Metric | Numerator | Denominator |
|---|---|---|
| Literature recall | Supported decisions | Evaluated literature references labeled `confirm` |
| Literature false support | Supported decisions | Evaluated literature references labeled `abstain` |
| Control false support | Supported decisions | Evaluated constructed controls |

Under the paper's scoring rule, execution errors count as withheld and
remain in the evaluated denominator. Output reports their count separately
from completed decisions. Missing tasks are excluded from denominators
when `--allow-partial` is specified.

Metrics are grouped by track. Literature metrics are also grouped by
reference strength: `strict`, `provisional`, and `combined`. Each metric
reports `supported`, `evaluated`, `completed`, `eligible`, `errors`, and
`missing` counts. `evaluated` includes submitted error rows; `completed`
counts valid decisions. A rate is `null` when its denominator is zero.
The output's `partial` flag indicates missing tasks or execution errors.

The scorer returns point estimates. The distributed summary tables also
provide the paper's Wilson and cluster-bootstrap intervals for recorded
results.

### Match the paper's scalar comparison

The main scientific comparison uses 46 scalar literature claims: 31 labeled
`confirm` and 15 labeled `abstain`. The default `all` scope also includes
brain-wide claims, giving 51 and 19 scientific literature references,
respectively. Select the scalar scope to match the main comparison:

```bash
neuroclaimbench score --benchmark . \
  --predictions ../outputs/confirm-full.jsonl --literature-scope scalar
```

This filters literature metrics to scalar tasks and includes all constructed
controls. Supply all 489 predictions; the scorer selects the metric subsets.
Expected results for the scientific scalar set and the 150 stress controls:

| Standard | Literature recall | Literature false support | Control false support |
|---|---:|---:|---:|
| Screen | 19/31 | 6/15 | 5/150 |
| Replicate | 15/31 | 2/15 | 1/150 |
| Full | 15/31 | 2/15 | 0/150 |

Literature reference agreement and CONFIRM support measure different things:
references summarize the adjudicated literature evidence, while CONFIRM
support means the selected reporting checks passed.

## Execute tasks

Obtain the prepared inputs specified in the
[data-access guide](DATA_ACCESS.md), then run:

```bash
neuroclaimbench run --benchmark . --data-root path/to/frozen-cohorts \
  --standard full --out ../outputs/rerun
```

Supply additional `--data-root` arguments for separate directories. Each
task's discovery and replication files must be together in one supplied
directory. Repeat `--task-id` to select specific tasks. Use a new output
directory for each run.

The command verifies cohort hashes before execution and writes per-task
results and `predictions.jsonl`. Unavailable or mismatched inputs produce
error rows. Exit code 1 indicates at least one failed task. Score the output
with the same `score` command; include `--allow-partial` when running a subset.

## File integrity and provenance

`SHA256SUMS` covers every distributed file except itself. The manifests
identify the package contents, schema, and record hashes. Source paths and
hashes inside scientific records identify the artifacts used to construct
the benchmark. Runtime input files are located through `--data-root`.
