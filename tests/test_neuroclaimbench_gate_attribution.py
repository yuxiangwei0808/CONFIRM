from nbs.analyze_neuroclaimbench_gate_attribution import analyze_records


def _record(
    item_id: str,
    *,
    disposition: str,
    cluster: str,
    failed: tuple[str, ...] = (),
):
    gates = {
        gate: gate not in failed
        for gate in (
            "search_provenance",
            "multiplicity",
            "confounding",
            "power",
            "multiverse",
            "replication",
        )
    }
    return {
        "benchmark_item_id": item_id,
        "semantic_cluster_id": cluster,
        "stratum": "scientific_literature",
        "reference_disposition": disposition,
        "confirmed": all(gates.values()),
        "gates": gates,
    }


def test_gate_attribution_reconciles_ladder_and_leave_one_out():
    records = [
        _record("confirmed", disposition="confirm", cluster="a"),
        _record(
            "replication-only",
            disposition="confirm",
            cluster="b",
            failed=("replication",),
        ),
        _record(
            "multiple-failures",
            disposition="confirm",
            cluster="c",
            failed=("multiplicity", "replication"),
        ),
    ]
    failures, leave_one_out, ladder = analyze_records(
        records,
        resamples=50,
        seed=7,
    )
    replication = next(
        row
        for row in leave_one_out
        if row["reference_disposition"] == "confirm"
        and row["removed_gate"] == "replication"
    )
    assert replication["baseline_confirmed_count"] == 1
    assert replication["counterfactual_confirmed_count"] == 2
    assert replication["added_confirmation_count"] == 1
    replication_failure = next(
        row
        for row in failures
        if row["reference_disposition"] == "confirm"
        and row["gate"] == "replication"
    )
    assert replication_failure["failed_count"] == 2
    assert replication_failure["exclusive_failure_count"] == 1
    final_stage = next(
        row
        for row in ladder
        if row["reference_disposition"] == "confirm"
        and row["stage_index"] == 6
    )
    assert final_stage["pass_count"] == 1
