from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pandas as pd

from bench import run_initial_claim_drafting as drafting
from bench import run_literature_claim_grounding as grounding


def _write_smri_cohorts(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    n = 30
    for cohort in ("ADNI_DISC", "OASIS3_REP"):
        table = pd.DataFrame(
            {
                "subject_id": [f"{cohort}-{idx:03d}" for idx in range(n)],
                "cohort": [cohort] * n,
                "site": ["site1"] * n,
                "age": [65 + idx for idx in range(n)],
                "sex": ["M" if idx % 2 == 0 else "F" for idx in range(n)],
                "dx": ["CN" if idx < n // 2 else "Dementia" for idx in range(n)],
                "eTIV": [1500.0 + idx for idx in range(n)],
                "smri_hippocampus": [4.0 - idx * 0.02 for idx in range(n)],
            }
        )
        table.to_parquet(root / f"{cohort}.parquet", index=False)


def _write_fc_cohorts(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    n = 30
    for cohort in ("ABCD_DISC", "HCP_Aging_REP"):
        table = pd.DataFrame(
            {
                "subject_id": [f"{cohort}-{idx:03d}" for idx in range(n)],
                "cohort": [cohort] * n,
                "site": ["site1"] * n,
                "age": [12 + idx for idx in range(n)],
                "sex": ["M" if idx % 2 == 0 else "F" for idx in range(n)],
                "dx": ["CN"] * n,
                "fc_fc_Default_Default": [0.1 + idx * 0.001 for idx in range(n)],
                "fc_fc_Default_DorsAttn": [0.2 + idx * 0.001 for idx in range(n)],
            }
        )
        table.to_parquet(root / f"{cohort}.parquet", index=False)


def _write_asd_mixed_fc_cohorts(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    n = 32
    for cohort in ("ABIDE1_DISC", "ABIDE1_REP"):
        pd.DataFrame(
            {
                "subject_id": [f"{cohort}-{idx:03d}" for idx in range(n)],
                "cohort": [cohort] * n,
                "site": ["site1"] * n,
                "age": [12 + idx for idx in range(n)],
                "sex": ["M" if idx % 2 == 0 else "F" for idx in range(n)],
                "dx": ["ASD" if idx % 2 == 0 else "HC" for idx in range(n)],
                "fc_mean_abs": [0.1 + idx * 0.001 for idx in range(n)],
                "fc_mean_positive": [0.2 + idx * 0.001 for idx in range(n)],
            }
        ).to_parquet(root / f"{cohort}.parquet", index=False)
    for cohort in ("ABIDE2_DISC", "ABIDE2_REP"):
        pd.DataFrame(
            {
                "subject_id": [f"{cohort}-{idx:03d}" for idx in range(n)],
                "cohort": [cohort] * n,
                "site": ["site1"] * n,
                "age": [12 + idx for idx in range(n)],
                "sex": ["M" if idx % 2 == 0 else "F" for idx in range(n)],
                "dx": ["1" if idx % 2 == 0 else "2" for idx in range(n)],
                "fc_fc_Default_Default": [0.1 + idx * 0.001 for idx in range(n)],
                "fc_fc_Default_DorsAttn": [0.2 + idx * 0.001 for idx in range(n)],
            }
        ).to_parquet(root / f"{cohort}.parquet", index=False)


def _write_adhd_label_cohorts(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    n = 40
    specs = {
        "ABCD_DISC": ["0.0", "1.0"],
        "ADHD200_REP": ["0", "1", "2", "3"],
    }
    for cohort, levels in specs.items():
        pd.DataFrame(
            {
                "subject_id": [f"{cohort}-{idx:03d}" for idx in range(n)],
                "cohort": [cohort] * n,
                "site": ["site1"] * n,
                "age": [10 + idx for idx in range(n)],
                "sex": ["M" if idx % 2 == 0 else "F" for idx in range(n)],
                "dx": [levels[idx % len(levels)] for idx in range(n)],
                "fc_fc_Default_Default": [0.1 + idx * 0.001 for idx in range(n)],
                "fc_fc_Default_DorsAttn": [0.2 + idx * 0.001 for idx in range(n)],
            }
        ).to_parquet(root / f"{cohort}.parquet", index=False)


def _write_record(path: Path) -> None:
    record = grounding.PubMedRecord(
        pmid="12345678",
        title="Hippocampal atrophy in Alzheimer disease",
        abstract="Alzheimer disease patients showed reduced hippocampal volume compared with controls after age and sex adjustment.",
        journal="Example Journal",
        year="2020",
        doi="10.0000/example",
        mesh_terms=["Alzheimer Disease", "Hippocampus"],
        query="Alzheimer hippocampal MRI",
        target_family="ad_aging",
        modality="sMRI",
        retrieved_at="2026-07-01T00:00:00",
    )
    path.write_text(json.dumps(record.model_dump(mode="json")) + "\n", encoding="utf-8")


def test_parse_pubmed_xml_extracts_abstract_record() -> None:
    xml = """
    <PubmedArticleSet>
      <PubmedArticle>
        <MedlineCitation>
          <PMID>123</PMID>
          <Article>
            <ArticleTitle>Example title</ArticleTitle>
            <Abstract><AbstractText>Example abstract text.</AbstractText></Abstract>
            <Journal><Title>Example Journal</Title><JournalIssue><PubDate><Year>2021</Year></PubDate></JournalIssue></Journal>
          </Article>
          <MeshHeadingList><MeshHeading><DescriptorName>Hippocampus</DescriptorName></MeshHeading></MeshHeadingList>
        </MedlineCitation>
        <PubmedData><ArticleIdList><ArticleId IdType="doi">10.1/test</ArticleId></ArticleIdList></PubmedData>
      </PubmedArticle>
    </PubmedArticleSet>
    """

    records = grounding._parse_pubmed_xml(
        xml,
        query="query",
        target_family="ad_aging",
        modality="sMRI",
        retrieved_at="now",
    )

    assert len(records) == 1
    assert records[0].pmid == "123"
    assert records[0].doi == "10.1/test"
    assert records[0].mesh_terms == ["Hippocampus"]


def test_literature_grounding_writes_stage1_compatible_claims(tmp_path: Path) -> None:
    data_root = tmp_path / "cohorts"
    records = tmp_path / "records.jsonl"
    claims_out = tmp_path / "literature_grounded_claims.csv"
    out_dir = tmp_path / "out"
    _write_smri_cohorts(data_root)
    _write_record(records)

    args = argparse.Namespace(
        model="standin",
        target_family=["ad_aging"],
        out_dir=str(out_dir),
        claims_out=str(claims_out),
        records_jsonl=str(records),
        max_records_per_query=20,
        max_claims_per_record=2,
        schema_retries=0,
        llm_max_tokens=8192,
        pubmed_email="",
        pubmed_api_key="",
        pubmed_timeout=30.0,
        pubmed_delay=0.0,
        data_root=[str(data_root)],
        no_progress=True,
    )

    summary = grounding.run(args)

    assert summary["n_pubmed_records"] == 1
    assert summary["n_extracted_claim_seeds"] == 1
    assert summary["n_executable_claim_questions"] == 1
    assert summary["status_counts"] == {"executable_now": 1}
    assert (out_dir / "pubmed_records.jsonl").exists()
    assert (out_dir / "llm_literature_extraction_prompts.jsonl").exists()
    assert (out_dir / "literature_claim_feasibility.csv").exists()

    rows = list(csv.DictReader(claims_out.open()))
    assert rows[0]["source_mode"] == "literature_grounded"
    assert rows[0]["label_basis"] == "pubmed_literature"
    assert rows[0]["source_citation"].startswith("PMID:12345678")
    assert "Source evidence:" not in rows[0]["question"]
    assert "evidence_snippet=" in rows[0]["notes"]
    assert rows[0]["discovery_cohort"] == "ADNI_DISC"
    assert rows[0]["replication_cohorts"] == "OASIS3_REP"
    assert set(rows[0]["allowed_covariates"].split(";")) == {"age", "sex", "eTIV"}
    assert "smri_hippocampus" in rows[0]["shared_outcome_columns_sample"]
    assert rows[0]["shared_outcome_prefixes"] == "smri_"
    assert rows[0]["group_var"] == "confirm_dx"
    assert rows[0]["case_label"] == "case"
    assert rows[0]["control_label"] == "control"
    assert rows[0]["source_pmid"] == "12345678"
    assert rows[0]["source_seed_id"]
    loaded = drafting.load_claim_questions("literature_grounded", claims_out, tmp_path / "missing.csv")
    assert len(loaded) == 1
    assert loaded[0].source_mode == "literature_grounded"
    assert loaded[0].discovery_cohort == "ADNI_DISC"


def test_feasibility_marks_unavailable_modality_as_adapter_needed(tmp_path: Path) -> None:
    data_root = tmp_path / "cohorts"
    _write_smri_cohorts(data_root)
    catalog = drafting._merge_catalogs([data_root])
    seed = grounding.LiteratureClaimSeed(
        seed_id="pet_seed",
        source_pmid="1",
        target_family="ad_aging",
        outcome_modality="FDG PET",
        predictor_or_group="Alzheimer disease diagnosis versus controls",
        outcome_family="FDG hypometabolism",
        expected_direction="negative",
        covariates=["age", "sex"],
        candidate_question="AD is associated with FDG hypometabolism.",
        evidence_snippet="FDG hypometabolism was observed.",
        support_level="direct",
        rationale="PET claim.",
    )

    result = grounding.evaluate_feasibility([seed], {}, catalog)[0]

    assert result.status == "requires_new_feature_adapter"


def test_feasibility_rejects_dti_structure_function_coupling_even_with_fc_data(tmp_path: Path) -> None:
    data_root = tmp_path / "cohorts"
    _write_fc_cohorts(data_root)
    catalog = drafting._merge_catalogs([data_root])
    seed = grounding.LiteratureClaimSeed(
        seed_id="dti_seed",
        source_pmid="2",
        target_family="normative_fmri",
        outcome_modality="DTI and resting-state fMRI",
        predictor_or_group="age",
        outcome_family="brain structure-function coupling measured by C-BSF index",
        expected_direction="negative",
        covariates=["sex", "site"],
        candidate_question="Age is associated with DTI and resting-state fMRI structure-function coupling.",
        evidence_snippet="C-BSF index changed with age.",
        support_level="direct",
        rationale="Requires DTI and fMRI coupling.",
    )

    result = grounding.evaluate_feasibility([seed], {}, catalog)[0]

    assert result.status == "requires_new_feature_adapter"
    assert "diffusion" in result.reason.lower() or "structure-function" in result.reason.lower()


def test_feasibility_rejects_task_fmri_and_graph_metrics(tmp_path: Path) -> None:
    data_root = tmp_path / "cohorts"
    _write_fc_cohorts(data_root)
    catalog = drafting._merge_catalogs([data_root])
    seeds = [
        grounding.LiteratureClaimSeed(
            seed_id="task_seed",
            source_pmid="3",
            target_family="normative_fmri",
            outcome_modality="task fMRI functional connectivity",
            predictor_or_group="semantic task condition versus control",
            outcome_family="semantic-network functional connectivity",
            expected_direction="positive",
            covariates=["age", "sex", "in-scanner task"],
            candidate_question="Semantic task condition changes functional connectivity.",
            evidence_snippet="Task fMRI connectivity changed during semantic task.",
            support_level="direct",
            rationale="Requires task fMRI.",
        ),
        grounding.LiteratureClaimSeed(
            seed_id="graph_seed",
            source_pmid="4",
            target_family="normative_fmri",
            outcome_modality="fMRI-FC",
            predictor_or_group="age",
            outcome_family="graph theory and system segregation metrics",
            expected_direction="negative",
            covariates=["sex"],
            candidate_question="Age is associated with graph metrics.",
            evidence_snippet="Graph-theory metrics varied with age.",
            support_level="direct",
            rationale="Requires graph metrics.",
        ),
    ]

    results = grounding.evaluate_feasibility(seeds, {}, catalog)

    assert [result.status for result in results] == ["requires_new_feature_adapter", "requires_new_feature_adapter"]
    assert "task-fmri" in results[0].reason.lower()
    assert "graph" in results[1].reason.lower()


def test_feasibility_rejects_csf_tau_predictors(tmp_path: Path) -> None:
    data_root = tmp_path / "cohorts"
    _write_smri_cohorts(data_root)
    catalog = drafting._merge_catalogs([data_root])
    seed = grounding.LiteratureClaimSeed(
        seed_id="tau_seed",
        source_pmid="5",
        target_family="ad_aging",
        outcome_modality="sMRI",
        predictor_or_group="CSF p-tau level",
        outcome_family="regional brain volume",
        expected_direction="negative",
        covariates=["age", "sex", "eTIV"],
        candidate_question="CSF p-tau is associated with regional brain volume.",
        evidence_snippet="CSF p-tau correlated with regional volume.",
        support_level="direct",
        rationale="Requires CSF biomarker predictor.",
    )

    result = grounding.evaluate_feasibility([seed], {}, catalog)[0]

    assert result.status == "requires_new_feature_adapter"
    assert "tau" in result.reason.lower()


def test_feasibility_rejects_microbleed_outcomes(tmp_path: Path) -> None:
    data_root = tmp_path / "cohorts"
    _write_smri_cohorts(data_root)
    catalog = drafting._merge_catalogs([data_root])
    seed = grounding.LiteratureClaimSeed(
        seed_id="microbleed_seed",
        source_pmid="7",
        target_family="ad_aging",
        outcome_modality="sMRI",
        predictor_or_group="Alzheimer pathology",
        outcome_family="cerebral microbleeds",
        expected_direction="positive",
        covariates=["age", "sex"],
        candidate_question="Alzheimer pathology is associated with cerebral microbleeds.",
        evidence_snippet="Cerebral microbleeds were associated with pathology.",
        support_level="direct",
        rationale="Requires microbleed features.",
    )

    result = grounding.evaluate_feasibility([seed], {}, catalog)[0]

    assert result.status == "requires_new_feature_adapter"
    assert "microbleed" in result.reason.lower()


def test_feasibility_rejects_imaging_predictor_behavior_outcome(tmp_path: Path) -> None:
    data_root = tmp_path / "cohorts"
    _write_fc_cohorts(data_root)
    catalog = drafting._merge_catalogs([data_root])
    seed = grounding.LiteratureClaimSeed(
        seed_id="brain_behavior_seed",
        source_pmid="8",
        target_family="normative_fmri",
        outcome_modality="behavioral cognition with fMRI-FC predictor",
        predictor_or_group="resting-state fMRI functional connectivity",
        outcome_family="executive function performance",
        expected_direction="positive",
        covariates=["age", "sex"],
        candidate_question="Functional connectivity predicts executive function.",
        evidence_snippet="Connectivity predicted executive function.",
        support_level="direct",
        rationale="Brain-to-behavior direction is not executable here.",
    )

    result = grounding.evaluate_feasibility([seed], {}, catalog)[0]

    assert result.status == "requires_new_feature_adapter"
    assert "brain-to-behavior" in result.reason.lower()


def test_feasibility_rejects_cerebellar_fc_without_local_network(tmp_path: Path) -> None:
    data_root = tmp_path / "cohorts"
    _write_fc_cohorts(data_root)
    catalog = drafting._merge_catalogs([data_root])
    seed = grounding.LiteratureClaimSeed(
        seed_id="cerebellar_seed",
        source_pmid="9",
        target_family="normative_fmri",
        outcome_modality="resting-state fMRI functional connectivity",
        predictor_or_group="age",
        outcome_family="cerebello-cortical connectivity",
        expected_direction="two_sided",
        covariates=["sex"],
        candidate_question="Age is associated with cerebello-cortical connectivity.",
        evidence_snippet="Cerebello-cortical connectivity varied with age.",
        support_level="direct",
        rationale="Requires cerebellar FC network.",
    )

    result = grounding.evaluate_feasibility([seed], {}, catalog)[0]

    assert result.status == "requires_new_feature_adapter"
    assert "cerebellar" in result.reason.lower()


def test_feasibility_keeps_plain_resting_state_fc_claim_executable(tmp_path: Path) -> None:
    data_root = tmp_path / "cohorts"
    _write_fc_cohorts(data_root)
    catalog = drafting._merge_catalogs([data_root])
    seed = grounding.LiteratureClaimSeed(
        seed_id="fc_seed",
        source_pmid="6",
        target_family="normative_fmri",
        outcome_modality="resting-state fMRI-FC",
        predictor_or_group="age",
        outcome_family="functional connectivity",
        expected_direction="two_sided",
        covariates=["sex", "site", "in-scanner task"],
        candidate_question="Age is associated with resting-state functional connectivity.",
        evidence_snippet="Resting-state functional connectivity varied with age.",
        support_level="direct",
        rationale="Uses local FC descriptors.",
    )

    result = grounding.evaluate_feasibility([seed], {}, catalog)[0]
    question = grounding._claim_question(seed, result)

    assert result.status == "executable_now"
    assert result.discovery_cohort == "ABCD_DISC"
    assert result.replication_cohort == "HCP_Aging_REP"
    assert "in-scanner task" not in question


def test_asd_cross_pair_rejected_when_fc_feature_families_do_not_overlap(tmp_path: Path) -> None:
    data_root = tmp_path / "cohorts"
    _write_asd_mixed_fc_cohorts(data_root)
    catalog = drafting._merge_catalogs([data_root])
    seed = grounding.LiteratureClaimSeed(
        seed_id="asd_fc_seed",
        source_pmid="10",
        target_family="asd",
        outcome_modality="resting-state fMRI-FC",
        predictor_or_group="ASD diagnosis versus controls",
        outcome_family="functional connectivity",
        expected_direction="two_sided",
        covariates=["age", "sex", "site"],
        candidate_question="ASD diagnosis is associated with resting-state functional connectivity.",
        evidence_snippet="ASD cases differed from controls in FC.",
        support_level="direct",
        rationale="Uses local FC descriptors.",
    )

    result = grounding.evaluate_feasibility([seed], {}, catalog)[0]

    assert result.status == "executable_now"
    assert result.discovery_cohort is not None
    assert result.replication_cohort is not None
    assert grounding._base_cohort(result.discovery_cohort) == grounding._base_cohort(result.replication_cohort)
    assert result.matched_outcome_examples
    assert set(result.matched_outcome_examples).issubset(
        set(next(row for row in catalog["cohorts"] if row["cohort"] == result.discovery_cohort)["idps"])
    )


def test_adhd_pair_uses_confirm_dx_for_incompatible_raw_labels(tmp_path: Path) -> None:
    data_root = tmp_path / "cohorts"
    _write_adhd_label_cohorts(data_root)
    catalog = drafting._merge_catalogs([data_root])
    seed = grounding.LiteratureClaimSeed(
        seed_id="adhd_fc_seed",
        source_pmid="11",
        target_family="adhd",
        outcome_modality="resting-state fMRI-FC",
        predictor_or_group="ADHD diagnosis versus controls",
        outcome_family="functional connectivity",
        expected_direction="two_sided",
        covariates=["age", "sex", "site"],
        candidate_question="ADHD diagnosis is associated with resting-state functional connectivity.",
        evidence_snippet="ADHD cases differed from controls in FC.",
        support_level="direct",
        rationale="Uses local FC descriptors.",
    )

    result = grounding.evaluate_feasibility([seed], {}, catalog)[0]
    rows = grounding._claim_rows([seed], {}, [result])

    assert result.status == "executable_now"
    assert result.discovery_cohort == "ABCD_DISC"
    assert result.replication_cohort == "ADHD200_REP"
    assert rows[0]["group_var"] == "confirm_dx"
    assert rows[0]["case_label"] == "case"
    assert rows[0]["control_label"] == "control"
    assert rows[0]["shared_outcome_columns_sample"]


def test_feasibility_rejects_unrepresentable_literature_concepts(tmp_path: Path) -> None:
    data_root = tmp_path / "cohorts"
    _write_fc_cohorts(data_root)
    catalog = drafting._merge_catalogs([data_root])
    seeds = [
        grounding.LiteratureClaimSeed(
            seed_id="effective_connectivity",
            source_pmid="12",
            target_family="psychosis",
            outcome_modality="resting-state fMRI effective connectivity via spectral DCM",
            predictor_or_group="schizophrenia diagnosis versus controls",
            outcome_family="fronto-striato-thalamic effective connectivity",
            expected_direction="two_sided",
            covariates=["age", "sex", "site"],
            candidate_question="Schizophrenia is associated with effective connectivity.",
            evidence_snippet="DCM effective connectivity differed.",
            support_level="direct",
            rationale="Requires effective connectivity.",
        ),
        grounding.LiteratureClaimSeed(
            seed_id="tdp_tau",
            source_pmid="13",
            target_family="ad_aging",
            outcome_modality="sMRI",
            predictor_or_group="tau and TDP-43 burden",
            outcome_family="medial temporal lobe atrophy",
            expected_direction="positive",
            covariates=["age", "sex", "eTIV"],
            candidate_question="Tau and TDP-43 predict atrophy.",
            evidence_snippet="Tau and TDP-43 were associated with atrophy.",
            support_level="direct",
            rationale="Requires pathology predictors.",
        ),
        grounding.LiteratureClaimSeed(
            seed_id="ivfc_behavior",
            source_pmid="14",
            target_family="normative_fmri",
            outcome_modality="behavioral/cognitive measures linked to rs-fMRI IVFC",
            predictor_or_group="IVFC of triple and cerebellar networks",
            outcome_family="executive function performance",
            expected_direction="two_sided",
            covariates=["age", "sex", "site"],
            candidate_question="IVFC predicts executive function.",
            evidence_snippet="IVFC explained behavior.",
            support_level="direct",
            rationale="Requires IVFC and behavior outcomes.",
        ),
    ]

    results = grounding.evaluate_feasibility(seeds, {}, catalog)

    assert [result.status for result in results] == [
        "requires_new_feature_adapter",
        "requires_new_feature_adapter",
        "requires_new_feature_adapter",
    ]
    assert "effective" in results[0].reason.lower()
    assert "tau" in results[1].reason.lower() or "tdp" in results[1].reason.lower()
    assert "individual" in results[2].reason.lower() or "brain-to-behavior" in results[2].reason.lower()


def test_feasibility_rejects_qmri_microstructure_without_matching_columns(tmp_path: Path) -> None:
    data_root = tmp_path / "cohorts"
    _write_smri_cohorts(data_root)
    catalog = drafting._merge_catalogs([data_root])
    seed = grounding.LiteratureClaimSeed(
        seed_id="qmri_seed",
        source_pmid="15",
        target_family="ad_aging",
        outcome_modality="quantitative MRI (R1, MTsat, R2*)",
        predictor_or_group="age",
        outcome_family="hippocampal microstructure",
        expected_direction="two_sided",
        covariates=["age", "sex"],
        candidate_question="Age is associated with hippocampal quantitative MRI microstructure.",
        evidence_snippet="Age was associated with R1, MTsat, and R2*.",
        support_level="direct",
        rationale="Requires qMRI columns, not generic hippocampal volume.",
    )

    result = grounding.evaluate_feasibility([seed], {}, catalog)[0]

    assert result.status == "requires_new_feature_adapter"


def test_feasibility_rejects_nonlocal_predictors_even_with_fc_columns(tmp_path: Path) -> None:
    data_root = tmp_path / "cohorts"
    _write_adhd_label_cohorts(data_root)
    catalog = drafting._merge_catalogs([data_root])
    seeds = [
        grounding.LiteratureClaimSeed(
            seed_id="neurofeedback",
            source_pmid="16",
            target_family="adhd",
            outcome_modality="fMRI or near-infrared spectroscopy neurofeedback",
            predictor_or_group="Frontal-activation neurofeedback versus control condition in ADHD",
            outcome_family="ADHD symptoms or cognition",
            expected_direction="negative",
            covariates=[],
            candidate_question="Neurofeedback improves ADHD symptoms.",
            evidence_snippet="Neurofeedback was compared with controls.",
            support_level="direct",
            rationale="Requires intervention and symptom columns.",
        ),
        grounding.LiteratureClaimSeed(
            seed_id="brief",
            source_pmid="17",
            target_family="adhd",
            outcome_modality="resting-state fMRI functional connectivity",
            predictor_or_group="behavioral regulation problems measured using the Behavior Rating Inventory of Executive Function",
            outcome_family="prefrontal pathway functional connectivity",
            expected_direction="negative",
            covariates=[],
            candidate_question="Behavioral regulation problems are associated with prefrontal FC.",
            evidence_snippet="BRIEF scores correlated with FC.",
            support_level="direct",
            rationale="Requires BRIEF predictor columns.",
        ),
    ]

    results = grounding.evaluate_feasibility(seeds, {}, catalog)

    assert [result.status for result in results] == ["requires_new_feature_adapter", "requires_new_feature_adapter"]
    assert all("predictor" in result.reason.lower() or "brain-to-behavior" in result.reason.lower() for result in results)


def test_claim_rows_skip_executable_results_without_feasible_covariates() -> None:
    seed = grounding.LiteratureClaimSeed(
        seed_id="no_covariates",
        source_pmid="18",
        target_family="ad_aging",
        outcome_modality="PET",
        predictor_or_group="age",
        outcome_family="FDG PET uptake",
        expected_direction="negative",
        covariates=[],
        candidate_question="Age is associated with FDG PET uptake.",
        evidence_snippet="PET uptake varied with age.",
        support_level="direct",
        rationale="Synthetic fixture with no standard covariates.",
    )
    feasibility = grounding.FeasibilityResult(
        seed_id="no_covariates",
        pmid="18",
        target_family="ad_aging",
        status="executable_now",
        reason="synthetic",
        discovery_cohort="ADNI_DISC",
        replication_cohort="ADNI_REP",
        matched_outcome_examples=["pet_fdg_suvr"],
    )

    rows = grounding._claim_rows([seed], {}, [feasibility])

    assert rows == []
