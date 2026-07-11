from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

import nbs_data.run_external_freesurfer as freesurfer_runner
from bench.prepare_nacc_external import cm3_to_mm3
from nbs_data.external_dataset_registry import (
    ExternalDatasetSpec,
    MetadataSpec,
    StructuralSpec,
    build_subject_manifest,
    load_registry,
)
from nbs_data.external_fmri import (
    _descriptor_feature_name,
    _prepare_descriptor_csv,
    _static_fc_worker,
    global_fc_features,
)
from nbs_data.external_metadata import load_metadata
from nbs_data.freesurfer.external_stats import (
    LEGACY_SUCCESS_MARKERS,
    adopt_legacy_completion,
    canonical_features,
    completion_check,
)
from nbs_data.run_external_freesurfer import (
    _gpu_task_queues,
    _progress_message,
    validate_fastsurfer_runtime,
    validate_freesurfer_runtime,
)


ROOT = Path(__file__).resolve().parents[1]


def test_external_data_launchers_are_isolated_from_confirm_launchers():
    processing_dir = ROOT / "scripts/data_processing"
    expected = {
        "_external_arcdev_common.sh",
        "audit_external_datasets_arcdev.sh",
        "launch_external_fmri_arcdev.sh",
        "launch_external_freesurfer_arcdev.sh",
        "run_external_data_remote.sh",
        "sync_external_evidence_arcdev.sh",
    }

    assert expected <= {path.name for path in processing_dir.iterdir()}
    assert all(path.suffix == ".sh" for path in processing_dir.iterdir() if path.is_file())
    for name in expected - {"_external_arcdev_common.sh", "run_external_data_remote.sh"}:
        assert not (ROOT / "scripts" / name).exists()


def test_remote_freesurfer_launcher_loads_module_and_accepts_explicit_homes():
    common = (ROOT / "scripts/data_processing/_external_arcdev_common.sh").read_text(
        encoding="utf-8"
    )
    sync = (ROOT / "scripts/data_processing/sync_external_evidence_arcdev.sh").read_text(
        encoding="utf-8"
    )
    remote_worker = (ROOT / "scripts/data_processing/run_external_data_remote.sh").read_text(
        encoding="utf-8"
    )
    launcher = (ROOT / "scripts/data_processing/launch_external_freesurfer_arcdev.sh").read_text(
        encoding="utf-8"
    )

    assert 'SSH_HOST="${SSH_HOST:-${ARCDEV_HOST:-arcdev}}"' in common
    assert 'ssh "$SSH_HOST"' in common
    assert 'bash -l "$REMOTE_CODE_DIR/scripts/data_processing/run_external_data_remote.sh"' in common
    assert 'SSH_HOST="${SSH_HOST:-${ARCDEV_HOST:-arcdev}}"' in sync
    assert 'ssh "$SSH_HOST"' in sync
    assert 'source /etc/profile' in remote_worker
    assert 'module load "$module_name"' in remote_worker
    assert 'REMOTE_FREESURFER_HOME:-${FREESURFER_HOME:-}' in remote_worker
    assert 'REMOTE_FASTSURFER_HOME="${REMOTE_FASTSURFER_HOME:-' in launcher
    assert 'REMOTE_FASTSURFER_PYTHON="${REMOTE_FASTSURFER_PYTHON:-' in launcher
    assert "flock -n 9" in remote_worker
    assert "already has a $STAGE log" in remote_worker


def _dataset(
    dataset_id: str,
    root: Path,
    pattern: str,
    layout: str,
    *,
    preferred_tokens: list[str] | None = None,
) -> ExternalDatasetSpec:
    return ExternalDatasetSpec(
        dataset_id=dataset_id,
        target_families=["ad_aging"],
        metadata=MetadataSpec(adapter="none"),
        structural=StructuralSpec(
            root=str(root),
            pattern=pattern,
            layout=layout,
            preferred_tokens=preferred_tokens or [],
        ),
    )


def _write_nifti(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = nib.Nifti1Image(np.ones((3, 4, 5), dtype=np.float32), np.eye(4))
    nib.save(image, path)


def _write(path: Path, text: str = "ok\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _complete_legacy_subject(root: Path) -> None:
    _write(root / "stats/aseg.stats", _aseg_text())
    _write(root / "stats/lh.aparc.DKTatlas.mapped.stats", _aparc_text(100.0, 2.0))
    _write(root / "stats/rh.aparc.DKTatlas.mapped.stats", _aparc_text(120.0, 2.4))
    _write(root / "surf/lh.white")
    _write(root / "surf/rh.white")
    for marker in LEGACY_SUCCESS_MARKERS:
        _write(root / marker)


def _aseg_text() -> str:
    return "\n".join(
        [
            "# Measure EstimatedTotalIntraCranialVol, eTIV, ICV, 1500000.0, mm^3",
            "# Measure BrainSegNotVent, BrainSegVolNotVent, Brain, 1000000.0, mm^3",
            "1 17 10 4000.0 Left-Hippocampus 0 0 0 0 0",
            "2 53 10 4200.0 Right-Hippocampus 0 0 0 0 0",
            "3 4 10 10000.0 Left-Lateral-Ventricle 0 0 0 0 0",
            "4 43 10 11000.0 Right-Lateral-Ventricle 0 0 0 0 0",
        ]
    )


def _aparc_text(offset: float, thickness: float) -> str:
    return "\n".join(
        [
            "# ColHeaders StructName NumVert SurfArea GrayVol ThickAvg ThickStd MeanCurv GausCurv FoldInd CurvInd",
            f"entorhinal 1 2 {offset} {thickness} 0 0 0 0 0",
            f"fusiform 1 2 {offset + 10} {thickness + 0.1} 0 0 0 0 0",
            f"middletemporal 1 2 {offset + 20} {thickness + 0.2} 0 0 0 0 0",
        ]
    )


def test_external_registry_is_pydantic_valid_and_has_declared_quarantines():
    registry = load_registry(ROOT / "configs/external_datasets.yml")

    by_id = {dataset.dataset_id: dataset for dataset in registry.datasets}
    assert by_id["Olin_ASD_SZ"].quarantine
    assert "feature_parity_blocked" in by_id["Olin_ASD_SZ"].quarantine_reasons
    assert by_id["Shile_Nanjing"].structural.existing_subjects_dir
    assert by_id["BLSA"].metadata.adapter == "blsa"


@pytest.mark.parametrize(
    ("layout", "relative_path", "expected_subject", "expected_session", "expected_site"),
    [
        ("bids_single_session", "sub001/ses_01/anat/T1.nii", "sub001", "ses_01", "TEST"),
        ("ehbs_single", "EHBS001/T1w.nii", "EHBS001", "ses-01", "TEST"),
        ("shile_bids", "siteA/S001/ses_01/anat/T1.nii", "siteA_S001", "ses_01", "siteA"),
        ("pk_mprc", "Scanner1/Z001/anat.nii", "Z001", "ses-01", "Scanner1"),
    ],
)
def test_single_session_layout_parsers(
    tmp_path,
    layout,
    relative_path,
    expected_subject,
    expected_session,
    expected_site,
):
    root = tmp_path / layout
    _write_nifti(root / relative_path)
    pattern = "**/T1w.nii" if layout == "ehbs_single" else "**/*.nii"

    result = build_subject_manifest(_dataset("TEST", root, pattern, layout))

    assert len(result.selected_rows) == 1
    row = result.selected_rows[0]
    assert (row.subject_id, row.session, row.site) == (
        expected_subject,
        expected_session,
        expected_site,
    )


def test_blsa_layout_selects_earliest_visit(tmp_path):
    root = tmp_path / "blsa"
    earliest = root / "BLSA_0001" / "01-0_10" / "anat" / "anat.nii"
    _write_nifti(earliest)
    _write_nifti(root / "BLSA_0001" / "04-0_10" / "anat" / "anat.nii")

    result = build_subject_manifest(_dataset("BLSA", root, "**/anat/anat.nii", "blsa_dated_visit"))

    assert result.selected_rows[0].session == "BLSA_0001_01-0_10"
    assert result.selected_rows[0].t1_path == str(earliest.resolve())


def test_aibl_selects_earliest_valid_scan_and_falls_back_only_after_qc_failure(tmp_path):
    root = tmp_path / "aibl"
    bad = root / "001" / "2008-01-01" / "anat" / "anat.nii"
    bad.parent.mkdir(parents=True)
    bad.write_text("not a nifti", encoding="utf-8")
    later = root / "001" / "2009-01-01" / "anat" / "anat.nii"
    _write_nifti(later)

    result = build_subject_manifest(_dataset("AIBL", root, "**/anat/anat.nii", "aibl_dated_visit"))

    assert len(result.selected_rows) == 1
    assert result.selected_rows[0].t1_path == str(later.resolve())
    assert result.selected_rows[0].selection_reason == "baseline_qc_fallback"
    assert result.selected_rows[0].rejected_earlier_candidates == 1


def test_olin_preserves_participant_folder_and_prefers_mprage(tmp_path):
    root = tmp_path / "olin"
    _write_nifti(root / "S5374KIW1" / "10_OTHER" / "T1.nii")
    preferred = root / "S5374KIW1" / "24_MPRAGE_3min" / "T1.nii"
    _write_nifti(preferred)

    dataset = _dataset(
        "Olin_ASD_SZ",
        root,
        "**/T1.nii",
        "olin_acquisition",
        preferred_tokens=["MPRAGE_3min", "MPRAGE"],
    )
    result = build_subject_manifest(dataset)

    assert result.selected_rows[0].subject_id == "S5374KIW1"
    assert result.selected_rows[0].t1_path == str(preferred.resolve())


def test_ambiguous_subject_id_at_multiple_sites_is_recorded_without_crashing(tmp_path):
    root = tmp_path / "pk"
    _write_nifti(root / "scanner_a" / "subject_1" / "anat.nii")
    _write_nifti(root / "scanner_b" / "subject_1" / "anat.nii")

    result = build_subject_manifest(_dataset("PK_MPRC", root, "**/anat.nii", "pk_mprc"))

    assert not result.selected_rows
    assert result.selection_exclusions == [
        {
            "dataset_id": "PK_MPRC",
            "subject_id": "subject_1",
            "reason": "ambiguous_site_assignment",
            "candidate_sites": ["scanner_a", "scanner_b"],
            "metadata_sites": [],
        }
    ]


def test_multi_scanner_subject_uses_authoritative_metadata_assignment(tmp_path):
    root = tmp_path / "pk"
    scanner1 = root / "Scanner1" / "Z32163" / "anat.nii"
    scanner2 = root / "Scanner2" / "Z32163" / "anat.nii"
    _write_nifti(scanner1)
    _write_nifti(scanner2)
    metadata = pd.DataFrame(
        {
            "subject_id": ["Z32163"],
            "session": ["ses-01"],
            "site": ["Scanner2"],
            "age": [51],
            "sex": ["M"],
            "dx": ["SZ"],
        }
    )

    result = build_subject_manifest(
        _dataset("PK_MPRC", root, "**/anat.nii", "pk_mprc"),
        metadata=metadata,
    )

    assert len(result.selected_rows) == 1
    assert result.selected_rows[0].t1_path == str(scanner2.resolve())
    assert result.selected_rows[0].site_selection_reason == "metadata_site_match"
    assert not result.selection_exclusions


def test_metadata_adapter_supplies_default_session_and_site_series(tmp_path):
    metadata_path = tmp_path / "ehbs.csv"
    pd.DataFrame({"subject_id": ["s1"], "age": [50], "sex": ["Female"]}).to_csv(metadata_path, index=False)
    dataset = ExternalDatasetSpec(
        dataset_id="EHBS",
        target_families=["normative_fmri"],
        metadata=MetadataSpec(adapter="ehbs", paths=[str(metadata_path)]),
    )

    metadata = load_metadata(dataset)

    assert metadata.loc[0, "session"] == "ses-01"
    assert metadata.loc[0, "site"] == "EHBS"


def test_cnp_metadata_join_normalizes_diagnosis_and_demographics(tmp_path):
    path = tmp_path / "participants.tsv"
    pd.DataFrame(
        {"participant_id": ["sub-1"], "diagnosis": ["SCHZ"], "age": [31], "gender": ["M"]}
    ).to_csv(path, sep="\t", index=False)
    dataset = ExternalDatasetSpec(
        dataset_id="LA5C",
        target_families=["psychosis"],
        metadata=MetadataSpec(adapter="cnp", paths=[str(path)]),
    )

    metadata = load_metadata(dataset)

    assert metadata.loc[0, ["subject_id", "session", "sex", "dx"]].tolist() == [
        "sub-1",
        "ses_01",
        "M",
        "SZ",
    ]


def test_aibl_metadata_join_matches_visit_and_diagnosis(tmp_path):
    master = tmp_path / "aibl.csv"
    diagnosis = tmp_path / "diagnosis.csv"
    pd.DataFrame(
        {
            "Subject": [1],
            "Modality": ["MRI"],
            "Description": ["MPRAGE"],
            "Acq Date": ["2008-01-01"],
            "Visit": [1],
            "Age": [72],
            "Sex": ["F"],
        }
    ).to_csv(master, index=False)
    pd.DataFrame(
        {"RID": [1], "VISCODE": ["bl"], "DXNORM": [0], "DXMCI": [0], "DXAD": [1], "SITEID": [2]}
    ).to_csv(diagnosis, index=False)
    dataset = ExternalDatasetSpec(
        dataset_id="AIBL",
        target_families=["ad_aging"],
        metadata=MetadataSpec(adapter="aibl", paths=[str(master), str(diagnosis)]),
    )

    metadata = load_metadata(dataset)

    assert metadata.loc[0, ["subject_id", "session", "age", "sex", "dx"]].tolist() == [
        "1",
        "2008-01-01",
        72,
        "F",
        "AD",
    ]


def test_blsa_metadata_join_uses_exact_session(tmp_path):
    path = tmp_path / "blsa.xlsx"
    pd.DataFrame(
        {
            "session": ["BLSA_0001_01-0_10"],
            "MPRAGE": [1],
            "dxfinal": [0],
            "scanner": ["3T"],
            "Age": [70],
            "sex": ["Female"],
        }
    ).to_excel(path, index=False)
    dataset = ExternalDatasetSpec(
        dataset_id="BLSA",
        target_families=["ad_aging"],
        metadata=MetadataSpec(adapter="blsa", paths=[str(path)]),
    )

    metadata = load_metadata(dataset)

    assert metadata.loc[0, ["subject_id", "session", "sex", "dx"]].tolist() == [
        "BLSA_0001",
        "BLSA_0001_01-0_10",
        "F",
        "CN",
    ]


def test_pk_mprc_metadata_join_supports_both_identifier_systems(tmp_path):
    path = tmp_path / "pk.xlsx"
    pd.DataFrame(
        {
            "zid": ["Z001"],
            "aid": ["A001"],
            "dx(0=crl,1=scz)": [1],
            "Age": [28],
            "sex (0=F,1=M)": [1],
            "class (scanner)": [1],
        }
    ).to_excel(path, index=False)
    dataset = ExternalDatasetSpec(
        dataset_id="PK_MPRC",
        target_families=["psychosis"],
        metadata=MetadataSpec(adapter="pk_mprc", paths=[str(path)]),
    )

    metadata = load_metadata(dataset)

    assert set(metadata["subject_id"]) == {"Z001", "A001"}
    assert set(metadata["dx"]) == {"SZ"}
    assert set(metadata["site"]) == {"Scanner1"}


def test_freesurfer_runtime_requires_exact_741(tmp_path):
    home = tmp_path / "freesurfer"
    binary = home / "bin/recon-all"
    _write(binary, "#!/usr/bin/env bash\necho freesurfer-linux-centos8_x86_64-7.4.1\n")
    binary.chmod(0o755)

    assert validate_freesurfer_runtime(home) == "7.4.1"

    binary.write_text("#!/usr/bin/env bash\necho freesurfer-7.4.2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="7.4.1 is required"):
        validate_freesurfer_runtime(home)

    assert validate_freesurfer_runtime(home, expected_version="7.4.2") == "7.4.2"


def test_fastsurfer_runtime_requires_compatible_local_python(tmp_path):
    home = tmp_path / "FastSurfer"
    runner = home / "run_fastsurfer.sh"
    python = home / ".venv/bin/python"
    python3 = home / ".venv/bin/python3"
    _write(runner, "#!/usr/bin/env bash\nexit 0\n")
    _write(python, "#!/usr/bin/env bash\necho 3.12.3\n")
    runner.chmod(0o755)
    python.chmod(0o755)
    python3.symlink_to("python")

    assert validate_fastsurfer_runtime(home, python) == "3.12.3"

    python.write_text("#!/usr/bin/env bash\necho missing yacs >&2\nexit 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="FastSurfer Python must be >=3.10"):
        validate_fastsurfer_runtime(home, python)


def test_freesurfer_cli_returns_nonzero_when_any_subject_failed(monkeypatch, tmp_path):
    monkeypatch.setattr(
        freesurfer_runner,
        "run",
        lambda args: {"status_counts": {"failed": 1}},
    )

    exit_code = freesurfer_runner.main(
        ["--datasets", "AIBL", "--out-root", str(tmp_path)]
    )

    assert exit_code == 1


def test_freesurfer_progress_prints_failure_reason():
    message = _progress_message(
        1,
        1,
        {
            "dataset_id": "AIBL",
            "subject_id": "10",
            "status": "failed",
            "error": "partial output preserved; rerun with --retry-failed",
        },
    )

    assert "status=failed" in message
    assert "partial output preserved" in message


def test_gpu_scheduler_uses_one_sequential_queue_per_device():
    tasks = [{"id": index} for index in range(5)]

    queues = _gpu_task_queues(tasks, ["0", "1"])

    assert queues == [
        ("0", [{"id": 0}, {"id": 2}, {"id": 4}]),
        ("1", [{"id": 1}, {"id": 3}]),
    ]


def test_aseg_alone_is_incomplete_and_legacy_adoption_creates_strict_receipt(tmp_path):
    subject = tmp_path / "subject"
    _write(subject / "stats/aseg.stats", _aseg_text())
    assert not completion_check(subject, allow_legacy=True).complete

    _complete_legacy_subject(subject)
    assert completion_check(subject, allow_legacy=True).complete
    assert not completion_check(subject).complete

    adopt_legacy_completion(
        subject,
        subject_id="subject",
        t1_path="/raw/T1.nii",
        t1_sha256="abc",
    )
    assert completion_check(subject).complete


def test_canonical_features_keep_cortical_volume_and_thickness_distinct(tmp_path):
    subject = tmp_path / "subject"
    _complete_legacy_subject(subject)
    adopt_legacy_completion(subject, subject_id="subject", t1_path="/raw/T1.nii", t1_sha256="abc")

    features = canonical_features(subject)

    assert features["eTIV"] == 1_500_000.0
    assert features["smri_hippocampus"] == 8_200.0
    assert features["smri_entorhinal"] == 220.0
    assert features["smri_ventricles"] == 21_000.0
    assert features["smri_thickness_entorhinal"] == pytest.approx(2.2)


def test_nacc_cm3_conversion_and_fc_feature_naming_are_explicit():
    converted = cm3_to_mm3(pd.Series([1.5, 2.0]))
    assert converted.tolist() == [1500.0, 2000.0]
    assert _descriptor_feature_name("fc_fc_DMN_VIS") == "fc_fc_DMN_VIS"
    assert _descriptor_feature_name("12") == "fc_fc_12"


def test_descriptor_import_requires_explicit_equal_length_index_before_metadata_join(tmp_path):
    descriptor = tmp_path / "descriptors.csv"
    index = tmp_path / "index.csv"
    metadata = tmp_path / "metadata.csv"
    pd.DataFrame({"Cont-Cont": [0.1, 0.2, 0.3]}).to_csv(descriptor, index=False)
    pd.DataFrame(
        {"subject": ["s1", "s2", "s3"], "session": ["ses01", "ses01", "ses01"]}
    ).to_csv(index, index=False)
    pd.DataFrame(
        {
            "subject_id": ["s1", "s3"],
            "session_id": ["ses01", "ses01"],
            "age": [50, 60],
            "sex": ["F", "M"],
        }
    ).to_csv(metadata, index=False)
    dataset = load_registry(ROOT / "configs/external_datasets.yml").selected(["EHBS"])[0].model_copy(
        update={
            "metadata": MetadataSpec(adapter="ehbs", paths=[str(metadata)]),
            "fmri": load_registry(ROOT / "configs/external_datasets.yml")
            .selected(["EHBS"])[0]
            .fmri.model_copy(
                update={
                    "descriptor_path": str(descriptor),
                    "descriptor_index_path": str(index),
                }
            ),
        }
    )

    frame, provenance = _prepare_descriptor_csv(dataset)

    assert frame["subject_id"].tolist() == ["s1", "s3"]
    assert frame["fc_fc_Cont_Cont"].tolist() == [0.1, 0.3]
    assert provenance["join_method"] == "explicit_descriptor_index"

    pd.DataFrame({"subject": ["s1"], "session": ["ses01"]}).to_csv(index, index=False)
    with pytest.raises(ValueError, match="descriptor/index row mismatch"):
        _prepare_descriptor_csv(dataset)


def test_global_fc_extractor_emits_only_registered_global_summaries():
    rng = np.random.default_rng(7)
    features = global_fc_features(rng.normal(size=(120, 100)))

    assert set(features) == {
        "fc_mean_abs",
        "fc_mean_positive",
        "fc_within_network",
        "fc_between_network",
    }
    assert all(np.isfinite(value) for value in features.values())


def test_static_fc_import_uses_first_valid_visit_and_only_order_invariant_summaries(tmp_path):
    from scipy.io import savemat

    bad = tmp_path / "01-0_10" / "sfc.mat"
    _write(bad, "bad mat")
    valid = tmp_path / "02-0_10" / "sfc.mat"
    valid.parent.mkdir(parents=True)
    savemat(valid, {"sFNC": np.asarray([[-0.2], [0.1], [0.3], [-0.1], [0.2], [0.4]])})

    result = _static_fc_worker(
        (
            "BLSA_0001",
            [(1, "01-0_10", str(bad)), (2, "02-0_10", str(valid))],
            "sFNC",
            4,
        )
    )

    assert result["status"] == "ready"
    assert result["session"] == "BLSA_0001_02-0_10"
    assert set(result) == {
        "status",
        "subject_id",
        "session",
        "source_fc_path",
        "source_fc_sha256",
        "fc_mean_abs",
        "fc_mean_positive",
    }
