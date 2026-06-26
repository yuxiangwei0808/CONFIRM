# Data Inventory and Integration Plan

Date: 2026-06-16

Scope: derivatives-first. Raw images are intentionally postponed. Current work should use tabular sMRI/PET/fMRI derivatives and metadata; raw-image workflows can be added later through FreeSurfer/fMRIPrep or external neuroimaging LLM tools.

## Summary

The newly added data are useful and materially expand the project:

- ADNI now supports sMRI, PET, and resting-state FC.
- OASIS-3 supports sMRI and PET.
- ABCD supports large-scale sMRI, rsfMRI QC/motion, cognition, CBCL, K-SADS ADHD, and demographics.
- UKB supports large-scale IDP/cognition/diagnosis claims, but through UKB field-coded/named IDPs rather than FreeSurfer `aseg/aparc` tables.
- COBRE/ABIDE2/BLSA bundle is mostly metadata/phenotype. It is useful for diagnosis/covariate labels, but not yet sufficient for regional neuroimaging claims unless matching derivative matrices/tables are added.

## Archive Findings

### ADNI Core

File:

- `data/raw/ADNIMERGE.xlsx`

Already usable. Contains:

- subject IDs: `RID`, `PTID`
- diagnosis: `DX`, `DX_bl`
- demographics: `AGE`, `PTGENDER`
- sMRI-derived biomarkers: `Hippocampus`, `Entorhinal`, `MidTemp`, `Fusiform`, `Ventricles`, `WholeBrain`, `ICV`
- PET scalar: `FDG`
- phase/site support: `ORIGPROT`, `COLPROT`, `FLDSTRENG`

Status: already used successfully in ADNI and ADNI-to-OASIS3 experiments.

### ADNI PET

Archive:

- `data/raw/adni_pet_summary_tables_info_20260616.tar.gz`

Inspected contents:

- `PET_Image_Analysis.zip`
- scan/search metadata CSVs for FDG, AV45, AV1451, FBB, PIB

Useful regional/summary PET tables found inside `PET_Image_Analysis.zip`:

- `UCBERKELEYFDG_8mm_02_17_23_29Aug2023.csv`: ROI-level FDG with `ROINAME`, `MEAN`, `MAX`, `STDEV`, `TOTVOX`
- `UCBERKELEY_AMY_6MM_29Aug2023.csv`: amyloid SUVR/centiloid-style regional table, 341 columns
- `UCBERKELEY_TAU_6MM_29Aug2023.csv`: tau SUVR regional table, 336 columns
- `UCBERKELEY_TAUPVC_6MM_29Aug2023.csv`: PVC tau SUVR table, 332 columns
- `BAIPETNMRCFDG_12_11_20_29Aug2023.csv`: compact FDG summaries including `HCI`, `SROI.AD`, `SROI.MCI`
- `BAIPETNMRCFTP_08_17_22_29Aug2023.csv`: compact tau summaries including `ENTORHINAL_SUVR`, `INFERIOR_TEMPORAL_SUVR`, `TAU_METAROI`
- `BAIPETNMRCAV45_10_23_20_29Aug2023.csv`: compact AV45 summaries

Status: very usable. Next adapter should extract baseline/nearest-visit PET summaries and prefix canonical columns as `pet_fdg_*`, `pet_amyloid_*`, `pet_tau_*`.

### ADNI Functional Connectivity

Archive:

- `data/raw/fc_ADNI.tar.gz`

Extracted files:

- `data_fc.h5`
- `ADNI_fMRI_metadata.csv`
- `progress_adni.json`
- `errors_adni.log`

HDF5 structure:

- group `fc/`
- 1,283 datasets: `sample_000000` ... `sample_001282`
- each sample shape: `450 x 450`, dtype `float32`
- matrices are finite, symmetric, and have diagonal mean 1.0
- group `metadata/` stores dataset names, file IDs, file paths, sessions, shapes, and subjects

Metadata:

- 1,283 fMRI runs
- 748 unique subjects
- each run has `shape_time=450`, `shape_features=450`

Join with `ADNIMERGE`:

- 401 / 748 FC subjects match `ADNIMERGE.PTID`
- 393 matched subjects have non-missing `DX`
- run-level DX counts after baseline merge: CN 416, MCI 321, Dementia 12, missing 534

Status: usable, with caveats.

Recommended first use:

- MCI-vs-CN or continuous cognitive/diagnosis-related FC claims, not AD Dementia-vs-CN, because Dementia has only 12 matched FC runs.
- injected motion/site/diagnosis confound traps
- FC scalar summaries: global mean absolute FC, positive FC, modular/network summaries if ROI labels are provided
- full edge-wise claims only after choosing a memory-efficient edge family or sparse feature set

Missing for interpretability:

- ROI labels / atlas definition for the 450 nodes.

This is not required for computation, but it is important for neurobiological interpretation. Without labels we can still run `fc_edge_i_j` or global/network-free summary claims, but cannot say which named network/region drove an effect.

### OASIS-3

Archives:

- `data/raw/oasis3_data_info.tar.gz`
- `data/raw/oasis3_data_info (1).tar.gz`
- duplicated/expanded in the multidataset archive

Already inspected/used:

- `OASIS3_Freesurfer_output.csv`: FreeSurfer volumes/thickness, `IntraCranialVol`
- `OASIS3_demographics.csv`: demographic fields
- `OASIS3_UDSb4_cdr.csv`: clinical CDR/MMSE/diagnosis-related fields

Additional PET tables present:

- `OASIS3_PUP.csv`
- `OASIS3_AV1451_PUP.csv`
- `OASIS3_AV1451L_PUP.csv`
- `OASIS3_amyloid_centiloid.csv`
- AV1451 Braak tauopathy summary

Status: sMRI already usable; PET is usable next.

### ABCD

Archive:

- `data/raw/abcd_freesurfer_metadata_behavior_release51_60_20260616.tar.gz`

High-value files sampled:

- `ab_p_demo.tsv`: 67,410 rows, demographics
- `mr_y_smri__vol__aseg.tsv`: 30,276 rows, 44 columns, subcortical FreeSurfer-style volumes
- `mr_y_smri__vol__dsk.tsv`: 30,276 rows, 73 columns, Desikan cortical volumes
- `mr_y_smri__thk__dsk.tsv`: 30,276 rows, 73 columns, Desikan cortical thickness
- `mr_y_qc__mot.tsv`: 29,863 rows, 116 columns, rsfMRI/dMRI/tfMRI motion metrics
- `mr_y_qc__raw__rsfmri.tsv`: 30,386 rows, 504 columns, rsfMRI QC/raw quality metrics
- `nc_y_nihtb.tsv`: 37,308 rows, NIH Toolbox cognition
- `mh_p_cbcl.tsv`: 67,295 rows, CBCL behavioral scores
- `mh_p_ksads__adhd.tsv`: 65,224 rows, ADHD symptoms/diagnosis items

Status: very usable for sMRI/development/behavior claims and QC/confound traps.

Recommended first use:

- age/development effects on cortical thickness/volume
- sex effects on global/subcortical measures
- brain-behavior fragile claims using NIH Toolbox or CBCL
- motion-confounded fMRI/QC traps
- ADHD symptom claims if K-SADS variable mapping is confirmed

### UK Biobank

Archive:

- `data/raw/ukb_idp_metadata_cognition_diagnosis_20260616.tar.gz`

Useful extracted files:

- `latest_2025_selected/ukb674036_2025_idp_demo_cog_selected.csv`: 502,366 rows, 5,718 columns, field-coded selected IDPs/cognition/demographics
- `descriptive_extracts/New_rda/my_ukb_data_smri.csv`: 37,929 rows, 6,972 named columns
- `descriptive_extracts/New_rda/my_ukb_data_fmri.csv`: 37,929 rows, 6,972 named columns
- diagnosis support tables and unaffected ID lists

Available feature families:

- global brain volumes / grey matter / white matter / ventricular CSF / WMH
- T2* hippocampus and dMRI cingulum-hippocampus metrics
- fMRI motion/QC metrics
- fluid intelligence and other cognitive fields
- diagnosis/self-report support

Important limitation:

- no dedicated FreeSurfer `aseg/aparc` table was found. UKB uses UKB field-coded IDPs, not ADNI/OASIS/ABCD-style FreeSurfer region tables.

Status: usable, but adapter must target UKB-specific IDP fields rather than FreeSurfer region matching.

Recommended first use:

- aging/global volume claims
- brain-behavior fragile claims at UKB scale
- motion/QC confound traps
- disease/self-report diagnosis claims only after ICD/self-report case definitions are fixed

### COBRE / ABIDE2 / BLSA Metadata Bundle

Archive:

- `data/raw/multidataset_cobre_abide2_oasis3_blsa_metadata_20260616.tar.gz`

Findings:

- COBRE: compact demographics/diagnosis/PANSS/cognitive/medication metadata. Example `pheno_comb_cobre_all.csv` has 252 rows and columns `URSI`, `Age`, `Sex`, `Diagnosis`.
- ABIDE2: composite phenotype CSVs, documentation, data legend. No compact regional neuroimaging table included in this bundle.
- BLSA: demographics/cognition/medical metadata and MRI/DTI/MPRAGE/REST workbook. No compact regional neuroimaging table included.
- OASIS3: useful sMRI/PET/clinical tables, already covered above.

Status:

- useful for labels and covariates.
- not enough for new regional neuroimaging claims unless matching derivative tables/matrices are added.

## Immediate Adapter Priorities

1. `AdniPetAdapter`
   - Add FDG/amyloid/tau regional and meta-ROI PET features.
   - Join by `RID`, `VISCODE`/date, and optionally nearest-to-baseline.
   - Canonical prefix: `pet_fdg_*`, `pet_amyloid_*`, `pet_tau_*`.

2. `AdniFcAdapter`
   - Read `data_fc.h5` + metadata.
   - Join subject IDs to `ADNIMERGE.PTID`.
   - Emit one row per subject or subject-session.
   - First feature set: `fc_mean_abs`, `fc_mean_positive`, `fc_global_sd`, plus optional selected edges.
   - Needs ROI labels for named-network interpretation; otherwise use generic edge IDs.

3. `AbcdSmriAdapter`
   - Join demographics, sMRI aseg/dsk volume/thickness, NIH Toolbox, CBCL/K-SADS ADHD, and motion/QC.
   - Emit baseline or selected session rows.
   - Canonical prefixes: `smri_*`, `beh_*`, `qc_*`.

4. `UkbIdpAdapter`
   - Use named `my_ukb_data_smri.csv` / `my_ukb_data_fmri.csv` first.
   - Emit global IDPs and cognitive variables.
   - Canonical prefixes: `smri_ukb_*`, `fmri_qc_*`, `beh_*`.

5. `Oasis3PetAdapter`
   - Add amyloid centiloid/tau/FDG-like PET summaries from OASIS3 PET tables.

## What I Need From You

Only one item is clearly needed for ADNI FC interpretability:

- ROI label / atlas mapping for the 450 x 450 FC matrices, if available.

If not available, we can still proceed with:

- global FC summaries,
- unnamed edge IDs,
- data-driven selected edge families.

For COBRE/ABIDE2/BLSA, I would need compact neuroimaging derivative tables or matrices if we want those datasets to contribute imaging claims, because the current bundle is mostly metadata/phenotype.

