# Remote Recheck Summary

Date: 2026-06-17  
Host: `arcdev`  
Mode: read-only path and table listing. No remote files were copied, edited, or deleted.

## Why This Recheck Was Done

The immediate experiment scope needed to be grounded in datasets we can actually use without waiting for NACC access or raw-image preprocessing.

## Verified Useful Remote Paths

The following paths exist on `arcdev` and are relevant to the near-term pipeline:

- `/data/users1/ywei/data`
- `/data/users1/ywei/data/ADNI/fmri`
- `/data/users1/ywei/data/OASIS3/fmri`
- `/data/users1/ywei/data/ABCD/fmri`
- `/data/users1/ywei/data/UKB/fmri`
- `/data/qneuromark/Data/COBRE/Data_info`
- `/data/qneuromark/Data/FBIRN/Data_info`
- `/data/qneuromark/Data/SZ_JH/Data_info`
- `/data/qneuromark/Data/AIBL/demos`

The following previously mentioned path does not exist:

- `/data/neuromark2/Data`

## Strongest Complete-Pipeline Data Available Now

The most reliable immediate pipeline should use the existing fMRI derivative cohorts under `/data/users1/ywei/data`, many of which are already copied locally under `data/raw_remote/arcdev/data/users1/ywei/data`.

Remote metadata files visible:

- ABCD: `/data/users1/ywei/data/ABCD/fmri/metadata.csv`
- ABIDE2: `/data/users1/ywei/data/ABIDE2/fmri/metadata.csv`
- ADHD200: `/data/users1/ywei/data/ADHD200/fmri/metadata_with_text_medical_all.csv`
- ADNI: `/data/users1/ywei/data/ADNI/fmri/metadata.csv`
- BLSA: `/data/users1/ywei/data/BLSA/fmri/metadata.csv`
- HCP: `/data/users1/ywei/data/HCP/fmri/metadata_with_text_medical.csv`
- HCP_Aging: `/data/users1/ywei/data/HCP_Aging/fmri/metadata.csv`
- OASIS3: `/data/users1/ywei/data/OASIS3/fmri/metadata.csv`
- UKB: `/data/users1/ywei/data/UKB/fmri/metadata.csv`
- EHBS: `/data/users1/ywei/data/EHBS/fmri/metadata_with_text_medical.csv`

Remote fMRI descriptor files visible:

- ABCD, ABIDE2, ADHD200, ADNI, EHBS, HCP, HCP_Aging, OASIS3, UKB have descriptor CSVs.
- Descriptor families include `fc_descriptors`, `fc_self_descriptors`, `ica_descriptors`, `ica_dyno_descriptors`, `region_self_descriptors`, and for some cohorts `dyno_descriptors`, `gradient_descriptors`, or `graph_descriptors`.

Local copied fMRI descriptor files already exist for:

- ABCD
- ABIDE2
- ADHD200
- ADNI
- HCP
- HCP_Aging
- OASIS3
- UKB

BLSA metadata is copied locally, but BLSA descriptor CSVs were not found in the copied local set.

## Other Local/Remote Candidates

### AIBL

Remote AIBL has useful small tables, including AV45, flutemetamol, APOE, neuropsych, MRI metadata, diagnosis, CDR, and MMSE. The user does not currently have AIBL access, so AIBL should not block the immediate pipeline.

### COBRE / FBIRN / SZ_JH

Remote clinical files are available for psychosis cohorts:

- COBRE diagnosis, PANSS, cognition, medication, refined phenotype files.
- FBIRN clinical, cognition, PANSS, CPZ, medication files.
- SZ_JH demo, medication, neuropsychological, SANS/SAPS files.

The missing piece is aligned imaging-derived feature tables for those subject IDs. Do not make psychosis imaging claims until that alignment is found or built.

### NACC

NACC should be deferred because the user does not currently have the needed clinical/diagnosis table. Local MRI/PET/CSF-style NACC tables are not enough for strong disease claims without the clinical anchor.

## Revised Immediate Scope

Build the complete pipeline on data already local and usable:

1. fMRI descriptor cohorts: ABCD, ABIDE2, ADHD200, ADNI, HCP, HCP_Aging, OASIS3, UKB.
2. Metadata-driven claims: age, sex, cognition, diagnosis where labels are explicit.
3. Imaging features: FC, ICA/dynamics, and region-level descriptor CSVs.
4. Gate evaluation: execution-only, confound, power, multiverse, replication.
5. Artifact-injection benchmark: site imbalance, label leakage, motion/quality proxy when available, synthetic null claims.

Defer:

- NACC disease claims.
- Psychosis imaging claims.
- Raw image processing.
