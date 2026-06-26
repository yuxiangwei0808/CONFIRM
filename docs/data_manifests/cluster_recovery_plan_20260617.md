# Cluster Recovery Plan (arcdev) — 2026-06-17

`ssh arcdev` (host arctrdgndev101). Remote python w/ scipy/h5py: `~/.conda/envs/playground/bin/python`. Read-only exploration done.

## Availability
- Recoverable data lives under `/data/qneuromark/Data/`.
- UNAVAILABLE (empty / flaky autofs mount): HCP_Aging, HCP_Development, ABCD, AMP_SCZ, BSNIP, BSNIP2, MCIC(data). Drop from plans unless remounted.

## Feature modalities
- **FC/ICA**: per-subject ROI timecourses `ZN_Neuromark/ZN_Prep_fMRI/<id>/RC_ROI.npy` shape (T,160); 160 components identical across cohorts. Static FNC = CPU correlation → fc_ features. (GSR_ROI.npy is the GSR variant — pick one consistently; use RC_ROI.)
- **sMRI regional tables**: only ADNI (FreeSurfer 7.1.1 stats tree), HCP (merged FS CSV), GSP (FS cols in phenotype CSV). BLSA/AIBL are VBM-only (no clean eTIV).

## Extraction targets + join keys
### (a) sMRI aging/sex positives
- ADNI: `ADNI/Updated/freesurfer_7.1.1/<PTID_ImageUID>/stats/{aseg,lh/rh.aparc}.stats` + `Data_info/ADNIMERGE.xlsx`; join dir `002_S_0295_S150056`→PTID. eTIV in aseg.stats. (We already have ADNI hippocampus from local ADNIMERGE; cluster adds full regional + eTIV.)
- GSP: `GSP/Data_info/DataRelease_2014-04-22.csv`; key `Subject_ID`; ICV + regional thickness; healthy; **age 5-yr binned**. Best for eTIV-adjusted SEX differences.
- HCP: `HCP/old_HCP/Data_info/CSV_freesurfer.csv`; key `Subject`; healthy; age binned (exact in RESTRICTED_*.csv).

### (b) SZ psychosis case/control + replication pair (COBRE ↔ FBIRN)
- COBRE: RC_ROI.npy (T,160)→FNC; dx `Data_info/COBRE_fromVince.csv` (SubjID, Dx 1=SZ/0=HC, Age, Sex, PANSS); join dir==SubjID; 204 fMRI, SZ 73/HC 70.
- FBIRN: RC_ROI.npy→FNC; dx `Data_info/fBIRN_CMINDS_4rsfMRI2_G.csv` (SubjectID, sDEMOG_DIAGNOSIS, sDEMOG_GENDER, nDEMOG_CUR_AGE); join dir==SubjectID; 370 fMRI, SZ 170/HC 162; multi-site (site in FBIRN_CLIN_*.xlsx).
- 3rd leg candidate: ChineseSZ (populated; dx field unverified).

### (c) Autism/ADHD real site IDs
- ABIDE1: `Autism/ABIDE1/Data_info/Phenotypic_V1_0b_preprocessed1.csv`; key FILE_ID (=feature dir, e.g. NYU_0051058); SITE_ID, DX_GROUP (1=autism,2=control), AGE_AT_SCAN, SEX; 1112.
- ABIDE2: `Autism/ABIDE2/Data_info/ABIDEII_Composite_Phenotypic.csv`; key SITE_ID+SUB_ID; 1115.
- ADHD200: site = directory (NYU/Peking/KKI/OHSU/WashU/Pittsburgh/NeuroIMAGE); per-site phenotypic CSVs in `ADHD/ADHD200/Data_info/`.

### Bonus
- OASIS3 multimodal: precomputed GICA + dFNC `OASIS/OASIS3/Multimodal_Updated/`; demos `oasis_matched_smri_fmri_demos.csv` (CDR/dx) — FC aging arm (~2154).

## Notes
- 160 components (not classic NeuroMark-1.0 53). Fine within qneuromark; note if mixing external 53-comp data.
- FC must be derived from RC_ROI.npy (cheap); only OASIS3 has precomputed FNC.

## Extraction corrections (from blocked attempt, 2026-06-17)
- **ACCESS**: `ssh arcdev` requires the GSU **VPN (winvpn)** to be connected, or the hostname `arctrdgndev101.rs.gsu.edu` does not resolve. VPN must be up for any extraction.
- **COBRE**: local demo is `pheno_comb_cobre_all.csv` (key `URSI`; TEXT `Diagnosis`: Schizophrenia 114 / Control 117 / Schizoaffective 11 / Bipolar 10; N=252). The earlier-assumed `COBRE_fromVince.csv` (SubjID, Dx 0/1) was NOT found locally — verify on cluster. Map SZ/HC from the text field and DECIDE explicitly on the 21 Schizoaffective+Bipolar (recommend: SZ vs Control only, exclude the in-between for a clean case/control). Verify whether RC_ROI feature dirs are named by URSI or SubjID before joining.
- **FBIRN**: `fBIRN_CMINDS_4rsfMRI2_G.csv` (N=332, SZ 170 / HC 162) has NO site column; dx already coded SZ/HC. Per-subject site needs another cluster source (scan-id prefix / site map) else fall back to site="FBIRN".
- **fc convention**: existing cohorts use Yeo-7 network-block cols (`fc_fc_<Net>_<Net>`) from Schaefer/CC200→Yeo-7 — does NOT apply to NeuroMark 160-ICA. For COBRE/FBIRN use the 4 canonical scalar descriptors (fc_mean_abs, fc_mean_positive, fc_within_network, fc_between_network) with a documented 160-component partition applied identically to both cohorts.
- Existing fMRI adapters (abide.py, adhd200.py) currently emit only fc_mean_abs + fc_mean_positive.

## CORRECTION (2026-06-17 eve): NeuroMark ICA features live under Results/, NOT Data/
The earlier "SZ breadth exhausted" finding was WRONG — it searched `/data/qneuromark/Data/<cohort>/ZN_Neuromark` (raw/sparse for many cohorts). The processed per-subject ICA timecourses live under **`/data/qneuromark/Results/ICA/<cohort>/`** (GIFT `*_ica_br<N>.mat`), with dynamic FNC under **`/data/qneuromark/Results/DFNC/<cohort>/`**.

**Cohorts with ICA results (Results/ICA):** ABCD, ABIDE1, ABIDE2, ADHD, ADHD_SJ_PK, ADNI, Autism_baby, BSNIP, BSNIP2, ChineseMDD, ChineseSZ, COBRE, ECT, EMBARC, FBIRN, GSP, HCP, HCP_Aging, HCP_Development, HCP_task, HIV_Duke, JH, LA5c, MDD_DIRECT, OASIS3, OSUCH, PANStudy, PK, UKBiobank, UKB_old.

**SZ/psychosis cohorts for replication breadth:** COBRE, FBIRN, ChineseSZ, BSNIP, BSNIP2, JH (+ maybe PANStudy). Extraction: per-subject static FNC from Results/ICA timecourses → 4 fc_ descriptors with the SAME 7-domain partition as COBRE/FBIRN; build clean SZ-vs-HC (BSNIP/BSNIP2 also contain bipolar/schizoaffective/relatives → exclude).
**Other domains available (future fragile/candidate, not clean positives):** MDD (ChineseMDD, MDD_DIRECT, EMBARC), more autism (ABIDE2, Autism_baby), more ADHD (ADHD, PK).
**Olin_ASD_SZ has NO `/data/qneuromark/Results/ICA/Olin_ASD_SZ` folder** (path the user gave doesn't exist there).
