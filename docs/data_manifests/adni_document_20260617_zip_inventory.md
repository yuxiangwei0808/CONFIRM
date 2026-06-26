# ADNI Document Zip Inventory

Generated: 2026-06-17

Source archive: `data/raw/ADNI_document-20260617T002101Z-3-001.zip`

This archive is useful for the next ADNI adapter. It is not only narrative documentation; it contains analysis-ready CSV metadata and assessment tables.

## Visible Top-level Contents

Large nested archives:

- `ADNI_document/Imaging.zip`
- `ADNI_document/Genetic.zip`
- `ADNI_document/ADSP_PHC.zip`
- `ADNI_document/Assessments.zip`
- `ADNI_document/Diagnosis.zip`
- `ADNI_document/Neuropsychological.zip`
- `ADNI_document/Non-clinical_Assessments.zip`
- `ADNI_document/ADOPIC_Analysis.zip`

Useful visible CSV/PDF files from the outer archive:

- `ADNI_document/ADNIMERGE_14May2025.csv`
- `ADNI_document/DATADIC_01Jun2026.csv`
- `ADNI_document/PTDEMOG_01Jun2026.csv`
- `ADNI_document/PTDXCONV_01Jun2026.csv`
- `ADNI_document/BAIPETNMRCFDG_12_11_20_21Apr2026.csv`
- `ADNI_document/Diagnosis/DXSUM_01Jun2026.csv`
- `ADNI_document/Diagnosis/BLCHANGE_01Jun2026.csv`
- `ADNI_document/Diagnosis/ADSXLIST_01Jun2026.csv`
- `ADNI_document/Assessments/CDR_01Jun2026.csv`
- `ADNI_document/Assessments/MMSE_01Jun2026.csv`
- `ADNI_document/Assessments/MOCA_01Jun2026.csv`
- `ADNI_document/Assessments/ADAS_01Jun2026.csv`
- `ADNI_document/Assessments/FAQ_01Jun2026.csv`
- `ADNI_document/Assessments/NEUROBAT_01Jun2026.csv`
- `ADNI_document/Assessments/NEUROPATH_01Jun2026.csv`

## Immediate Use

The next adapter can selectively extract the small CSVs above and join them with the existing `data/raw/ADNIMERGE.xlsx`. This should support stronger ADNI disease/cognition/PET claims without needing raw images.

Likely first claims:

- AD versus cognitively normal diagnosis effects on cognitive scores.
- FDG-PET hypometabolism-style summary associations if the `BAIPETNMRCFDG` table has usable regional measures.
- Longitudinal diagnosis conversion or baseline diagnosis claims if `PTDXCONV`/`DXSUM` expose stable diagnosis fields.

## Still Unknown

The nested `Imaging.zip` has not been unpacked in this staging pass because it is large. It may contain image metadata, processed MRI summaries, or additional modality-specific tables. The next implementation pass should inspect it selectively, not unpack the whole archive unless needed.
