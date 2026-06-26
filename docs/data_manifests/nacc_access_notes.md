# NACC Access Notes

Generated: 2026-06-17

NACC disease claims are deferred until the needed clinical phenotype data are available locally.

Official access path:

- Data request process: https://www.naccdata.org/data-request-process/
- Data Front Door: https://www.naccdata.org/data-front-door/
- Multimodal Query Tool: https://www.naccdata.org/multimodal-query-tool/
- Imaging data overview: https://www.naccdata.org/about-nacc-data/imaging-data/
- MRI Preview System: https://www.naccdata.org/mri-preview-system/
- Query System: https://www.naccdata.org/query-system/

What to request later:

1. Quick Access File / UDS clinical phenotype data with `NACCID`, visit date or visit number, age, sex, diagnosis/cognitive status, CDR, MMSE or other cognitive scores.
2. MRI calculated/summary measures if available, including volumes, cortical thickness, surface area, intracranial volume, hippocampal volume, and QC fields.
3. Amyloid PET or tau PET summary/SUVR/centiloid values if available.
4. CSF biomarker data with assay/method metadata for A beta, p-tau, and t-tau.
5. A `NACCID` list filtered through the Multimodal Query Tool if an imaging-rich subset is needed before a full request.

Local status:

- Already local: NACC MRI table, amyloid-PET metadata table, CSF biomarker table.
- Missing for disease claims: UDS clinical phenotype/demographic diagnosis table that can be joined to `NACCID`.
