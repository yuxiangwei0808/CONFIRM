# Archive manifest: one-off remote table conversion

- **Archived:** 2026-07-30
- **Original path:** `nbs_data/prepare_remote_tables.py`
- **Reason:** This historical discovery utility embeds a retired remote-layout
  assumption and is not imported or launched by the active preparation stack.
- **Replacement:** `external_dataset_registry.py`,
  `prepare_external_evidence.py`, and the local dataset registry.
- **Verified archived-content SHA-256:**
  `314ebc2ce13f1a16ffd0a9080420616045fb79fc0cda1d36cb98679c6108f16a`
