# Archive manifest: data discovery records

- **Archived:** 2026-07-30
- **Original paths:** `docs/DATA_DISCOVERY_REMOTE.md`,
  `docs/DATA_REQUIREMENTS_NEXT.md`, and `docs/data_manifests/`
- **Reason:** Historical cluster reconnaissance and one-off inventory records;
  they are not inputs to the active data-preparation pipeline.
- **Replacement:** `docs/LAB_HANDOFF.md`,
  `configs/external_datasets.example.yml`, and the ignored
  `configs/external_datasets.local.yml`.
- **Pre-move content-list SHA-256:**
  `dceae0e5b0da59c5a5616f0c5c9b8f9981077a60627e7c76f7df63a4bc2a2d50`
- **Verified archived-content SHA-256:**
  `77bce0967d5d892b419720ec66f3cc6343b8a6e8bcc849fafcf28be1e94774d0`

Recompute the content-list fingerprint with:

```bash
find . -type f ! -name ARCHIVE_MANIFEST.md -print0 | xargs -0 shasum -a 256 | awk '{print $1}' | sort | shasum -a 256
```
