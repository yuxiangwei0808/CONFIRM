"""Build the public source and metadata archives from an explicit file list."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
VERSION = "0.1.1"
BENCHMARK_FILES = (
    "README.md", "DATA_DICTIONARY.md", "DATA_ACCESS.md", "USAGE.md", "CITATION.cff", "LICENSE",
    "PACKAGE_MANIFEST.json", "RELEASE_MANIFEST.json", "SHA256SUMS", "alignment_policy.json",
    "analysis_audit.json", "analysis_manifest.json", "benchmark_manifest.json", "benchmark_results.csv",
    "benchmark_splits.json", "benchmark_strata_summary.csv", "benchmark_summary.json", "cases.jsonl",
    "cluster_bootstrap_sensitivity.csv", "feedback_crosswalk_manifest.json", "feedback_parent_crosswalk.csv",
    "feedback_reference_summary.csv", "outcomes.jsonl", "reference_agreement_audit.csv", "references.jsonl",
    "target_reference_summary.csv", "tasks.jsonl", "unresolved_case_summary.csv", "v2_to_v2.1_crosswalk.csv",
)
BASE_FILES = (
    "README.md", "LICENSE", "NOTICE", "CITATION.cff", "RELEASE_NOTES.md", "pyproject.toml",
    "MANIFEST.in", ".gitignore", "requirements.txt",
    "docs/FRAMEWORK.md", "docs/DATA.md", "docs/NEUROCLAIMBENCH.md",
    "examples/README.md", "examples/audit_contract.py",
    "configs/cohort_maps/_TEMPLATE.yaml", "configs/contracts/example_age_atrophy.yaml",
    "scripts/package_public_release.py",
    "src/bench/__init__.py", "src/bench/io.py", "src/bench/benchmark.py", "src/bench/release_cli.py",
    "tests/test_contract_schema.py", "tests/test_analysis_identifiability.py", "tests/test_synthetic_gates.py",
    "tests/test_brainwide.py", "tests/test_confirmation_subtype.py", "tests/test_search_provenance.py",
    "tests/test_claim_search.py", "tests/test_claim_proposals.py", "tests/test_llm_factory.py",
    "tests/test_public_release.py",
    "tests/test_release_regressions.py",
    "tests/test_claim_search_pipeline_option.py",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_files() -> list[str]:
    files = [*BASE_FILES]
    for pattern in ("src/confirm/*.py", "src/confirm/ingest/*.py"):
        files.extend(p.relative_to(ROOT).as_posix() for p in ROOT.glob(pattern))
    files.extend(f"benchmark/neuroclaimbench-v2.1/{name}" for name in BENCHMARK_FILES)
    if len(files) != len(set(files)):
        raise ValueError("Duplicate path in release file list")
    for name in files:
        path = ROOT / name
        if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(ROOT):
            raise ValueError(f"Missing or non-local release file: {name}")
    return sorted(files)


def zip_tree(root: Path, destination: Path) -> None:
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            info = zipfile.ZipInfo(f"{root.name}/{path.relative_to(root).as_posix()}", (2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes())


def build(output: Path) -> dict:
    files = source_files()
    benchmark = ROOT / "benchmark/neuroclaimbench-v2.1"
    frozen = json.loads((benchmark / "benchmark_manifest.json").read_text())["files"]
    for name in ("cases", "references", "tasks", "outcomes"):
        if sha256(benchmark / f"{name}.jsonl") != frozen[name]["sha256"]:
            raise ValueError(f"Frozen benchmark record changed: {name}")
    for line in (benchmark / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split(maxsplit=1)
        if name not in BENCHMARK_FILES or sha256(benchmark / name) != expected:
            raise ValueError(f"Benchmark checksum or release inventory mismatch: {name}")
    output.mkdir(parents=True, exist_ok=True)
    target = output / f"confirm-{VERSION}"
    if target.exists():
        marker = target / "PACKAGE_MANIFEST.json"
        if not marker.is_file() or json.loads(marker.read_text()).get("generator") != "package_public_release.py":
            raise ValueError(f"Refusing to replace an unrelated directory: {target}")
    with tempfile.TemporaryDirectory(prefix=".package-", dir=output) as temporary:
        staged = Path(temporary) / target.name
        for name in files:
            dest = staged / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, dest)
        manifest = {"name": "CONFIRM", "version": VERSION,
                    "generator": "package_public_release.py", "code_license": "Apache-2.0",
                    "benchmark_license": "CC-BY-4.0", "files": {name: sha256(staged / name) for name in files}}
        (staged / "PACKAGE_MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        (staged / "SHA256SUMS").write_text("".join(
            f"{sha256(p)}  {p.relative_to(staged).as_posix()}\n"
            for p in sorted(staged.rglob("*")) if p.is_file()))
        zip_tree(staged, output / f"confirm-{VERSION}.zip")
        zip_tree(staged / "benchmark/neuroclaimbench-v2.1", output / "neuroclaimbench-v2.1.zip")
        if target.exists():
            shutil.rmtree(target)
        shutil.move(str(staged), target)
    archives = [output / f"confirm-{VERSION}.zip", output / "neuroclaimbench-v2.1.zip"]
    (output / "SHA256SUMS").write_text("".join(f"{sha256(p)}  {p.name}\n" for p in archives))
    return {"source_directory": str(target.resolve()), "source_files": len(files),
            "archives": [{"path": str(p.resolve()), "bytes": p.stat().st_size, "sha256": sha256(p)} for p in archives]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "release")
    args = parser.parse_args()
    print(json.dumps(build(args.out.resolve()), indent=2))


if __name__ == "__main__":
    main()
