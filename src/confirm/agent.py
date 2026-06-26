"""CONFIRM orchestration and LLM agent layer."""

from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from bench.claim_library import lookup_ref_effect
from confirm.analysis import audit_confound_completeness, run_primary
from confirm.brainwide import run_brainwide
from confirm.contract import ClaimContract, load_contract
from confirm.llm import LLMClient, get_llm
from confirm.multiverse import run_brainwide_multiverse, run_multiverse
from confirm.power import power_check
from confirm.provenance import make_receipt, write_receipt
from confirm.replication import replicate, replicate_brainwide
from confirm.results import EffectResult, PowerResult, RegionTable
from confirm.schema import idp_columns, validate_canonical
from confirm.verdict import Verdict, decide, decide_brainwide

SEED = 20260615

DOMAIN_PRIOR_SYSTEM_PROMPT = """You draft CONFIRM claim contracts for CPU-only neuroimaging derivative analyses.
Rules:
- Emit ONLY YAML or JSON matching the ClaimContract schema.
- Do not include prose, markdown fences, or computed results.
- Prefer adjustment for age, sex, eTIV, and site when those variables are available.
- Use FDR for imaging families.
- Require cross-cohort replication before any confirmed claim.
- The LLM drafts contracts and prose only; executed code computes all numbers and gates.
- Do not weaken deterministic gates to make a claim pass.
"""


def _cohort_path(data_dir: Path, cohort: str) -> Path:
    path = data_dir / f"{cohort}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Canonical cohort parquet not found: {path}")
    return path


def _load_canonical(path: Path) -> pd.DataFrame:
    return validate_canonical(pd.read_parquet(path))


def build_data_catalog(data_dir: str | Path) -> dict[str, Any]:
    """Build a compact catalog from canonical cohort parquet files."""

    root = Path(data_dir)
    cohorts: list[dict[str, Any]] = []
    for path in sorted(root.glob("*.parquet")):
        try:
            df = _load_canonical(path)
        except Exception as exc:
            cohorts.append({"file": str(path), "status": "unreadable", "error": str(exc)})
            continue
        idps = idp_columns(df.columns)
        cohorts.append(
            {
                "cohort": path.stem,
                "file": str(path),
                "n": int(len(df)),
                "columns": list(df.columns),
                "idps": idps,
                "region_names": idps,
                "atlases": sorted({("smri" if col.startswith("smri_") else "pet" if col.startswith("pet_") else "fc") for col in idps}),
                "dx_levels": sorted(str(value) for value in df["dx"].dropna().unique()) if "dx" in df.columns else [],
            }
        )
    return {"data_dir": str(root), "cohorts": cohorts}


def _strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return stripped


def _parse_contract_text(text: str) -> dict[str, Any]:
    payload = _strip_code_fence(text)
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        data = yaml.safe_load(payload)
        if not isinstance(data, dict):
            raise ValueError("LLM output did not parse to a mapping")
        return data


_EXAMPLE_CONTRACT = {
    "claim_id": "ad_hippocampal_atrophy",
    "question": "Is hippocampal volume reduced in Alzheimer's disease vs controls?",
    "estimand": {
        "type": "group_diff",
        "outcome": "smri_hippocampus",
        "predictor": "dx",
        "group": {"var": "dx", "case": "Dementia", "control": "CN"},
        "direction": "negative",
        "unit": "scalar",
    },
    "covariates": ["age", "sex", "eTIV"],
    "inclusion": None,
    "discovery_cohort": "ADNI",
    "replication_cohorts": ["OASIS3"],
    "gates": {
        "multiplicity": {"method": "fdr_bh", "alpha": 0.05, "family_size": 1},
        "confound": {"require_covariates": ["age", "sex", "eTIV"], "motion_check": False},
        "power": {"min_power": 0.8, "ref_effect": None},
        "multiverse": {"min_fraction_consistent": 0.6},
        "replication": {"alpha": 0.05, "require_same_sign": True, "require_ci_overlap": False, "harmonize": "combat"},
    },
    "reporting_language_allowed": ["confirmed", "non_replicated", "under_powered", "fragile"],
}


def _contract_prompt(question: str, catalog: dict[str, Any], previous_error: str | None = None) -> str:
    cohorts = [c for c in catalog.get("cohorts", []) if "cohort" in c]
    names = [c["cohort"] for c in cohorts]
    details = {c["cohort"]: {"n": c.get("n"), "idps": c.get("idps", []), "dx_levels": c.get("dx_levels", [])} for c in cohorts}
    instructions = (
        "Output ONLY one YAML object (no prose, no markdown fences). It must have EXACTLY these "
        "top-level keys: claim_id, question, estimand, covariates, inclusion, discovery_cohort, "
        "replication_cohorts, gates, reporting_language_allowed.\n"
        "- discovery_cohort: ONE cohort NAME string from AVAILABLE_COHORTS.\n"
        "- replication_cohorts: a LIST of cohort NAME strings from AVAILABLE_COHORTS (must differ from discovery; "
        "for a 'replicate across cohorts' question, include at least one).\n"
        "- estimand.predictor is the variable name (for group_diff use the grouping variable, e.g. 'dx'); "
        "estimand.group is {var, case, control} for group_diff, else null. estimand.direction is negative|positive|two_sided.\n"
        "- estimand.outcome MUST be an IDP name that exists in BOTH the discovery and replication cohorts' idps. "
        "For a brain-wide pattern set estimand.unit='brainwide' and estimand.outcome to a region prefix like 'smri_'.\n"
        "- gates is an OBJECT (not a list) exactly like the EXAMPLE. reporting_language_allowed is the list in the EXAMPLE.\n"
        "- Match the EXACT structure of EXAMPLE_CONTRACT below; only change values, never key names."
    )
    payload = {
        "QUESTION": question,
        "AVAILABLE_COHORTS": names,
        "COHORT_DETAILS": details,
        "INSTRUCTIONS": instructions,
        "EXAMPLE_CONTRACT": _EXAMPLE_CONTRACT,
    }
    if previous_error:
        payload["FIX_THIS_VALIDATION_ERROR_FROM_YOUR_LAST_OUTPUT"] = previous_error
    return yaml.safe_dump(payload, sort_keys=False)


def draft_contract(question: str, catalog: dict[str, Any], llm: LLMClient | None = None) -> ClaimContract:
    """Draft and validate a claim contract from a natural-language question."""

    client = llm or get_llm()
    last_error: str | None = None
    for _ in range(3):
        prompt = _contract_prompt(question, catalog, last_error)
        text = client.complete(DOMAIN_PRIOR_SYSTEM_PROMPT, prompt)
        try:
            return ClaimContract.model_validate(_parse_contract_text(text))
        except Exception as exc:
            last_error = str(exc)
    raise ValueError(f"LLM failed to draft a valid claim contract after 3 attempts: {last_error}")


def draft_contract_from_question(nl_question: str, *, context: dict[str, object] | None = None) -> ClaimContract:
    """Backward-compatible wrapper for the B0 hook name."""

    catalog = dict(context or {})
    if "cohorts" not in catalog and "data_dir" in catalog:
        catalog = build_data_catalog(str(catalog["data_dir"]))
    return draft_contract(nl_question, catalog)


def _run_scalar_contract(
    contract: ClaimContract,
    discovery_df: pd.DataFrame,
    replication_dfs: list[pd.DataFrame],
    ref_effect: float | None,
) -> tuple[Verdict, dict[str, Any]]:
    confound_audit = audit_confound_completeness(discovery_df, contract)
    primary = run_primary(discovery_df, contract)
    multiverse = run_multiverse(discovery_df, contract, forks=None)
    power = power_check(primary, contract, ref_effect=ref_effect)
    replication = replicate(primary, discovery_df, replication_dfs, contract)
    verdict = decide(primary, multiverse, power, replication, contract, confound_audit=confound_audit)
    return verdict, {
        "primary": primary,
        "confound_completeness": confound_audit,
        "multiverse": multiverse,
        "power": power,
        "replication": replication,
        "verdict": verdict,
    }


def _best_region_effect(regions: RegionTable) -> EffectResult:
    ordered = sorted(regions.regions, key=lambda region: (not region.significant, region.effect.p))
    return ordered[0].effect


def _run_brainwide_contract(
    contract: ClaimContract,
    discovery_df: pd.DataFrame,
    replication_dfs: list[pd.DataFrame],
) -> tuple[Verdict, dict[str, Any]]:
    confound_audit = audit_confound_completeness(discovery_df, contract)
    regions = run_brainwide(discovery_df, contract)
    multiverse = run_brainwide_multiverse(discovery_df, regions, contract)
    power = power_check(_best_region_effect(regions), contract, ref_effect=contract.gates.power.ref_effect)
    replication = replicate_brainwide(regions, discovery_df, replication_dfs, contract)
    verdict = decide_brainwide(regions, multiverse, power, replication, contract, confound_audit=confound_audit)
    return verdict, {
        "regions": regions,
        "confound_completeness": confound_audit,
        "multiverse": multiverse,
        "power": power,
        "replication": replication,
        "verdict": verdict,
    }


def _execute_contract(
    contract: ClaimContract,
    data_root: Path,
    ref_effect: float | None = None,
) -> tuple[Verdict, dict[str, Any], list[Path]]:
    discovery_path = _cohort_path(data_root, contract.discovery_cohort)
    replication_paths = [_cohort_path(data_root, cohort) for cohort in contract.replication_cohorts]
    discovery_df = _load_canonical(discovery_path)
    replication_dfs = [_load_canonical(path) for path in replication_paths]
    if contract.estimand.unit == "brainwide":
        verdict, results = _run_brainwide_contract(contract, discovery_df, replication_dfs)
    else:
        verdict, results = _run_scalar_contract(contract, discovery_df, replication_dfs, ref_effect)
    return verdict, {"contract": contract.model_dump(), **results}, [discovery_path, *replication_paths]


def _numbers_in_bundle(bundle: Any) -> list[float]:
    values: list[float] = []
    if hasattr(bundle, "to_dict"):
        return _numbers_in_bundle(bundle.to_dict())
    if isinstance(bundle, dict):
        for value in bundle.values():
            values.extend(_numbers_in_bundle(value))
    elif isinstance(bundle, (list, tuple)):
        for value in bundle:
            values.extend(_numbers_in_bundle(value))
    elif isinstance(bundle, bool):
        pass
    elif isinstance(bundle, (int, float)) and math.isfinite(float(bundle)):
        values.append(float(bundle))
    return values


_NUMBER_RE = re.compile(r"(?<![\w-])-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?(?![\w-])")


def _number_allowed(text_number: str, allowed: list[float]) -> bool:
    value = float(text_number)
    return any(math.isclose(value, allowed_value, rel_tol=1e-3, abs_tol=1e-3) for allowed_value in allowed)


def _strip_unapproved_numbers(narrative: str, results: dict[str, Any]) -> str:
    allowed = _numbers_in_bundle(results)
    bad = [match.group(0) for match in _NUMBER_RE.finditer(narrative) if not _number_allowed(match.group(0), allowed)]
    if not bad:
        return narrative
    sentences = re.split(r"(?<=[.!?])\s+", narrative.strip())
    kept = [
        sentence
        for sentence in sentences
        if not any(not _number_allowed(match.group(0), allowed) for match in _NUMBER_RE.finditer(sentence))
    ]
    suffix = "Numeric statements not present in the computed result bundle were removed."
    return (" ".join(kept).strip() + " " + suffix).strip()


def interpret(verdict: Verdict, region_table_or_effect: Any, atlas: str | None = None, llm: LLMClient | None = None) -> str:
    """Generate an interpretation narrative with numeric anti-hallucination checks."""

    client = llm or get_llm()
    results = {
        "verdict": verdict.to_dict(),
        "result": region_table_or_effect.to_dict() if hasattr(region_table_or_effect, "to_dict") else region_table_or_effect,
        "atlas": atlas,
    }
    system = (
        "You interpret CONFIRM engine outputs. Use only numbers present in the supplied JSON. "
        "Name regions only if they appear in the supplied result. Do not invent methods or results."
    )
    user = json.dumps(results, indent=2, sort_keys=True)
    narrative = client.complete(system, user).strip()
    return _strip_unapproved_numbers(narrative, results)


def run_claim(contract_path: str | Path, data_dir: str | Path, out_dir: str | Path, command: list[str] | None = None) -> Verdict:
    """Run the full CONFIRM gate chain for one claim contract."""

    contract = load_contract(contract_path)
    data_root = Path(data_dir)
    ref_effect = contract.gates.power.ref_effect
    if ref_effect is None and contract.estimand.unit == "scalar":
        ref_effect = lookup_ref_effect(contract_path, contract.claim_id)
    verdict, results, cohort_paths = _execute_contract(contract, data_root, ref_effect=ref_effect)
    receipt = make_receipt(
        contract_path=contract_path,
        cohort_paths=cohort_paths,
        command=command,
        seed=SEED,
        results=results,
    )
    write_receipt(out_dir, receipt)
    return verdict


def run_question(question: str, data_dir: str | Path, out: str | Path, approve: bool = True) -> Verdict:
    """Draft, optionally approve, execute, interpret, and receipt a natural-language question."""

    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    catalog = build_data_catalog(data_dir)
    llm = get_llm()
    contract = draft_contract(question, catalog, llm=llm)
    contract_text = yaml.safe_dump(contract.model_dump(mode="json"), sort_keys=False)
    if approve:
        print(contract_text)
        answer = input("Approve this CONFIRM contract? [y/N] ").strip().lower()
        if answer not in {"y", "yes"}:
            raise SystemExit("Contract not approved.")

    contract_path = out_dir / "drafted_contract.yaml"
    contract_path.write_text(contract_text, encoding="utf-8")
    verdict, results, cohort_paths = _execute_contract(contract, Path(data_dir), ref_effect=contract.gates.power.ref_effect)
    primary_result = results.get("regions") or results.get("primary")
    narrative = interpret(verdict, primary_result, atlas=contract.estimand.region_set, llm=llm)
    (out_dir / "narrative.txt").write_text(narrative + "\n", encoding="utf-8")

    prompt_hash = hashlib.sha256(_contract_prompt(question, catalog).encode("utf-8")).hexdigest()
    receipt = make_receipt(
        contract_path=contract_path,
        cohort_paths=cohort_paths,
        command=sys.argv,
        seed=SEED,
        results={
            **results,
            "agent": {
                "question": question,
                "llm_model": getattr(llm, "model", type(llm).__name__),
                "prompt_hash": prompt_hash,
                "drafted_contract": contract.model_dump(mode="json"),
                "narrative": narrative,
            },
        },
    )
    write_receipt(out_dir, receipt)
    return verdict
