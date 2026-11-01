"""Check the visual guide's evidence, local links, and publication files."""
from pathlib import Path
import json
import re

ROOT = Path(__file__).resolve().parent
SITE = ROOT / "site"


def validate():
    payload = json.loads((SITE / "data/examples.json").read_text())
    expected = {"supported": [True, True, True], "site_confounded": [False, False, False],
                "nonreplicating": [True, False, False]}
    assert payload["synthetic"] and payload["llm_calls"] == 0
    assert len(payload["examples"]) == 3
    for example in payload["examples"]:
        key = example["id"]
        assert [example["support"][s] for s in ["screen", "replicate", "full"]] == expected[key]
        g = example["results"]["verdict"]["gates"]
        base = all(g[name] for name in ["search_provenance", "multiplicity", "confound", "confound_completeness"])
        assert [base, base and g["replication"], base and g["replication"] and g["power"] and g["multiverse"]] == expected[key]
        assert example["results"]["primary"]["n"] == 600
        assert example["results"]["replication"]["cohort_results"][0]["effect"]["n"] == 600
        assert all(len(points) == 180 for points in example["points"].values())
        record = json.loads((SITE / f"data/{key}-record.json").read_text())
        plan = json.loads((SITE / f"data/{key}-plan.json").read_text())
        assert record["results"] == example["results"]
        assert plan == example["contract"] == record["results"]["contract"]
        assert record["contract_sha256"] == example["contract_sha256"]
    trace = json.loads((SITE / "data/followup.json").read_text())
    assert trace["kind"] == "recorded_benchmark_control"
    assert trace["failed_parent_gates"] == ["confound_completeness"]
    assert trace["parent_label"] == "fragile" and len(trace["candidates"]) == 2
    assert "site" not in trace["parent_contract"]["covariates"]
    assert trace["final_family_size"] == 23
    for c in trace["candidates"]:
        assert "site" in c["contract"]["covariates"]
        assert c["evaluated"] and c["validation"]["ok"] and not c["supported"]
        assert c["contract"]["estimand"]["group"] == trace["parent_contract"]["estimand"]["group"]
    assert trace["candidates"][0]["primary"]["p"] > .6
    assert trace["candidates"][1]["region_count"] == 21
    assert trace["candidates"][1]["significant_regions"] == 0
    for path in SITE.rglob("*"):
        if not path.is_file():
            continue
        assert not path.is_symlink()
        text = path.read_text()
        assert not re.search(r"/Users/|/home/users/|/data/users\d*/|sk-[A-Za-z0-9_-]{24,}", text), path
        if path.suffix == ".json":
            json.loads(text, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
        if path.suffix == ".html":
            for link in re.findall(r'(?:href|src)="([^"]+)"', text):
                if link.startswith(("https://", "http://", "#")):
                    continue
                assert (path.parent / link).is_file(), link
    print("Guide validation passed: three synthetic examples, nine decisions, recorded follow-up, downloads, and local links.")


if __name__ == "__main__":
    validate()
