"""Check paper evidence, guide records, local links, and publication files."""
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit
import csv
import hashlib
import json
import re
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent
SITE = ROOT / "site"
GUIDE = SITE / "guide"


class Page(HTMLParser):
    def __init__(self, path):
        super().__init__()
        self.links = []
        self.ids = set()
        self.rows = []
        self.row = None
        self.cell = None
        self.feed(path.read_text())

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        self.links.extend(attrs[key] for key in ("src", "href") if key in attrs)
        if "id" in attrs:
            assert attrs["id"] not in self.ids, attrs["id"]
            self.ids.add(attrs["id"])
        if tag == "tr":
            self.row = []
        if tag in ("td", "th"):
            self.cell = ""

    def handle_data(self, data):
        if self.cell is not None:
            self.cell += data

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self.cell is not None:
            self.row.append(" ".join(self.cell.split()))
            self.cell = None
        if tag == "tr":
            self.rows.append(self.row)
            self.row = None


def validate_paper():
    manifest = json.loads((SITE / "paper-data/manifest.json").read_text())
    for record in manifest["figures"] + manifest["sources"]:
        if "file" not in record:
            continue
        path = SITE / record["file"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"], path
        if path.suffix == ".svg":
            svg = ET.fromstring(path.read_text())
            assert not list(svg.iter("{http://www.w3.org/2000/svg}image")), path
    results = json.loads((SITE / "paper-data/results.json").read_text())
    with (SITE / "paper-data/frontier.csv").open() as source:
        frontier = list(csv.DictReader(source))
    for standard, method in [("screen", "discovery"), ("replicate", "replicated"), ("full", "confirmed")]:
        for name, metric in [("literature", "recovery"), ("controls", "synthetic")]:
            row, = [r for r in frontier if r["method"] == method and r["metric"] == metric]
            assert results["standards"][standard][name] == {
                "supported": int(row["supported_count"]), "total": int(row["denominator"])}
    baseline_controls = [int(r["supported_count"]) for r in frontier
                         if r["family"] != "confirm" and r["metric"] == "synthetic"]
    assert len(baseline_controls) == 4 and (min(baseline_controls), max(baseline_controls)) == (34, 57)
    with (SITE / "paper-data/retrofit.csv").open() as source:
        retrofit = {(r["arm"], r["metric"]): (int(r["supported_count"]), int(r["denominator"]))
                    for r in csv.DictReader(source)}
    assert retrofit["agent_alone", "recovery"] == retrofit["veto_full", "recovery"] == (15, 31)
    assert retrofit["agent_alone", "synthetic_support"] == (34, 150)
    assert retrofit["veto_full", "synthetic_support"] == (0, 150)
    rows = Page(SITE / "index.html").rows[1:]
    expected = [[r["method"], f'{r["llm_calls"]:,}', f'{r["candidates_evaluated"]:,}',
                 str(r["supported_on_search"]), f'{r["holdout_percent"]}%'] for r in results["followup"]]
    assert rows == expected, (rows, expected)
    benchmark = results["benchmark"]
    assert benchmark["executable_tasks"] == sum(benchmark[k] for k in ["scientific", "controls", "external_transfer"])


def validate():
    validate_paper()
    payload = json.loads((GUIDE / "data/examples.json").read_text())
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
        record = json.loads((GUIDE / f"data/{key}-record.json").read_text())
        plan = json.loads((GUIDE / f"data/{key}-plan.json").read_text())
        assert record["results"] == example["results"]
        assert plan == example["contract"] == record["results"]["contract"]
        assert record["contract_sha256"] == example["contract_sha256"]
    trace = json.loads((GUIDE / "data/followup.json").read_text())
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
            for link in Page(path).links:
                url = urlsplit(link)
                if url.scheme or url.netloc:
                    continue
                target = (path.parent / unquote(url.path)).resolve() if url.path else path.resolve()
                assert target.is_relative_to(SITE.resolve()), link
                if target.is_dir():
                    target /= "index.html"
                assert target.is_file(), (path, link)
                if url.fragment and target.suffix == ".html":
                    assert unquote(url.fragment) in Page(target).ids, (path, link)
    print("Site validation passed: paper figures and results, three synthetic examples, nine decisions, recorded follow-up, downloads, and local links.")


if __name__ == "__main__":
    validate()
