"""Export the paper's vector figures and reported results into the project site."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import pymupdf

SITE = Path(__file__).resolve().parent / "site"
FIGURES = {
    "overview": "CONFIRM_Figure1_overleaf.pdf",
    "frontier": "fig_confirm_frontier.pdf",
    "retrofit": "fig_retrofit_composition.pdf",
    "worked-example": "fig_worked_example.pdf",
}


def import_paper(paper: Path) -> None:
    (SITE / "figures").mkdir(parents=True, exist_ok=True)
    (SITE / "paper-data").mkdir(parents=True, exist_ok=True)
    manifest = {"figures": [], "sources": []}
    for name, filename in FIGURES.items():
        source = paper / "figures" / filename
        target = SITE / "figures" / f"{name}.svg"
        with pymupdf.open(source) as document:
            if len(document) != 1:
                raise ValueError(f"Expected one figure page: {source}")
            target.write_text(document[0].get_svg_image(text_as_path=True))
        manifest["figures"].append({"id": name, "source": f"figures/{filename}",
                                    "file": target.relative_to(SITE).as_posix(),
                                    "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                                    "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                                    "conversion": f"PyMuPDF {pymupdf.VersionBind}; text_as_path=True"})
    records = {}
    for name, filename in [("frontier", "fig_confirm_frontier_source.csv"),
                           ("retrofit", "fig_retrofit_composition_source.csv")]:
        source = paper / "figures" / filename
        target = SITE / "paper-data" / f"{name}.csv"
        target.write_text(source.read_text())
        records[name] = list(csv.DictReader(io.StringIO(source.read_text())))
        manifest["sources"].append({"source": f"figures/{filename}",
                                    "file": target.relative_to(SITE).as_posix(),
                                    "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                                    "sha256": hashlib.sha256(target.read_bytes()).hexdigest()})
    standards = {}
    for name, method in [("screen", "discovery"), ("replicate", "replicated"), ("full", "confirmed")]:
        rows = {row["metric"]: row for row in records["frontier"] if row["method"] == method}
        standards[name] = {key: {"supported": int(rows[metric]["supported_count"]),
                                 "total": int(rows[metric]["denominator"])}
                           for key, metric in [("literature", "recovery"), ("controls", "synthetic")]}
    table_source = paper / "figures/tab_feedback_control_main.tex"
    table = table_source.read_text()
    followup = []
    for name in ["Failure-blind retry", "Self-Refine", "CONFIRM diagnosis"]:
        match = re.search(re.escape(name) + r"\s*&\s*([\d,]+)\s*&\s*([\d,]+)\s*&\s*(\d+)\s*&\s*(\d+)\s*&\s*(\d+)", table)
        if not match:
            raise ValueError(f"Missing feedback row: {name}")
        values = [int(value.replace(",", "")) for value in match.groups()]
        followup.append(dict(zip(["method", "llm_calls", "candidates_evaluated", "supported_on_search", "parent_percent", "holdout_percent"], [name, *values])))
    manifest["sources"].append({"source": "figures/tab_feedback_control_main.tex",
                                "sha256": hashlib.sha256(table_source.read_bytes()).hexdigest()})
    payload = {"standards": standards, "followup": followup,
               "followup_design": {"parent_claims": 215, "rounds": 3, "candidates_per_round": 5},
               "benchmark": {"executable_tasks": 489, "scientific": 279, "controls": 150, "external_transfer": 60},
               "scope": "The main scalar comparison uses 31 literature-supported claims and 150 designed controls. These are subsets of the 489 executable benchmark tasks.",
               "holdout_definition": "Percentage of candidates evaluated on holdout data that receive support there; rounded as in the paper's representative run."}
    (SITE / "paper-data/results.json").write_text(json.dumps(payload, indent=2) + "\n")
    (SITE / "paper-data/manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Imported four vector figures and the paper's comparison and follow-up results.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paper", type=Path)
    import_paper(parser.parse_args().paper.resolve())
