"""Small public PubMed retrieval service shared by benchmark runners."""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class PubMedRecord(BaseModel):
    """A retrieved PubMed abstract with deterministic query provenance."""

    model_config = ConfigDict(extra="forbid")

    pmid: str
    title: str
    abstract: str
    journal: str = ""
    year: str = ""
    doi: str = ""
    mesh_terms: list[str] = Field(default_factory=list)
    query: str
    target_family: str
    modality: str
    retrieved_at: str


def _pubmed_url(endpoint: str, params: dict[str, str]) -> str:
    return (
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        f"{endpoint}?{urllib.parse.urlencode(params)}"
    )


def _urlopen_text(url: str, *, timeout: float) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read().decode("utf-8")


def search_pubmed_ids(
    query: str,
    *,
    max_records: int,
    email: str,
    api_key: str,
    timeout: float,
) -> list[str]:
    """Return PubMed IDs in PubMed relevance order."""

    params = {
        "db": "pubmed",
        "term": query,
        "retmode": "json",
        "retmax": str(max_records),
        "sort": "relevance",
        "tool": "confirm_claim_grounding",
    }
    if email:
        params["email"] = email
    if api_key:
        params["api_key"] = api_key
    data = json.loads(
        _urlopen_text(_pubmed_url("esearch.fcgi", params), timeout=timeout)
    )
    return [
        str(item)
        for item in data.get("esearchresult", {}).get("idlist", [])
    ]


def fetch_pubmed_records(
    ids: list[str],
    *,
    query: str,
    target_family: str,
    modality: str,
    email: str,
    api_key: str,
    timeout: float,
    retrieved_at: str,
) -> list[PubMedRecord]:
    """Fetch PubMed metadata and abstracts for an ordered ID list."""

    if not ids:
        return []
    params = {
        "db": "pubmed",
        "id": ",".join(ids),
        "retmode": "xml",
        "tool": "confirm_claim_grounding",
    }
    if email:
        params["email"] = email
    if api_key:
        params["api_key"] = api_key
    xml_text = _urlopen_text(
        _pubmed_url("efetch.fcgi", params),
        timeout=timeout,
    )
    return parse_pubmed_xml(
        xml_text,
        query=query,
        target_family=target_family,
        modality=modality,
        retrieved_at=retrieved_at,
    )


def parse_pubmed_xml(
    xml_text: str,
    *,
    query: str,
    target_family: str,
    modality: str,
    retrieved_at: str,
) -> list[PubMedRecord]:
    """Parse PubMed XML into the metadata consumed by adjudication."""

    root = ET.fromstring(xml_text)
    records: list[PubMedRecord] = []
    for article in root.findall(".//PubmedArticle"):
        pmid = _text(article.find(".//MedlineCitation/PMID"))
        title = _text(article.find(".//ArticleTitle"))
        abstract = " ".join(
            part
            for part in (
                _text(node)
                for node in article.findall(".//Abstract/AbstractText")
            )
            if part
        )
        if not abstract:
            continue
        doi = ""
        for item in article.findall(".//ArticleIdList/ArticleId"):
            if item.attrib.get("IdType") == "doi":
                doi = _text(item)
                break
        year = _text(article.find(".//JournalIssue/PubDate/Year"))
        if not year:
            medline_date = _text(
                article.find(".//JournalIssue/PubDate/MedlineDate")
            )
            year = medline_date[:4] if medline_date else ""
        records.append(
            PubMedRecord(
                pmid=pmid,
                title=title,
                abstract=abstract,
                journal=_text(article.find(".//Journal/Title")),
                year=year,
                doi=doi,
                mesh_terms=[
                    _text(node)
                    for node in article.findall(
                        ".//MeshHeading/DescriptorName"
                    )
                    if _text(node)
                ],
                query=query,
                target_family=target_family,
                modality=modality,
                retrieved_at=retrieved_at,
            )
        )
    return records


def _text(node: Optional[ET.Element]) -> str:
    if node is None:
        return ""
    return " ".join("".join(node.itertext()).split())
