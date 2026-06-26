"""Cohort adapters that normalize public or user-provided data into CONFIRM tables."""

from confirm.ingest.abide import AbideAdapter
from confirm.ingest.adhd200 import Adhd200Adapter
from confirm.ingest.adni import AdniAdapter
from confirm.ingest.base import CohortAdapter
from confirm.ingest.generic_csv import GenericCsvAdapter
from confirm.ingest.oasis1 import Oasis1Adapter
from confirm.ingest.oasis3 import Oasis3Adapter

__all__ = [
    "AbideAdapter",
    "Adhd200Adapter",
    "AdniAdapter",
    "CohortAdapter",
    "GenericCsvAdapter",
    "Oasis1Adapter",
    "Oasis3Adapter",
]
