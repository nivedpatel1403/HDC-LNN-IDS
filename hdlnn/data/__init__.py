"""Data ingestion, canonicalization, quality auditing, and splitting package."""

from hdlnn.data.interfaces import ILoader
from hdlnn.data.loaders.ciciot import CICIoTLoader
from hdlnn.data.loaders.nslkdd import NSLKDDLoader
from hdlnn.data.loaders.unsw import UNSWLoader
from hdlnn.data.quality import DataQualityReport, check_quality, log_quality_report
from hdlnn.data.splitter import Splitter

__all__ = [
    "ILoader",
    "UNSWLoader",
    "CICIoTLoader",
    "NSLKDDLoader",
    "DataQualityReport",
    "check_quality",
    "log_quality_report",
    "Splitter",
]
