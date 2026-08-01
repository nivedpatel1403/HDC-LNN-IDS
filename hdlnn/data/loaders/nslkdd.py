"""NSL-KDD raw dataset loader stub.

Column mapping not yet finalized — requires the actual NSL-KDD raw file schema to complete.
Used exclusively for comparability with prior published HDC-IDS baselines per ADR-004.
"""

from hdlnn.contracts.flow import CanonicalFlow
from hdlnn.data.interfaces import ILoader

# TODO: Define NSL-KDD column mapping dictionary once raw schema is finalized
NSLKDD_COLUMN_MAP: dict[str, str] = {}


class NSLKDDLoader(ILoader):
    """Stub loader for NSL-KDD dataset used for baseline comparability."""

    def load(self, raw_path: str) -> list[CanonicalFlow]:
        """Raise NotImplementedError as NSLKDDLoader mapping is not yet implemented."""
        raise NotImplementedError("NSLKDDLoader mapping not yet implemented")
