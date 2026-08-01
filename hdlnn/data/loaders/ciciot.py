"""CICIoT2023 raw dataset loader stub.

Column mapping not yet finalized — requires the actual CICIoT2023 raw file schema to complete.
Used exclusively for zero-shot cross-dataset evaluation per ADR-004, never trained on directly.
"""

from hdlnn.contracts.flow import CanonicalFlow
from hdlnn.data.interfaces import ILoader

# TODO: Define CICIoT2023 column mapping dictionary once raw schema is finalized
CICIOT_COLUMN_MAP: dict[str, str] = {}


class CICIoTLoader(ILoader):
    """Stub loader for CICIoT2023 dataset used in zero-shot cross-dataset evaluation."""

    def load(self, raw_path: str) -> list[CanonicalFlow]:
        """Raise NotImplementedError as CICIoT2023 mapping is not yet implemented."""
        raise NotImplementedError("CICIoTLoader mapping not yet implemented")
