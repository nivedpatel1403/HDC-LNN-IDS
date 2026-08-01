"""Abstract base interfaces for data loaders."""

from abc import ABC, abstractmethod
from hdlnn.contracts.flow import CanonicalFlow


class ILoader(ABC):
    """Abstract base class interface for dataset loaders."""

    @abstractmethod
    def load(self, raw_path: str) -> list[CanonicalFlow]:
        """Load raw dataset file and parse into canonical flow contracts.

        Args:
            raw_path: Absolute or relative file path to the raw dataset file.

        Returns:
            List of CanonicalFlow Pydantic model instances.
        """
        pass
