"""Abstract base interface for HDC encoders."""

from abc import ABC, abstractmethod

from hdlnn.contracts.encoded import EncodedHypervector
from hdlnn.contracts.flow import CanonicalFlow


class IEncoder(ABC):
    """Abstract interface for Hyperdimensional Computing (HDC) encoders."""

    @abstractmethod
    def fit(self, train_flows: list[CanonicalFlow]) -> None:
        """Fit the encoder (ItemMemory and numeric level hypervectors) using train flows only.

        Args:
            train_flows: List of CanonicalFlow objects from the training split.
        """
        pass

    @abstractmethod
    def encode(self, flow: CanonicalFlow) -> EncodedHypervector:
        """Encode a single CanonicalFlow into an EncodedHypervector contract.

        Args:
            flow: CanonicalFlow object to encode.

        Returns:
            EncodedHypervector contract model instance.
        """
        pass
