"""Abstract base interface for sequence models (LNN cells and baselines)."""

from abc import ABC, abstractmethod

from hdlnn.contracts.encoded import EncodedHypervector
from hdlnn.contracts.trajectory import TrajectoryState


class ISequenceModel(ABC):
    """Abstract baseline-agnostic interface for continuous-time sequence models."""

    @abstractmethod
    def forward(
        self, sequence: list[EncodedHypervector], entity_id: str = "default"
    ) -> list[TrajectoryState]:
        """Process a full ordered sequence of hypervectors for an entity in batch mode.

        Args:
            sequence: Ordered list of EncodedHypervector contract objects.
            entity_id: Identifier string for the sequence entity (e.g. source IP).

        Returns:
            List of TrajectoryState contract instances corresponding to each sequence step.
        """
        pass

    @abstractmethod
    def step(
        self,
        hv: EncodedHypervector,
        prev_state: TrajectoryState | None = None,
        entity_id: str | None = None,
    ) -> TrajectoryState:
        """Process a single record in streaming mode and return an updated TrajectoryState.

        Args:
            hv: Single EncodedHypervector contract object.
            prev_state: Optional previous TrajectoryState carrying hidden state reference.
            entity_id: Optional entity ID string (inferred from prev_state if omitted).

        Returns:
            Updated TrajectoryState contract instance for the new record.
        """
        pass
