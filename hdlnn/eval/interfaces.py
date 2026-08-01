"""Abstract base interface for evaluation pipelines."""

from abc import ABC, abstractmethod

from hdlnn.contracts.flow import CanonicalFlow
from hdlnn.eval.run_result import RunResult


class IPipeline(ABC):
    """Abstract interface for end-to-end evaluation pipeline adapters."""

    @abstractmethod
    def run(
        self,
        train_flows: list[CanonicalFlow],
        val_flows: list[CanonicalFlow],
        test_flows: list[CanonicalFlow],
    ) -> RunResult:
        """Run an end-to-end pipeline execution over train, val, and test flow splits.

        Args:
            train_flows: List of CanonicalFlow contract objects for training.
            val_flows: List of CanonicalFlow contract objects for validation.
            test_flows: List of CanonicalFlow contract objects for testing.

        Returns:
            RunResult container object populated with execution metrics and artifacts.
        """
        pass
