"""Exception hierarchy for the HD-LNN architecture."""


class HdlnnError(Exception):
    """Base exception class for all HD-LNN errors."""

    pass


class DataValidationError(HdlnnError):
    """Raised when data validation fails for canonical flows or features."""

    pass


class ItemMemoryError(HdlnnError):
    """Raised when HDC item memory operations encounter errors."""

    pass


class ConfigError(HdlnnError):
    """Raised when configuration loading or validation fails."""

    pass


class ManifoldFitError(HdlnnError):
    """Raised when manifold drift estimation or divergence calculation fails."""

    pass
