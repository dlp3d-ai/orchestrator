try:
    from ..utils.exception import MissingAPIKeyException
except Exception:

    class MissingAPIKeyException(Exception):
        """Fallback used when generated protocol modules are unavailable in unit tests."""


class LLMProviderError(RuntimeError):
    """Base error for shared LLM provider failures."""


class LLMEmptyResponseError(LLMProviderError):
    """Raised when a provider returns no usable text content."""


class LLMProviderCallError(LLMProviderError):
    """Raised when the provider SDK/API call fails."""


__all__ = [
    "LLMEmptyResponseError",
    "LLMProviderCallError",
    "LLMProviderError",
    "MissingAPIKeyException",
]
