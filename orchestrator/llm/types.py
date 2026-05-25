from dataclasses import dataclass
from typing import Any, Dict, List, Optional

LLMMessage = Dict[str, Any]


@dataclass
class LLMUsage:
    """Normalized token usage returned by an LLM provider.

    Attributes:
        prompt_tokens (int):
            Number of input tokens.
        completion_tokens (int):
            Number of output tokens.
        total_tokens (int):
            Total token count.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class LLMCompletionResult:
    """Normalized non-streaming LLM completion result.

    Attributes:
        content (str):
            Text content returned by the provider.
        usage (LLMUsage):
            Normalized token usage.
        raw_response (Any):
            Original provider response for debugging and provider-specific data.
    """

    content: str
    usage: LLMUsage
    raw_response: Any = None


@dataclass
class LLMStreamChunk:
    """Normalized streaming LLM response chunk.

    Attributes:
        text_delta (str):
            Text delta emitted by the provider.
        usage (Optional[LLMUsage]):
            Optional usage metadata, usually available on the final chunk.
        raw_chunk (Any):
            Original provider chunk for debugging and provider-specific data.
    """

    text_delta: str = ""
    usage: Optional[LLMUsage] = None
    raw_chunk: Any = None


def normalize_messages(messages: List[LLMMessage]) -> List[LLMMessage]:
    """Copy chat messages into plain dictionaries.

    Args:
        messages (List[LLMMessage]):
            Messages provided by business adapters.

    Returns:
        List[LLMMessage]:
            Shallow-copied message dictionaries safe to pass to SDK clients.
    """
    return [dict(message) for message in messages]
