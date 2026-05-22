from dataclasses import dataclass
from typing import Any, Dict, List, Optional

LLMMessage = Dict[str, Any]


@dataclass
class LLMUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class LLMCompletionResult:
    content: str
    usage: LLMUsage
    raw_response: Any = None


@dataclass
class LLMStreamChunk:
    text_delta: str = ""
    usage: Optional[LLMUsage] = None
    raw_chunk: Any = None


def normalize_messages(messages: List[LLMMessage]) -> List[LLMMessage]:
    return [dict(message) for message in messages]
