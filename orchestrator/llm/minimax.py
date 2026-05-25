import re
from typing import Optional

from .openai_chat import OpenAIChatProviderConfig

MINIMAX_DEFAULT_MODEL = "MiniMax-M2.7"
MINIMAX_DEFAULT_BASE_URL = "https://api.minimaxi.com/v1"
MINIMAX_API_KEY_FIELD = "minimax_api_key"
MINIMAX_EXTRA_BODY = {"reasoning_split": True}
MINIMAX_MIN_MEMORY_COMPLETION_TOKENS = 1024
_MINIMAX_THINKING_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def strip_minimax_thinking(content: str) -> str:
    """Remove MiniMax inline thinking blocks from provider content.

    Args:
        content (str):
            Text returned by MiniMax.

    Returns:
        str:
            Text with ``<think>...</think>`` blocks removed.
    """
    return _MINIMAX_THINKING_PATTERN.sub("", content).strip()


def build_minimax_config(
    *,
    model_name: str,
    base_url: Optional[str],
    timeout: Optional[float],
    proxy_url: Optional[str],
) -> OpenAIChatProviderConfig:
    """Build the OpenAI-compatible provider config for MiniMax.

    Args:
        model_name (str):
            MiniMax model name.
        base_url (Optional[str]):
            OpenAI-compatible MiniMax base URL.
        timeout (Optional[float]):
            Request timeout in seconds.
        proxy_url (Optional[str]):
            Optional HTTP proxy URL.

    Returns:
        OpenAIChatProviderConfig:
            Provider config consumed by the shared OpenAI-compatible layer.
    """
    return OpenAIChatProviderConfig(
        provider_name="MiniMax",
        api_key_field=MINIMAX_API_KEY_FIELD,
        model_name=model_name,
        base_url=base_url,
        timeout=timeout,
        proxy_url=proxy_url,
    )
