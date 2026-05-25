from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx

from .errors import LLMEmptyResponseError, LLMProviderCallError, MissingAPIKeyException
from .types import LLMCompletionResult, LLMMessage, LLMStreamChunk, LLMUsage, normalize_messages


@dataclass
class AnthropicMessagesProviderConfig:
    """Configuration for Anthropic Messages API calls.

    Attributes:
        api_key_field (str):
            Field name used to read the Anthropic API key from user settings.
        model_name (str):
            Default Anthropic model name.
        timeout (Optional[float]):
            Optional request timeout in seconds.
        proxy_url (Optional[str]):
            Optional HTTP proxy URL for provider requests.
    """

    api_key_field: str
    model_name: str
    timeout: Optional[float] = None
    proxy_url: Optional[str] = None


def _get_api_key(api_keys: Optional[Dict[str, Any]], config: AnthropicMessagesProviderConfig) -> str:
    """Read the Anthropic API key from user settings.

    Args:
        api_keys (Optional[Dict[str, Any]]):
            User-provided API key mapping.
        config (AnthropicMessagesProviderConfig):
            Anthropic provider configuration.

    Returns:
        str:
            API key value.

    Raises:
        MissingAPIKeyException:
            Raised when the configured key field is absent or empty.
    """
    api_key = (api_keys or {}).get(config.api_key_field, "")
    if not api_key:
        raise MissingAPIKeyException("Anthropic API key is not found in the API keys.")
    return api_key


def _build_http_client(config: AnthropicMessagesProviderConfig) -> Optional[httpx.AsyncClient]:
    """Create a proxied HTTP client when a proxy is configured.

    Args:
        config (AnthropicMessagesProviderConfig):
            Anthropic provider configuration.

    Returns:
        Optional[httpx.AsyncClient]:
            A proxied HTTP client, or None to let the SDK create its default
            client.
    """
    if config.proxy_url is None:
        return None
    return httpx.AsyncClient(proxy=config.proxy_url)


def create_client(api_keys: Optional[Dict[str, Any]], config: AnthropicMessagesProviderConfig) -> Any:
    """Create an Anthropic async client.

    Args:
        api_keys (Optional[Dict[str, Any]]):
            User-provided API key mapping.
        config (AnthropicMessagesProviderConfig):
            Anthropic provider configuration.

    Returns:
        Any:
            Async Anthropic client instance.
    """
    import anthropic

    kwargs: Dict[str, Any] = {
        "api_key": _get_api_key(api_keys, config),
        "http_client": _build_http_client(config),
    }
    if config.timeout is not None:
        kwargs["timeout"] = config.timeout
    return anthropic.AsyncAnthropic(**kwargs)


def usage_from_anthropic(raw_usage: Any) -> LLMUsage:
    """Normalize Anthropic usage metadata.

    Args:
        raw_usage (Any):
            Anthropic usage object returned by the SDK.

    Returns:
        LLMUsage:
            Normalized token usage. Missing values default to zero.
    """
    if raw_usage is None:
        return LLMUsage()
    prompt_tokens = int(getattr(raw_usage, "input_tokens", 0) or 0)
    completion_tokens = int(getattr(raw_usage, "output_tokens", 0) or 0)
    return LLMUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )


async def stream(
    *,
    api_keys: Optional[Dict[str, Any]],
    config: AnthropicMessagesProviderConfig,
    system: str,
    messages: List[LLMMessage],
    model_override: Optional[str] = None,
    max_tokens: int = 1000,
    temperature: float = 1,
    client: Optional[Any] = None,
) -> AsyncIterator[LLMStreamChunk]:
    """Run a streaming Anthropic Messages API call.

    Args:
        api_keys (Optional[Dict[str, Any]]):
            User-provided API key mapping.
        config (AnthropicMessagesProviderConfig):
            Anthropic provider configuration.
        system (str):
            System prompt content.
        messages (List[LLMMessage]):
            Conversation messages.
        model_override (Optional[str], optional):
            Per-request model override. Defaults to None.
        max_tokens (int, optional):
            Maximum completion token count. Defaults to 1000.
        temperature (float, optional):
            Sampling temperature. Defaults to 1.
        client (Optional[Any], optional):
            Existing SDK client to reuse. Defaults to None.

    Yields:
        LLMStreamChunk:
            Normalized text deltas and final usage metadata.

    Raises:
        LLMProviderCallError:
            Raised when the Anthropic streaming call fails.
    """
    owned_client = client is None
    llm_client = client or create_client(api_keys, config)
    try:
        async with llm_client.messages.stream(
            model=model_override or config.model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=normalize_messages(messages),
        ) as response_stream:
            async for text_delta in response_stream.text_stream:
                yield LLMStreamChunk(text_delta=text_delta or "")
            final_message = await response_stream.get_final_message()
            if final_message is not None:
                yield LLMStreamChunk(usage=usage_from_anthropic(getattr(final_message, "usage", None)))
    except Exception as exc:
        raise LLMProviderCallError(f"Anthropic messages stream failed: {exc}") from exc
    finally:
        if owned_client and hasattr(llm_client, "close"):
            await llm_client.close()


async def complete(
    *,
    api_keys: Optional[Dict[str, Any]],
    config: AnthropicMessagesProviderConfig,
    system: str,
    messages: List[LLMMessage],
    model_override: Optional[str] = None,
    max_tokens: int = 1000,
    temperature: float = 1,
    client: Optional[Any] = None,
) -> LLMCompletionResult:
    """Run a non-streaming Anthropic Messages API call.

    Args:
        api_keys (Optional[Dict[str, Any]]):
            User-provided API key mapping.
        config (AnthropicMessagesProviderConfig):
            Anthropic provider configuration.
        system (str):
            System prompt content.
        messages (List[LLMMessage]):
            Conversation messages.
        model_override (Optional[str], optional):
            Per-request model override. Defaults to None.
        max_tokens (int, optional):
            Maximum completion token count. Defaults to 1000.
        temperature (float, optional):
            Sampling temperature. Defaults to 1.
        client (Optional[Any], optional):
            Existing SDK client to reuse. Defaults to None.

    Returns:
        LLMCompletionResult:
            Normalized completion content, usage, and raw response.

    Raises:
        LLMProviderCallError:
            Raised when the Anthropic call fails.
        LLMEmptyResponseError:
            Raised when the provider returns no text content.
    """
    owned_client = client is None
    llm_client = client or create_client(api_keys, config)
    try:
        response = await llm_client.messages.create(
            model=model_override or config.model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=normalize_messages(messages),
        )
    except Exception as exc:
        raise LLMProviderCallError(f"Anthropic messages completion failed: {exc}") from exc
    finally:
        if owned_client and hasattr(llm_client, "close"):
            await llm_client.close()
    content = "".join(
        getattr(block, "text", "") for block in getattr(response, "content", []) if getattr(block, "type", "") == "text"
    )
    if not content:
        raise LLMEmptyResponseError("Anthropic returned empty content")
    return LLMCompletionResult(
        content=content,
        usage=usage_from_anthropic(getattr(response, "usage", None)),
        raw_response=response,
    )
