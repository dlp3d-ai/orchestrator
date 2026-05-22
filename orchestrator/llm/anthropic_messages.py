from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx

from .errors import LLMEmptyResponseError, LLMProviderCallError, MissingAPIKeyException
from .types import LLMCompletionResult, LLMMessage, LLMStreamChunk, LLMUsage, normalize_messages


@dataclass
class AnthropicMessagesProviderConfig:
    api_key_field: str
    model_name: str
    timeout: Optional[float] = None
    proxy_url: Optional[str] = None


def _get_api_key(api_keys: Optional[Dict[str, Any]], config: AnthropicMessagesProviderConfig) -> str:
    api_key = (api_keys or {}).get(config.api_key_field, "")
    if not api_key:
        raise MissingAPIKeyException("Anthropic API key is not found in the API keys.")
    return api_key


def _build_http_client(config: AnthropicMessagesProviderConfig) -> Optional[httpx.AsyncClient]:
    if config.proxy_url is None:
        return None
    return httpx.AsyncClient(proxy=config.proxy_url)


def create_client(api_keys: Optional[Dict[str, Any]], config: AnthropicMessagesProviderConfig) -> Any:
    import anthropic

    kwargs: Dict[str, Any] = {
        "api_key": _get_api_key(api_keys, config),
        "http_client": _build_http_client(config),
    }
    if config.timeout is not None:
        kwargs["timeout"] = config.timeout
    return anthropic.AsyncAnthropic(**kwargs)


def usage_from_anthropic(raw_usage: Any) -> LLMUsage:
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
