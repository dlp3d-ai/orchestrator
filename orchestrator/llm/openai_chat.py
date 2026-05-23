from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

import httpx

from .errors import LLMEmptyResponseError, LLMProviderCallError, MissingAPIKeyException
from .types import LLMCompletionResult, LLMMessage, LLMStreamChunk, LLMUsage, normalize_messages


@dataclass
class OpenAIChatProviderConfig:
    provider_name: str
    api_key_field: str
    model_name: str
    base_url: Optional[str] = None
    timeout: Optional[float] = None
    proxy_url: Optional[str] = None


_RESPONSE_FORMAT_UNSUPPORTED_CACHE: set[Tuple[str, Optional[str], str, str]] = set()
_RESPONSE_FORMAT_UNSUPPORTED_SIGNALS = (
    "guided_grammar",
    "compile_grammar_error",
    "Unsupported tokenizer type",
    "json_schema",
    "response_format",
)


def _response_format_cache_key(
    config: OpenAIChatProviderConfig,
    model_name: str,
    response_format: Dict[str, Any],
) -> Tuple[str, Optional[str], str, str]:
    response_format_type = response_format.get("type", "unknown")
    return (config.provider_name, config.base_url, model_name, str(response_format_type))


def _is_response_format_unsupported_error(exc: Exception) -> bool:
    error_text = str(exc)
    return any(signal in error_text for signal in _RESPONSE_FORMAT_UNSUPPORTED_SIGNALS)


def _get_api_key(api_keys: Optional[Dict[str, Any]], config: OpenAIChatProviderConfig) -> str:
    api_key = (api_keys or {}).get(config.api_key_field, "")
    if not api_key:
        raise MissingAPIKeyException(f"{config.provider_name} API key is not found in the API keys.")
    return api_key


def _build_http_client(config: OpenAIChatProviderConfig) -> Optional[httpx.AsyncClient]:
    if config.proxy_url is None:
        return None
    return httpx.AsyncClient(proxy=config.proxy_url)


def create_client(api_keys: Optional[Dict[str, Any]], config: OpenAIChatProviderConfig) -> Any:
    import openai

    kwargs: Dict[str, Any] = {
        "api_key": _get_api_key(api_keys, config),
        "http_client": _build_http_client(config),
    }
    if config.base_url:
        kwargs["base_url"] = config.base_url
    if config.timeout is not None:
        kwargs["timeout"] = config.timeout
    return openai.AsyncOpenAI(**kwargs)


def usage_from_openai(raw_usage: Any) -> LLMUsage:
    if raw_usage is None:
        return LLMUsage()
    prompt_tokens = int(getattr(raw_usage, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(raw_usage, "completion_tokens", 0) or 0)
    total_tokens = int(getattr(raw_usage, "total_tokens", prompt_tokens + completion_tokens) or 0)
    return LLMUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
    )


async def complete(
    *,
    api_keys: Optional[Dict[str, Any]],
    config: OpenAIChatProviderConfig,
    messages: List[LLMMessage],
    model_override: Optional[str] = None,
    temperature: float = 1,
    max_tokens: Optional[int] = None,
    response_format: Optional[Dict[str, Any]] = None,
    extra_body: Optional[Dict[str, Any]] = None,
    client: Optional[Any] = None,
) -> LLMCompletionResult:
    owned_client = client is None
    llm_client = client or create_client(api_keys, config)
    model_name = model_override or config.model_name
    response_format_cache_key = (
        _response_format_cache_key(config, model_name, response_format) if response_format is not None else None
    )
    should_send_response_format = (
        response_format is not None and response_format_cache_key not in _RESPONSE_FORMAT_UNSUPPORTED_CACHE
    )
    kwargs: Dict[str, Any] = {
        "model": model_name,
        "messages": normalize_messages(messages),
        "temperature": temperature,
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if should_send_response_format:
        kwargs["response_format"] = response_format
    if extra_body is not None:
        kwargs["extra_body"] = extra_body
    try:
        try:
            response = await llm_client.chat.completions.create(**kwargs)
        except Exception as exc:
            if not should_send_response_format or not _is_response_format_unsupported_error(exc):
                raise
            if response_format_cache_key is not None:
                _RESPONSE_FORMAT_UNSUPPORTED_CACHE.add(response_format_cache_key)
            retry_kwargs = dict(kwargs)
            retry_kwargs.pop("response_format", None)
            response = await llm_client.chat.completions.create(**retry_kwargs)
    except Exception as exc:
        raise LLMProviderCallError(f"{config.provider_name} chat completion failed: {exc}") from exc
    finally:
        if owned_client and hasattr(llm_client, "close"):
            await llm_client.close()
    content = response.choices[0].message.content if response.choices else None
    if content is None:
        raise LLMEmptyResponseError(f"{config.provider_name} returned empty content")
    return LLMCompletionResult(
        content=content,
        usage=usage_from_openai(response.usage),
        raw_response=response,
    )


async def stream(
    *,
    api_keys: Optional[Dict[str, Any]],
    config: OpenAIChatProviderConfig,
    messages: List[LLMMessage],
    model_override: Optional[str] = None,
    temperature: float = 1,
    max_tokens: Optional[int] = None,
    stream_options: Optional[Dict[str, Any]] = None,
    extra_body: Optional[Dict[str, Any]] = None,
    client: Optional[Any] = None,
) -> AsyncIterator[LLMStreamChunk]:
    owned_client = client is None
    llm_client = client or create_client(api_keys, config)
    kwargs: Dict[str, Any] = {
        "model": model_override or config.model_name,
        "messages": normalize_messages(messages),
        "temperature": temperature,
        "stream": True,
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if stream_options is not None:
        kwargs["stream_options"] = stream_options
    if extra_body is not None:
        kwargs["extra_body"] = extra_body
    try:
        response_stream = await llm_client.chat.completions.create(**kwargs)
        async for chunk in response_stream:
            text_delta = ""
            if len(chunk.choices) > 0 and chunk.choices[0].delta.content is not None:
                text_delta = chunk.choices[0].delta.content
            yield LLMStreamChunk(
                text_delta=text_delta,
                usage=usage_from_openai(chunk.usage) if getattr(chunk, "usage", None) else None,
                raw_chunk=chunk,
            )
    except Exception as exc:
        raise LLMProviderCallError(f"{config.provider_name} stream chat completion failed: {exc}") from exc
    finally:
        if owned_client and hasattr(llm_client, "close"):
            await llm_client.close()
