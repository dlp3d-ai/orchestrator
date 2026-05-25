import asyncio
from types import SimpleNamespace

import pytest

from orchestrator.llm.errors import LLMEmptyResponseError, LLMProviderCallError, MissingAPIKeyException
from orchestrator.llm.openai_chat import (
    _RESPONSE_FORMAT_UNSUPPORTED_CACHE,
    OpenAIChatProviderConfig,
    complete,
    create_client,
    stream,
)


class FakeCompletions:
    def __init__(self, response=None, stream_chunks=None, error=None):
        self.response = response
        self.stream_chunks = stream_chunks
        self.error = error
        self.kwargs = None

    async def create(self, **kwargs):
        self.kwargs = kwargs
        if self.error:
            raise self.error
        if kwargs.get("stream"):
            return FakeAsyncIterator(self.stream_chunks)
        return self.response


class FakeSequenceCompletions:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class FakeAsyncIterator:
    def __init__(self, chunks):
        self.chunks = chunks or []

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self.chunks:
            raise StopAsyncIteration
        return self.chunks.pop(0)


class FakeClient:
    def __init__(self, completions):
        self.chat = SimpleNamespace(completions=completions)


def make_response(content="ok", prompt_tokens=3, completion_tokens=4):
    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )
    message = SimpleNamespace(content=content)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)


def make_stream_chunk(text="", usage=None):
    delta = SimpleNamespace(content=text)
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta)] if text else [], usage=usage)


@pytest.fixture(autouse=True)
def clear_response_format_cache():
    _RESPONSE_FORMAT_UNSUPPORTED_CACHE.clear()
    yield
    _RESPONSE_FORMAT_UNSUPPORTED_CACHE.clear()


def test_complete_normalizes_content_usage_and_request_kwargs():
    completions = FakeCompletions(response=make_response(content='{"type":"accept"}'))
    client = FakeClient(completions)
    config = OpenAIChatProviderConfig(
        provider_name="DeepSeek",
        api_key_field="deepseek_api_key",
        model_name="deepseek-chat",
        base_url="https://api.deepseek.com",
    )

    result = asyncio.run(
        complete(
            api_keys=None,
            client=client,
            config=config,
            model_override="override-model",
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=12,
            response_format={"type": "json_object"},
            extra_body={"thinking": {"enabled": False}},
        )
    )

    assert result.content == '{"type":"accept"}'
    assert result.usage.prompt_tokens == 3
    assert result.usage.completion_tokens == 4
    assert completions.kwargs["model"] == "override-model"
    assert completions.kwargs["max_tokens"] == 12
    assert completions.kwargs["response_format"] == {"type": "json_object"}
    assert completions.kwargs["extra_body"] == {"thinking": {"enabled": False}}


def test_stream_normalizes_text_and_usage_chunks():
    usage = SimpleNamespace(prompt_tokens=5, completion_tokens=6, total_tokens=11)
    completions = FakeCompletions(
        stream_chunks=[make_stream_chunk("he"), make_stream_chunk("llo"), make_stream_chunk(usage=usage)]
    )
    client = FakeClient(completions)
    config = OpenAIChatProviderConfig(provider_name="OpenAI", api_key_field="openai_api_key", model_name="gpt-test")

    async def collect_chunks():
        return [
            chunk
            async for chunk in stream(
                api_keys=None,
                client=client,
                config=config,
                messages=[{"role": "user", "content": "hello"}],
                stream_options={"include_usage": True},
                extra_body={"thinking": {"enabled": False}},
            )
        ]

    chunks = asyncio.run(collect_chunks())

    assert [chunk.text_delta for chunk in chunks] == ["he", "llo", ""]
    assert chunks[-1].usage.prompt_tokens == 5
    assert completions.kwargs["stream"] is True
    assert completions.kwargs["stream_options"] == {"include_usage": True}
    assert completions.kwargs["extra_body"] == {"thinking": {"enabled": False}}


def test_complete_wraps_provider_errors_and_empty_content():
    config = OpenAIChatProviderConfig(provider_name="OpenAI", api_key_field="openai_api_key", model_name="gpt-test")
    with pytest.raises(LLMProviderCallError):
        asyncio.run(
            complete(
                api_keys=None,
                client=FakeClient(FakeCompletions(error=RuntimeError("boom"))),
                config=config,
                messages=[],
            )
        )

    with pytest.raises(LLMEmptyResponseError):
        asyncio.run(
            complete(
                api_keys=None,
                client=FakeClient(FakeCompletions(response=make_response(content=None))),
                config=config,
                messages=[],
            )
        )


def test_create_client_requires_configured_api_key():
    config = OpenAIChatProviderConfig(provider_name="XAI", api_key_field="xai_api_key", model_name="grok")
    with pytest.raises(MissingAPIKeyException):
        create_client({}, config)


def test_complete_retries_without_response_format_for_guided_grammar_errors():
    completions = FakeSequenceCompletions(
        [
            RuntimeError("guided_grammar has compile_grammar_error: Unsupported tokenizer type"),
            make_response(content="<output>fallback text</output>"),
        ]
    )
    client = FakeClient(completions)
    config = OpenAIChatProviderConfig(
        provider_name="SenseNova",
        api_key_field="sensenova_api_key",
        model_name="sensenova-6.7-flash-lite",
        base_url="https://token.sensenova.cn/v1",
    )

    result = asyncio.run(
        complete(
            api_keys=None,
            client=client,
            config=config,
            messages=[{"role": "user", "content": "hello"}],
            response_format={"type": "json_schema"},
        )
    )

    assert result.content == "<output>fallback text</output>"
    assert len(completions.calls) == 2
    assert completions.calls[0]["response_format"] == {"type": "json_schema"}
    assert "response_format" not in completions.calls[1]


def test_complete_skips_response_format_after_unsupported_format_is_cached():
    config = OpenAIChatProviderConfig(
        provider_name="SenseNova",
        api_key_field="sensenova_api_key",
        model_name="sensenova-6.7-flash-lite",
        base_url="https://token.sensenova.cn/v1",
    )
    response_format = {"type": "json_schema"}

    first_completions = FakeSequenceCompletions(
        [
            RuntimeError("response_format json_schema guided_grammar compile_grammar_error"),
            make_response(content="accept"),
        ]
    )
    asyncio.run(
        complete(
            api_keys=None,
            client=FakeClient(first_completions),
            config=config,
            messages=[{"role": "user", "content": "hello"}],
            response_format=response_format,
        )
    )

    second_completions = FakeSequenceCompletions([make_response(content="accept")])
    result = asyncio.run(
        complete(
            api_keys=None,
            client=FakeClient(second_completions),
            config=config,
            messages=[{"role": "user", "content": "hello"}],
            response_format=response_format,
        )
    )

    assert result.content == "accept"
    assert len(second_completions.calls) == 1
    assert "response_format" not in second_completions.calls[0]


def test_complete_does_not_retry_unrelated_errors_with_response_format():
    completions = FakeSequenceCompletions([RuntimeError("401 invalid api key")])
    config = OpenAIChatProviderConfig(provider_name="OpenAI", api_key_field="openai_api_key", model_name="gpt-test")

    with pytest.raises(LLMProviderCallError):
        asyncio.run(
            complete(
                api_keys=None,
                client=FakeClient(completions),
                config=config,
                messages=[{"role": "user", "content": "hello"}],
                response_format={"type": "json_schema"},
            )
        )

    assert len(completions.calls) == 1
    assert completions.calls[0]["response_format"] == {"type": "json_schema"}
    assert not _RESPONSE_FORMAT_UNSUPPORTED_CACHE


def test_complete_does_not_retry_without_response_format_even_when_error_mentions_json_schema():
    completions = FakeSequenceCompletions([RuntimeError("json_schema compile_grammar_error")])
    config = OpenAIChatProviderConfig(provider_name="OpenAI", api_key_field="openai_api_key", model_name="gpt-test")

    with pytest.raises(LLMProviderCallError):
        asyncio.run(
            complete(
                api_keys=None,
                client=FakeClient(completions),
                config=config,
                messages=[{"role": "user", "content": "hello"}],
            )
        )

    assert len(completions.calls) == 1
    assert "response_format" not in completions.calls[0]
    assert not _RESPONSE_FORMAT_UNSUPPORTED_CACHE


def test_complete_response_format_cache_isolated_by_model_override():
    config = OpenAIChatProviderConfig(
        provider_name="SenseNova",
        api_key_field="sensenova_api_key",
        model_name="sensenova-default",
        base_url="https://token.sensenova.cn/v1",
    )
    response_format = {"type": "json_schema"}
    first_completions = FakeSequenceCompletions(
        [
            RuntimeError("guided_grammar compile_grammar_error"),
            make_response(content="fallback"),
        ]
    )
    asyncio.run(
        complete(
            api_keys=None,
            client=FakeClient(first_completions),
            config=config,
            model_override="sensenova-6.7-flash-lite",
            messages=[{"role": "user", "content": "hello"}],
            response_format=response_format,
        )
    )

    second_completions = FakeSequenceCompletions([make_response(content="structured")])
    asyncio.run(
        complete(
            api_keys=None,
            client=FakeClient(second_completions),
            config=config,
            model_override="different-model",
            messages=[{"role": "user", "content": "hello"}],
            response_format=response_format,
        )
    )

    assert len(second_completions.calls) == 1
    assert second_completions.calls[0]["response_format"] == response_format
