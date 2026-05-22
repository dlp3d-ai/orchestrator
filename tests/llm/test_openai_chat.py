import asyncio
from types import SimpleNamespace

import pytest

from orchestrator.llm.errors import LLMEmptyResponseError, LLMProviderCallError, MissingAPIKeyException
from orchestrator.llm.openai_chat import OpenAIChatProviderConfig, complete, create_client, stream


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
