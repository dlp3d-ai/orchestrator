import asyncio
from types import SimpleNamespace

import pytest

from orchestrator.llm.anthropic_messages import AnthropicMessagesProviderConfig, complete, stream
from orchestrator.llm.errors import LLMEmptyResponseError, LLMProviderCallError


class FakeMessages:
    def __init__(self, completion_response=None, stream_context=None, error=None):
        self.completion_response = completion_response
        self.stream_context = stream_context
        self.error = error
        self.create_kwargs = None
        self.stream_kwargs = None

    async def create(self, **kwargs):
        self.create_kwargs = kwargs
        if self.error:
            raise self.error
        return self.completion_response

    def stream(self, **kwargs):
        self.stream_kwargs = kwargs
        if self.error:
            raise self.error
        return self.stream_context


class FakeClient:
    def __init__(self, messages):
        self.messages = messages


class FakeStreamContext:
    def __init__(self, text_chunks, final_message=None):
        self.text_stream = FakeAsyncIterator(text_chunks)
        self.final_message = final_message

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def get_final_message(self):
        return self.final_message


class FakeAsyncIterator:
    def __init__(self, chunks):
        self.chunks = list(chunks)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self.chunks:
            raise StopAsyncIteration
        return self.chunks.pop(0)


def make_usage(input_tokens=2, output_tokens=3):
    return SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens)


def test_complete_normalizes_text_blocks_and_usage():
    response = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="hello"), SimpleNamespace(type="tool_use", text="ignored")],
        usage=make_usage(),
    )
    messages = FakeMessages(completion_response=response)
    config = AnthropicMessagesProviderConfig(api_key_field="anthropic_api_key", model_name="claude-test")

    result = asyncio.run(complete(
        api_keys=None,
        client=FakeClient(messages),
        config=config,
        system="system",
        messages=[{"role": "user", "content": "hi"}],
        model_override="claude-override",
    ))

    assert result.content == "hello"
    assert result.usage.prompt_tokens == 2
    assert result.usage.completion_tokens == 3
    assert messages.create_kwargs["model"] == "claude-override"


def test_stream_yields_text_and_final_usage():
    final_message = SimpleNamespace(usage=make_usage(input_tokens=7, output_tokens=8))
    stream_context = FakeStreamContext(["he", "llo"], final_message=final_message)
    messages = FakeMessages(stream_context=stream_context)
    config = AnthropicMessagesProviderConfig(api_key_field="anthropic_api_key", model_name="claude-test")

    async def collect_chunks():
        return [
            chunk
            async for chunk in stream(
                api_keys=None,
                client=FakeClient(messages),
                config=config,
                system="system",
                messages=[{"role": "user", "content": "hi"}],
            )
        ]

    chunks = asyncio.run(collect_chunks())

    assert [chunk.text_delta for chunk in chunks] == ["he", "llo", ""]
    assert chunks[-1].usage.prompt_tokens == 7
    assert messages.stream_kwargs["system"] == "system"


def test_anthropic_errors_are_wrapped_and_empty_content_is_rejected():
    config = AnthropicMessagesProviderConfig(api_key_field="anthropic_api_key", model_name="claude-test")
    with pytest.raises(LLMProviderCallError):
        asyncio.run(complete(
            api_keys=None,
            client=FakeClient(FakeMessages(error=RuntimeError("boom"))),
            config=config,
            system="system",
            messages=[],
        ))

    with pytest.raises(LLMEmptyResponseError):
        asyncio.run(complete(
            api_keys=None,
            client=FakeClient(FakeMessages(completion_response=SimpleNamespace(content=[], usage=None))),
            config=config,
            system="system",
            messages=[],
        ))
