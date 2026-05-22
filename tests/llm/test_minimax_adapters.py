import asyncio
from types import SimpleNamespace

import pytest

from orchestrator.classification.minimax_classification_client import MiniMaxClassificationClient
from orchestrator.conversation.minimax_conversation_client import MiniMaxConversationClient
from orchestrator.data_structures.classification import ClassificationType
from orchestrator.llm.errors import MissingAPIKeyException
from orchestrator.llm.minimax import MINIMAX_API_KEY_FIELD, MINIMAX_DEFAULT_BASE_URL, MINIMAX_DEFAULT_MODEL
from orchestrator.llm.openai_chat import create_client
from orchestrator.llm.types import LLMCompletionResult, LLMStreamChunk, LLMUsage
from orchestrator.memory.minimax_memory_client import MiniMaxMemoryClient
from orchestrator.reaction.minimax_reaction_client import MiniMaxReactionClient


def _assert_minimax_provider_config(adapter):
    assert adapter.minimax_model_name == MINIMAX_DEFAULT_MODEL
    assert adapter.minimax_url == MINIMAX_DEFAULT_BASE_URL
    assert adapter.llm_provider_config.provider_name == "MiniMax"
    assert adapter.llm_provider_config.api_key_field == MINIMAX_API_KEY_FIELD
    assert adapter.llm_provider_config.model_name == MINIMAX_DEFAULT_MODEL
    assert adapter.llm_provider_config.base_url == MINIMAX_DEFAULT_BASE_URL


def test_minimax_adapters_default_to_openai_compatible_config():
    _assert_minimax_provider_config(MiniMaxClassificationClient(name="classification", motion_keywords=[]))
    _assert_minimax_provider_config(MiniMaxReactionClient(name="reaction", motion_keywords=[]))
    _assert_minimax_provider_config(MiniMaxMemoryClient(name="memory", db_client=object()))
    _assert_minimax_provider_config(
        MiniMaxConversationClient(name="conversation", agent_prompts_file="configs/agent_prompts.yaml")
    )


def test_minimax_config_requires_minimax_api_key():
    adapter = MiniMaxClassificationClient(name="classification", motion_keywords=[])

    with pytest.raises(MissingAPIKeyException):
        create_client({}, adapter.llm_provider_config)

    memory_adapter = MiniMaxMemoryClient(name="memory", db_client=object())
    with pytest.raises(MissingAPIKeyException):
        asyncio.run(
            memory_adapter.call_llm(
                system_prompt="memory",
                user_input="conversation",
                max_tokens=64,
                api_keys={},
            )
        )


def test_minimax_classification_uses_common_openai_chat_layer(monkeypatch):
    calls = []

    async def fake_complete(**kwargs):
        calls.append(kwargs)
        return LLMCompletionResult(content="reject", usage=LLMUsage())

    monkeypatch.setattr(
        "orchestrator.classification.minimax_classification_client.complete",
        fake_complete,
    )

    adapter = MiniMaxClassificationClient(name="classification", motion_keywords=[])
    adapter.input_buffer["request-1"] = {
        "llm_client": object(),
        "classification_model_override": "",
        "api_keys": {MINIMAX_API_KEY_FIELD: "test-key"},
    }

    result = asyncio.run(adapter.classify("request-1", "classify", "hello"))

    assert result is ClassificationType.REJECT
    assert len(calls) == 1
    assert calls[0]["config"] is adapter.llm_provider_config
    assert calls[0]["model_override"] == MINIMAX_DEFAULT_MODEL
    assert "extra_body" not in calls[0]


def test_minimax_reaction_uses_common_openai_chat_layer(monkeypatch):
    calls = []
    content = """
    <happiness_delta>1</happiness_delta>
    <sadness_delta>-1</sadness_delta>
    <relationship_delta>2</relationship_delta>
    <speech_keywords>hello</speech_keywords>
    <motion_keywords>wave</motion_keywords>
    """

    async def fake_complete(**kwargs):
        calls.append(kwargs)
        return LLMCompletionResult(content=content, usage=LLMUsage())

    monkeypatch.setattr(
        "orchestrator.reaction.minimax_reaction_client.complete",
        fake_complete,
    )

    adapter = MiniMaxReactionClient(name="reaction", motion_keywords=[])
    adapter.input_buffer["request-1"] = {
        "llm_client": object(),
        "reaction_model_override": "",
        "api_keys": {MINIMAX_API_KEY_FIELD: "test-key"},
    }

    result = asyncio.run(
        adapter.get_reaction_delta(
            "request-1",
            prompt="reaction",
            text="agent text",
            tag="",
            user_input="hello",
        )
    )

    assert result.emotion_delta.happiness_delta == 1
    assert result.relationship_delta == 2
    assert result.motion[0].speech_keywords == "hello"
    assert result.motion[0].motion_keywords == "wave"
    assert len(calls) == 1
    assert calls[0]["config"] is adapter.llm_provider_config
    assert "extra_body" not in calls[0]


def test_minimax_memory_uses_common_openai_chat_layer_and_preserves_output_extraction(monkeypatch):
    calls = []

    async def fake_complete(**kwargs):
        calls.append(kwargs)
        return LLMCompletionResult(
            content="<output>condensed memory</output>",
            usage=LLMUsage(prompt_tokens=3, completion_tokens=4, total_tokens=7),
        )

    monkeypatch.setattr(
        "orchestrator.memory.minimax_memory_client.complete",
        fake_complete,
    )

    adapter = MiniMaxMemoryClient(name="memory", db_client=object())
    result = asyncio.run(
        adapter.call_llm(
            system_prompt="memory",
            user_input="conversation",
            max_tokens=64,
            api_keys={MINIMAX_API_KEY_FIELD: "test-key"},
        )
    )

    assert result == "condensed memory"
    assert len(calls) == 1
    assert calls[0]["config"] is adapter.llm_provider_config
    assert calls[0]["max_tokens"] == 64
    assert "extra_body" not in calls[0]


def test_minimax_conversation_stream_uses_common_openai_chat_layer(monkeypatch):
    calls = []

    async def fake_stream(**kwargs):
        calls.append(kwargs)
        yield LLMStreamChunk(text_delta="hello")
        yield LLMStreamChunk(usage=LLMUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2))

    monkeypatch.setattr(
        "orchestrator.conversation.minimax_conversation_client.stream",
        fake_stream,
    )

    fed_bodies = []

    class FakePayload:
        async def feed_stream(self, body):
            fed_bodies.append(body)

    fake_node = SimpleNamespace(name="downstream", payload=FakePayload())
    fake_dag_node = SimpleNamespace(downstreams=[fake_node])
    fake_dag = SimpleNamespace(get_node=lambda _: fake_dag_node)

    adapter = MiniMaxConversationClient(name="conversation", agent_prompts_file="configs/agent_prompts.yaml")
    adapter.input_buffer["request-1"] = {
        "chat_task": {
            "dag": fake_dag,
            "dag_start_time": None,
            "style_list": [],
            "node_name": "conversation",
            "conversation_model_override": "",
            "user_prompt": "user prompt",
            "llm_client": object(),
            "user_id": "user-1",
        }
    }

    result = asyncio.run(adapter._llm_stream_chat("message", "context", [], "zh", "request-1"))

    assert result == "hello"
    assert fed_bodies[0].text_segment == "hello"
    assert len(calls) == 1
    assert calls[0]["config"] is adapter.llm_provider_config
    assert calls[0]["stream_options"] == {"include_usage": True}
    assert "extra_body" not in calls[0]
