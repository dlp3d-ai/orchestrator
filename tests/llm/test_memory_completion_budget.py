import pytest

from orchestrator.memory.memory_adapter import BaseMemoryAdapter
from orchestrator.memory.memory_processor import MemoryProcessor
from orchestrator.memory.task_manager import TaskManager


class FakeMemoryAdapter(BaseMemoryAdapter):
    """Memory adapter test double that records LLM call arguments."""

    def __init__(self):
        super().__init__(name="fake_memory", db_client=object())
        self.calls = []

    async def call_llm(
        self,
        system_prompt,
        user_input,
        max_tokens,
        response_format=None,
        tag_prompt=None,
        api_keys=None,
        model_override=None,
    ):
        self.calls.append(
            {
                "max_tokens": max_tokens,
                "response_format": response_format,
                "tag_prompt": tag_prompt,
                "api_keys": api_keys,
                "model_override": model_override,
            }
        )
        return "ok"


@pytest.mark.asyncio
async def test_memory_processor_uses_completion_token_budget_separate_from_summary_length():
    adapter = FakeMemoryAdapter()
    processor = MemoryProcessor(
        db_client=object(),
        task_manager=TaskManager(),
        memory_adapter=adapter,
        medium_term_char_threshold=100,
        long_term_char_threshold=100,
        profile_memory_char_threshold=500,
    )

    await processor._merge_short_and_medium_term(
        short_term_memories=[{"role": "user", "content": "hello", "relationship": "Stranger"}],
        latest_medium_term_memory={"content": "old"},
    )
    await processor._merge_long_term_and_medium_term(
        long_term_content="long",
        medium_term_content="medium",
    )
    await processor._update_profile_memory(
        profile_memory={"content": "profile"},
        new_short_term_memories=[{"role": "assistant", "content": "reply"}],
    )

    assert [call["max_tokens"] for call in adapter.calls] == [512, 512, 2000]
