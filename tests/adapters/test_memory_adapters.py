import os
import time

import pytest

from orchestrator.io.memory.mongodb_memory_client import MongoDBMemoryClient
from orchestrator.memory.memory_processor import MemoryProcessor
from orchestrator.memory.sensenova_omni_memory_client import SenseNovaOmniMemoryClient
from orchestrator.memory.task_manager import TaskManager
from orchestrator.memory.xai_memory_client import XAIMemoryClient
from orchestrator.utils.log import logging

# Test character ID
TEST_CHARACTER_ID = "88801af2-6d2e-48f0-a413-c0058a448a26"
# MongoDB
MONGODB_HOST = "mongodb"
MONGODB_PORT = 27017
MONGODB_DB = "memory_test"
MONGODB_AUTH_DATABASE = "memory_test"
MONGODB_USER = "orchestrator"
MONGODB_PASSWORD = "orchestrator_password"


@pytest.fixture(scope="session")
def mongodb_memory_client() -> MongoDBMemoryClient:
    """Create a MongoDBMemoryClient instance for testing.

    Returns:
        MongoDBMemoryClient:
            Configured MongoDB memory client instance for test database.
    """
    return MongoDBMemoryClient(
        host=MONGODB_HOST,
        port=MONGODB_PORT,
        username=MONGODB_USER,
        password=MONGODB_PASSWORD,
        database=MONGODB_DB,
        auth_database=MONGODB_AUTH_DATABASE,
        logger_cfg={"console_level": logging.DEBUG},
    )


@pytest.fixture
def xai_memory_client(mongodb_memory_client: MongoDBMemoryClient):
    """Create a XAIMemoryClient instance for testing.

    Returns:
        XAIMemoryClient:
            Configured XAI memory client instance for test database.
    """
    return XAIMemoryClient(
        name="test_xai_memory",
        db_client=mongodb_memory_client,
        proxy_url=os.environ.get("PROXY_URL", None),
    )


@pytest.fixture
def sensenova_omni_memory_client(mongodb_memory_client: MongoDBMemoryClient):
    """Create a SenseNovaOmniMemoryClient instance for testing.

    Returns:
        SenseNovaOmniMemoryClient:
            Configured SenseNova Omni memory client instance for test database.
    """
    return SenseNovaOmniMemoryClient(
        name="test_sensenova_omni_memory",
        db_client=mongodb_memory_client,
    )


@pytest.mark.asyncio
async def test_xai_memory_client_call_llm(xai_memory_client: XAIMemoryClient):
    """Test XAI memory client call_llm method.

    This test verifies that the XAI memory client can successfully call the LLM
    to merge short-term and medium-term memories. The test creates a MemoryProcessor
    instance with test conversation data and validates that the memory merging
    operation completes within 30 seconds and returns a non-empty string result.

    Args:
        xai_memory_client (XAIMemoryClient):
            XAI memory client fixture for testing.
    """
    xai_api_key = os.environ.get("XAI_API_KEY")
    if not xai_api_key:
        pytest.skip("xai_api_key is not set, skipping test test_xai_memory_client_call_llm")

    logger_cfg = dict(
        logger_name="test_xai_memory_call_llm",
        console_level=logging.DEBUG,
    )

    # test data
    short_term_memories = [
        {"role": "user", "content": "你好，我叫小明", "relationship": "Stranger"},
        {"role": "assistant", "content": "你好小明，很高兴认识你！"},
    ]

    latest_medium_term_memory = {"content": "关系阶段：陌生人，主要话题：用户询问我的姓名和职责。"}

    # create MemoryProcessor instance
    task_manager = TaskManager(logger_cfg=logger_cfg)
    memory_processor = MemoryProcessor(
        db_client=xai_memory_client.db_client,
        task_manager=task_manager,
        memory_adapter=xai_memory_client,
        medium_term_char_threshold=100,
        logger_cfg=logger_cfg,
    )

    # test _merge_short_and_medium_term
    start_time = time.time()
    result = await memory_processor._merge_short_and_medium_term(
        short_term_memories=short_term_memories,
        latest_medium_term_memory=latest_medium_term_memory,
        api_keys={
            "xai_api_key": xai_api_key,
        },
    )

    end_time = time.time()
    duration = end_time - start_time

    # validate result
    assert result is not None
    assert isinstance(result, str)
    assert len(result) > 0
    assert duration < 30  # should complete within 30 seconds

    print(f"XAI Memory Client call_llm test completed in {duration:.2f} seconds")
    print(f"Result: {result}")


@pytest.mark.asyncio
async def test_sensenova_omni_memory_client_call_llm(sensenova_omni_memory_client: SenseNovaOmniMemoryClient):
    """Test SenseNova Omni memory client call_llm method.

    This test verifies that the SenseNova Omni memory client can successfully call
    the LLM to merge short-term and medium-term memories. The test creates a
    MemoryProcessor instance with test conversation data and validates that the
    memory merging operation completes within 30 seconds and returns a non-empty
    string result.

    Args:
        sensenova_omni_memory_client (SenseNovaOmniMemoryClient):
            SenseNova Omni memory client fixture for testing.
    """
    sensenova_ak = os.environ.get("SENSENOVA_AK")
    sensenova_sk = os.environ.get("SENSENOVA_SK")
    if not sensenova_ak or not sensenova_sk:
        pytest.skip("sensenova_ak or sensenova_sk is not set, skipping test test_sensenova_omni_memory_client_call_llm")

    logger_cfg = dict(
        logger_name="test_sensenova_omni_memory_call_llm",
        console_level=logging.DEBUG,
    )

    # test data
    short_term_memories = [
        {"role": "user", "content": "你好，我叫小红", "relationship": "Stranger"},
        {"role": "assistant", "content": "你好小红，很高兴认识你！"},
    ]

    latest_medium_term_memory = {"content": "关系阶段：陌生人，主要话题：用户询问我的姓名和职责。"}

    # create MemoryProcessor instance
    task_manager = TaskManager(logger_cfg=logger_cfg)
    memory_processor = MemoryProcessor(
        db_client=sensenova_omni_memory_client.db_client,
        task_manager=task_manager,
        memory_adapter=sensenova_omni_memory_client,
        medium_term_char_threshold=100,
        logger_cfg=logger_cfg,
    )

    # test _merge_short_and_medium_term
    start_time = time.time()
    result = await memory_processor._merge_short_and_medium_term(
        short_term_memories=short_term_memories,
        latest_medium_term_memory=latest_medium_term_memory,
        api_keys={
            "sensenova_ak": sensenova_ak,
            "sensenova_sk": sensenova_sk,
        },
    )

    end_time = time.time()
    duration = end_time - start_time

    # validate result
    assert result is not None
    assert isinstance(result, str)
    assert len(result) > 0
    assert duration < 30  # should complete within 30 seconds

    print(f"Omni Memory Client call_llm test completed in {duration:.2f} seconds")
    print(f"Result: {result}")
