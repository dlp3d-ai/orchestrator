import os
import time

import pytest

from orchestrator.io.memory.mongodb_memory_client import MongoDBMemoryClient
from orchestrator.memory.deepseek_memory_client import DeepSeekMemoryClient
from orchestrator.memory.gemini_memory_client import GeminiMemoryClient
from orchestrator.memory.memory_processor import MemoryProcessor
from orchestrator.memory.openai_memory_client import OpenAIMemoryClient
from orchestrator.memory.sensechat_memory_client import SenseChatMemoryClient
from orchestrator.memory.sensenova_memory_client import SenseNovaMemoryClient
from orchestrator.memory.sensenova_omni_memory_client import SenseNovaOmniMemoryClient
from orchestrator.memory.task_manager import TaskManager
from orchestrator.memory.xai_memory_client import XAIMemoryClient
from orchestrator.utils.log import logging

# Test character ID
TEST_CHARACTER_ID = "88801af2-6d2e-48f0-a413-c0058a448a26"
# MongoDB connection configuration
MONGODB_HOST = os.environ.get("MONGODB_HOST")
MONGODB_PORT = int(os.environ.get("MONGODB_PORT", 27017))
MONGODB_MEMORY_DB = os.environ.get("MONGODB_MEMORY_DB")
MONGODB_AUTH_DATABASE = MONGODB_MEMORY_DB
MONGODB_MEMORY_USER = os.environ.get("MONGODB_MEMORY_USER")
MONGODB_MEMORY_PASSWORD = os.environ.get("MONGODB_MEMORY_PASSWORD")


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
        username=MONGODB_MEMORY_USER,
        password=MONGODB_MEMORY_PASSWORD,
        database=MONGODB_MEMORY_DB,
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
def openai_memory_client(mongodb_memory_client: MongoDBMemoryClient):
    """Create a OpenAIMemoryClient instance for testing.

    Returns:
        OpenAIMemoryClient:
            Configured OpenAI memory client instance for test database.
    """
    return OpenAIMemoryClient(
        name="test_openai_memory",
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
        name="test_sensenovaomni_memory",
        db_client=mongodb_memory_client,
    )


@pytest.fixture
def sensenova_memory_client(mongodb_memory_client: MongoDBMemoryClient):
    """Create a SenseNovaMemoryClient instance for testing.

    Returns:
        SenseNovaMemoryClient:
            Configured SenseNova memory client instance for test database.
    """
    return SenseNovaMemoryClient(
        name="test_sensenova_memory",
        db_client=mongodb_memory_client,
        proxy_url=os.environ.get("PROXY_URL", None),
    )


@pytest.fixture
def sensechat_memory_client(mongodb_memory_client: MongoDBMemoryClient):
    """Create a SenseChatMemoryClient instance for testing.

    Returns:
        SenseChatMemoryClient:
            Configured SenseChat memory client instance for test database.
    """
    return SenseChatMemoryClient(
        name="test_sensechat_memory",
        db_client=mongodb_memory_client,
        proxy_url=os.environ.get("PROXY_URL", None),
    )


@pytest.fixture
def gemini_memory_client(mongodb_memory_client: MongoDBMemoryClient):
    """Create a GeminiMemoryClient instance for testing.

    Returns:
        GeminiMemoryClient:
            Configured Gemini memory client instance for test database.
    """
    return GeminiMemoryClient(
        name="test_gemini_memory",
        db_client=mongodb_memory_client,
        proxy_url=os.environ.get("PROXY_URL", None),
    )


@pytest.fixture
def deepseek_memory_client(mongodb_memory_client: MongoDBMemoryClient):
    """Create a DeepSeekMemoryClient instance for testing.

    Returns:
        DeepSeekMemoryClient:
            Configured DeepSeek memory client instance for test database.
    """
    return DeepSeekMemoryClient(
        name="test_deepseek_memory",
        db_client=mongodb_memory_client,
        proxy_url=os.environ.get("PROXY_URL", None),
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
    if not MONGODB_HOST:
        pytest.skip("MONGODB_HOST is not set, skipping test_xai_memory_client_call_llm")

    logger_cfg = dict(logger_name="test_xai_memory_call_llm", file_level=logging.DEBUG, logger_path="logs/pytest.log")

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
    sensenovaomni_ak = os.environ.get("SENSENOVAOMNI_AK")
    sensenovaomni_sk = os.environ.get("SENSENOVAOMNI_SK")
    if not sensenovaomni_ak or not sensenovaomni_sk:
        pytest.skip(
            "sensenovaomni_ak or sensenovaomni_sk is not set, skipping test test_sensenova_omni_memory_client_call_llm"
        )
    if not MONGODB_HOST:
        pytest.skip("MONGODB_HOST is not set, skipping test_sensenova_omni_memory_client_call_llm")

    logger_cfg = dict(
        logger_name="test_sensenova_omni_memory_call_llm", file_level=logging.DEBUG, logger_path="logs/pytest.log"
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
            "sensenovaomni_ak": sensenovaomni_ak,
            "sensenovaomni_sk": sensenovaomni_sk,
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


@pytest.mark.asyncio
async def test_openai_memory_client_call_llm(openai_memory_client: OpenAIMemoryClient):
    """Test OpenAI memory client call_llm method.

    This test verifies that the OpenAI memory client can successfully call the LLM
    to merge short-term and medium-term memories. The test creates a MemoryProcessor
    instance with test conversation data and validates that the memory merging
    operation completes within 30 seconds and returns a non-empty string result.

    Args:
        openai_memory_client (OpenAIMemoryClient):
            OpenAI memory client fixture for testing.
    """
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        pytest.skip("openai_api_key is not set, skipping test test_openai_memory_client_call_llm")
    if not MONGODB_HOST:
        pytest.skip("MONGODB_HOST is not set, skipping test_openai_memory_client_call_llm")

    logger_cfg = dict(
        logger_name="test_openai_memory_call_llm", file_level=logging.DEBUG, logger_path="logs/pytest.log"
    )

    # test data
    short_term_memories = [
        {"role": "user", "content": "你好，我叫小张", "relationship": "Stranger"},
        {"role": "assistant", "content": "你好小张，很高兴认识你！"},
    ]

    latest_medium_term_memory = {"content": "关系阶段：陌生人，主要话题：用户询问我的姓名和职责。"}

    # create MemoryProcessor instance
    task_manager = TaskManager(logger_cfg=logger_cfg)
    memory_processor = MemoryProcessor(
        db_client=openai_memory_client.db_client,
        task_manager=task_manager,
        memory_adapter=openai_memory_client,
        medium_term_char_threshold=100,
        logger_cfg=logger_cfg,
    )

    # test _merge_short_and_medium_term
    start_time = time.time()
    result = await memory_processor._merge_short_and_medium_term(
        short_term_memories=short_term_memories,
        latest_medium_term_memory=latest_medium_term_memory,
        api_keys={
            "openai_api_key": openai_api_key,
        },
    )

    end_time = time.time()
    duration = end_time - start_time

    # validate result
    assert result is not None
    assert isinstance(result, str)
    assert len(result) > 0
    assert duration < 30  # should complete within 30 seconds

    print(f"OpenAI Memory Client call_llm test completed in {duration:.2f} seconds")
    print(f"Result: {result}")


@pytest.mark.asyncio
async def test_sensenova_memory_client_call_llm(sensenova_memory_client: SenseNovaMemoryClient):
    """Test SenseNova memory client call_llm method.

    This test verifies that the SenseNova memory client can successfully call the LLM
    to merge short-term and medium-term memories. The test creates a MemoryProcessor
    instance with test conversation data and validates that the memory merging
    operation completes within 30 seconds and returns a non-empty string result.

    Args:
        sensenova_memory_client (SenseNovaMemoryClient):
            SenseNova memory client fixture for testing.
    """
    sensenova_ak = os.environ.get("SENSENOVA_AK")
    sensenova_sk = os.environ.get("SENSENOVA_SK")
    if not sensenova_ak or not sensenova_sk:
        pytest.skip("sensenova_ak or sensenova_sk is not set, skipping test test_sensenova_memory_client_call_llm")
    if not MONGODB_HOST:
        pytest.skip("MONGODB_HOST is not set, skipping test_sensenova_memory_client_call_llm")

    logger_cfg = dict(
        logger_name="test_sensenova_memory_call_llm", file_level=logging.DEBUG, logger_path="logs/pytest.log"
    )

    short_term_memories = [
        {"role": "user", "content": "你好，我叫小刘", "relationship": "Stranger"},
        {"role": "assistant", "content": "你好小刘，很高兴认识你！"},
    ]

    latest_medium_term_memory = {"content": "关系阶段：陌生人，主要话题：用户询问我的姓名和职责。"}

    task_manager = TaskManager(logger_cfg=logger_cfg)
    memory_processor = MemoryProcessor(
        db_client=sensenova_memory_client.db_client,
        task_manager=task_manager,
        memory_adapter=sensenova_memory_client,
        medium_term_char_threshold=100,
        logger_cfg=logger_cfg,
    )

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

    assert result is not None
    assert isinstance(result, str)
    assert len(result) > 0
    assert duration < 30

    print(f"SenseNova Memory Client call_llm test completed in {duration:.2f} seconds")
    print(f"Result: {result}")


@pytest.mark.asyncio
async def test_sensechat_memory_client_call_llm(sensechat_memory_client: SenseChatMemoryClient):
    """Test SenseChat memory client call_llm method.

    This test verifies that the SenseChat memory client can successfully call the LLM
    to merge short-term and medium-term memories. The test creates a MemoryProcessor
    instance with test conversation data and validates that the memory merging
    operation completes within 30 seconds and returns a non-empty string result.

    Args:
        sensechat_memory_client (SenseChatMemoryClient):
            SenseChat memory client fixture for testing.
    """
    sensechat_ak = os.environ.get("SENSECHAT_AK")
    sensechat_sk = os.environ.get("SENSECHAT_SK")
    if not sensechat_ak or not sensechat_sk:
        pytest.skip("sensechat_ak or sensechat_sk is not set, skipping test test_sensechat_memory_client_call_llm")
    if not MONGODB_HOST:
        pytest.skip("MONGODB_HOST is not set, skipping test_sensechat_memory_client_call_llm")

    logger_cfg = dict(
        logger_name="test_sensechat_memory_call_llm", file_level=logging.DEBUG, logger_path="logs/pytest.log"
    )

    short_term_memories = [
        {"role": "user", "content": "你好，我叫小王", "relationship": "Stranger"},
        {"role": "assistant", "content": "你好小王，很高兴认识你！"},
    ]

    latest_medium_term_memory = {"content": "关系阶段：陌生人，主要话题：用户询问我的姓名和职责。"}

    task_manager = TaskManager(logger_cfg=logger_cfg)
    memory_processor = MemoryProcessor(
        db_client=sensechat_memory_client.db_client,
        task_manager=task_manager,
        memory_adapter=sensechat_memory_client,
        medium_term_char_threshold=100,
        logger_cfg=logger_cfg,
    )

    start_time = time.time()
    result = await memory_processor._merge_short_and_medium_term(
        short_term_memories=short_term_memories,
        latest_medium_term_memory=latest_medium_term_memory,
        api_keys={
            "sensechat_ak": sensechat_ak,
            "sensechat_sk": sensechat_sk,
        },
    )

    end_time = time.time()
    duration = end_time - start_time

    assert result is not None
    assert isinstance(result, str)
    assert len(result) > 0
    assert duration < 30

    print(f"SenseChat Memory Client call_llm test completed in {duration:.2f} seconds")
    print(f"Result: {result}")


@pytest.mark.asyncio
async def test_gemini_memory_client_call_llm(gemini_memory_client: GeminiMemoryClient):
    """Test Gemini memory client call_llm method.

    This test verifies that the Gemini memory client can successfully call the LLM
    to merge short-term and medium-term memories. The test creates a MemoryProcessor
    instance with test conversation data and validates that the memory merging
    operation completes within 30 seconds and returns a non-empty string result.

    Args:
        gemini_memory_client (GeminiMemoryClient):
            Gemini memory client fixture for testing.
    """
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_api_key:
        pytest.skip("gemini_api_key is not set, skipping test test_gemini_memory_client_call_llm")
    if not MONGODB_HOST:
        pytest.skip("MONGODB_HOST is not set, skipping test_gemini_memory_client_call_llm")

    logger_cfg = dict(
        logger_name="test_gemini_memory_call_llm", file_level=logging.DEBUG, logger_path="logs/pytest.log"
    )

    short_term_memories = [
        {"role": "user", "content": "你好，我叫小李", "relationship": "Stranger"},
        {"role": "assistant", "content": "你好小李，很高兴认识你！"},
    ]

    latest_medium_term_memory = {"content": "关系阶段：陌生人，主要话题：用户询问我的姓名和职责。"}

    task_manager = TaskManager(logger_cfg=logger_cfg)
    memory_processor = MemoryProcessor(
        db_client=gemini_memory_client.db_client,
        task_manager=task_manager,
        memory_adapter=gemini_memory_client,
        medium_term_char_threshold=100,
        logger_cfg=logger_cfg,
    )

    start_time = time.time()
    result = await memory_processor._merge_short_and_medium_term(
        short_term_memories=short_term_memories,
        latest_medium_term_memory=latest_medium_term_memory,
        api_keys={
            "gemini_api_key": gemini_api_key,
        },
    )

    end_time = time.time()
    duration = end_time - start_time

    assert result is not None
    assert isinstance(result, str)
    assert len(result) > 0
    assert duration < 30

    print(f"Gemini Memory Client call_llm test completed in {duration:.2f} seconds")
    print(f"Result: {result}")


@pytest.mark.asyncio
async def test_deepseek_memory_client_call_llm(deepseek_memory_client: DeepSeekMemoryClient):
    """Test DeepSeek memory client call_llm method.

    This test verifies that the DeepSeek memory client can successfully call the LLM
    to merge short-term and medium-term memories. The test creates a MemoryProcessor
    instance with test conversation data and validates that the memory merging
    operation completes within 30 seconds and returns a non-empty string result.

    Args:
        deepseek_memory_client (DeepSeekMemoryClient):
            DeepSeek memory client fixture for testing.
    """
    deepseek_api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not deepseek_api_key:
        pytest.skip("deepseek_api_key is not set, skipping test test_deepseek_memory_client_call_llm")
    if not MONGODB_HOST:
        pytest.skip("MONGODB_HOST is not set, skipping test_deepseek_memory_client_call_llm")

    logger_cfg = dict(
        logger_name="test_deepseek_memory_call_llm", file_level=logging.DEBUG, logger_path="logs/pytest.log"
    )

    short_term_memories = [
        {"role": "user", "content": "你好，我叫小陈", "relationship": "Stranger"},
        {"role": "assistant", "content": "你好小陈，很高兴认识你！"},
    ]

    latest_medium_term_memory = {"content": "关系阶段：陌生人，主要话题：用户询问我的姓名和职责。"}

    task_manager = TaskManager(logger_cfg=logger_cfg)
    memory_processor = MemoryProcessor(
        db_client=deepseek_memory_client.db_client,
        task_manager=task_manager,
        memory_adapter=deepseek_memory_client,
        medium_term_char_threshold=100,
        logger_cfg=logger_cfg,
    )

    start_time = time.time()
    result = await memory_processor._merge_short_and_medium_term(
        short_term_memories=short_term_memories,
        latest_medium_term_memory=latest_medium_term_memory,
        api_keys={
            "deepseek_api_key": deepseek_api_key,
        },
    )

    end_time = time.time()
    duration = end_time - start_time

    assert result is not None
    assert isinstance(result, str)
    assert len(result) > 0
    assert duration < 30

    print(f"DeepSeek Memory Client call_llm test completed in {duration:.2f} seconds")
    print(f"Result: {result}")
