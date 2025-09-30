import asyncio
import logging
import os
import time
import uuid

import pytest
import pytest_asyncio
import yaml

from orchestrator.conversation.builder import build_conversation_adapter
from orchestrator.data_structures.classification import (
    ClassificationChunkBody,
    ClassificationChunkEnd,
    ClassificationChunkStart,
    ClassificationType,
)
from orchestrator.data_structures.process_flow import DAGNode, DAGStatus, DirectedAcyclicGraph
from orchestrator.data_structures.text_chunk import TextChunkBody, TextChunkEnd, TextChunkStart
from orchestrator.io.memory.mongodb_memory_client import MongoDBMemoryClient
from orchestrator.memory.memory_adapter import INITIAL_EMOTION_STATE
from orchestrator.memory.sensenova_omni_memory_client import SenseNovaOmniMemoryClient
from orchestrator.profile.text_stream_profile import TextStreamProfile

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


@pytest_asyncio.fixture(scope="session", autouse=True)
async def setup_test_memory(mongodb_memory_client: MongoDBMemoryClient):
    """Set up and clean up test memory before and after all tests.

    Args:
        mongodb_memory_client (MongoDBMemoryClient):
            MongoDB memory client instance for database operations.
    """
    # Create DynamoDB client
    db_client = mongodb_memory_client

    # Clean up all existing memories
    try:
        await db_client.remove_long_term_memory(TEST_CHARACTER_ID)
    except Exception:
        pass  # Ignore if not exists

    try:
        await db_client.remove_medium_term_memory_not_after(TEST_CHARACTER_ID)
    except Exception:
        pass  # Ignore if not exists

    try:
        await db_client.remove_chat_history_memory(TEST_CHARACTER_ID)
    except Exception:
        pass  # Ignore if not exists

    # Create initial chat history memory
    current_time = time.time()

    # Create some initial conversation records
    chat_timestamps = [
        current_time - 3600,  # 1 hour ago
        current_time - 3599,
        current_time - 1800,  # 30 minutes ago
        current_time - 1799,
        current_time - 900,  # 15 minutes ago
        current_time - 899,
    ]

    # Add initial conversation history
    for i, timestamp in enumerate(chat_timestamps):
        if i % 2 == 0:  # User message
            await db_client.append_chat_history(
                character_id=TEST_CHARACTER_ID,
                unix_timestamp=timestamp,
                role="user",
                content=f"测试用户消息 {i+1}",
                relationship="Friend",
            )
        else:  # Assistant message
            await db_client.append_chat_history(
                character_id=TEST_CHARACTER_ID,
                unix_timestamp=timestamp,
                role="assistant",
                content=f"测试助手回复 {i+1}",
                happiness=15,
                sadness=14,
                fear=14,
                anger=14,
                disgust=14,
                surprise=15,
                shyness=14,
            )

    # Create medium-term memory
    medium_term_timestamp = db_client.convert_unix_timestamp_to_str(chat_timestamps[0] - 1)
    last_short_term_timestamp = db_client.convert_unix_timestamp_to_str(chat_timestamps[-1])

    await db_client.append_medium_term_memory(character_id=TEST_CHARACTER_ID, start_timestamp=medium_term_timestamp)
    await db_client.update_medium_term_memory(
        character_id=TEST_CHARACTER_ID,
        start_timestamp=medium_term_timestamp,
        content="测试中期记忆：用户和助手进行了一些基础对话，建立了友好的关系。",
        last_short_term_timestamp=last_short_term_timestamp,
    )

    # Create long-term memory
    last_medium_term_timestamp = medium_term_timestamp
    await db_client.set_long_term_memory(
        character_id=TEST_CHARACTER_ID,
        content="测试长期记忆：这是一个测试角色，用于验证对话功能。用户和助手之间建立了稳定的友谊关系。",
        last_short_term_timestamp=last_short_term_timestamp,
        last_medium_term_timestamp=last_medium_term_timestamp,
    )

    yield  # Let tests run

    # Clean up after tests (optional, as next test will reinitialize)
    try:
        await db_client.remove_long_term_memory(TEST_CHARACTER_ID)
        await db_client.remove_medium_term_memory_not_after(TEST_CHARACTER_ID)
        await db_client.remove_chat_history_memory(TEST_CHARACTER_ID)
    except Exception:
        pass


@pytest.fixture
def test_memory_adapter(mongodb_memory_client: MongoDBMemoryClient):
    """Create a test memory adapter for conversation testing.

    Args:
        mongodb_memory_client (MongoDBMemoryClient):
            MongoDB memory client instance for database operations.

    Returns:
        SenseNovaOmniMemoryClient:
            Configured SenseNova Omni memory client for testing.
    """
    return SenseNovaOmniMemoryClient(
        name="test_sensenova_omni_memory",
        db_client=mongodb_memory_client,
    )


agent_prompts_file_path = "configs/agent_prompts.yaml"
with open(agent_prompts_file_path, "r", encoding="utf-8") as file:
    agent_prompts = yaml.safe_load(file)


@pytest.mark.asyncio
async def test_openai_stream(
    test_memory_adapter: SenseNovaOmniMemoryClient,
    mongodb_memory_client: MongoDBMemoryClient,
):
    """Test OpenAI conversation streaming functionality.

    Args:
        test_memory_adapter (SenseNovaOmniMemoryClient):
            SenseNova Omni memory client for conversation testing.
        mongodb_memory_client (MongoDBMemoryClient):
            MongoDB memory client for database operations.
    """
    sensenova_ak = os.environ.get("SENSENOVA_AK")
    sensenova_sk = os.environ.get("SENSENOVA_SK")
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not sensenova_ak or not sensenova_sk or not openai_api_key:
        pytest.skip("SENSENOVA_AK or SENSENOVA_SK or OPENAI_API_KEY is not set, skipping test_openai_stream")

    logger_cfg = dict(logger_name="test_openai_streaming", file_level=logging.DEBUG, logger_path="logs/pytest.log")
    openai_client_cfg = dict(
        type="OpenAIConversationClient",
        name="openai_client",
        agent_prompts_file="configs/agent_prompts.yaml",
        openai_model_name="gpt-4.1-2025-04-14",
        proxy_url=os.environ.get("PROXY_URL", None),
        logger_cfg=logger_cfg,
    )
    cascade_memories = await test_memory_adapter.db_client.get_cascade_memories(
        character_id=TEST_CHARACTER_ID,
    )
    adapter = build_conversation_adapter(openai_client_cfg)
    asyncio.create_task(adapter.run())
    profile = TextStreamProfile(mark_status_on_end=True, logger_cfg=logger_cfg)
    asyncio.create_task(profile.run())
    graph = DirectedAcyclicGraph(
        name="test_openai_streaming",
        conf=dict(
            user_prompt=agent_prompts["keqing_default"],
            language="zh",
            character_id=TEST_CHARACTER_ID,
            user_settings=dict(
                openai_api_key=openai_api_key,
                sensenova_ak=sensenova_ak,
                sensenova_sk=sensenova_sk,
            ),
            profile_memory=None,
            cascade_memories=cascade_memories,
            relationship=("Lover", 100),
            emotion=INITIAL_EMOTION_STATE,
            memory_adapter=test_memory_adapter,
            memory_db_client=mongodb_memory_client,
        ),
        logger_cfg=logger_cfg,
    )
    chat_node = DAGNode(
        name="chat_node",
        payload=adapter,
    )
    profile_node = DAGNode(
        name="profile_node",
        payload=profile,
    )
    graph.add_node(chat_node)
    graph.add_node(profile_node)
    graph.add_edge(chat_node.name, profile_node.name)
    graph.set_status(DAGStatus.RUNNING)
    request_id = str(uuid.uuid4())
    start_chunk = TextChunkStart(request_id=request_id, dag=graph, node_name=chat_node.name)
    await adapter.feed_stream(start_chunk)
    message = "用户已进入对话"
    for char in message:
        body_chunk = TextChunkBody(
            request_id=request_id,
            text_segment=char,
        )
        await adapter.feed_stream(body_chunk)
    end_chunk = TextChunkEnd(
        request_id=request_id,
    )
    await adapter.feed_stream(end_chunk)
    start_time = time.time()
    while graph.status != DAGStatus.COMPLETED:
        await asyncio.sleep(0.1)
        if time.time() - start_time > 10:
            raise TimeoutError("OpenAI stream timeout")
    await adapter.interrupt()
    await profile.interrupt()
    await asyncio.sleep(adapter.sleep_time * 5)


@pytest.mark.asyncio
async def test_gemini_stream(
    test_memory_adapter: SenseNovaOmniMemoryClient,
    mongodb_memory_client: MongoDBMemoryClient,
):
    """Test Gemini conversation streaming functionality.

    Args:
        test_memory_adapter (SenseNovaOmniMemoryClient):
            SenseNova Omni memory client for conversation testing.
        mongodb_memory_client (MongoDBMemoryClient):
            MongoDB memory client for database operations.
    """
    sensenova_ak = os.environ.get("SENSENOVA_AK")
    sensenova_sk = os.environ.get("SENSENOVA_SK")
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    if not sensenova_ak or not sensenova_sk or not gemini_api_key:
        pytest.skip("SENSENOVA_AK or SENSENOVA_SK or GEMINI_API_KEY is not set, skipping test_gemini_stream")

    logger_cfg = dict(logger_name="test_gemini_streaming", file_level=logging.DEBUG, logger_path="logs/pytest.log")
    gemini_client_cfg = dict(
        type="GeminiConversationClient",
        name="gemini_client",
        agent_prompts_file="configs/agent_prompts.yaml",
        gemini_model_name="gemini-2.5-flash-lite",
        proxy_url=os.environ.get("PROXY_URL", None),
        logger_cfg=logger_cfg,
    )
    cascade_memories = await test_memory_adapter.db_client.get_cascade_memories(
        character_id=TEST_CHARACTER_ID,
    )
    profile_memory = await test_memory_adapter.db_client.get_profile_memory(
        character_id=TEST_CHARACTER_ID,
    )
    adapter = build_conversation_adapter(gemini_client_cfg)
    asyncio.create_task(adapter.run())
    profile = TextStreamProfile(mark_status_on_end=True, logger_cfg=logger_cfg)
    asyncio.create_task(profile.run())
    graph = DirectedAcyclicGraph(
        name="test_gemini_streaming",
        conf=dict(
            user_prompt=agent_prompts["keqing_default"],
            character_id=TEST_CHARACTER_ID,
            user_settings=dict(
                gemini_api_key=gemini_api_key,
                sensenova_ak=sensenova_ak,
                sensenova_sk=sensenova_sk,
            ),
            language="zh",
            profile_memory=profile_memory,
            cascade_memories=cascade_memories,
            relationship=("Lover", 100),
            emotion=INITIAL_EMOTION_STATE,
            memory_adapter=test_memory_adapter,
            memory_db_client=mongodb_memory_client,
        ),
        logger_cfg=logger_cfg,
    )
    chat_node = DAGNode(
        name="chat_node",
        payload=adapter,
    )
    profile_node = DAGNode(
        name="profile_node",
        payload=profile,
    )
    graph.add_node(chat_node)
    graph.add_node(profile_node)
    graph.add_edge(chat_node.name, profile_node.name)
    graph.set_status(DAGStatus.RUNNING)
    request_id = str(uuid.uuid4())
    start_chunk = TextChunkStart(request_id=request_id, dag=graph, node_name=chat_node.name)
    await adapter.feed_stream(start_chunk)
    message = "不喜欢你。"
    for char in message:
        body_chunk = TextChunkBody(
            request_id=request_id,
            text_segment=char,
        )
        await adapter.feed_stream(body_chunk)
    end_chunk = TextChunkEnd(
        request_id=request_id,
    )
    await adapter.feed_stream(end_chunk)
    start_time = time.time()
    while graph.status != DAGStatus.COMPLETED:
        await asyncio.sleep(0.1)
        if time.time() - start_time > 10:
            raise TimeoutError("Gemini stream timeout")
    await adapter.interrupt()
    await profile.interrupt()
    await asyncio.sleep(adapter.sleep_time * 5)


@pytest.mark.asyncio
async def test_deepseek_stream(
    test_memory_adapter: SenseNovaOmniMemoryClient,
    mongodb_memory_client: MongoDBMemoryClient,
):
    """Test DeepSeek conversation streaming functionality.

    Args:
        test_memory_adapter (SenseNovaOmniMemoryClient):
            SenseNova Omni memory client for conversation testing.
        mongodb_memory_client (MongoDBMemoryClient):
            MongoDB memory client for database operations.
    """
    sensenova_ak = os.environ.get("SENSENOVA_AK")
    sensenova_sk = os.environ.get("SENSENOVA_SK")
    deepseek_api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not sensenova_ak or not sensenova_sk or not deepseek_api_key:
        pytest.skip("SENSENOVA_AK or SENSENOVA_SK or DEEPSEEK_API_KEY is not set, skipping test_deepseek_stream")

    logger_cfg = dict(logger_name="test_deepseek_streaming", file_level=logging.DEBUG, logger_path="logs/pytest.log")
    deepseek_client_cfg = dict(
        type="DeepSeekConversationClient",
        name="deepseek_client",
        agent_prompts_file="configs/agent_prompts.yaml",
        deepseek_model_name="deepseek-chat",
        proxy_url=os.environ.get("PROXY_URL", None),
        logger_cfg=logger_cfg,
    )
    cascade_memories = await test_memory_adapter.db_client.get_cascade_memories(
        character_id=TEST_CHARACTER_ID,
    )
    adapter = build_conversation_adapter(deepseek_client_cfg)
    asyncio.create_task(adapter.run())
    profile = TextStreamProfile(mark_status_on_end=True, logger_cfg=logger_cfg)
    asyncio.create_task(profile.run())
    graph = DirectedAcyclicGraph(
        name="test_deepseek_streaming",
        conf=dict(
            user_prompt=agent_prompts["keqing_default"],
            character_id=TEST_CHARACTER_ID,
            user_settings=dict(
                deepseek_api_key=deepseek_api_key,
                sensenova_ak=sensenova_ak,
                sensenova_sk=sensenova_sk,
            ),
            language="zh",
            cascade_memories=cascade_memories,
            relationship=("Lover", 100),
            emotion=INITIAL_EMOTION_STATE,
            memory_adapter=test_memory_adapter,
            memory_db_client=mongodb_memory_client,
        ),
        logger_cfg=logger_cfg,
    )
    chat_node = DAGNode(
        name="chat_node",
        payload=adapter,
    )
    profile_node = DAGNode(
        name="profile_node",
        payload=profile,
    )
    graph.add_node(chat_node)
    graph.add_node(profile_node)
    graph.add_edge(chat_node.name, profile_node.name)
    graph.set_status(DAGStatus.RUNNING)
    request_id = str(uuid.uuid4())
    start_chunk = TextChunkStart(request_id=request_id, dag=graph, node_name=chat_node.name)
    await adapter.feed_stream(start_chunk)
    message = "Can you hug me?"
    for char in message:
        body_chunk = TextChunkBody(
            request_id=request_id,
            text_segment=char,
        )
        await adapter.feed_stream(body_chunk)
    end_chunk = TextChunkEnd(
        request_id=request_id,
    )
    await adapter.feed_stream(end_chunk)
    start_time = time.time()
    while graph.status != DAGStatus.COMPLETED:
        await asyncio.sleep(0.1)
        if time.time() - start_time > 10:
            raise TimeoutError("DeepSeek stream timeout")
    await adapter.interrupt()
    await profile.interrupt()
    await asyncio.sleep(adapter.sleep_time * 5)


@pytest.mark.asyncio
async def test_xai_stream(
    test_memory_adapter: SenseNovaOmniMemoryClient,
    mongodb_memory_client: MongoDBMemoryClient,
):
    """Test XAI conversation streaming functionality.

    Args:
        test_memory_adapter (SenseNovaOmniMemoryClient):
            SenseNova Omni memory client for conversation testing.
        mongodb_memory_client (MongoDBMemoryClient):
            MongoDB memory client for database operations.
    """
    sensenova_ak = os.environ.get("SENSENOVA_AK")
    sensenova_sk = os.environ.get("SENSENOVA_SK")
    xai_api_key = os.environ.get("XAI_API_KEY")
    if not sensenova_ak or not sensenova_sk or not xai_api_key:
        pytest.skip("SENSENOVA_AK or SENSENOVA_SK or XAI_API_KEY is not set, skipping test_xai_stream")

    logger_cfg = dict(logger_name="test_xai_streaming", file_level=logging.DEBUG, logger_path="logs/pytest.log")
    xai_client_cfg = dict(
        type="XAIConversationClient",
        name="xai_client",
        agent_prompts_file="configs/agent_prompts.yaml",
        xai_model_name="grok-3",
        proxy_url=os.environ.get("PROXY_URL", None),
        logger_cfg=logger_cfg,
    )
    cascade_memories = await test_memory_adapter.db_client.get_cascade_memories(
        character_id=TEST_CHARACTER_ID,
    )
    adapter = build_conversation_adapter(xai_client_cfg)
    asyncio.create_task(adapter.run())
    profile = TextStreamProfile(mark_status_on_end=True, logger_cfg=logger_cfg)
    asyncio.create_task(profile.run())
    graph = DirectedAcyclicGraph(
        name="test_xai_streaming",
        conf=dict(
            user_prompt=agent_prompts["keqing_default"],
            character_id=TEST_CHARACTER_ID,
            user_settings=dict(
                xai_api_key=xai_api_key,
                sensenova_ak=sensenova_ak,
                sensenova_sk=sensenova_sk,
            ),
            language="zh",
            cascade_memories=cascade_memories,
            relationship=("Lover", 100),
            emotion=INITIAL_EMOTION_STATE,
            memory_adapter=test_memory_adapter,
            memory_db_client=mongodb_memory_client,
        ),
        logger_cfg=logger_cfg,
    )
    chat_node = DAGNode(
        name="chat_node",
        payload=adapter,
    )
    profile_node = DAGNode(
        name="profile_node",
        payload=profile,
    )
    graph.add_node(chat_node)
    graph.add_node(profile_node)
    graph.add_edge(chat_node.name, profile_node.name)
    graph.set_status(DAGStatus.RUNNING)
    request_id = str(uuid.uuid4())
    start_chunk = TextChunkStart(request_id=request_id, dag=graph, node_name=chat_node.name)
    await adapter.feed_stream(start_chunk)
    message = "你在干嘛？"
    for char in message:
        body_chunk = TextChunkBody(
            request_id=request_id,
            text_segment=char,
        )
        await adapter.feed_stream(body_chunk)
    end_chunk = TextChunkEnd(
        request_id=request_id,
    )
    await adapter.feed_stream(end_chunk)
    start_time = time.time()
    while graph.status != DAGStatus.COMPLETED:
        await asyncio.sleep(0.1)
        if time.time() - start_time > 10:
            raise TimeoutError("XAI stream timeout")
    await adapter.interrupt()
    await profile.interrupt()
    await asyncio.sleep(adapter.sleep_time * 5)


@pytest.mark.asyncio
async def test_anthropic_stream(
    test_memory_adapter: SenseNovaOmniMemoryClient,
    mongodb_memory_client: MongoDBMemoryClient,
):
    """Test Anthropic conversation streaming functionality.

    Args:
        test_memory_adapter (SenseNovaOmniMemoryClient):
            SenseNova Omni memory client for conversation testing.
        mongodb_memory_client (MongoDBMemoryClient):
            MongoDB memory client for database operations.
    """
    sensenova_ak = os.environ.get("SENSENOVA_AK")
    sensenova_sk = os.environ.get("SENSENOVA_SK")
    anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not sensenova_ak or not sensenova_sk or not anthropic_api_key:
        pytest.skip("SENSENOVA_AK or SENSENOVA_SK or ANTHROPIC_API_KEY is not set, skipping test_anthropic_stream")

    logger_cfg = dict(logger_name="test_anthropic_streaming", file_level=logging.DEBUG, logger_path="logs/pytest.log")
    anthropic_client_cfg = dict(
        type="AnthropicConversationClient",
        name="anthropic_client",
        agent_prompts_file="configs/agent_prompts.yaml",
        anthropic_model_name="claude-sonnet-4-5-20250929",
        proxy_url=os.environ.get("PROXY_URL", None),
        logger_cfg=logger_cfg,
    )
    cascade_memories = await test_memory_adapter.db_client.get_cascade_memories(
        character_id=TEST_CHARACTER_ID,
    )
    adapter = build_conversation_adapter(anthropic_client_cfg)
    asyncio.create_task(adapter.run())
    profile = TextStreamProfile(mark_status_on_end=True, logger_cfg=logger_cfg)
    asyncio.create_task(profile.run())
    graph = DirectedAcyclicGraph(
        name="test_anthropic_streaming",
        conf=dict(
            user_prompt=agent_prompts["keqing_default"],
            character_id=TEST_CHARACTER_ID,
            user_settings=dict(
                anthropic_api_key=anthropic_api_key,
                sensenova_ak=sensenova_ak,
                sensenova_sk=sensenova_sk,
            ),
            language="zh",
            cascade_memories=cascade_memories,
            relationship=("Lover", 100),
            emotion=INITIAL_EMOTION_STATE,
            memory_adapter=test_memory_adapter,
            memory_db_client=mongodb_memory_client,
        ),
        logger_cfg=logger_cfg,
    )
    chat_node = DAGNode(
        name="chat_node",
        payload=adapter,
    )
    profile_node = DAGNode(
        name="profile_node",
        payload=profile,
    )
    graph.add_node(chat_node)
    graph.add_node(profile_node)
    graph.add_edge(chat_node.name, profile_node.name)
    graph.set_status(DAGStatus.RUNNING)
    request_id = str(uuid.uuid4())
    start_chunk = TextChunkStart(request_id=request_id, dag=graph, node_name=chat_node.name)
    await adapter.feed_stream(start_chunk)
    message = "你最近在忙什么？"
    for char in message:
        body_chunk = TextChunkBody(
            request_id=request_id,
            text_segment=char,
        )
        await adapter.feed_stream(body_chunk)
    end_chunk = TextChunkEnd(
        request_id=request_id,
    )
    await adapter.feed_stream(end_chunk)
    start_time = time.time()
    while graph.status != DAGStatus.COMPLETED:
        await asyncio.sleep(0.1)
        if time.time() - start_time > 10:
            raise TimeoutError("Anthropic stream timeout")
    await adapter.interrupt()
    await profile.interrupt()
    await asyncio.sleep(adapter.sleep_time * 5)


@pytest.mark.asyncio
async def test_sensenova_omni_stream(
    test_memory_adapter: SenseNovaOmniMemoryClient,
    mongodb_memory_client: MongoDBMemoryClient,
):
    """Test SenseNova Omni conversation streaming functionality.

    Args:
        test_memory_adapter (SenseNovaOmniMemoryClient):
            SenseNova Omni memory client for conversation testing.
        mongodb_memory_client (MongoDBMemoryClient):
            MongoDB memory client for database operations.
    """
    sensenova_ak = os.environ.get("SENSENOVA_AK")
    sensenova_sk = os.environ.get("SENSENOVA_SK")
    if not sensenova_ak or not sensenova_sk:
        pytest.skip("SENSENOVA_AK or SENSENOVA_SK is not set, skipping test_sensenova_omni_stream")

    logger_cfg = dict(
        logger_name="test_sensenova_omni_streaming", file_level=logging.DEBUG, logger_path="logs/pytest.log"
    )
    sensenova_omni_client_cfg = dict(
        type="SenseNovaOmniConversationClient",
        name="sensenova_omni_client",
        agent_prompts_file="configs/agent_prompts.yaml",
        wss_url="wss://api-gai.sensetime.com/agent-5o/duplex/ws2",
        logger_cfg=logger_cfg,
    )
    cascade_memories = await test_memory_adapter.db_client.get_cascade_memories(
        character_id=TEST_CHARACTER_ID,
    )
    adapter = build_conversation_adapter(sensenova_omni_client_cfg)
    asyncio.create_task(adapter.run())
    profile = TextStreamProfile(mark_status_on_end=True, logger_cfg=logger_cfg)
    asyncio.create_task(profile.run())
    graph = DirectedAcyclicGraph(
        name="test_sensenova_omni_streaming",
        conf=dict(
            user_prompt=agent_prompts["keqing_default"],
            character_id=TEST_CHARACTER_ID,
            user_settings=dict(
                sensenova_ak=sensenova_ak,
                sensenova_sk=sensenova_sk,
            ),
            language="zh",
            cascade_memories=cascade_memories,
            relationship=("Lover", 100),
            emotion=INITIAL_EMOTION_STATE,
            memory_adapter=test_memory_adapter,
            memory_db_client=mongodb_memory_client,
        ),
        logger_cfg=logger_cfg,
    )
    chat_node = DAGNode(
        name="chat_node",
        payload=adapter,
    )
    profile_node = DAGNode(
        name="profile_node",
        payload=profile,
    )
    graph.add_node(chat_node)
    graph.add_node(profile_node)
    graph.add_edge(chat_node.name, profile_node.name)
    graph.set_status(DAGStatus.RUNNING)
    request_id = str(uuid.uuid4())
    start_chunk = TextChunkStart(request_id=request_id, dag=graph, node_name=chat_node.name)
    await adapter.feed_stream(start_chunk)
    message = "你喜欢吃什么"
    for char in message:
        body_chunk = TextChunkBody(
            request_id=request_id,
            text_segment=char,
        )
        await adapter.feed_stream(body_chunk)
    end_chunk = TextChunkEnd(
        request_id=request_id,
    )
    await adapter.feed_stream(end_chunk)
    start_time = time.time()
    while graph.status != DAGStatus.COMPLETED:
        await asyncio.sleep(0.1)
        if time.time() - start_time > 10:
            raise TimeoutError("SenseNova Omni stream timeout")
    await adapter.interrupt()
    await profile.interrupt()
    await asyncio.sleep(adapter.sleep_time * 5)


@pytest.mark.asyncio
async def test_blank_input_stream(
    test_memory_adapter: SenseNovaOmniMemoryClient,
    mongodb_memory_client: MongoDBMemoryClient,
):
    """Test conversation streaming with blank input.

    Args:
        test_memory_adapter (SenseNovaOmniMemoryClient):
            SenseNova Omni memory client for conversation testing.
        mongodb_memory_client (MongoDBMemoryClient):
            MongoDB memory client for database operations.
    """
    sensenova_ak = os.environ.get("SENSENOVA_AK")
    sensenova_sk = os.environ.get("SENSENOVA_SK")
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not sensenova_ak or not sensenova_sk or not openai_api_key:
        pytest.skip("SENSENOVA_AK or SENSENOVA_SK or OPENAI_API_KEY is not set, skipping test_openai_stream")

    logger_cfg = dict(logger_name="test_openai_streaming", file_level=logging.DEBUG, logger_path="logs/pytest.log")
    openai_client_cfg = dict(
        type="OpenAIConversationClient",
        name="openai_client",
        agent_prompts_file="configs/agent_prompts.yaml",
        openai_model_name="gpt-4.1-2025-04-14",
        proxy_url=os.environ.get("PROXY_URL", None),
        logger_cfg=logger_cfg,
    )
    cascade_memories = await test_memory_adapter.db_client.get_cascade_memories(
        character_id=TEST_CHARACTER_ID,
    )
    adapter = build_conversation_adapter(openai_client_cfg)
    asyncio.create_task(adapter.run())
    profile = TextStreamProfile(mark_status_on_end=True, logger_cfg=logger_cfg)
    asyncio.create_task(profile.run())
    graph = DirectedAcyclicGraph(
        name="test_blank_input_stream",
        conf=dict(
            user_prompt=agent_prompts["keqing_default"],
            character_id=TEST_CHARACTER_ID,
            user_settings=dict(
                openai_api_key=openai_api_key,
                sensenova_ak=sensenova_ak,
                sensenova_sk=sensenova_sk,
            ),
            language="zh",
            cascade_memories=cascade_memories,
            relationship=("Lover", 100),
            emotion=INITIAL_EMOTION_STATE,
            memory_adapter=test_memory_adapter,
            memory_db_client=mongodb_memory_client,
        ),
        logger_cfg=logger_cfg,
    )
    chat_node = DAGNode(
        name="chat_node",
        payload=adapter,
    )
    profile_node = DAGNode(
        name="profile_node",
        payload=profile,
    )
    graph.add_node(chat_node)
    graph.add_node(profile_node)
    graph.add_edge(chat_node.name, profile_node.name)
    graph.set_status(DAGStatus.RUNNING)
    request_id = str(uuid.uuid4())
    start_chunk = TextChunkStart(request_id=request_id, dag=graph, node_name=chat_node.name)
    await adapter.feed_stream(start_chunk)
    message = ""
    for char in message:
        body_chunk = TextChunkBody(
            request_id=request_id,
            text_segment=char,
        )
        await adapter.feed_stream(body_chunk)
    end_chunk = TextChunkEnd(
        request_id=request_id,
    )
    await adapter.feed_stream(end_chunk)
    start_time = time.time()
    while graph.status != DAGStatus.COMPLETED:
        await asyncio.sleep(0.1)
        if time.time() - start_time > 10:
            raise TimeoutError("OpenAI stream timeout")
    await adapter.interrupt()
    await profile.interrupt()
    await asyncio.sleep(adapter.sleep_time * 5)


@pytest.mark.asyncio
async def test_classification_input_stream(
    test_memory_adapter: SenseNovaOmniMemoryClient,
    mongodb_memory_client: MongoDBMemoryClient,
):
    """Test conversation streaming with classification input.

    Args:
        test_memory_adapter (SenseNovaOmniMemoryClient):
            SenseNova Omni memory client for conversation testing.
        mongodb_memory_client (MongoDBMemoryClient):
            MongoDB memory client for database operations.
    """
    sensenova_ak = os.environ.get("SENSENOVA_AK")
    sensenova_sk = os.environ.get("SENSENOVA_SK")
    xai_api_key = os.environ.get("XAI_API_KEY")
    if not sensenova_ak or not sensenova_sk or not xai_api_key:
        pytest.skip("SENSENOVA_AK or SENSENOVA_SK or XAI_API_KEY is not set, skipping test_classification_input_stream")

    logger_cfg = dict(logger_name="test_xai_streaming", file_level=logging.DEBUG, logger_path="logs/pytest.log")
    xai_client_cfg = dict(
        type="XAIConversationClient",
        name="xai_client",
        agent_prompts_file="configs/agent_prompts.yaml",
        xai_model_name="grok-3",
        proxy_url=os.environ.get("PROXY_URL", None),
        logger_cfg=logger_cfg,
    )
    cascade_memories = await test_memory_adapter.db_client.get_cascade_memories(
        character_id=TEST_CHARACTER_ID,
    )
    adapter = build_conversation_adapter(xai_client_cfg)
    asyncio.create_task(adapter.run())
    profile = TextStreamProfile(mark_status_on_end=True, logger_cfg=logger_cfg)
    asyncio.create_task(profile.run())
    graph = DirectedAcyclicGraph(
        name="test_classification_input_stream",
        conf=dict(
            user_prompt=agent_prompts["keqing_default"],
            reject_prompt=agent_prompts["system_reject"],
            character_id=TEST_CHARACTER_ID,
            user_settings=dict(
                xai_api_key=xai_api_key,
                sensenova_ak=sensenova_ak,
                sensenova_sk=sensenova_sk,
            ),
            language="zh",
            cascade_memories=cascade_memories,
            relationship=("Lover", 100),
            emotion=INITIAL_EMOTION_STATE,
            memory_adapter=test_memory_adapter,
            memory_db_client=mongodb_memory_client,
        ),
        logger_cfg=logger_cfg,
    )
    chat_node = DAGNode(
        name="chat_node",
        payload=adapter,
    )
    profile_node = DAGNode(
        name="profile_node",
        payload=profile,
    )
    graph.add_node(chat_node)
    graph.add_node(profile_node)
    graph.add_edge(chat_node.name, profile_node.name)
    graph.set_status(DAGStatus.RUNNING)
    request_id = str(uuid.uuid4())
    cls_start_chunk = ClassificationChunkStart(
        request_id=request_id,
        dag=graph,
        node_name=chat_node.name,
    )
    await adapter.feed_stream(cls_start_chunk)
    cls_body_chunk = ClassificationChunkBody(
        request_id=request_id,
        classification_result=ClassificationType.REJECT,
        message="抱抱？",
    )
    await adapter.feed_stream(cls_body_chunk)
    cls_end_chunk = ClassificationChunkEnd(
        request_id=request_id,
    )
    await adapter.feed_stream(cls_end_chunk)
    start_time = time.time()
    while graph.status != DAGStatus.COMPLETED:
        await asyncio.sleep(0.1)
        if time.time() - start_time > 10:
            raise TimeoutError("XAI stream timeout")
    await adapter.interrupt()
    await profile.interrupt()
    await asyncio.sleep(adapter.sleep_time * 5)
