import asyncio
import json
import os
import time
import uuid

import pytest

from orchestrator.data_structures.conversation import (
    ClassificationType,
    ClassifiedTextChunkBody,
    ClassifiedTextChunkEnd,
    ClassifiedTextChunkStart,
)
from orchestrator.data_structures.process_flow import DAGNode, DAGStatus, DirectedAcyclicGraph
from orchestrator.io.memory.mongodb_memory_client import MongoDBMemoryClient
from orchestrator.memory.memory_adapter import INITIAL_EMOTION_STATE, INITIAL_RELATIONSHIP_STATE
from orchestrator.memory.sensenova_omni_memory_client import SenseNovaOmniMemoryClient
from orchestrator.profile.reaction_stream_profile import ReactionStreamProfile
from orchestrator.reaction.builder import build_reaction_adapter
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
def test_memory_adapter(mongodb_memory_client: MongoDBMemoryClient):
    """Create a SenseNovaOmniMemoryClient instance for testing.

    Args:
        mongodb_memory_client (MongoDBMemoryClient):
            MongoDB memory client fixture for database operations.

    Returns:
        SenseNovaOmniMemoryClient:
            Configured SenseNova Omni memory client instance for testing.
    """
    return SenseNovaOmniMemoryClient(
        name="test_memory_adapter",
        db_client=mongodb_memory_client,
    )


test_client_name = "test_client_name"

motion_keywords = []
with open("configs/motion_kws.json", "r", encoding="utf-8") as f:
    motions = json.load(f)
    for motion in motions:
        if motion["motion_keywords_ch"]:
            motion_keywords.extend(motion["motion_keywords_ch"].split(","))
motion_keywords = list(set(motion_keywords))
print(f"Loaded {len(motion_keywords)} motion keywords")


@pytest.mark.asyncio
async def test_openai_reaction_client_stream(
    test_memory_adapter: SenseNovaOmniMemoryClient,
    mongodb_memory_client: MongoDBMemoryClient,
):
    """Test OpenAI reaction client streaming functionality.

    This test verifies that the OpenAI reaction adapter can process classified
    text chunks in streaming mode and generate appropriate emotional reactions.

    Args:
        test_memory_adapter (SenseNovaOmniMemoryClient):
            Memory adapter fixture for handling conversation memory.
        mongodb_memory_client (MongoDBMemoryClient):
            MongoDB client fixture for database operations.
    """
    sensenovaomni_ak = os.environ.get("SENSENOVAOMNI_AK")
    sensenovaomni_sk = os.environ.get("SENSENOVAOMNI_SK")
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not sensenovaomni_ak or not sensenovaomni_sk or not openai_api_key:
        pytest.skip(
            "SENSENOVAOMNI_AK or SENSENOVAOMNI_SK or OPENAI_API_KEY is not set, skipping test_openai_reaction_client_stream"
        )
    if not MONGODB_HOST:
        pytest.skip("MONGODB_HOST is not set, skipping test_openai_reaction_client_stream")

    logger_cfg = dict(
        logger_name="test_openai_reaction_client_stream", file_level=logging.DEBUG, logger_path="logs/pytest.log"
    )
    reaction_client_cfg = dict(
        type="OpenAIReactionClient",
        name="openai_reaction_client",
        motion_keywords=motion_keywords,
        openai_model_name="gpt-4.1-mini-2025-04-14",
        proxy_url=os.environ.get("PROXY_URL", None),
        logger_cfg=logger_cfg,
    )
    relationship = await mongodb_memory_client.get_relationship(
        character_id=TEST_CHARACTER_ID,
    )
    emotion = await mongodb_memory_client.get_emotion(
        character_id=TEST_CHARACTER_ID,
    )
    if relationship is None:
        relationship = (INITIAL_RELATIONSHIP_STATE["stage"], INITIAL_RELATIONSHIP_STATE["value"])
    if emotion is None:
        emotion = INITIAL_EMOTION_STATE

    adapter = build_reaction_adapter(reaction_client_cfg)
    asyncio.create_task(adapter.run())
    profile = ReactionStreamProfile(mark_status_on_end=True, logger_cfg=logger_cfg)
    asyncio.create_task(profile.run())
    graph = DirectedAcyclicGraph(
        name="test_reaction_stream",
        conf={
            "character_id": TEST_CHARACTER_ID,
            "language": "zh",
            "user_settings": dict(
                openai_api_key=openai_api_key,
                sensenovaomni_ak=sensenovaomni_ak,
                sensenovaomni_sk=sensenovaomni_sk,
            ),
            "relationship": relationship,
            "emotion": emotion,
            "memory_adapter": test_memory_adapter,
            "memory_db_client": mongodb_memory_client,
        },
        logger_cfg=logger_cfg,
    )
    reaction_node = DAGNode(
        name="reaction_node",
        payload=adapter,
    )
    profile_node = DAGNode(
        name="profile_node",
        payload=profile,
    )
    graph.add_node(reaction_node)
    graph.add_node(profile_node)
    graph.add_edge(reaction_node.name, profile_node.name)
    graph.set_status(DAGStatus.RUNNING)
    request_id = str(uuid.uuid4())
    start_chunk = ClassifiedTextChunkStart(
        request_id=request_id,
        dag=graph,
        node_name=reaction_node.name,
        classification_result=ClassificationType.ACCEPT,
        client_name=test_client_name,
        user_input="我给你准备了一个小礼物，期不期待？",
    )
    await adapter.feed_stream(start_chunk)
    text = "<style>惊讶</style>你居然给我准备礼物？这让我有点意外呢……说吧，到底是什么？别让我等太久！"
    for char in text:
        body_chunk = ClassifiedTextChunkBody(
            request_id=request_id,
            text_segment=char,
        )
        await adapter.feed_stream(body_chunk)
    end_chunk = ClassifiedTextChunkEnd(request_id=request_id)
    await adapter.feed_stream(end_chunk)
    start_time = time.time()
    while graph.status != DAGStatus.COMPLETED:
        await asyncio.sleep(0.1)
        if time.time() - start_time > 10:
            raise TimeoutError("OpenAI reaction stream timeout")
    await adapter.interrupt()
    await profile.interrupt()
    await asyncio.sleep(adapter.sleep_time * 5)


@pytest.mark.asyncio
async def test_xai_reaction_client_stream(
    test_memory_adapter: SenseNovaOmniMemoryClient,
    mongodb_memory_client: MongoDBMemoryClient,
):
    """Test XAI reaction client streaming functionality.

    This test verifies that the XAI reaction adapter can process classified
    text chunks in streaming mode and generate appropriate emotional reactions.

    Args:
        test_memory_adapter (SenseNovaOmniMemoryClient):
            Memory adapter fixture for handling conversation memory.
        mongodb_memory_client (MongoDBMemoryClient):
            MongoDB client fixture for database operations.
    """
    sensenovaomni_ak = os.environ.get("SENSENOVAOMNI_AK")
    sensenovaomni_sk = os.environ.get("SENSENOVAOMNI_SK")
    xai_api_key = os.environ.get("XAI_API_KEY")
    if not sensenovaomni_ak or not sensenovaomni_sk or not xai_api_key:
        pytest.skip(
            "SENSENOVAOMNI_AK or SENSENOVAOMNI_SK or XAI_API_KEY is not set, skipping test_xai_reaction_client_stream"
        )

    if not xai_api_key:
        pytest.skip("XAI_API_KEY is not available")
    if not MONGODB_HOST:
        pytest.skip("MONGODB_HOST is not set, skipping test_xai_reaction_client_stream")
    logger_cfg = dict(
        logger_name="test_xai_reaction_client_stream", file_level=logging.DEBUG, logger_path="logs/pytest.log"
    )
    reaction_client_cfg = dict(
        type="XAIReactionClient",
        name="xai_reaction_client",
        motion_keywords=motion_keywords,
        proxy_url=os.environ.get("PROXY_URL", None),
        xai_model_name="grok-3",
        logger_cfg=logger_cfg,
    )
    relationship = await mongodb_memory_client.get_relationship(
        character_id=TEST_CHARACTER_ID,
    )
    emotion = await mongodb_memory_client.get_emotion(
        character_id=TEST_CHARACTER_ID,
    )
    if relationship is None:
        relationship = (INITIAL_RELATIONSHIP_STATE["stage"], INITIAL_RELATIONSHIP_STATE["value"])
    if emotion is None:
        emotion = INITIAL_EMOTION_STATE

    adapter = build_reaction_adapter(reaction_client_cfg)
    asyncio.create_task(adapter.run())
    profile = ReactionStreamProfile(mark_status_on_end=True, logger_cfg=logger_cfg)
    asyncio.create_task(profile.run())
    graph = DirectedAcyclicGraph(
        name="test_reaction_stream",
        conf={
            "character_id": TEST_CHARACTER_ID,
            "language": "zh",
            "user_settings": dict(
                xai_api_key=xai_api_key,
                sensenovaomni_ak=sensenovaomni_ak,
                sensenovaomni_sk=sensenovaomni_sk,
            ),
            "relationship": relationship,
            "emotion": emotion,
            "memory_adapter": test_memory_adapter,
            "memory_db_client": mongodb_memory_client,
        },
        logger_cfg=logger_cfg,
    )
    reaction_node = DAGNode(
        name="reaction_node",
        payload=adapter,
    )
    profile_node = DAGNode(
        name="profile_node",
        payload=profile,
    )
    graph.add_node(reaction_node)
    graph.add_node(profile_node)
    graph.add_edge(reaction_node.name, profile_node.name)
    graph.set_status(DAGStatus.RUNNING)
    request_id = str(uuid.uuid4())
    start_chunk = ClassifiedTextChunkStart(
        request_id=request_id,
        dag=graph,
        node_name=reaction_node.name,
        classification_result=ClassificationType.ACCEPT,
        client_name=test_client_name,
        user_input="我给你准备了一个小礼物，期不期待？",
    )
    await adapter.feed_stream(start_chunk)
    text = "<style>惊讶</style>你居然给我准备礼物？这让我有点意外呢……说吧，到底是什么？别让我等太久！"
    for char in text:
        body_chunk = ClassifiedTextChunkBody(
            request_id=request_id,
            text_segment=char,
        )
        await adapter.feed_stream(body_chunk)
    end_chunk = ClassifiedTextChunkEnd(request_id=request_id)
    await adapter.feed_stream(end_chunk)
    start_time = time.time()
    while graph.status != DAGStatus.COMPLETED:
        await asyncio.sleep(0.1)
        if time.time() - start_time > 10:
            raise TimeoutError("XAI reaction stream timeout")
    await adapter.interrupt()
    await profile.interrupt()
    await asyncio.sleep(adapter.sleep_time * 5)


@pytest.mark.asyncio
async def test_gemini_reaction_client_stream(
    test_memory_adapter: SenseNovaOmniMemoryClient,
    mongodb_memory_client: MongoDBMemoryClient,
):
    """Test Gemini reaction client streaming functionality.

    This test verifies that the Gemini reaction adapter can process classified
    text chunks in streaming mode and generate appropriate emotional reactions.

    Args:
        test_memory_adapter (SenseNovaOmniMemoryClient):
            Memory adapter fixture for handling conversation memory.
        mongodb_memory_client (MongoDBMemoryClient):
            MongoDB client fixture for database operations.
    """
    sensenovaomni_ak = os.environ.get("SENSENOVAOMNI_AK")
    sensenovaomni_sk = os.environ.get("SENSENOVAOMNI_SK")
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    if not sensenovaomni_ak or not sensenovaomni_sk or not gemini_api_key:
        pytest.skip(
            "SENSENOVAOMNI_AK or SENSENOVAOMNI_SK or GEMINI_API_KEY is not set, skipping test_gemini_reaction_client_stream"
        )

    if not gemini_api_key:
        pytest.skip("GEMINI_API_KEY is not available")
    if not MONGODB_HOST:
        pytest.skip("MONGODB_HOST is not set, skipping test_gemini_reaction_client_stream")
    logger_cfg = dict(
        logger_name="test_gemini_reaction_client_stream", file_level=logging.DEBUG, logger_path="logs/pytest.log"
    )
    reaction_client_cfg = dict(
        type="GeminiReactionClient",
        name="gemini_reaction_client",
        motion_keywords=motion_keywords,
        proxy_url=os.environ.get("PROXY_URL", None),
        gemini_model_name="gemini-2.5-flash-lite",
        logger_cfg=logger_cfg,
    )
    relationship = await mongodb_memory_client.get_relationship(
        character_id=TEST_CHARACTER_ID,
    )
    emotion = await mongodb_memory_client.get_emotion(
        character_id=TEST_CHARACTER_ID,
    )
    if relationship is None:
        relationship = (INITIAL_RELATIONSHIP_STATE["stage"], INITIAL_RELATIONSHIP_STATE["value"])
    if emotion is None:
        emotion = INITIAL_EMOTION_STATE

    adapter = build_reaction_adapter(reaction_client_cfg)
    asyncio.create_task(adapter.run())
    profile = ReactionStreamProfile(mark_status_on_end=True, logger_cfg=logger_cfg)
    asyncio.create_task(profile.run())
    graph = DirectedAcyclicGraph(
        name="test_reaction_stream",
        conf={
            "character_id": TEST_CHARACTER_ID,
            "language": "en",
            "user_settings": dict(
                gemini_api_key=gemini_api_key,
                sensenovaomni_ak=sensenovaomni_ak,
                sensenovaomni_sk=sensenovaomni_sk,
            ),
            "relationship": relationship,
            "emotion": emotion,
            "memory_adapter": test_memory_adapter,
            "memory_db_client": mongodb_memory_client,
        },
        logger_cfg=logger_cfg,
    )
    reaction_node = DAGNode(
        name="reaction_node",
        payload=adapter,
    )
    profile_node = DAGNode(
        name="profile_node",
        payload=profile,
    )
    graph.add_node(reaction_node)
    graph.add_node(profile_node)
    graph.add_edge(reaction_node.name, profile_node.name)
    graph.set_status(DAGStatus.RUNNING)
    request_id = str(uuid.uuid4())
    start_chunk = ClassifiedTextChunkStart(
        request_id=request_id,
        dag=graph,
        node_name=reaction_node.name,
        classification_result=ClassificationType.ACCEPT,
        client_name=test_client_name,
        user_input="I have prepared a small gift for you, are you excited?",
    )
    await adapter.feed_stream(start_chunk)
    text = "<style>surprise</style>You actually prepared a gift for me? This is a bit unexpected... What is it?"
    for char in text:
        body_chunk = ClassifiedTextChunkBody(
            request_id=request_id,
            text_segment=char,
        )
        await adapter.feed_stream(body_chunk)
    end_chunk = ClassifiedTextChunkEnd(request_id=request_id)
    await adapter.feed_stream(end_chunk)
    start_time = time.time()
    while graph.status != DAGStatus.COMPLETED:
        await asyncio.sleep(0.1)
        if time.time() - start_time > 10:
            raise TimeoutError("Gemini reaction stream timeout")
    await adapter.interrupt()
    await profile.interrupt()
    await asyncio.sleep(adapter.sleep_time * 5)


@pytest.mark.asyncio
async def test_sensenova_omni_reaction_client_stream(
    test_memory_adapter: SenseNovaOmniMemoryClient,
    mongodb_memory_client: MongoDBMemoryClient,
):
    """Test SenseNova Omni reaction client streaming functionality.

    This test verifies that the SenseNova Omni reaction adapter can process
    classified text chunks in streaming mode and generate appropriate emotional
    reactions.

    Args:
        test_memory_adapter (SenseNovaOmniMemoryClient):
            Memory adapter fixture for handling conversation memory.
        mongodb_memory_client (MongoDBMemoryClient):
            MongoDB client fixture for database operations.
    """
    sensenovaomni_ak = os.environ.get("SENSENOVAOMNI_AK")
    sensenovaomni_sk = os.environ.get("SENSENOVAOMNI_SK")
    if not sensenovaomni_ak or not sensenovaomni_sk:
        pytest.skip(
            "SENSENOVAOMNI_AK or SENSENOVAOMNI_SK is not set, skipping test_sensenova_omni_reaction_client_stream"
        )
    if not MONGODB_HOST:
        pytest.skip("MONGODB_HOST is not set, skipping test_sensenova_omni_reaction_client_stream")

    logger_cfg = dict(
        logger_name="test_sensenova_omni_reaction_client_stream",
        file_level=logging.DEBUG,
        logger_path="logs/pytest.log",
    )
    reaction_client_cfg = dict(
        type="SenseNovaOmniReactionClient",
        name="sensenova_omni_reaction_client",
        motion_keywords=motion_keywords,
        wss_url="wss://api-gai.sensetime.com/agent-5o/duplex/ws2",
        proxy_url=os.environ.get("PROXY_URL", None),
        logger_cfg=logger_cfg,
    )
    relationship = await mongodb_memory_client.get_relationship(
        character_id=TEST_CHARACTER_ID,
    )
    emotion = await mongodb_memory_client.get_emotion(
        character_id=TEST_CHARACTER_ID,
    )
    if relationship is None:
        relationship = (INITIAL_RELATIONSHIP_STATE["stage"], INITIAL_RELATIONSHIP_STATE["value"])
    if emotion is None:
        emotion = INITIAL_EMOTION_STATE

    adapter = build_reaction_adapter(reaction_client_cfg)
    asyncio.create_task(adapter.run())
    profile = ReactionStreamProfile(mark_status_on_end=True, logger_cfg=logger_cfg)
    asyncio.create_task(profile.run())
    graph = DirectedAcyclicGraph(
        name="test_reaction_stream",
        conf={
            "character_id": TEST_CHARACTER_ID,
            "language": "zh",
            "user_settings": dict(
                sensenovaomni_ak=sensenovaomni_ak,
                sensenovaomni_sk=sensenovaomni_sk,
            ),
            "relationship": relationship,
            "emotion": emotion,
            "memory_adapter": test_memory_adapter,
            "memory_db_client": mongodb_memory_client,
        },
        logger_cfg=logger_cfg,
    )
    reaction_node = DAGNode(
        name="reaction_node",
        payload=adapter,
    )
    profile_node = DAGNode(
        name="profile_node",
        payload=profile,
    )
    graph.add_node(reaction_node)
    graph.add_node(profile_node)
    graph.add_edge(reaction_node.name, profile_node.name)
    graph.set_status(DAGStatus.RUNNING)
    request_id = str(uuid.uuid4())
    start_chunk = ClassifiedTextChunkStart(
        request_id=request_id,
        dag=graph,
        node_name=reaction_node.name,
        classification_result=ClassificationType.ACCEPT,
        client_name=test_client_name,
        user_input="我为你准备了一个小礼物，期不期待？",
    )
    await adapter.feed_stream(start_chunk)
    text = "<style>惊讶</style>你居然给我准备礼物？这让我有点意外呢……说吧到底是什么别让我等太久一定要记住！记住。"
    for char in text:
        body_chunk = ClassifiedTextChunkBody(
            request_id=request_id,
            text_segment=char,
        )
        await adapter.feed_stream(body_chunk)
    end_chunk = ClassifiedTextChunkEnd(request_id=request_id)
    await adapter.feed_stream(end_chunk)
    start_time = time.time()
    while graph.status != DAGStatus.COMPLETED:
        await asyncio.sleep(0.1)
        if time.time() - start_time > 10:
            raise TimeoutError("Gemini reaction stream timeout")
    await adapter.interrupt()
    await profile.interrupt()
    await asyncio.sleep(adapter.sleep_time * 5)


@pytest.mark.asyncio
async def test_dummy_reaction_client_stream(
    test_memory_adapter: SenseNovaOmniMemoryClient,
    mongodb_memory_client: MongoDBMemoryClient,
):
    """Test dummy reaction client streaming functionality.

    This test verifies that the dummy reaction adapter can process classified
    text chunks in streaming mode and generate basic responses without external
    API calls.

    Args:
        test_memory_adapter (SenseNovaOmniMemoryClient):
            Memory adapter fixture for handling conversation memory.
        mongodb_memory_client (MongoDBMemoryClient):
            MongoDB client fixture for database operations.
    """
    logger_cfg = dict(
        logger_name="test_dummy_reaction_client_stream", file_level=logging.DEBUG, logger_path="logs/pytest.log"
    )
    reaction_client_cfg = dict(
        type="DummyReactionClient",
        name="dummy_reaction_client",
        motion_keywords=motion_keywords,
        logger_cfg=logger_cfg,
    )
    relationship = await mongodb_memory_client.get_relationship(
        character_id=TEST_CHARACTER_ID,
    )
    emotion = await mongodb_memory_client.get_emotion(
        character_id=TEST_CHARACTER_ID,
    )
    if relationship is None:
        relationship = (INITIAL_RELATIONSHIP_STATE["stage"], INITIAL_RELATIONSHIP_STATE["value"])
    if emotion is None:
        emotion = INITIAL_EMOTION_STATE

    adapter = build_reaction_adapter(reaction_client_cfg)
    asyncio.create_task(adapter.run())
    profile = ReactionStreamProfile(mark_status_on_end=True, logger_cfg=logger_cfg)
    asyncio.create_task(profile.run())
    graph = DirectedAcyclicGraph(
        name="test_reaction_stream",
        conf={
            "character_id": TEST_CHARACTER_ID,
            "language": "zh",
            "relationship": relationship,
            "emotion": emotion,
            "memory_adapter": test_memory_adapter,
            "memory_db_client": mongodb_memory_client,
        },
        logger_cfg=logger_cfg,
    )
    reaction_node = DAGNode(
        name="reaction_node",
        payload=adapter,
    )
    profile_node = DAGNode(
        name="profile_node",
        payload=profile,
    )
    graph.add_node(reaction_node)
    graph.add_node(profile_node)
    graph.add_edge(reaction_node.name, profile_node.name)
    graph.set_status(DAGStatus.RUNNING)
    request_id = str(uuid.uuid4())
    start_chunk = ClassifiedTextChunkStart(
        request_id=request_id,
        dag=graph,
        node_name=reaction_node.name,
        classification_result=ClassificationType.ACCEPT,
        client_name=test_client_name,
        user_input="你叫什么名字？",
    )
    await adapter.feed_stream(start_chunk)
    text = "你好，我叫韩梅梅，今天很高兴认识你。"
    for char in text:
        body_chunk = ClassifiedTextChunkBody(
            request_id=request_id,
            text_segment=char,
        )
        await adapter.feed_stream(body_chunk)
    end_chunk = ClassifiedTextChunkEnd(request_id=request_id)
    await adapter.feed_stream(end_chunk)
    start_time = time.time()
    while graph.status != DAGStatus.COMPLETED:
        await asyncio.sleep(0.1)
        if time.time() - start_time > 10:
            raise TimeoutError("Dummy reaction stream timeout")
    await adapter.interrupt()
    await profile.interrupt()
    await asyncio.sleep(adapter.sleep_time * 5)
