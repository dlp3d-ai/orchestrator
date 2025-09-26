import asyncio
import logging
import os
import time

import pytest

from orchestrator.aggregator.conversation_aggregator import ConversationAggregator
from orchestrator.data_structures.classification import (
    ClassificationChunkBody,
    ClassificationChunkEnd,
    ClassificationChunkStart,
    ClassificationType,
)
from orchestrator.data_structures.conversation import (
    ConversationChunkBody,
    ConversationChunkEnd,
    ConversationChunkStart,
    RejectChunkBody,
    RejectChunkEnd,
    RejectChunkStart,
)
from orchestrator.data_structures.process_flow import DAGNode, DAGStatus, DirectedAcyclicGraph
from orchestrator.io.memory.mongodb_memory_client import MongoDBMemoryClient
from orchestrator.memory.memory_adapter import INITIAL_EMOTION_STATE
from orchestrator.memory.sensenova_omni_memory_client import SenseNovaOmniMemoryClient
from orchestrator.profile.classified_text_stream_profile import ClassifiedStreamProfile

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


@pytest.mark.asyncio
async def test_conversation_aggregator_accept(mongodb_memory_client: MongoDBMemoryClient):
    """Test ConversationAggregator correctly handles ACCEPT classification
    results.

    This test verifies that the ConversationAggregator can correctly handle the
    ACCEPT classification result.
    """
    logger_cfg = dict(
        logger_name="test_conversation_aggregator_accept",
        console_level=logging.DEBUG,
    )
    memory = SenseNovaOmniMemoryClient(
        name="test_memory_adapter",
        db_client=mongodb_memory_client,
    )
    aggregator = ConversationAggregator(
        logger_cfg=logger_cfg,
    )
    cascade_memories = await memory.db_client.get_cascade_memories(
        character_id=TEST_CHARACTER_ID,
    )
    dag = DirectedAcyclicGraph(
        name="test_dag",
        conf=dict(
            start_time=time.time(),
            client_name="client_name",
            character_id=TEST_CHARACTER_ID,
            cascade_memories=cascade_memories,
            relationship=("Lover", 100),
            emotion=INITIAL_EMOTION_STATE,
            memory_adapter=memory,
            memory_db_client=mongodb_memory_client,
        ),
        logger_cfg=logger_cfg,
    )

    agg_node = DAGNode("agg", aggregator)

    profile = ClassifiedStreamProfile(mark_status_on_end=True, logger_cfg=logger_cfg)
    profile_node = DAGNode("sink", profile)
    dag.add_node(agg_node)
    dag.add_node(profile_node)
    dag.add_edge("agg", "sink")

    request_id = "req1"

    await aggregator.feed_stream(
        ConversationChunkStart(
            request_id=request_id,
            node_name="agg",
            dag=dag,
            client_name="client_name",
            user_input="可以抱抱我吗？",
        )
    )
    await aggregator.feed_stream(ConversationChunkBody(request_id=request_id, text_segment="来，抱一个，"))

    await aggregator.feed_stream(ClassificationChunkStart(request_id=request_id, node_name="agg", dag=dag))
    await aggregator.feed_stream(
        ClassificationChunkBody(
            request_id=request_id,
            message="可以抱抱我吗？",
            classification_result=ClassificationType.ACCEPT,
        )
    )
    await aggregator.feed_stream(ClassificationChunkEnd(request_id=request_id))

    await aggregator.feed_stream(ConversationChunkBody(request_id=request_id, text_segment="我最喜欢你了"))
    await aggregator.feed_stream(ConversationChunkEnd(request_id=request_id))

    dag.set_status(DAGStatus.RUNNING)
    agg_task = asyncio.create_task(aggregator.run())
    profile_task = asyncio.create_task(profile.run())

    start_time = time.time()
    while dag.status != DAGStatus.COMPLETED:
        await asyncio.sleep(0.1)
        if time.time() - start_time > 10:
            raise TimeoutError("Test timeout")

    conversation_history = await memory.db_client.get_chat_histories_after(
        character_id=TEST_CHARACTER_ID,
        after_timestamp=MongoDBMemoryClient.convert_unix_timestamp_to_str(start_time),
    )
    print(f"Chat history: {conversation_history}")


@pytest.mark.asyncio
async def test_conversation_aggregator_reject(mongodb_memory_client: MongoDBMemoryClient):
    """Test ConversationAggregator correctly handles REJECT classification
    results.

    This test verifies that the ConversationAggregator can correctly handle the
    REJECT classification result.
    """
    logger_cfg = dict(
        logger_name="test_conversation_aggregator_reject",
        console_level=logging.DEBUG,
    )
    memory = SenseNovaOmniMemoryClient(
        name="test_memory_adapter",
        db_client=mongodb_memory_client,
    )
    cascade_memories = await memory.db_client.get_cascade_memories(
        character_id=TEST_CHARACTER_ID,
    )
    aggregator = ConversationAggregator(queue_size=10, sleep_time=0.01)
    dag = DirectedAcyclicGraph(
        name="test_dag",
        conf=dict(
            start_time=time.time(),
            client_name="client_name",
            character_id=TEST_CHARACTER_ID,
            cascade_memories=cascade_memories,
            relationship=("Lover", 100),
            emotion=INITIAL_EMOTION_STATE,
            memory_adapter=memory,
            memory_db_client=mongodb_memory_client,
        ),
        logger_cfg=logger_cfg,
    )

    agg_node = DAGNode("agg", aggregator)

    profile = ClassifiedStreamProfile(mark_status_on_end=True, logger_cfg=logger_cfg)
    profile_node = DAGNode("sink", profile)
    dag.add_node(agg_node)
    dag.add_node(profile_node)
    dag.add_edge("agg", "sink")

    request_id = "req2"

    await aggregator.feed_stream(RejectChunkStart(request_id=request_id, node_name="agg", dag=dag))
    await aggregator.feed_stream(
        RejectChunkBody(
            request_id=request_id,
            text_segment="不可以，",
        )
    )

    await aggregator.feed_stream(
        ConversationChunkStart(
            request_id=request_id,
            node_name="agg",
            dag=dag,
            client_name="client_name",
            user_input="可以抱抱我吗？",
        )
    )
    await aggregator.feed_stream(ConversationChunkBody(request_id=request_id, text_segment="来，抱一个，"))
    await aggregator.feed_stream(ConversationChunkEnd(request_id=request_id))

    await aggregator.feed_stream(
        RejectChunkBody(
            request_id=request_id,
            text_segment="不好意思哦",
        )
    )
    await aggregator.feed_stream(RejectChunkEnd(request_id=request_id))

    await aggregator.feed_stream(ClassificationChunkStart(request_id=request_id, node_name="agg", dag=dag))
    await aggregator.feed_stream(
        ClassificationChunkBody(
            request_id=request_id,
            message="可以抱抱我吗？",
            classification_result=ClassificationType.REJECT,
        )
    )
    await aggregator.feed_stream(ClassificationChunkEnd(request_id=request_id))

    dag.set_status(DAGStatus.RUNNING)
    agg_task = asyncio.create_task(aggregator.run())
    profile_task = asyncio.create_task(profile.run())

    start_time = time.time()
    while dag.status != DAGStatus.COMPLETED:
        await asyncio.sleep(0.1)
        if time.time() - start_time > 10:
            raise TimeoutError("Test timeout")

    conversation_history = await memory.db_client.get_chat_histories_after(
        character_id=TEST_CHARACTER_ID,
        after_timestamp=MongoDBMemoryClient.convert_unix_timestamp_to_str(start_time),
    )
    print(f"Chat history: {conversation_history}")


@pytest.mark.asyncio
async def test_conversation_aggregator_leave(mongodb_memory_client: MongoDBMemoryClient):
    """Test ConversationAggregator correctly handles LEAVE classification
    results.

    This test verifies that the ConversationAggregator can correctly handle the
    LEAVE classification result.
    """
    logger_cfg = dict(
        logger_name="test_conversation_aggregator_leave",
        console_level=logging.DEBUG,
    )
    memory = SenseNovaOmniMemoryClient(
        name="test_memory_adapter",
        db_client=mongodb_memory_client,
    )
    cascade_memories = await memory.db_client.get_cascade_memories(
        character_id=TEST_CHARACTER_ID,
    )
    aggregator = ConversationAggregator(queue_size=10, sleep_time=0.01)
    dag = DirectedAcyclicGraph(
        name="test_dag",
        conf=dict(
            start_time=time.time(),
            client_name="client_name",
            character_id=TEST_CHARACTER_ID,
            cascade_memories=cascade_memories,
            relationship=("Lover", 100),
            emotion=INITIAL_EMOTION_STATE,
            memory_adapter=memory,
            memory_db_client=mongodb_memory_client,
        ),
        logger_cfg=logger_cfg,
    )

    agg_node = DAGNode("agg", aggregator)
    dag.add_node(agg_node)

    request_id = "req3"

    await aggregator.feed_stream(ClassificationChunkStart(request_id=request_id, node_name="agg", dag=dag))
    await aggregator.feed_stream(
        ClassificationChunkBody(
            request_id=request_id,
            message="拜拜",
            classification_result=ClassificationType.LEAVE,
        )
    )
    await aggregator.feed_stream(ClassificationChunkEnd(request_id=request_id))

    dag.set_status(DAGStatus.RUNNING)
    agg_task = asyncio.create_task(aggregator.run())

    start_time = time.time()
    while request_id in aggregator.input_buffer:
        await asyncio.sleep(0.1)
        if time.time() - start_time > 10:
            raise TimeoutError("Test timeout")

    dag.set_status(DAGStatus.COMPLETED)
