import asyncio
import json
import os
import time
import uuid

import pytest

from orchestrator.classification.builder import build_classification_adapter
from orchestrator.data_structures.process_flow import DAGNode, DAGStatus, DirectedAcyclicGraph
from orchestrator.data_structures.text_chunk import TextChunkBody, TextChunkEnd, TextChunkStart
from orchestrator.profile.classification_stream_profile import ClassificationStreamProfile
from orchestrator.utils.log import logging

motion_keywords = []
with open("configs/motion_kws.json", "r", encoding="utf-8") as f:
    motions = json.load(f)
    for motion in motions:
        if motion["motion_keywords_ch"]:
            motion_keywords.extend(motion["motion_keywords_ch"].split(","))
motion_keywords = list(set(motion_keywords))
print(f"Loaded {len(motion_keywords)} motion keywords")


@pytest.mark.asyncio
async def test_openai_classification_client_stream():
    """Test OpenAI classification client streaming functionality.

    This test verifies that the OpenAI classification adapter can process text
    chunks in streaming mode and correctly classify motion keywords.

    The test will be skipped if OPENAI_API_KEY environment variable is not set.
    """
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        pytest.skip("openai_api_key is not set, skipping test test_openai_classification_client_stream")

    logger_cfg = dict(
        logger_name="test_openai_classification_client_stream", file_level=logging.DEBUG, logger_path="logs/pytest.log"
    )
    classification_client_cfg = dict(
        type="OpenAIClassificationClient",
        name="openai_classification_client",
        motion_keywords=motion_keywords,
        openai_model_name="gpt-4o-mini-2024-07-18",
        proxy_url=os.environ.get("PROXY_URL", None),
        logger_cfg=logger_cfg,
    )
    adapter = build_classification_adapter(classification_client_cfg)
    asyncio.create_task(adapter.run())
    profile = ClassificationStreamProfile(mark_status_on_end=True, logger_cfg=logger_cfg)
    asyncio.create_task(profile.run())
    graph = DirectedAcyclicGraph(
        name="test_classification_stream",
        conf=dict(
            language="zh",
            user_settings=dict(
                openai_api_key=openai_api_key,
            ),
        ),
        logger_cfg=logger_cfg,
    )
    classification_node = DAGNode(
        name="classification_node",
        payload=adapter,
    )
    profile_node = DAGNode(
        name="profile_node",
        payload=profile,
    )
    graph.add_node(classification_node)
    graph.add_node(profile_node)
    graph.add_edge(classification_node.name, profile_node.name)
    graph.set_status(DAGStatus.RUNNING)
    request_id = str(uuid.uuid4())
    start_chunk = TextChunkStart(
        request_id=request_id,
        dag=graph,
        node_name=classification_node.name,
    )
    await adapter.feed_stream(start_chunk)
    text = "请转个圈"
    for char in text:
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
            raise TimeoutError("Classification stream timeout")
    await adapter.interrupt()
    await profile.interrupt()
    await asyncio.sleep(adapter.sleep_time * 5)


@pytest.mark.asyncio
async def test_xai_classification_client_stream():
    """Test XAI classification client streaming functionality.

    This test verifies that the XAI classification adapter can process text
    chunks in streaming mode and correctly classify motion keywords.

    The test will be skipped if XAI_API_KEY environment variable is not set.
    """
    xai_api_key = os.environ.get("XAI_API_KEY")
    if not xai_api_key:
        pytest.skip("xai_api_key is not set, skipping test test_xai_classification_client_stream")

    logger_cfg = dict(
        logger_name="test_xai_classification_client_stream", file_level=logging.DEBUG, logger_path="logs/pytest.log"
    )
    classification_client_cfg = dict(
        type="XAIClassificationClient",
        name="xai_classification_client",
        motion_keywords=motion_keywords,
        xai_model_name="grok-3",
        proxy_url=os.environ.get("PROXY_URL", None),
        logger_cfg=logger_cfg,
    )
    adapter = build_classification_adapter(classification_client_cfg)
    asyncio.create_task(adapter.run())
    profile = ClassificationStreamProfile(mark_status_on_end=True, logger_cfg=logger_cfg)
    asyncio.create_task(profile.run())
    graph = DirectedAcyclicGraph(
        name="test_classification_stream",
        conf=dict(
            language="zh",
            user_settings=dict(
                xai_api_key=xai_api_key,
            ),
        ),
        logger_cfg=logger_cfg,
    )
    classification_node = DAGNode(
        name="classification_node",
        payload=adapter,
    )
    profile_node = DAGNode(
        name="profile_node",
        payload=profile,
    )
    graph.add_node(classification_node)
    graph.add_node(profile_node)
    graph.add_edge(classification_node.name, profile_node.name)
    graph.set_status(DAGStatus.RUNNING)
    request_id = str(uuid.uuid4())
    start_chunk = TextChunkStart(
        request_id=request_id,
        dag=graph,
        node_name=classification_node.name,
    )
    await adapter.feed_stream(start_chunk)
    text = "请跳支舞"
    for char in text:
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
            raise TimeoutError("Classification stream timeout")
    await adapter.interrupt()
    await profile.interrupt()
    await asyncio.sleep(adapter.sleep_time * 5)


@pytest.mark.asyncio
async def test_gemini_classification_client_stream():
    """Test Gemini classification client streaming functionality.

    This test verifies that the Gemini classification adapter can process text
    chunks in streaming mode and correctly classify motion keywords.

    The test will be skipped if GEMINI_API_KEY environment variable is not set.
    """
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_api_key:
        pytest.skip("gemini_api_key is not set, skipping test test_gemini_classification_client_stream")

    logger_cfg = dict(
        logger_name="test_gemini_classification_client_stream", file_level=logging.DEBUG, logger_path="logs/pytest.log"
    )
    classification_client_cfg = dict(
        type="GeminiClassificationClient",
        name="gemini_classification_client",
        motion_keywords=motion_keywords,
        gemini_model_name="gemini-2.5-flash-lite",
        proxy_url=os.environ.get("PROXY_URL", None),
        logger_cfg=logger_cfg,
    )
    adapter = build_classification_adapter(classification_client_cfg)
    asyncio.create_task(adapter.run())
    profile = ClassificationStreamProfile(mark_status_on_end=True, logger_cfg=logger_cfg)
    asyncio.create_task(profile.run())
    graph = DirectedAcyclicGraph(
        name="test_classification_stream",
        conf=dict(
            language="en",
            user_settings=dict(
                gemini_api_key=gemini_api_key,
            ),
        ),
        logger_cfg=logger_cfg,
    )
    classification_node = DAGNode(
        name="classification_node",
        payload=adapter,
    )
    profile_node = DAGNode(
        name="profile_node",
        payload=profile,
    )
    graph.add_node(classification_node)
    graph.add_node(profile_node)
    graph.add_edge(classification_node.name, profile_node.name)
    graph.set_status(DAGStatus.RUNNING)
    request_id = str(uuid.uuid4())
    start_chunk = TextChunkStart(
        request_id=request_id,
        dag=graph,
        node_name=classification_node.name,
    )
    await adapter.feed_stream(start_chunk)
    text = "Sing a song, please."
    for char in text:
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
            raise TimeoutError("Classification stream timeout")
    await adapter.interrupt()
    await profile.interrupt()
    await asyncio.sleep(adapter.sleep_time * 5)


@pytest.mark.asyncio
async def test_sensenova_omni_classification_client_stream():
    """Test SenseNova Omni classification client streaming functionality.

    This test verifies that the SenseNova Omni classification adapter can
    process text chunks in streaming mode and correctly classify motion
    keywords.

    The test will be skipped if SENSENOVA_AK or SENSENOVA_SK environment
    variables are not set.
    """
    sensenova_ak = os.environ.get("SENSENOVA_AK")
    sensenova_sk = os.environ.get("SENSENOVA_SK")
    if not sensenova_ak or not sensenova_sk:
        pytest.skip(
            "sensenova_ak or sensenova_sk is not set, skipping test test_sensenova_omni_classification_client_stream"
        )

    logger_cfg = dict(
        logger_name="test_sensenova_omni_classification_client_stream",
        file_level=logging.DEBUG,
        logger_path="logs/pytest.log",
    )
    classification_client_cfg = dict(
        type="SenseNovaOmniClassificationClient",
        name="sensenova_omni_classification_client",
        motion_keywords=motion_keywords,
        wss_url="wss://api-gai.sensetime.com/agent-5o/duplex/ws2",
        proxy_url=os.environ.get("PROXY_URL", None),
        logger_cfg=logger_cfg,
    )
    adapter = build_classification_adapter(classification_client_cfg)
    asyncio.create_task(adapter.run())
    profile = ClassificationStreamProfile(mark_status_on_end=True, logger_cfg=logger_cfg)
    asyncio.create_task(profile.run())
    graph = DirectedAcyclicGraph(
        name="test_classification_stream",
        conf=dict(
            language="zh",
            user_settings=dict(
                sensenova_ak=sensenova_ak,
                sensenova_sk=sensenova_sk,
            ),
        ),
        logger_cfg=logger_cfg,
    )
    classification_node = DAGNode(
        name="classification_node",
        payload=adapter,
    )
    profile_node = DAGNode(
        name="profile_node",
        payload=profile,
    )
    graph.add_node(classification_node)
    graph.add_node(profile_node)
    graph.add_edge(classification_node.name, profile_node.name)
    graph.set_status(DAGStatus.RUNNING)
    request_id = str(uuid.uuid4())
    start_chunk = TextChunkStart(
        request_id=request_id,
        dag=graph,
        node_name=classification_node.name,
    )
    await adapter.feed_stream(start_chunk)
    text = "你会跳舞吗"
    for char in text:
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
            raise TimeoutError("Classification stream timeout")
    await adapter.interrupt()
    await profile.interrupt()
    await asyncio.sleep(adapter.sleep_time * 5)
