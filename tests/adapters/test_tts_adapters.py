import asyncio
import os
import time
import uuid

import pytest

from orchestrator.data_structures.process_flow import DAGNode, DAGStatus, DirectedAcyclicGraph
from orchestrator.data_structures.text_chunk import TextChunkBody, TextChunkEnd, TextChunkStart
from orchestrator.generation.text2speech.builder import build_tts_adapter
from orchestrator.profile.audio_stream_profile import AudioStreamProfile
from orchestrator.utils.log import logging


@pytest.mark.asyncio
async def test_sensetime_tts_client_stream():
    """Test SenseTime TTS client streaming functionality.

    This test verifies that the SenseTime TTS adapter can process text chunks
    in streaming mode and generate audio output.

    The test will be skipped if ZOETROPE_TTS_WS_URL environment variable is not
    set.
    """
    tts_ws_url = os.environ.get("ZOETROPE_TTS_WS_URL")
    if not tts_ws_url:
        pytest.skip("ZOETROPE_TTS_WS_URL is not set, skipping test test_sensetime_tts_client_stream")

    logger_cfg = dict(
        logger_name="test_sensetime_tts_client_stream", file_level=logging.DEBUG, logger_path="logs/pytest.log"
    )
    tts_client_cfg = dict(
        type="SensetimeTTSClient",
        name="zoetrope_tts_client",
        tts_ws_url=tts_ws_url,
    )
    voice_name = "baimianmian"
    voice_speed = 1.0
    text = "Hi, my name is Han Meimei. Nice to meet you."
    adapter = build_tts_adapter(tts_client_cfg)
    asyncio.create_task(adapter.run())
    profile = AudioStreamProfile(mark_status_on_end=True, save_dir="output", logger_cfg=logger_cfg)
    asyncio.create_task(profile.run())
    graph = DirectedAcyclicGraph(
        name="test_sensetime_tts_client_stream",
        conf=dict(
            voice_name=voice_name,
            voice_speed=voice_speed,
            language="en",
        ),
        logger_cfg=logger_cfg,
    )
    tts_node = DAGNode(
        name="tts_node",
        payload=adapter,
    )
    profile_node = DAGNode(
        name="profile_node",
        payload=profile,
    )
    graph.add_node(tts_node)
    graph.add_node(profile_node)
    graph.add_edge(tts_node.name, profile_node.name)
    request_id = str(uuid.uuid4())
    start_chunk = TextChunkStart(
        request_id=request_id,
        node_name=tts_node.name,
        dag=graph,
    )
    graph.status = DAGStatus.RUNNING
    await adapter.feed_stream(start_chunk)
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
            raise TimeoutError("TTS stream timeout")
    await adapter.interrupt()
    await profile.interrupt()
    await asyncio.sleep(adapter.sleep_time * 5)


@pytest.mark.asyncio
async def test_huoshan_tts_client_stream():
    """Test Huoshan TTS client streaming functionality.

    This test verifies that the Huoshan TTS adapter can process text chunks in
    streaming mode and generate audio output.

    The test will be skipped if HUOSHAN_APPID or HUOSHAN_TOKEN environment
    variables are not set.
    """
    huoshan_app_id = os.environ.get("HUOSHAN_APPID")
    huoshan_token = os.environ.get("HUOSHAN_TOKEN")
    if not huoshan_app_id or not huoshan_token:
        pytest.skip("HUOSHAN_APPID or HUOSHAN_TOKEN is not set, skipping test test_huoshan_tts_client_stream")

    logger_cfg = dict(
        logger_name="test_huoshan_tts_client_stream", file_level=logging.DEBUG, logger_path="logs/pytest.log"
    )
    tts_client_cfg = dict(
        type="HuoshanTTSClient",
        name="huoshan_tts_client",
        tts_ws_url="wss://openspeech.bytedance.com/api/v1/tts/ws_binary",
        cluster="volcano_tts",
        logger_cfg=logger_cfg,
    )
    voice_name = "BV700_V2_streaming"
    voice_speed = 1.0
    text = "我是刻晴，璃月七星中的玉衡星。"
    adapter = build_tts_adapter(tts_client_cfg)
    asyncio.create_task(adapter.run())
    profile = AudioStreamProfile(
        mark_status_on_end=True,
        save_dir="output",
        logger_cfg=logger_cfg,
    )
    asyncio.create_task(profile.run())
    graph = DirectedAcyclicGraph(
        name="test_huoshan_tts_client_stream",
        conf=dict(
            user_settings=dict(
                huoshan_app_id=huoshan_app_id,
                huoshan_token=huoshan_token,
            ),
            voice_name=voice_name,
            voice_speed=voice_speed,
            language="zh",
        ),
        logger_cfg=logger_cfg,
    )
    tts_node = DAGNode(
        name="tts_node",
        payload=adapter,
    )
    profile_node = DAGNode(
        name="profile_node",
        payload=profile,
    )
    graph.add_node(tts_node)
    graph.add_node(profile_node)
    graph.add_edge(tts_node.name, profile_node.name)
    request_id = str(uuid.uuid4())
    start_chunk = TextChunkStart(
        request_id=request_id,
        node_name=tts_node.name,
        dag=graph,
    )
    graph.status = DAGStatus.RUNNING
    await adapter.feed_stream(start_chunk)
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
            raise TimeoutError("TTS stream timeout")
    await adapter.interrupt()
    await profile.interrupt()
    await asyncio.sleep(adapter.sleep_time * 5)


@pytest.mark.asyncio
async def test_softsugar_tts_client_stream():
    """Test SoftSugar TTS client streaming functionality.

    This test verifies that the SoftSugar TTS adapter can process text chunks
    in streaming mode and generate audio output.

    The test will be skipped if SOFTSUGAR_APP_ID or SOFTSUGAR_APP_KEY
    environment variables are not set.
    """
    softsugar_app_id = os.environ.get("SOFTSUGAR_APP_ID")
    softsugar_app_key = os.environ.get("SOFTSUGAR_APP_KEY")
    if not softsugar_app_id or not softsugar_app_key:
        pytest.skip("SOFTSUGAR_APP_ID or SOFTSUGAR_APP_KEY is not set, skipping test test_softsugar_tts_client_stream")

    logger_cfg = dict(
        logger_name="test_softsugar_tts_client_stream", file_level=logging.DEBUG, logger_path="logs/pytest.log"
    )
    tts_client_cfg = dict(
        type="SoftSugarTTSClient",
        name="softsugar_tts_client",
        tts_ws_url="ws://aigc.softsugar.com/api/voice/stream/v3",
        softsugar_token_url="https://aigc.softsugar.com/api/uc/v1/access/api/token",
        logger_cfg=logger_cfg,
    )
    voice_name = "8wfZav:AEA_Z10Mqp9GCwDGMrz8xIzi3VScxNzUtLCg"
    voice_speed = 1.0
    text = "你好，我是刻晴，璃月七星中的玉衡星。"
    adapter = build_tts_adapter(tts_client_cfg)
    asyncio.create_task(adapter.run())
    profile = AudioStreamProfile(
        mark_status_on_end=True,
        save_dir="output",
        logger_cfg=logger_cfg,
    )
    asyncio.create_task(profile.run())
    graph = DirectedAcyclicGraph(
        name="test_softsugar_tts_client_stream",
        conf=dict(
            user_settings=dict(
                softsugar_app_id=softsugar_app_id,
                softsugar_app_key=softsugar_app_key,
            ),
            voice_name=voice_name,
            voice_speed=voice_speed,
            language="en",
        ),
        logger_cfg=logger_cfg,
    )
    tts_node = DAGNode(
        name="tts_node",
        payload=adapter,
    )
    profile_node = DAGNode(
        name="profile_node",
        payload=profile,
    )
    graph.add_node(tts_node)
    graph.add_node(profile_node)
    graph.add_edge(tts_node.name, profile_node.name)
    request_id = str(uuid.uuid4())
    start_chunk = TextChunkStart(
        request_id=request_id,
        node_name=tts_node.name,
        dag=graph,
    )
    graph.status = DAGStatus.RUNNING
    await adapter.feed_stream(start_chunk)
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
            raise TimeoutError("TTS stream timeout")
    await adapter.interrupt()
    await profile.interrupt()
    await asyncio.sleep(adapter.sleep_time * 5)


@pytest.mark.asyncio
async def test_sensenova_tts_client_stream():
    """Test SenseNova TTS client streaming functionality.

    This test verifies that the SenseNova TTS adapter can process text chunks
    in streaming mode and generate audio output.

    The test will be skipped if NOVATTS_API_KEY environment variable is not
    set.
    """
    nova_tts_api_key = os.environ.get("NOVATTS_API_KEY")
    if not nova_tts_api_key:
        pytest.skip("NOVATTS_API_KEY is not set, skipping test test_sensenova_tts_client_stream")

    logger_cfg = dict(
        logger_name="test_sensenova_tts_client_stream", file_level=logging.DEBUG, logger_path="logs/pytest.log"
    )
    tts_client_cfg = dict(
        type="SensenovaTTSClient",
        name="sensenova_tts_client",
        tts_ws_url="wss://api-gai-internal.sensetime.com/agent-5o/signpost/v0/prod-agent-5o-tts-sovits-cosyvoice2-public/ws/stream",
        logger_cfg=logger_cfg,
    )
    voice_name = "female_xiaoxiao_m2"
    voice_speed = 1.0
    text = "我叫刻晴，璃月七星中的玉衡星。"
    adapter = build_tts_adapter(tts_client_cfg)
    asyncio.create_task(adapter.run())
    profile = AudioStreamProfile(
        mark_status_on_end=True,
        save_dir="output",
        logger_cfg=logger_cfg,
    )
    asyncio.create_task(profile.run())
    graph = DirectedAcyclicGraph(
        name="test_sensenova_tts_client_stream",
        conf=dict(
            user_settings=dict(
                nova_tts_api_key=nova_tts_api_key,
            ),
            voice_name=voice_name,
            voice_speed=voice_speed,
            language="en",
        ),
        logger_cfg=logger_cfg,
    )
    tts_node = DAGNode(
        name="tts_node",
        payload=adapter,
    )
    profile_node = DAGNode(
        name="profile_node",
        payload=profile,
    )
    graph.add_node(tts_node)
    graph.add_node(profile_node)
    graph.add_edge(tts_node.name, profile_node.name)
    request_id = str(uuid.uuid4())
    start_chunk = TextChunkStart(
        request_id=request_id,
        node_name=tts_node.name,
        dag=graph,
    )
    graph.status = DAGStatus.RUNNING
    await adapter.feed_stream(start_chunk)
    for char in text:
        body_chunk = TextChunkBody(
            request_id=request_id,
            text_segment=char,
            style="",
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
            raise TimeoutError("TTS stream timeout")
    await adapter.interrupt()
    await profile.interrupt()
    await asyncio.sleep(adapter.sleep_time * 5)


@pytest.mark.asyncio
async def test_elevenlabs_tts_client_stream():
    """Test ElevenLabs TTS client streaming functionality.

    This test verifies that the ElevenLabs TTS adapter can process text chunks
    in streaming mode and generate audio output.

    The test will be skipped if ELEVENLABS_API_KEY environment variable is not
    set.
    """
    elevenlabs_api_key = os.environ.get("ELEVENLABS_API_KEY")
    if not elevenlabs_api_key:
        pytest.skip("ELEVENLABS_API_KEY is not set, skipping test test_elevenlabs_tts_client_stream")

    logger_cfg = dict(
        logger_name="test_elevenlabs_tts_client_non_stream", file_level=logging.DEBUG, logger_path="logs/pytest.log"
    )
    tts_client_cfg = dict(
        type="ElevenLabsTTSClient",
        name="elevenlabs_tts_client",
        elevenlabs_model_name="eleven_flash_v2_5",
        proxy_url=os.environ.get("PROXY_URL", None),
        logger_cfg=logger_cfg,
    )
    voice_name = "EXAVITQu4vr4xnSDxMaL"
    voice_speed = 1.0
    text = "Hi, my name is Ke Qing. Nice to meet you."
    adapter = build_tts_adapter(tts_client_cfg)
    asyncio.create_task(adapter.run())
    profile = AudioStreamProfile(
        mark_status_on_end=True,
        save_dir="output",
        logger_cfg=logger_cfg,
    )
    asyncio.create_task(profile.run())
    graph = DirectedAcyclicGraph(
        name="test_elevenlabs_tts_client_stream",
        conf=dict(
            user_settings=dict(
                elevenlabs_api_key=elevenlabs_api_key,
            ),
            voice_name=voice_name,
            voice_speed=voice_speed,
            language="en",
        ),
        logger_cfg=logger_cfg,
    )
    tts_node = DAGNode(
        name="tts_node",
        payload=adapter,
    )
    profile_node = DAGNode(
        name="profile_node",
        payload=profile,
    )
    graph.add_node(tts_node)
    graph.add_node(profile_node)
    graph.add_edge(tts_node.name, profile_node.name)
    request_id = str(uuid.uuid4())
    start_chunk = TextChunkStart(
        request_id=request_id,
        node_name=tts_node.name,
        dag=graph,
    )
    graph.status = DAGStatus.RUNNING
    await adapter.feed_stream(start_chunk)
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
            raise TimeoutError("TTS stream timeout")
    await adapter.interrupt()
    await profile.interrupt()
    await asyncio.sleep(adapter.sleep_time * 5)
