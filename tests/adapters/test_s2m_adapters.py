import asyncio
import io
import os
import time
import uuid

import pytest

from orchestrator.data_structures.audio_chunk import (
    AudioWithSubtitleChunkBody,
    AudioWithSubtitleChunkEnd,
    AudioWithSubtitleChunkStart,
)
from orchestrator.data_structures.process_flow import DAGNode, DAGStatus, DirectedAcyclicGraph
from orchestrator.generation.speech2motion.builder import build_speech2motion_adapter
from orchestrator.profile.motion_stream_profile import MotionStreamProfile
from orchestrator.utils.log import logging

VERIFY = False
WS_URL = os.environ.get("S2M_WS_URL", "")
pytestmark = pytest.mark.skipif(WS_URL == "", reason="S2M_WS_URL is not set, skipping test")
# TODO: Restore non-streaming S2M tests when needed


@pytest.mark.asyncio
async def test_speech2motion_client_stream_wav():
    """Test speech2motion streaming client with WAV audio format.

    Tests the streaming speech2motion client using WAV audio files with
    speech text and timing information. Verifies that the client can process
    speech text and generate motion animation data through WebSocket communication.

    Raises:
        TimeoutError:
            If the Speech2Motion stream processing times out.
    """
    logger_cfg = dict(
        logger_name="test_speech2motion_client_stream_wav", file_level=logging.DEBUG, logger_path="logs/pytest.log"
    )
    s2m_cfg = dict(
        type="Speech2MotionStreamingClient",
        verify=VERIFY,
        ws_url=WS_URL,
        timeout=20.0,
        logger_cfg=logger_cfg,
    )
    adapter = build_speech2motion_adapter(s2m_cfg)
    asyncio.create_task(adapter.run())
    profile = MotionStreamProfile(mark_status_on_end=True, logger_cfg=logger_cfg)
    asyncio.create_task(profile.run())
    graph = DirectedAcyclicGraph(
        name="test_speech2motion_client_v2_stream",
        conf=dict(
            avatar="KQ-default",
            character_id="orchestrator-pytest",
        ),
        logger_cfg=logger_cfg,
    )
    s2m_node = DAGNode(
        name="s2m_node",
        payload=adapter,
    )
    graph.add_node(s2m_node)
    profile_node = DAGNode(
        name="profile_node",
        payload=profile,
    )
    graph.add_node(profile_node)
    graph.add_edge(s2m_node.name, profile_node.name)
    graph.set_status(DAGStatus.RUNNING)
    request_id = str(uuid.uuid4())
    start_chunk = AudioWithSubtitleChunkStart(
        request_id=request_id,
        audio_type="wav",
        node_name=s2m_node.name,
        dag=graph,
    )
    await adapter.feed_stream(start_chunk)
    body_chunk = AudioWithSubtitleChunkBody(
        request_id=request_id,
        duration=1.355,
        audio_io=io.BytesIO(),
        seq_number=0,
        speech_text="2025年，",
        speech_time=[(0, 0.225), (1, 0.375), (2, 0.525), (3, 0.655), (4, 0.875)],
    )
    await adapter.feed_stream(body_chunk)
    body_chunk = AudioWithSubtitleChunkBody(
        request_id=request_id,
        duration=0.9,
        audio_io=io.BytesIO(),
        seq_number=1,
        speech_text="测测试。",
        speech_time=[(0, 1.56 - 1.355), (1, 1.73 - 1.355), (2, 1.93 - 1.355)],
    )
    await adapter.feed_stream(body_chunk)
    end_chunk = AudioWithSubtitleChunkEnd(
        request_id=request_id,
    )
    await adapter.feed_stream(end_chunk)
    start_time = time.time()
    while graph.status != DAGStatus.COMPLETED:
        await asyncio.sleep(0.1)
        if time.time() - start_time > 10:
            raise TimeoutError("Speech2Motion stream timeout")
    await adapter.interrupt()
    await profile.interrupt()
    await asyncio.sleep(adapter.sleep_time * 5)
    assert len(adapter.input_buffer) == 0
    assert len(profile.input_buffer) == 0


@pytest.mark.asyncio
async def test_speech2motion_client_v2_stream_pcm():
    """Test speech2motion streaming client with PCM audio format.

    Tests the streaming speech2motion client using PCM audio data with
    speech text and timing information. Verifies that the client can process
    speech text segments and generate motion animation data with proper
    timing synchronization.

    Raises:
        TimeoutError:
            If the Speech2Motion stream processing times out.
    """
    logger_cfg = dict(
        logger_name="test_speech2motion_client_v2_stream_pcm", file_level=logging.DEBUG, logger_path="logs/pytest.log"
    )
    s2m_cfg = dict(
        type="Speech2MotionStreamingClient",
        verify=VERIFY,
        ws_url=WS_URL,
        timeout=20.0,
        logger_cfg=logger_cfg,
    )
    adapter = build_speech2motion_adapter(s2m_cfg)
    asyncio.create_task(adapter.run())
    profile = MotionStreamProfile(mark_status_on_end=True, logger_cfg=logger_cfg)
    asyncio.create_task(profile.run())
    graph = DirectedAcyclicGraph(
        name="test_speech2motion_client_v2_stream",
        conf=dict(
            avatar="KQ-default",
            character_id="orchestrator-pytest",
        ),
        logger_cfg=logger_cfg,
    )
    s2m_node = DAGNode(
        name="s2m_node",
        payload=adapter,
    )
    graph.add_node(s2m_node)
    profile_node = DAGNode(
        name="profile_node",
        payload=profile,
    )
    graph.add_node(profile_node)
    graph.add_edge(s2m_node.name, profile_node.name)
    graph.set_status(DAGStatus.RUNNING)
    request_id = str(uuid.uuid4())
    start_chunk = AudioWithSubtitleChunkStart(
        request_id=request_id,
        audio_type="pcm",
        node_name=s2m_node.name,
        dag=graph,
        sample_width=2,
        n_channels=1,
        frame_rate=16000,
    )
    await adapter.feed_stream(start_chunk)
    body_chunk = AudioWithSubtitleChunkBody(
        request_id=request_id,
        duration=1.355,
        audio_io=io.BytesIO(),
        seq_number=0,
        speech_text="2025年，",
        speech_time=[(0, 0.225), (1, 0.375), (2, 0.525), (3, 0.655), (4, 0.875)],
    )
    await adapter.feed_stream(body_chunk)
    body_chunk = AudioWithSubtitleChunkBody(
        request_id=request_id,
        duration=0.9,
        audio_io=io.BytesIO(),
        seq_number=1,
        speech_text="测测试。",
        speech_time=[(0, 1.56 - 1.355), (1, 1.73 - 1.355), (2, 1.93 - 1.355)],
    )
    await adapter.feed_stream(body_chunk)
    end_chunk = AudioWithSubtitleChunkEnd(
        request_id=request_id,
    )
    await adapter.feed_stream(end_chunk)
    start_time = time.time()
    while graph.status != DAGStatus.COMPLETED:
        await asyncio.sleep(0.1)
        if time.time() - start_time > 10:
            raise TimeoutError("Speech2Motion stream timeout")
    await adapter.interrupt()
    await profile.interrupt()
    await asyncio.sleep(adapter.sleep_time * 5)
    assert len(adapter.input_buffer) == 0
    assert len(profile.input_buffer) == 0
