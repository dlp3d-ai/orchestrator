import asyncio
import io
import logging
import time

import pytest

from orchestrator.aggregator.tts_reaction_aggregator import TTSReactionAggregator
from orchestrator.data_structures.audio_chunk import (
    AudioWithSubtitleChunkBody,
    AudioWithSubtitleChunkEnd,
    AudioWithSubtitleChunkStart,
)
from orchestrator.data_structures.process_flow import DAGNode, DAGStatus, DirectedAcyclicGraph
from orchestrator.data_structures.reaction import Reaction, ReactionChunkBody, ReactionChunkEnd, ReactionChunkStart
from orchestrator.profile.audio_reaction_stream_profile import AudioReactionStreamProfile


@pytest.mark.asyncio
async def test_tts_reaction_aggregator_basic_flow():
    """Test TTSReactionAggregator can correctly stream-align TTS and Reaction
    data and output start, body, end chunks.

    This test verifies the basic functionality of the TTSReactionAggregator by:
    1. Creating an aggregator and DAG with downstream profile
    2. Feeding start signals for both TTS and Reaction streams
    3. Sending aligned body chunks with audio and reaction data
    4. Sending end signals to complete the streams
    5. Verifying the DAG completes successfully within timeout
    """
    logger_cfg = dict(
        logger_name="test_tts_reaction_aggregator_basic_flow",
        console_level=logging.DEBUG,
    )
    # Create aggregator and DAG
    aggregator = TTSReactionAggregator(queue_size=10, sleep_time=0.01)
    dag = DirectedAcyclicGraph("test_dag", {})
    # Create nodes and connect downstream
    agg_node = DAGNode("agg", aggregator)

    # Use Profile as downstream receiver
    profile = AudioReactionStreamProfile(mark_status_on_end=True, logger_cfg=logger_cfg)
    profile_node = DAGNode("sink", profile)
    dag.add_node(agg_node)
    dag.add_node(profile_node)
    dag.add_edge("agg", "sink")

    request_id = "req1"
    # Send start signals first
    await aggregator.feed_stream(
        AudioWithSubtitleChunkStart(request_id=request_id, audio_type="wav", node_name="agg", dag=dag)
    )
    await aggregator.feed_stream(ReactionChunkStart(request_id=request_id, node_name="agg", dag=dag))

    # Create reaction model with speech text and motion data
    reaction_model = {
        "speech_text": "你好",
        "label_expression": "(Stranger) & (Neutral)",
        "motion_keywords": [(0, "点头")],
        "face_emotion": "Neutral",
    }
    reaction_model = Reaction(**reaction_model)
    await aggregator.feed_stream(ReactionChunkBody(request_id=request_id, seq_number=0, reaction=reaction_model))
    # Send TTS body chunk
    audio_buffer = io.BytesIO(b"audio_bytes")
    tts_body = AudioWithSubtitleChunkBody(
        request_id=request_id,
        duration=0.5,
        audio_io=audio_buffer,
        seq_number=0,
        speech_text="你好",
        speech_time=[(0, 0.0)],
    )
    await aggregator.feed_stream(tts_body)

    # Send end signals
    await aggregator.feed_stream(AudioWithSubtitleChunkEnd(request_id=request_id))
    await aggregator.feed_stream(ReactionChunkEnd(request_id=request_id))

    # Run aggregator and Profile
    dag.set_status(DAGStatus.RUNNING)
    agg_task = asyncio.create_task(aggregator.run())
    profile_task = asyncio.create_task(profile.run())

    # Wait for DAG status to become COMPLETED
    start_time = time.time()
    while dag.status != DAGStatus.COMPLETED:
        await asyncio.sleep(0.1)
        if time.time() - start_time > 10:
            raise TimeoutError("Test timeout")


@pytest.mark.asyncio
async def test_tts_reaction_aggregator_misaligned_chunks():
    """Test TTSReactionAggregator can correctly align and output chunks when
    TTS and Reaction splits are misaligned.

    This test verifies the aggregator's ability to handle misaligned chunk boundaries by:
    1. Creating an aggregator and DAG with downstream profile
    2. Sending start signals for both streams
    3. Sending reaction chunks with different text boundaries than TTS chunks
    4. Verifying the aggregator can align and process the misaligned data
    5. Ensuring the DAG completes successfully within timeout
    """
    logger_cfg = dict(
        logger_name="test_tts_reaction_aggregator_misaligned_chunks",
        console_level=logging.DEBUG,
    )
    # Create aggregator and DAG
    aggregator = TTSReactionAggregator(queue_size=10, sleep_time=0.01)
    dag = DirectedAcyclicGraph("test_dag", {})
    agg_node = DAGNode("agg", aggregator)

    # Use Profile as downstream receiver
    profile = AudioReactionStreamProfile(mark_status_on_end=True, logger_cfg=logger_cfg)
    profile_node = DAGNode("sink", profile)
    dag.add_node(agg_node)
    dag.add_node(profile_node)
    dag.add_edge("agg", "sink")

    request_id = "req2"
    # Send start signals
    await aggregator.feed_stream(
        AudioWithSubtitleChunkStart(request_id=request_id, audio_type="wav", node_name="agg", dag=dag)
    )
    await aggregator.feed_stream(ReactionChunkStart(request_id=request_id, node_name="agg", dag=dag))

    # Create reaction models with different text boundaries
    reaction1 = Reaction(
        **{
            "speech_text": "你好，",
            "label_expression": "(Stranger) & (Neutral)",
            "motion_keywords": [(0, "点头")],
            "face_emotion": "Neutral",
        }
    )
    reaction2 = Reaction(
        **{
            "speech_text": "我的名字，叫韩梅梅",
            "label_expression": "(Stranger) & (Neutral)",
            "motion_keywords": [(0, "指向自己")],
            "face_emotion": "Neutral",
        }
    )
    reaction3 = Reaction(
        **{
            "speech_text": "很高兴认识你哟~",
            "label_expression": "(Stranger) & (Neutral)",
            "motion_keywords": [(1, "微笑")],
            "face_emotion": "Neutral",
        }
    )
    await aggregator.feed_stream(ReactionChunkBody(request_id=request_id, seq_number=0, reaction=reaction1))

    # TTS chunks split as ["你好，我的名字，", "叫韩梅梅", "很高兴认识你哟。"]
    # This creates misalignment with reaction chunks
    audio_buffer1 = io.BytesIO(b"audio1")
    tts1 = AudioWithSubtitleChunkBody(
        request_id=request_id,
        duration=0.3,
        audio_io=audio_buffer1,
        seq_number=0,
        speech_text="你好，我的名字，",
        speech_time=[(0, 0.0), (1, 0.1), (2, 0.2)],
    )
    await aggregator.feed_stream(tts1)

    await aggregator.feed_stream(ReactionChunkBody(request_id=request_id, seq_number=1, reaction=reaction2))

    audio_buffer2 = io.BytesIO(b"audio2")
    tts2 = AudioWithSubtitleChunkBody(
        request_id=request_id,
        duration=0.2,
        audio_io=audio_buffer2,
        seq_number=1,
        speech_text="叫韩梅梅",
        speech_time=[(0, 0.0), (1, 0.05), (2, 0.15)],
    )
    await aggregator.feed_stream(tts2)

    audio_buffer3 = io.BytesIO(b"audio3")
    tts3 = AudioWithSubtitleChunkBody(
        request_id=request_id,
        duration=0.1,
        audio_io=audio_buffer3,
        seq_number=2,
        speech_text="很高兴认识你哟。",
        speech_time=[(0, 0.0), (1, 0.05), (2, 0.08)],
    )
    await aggregator.feed_stream(tts3)
    await aggregator.feed_stream(ReactionChunkBody(request_id=request_id, seq_number=2, reaction=reaction3))

    # Send end signals
    await aggregator.feed_stream(AudioWithSubtitleChunkEnd(request_id=request_id))
    await aggregator.feed_stream(ReactionChunkEnd(request_id=request_id))

    # Run aggregator and Profile
    dag.set_status(DAGStatus.RUNNING)
    agg_task = asyncio.create_task(aggregator.run())
    profile_task = asyncio.create_task(profile.run())

    # Wait for DAG status to become COMPLETED
    start_time = time.time()
    while dag.status != DAGStatus.COMPLETED:
        await asyncio.sleep(0.1)
        if time.time() - start_time > 10:
            raise TimeoutError("Test timeout")
