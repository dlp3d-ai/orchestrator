import asyncio
import io
import os
import time
import uuid
import wave
from typing import List, Union

import numpy as np
import pytest

from orchestrator.data_structures.audio_chunk import AudioChunkBody, AudioChunkEnd, AudioChunkStart
from orchestrator.data_structures.process_flow import DAGNode, DAGStatus, DirectedAcyclicGraph
from orchestrator.generation.speech_recognition.builder import build_asr_adapter
from orchestrator.profile.text_stream_profile import TextStreamProfile
from orchestrator.utils.log import logging


def _wav_to_chunks(
    input_wav_path: str,
    segment_duration: float,
    request_id: str,
    dag: DirectedAcyclicGraph,
    node_name: str,
    frame_rate: int = 16000,
) -> List[Union[AudioChunkStart, AudioChunkBody, AudioChunkEnd]]:
    """Convert WAV file to audio chunks for streaming processing.

    This function reads a WAV file and segments it into audio chunks suitable
    for streaming ASR processing. It validates audio format requirements and
    ensures proper byte order for PCM data.

    Args:
        input_wav_path (str):
            Path to the input WAV file.
        segment_duration (float):
            Duration of each audio segment in seconds.
        request_id (str):
            Unique identifier for the request.
        dag (DirectedAcyclicGraph):
            The DAG instance for processing flow.
        node_name (str):
            Name of the node in the DAG.
        frame_rate (int, optional):
            Target frame rate in Hz. Defaults to 16000.

    Returns:
        List[Union[AudioChunkStart, AudioChunkBody, AudioChunkEnd]]:
            List of audio chunks including start, body, and end chunks.

    Raises:
        ValueError:
            If audio format requirements are not met (mono, 16-bit, little-endian).
    """
    # Open WAV file
    start_chunk = AudioChunkStart(
        request_id=request_id,
        audio_type="pcm",
        node_name=node_name,
        dag=dag,
        n_channels=1,
        sample_width=2,
        frame_rate=frame_rate,
    )
    ret_list = [start_chunk]
    with wave.open(input_wav_path, "rb") as wav_file:
        # Get audio parameters
        n_channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        frame_rate = wav_file.getframerate()
        n_frames = wav_file.getnframes()

        # Check and ensure audio parameters meet requirements
        if n_channels != 1:
            raise ValueError("Input audio must be mono (1 channel)")
        if frame_rate != frame_rate:
            raise ValueError(f"Input audio must be {frame_rate}Hz")
        if sample_width != 2:  # Ensure it's 16-bit
            raise ValueError(f"Input audio must be 16-bit, got {sample_width} bytes")

        # Check byte order (little-endian)
        if wav_file.getcomptype() != "NONE":
            raise ValueError("Input audio must be uncompressed PCM")
        if wav_file.getcomptype() == "NONE" and wav_file.getsampwidth() == 2:
            # For 16-bit PCM, check if it's little-endian
            if wav_file.getnframes() > 0:
                first_frame = wav_file.readframes(1)
                if first_frame[0] != 0 or first_frame[1] != 0:  # Simple little-endian check
                    raise ValueError("Input audio must be little-endian")
                wav_file.rewind()  # Reset file pointer to start position

        # Calculate frames per segment
        frames_per_segment = int(frame_rate * segment_duration)

        # Read audio data
        frames = wav_file.readframes(n_frames)

        # Convert byte data to numpy array, ensuring little-endian byte order
        if sample_width == 2:  # 16-bit
            dtype = np.int16
        elif sample_width == 4:  # 32-bit
            dtype = np.int32
        else:
            raise ValueError(f"Unsupported sample width: {sample_width}")

        audio_data = np.frombuffer(frames, dtype=dtype)

        # Ensure data is little-endian
        if audio_data.dtype.byteorder == ">":  # If big-endian
            audio_data = audio_data.byteswap()  # Convert to little-endian

        # Calculate number of segments needed
        n_segments = (len(audio_data) + frames_per_segment - 1) // frames_per_segment

        # Save PCM data segments to BytesIO
        for i in range(n_segments):
            start_idx = i * frames_per_segment
            end_idx = min((i + 1) * frames_per_segment, len(audio_data))
            segment_data = audio_data[start_idx:end_idx]
            segment_duration = len(segment_data) / frame_rate
            # Generate output filename
            output_pcm = io.BytesIO()
            # Save PCM data
            output_pcm.write(segment_data.tobytes())
            output_pcm.seek(0)
            body_chunk = AudioChunkBody(
                request_id=request_id,
                audio_io=output_pcm,
                duration=segment_duration,
                seq_number=i,
            )
            ret_list.append(body_chunk)
    end_chunk = AudioChunkEnd(
        request_id=request_id,
    )
    ret_list.append(end_chunk)
    return ret_list


@pytest.mark.asyncio
async def test_sensetime_asr_client_stream():
    """Test Sensetime ASR client with streaming PCM mode.

    This test verifies the streaming functionality of the Sensetime ASR client
    by processing audio data in real-time chunks and validating the complete
    processing pipeline including DAG execution and buffer management.

    The test is skipped if ZOETROPE_ASR_WS_URL environment variable is not set.
    """
    ws_url = os.environ.get("ZOETROPE_ASR_WS_URL")
    if not ws_url:
        pytest.skip(
            "ZOETROPE_ASR_WS_URL environment variable is not set, skipping test test_sensetime_asr_client_stream"
        )

    logger_cfg = dict(
        logger_name="test_sensetime_asr_client_stream",
        console_level=logging.DEBUG,
    )
    asr_cfg = dict(
        type="SensetimeASRClient",
        name="sensetime_asr_client",
        ws_url=ws_url,
        logger_cfg=logger_cfg,
    )
    adapter = build_asr_adapter(asr_cfg)
    asyncio.create_task(adapter.run())
    profile = TextStreamProfile(mark_status_on_end=True, logger_cfg=logger_cfg)
    asyncio.create_task(profile.run())
    graph = DirectedAcyclicGraph(
        name="test_sensetime_asr_client_stream",
        conf=dict(),
        logger_cfg=logger_cfg,
    )
    asr_node = DAGNode(
        name="asr_node",
        payload=adapter,
    )
    profile_node = DAGNode(
        name="pcm_profile_node",
        payload=profile,
    )
    graph.add_node(asr_node)
    graph.add_node(profile_node)
    graph.add_edge(asr_node.name, profile_node.name)
    request_id = str(uuid.uuid4())
    audio_path = "input/test_audio.wav"
    chunks = _wav_to_chunks(audio_path, adapter.CHUNK_DURATION, request_id, graph, asr_node.name)
    graph.conf["start_time"] = time.time()
    graph.status = DAGStatus.RUNNING
    last_send_time = 0.0
    for chunk in chunks:
        cur_time = time.time()
        if not isinstance(chunk, AudioChunkBody):
            await adapter.feed_stream(chunk)
        else:
            elapsed_since_last_send = cur_time - last_send_time
            if last_send_time != 0.0 and elapsed_since_last_send < adapter.CHUNK_DURATION:
                wait_duration = adapter.CHUNK_DURATION - elapsed_since_last_send
                if wait_duration > 0:  # Ensure wait_duration is positive
                    await asyncio.sleep(wait_duration)

            await adapter.feed_stream(chunk)
            last_send_time = time.time()  # Update last_send_time to the actual send time of the current chunk
    start_time = time.time()
    while graph.status != DAGStatus.COMPLETED:
        await asyncio.sleep(0.1)
    if time.time() - start_time > 10:
        raise TimeoutError("ASR stream timeout")
    await adapter.interrupt()
    await profile.interrupt()
    await asyncio.sleep(adapter.sleep_time * 5)
    assert len(adapter.input_buffer) == 0
    assert len(profile.input_buffer) == 0


@pytest.mark.asyncio
async def test_softsugar_asr_client_stream():
    """Test SoftSugar ASR client with streaming mode.

    This test verifies the streaming functionality of the SoftSugar ASR client
    by processing audio data in real-time chunks and validating the complete
    processing pipeline including DAG execution and buffer management.

    Requires SOFTSUGAR_APP_ID and SOFTSUGAR_APP_KEY environment variables.
    """
    softsugar_app_id = os.environ.get("SOFTSUGAR_APP_ID")
    softsugar_app_key = os.environ.get("SOFTSUGAR_APP_KEY")
    if not softsugar_app_id or not softsugar_app_key:
        pytest.skip("softsugar_app_id or softsugar_app_key is not set, skipping test test_softsugar_asr_client_stream")

    logger_cfg = dict(
        logger_name="test_softsugar_asr_client_stream",
        console_level=logging.DEBUG,
    )
    asr_cfg = dict(
        type="SoftSugarASRClient",
        name="softsugar_asr_client",
        ws_url="ws://aigc.softsugar.com/api/voice/stream/v1",
        softsugar_api="https://aigc.softsugar.com",
        logger_cfg=logger_cfg,
    )
    adapter = build_asr_adapter(asr_cfg)
    asyncio.create_task(adapter.run())
    profile = TextStreamProfile(mark_status_on_end=True, logger_cfg=logger_cfg)
    asyncio.create_task(profile.run())
    graph = DirectedAcyclicGraph(
        name="test_softsugar_asr_client_stream",
        conf=dict(
            start_time=0.0,
            user_settings=dict(
                softsugar_app_id=softsugar_app_id,
                softsugar_app_key=softsugar_app_key,
            ),
        ),
        logger_cfg=logger_cfg,
    )
    asr_node = DAGNode(
        name="asr_node",
        payload=adapter,
    )
    profile_node = DAGNode(
        name="profile_node",
        payload=profile,
    )
    graph.add_node(asr_node)
    graph.add_node(profile_node)
    graph.add_edge(asr_node.name, profile_node.name)
    request_id = str(uuid.uuid4())
    audio_path = "input/test_audio.wav"
    chunks = _wav_to_chunks(audio_path, adapter.CHUNK_DURATION, request_id, graph, asr_node.name)
    graph.conf["start_time"] = time.time()
    graph.status = DAGStatus.RUNNING
    last_send_time = 0.0
    for chunk in chunks:
        cur_time = time.time()
        if not isinstance(chunk, AudioChunkBody):
            await adapter.feed_stream(chunk)
        else:
            elapsed_since_last_send = cur_time - last_send_time
            if last_send_time != 0.0 and elapsed_since_last_send < adapter.CHUNK_DURATION:
                wait_duration = adapter.CHUNK_DURATION - elapsed_since_last_send
                if wait_duration > 0:  # Ensure wait_duration is positive
                    await asyncio.sleep(wait_duration)

            await adapter.feed_stream(chunk)
            last_send_time = time.time()  # Update last_send_time to the actual send time of the current chunk
    while graph.status != DAGStatus.COMPLETED:
        await asyncio.sleep(0.1)
        if time.time() - graph.conf["start_time"] > 10:
            raise TimeoutError("ASR stream timeout")
    await adapter.interrupt()
    await profile.interrupt()
    await asyncio.sleep(adapter.sleep_time * 5)
    assert len(adapter.input_buffer) == 0
    assert len(profile.input_buffer) == 0


@pytest.mark.asyncio
async def test_openai_realtime_asr_client_stream():
    """Test OpenAI Realtime ASR client with streaming mode.

    This test verifies the streaming functionality of the OpenAI Realtime ASR
    client by processing audio data in real-time chunks and validating the
    complete processing pipeline including DAG execution and buffer management.

    Requires OPENAI_API_KEY and PROXY_URL environment variables.
    """
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        pytest.skip("openai_api_key is not set, skipping test test_openai_realtime_asr_client_stream")

    logger_cfg = dict(
        logger_name="test_openai_realtime_asr_client_stream",
        console_level=logging.DEBUG,
    )
    asr_cfg = dict(
        type="OpenAIRealtimeASRClient",
        name="openai_realtime_asr_client",
        wss_url="wss://api.openai.com/v1/realtime?intent=transcription",
        openai_model_name="gpt-4o-mini-transcribe",
        proxy_url=os.environ.get("PROXY_URL", None),
        logger_cfg=logger_cfg,
    )
    adapter = build_asr_adapter(asr_cfg)
    asyncio.create_task(adapter.run())
    profile = TextStreamProfile(mark_status_on_end=True, logger_cfg=logger_cfg)
    asyncio.create_task(profile.run())
    graph = DirectedAcyclicGraph(
        name="test_softsugar_asr_client_stream",
        conf=dict(
            start_time=0.0,
            user_settings=dict(
                openai_api_key=openai_api_key,
            ),
        ),
        logger_cfg=logger_cfg,
    )
    asr_node = DAGNode(
        name="asr_node",
        payload=adapter,
    )
    profile_node = DAGNode(
        name="profile_node",
        payload=profile,
    )
    graph.add_node(asr_node)
    graph.add_node(profile_node)
    graph.add_edge(asr_node.name, profile_node.name)
    request_id = str(uuid.uuid4())
    audio_path = "input/test_audio.wav"
    chunks = _wav_to_chunks(audio_path, adapter.CHUNK_DURATION, request_id, graph, asr_node.name, frame_rate=24000)
    graph.conf["start_time"] = time.time()
    graph.status = DAGStatus.RUNNING
    last_send_time = 0.0
    for chunk in chunks:
        cur_time = time.time()
        if not isinstance(chunk, AudioChunkBody):
            await adapter.feed_stream(chunk)
        else:
            elapsed_since_last_send = cur_time - last_send_time
            if last_send_time != 0.0 and elapsed_since_last_send < adapter.CHUNK_DURATION:
                wait_duration = adapter.CHUNK_DURATION - elapsed_since_last_send
                if wait_duration > 0:  # Ensure wait_duration is positive
                    await asyncio.sleep(wait_duration)

            await adapter.feed_stream(chunk)
            last_send_time = time.time()  # Update last_send_time to the actual send time of the current chunk
    while graph.status != DAGStatus.COMPLETED:
        await asyncio.sleep(0.1)
        if time.time() - graph.conf["start_time"] > 10:
            raise TimeoutError("ASR stream timeout")
    await adapter.interrupt()
    await profile.interrupt()
    await asyncio.sleep(0.5)

    # Get all tasks in current event loop
    current_task = asyncio.current_task()
    tasks = [task for task in asyncio.all_tasks() if task is not current_task]

    if tasks:
        # Give tasks some time to complete, then cancel unfinished tasks
        try:
            await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=2.0)
        except asyncio.TimeoutError:
            # If timeout, cancel all tasks
            print(f"Timeout waiting for tasks to complete, starting to cancel {len(tasks)} unfinished tasks:")
            for task in tasks:
                if not task.done():
                    # Get task name and function information
                    task_name = task.get_name()
                    coro = task.get_coro() if hasattr(task, "get_coro") else None
                    func_name = "unknown"
                    if coro and hasattr(coro, "__qualname__"):
                        func_name = coro.__qualname__
                    elif coro and hasattr(coro, "__name__"):
                        func_name = coro.__name__

                    print(f"  Canceling task: {task_name} -> {func_name}")
                    task.cancel()
            print("Waiting for all tasks to be canceled...")
            # Wait for cancellation to complete
            await asyncio.gather(*tasks, return_exceptions=True)

    assert len(adapter.input_buffer) == 0
    assert len(profile.input_buffer) == 0
