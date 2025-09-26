import asyncio
import io
import os
import time
import uuid
import wave
from typing import List, Union

import numpy as np
import pytest

from orchestrator.data_structures.audio_chunk import (
    AudioChunkBody,
    AudioChunkEnd,
    AudioChunkStart,
    AudioWithReactionChunkBody,
    AudioWithReactionChunkEnd,
    AudioWithReactionChunkStart,
)
from orchestrator.data_structures.process_flow import DAGNode, DAGStatus, DirectedAcyclicGraph
from orchestrator.data_structures.reaction import Reaction
from orchestrator.generation.audio2face.builder import build_audio2face_adapter
from orchestrator.profile.face_stream_profile import FaceStreamProfile
from orchestrator.utils.log import logging

VERIFY = False
WS_URL = os.environ.get("A2F_WS_URL", "")
pytestmark = pytest.mark.skipif(WS_URL == "", reason="A2F_WS_URL is not set, skipping test")

# TODO: Restore non-streaming A2F tests when needed


def _wav_to_audio_chunks(
    input_wav_path: str,
    segment_duration: float,
    request_id: str,
    dag: DirectedAcyclicGraph,
    node_name: str,
) -> List[Union[AudioChunkStart, AudioChunkBody, AudioChunkEnd]]:
    """Convert WAV file to audio chunks for streaming processing.

    Reads a WAV file and segments it into audio chunks with specified duration.
    Validates audio format requirements and handles byte order conversion.

    Args:
        input_wav_path (str):
            Path to the input WAV file.
        segment_duration (float):
            Duration of each audio segment in seconds.
        request_id (str):
            Unique identifier for the request.
        dag (DirectedAcyclicGraph):
            Directed acyclic graph for workflow management.
        node_name (str):
            Name of the processing node.

    Returns:
        List[Union[AudioChunkStart, AudioChunkBody, AudioChunkEnd]]:
            List of audio chunks including start, body, and end chunks.

    Raises:
        ValueError:
            If audio format requirements are not met.
    """
    # Open WAV file
    start_chunk = AudioChunkStart(
        request_id=request_id,
        audio_type="pcm",
        node_name=node_name,
        dag=dag,
        n_channels=1,
        sample_width=2,
        frame_rate=16000,
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
        if frame_rate != 16000:
            raise ValueError("Input audio must be 16000Hz")
        if sample_width != 2:  # Ensure it is 16-bit
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

        # Convert byte data to numpy array, ensure little-endian
        if sample_width == 2:  # 16-bit
            dtype = np.int16
        elif sample_width == 4:  # 32-bit
            dtype = np.int32
        else:
            raise ValueError(f"Unsupported sample width: {sample_width}")

        audio_data = np.frombuffer(frames, dtype=dtype)

        # Ensure data is little-endian
        if audio_data.dtype.byteorder == ">":  # If it is big-endian
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


def _wav_to_audio_with_reaction_chunks(
    input_wav_path: str,
    segment_duration: float,
    request_id: str,
    dag: DirectedAcyclicGraph,
    node_name: str,
) -> List[Union[AudioWithReactionChunkStart, AudioWithReactionChunkBody, AudioWithReactionChunkEnd]]:
    """Convert WAV file to audio chunks with reaction data.

    Converts a WAV file to audio chunks and adds reaction information.
    Creates neutral emotion reactions for each audio chunk.

    Args:
        input_wav_path (str):
            Path to the input WAV file.
        segment_duration (float):
            Duration of each audio segment in seconds.
        request_id (str):
            Unique identifier for the request.
        dag (DirectedAcyclicGraph):
            Directed acyclic graph for workflow management.
        node_name (str):
            Name of the processing node.

    Returns:
        List[Union[AudioWithReactionChunkStart, AudioWithReactionChunkBody, AudioWithReactionChunkEnd]]:
            List of audio chunks with reaction data.

    Raises:
        ValueError:
            If unknown audio chunk type is encountered.
    """
    audio_chunks = _wav_to_audio_chunks(
        input_wav_path=input_wav_path,
        segment_duration=segment_duration,
        request_id=request_id,
        dag=dag,
        node_name=node_name,
    )
    ret_list = []
    for audio_chunk in audio_chunks:
        if isinstance(audio_chunk, AudioChunkBody):
            reaction_chunk = AudioWithReactionChunkBody(
                request_id=audio_chunk.request_id,
                duration=audio_chunk.duration,
                seq_number=audio_chunk.seq_number,
                audio_io=audio_chunk.audio_io,
                speech_text="",  # A2F doesn't care about speech text
                speech_time=[],  # A2F doesn't care about speech timing
                reaction=Reaction(
                    speech_text="",
                    label_expression=None,
                    motion_keywords=None,
                    face_emotion="Neutral",
                ),
            )
            ret_list.append(reaction_chunk)
        elif isinstance(audio_chunk, AudioChunkStart):
            new_chunk = AudioWithReactionChunkStart(
                request_id=audio_chunk.request_id,
                audio_type=audio_chunk.audio_type,
                node_name=audio_chunk.node_name,
                dag=audio_chunk.dag,
                n_channels=audio_chunk.n_channels,
                sample_width=audio_chunk.sample_width,
                frame_rate=audio_chunk.frame_rate,
            )
            ret_list.append(new_chunk)
        elif isinstance(audio_chunk, AudioChunkEnd):
            new_chunk = AudioWithReactionChunkEnd(
                request_id=audio_chunk.request_id,
            )
            ret_list.append(new_chunk)
        else:
            raise ValueError(f"Unknown audio chunk type: {type(audio_chunk)}")
    return ret_list


@pytest.mark.asyncio
async def test_audio2face_streaming_client_wav():
    """Test audio2face streaming client with WAV audio format.

    Tests the streaming audio2face client using WAV audio files.
    Verifies that the client can process WAV audio and generate
    face animation data through WebSocket communication.

    Raises:
        TimeoutError:
            If the A2F stream processing times out.
    """
    logger_cfg = dict(
        logger_name="test_audio2face_streaming_client_wav", file_level=logging.DEBUG, logger_path="logs/pytest.log"
    )
    a2f_cfg = dict(
        type="Audio2FaceStreamingClient",
        verify=VERIFY,
        ws_url=WS_URL,
        timeout=10.0,
        logger_cfg=logger_cfg,
    )
    adapter = build_audio2face_adapter(a2f_cfg)
    asyncio.create_task(adapter.run())
    profile = FaceStreamProfile(mark_status_on_end=True, logger_cfg=logger_cfg)
    asyncio.create_task(profile.run())
    graph = DirectedAcyclicGraph(
        name="test_audio2face_streaming_client_wav",
        conf=dict(
            avatar="KQ-default",
            response_chunk_n_frames=10,
        ),
        logger_cfg=logger_cfg,
    )
    a2f_node = DAGNode(
        name="a2f_node",
        payload=adapter,
    )
    graph.add_node(a2f_node)
    profile_node = DAGNode(
        name="profile_node",
        payload=profile,
    )
    graph.add_node(profile_node)
    graph.add_edge(a2f_node.name, profile_node.name)
    graph.set_status(DAGStatus.RUNNING)
    request_id = str(uuid.uuid4())
    start_chunk = AudioChunkStart(
        request_id=request_id,
        audio_type="wav",
        node_name=a2f_node.name,
        dag=graph,
    )
    await adapter.feed_stream(start_chunk)
    audio_path = "input/test_audio.wav"
    # Body chunk 0
    audio_io = io.BytesIO()
    with open(audio_path, "rb") as reader:
        audio_io.write(reader.read())
    audio_io.seek(0)
    with wave.open(audio_io, "rb") as reader:
        duration = reader.getnframes() / reader.getframerate()
    audio_io.seek(0)
    body_chunk = AudioChunkBody(
        request_id=request_id,
        duration=duration,
        seq_number=0,
        audio_io=audio_io,
    )
    await adapter.feed_stream(body_chunk)
    # Body chunk 1
    audio_io = io.BytesIO()
    with open(audio_path, "rb") as reader:
        audio_io.write(reader.read())
    audio_io.seek(0)
    with wave.open(audio_io, "rb") as reader:
        duration = reader.getnframes() / reader.getframerate()
    audio_io.seek(0)
    body_chunk = AudioChunkBody(
        request_id=request_id,
        duration=duration,
        seq_number=1,
        audio_io=audio_io,
    )
    await adapter.feed_stream(body_chunk)
    end_chunk = AudioChunkEnd(
        request_id=request_id,
    )
    await adapter.feed_stream(end_chunk)
    start_time = time.time()
    while graph.status != DAGStatus.COMPLETED:
        await asyncio.sleep(0.1)
        if time.time() - start_time > 10:
            raise TimeoutError("A2F stream timeout")
    await adapter.interrupt()
    await profile.interrupt()
    await asyncio.sleep(adapter.sleep_time * 5)
    assert len(adapter.input_buffer) == 0
    assert len(profile.input_buffer) == 0


@pytest.mark.asyncio
async def test_audio2face_streaming_client_pcm():
    """Test audio2face streaming client with PCM audio format.

    Tests the streaming audio2face client using PCM audio data.
    Verifies that the client can process PCM audio segments and generate
    face animation data with reaction information.

    Raises:
        TimeoutError:
            If the A2F stream processing times out.
    """
    logger_cfg = dict(
        logger_name="test_audio2face_streaming_client_pcm", file_level=logging.DEBUG, logger_path="logs/pytest.log"
    )
    a2f_cfg = dict(
        type="Audio2FaceStreamingClient",
        verify=VERIFY,
        ws_url=WS_URL,
        timeout=10.0,
        logger_cfg=logger_cfg,
    )
    adapter = build_audio2face_adapter(a2f_cfg)
    asyncio.create_task(adapter.run())
    profile = FaceStreamProfile(mark_status_on_end=True, logger_cfg=logger_cfg)
    asyncio.create_task(profile.run())
    graph = DirectedAcyclicGraph(
        name="test_audio2face_streaming_client_pcm",
        conf=dict(
            avatar="KQ-default",
            response_chunk_n_frames=10,
        ),
        logger_cfg=logger_cfg,
    )
    a2f_node = DAGNode(
        name="a2f_node",
        payload=adapter,
    )
    graph.add_node(a2f_node)
    profile_node = DAGNode(
        name="profile_node",
        payload=profile,
    )
    graph.add_node(profile_node)
    graph.add_edge(a2f_node.name, profile_node.name)
    chunks = _wav_to_audio_with_reaction_chunks(
        input_wav_path="input/test_audio.wav",
        segment_duration=0.1,
        request_id=str(uuid.uuid4()),
        dag=graph,
        node_name=a2f_node.name,
    )
    graph.set_status(DAGStatus.RUNNING)
    for chunk in chunks:
        await adapter.feed_stream(chunk)
    start_time = time.time()
    while graph.status != DAGStatus.COMPLETED:
        await asyncio.sleep(0.1)
        if time.time() - start_time > 10:
            raise TimeoutError("A2F stream timeout")
    await adapter.interrupt()
    await profile.interrupt()
    await asyncio.sleep(adapter.sleep_time * 5)
    assert len(adapter.input_buffer) == 0
    assert len(profile.input_buffer) == 0
