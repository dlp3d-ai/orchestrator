import asyncio
import io
import os
import time
import uuid
import wave
from typing import List, Union

import numpy as np
import pytest
import yaml

from orchestrator.conversation.builder import build_conversation_adapter
from orchestrator.data_structures.audio_chunk import AudioChunkBody, AudioChunkEnd, AudioChunkStart
from orchestrator.data_structures.process_flow import DAGNode, DAGStatus, DirectedAcyclicGraph
from orchestrator.io.memory.mongodb_memory_client import MongoDBMemoryClient
from orchestrator.memory.memory_adapter import INITIAL_EMOTION_STATE
from orchestrator.memory.sensenova_omni_memory_client import SenseNovaOmniMemoryClient
from orchestrator.profile.audio_stream_profile import AudioStreamProfile
from orchestrator.utils.log import logging

# Test character ID for audio conversation testing
TEST_CHARACTER_ID = "88801af2-6d2e-48f0-a413-c0058a448a26"
# MongoDB connection configuration
MONGODB_HOST = "mongodb"
MONGODB_PORT = 27017
MONGODB_DB = "memory_test"
MONGODB_AUTH_DATABASE = "memory_test"
MONGODB_USER = "orchestrator"
MONGODB_PASSWORD = "orchestrator_password"


@pytest.fixture
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


@pytest.fixture(scope="session", autouse=True)
async def setup_test_memory():
    """Set up test memory before all tests start.

    Cleans existing memory and initializes test memory data for audio
    conversation testing.
    """
    # Create MongoDB client for test memory setup
    db_client = MongoDBMemoryClient(
        host=MONGODB_HOST,
        port=MONGODB_PORT,
        username=MONGODB_USER,
        password=MONGODB_PASSWORD,
        database=(MONGODB_DB or "character"),
        logger_cfg={"console_level": logging.DEBUG},
    )

    # Clean all existing memory data
    try:
        await db_client.remove_long_term_memory(TEST_CHARACTER_ID)
    except Exception:
        pass  # Ignore if memory doesn't exist

    try:
        await db_client.remove_medium_term_memory_not_after(TEST_CHARACTER_ID)
    except Exception:
        pass  # Ignore if memory doesn't exist

    try:
        await db_client.remove_chat_history_memory(TEST_CHARACTER_ID)
    except Exception:
        pass  # Ignore if memory doesn't exist

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

    # Add initial chat history
    for i, timestamp in enumerate(chat_timestamps):
        if i % 2 == 0:  # User messages
            await db_client.append_chat_history(
                character_id=TEST_CHARACTER_ID,
                unix_timestamp=timestamp,
                role="user",
                content=f"测试用户消息 {i+1}",
                relationship="Friend",
            )
        else:  # Assistant messages
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
    medium_term_timestamp = MongoDBMemoryClient.convert_unix_timestamp_to_str(chat_timestamps[0] - 1)
    last_short_term_timestamp = MongoDBMemoryClient.convert_unix_timestamp_to_str(chat_timestamps[-1])

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
        content="测试长期记忆：这是一个测试角色，用于验证音频对话功能。用户和助手之间建立了稳定的友谊关系。",
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


def _wav_to_chunks(
    input_wav_path: str, segment_duration: float, request_id: str, dag: DirectedAcyclicGraph, node_name: str
) -> List[Union[AudioChunkStart, AudioChunkBody, AudioChunkEnd]]:
    """Convert WAV file to audio chunks for testing.

    Splits a WAV file into smaller chunks for streaming audio conversation testing.
    Handles different audio formats and ensures proper byte ordering.

    Args:
        input_wav_path (str):
            Path to the input WAV file.
        segment_duration (float):
            Duration of each audio segment in seconds.
        request_id (str):
            Unique identifier for the test request.
        dag (DirectedAcyclicGraph):
            The DAG instance for the test.
        node_name (str):
            Name of the node in the DAG.

    Returns:
        List[Union[AudioChunkStart, AudioChunkBody, AudioChunkEnd]]:
            List of audio chunks for streaming.
    """
    ret_list: List[Union[AudioChunkStart, AudioChunkBody, AudioChunkEnd]] = list()
    with wave.open(input_wav_path, "rb") as wav_file:
        # Get audio parameters
        wf_n_channels = wav_file.getnchannels()
        wf_sample_width = wav_file.getsampwidth()
        wf_frame_rate = wav_file.getframerate()
        n_frames = wav_file.getnframes()

        start_chunk = AudioChunkStart(
            request_id=request_id,
            audio_type="pcm",
            node_name=node_name,
            dag=dag,
            n_channels=wf_n_channels,
            sample_width=wf_sample_width,
            frame_rate=wf_frame_rate,
        )
        ret_list.append(start_chunk)

        # Check byte order (little-endian)
        if wav_file.getcomptype() != "NONE":
            raise ValueError("Input audio must be uncompressed PCM")
        # Note: Endianness check might need more robust handling if issues arise

        # Calculate frames per segment
        frames_per_segment = int(wf_frame_rate * segment_duration)

        # Read audio data
        frames_data = wav_file.readframes(n_frames)

        # Convert byte data to numpy array, ensuring little-endian byte order
        if wf_sample_width == 2:  # 16-bit
            dtype = np.int16
        elif wf_sample_width == 4:  # 32-bit
            dtype = np.int32
        else:
            raise ValueError(f"Unsupported sample width: {wf_sample_width}")

        audio_data = np.frombuffer(frames_data, dtype=dtype)

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
            current_segment_duration = len(segment_data) / wf_frame_rate
            # Generate output file name
            output_pcm = io.BytesIO()
            # Save PCM data
            output_pcm.write(segment_data.tobytes())
            output_pcm.seek(0)
            body_chunk = AudioChunkBody(
                request_id=request_id,
                audio_io=output_pcm,
                duration=current_segment_duration,
                seq_number=i,
            )
            ret_list.append(body_chunk)
    end_chunk = AudioChunkEnd(
        request_id=request_id,
    )
    ret_list.append(end_chunk)
    return ret_list


agent_prompts_file_path = "configs/agent_prompts.yaml"
with open(agent_prompts_file_path, "r", encoding="utf-8") as file:
    agent_prompts = yaml.safe_load(file)


@pytest.mark.asyncio
async def test_openai_audio_client_stream_pcm_16khz(
    mongodb_memory_client: MongoDBMemoryClient,
):
    """Test OpenAI audio client streaming with 16kHz PCM audio.

    Tests the complete audio conversation pipeline using OpenAI's realtime API
    with 16kHz PCM audio input and validates the streaming functionality.

    Args:
        mongodb_memory_client (MongoDBMemoryClient):
            MongoDB client for memory operations during testing.
    """
    sensenova_ak = os.environ.get("SENSENOVA_AK")
    sensenova_sk = os.environ.get("SENSENOVA_SK")
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key or not sensenova_ak or not sensenova_sk:
        pytest.skip(
            "OPENAI_API_KEY or SENSENOVA_AK or SENSENOVA_SK not set, skipping test_openai_audio_client_stream_pcm_16khz"
        )

    logger_cfg = dict(
        logger_name="test_openai_audio_client_stream_pcm_16khz", file_level=logging.DEBUG, logger_path="logs/pytest.log"
    )

    test_memory_adapter = SenseNovaOmniMemoryClient(
        name="test_memory_adapter",
        db_client=mongodb_memory_client,
    )
    cascade_memories = await test_memory_adapter.db_client.get_cascade_memories(
        character_id=TEST_CHARACTER_ID,
    )

    audio_conv_cfg = dict(
        type="OpenAIAudioClient",
        name="openai_audio_client",
        agent_prompts_file=agent_prompts_file_path,
        wss_url="wss://api.openai.com/v1/realtime",  # Standard OpenAI Realtime URL
        proxy_url=os.environ.get("PROXY_URL", None),
        request_timeout=5,
        logger_cfg=logger_cfg,
    )

    adapter = build_conversation_adapter(audio_conv_cfg)
    asyncio.create_task(adapter.run())
    profile = AudioStreamProfile(mark_status_on_end=True, save_dir="output", logger_cfg=logger_cfg)
    asyncio.create_task(profile.run())

    graph = DirectedAcyclicGraph(
        name="test_openai_audio_client_stream",
        conf=dict(
            conversation_voice_name="alloy",  # Example voice
            language="en",
            character_id=TEST_CHARACTER_ID,
            user_settings=dict(
                sensenova_ak=sensenova_ak,
                sensenova_sk=sensenova_sk,
                openai_api_key=openai_api_key,
            ),
            user_prompt=agent_prompts["keqing_default"],
            cascade_memories=cascade_memories,
            relationship=("Lover", 100),
            emotion=INITIAL_EMOTION_STATE,
            memory_adapter=test_memory_adapter,
            memory_db_client=mongodb_memory_client,
        ),
        logger_cfg=logger_cfg,
    )

    audio_conv_node = DAGNode(
        name="audio_conv_node",
        payload=adapter,
    )
    profile_node = DAGNode(
        name="audio_profile_node",  # Changed from pcm_profile_node to avoid potential conflicts
        payload=profile,
    )
    graph.add_node(audio_conv_node)
    graph.add_node(profile_node)
    graph.add_edge(audio_conv_node.name, profile_node.name)

    request_id = str(uuid.uuid4())
    audio_path = "input/test_audio_16kHz.wav"
    chunks = _wav_to_chunks(audio_path, adapter.CHUNK_DURATION, request_id, graph, audio_conv_node.name)
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
            raise TimeoutError("OpenAI Audio Client stream timeout")
    await adapter.interrupt()
    await profile.interrupt()
    await asyncio.sleep(adapter.sleep_time * 5)
    assert len(adapter.input_buffer) == 0
    assert len(profile.input_buffer) == 0

    assert graph.status == DAGStatus.COMPLETED, f"Graph did not complete successfully. Final status: {graph.status}"


@pytest.mark.asyncio
async def test_openai_audio_client_stream_pcm_24khz(mongodb_memory_client: MongoDBMemoryClient):
    """Test OpenAI audio client streaming with 24kHz PCM audio.

    Tests the complete audio conversation pipeline using OpenAI's realtime API
    with 24kHz PCM audio input and validates the streaming functionality.

    Args:
        mongodb_memory_client (MongoDBMemoryClient):
            MongoDB client for memory operations during testing.
    """
    sensenova_ak = os.environ.get("SENSENOVA_AK")
    sensenova_sk = os.environ.get("SENSENOVA_SK")
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key or not sensenova_ak or not sensenova_sk:
        pytest.skip(
            "OPENAI_API_KEY or SENSENOVA_AK or SENSENOVA_SK not set, skipping test_openai_audio_client_stream_pcm_24khz"
        )

    logger_cfg = dict(
        logger_name="test_openai_audio_client_stream_pcm_24khz", file_level=logging.DEBUG, logger_path="logs/pytest.log"
    )

    test_memory_adapter = SenseNovaOmniMemoryClient(
        name="test_memory_adapter",
        db_client=mongodb_memory_client,
    )
    cascade_memories = await test_memory_adapter.db_client.get_cascade_memories(
        character_id=TEST_CHARACTER_ID,
    )

    audio_conv_cfg = dict(
        type="OpenAIAudioClient",
        name="openai_audio_client",
        agent_prompts_file=agent_prompts_file_path,
        wss_url="wss://api.openai.com/v1/realtime",  # Standard OpenAI Realtime URL
        proxy_url=os.environ.get("PROXY_URL", None),
        request_timeout=5,
        logger_cfg=logger_cfg,
    )
    adapter = build_conversation_adapter(audio_conv_cfg)
    asyncio.create_task(adapter.run())
    profile = AudioStreamProfile(mark_status_on_end=True, save_dir="output", logger_cfg=logger_cfg)
    asyncio.create_task(profile.run())

    graph = DirectedAcyclicGraph(
        name="test_openai_audio_client_stream",
        conf=dict(
            conversation_voice_name="alloy",  # Example voice
            language="en",
            character_id=TEST_CHARACTER_ID,
            user_settings=dict(
                sensenova_ak=sensenova_ak,
                sensenova_sk=sensenova_sk,
                openai_api_key=openai_api_key,
            ),
            user_prompt=agent_prompts["keqing_default"],
            cascade_memories=cascade_memories,
            relationship=("Lover", 100),
            emotion=INITIAL_EMOTION_STATE,
            memory_adapter=test_memory_adapter,
            memory_db_client=mongodb_memory_client,
        ),
        logger_cfg=logger_cfg,
    )

    audio_conv_node = DAGNode(
        name="audio_conv_node",
        payload=adapter,
    )
    profile_node = DAGNode(
        name="audio_profile_node",  # Changed from pcm_profile_node to avoid potential conflicts
        payload=profile,
    )
    graph.add_node(audio_conv_node)
    graph.add_node(profile_node)
    graph.add_edge(audio_conv_node.name, profile_node.name)

    request_id = str(uuid.uuid4())
    audio_path = "input/test_audio_24kHz.wav"
    chunks = _wav_to_chunks(audio_path, adapter.CHUNK_DURATION, request_id, graph, audio_conv_node.name)
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
        if time.time() - start_time > 30:
            raise TimeoutError("OpenAI Audio Client stream timeout")
    await adapter.interrupt()
    await profile.interrupt()
    await asyncio.sleep(adapter.sleep_time * 5)
    assert len(adapter.input_buffer) == 0
    assert len(profile.input_buffer) == 0

    assert graph.status == DAGStatus.COMPLETED, f"Graph did not complete successfully. Final status: {graph.status}"
