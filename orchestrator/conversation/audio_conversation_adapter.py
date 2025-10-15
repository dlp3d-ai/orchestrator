import asyncio
from abc import ABC, abstractmethod
from asyncio import QueueFull
from typing import Any, Dict, Union

import yaml

from ..data_structures.audio_chunk import AudioChunkBody, AudioChunkEnd, AudioChunkStart
from ..utils.exception import MissingAPIKeyException, failure_callback
from ..utils.log import setup_logger
from ..utils.streamable import ChunkWithoutStartError, Streamable


class AudioConversationAdapter(Streamable, ABC):
    """Base class for audio conversation adapters.

    This abstract base class provides the foundation for implementing audio-
    based conversation adapters that handle real-time audio input and output.
    It extends the Streamable interface and provides common functionality for
    audio processing pipelines.

    Audio conversation adapters do not handle reject classification results.
    """

    AVAILABLE_FOR_STREAM = True
    N_CHANNELS: int = 1
    SAMPLE_WIDTH: int = 2
    FRAME_RATE: int = 16000

    def __init__(
        self,
        name: str,
        agent_prompts_file: str,
        proxy_url: Union[None, str] = None,
        n_workers: int = 1,
        request_timeout: float = 20.0,
        queue_size: int = 100,
        sleep_time: float = 0.01,
        clean_interval: float = 10.0,
        expire_time: float = 120.0,
        logger_cfg: Union[None, Dict[str, Any]] = None,
    ):
        """Initialize the audio conversation adapter.

        Sets up the base configuration for audio conversation processing
        including audio format parameters, queue management, and logging.

        Args:
            name (str):
                The name of the conversation adapter.
            agent_prompts_file (str):
                The path to the agent prompts file containing conversation instructions.
            proxy_url (Union[None, str], optional):
                The proxy URL for the conversation connection.
                Defaults to None.
            n_workers (int, optional):
                The number of worker threads for the conversation processing.
                Defaults to 1.
            sleep_time (float, optional):
                The sleep interval between operations in seconds.
                Defaults to 0.01.
            clean_interval (float, optional):
                The interval to clean expired requests in seconds.
                Defaults to 10.0.
            expire_time (float, optional):
                The time after which requests expire in seconds.
                Defaults to 120.0.
            logger_cfg (Union[None, Dict[str, Any]], optional):
                Logger configuration dictionary. Defaults to None.
        """
        Streamable.__init__(
            self,
            queue_size=queue_size,
            sleep_time=sleep_time,
            clean_interval=clean_interval,
            expire_time=expire_time,
            logger_cfg=logger_cfg,
        )
        ABC.__init__(self)
        self.name = name
        self.logger_cfg["logger_name"] = name
        self.logger = setup_logger(**self.logger_cfg)

        with open(agent_prompts_file, "r", encoding="utf-8") as file:
            self.agent_prompts = yaml.safe_load(file)
        self.proxy_url = proxy_url
        self.request_timeout = request_timeout

    async def feed_stream(
        self,
        chunk: Union[AudioChunkStart, AudioChunkBody, AudioChunkEnd],
    ) -> None:
        """Feed audio chunk to the conversation adapter.

        Adds audio chunks to the processing queue for real-time processing.
        Handles different types of audio chunks (start, body, end).

        Args:
            chunk (Union[AudioChunkStart, AudioChunkBody, AudioChunkEnd]):
                The audio chunk to process.
        """
        try:
            self.queue.put_nowait(chunk)
        except QueueFull as e:
            msg = "The queue is full"
            self.logger.error(msg)
            raise e

    async def _handle_start(
        self,
        chunk: AudioChunkStart,
        cur_time: float,
    ) -> None:
        """Handle the start of an audio stream.

        Initializes a new conversation session and validates audio format.
        Sets up the input buffer with conversation parameters and starts
        the session creation process.

        Args:
            chunk (AudioChunkStart):
                The start chunk containing audio format information.
            cur_time (float):
                Current timestamp for the operation.
        """
        request_id = chunk.request_id
        if chunk.audio_type == "pcm":
            if chunk.n_channels != 1:
                msg = (
                    "Only mono audio is supported for audio conversation, "
                    + f"but got {chunk.n_channels} channels for request {request_id}."
                )
                self.logger.error(msg)
                raise NotImplementedError(msg)
            if chunk.sample_width != 2:
                msg = (
                    "Only 16-bit audio is supported for audio conversation, "
                    + f"but got {chunk.sample_width} bytes for request {request_id}."
                )
                self.logger.error(msg)
                raise NotImplementedError(msg)
        conf = chunk.dag.conf
        dag_start_time = conf.get("start_time", None)
        memory_adapter = conf.get("memory_adapter")
        memory_db_client = conf.get("memory_db_client")
        conversation_model_override = conf.get("conversation_model_override", None)
        api_keys = conf.get("user_settings", {})
        voice_name = conf["conversation_voice_name"]
        language = conf.get("language", "zh")
        character_id = conf.get("character_id", None)
        profile_memory = conf.get("profile_memory", None)
        cascade_memories = conf.get("cascade_memories", None)
        emotion = conf.get("emotion")
        relationship = conf.get("relationship")
        user_prompt = conf.get("user_prompt")
        callback_bytes_fn = chunk.dag.conf.get("callback_bytes_fn", None)
        self.input_buffer[request_id] = {
            "dag_start_time": dag_start_time,
            "start_time": cur_time,
            "last_update_time": cur_time,
            "voice_name": voice_name,
            "language": language,
            "memory_adapter": memory_adapter,
            "memory_db_client": memory_db_client,
            "conversation_model_override": conversation_model_override,
            "api_keys": api_keys,
            "character_id": character_id,
            "profile_memory": profile_memory,
            "cascade_memories": cascade_memories,
            "relationship": relationship,
            "user_prompt": user_prompt,
            "emotion": emotion,
            "dag": chunk.dag,
            "node_name": chunk.node_name,
            "audio_type": chunk.audio_type,
            "downstream_warned": False,
            "chunk_received": 0,
            "chunk_sent": 0,
            "input_chunk_received": 0,
            "input_chunk_sent": 0,
            "callback_bytes_fn": callback_bytes_fn,
        }
        if chunk.audio_type == "pcm":
            self.input_buffer[request_id]["n_channels"] = chunk.n_channels
            self.input_buffer[request_id]["sample_width"] = chunk.sample_width
            self.input_buffer[request_id]["frame_rate"] = chunk.frame_rate
        task = asyncio.create_task(self._create_session(request_id, cascade_memories, voice_name))
        task.add_done_callback(lambda t: self._handle_init_task_exception(t, request_id))

    def _handle_init_task_exception(self, task: asyncio.Task, request_id: str) -> None:
        """Handle exceptions from the initialization task.

        Args:
            task (asyncio.Task): The completed task.
            request_id (str): The request ID associated with the task.
        """
        if task.exception() is not None:
            exception = task.exception()
            if isinstance(exception, MissingAPIKeyException):
                msg = f"Missing API key during LLM audio conversation client initialization: {exception}"
                self.logger.error(msg)
                # Create an async task to handle the failure callback
                asyncio.create_task(self._send_failure_callback(msg, request_id))
            else:
                msg = f"Unexpected error during LLM audio conversation client initialization: {exception}"
                self.logger.error(msg)
                # Create an async task to handle the failure callback for other exceptions too
                asyncio.create_task(self._send_failure_callback(f"Unexpected error: {exception}", request_id))

    async def _send_failure_callback(self, msg: str, request_id: str) -> None:
        """Send failure callback asynchronously.

        Args:
            msg (str): The error message to send.
            request_id (str): The request ID to get the callback function.
        """
        try:
            if request_id in self.input_buffer:
                callback_bytes_fn = self.input_buffer[request_id].get("callback_bytes_fn")
                if callback_bytes_fn:
                    await failure_callback(msg, callback_bytes_fn)
            else:
                self.logger.warning(f"Request {request_id} not found in input buffer")
        except Exception as e:
            self.logger.error(f"Failed to send failure callback for request {request_id}: {e}")

    async def _handle_body(
        self,
        chunk: AudioChunkBody,
        cur_time: float,
    ) -> None:
        """Handle audio data chunks during streaming.

        Processes incoming audio data and forwards it to the conversation
        service for real-time processing.

        Args:
            chunk (AudioChunkBody):
                The audio data chunk containing PCM audio bytes.
            cur_time (float):
                Current timestamp for the operation.
        """
        request_id = chunk.request_id
        if request_id not in self.input_buffer:
            msg = f"Request {request_id} not found in input buffer, but received a body message."
            self.logger.error(msg)
            raise ChunkWithoutStartError(msg)
        self.input_buffer[request_id]["last_update_time"] = cur_time
        seq_number = self.input_buffer[request_id]["input_chunk_received"]
        self.input_buffer[request_id]["input_chunk_received"] += 1
        audio_bytes = chunk.audio_io.read()
        asyncio.create_task(self._send_audio(request_id, audio_bytes, seq_number))

    async def _handle_end(
        self,
        chunk: AudioChunkEnd,
        cur_time: float,
    ) -> None:
        """Handle the end of an audio stream.

        Signals the completion of audio input and triggers the final
        processing and response generation.

        Args:
            chunk (AudioChunkEnd):
                The end chunk signaling stream completion.
            cur_time (float):
                Current timestamp for the operation.
        """
        request_id = chunk.request_id
        if request_id not in self.input_buffer:
            msg = f"Request {request_id} not found in input buffer, but received an end message."
            self.logger.error(msg)
            raise ChunkWithoutStartError(msg)
        self.input_buffer[request_id]["last_update_time"] = cur_time
        asyncio.create_task(self._commit_audio(request_id))

    @abstractmethod
    async def _create_session(
        self,
        request_id: str,
        cascade_memories: Union[None, Dict[str, Any]],
        voice_name: str,
    ) -> None:
        """Create a session with the audio conversation service.

        Establishes a connection to the underlying audio conversation
        service and initializes the session with conversation context.

        Args:
            request_id (str):
                Unique identifier for the conversation request.
            cascade_memories (Union[None, Dict[str, Any]]):
                Memory context from previous conversations, or None if no context.
            voice_name (str):
                The voice to use for audio generation.
        """
        raise NotImplementedError

    @abstractmethod
    async def _send_audio(self, request_id: str, audio_bytes: bytes, seq_number: int) -> None:
        """Send audio data to the conversation service.

        Processes and sends audio chunks to the conversation service
        for real-time processing.

        Args:
            request_id (str):
                Unique identifier for the conversation request.
            audio_bytes (bytes):
                Raw audio data bytes to send.
            seq_number (int):
                Sequence number of the audio chunk for ordering.
        """
        raise NotImplementedError

    @abstractmethod
    async def _commit_audio(self, request_id: str) -> None:
        """Commit audio input to the conversation service.

        Signals the end of audio input and triggers the AI response
        generation process.

        Args:
            request_id (str):
                Unique identifier for the conversation request.
        """
        raise NotImplementedError

    @abstractmethod
    async def _receive_pcm(self, request_id: str) -> None:
        """Receive audio response from the conversation service.

        Handles incoming audio responses and processes them for
        downstream consumption.

        Args:
            request_id (str):
                Unique identifier for the conversation request.
        """
        raise NotImplementedError

    @abstractmethod
    async def _close_session(self, request_id: str) -> None:
        """Close the session with the conversation service.

        Terminates the connection to the conversation service and
        performs necessary cleanup operations.

        Args:
            request_id (str):
                Unique identifier for the conversation request.
        """
        raise NotImplementedError

    @abstractmethod
    async def _send_stream_start_task(self, request_id: str) -> None:
        """Send stream start signal to downstream nodes.

        Notifies downstream processing nodes that audio streaming
        has begun and provides necessary metadata.

        Args:
            request_id (str):
                Unique identifier for the conversation request.
        """
        raise NotImplementedError
