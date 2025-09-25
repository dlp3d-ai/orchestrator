import asyncio
import io
import ssl
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Literal, Tuple, Union

import websockets
from pydantic import BaseModel

from ...data_structures import speech2motion_v3_pb2 as s2m_pb2
from ...data_structures.audio_chunk import (
    AudioWithSubtitleChunkBody,
    AudioWithSubtitleChunkEnd,
    AudioWithSubtitleChunkStart,
)
from ...data_structures.motion_chunk import MotionChunkBody, MotionChunkEnd, MotionChunkStart
from ...data_structures.process_flow import DAGStatus
from ...utils.executor_registry import ExecutorRegistry
from ...utils.streamable import ChunkWithoutStartError
from .speech2motion_adapter import Speech2MotionAdapter


class StreamingSpeech2MotionV2ChunkStart(BaseModel):
    """V2 streaming motion generation start request.

    Initializes a streaming motion generation session with user information,
    avatar selection, and motion generation parameters.

    Args:
        request_id (str):
            Request ID for identifying an independent streaming motion generation request.
        user_id (str):
            User ID for distinguishing request users and accessing related memory.
        avatar (str):
            User-selected avatar for motion generation.
        max_front_extension_duration (float, optional):
            Maximum additional duration that generated motion can extend before
            the specified duration. Defaults to 0.0.
        max_rear_extension_duration (float, optional):
            Maximum additional duration that generated motion can extend after
            the specified duration. Defaults to 0.0.
        memory_duration_override (float | None, optional):
            Memory duration in seconds. Defaults to None, uses default
            `memory_duration` parameter.
        first_body_fast_response_override (bool | None, optional):
            Whether to override the fast response configuration on first Body request.
            Defaults to None, uses default `first_body_fast_response` parameter.
        idle_long_extendable (bool, optional):
            Whether to allow extending motion duration during idle time.
            Defaults to False.
        return_content (Literal["motion_clip", "log"], optional):
            Type of content to return. Defaults to "motion_clip", returns motion clips.
            If set to "log", returns log information.
        response_chunk_n_frames (int | None, optional):
            Number of frames per response chunk. Defaults to None, uses default
            `response_chunk_n_frames` parameter.
    """

    request_id: str
    user_id: str
    avatar: str
    max_front_extension_duration: float = 0.0
    max_rear_extension_duration: float = 0.0
    memory_duration_override: float | None = None
    first_body_fast_response_override: bool | None = None
    idle_long_extendable: bool = False
    return_content: Literal["motion_clip", "log"] = "motion_clip"
    response_chunk_n_frames: int | None = None


class StreamingSpeech2MotionV2ChunkBody(BaseModel):
    """V2 streaming motion generation body request.

    Processes speech text and motion keywords to generate corresponding motion segments.

    Args:
        request_id (str):
            Request ID for identifying an independent streaming motion generation request.
        duration (float):
            Duration of the current motion generation in seconds.
        speech_text (str):
            Current speech text content.
        sequence_number (int):
            Sequence number of the current request among all Body requests, starting from 0.
        speech_time (list[tuple[int, float]] | None, optional):
            List of tuples containing character index and start time in seconds.
            Defaults to None, meaning each character takes the same time.
        motion_keywords (list[tuple[int, str]] | None, optional):
            List of tuples containing starting character index and motion description keywords.
            Defaults to None.
    """

    request_id: str
    duration: float
    speech_text: str
    sequence_number: int
    speech_time: list[tuple[int, float]] | None = None
    motion_keywords: list[tuple[int, str]] | None = None


class StreamingSpeech2MotionV2ChunkEnd(BaseModel):
    """V2 streaming motion generation end request.

    Signals the end of a streaming motion generation session and processes
    remaining extension motions.

    Args:
        request_id (str):
            Request ID for identifying an independent streaming motion generation request.
    """

    request_id: str


class StreamingSpeech2MotionV3ChunkStart(BaseModel):
    """V3 streaming motion generation start request.

    Initializes a streaming motion generation session with user information,
    avatar selection, and application configuration.

    Args:
        request_id (str):
            Request ID for identifying an independent streaming motion generation request.
        user_id (str):
            User ID for distinguishing request users and accessing related memory.
        avatar (str):
            User-selected avatar for motion generation.
        app_name (Literal["babylon", "python_backend"], optional):
            Application name for the motion generation service.
            Defaults to "python_backend".
        max_front_extension_duration (float, optional):
            Maximum additional duration that generated motion can extend before
            the specified duration. Defaults to 0.0.
        max_rear_extension_duration (float, optional):
            Maximum additional duration that generated motion can extend after
            the specified duration. Defaults to 0.0.
        memory_duration_override (float | None, optional):
            Memory duration in seconds. Defaults to None.
    """

    request_id: str
    user_id: str
    avatar: str
    app_name: Literal["babylon", "python_backend"] = "python_backend"
    max_front_extension_duration: float = 0.0
    max_rear_extension_duration: float = 0.0
    memory_duration_override: float | None = None


class StreamingSpeech2MotionV3ChunkBody(BaseModel):
    """V3 streaming motion generation body request.

    Processes speech text and motion keywords to generate corresponding motion segments.

    Args:
        request_id (str):
            Request ID for identifying an independent streaming motion generation request.
        duration (float):
            Duration of the current motion generation in seconds.
        speech_text (str):
            Current speech text content.
        sequence_number (int):
            Sequence number of the current request among all Body requests, starting from 0.
        speech_time (list[tuple[int, float]] | None, optional):
            List of tuples containing character index and start time in seconds.
            Defaults to None.
        motion_keywords (list[tuple[int, str]] | None, optional):
            List of tuples containing starting character index and motion description keywords.
            Defaults to None.
        label_expression (str | None, optional):
            Label expression for motion filtering. Defaults to None.
    """

    request_id: str
    duration: float
    speech_text: str
    sequence_number: int
    speech_time: list[tuple[int, float]] | None = None
    motion_keywords: list[tuple[int, str]] | None = None
    label_expression: str | None = None


class StreamingSpeech2MotionV3ChunkEnd(BaseModel):
    """V3 streaming motion generation end request.

    Signals the end of a streaming motion generation session and processes
    remaining extension motions.

    Args:
        request_id (str):
            Request ID for identifying an independent streaming motion generation request.
    """

    request_id: str


class Speech2MotionStreamingClient(Speech2MotionAdapter):
    """Streaming client for speech-to-motion generation using WebSocket
    connection.

    This client handles real-time speech text streaming to generate motion
    animation data through WebSocket communication. It supports both V2 and V3
    protocols and can process speech with motion keywords and timing
    information.
    """

    _ssl_context_cache: Union[None, ssl.SSLContext] = None
    AVAILABLE_FOR_STREAM = True
    ExecutorRegistry.register_class("Speech2MotionStreamingClient")

    def __init__(
        self,
        ws_url: str,
        api_key: Union[None, str] = None,
        verify: bool = True,
        timeout: float = 20.0,
        queue_size: int = 100,
        sleep_time: float = 0.01,
        clean_interval: float = 10.0,
        expire_time: float = 120.0,
        max_workers: int = 1,
        thread_pool_executor: ThreadPoolExecutor | None = None,
        logger_cfg: Union[None, Dict[str, Any]] = None,
    ):
        """Initialize the speech2motion streaming client.

        Args:
            ws_url (str):
                WebSocket URL of the speech2motion service.
            api_key (Union[None, str], optional):
                API key for authentication with the speech2motion service.
                Defaults to None.
            verify (bool, optional):
                Whether to verify SSL certificates for secure connections.
                Defaults to True.
            timeout (float, optional):
                Timeout in seconds for WebSocket operations.
                Defaults to 20.0.
            queue_size (int, optional):
                Maximum size of the internal processing queue.
                Defaults to 100.
            sleep_time (float, optional):
                Sleep interval in seconds between processing cycles.
                Defaults to 0.01.
            clean_interval (float, optional):
                Interval in seconds for cleaning expired requests.
                Defaults to 10.0.
            expire_time (float, optional):
                Time in seconds after which requests expire.
                Defaults to 120.0.
            max_workers (int, optional):
                Maximum number of worker threads for processing.
                Defaults to 1.
            thread_pool_executor (ThreadPoolExecutor | None, optional):
                External thread pool executor to use. If None, creates a new one.
                Defaults to None.
            logger_cfg (Union[None, Dict[str, Any]], optional):
                Logger configuration dictionary.
                Defaults to None.
        """
        Speech2MotionAdapter.__init__(
            self,
            queue_size=queue_size,
            sleep_time=sleep_time,
            clean_interval=clean_interval,
            expire_time=expire_time,
            logger_cfg=logger_cfg,
        )
        self.ws_url = ws_url
        self.api_key = api_key
        self.verify = verify
        self.timeout = timeout
        self.executor = (
            thread_pool_executor if thread_pool_executor is not None else ThreadPoolExecutor(max_workers=max_workers)
        )
        self.executor_external = True if thread_pool_executor is not None else False

    def __del__(self) -> None:
        """Destructor for cleaning up thread pool executor.

        Shuts down the internal thread pool executor if it was created by this
        instance (not provided externally).
        """
        if not self.executor_external:
            self.executor.shutdown(wait=True)

    async def generate_speech2motion(
        self,
        duration: float,
        speech_text: str,
        avatar: str,
        speech_time: Union[List[Tuple[int, float]], None] = None,
        motion_keywords: Union[List[Tuple[float, str]], None] = None,
        user_id: Union[str, None] = None,
        max_front_extension_duration: float = 0.0,
        max_rear_extension_duration: float = 0.0,
        first_body_fast_response_override: bool = False,
        request_id: Union[str, None] = None,
        **kwargs: Any,
    ) -> io.BytesIO:
        """Generate motion animation from speech text (non-streaming mode).

        Note: This method is not implemented as this client only supports
        streaming mode. Use the streaming interface instead.

        Args:
            duration (float):
                Total duration of the motion in seconds.
            speech_text (str):
                Speech text content for motion generation.
            avatar (str):
                Avatar name for motion generation.
            speech_time (Union[List[Tuple[int, float]], None], optional):
                List of tuples containing character index and start time in seconds.
                Defaults to None, every character takes the same time.
            motion_keywords (Union[List[Tuple[float, str]], None], optional):
                List of tuples containing start time in seconds and motion keywords.
                Defaults to None.
            user_id (Union[str, None], optional):
                User ID for memory filtering. Defaults to None.
            max_front_extension_duration (float, optional):
                Maximum additional duration that generated motion can extend before
                the specified duration. Defaults to 0.0.
            max_rear_extension_duration (float, optional):
                Maximum additional duration that generated motion can extend after
                the specified duration. Defaults to 0.0.
            first_body_fast_response_override (bool, optional):
                Whether to override fast response configuration on first Body request.
                Defaults to False.
            request_id (Union[str, None], optional):
                Unique identifier for request tracking and logging.
                Defaults to None.
            **kwargs (Any):
                Additional arguments passed to the speech2motion adapter.

        Returns:
            io.BytesIO:
                Generated motion animation data.

        Raises:
            NotImplementedError:
                This method is not supported in streaming mode.
        """
        # TODO: Support non-streaming generation when needed
        msg = "Non-streaming speech2motion generation is not supported yet."
        self.logger.error(msg)
        raise NotImplementedError(msg)

    async def _handle_start(self, chunk: AudioWithSubtitleChunkStart, cur_time: float) -> None:
        """Handle start chunks from audio streams.

        Initializes the input buffer for a new request and sets up tracking
        for speech-to-motion processing. Immediately starts the stream task.

        Args:
            chunk (AudioWithSubtitleChunkStart):
                Start chunk from audio stream.
            cur_time (float):
                Current timestamp for tracking.
        """
        dag = chunk.dag
        conf = dag.conf
        request_id = chunk.request_id
        avatar = conf["avatar"]
        origin_seed = conf.get("origin_seed", None)
        character_id = conf.get("character_id", None)
        max_front_extension_duration = conf.get("max_front_extension_duration", 0.0)
        max_rear_extension_duration = conf.get("max_rear_extension_duration", 0.0)
        first_body_fast_response_override = conf.get("first_body_fast_response", False)
        idle_long_extendable = conf.get("idle_long_extendable", False)
        response_chunk_n_frames = conf.get("response_chunk_n_frames", 10)
        app_name = conf.get("app_name", "python_backend")
        label_expression = conf.get("label_expression", None)
        self.input_buffer[request_id] = {
            "start_time": cur_time,
            "last_update_time": cur_time,
            "dag": dag,
            "avatar": avatar,
            "origin_seed": origin_seed,
            "character_id": character_id,
            "max_front_extension_duration": max_front_extension_duration,
            "max_rear_extension_duration": max_rear_extension_duration,
            "first_body_fast_response_override": first_body_fast_response_override,
            "idle_long_extendable": idle_long_extendable,
            "node_name": chunk.node_name,
            "first_chunk_handle_time": None,
            "chunk_received_from_upstream": 0,
            "chunk_sent_to_server": 0,
            "chunk_received_from_server": 0,
            "downstream_warned": False,
            "response_chunk_n_frames": response_chunk_n_frames,
            "app_name": app_name,
            "ws_client": None,
            "global_label_expression": label_expression,
        }
        asyncio.create_task(self._start_stream_task(request_id))

    async def _handle_body(self, chunk: AudioWithSubtitleChunkBody, cur_time: float) -> None:
        """Handle body chunks from audio streams.

        Processes speech text chunks and sends them to the WebSocket server.
        Extracts motion keywords and label expressions if available.

        Args:
            chunk (AudioWithSubtitleChunkBody):
                Body chunk from audio stream.
            cur_time (float):
                Current timestamp for tracking.
        """
        request_id = chunk.request_id
        if request_id not in self.input_buffer:
            msg = f"Request {request_id} not found in input buffer, but received a body message."
            self.logger.error(msg)
            raise ChunkWithoutStartError(msg)
        self.input_buffer[request_id]["last_update_time"] = cur_time
        seq_number = self.input_buffer[request_id]["chunk_received_from_upstream"]
        self.input_buffer[request_id]["chunk_received_from_upstream"] += 1
        dag = self.input_buffer[request_id]["dag"]
        if dag.status == DAGStatus.RUNNING:
            if len(chunk.speech_time) == 0:
                speech_time = None
            else:
                speech_time = chunk.speech_time
            chunk_class_str = chunk.__class__.__name__
            if chunk_class_str == "AudioWithReactionChunkBody":
                motion_keywords = chunk.reaction.motion_keywords
                label_expression = chunk.reaction.label_expression
            else:
                motion_keywords = None
                label_expression = None
            asyncio.create_task(
                self._send_body_request_task(
                    request_id=request_id,
                    duration=chunk.duration,
                    speech_text=chunk.speech_text,
                    seq_number=seq_number,
                    speech_time=speech_time,
                    motion_keywords=motion_keywords,
                    label_expression=label_expression,
                )
            )

    async def _handle_end(self, chunk: AudioWithSubtitleChunkEnd, cur_time: float) -> None:
        """Handle end chunks from audio streams.

        Signals the end of speech stream processing and triggers cleanup tasks.

        Args:
            chunk (AudioWithSubtitleChunkEnd):
                End chunk from audio stream.
            cur_time (float):
                Current timestamp for tracking.
        """
        request_id = chunk.request_id
        if request_id not in self.input_buffer:
            msg = f"Request {request_id} not found in input buffer, but received an end message."
            self.logger.error(msg)
            raise ChunkWithoutStartError(msg)
        self.input_buffer[request_id]["last_update_time"] = cur_time
        # Trigger last task when received end chunk
        dag = self.input_buffer[request_id]["dag"]
        if dag.status == DAGStatus.RUNNING:
            asyncio.create_task(self._send_stream_end_task(request_id))

    async def _start_stream_task(
        self,
        request_id: str,
    ) -> None:
        """Start WebSocket stream for speech2motion processing.

        Establishes WebSocket connection and sends initial configuration
        to the speech2motion server, then starts receiving responses.

        Args:
            request_id (str):
                Unique identifier for the request.
        """
        loop = asyncio.get_running_loop()
        if self.verify:
            ssl_context = None
        else:
            if self.__class__._ssl_context_cache is None:
                ssl_context = await loop.run_in_executor(self.executor, ssl.create_default_context)
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_NONE
                self.__class__._ssl_context_cache = ssl_context
            else:
                ssl_context = self.__class__._ssl_context_cache
        ws_client = await websockets.connect(
            self.ws_url,
            ssl=ssl_context,
            ping_timeout=self.timeout,
            close_timeout=self.timeout,
        )
        chunk_start = s2m_pb2.Speech2MotionV3Request()
        if self.ws_url.endswith("/v2/streaming_speech2motion/ws"):
            chunk_start.class_name = "StreamingSpeech2MotionV2ChunkStart"
            chunk_start.return_content = "motion_clip"
            chunk_start.idle_long_extendable = self.input_buffer[request_id]["idle_long_extendable"]
        else:
            chunk_start.class_name = "StreamingSpeech2MotionV3ChunkStart"
        chunk_start.request_id = request_id
        chunk_start.user_id = self.input_buffer[request_id]["character_id"]
        chunk_start.avatar = self.input_buffer[request_id]["avatar"]
        chunk_start.max_front_extension_duration = self.input_buffer[request_id]["max_front_extension_duration"]
        chunk_start.max_rear_extension_duration = self.input_buffer[request_id]["max_rear_extension_duration"]
        chunk_start.response_chunk_n_frames_value = self.input_buffer[request_id]["response_chunk_n_frames"]
        chunk_start.app_name = self.input_buffer[request_id]["app_name"]
        pb_bytes = await loop.run_in_executor(self.executor, chunk_start.SerializeToString)
        await ws_client.send(pb_bytes)
        self.input_buffer[request_id]["ws_client"] = ws_client
        asyncio.create_task(self._receive_stream_task(request_id))

    async def _send_body_request_task(
        self,
        request_id: str,
        duration: float,
        speech_text: str,
        seq_number: int,
        speech_time: Union[List[Tuple[int, float]], None] = None,
        motion_keywords: Union[List[Tuple[int, str]], None] = None,
        label_expression: Union[str, None] = None,
    ) -> None:
        """Send speech text data to WebSocket server for processing.

        Handles speech text chunks with timing information and motion keywords,
        sends them to the server for motion generation.

        Args:
            request_id (str):
                Unique identifier for the request.
            duration (float):
                Duration of the speech segment in seconds.
            speech_text (str):
                Speech text content for motion generation.
            seq_number (int):
                Sequence number for ordering speech chunks.
            speech_time (Union[List[Tuple[int, float]], None], optional):
                List of tuples containing character index and start time in seconds.
                Defaults to None.
            motion_keywords (Union[List[Tuple[int, str]], None], optional):
                List of tuples containing character index and motion keywords.
                motion_keywords[i][1] is the keyword,
                motion_keywords[i][0] is the character index in speech_text
                that the motion should align with. Defaults to None.
            label_expression (Union[str, None], optional):
                Label expression for limiting motion range. Defaults to None,
                meaning no restriction.
        """
        if seq_number == 0:
            self.input_buffer[request_id]["first_chunk_handle_time"] = time.time()
        try:
            ws_client = self.input_buffer[request_id].get("ws_client", None)
            while ws_client is None:
                await asyncio.sleep(self.sleep_time)
                ws_client = self.input_buffer[request_id].get("ws_client", None)
            dag = self.input_buffer[request_id]["dag"]
            while self.input_buffer[request_id]["chunk_sent_to_server"] < seq_number:
                if dag.status == DAGStatus.RUNNING:
                    await asyncio.sleep(self.sleep_time)
                else:
                    msg = f"Streaming audio sending interrupted by DAG status {dag.status}"
                    msg = msg + f" for request {request_id}"
                    self.logger.warning(msg)
                    return
            chunk_body = s2m_pb2.Speech2MotionV3Request()
            if self.ws_url.endswith("/v2/streaming_speech2motion/ws"):
                chunk_body.class_name = "StreamingSpeech2MotionV2ChunkBody"
            else:
                chunk_body.class_name = "StreamingSpeech2MotionV3ChunkBody"
            chunk_body.request_id = request_id
            chunk_body.duration = duration
            chunk_body.speech_text = speech_text
            chunk_body.sequence_number = seq_number
            if speech_time is not None:
                for char_index, char_time in speech_time:
                    pb_char_time = s2m_pb2.SpeechTime()
                    pb_char_time.char_index = char_index
                    pb_char_time.start_time = char_time
                    chunk_body.speech_time.append(pb_char_time)
            if motion_keywords is not None:
                for start_char_index, keyword in motion_keywords:
                    pb_motion_keyword = s2m_pb2.MotionKeyword()
                    pb_motion_keyword.start_char_index = start_char_index
                    pb_motion_keyword.keyword = keyword
                    chunk_body.motion_keywords.append(pb_motion_keyword)
            if label_expression is not None:
                chunk_body.label_expression = label_expression
            elif self.input_buffer[request_id].get("global_label_expression", None) is not None:
                chunk_body.label_expression = self.input_buffer[request_id]["global_label_expression"]
            loop = asyncio.get_running_loop()
            data = await loop.run_in_executor(self.executor, chunk_body.SerializeToString)
            await ws_client.send(data)
            self.input_buffer[request_id]["chunk_sent_to_server"] += 1
        except Exception as e:
            msg = f"Error in streaming speech2motion generation: {e}"
            msg = msg + f" for request {request_id}"
            traceback_str = traceback.format_exc()
            msg += f"\n{traceback_str}"
            self.logger.error(msg)
            dag = self.input_buffer[request_id]["dag"]
            dag.set_status(DAGStatus.FAILED)

    async def _send_stream_end_task(self, request_id: str) -> None:
        """Send stream end signal to WebSocket server.

        Waits for all pending speech chunks to be sent, then sends
        the end signal to complete the speech stream.

        Args:
            request_id (str):
                Unique identifier for the request.
        """
        dag = self.input_buffer[request_id]["dag"]
        ws_client = self.input_buffer[request_id].get("ws_client", None)
        while ws_client is None:
            await asyncio.sleep(self.sleep_time)
            ws_client = self.input_buffer[request_id].get("ws_client", None)
        while (
            self.input_buffer[request_id]["chunk_received_from_upstream"]
            > self.input_buffer[request_id]["chunk_sent_to_server"]
        ):
            if dag.status == DAGStatus.RUNNING:
                await asyncio.sleep(self.sleep_time)
            else:
                msg = f"Streaming end signal sending interrupted by DAG status {dag.status}"
                msg += f" for request {request_id}"
                self.logger.warning(msg)
                return
        chunk_end = s2m_pb2.Speech2MotionV3Request()
        if self.ws_url.endswith("/v2/streaming_speech2motion/ws"):
            chunk_end.class_name = "StreamingSpeech2MotionV2ChunkEnd"
        else:
            chunk_end.class_name = "StreamingSpeech2MotionV3ChunkEnd"
        chunk_end.request_id = request_id
        loop = asyncio.get_running_loop()
        data = await loop.run_in_executor(self.executor, chunk_end.SerializeToString)
        await ws_client.send(data)

    async def _receive_stream_task(self, request_id: str) -> None:
        """Receive motion animation data from WebSocket server.

        Continuously receives motion data from the server and forwards
        it to downstream nodes. Handles start, body, and end chunks, and
        manages WebSocket connection lifecycle.

        Args:
            request_id (str):
                Unique identifier for the request.
        """
        dag = self.input_buffer[request_id]["dag"]
        node_name = self.input_buffer[request_id]["node_name"]
        dag_node = dag.get_node(node_name)
        downstream_nodes = dag_node.downstreams
        downstream_warned = self.input_buffer[request_id]["downstream_warned"]
        if len(downstream_nodes) == 0 and not downstream_warned:
            self.logger.warning(f"Request {request_id} has no downstreams, so the result is discarded.")
            self.input_buffer[request_id]["downstream_warned"] = True
        ws_client = self.input_buffer[request_id].get("ws_client", None)
        while ws_client is None:
            await asyncio.sleep(self.sleep_time)
            ws_client = self.input_buffer[request_id].get("ws_client", None)
        dag_start_time = self.input_buffer[request_id]["dag"].conf.get("start_time", None)
        chunk_end_received = False
        loop = asyncio.get_running_loop()
        while not chunk_end_received:
            if dag.status != DAGStatus.RUNNING:
                msg = f"Streaming receiving interrupted by DAG status {dag.status} for request {request_id}"
                self.logger.warning(msg)
                break
            try:
                response = await ws_client.recv()
                coroutines = list()
                resp = s2m_pb2.Speech2MotionV3Response()
                await loop.run_in_executor(self.executor, resp.ParseFromString, response)
                if (
                    resp.class_name == "Speech2MotionV2ResponseChunkStart"
                    or resp.class_name == "Speech2MotionV3ResponseChunkStart"
                ):
                    if resp.HasField("timeline_start_idx_value"):
                        timeline_start_idx = resp.timeline_start_idx_value
                    else:
                        timeline_start_idx = None
                    if len(resp.blendshape_names) > 0:
                        blendshape_names = list(resp.blendshape_names)
                    else:
                        blendshape_names = None
                    joint_names = list(resp.joint_names)
                    for node in downstream_nodes:
                        payload = node.payload
                        next_node_name = node.name
                        start_chunk = MotionChunkStart(
                            request_id=request_id,
                            joint_names=joint_names,
                            restpose_name=str(resp.restpose_name),
                            dtype=resp.dtype,
                            timeline_start_idx=timeline_start_idx,
                            node_name=next_node_name,
                            blendshape_names=blendshape_names,
                            dag=dag,
                        )
                        coroutines.append(payload.feed_stream(start_chunk))
                elif (
                    resp.class_name == "Speech2MotionV2ResponseChunkBody"
                    or resp.class_name == "Speech2MotionV3ResponseChunkBody"
                ):
                    seq_number = self.input_buffer[request_id]["chunk_received_from_server"]
                    for node in downstream_nodes:
                        payload = node.payload
                        body_chunk = MotionChunkBody(
                            request_id=request_id,
                            seq_number=seq_number,
                            data=resp.data,
                        )
                        coroutines.append(payload.feed_stream(body_chunk))
                    self.input_buffer[request_id]["chunk_received_from_server"] += 1
                    first_chunk_handle_time = self.input_buffer[request_id]["first_chunk_handle_time"]
                    if seq_number == 0 and first_chunk_handle_time is not None:
                        cur_time = time.time()
                        latency = cur_time - first_chunk_handle_time
                        msg = (
                            f"Request {request_id} first MotionChunkBody delay {latency:.2f}s "
                            + "from receiving first AudioWithSubtitleChunkBody."
                        )
                        if dag_start_time is not None:
                            latency = cur_time - dag_start_time
                            msg = msg[:-1] + f", delay {latency:.2f}s from dag start."
                        self.logger.debug(msg)
                elif (
                    resp.class_name == "Speech2MotionV2ResponseChunkEnd"
                    or resp.class_name == "Speech2MotionV3ResponseChunkEnd"
                ):
                    chunk_end_received = True
                    for node in downstream_nodes:
                        payload = node.payload
                        end_chunk = MotionChunkEnd(
                            request_id=request_id,
                        )
                        coroutines.append(payload.feed_stream(end_chunk))
                    first_chunk_handle_time = self.input_buffer[request_id]["first_chunk_handle_time"]
                    if first_chunk_handle_time is not None:
                        cur_time = time.time()
                        latency = cur_time - first_chunk_handle_time
                        msg = (
                            f"Request {request_id} MotionChunkEnd delay {latency:.2f}s "
                            + "from receiving first AudioWithSubtitleChunkBody."
                        )
                        if dag_start_time is not None:
                            latency = cur_time - dag_start_time
                            msg = msg[:-1] + f", delay {latency:.2f}s from dag start."
                        self.logger.debug(msg)
                else:
                    msg = f"Unknown response class name: {resp.class_name}"
                    self.logger.error(msg)
                asyncio.gather(*coroutines)
            except Exception as e:
                self.logger.error(f"Error in receiving stream: {e}")
                dag.set_status(DAGStatus.FAILED)
        # Pop request_id from input buffer if chunk_end_received
        # If not chunk_end_received, DAG is interrupted,
        # keep request_id to receive data from upstream
        if chunk_end_received:
            self.input_buffer.pop(request_id)
        await ws_client.close()
