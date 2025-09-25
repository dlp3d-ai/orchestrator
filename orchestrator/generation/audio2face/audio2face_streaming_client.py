import asyncio
import io
import ssl
import time
import traceback
import wave
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Literal, Union

import websockets

from ...data_structures import audio2face_v1_pb2 as a2f_pb2
from ...data_structures.audio_chunk import (
    AudioChunkBody,
    AudioChunkEnd,
    AudioChunkStart,
    AudioWithReactionChunkBody,
    AudioWithReactionChunkEnd,
    AudioWithReactionChunkStart,
)
from ...data_structures.face_chunk import FaceChunkBody, FaceChunkEnd, FaceChunkStart
from ...data_structures.process_flow import DAGStatus
from ...utils.audio import resample_pcm
from ...utils.executor_registry import ExecutorRegistry
from ...utils.streamable import ChunkWithoutStartError
from .audio2face_adapter import Audio2FaceAdapter


class Audio2FaceStreamingClient(Audio2FaceAdapter):
    """Streaming client for audio-to-face generation using WebSocket
    connection.

    This client handles real-time audio streaming to generate facial animation
    data (blendshapes) through WebSocket communication. It supports both PCM
    and WAV audio formats and can process audio with emotion information.
    """

    N_CHANNELS: int = 1
    SAMPLE_WIDTH: int = 2
    FRAME_RATE: int = 16000

    _ssl_context_cache: Union[None, ssl.SSLContext] = None
    AVAILABLE_FOR_STREAM = True
    ExecutorRegistry.register_class("Audio2FaceStreamingClient")

    def __init__(
        self,
        ws_url: str,
        api_key: Union[None, str] = None,
        timeout: float = 10.0,
        verify: bool = True,
        queue_size: int = 100,
        sleep_time: float = 0.01,
        clean_interval: float = 10.0,
        expire_time: float = 120.0,
        max_workers: int = 1,
        thread_pool_executor: ThreadPoolExecutor | None = None,
        logger_cfg: Union[None, Dict[str, Any]] = None,
    ):
        """Initialize the audio2face streaming client.

        Args:
            ws_url (str):
                WebSocket URL of the audio2face service.
            api_key (Union[None, str], optional):
                API key for authentication with the audio2face service.
                Defaults to None.
            timeout (float, optional):
                Timeout in seconds for WebSocket operations.
                Defaults to 10.0.
            verify (bool, optional):
                Whether to verify SSL certificates for secure connections.
                Defaults to True.
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
                Maximum number of worker threads. Defaults to 1.
            thread_pool_executor (ThreadPoolExecutor | None, optional):
                Thread pool executor.
                If None, a new thread pool executor will be created based on
                max_workers. Defaults to None.
            logger_cfg (Union[None, Dict[str, Any]], optional):
                Logger configuration dictionary.
                Defaults to None.
        """
        Audio2FaceAdapter.__init__(
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
        """Destructor, cleanup thread pool executor."""
        if not self.executor_external:
            self.executor.shutdown(wait=True)

    async def generate_audio2face(
        self,
        audio: Union[str, io.BytesIO],
        request_id: Union[str, None] = None,
        output_type: Literal["arkit_npz", "arkit_csv"] = "arkit_npz",
        **kwargs: Any,
    ) -> io.BytesIO:
        """Generate face animation from audio (non-streaming mode).

        Note: This method is not implemented as this client only supports
        streaming mode. Use the streaming interface instead.

        Args:
            audio (Union[str, io.BytesIO]):
                Audio data to generate face animation from.
            request_id (Union[str, None], optional):
                Unique identifier for request tracking and logging.
                Defaults to None.
            output_type (Literal["arkit_npz", "arkit_csv"], optional):
                Output format for the generated face data.
                Defaults to "arkit_npz".
            **kwargs (Any):
                Additional arguments passed to the audio2face adapter.

        Returns:
            io.BytesIO:
                Generated face animation data.

        Raises:
            NotImplementedError:
                This method is not supported in streaming mode.
        """
        # TODO: Support non-streaming generation when needed
        msg = "Non-streaming audio2face generation is not supported yet."
        self.logger.error(msg)
        raise NotImplementedError(msg)

    async def _handle_start(self, chunk: Union[AudioChunkStart, AudioWithReactionChunkStart], cur_time: float) -> None:
        """Handle start chunks from audio streams.

        Initializes the input buffer for a new request and sets up tracking
        for audio processing. For PCM audio, immediately starts the stream task.

        Args:
            chunk (Union[AudioChunkStart, AudioWithReactionChunkStart]):
                Start chunk from audio stream.
            cur_time (float):
                Current timestamp for tracking.
        """
        dag = chunk.dag
        conf = dag.conf
        request_id = chunk.request_id
        audio_type = chunk.audio_type
        profile_name = conf["avatar"]
        response_chunk_n_frames = conf.get("response_chunk_n_frames", 10)
        self.input_buffer[request_id] = {
            "start_time": cur_time,
            "last_update_time": cur_time,
            "dag": dag,
            "audio_src_type": audio_type,
            "audio_src_n_channels": None,
            "audio_src_sample_width": None,
            "audio_src_frame_rate": None,
            "response_chunk_n_frames": response_chunk_n_frames,
            "profile_name": profile_name,
            "node_name": chunk.node_name,
            "first_chunk_handle_time": None,
            "chunk_received_from_upstream": 0,
            "chunk_sent_to_server": 0,
            "chunk_received_from_server": 0,
            "downstream_warned": False,
            "ws_client": None,
        }
        if audio_type.lower() == "pcm":
            self.input_buffer[request_id]["audio_src_n_channels"] = chunk.n_channels
            self.input_buffer[request_id]["audio_src_sample_width"] = chunk.sample_width
            self.input_buffer[request_id]["audio_src_frame_rate"] = chunk.frame_rate
            asyncio.create_task(self._start_stream_task(request_id))

    async def _handle_body(self, chunk: Union[AudioChunkBody, AudioWithReactionChunkBody], cur_time: float) -> None:
        """Handle body chunks from audio streams.

        Processes audio data chunks and sends them to the WebSocket server.
        Extracts emotion information if available and handles DAG status checking.

        Args:
            chunk (Union[AudioChunkBody, AudioWithReactionChunkBody]):
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
        audio_io = chunk.audio_io
        if hasattr(chunk, "reaction"):
            # Neutral emotion has no corresponding offset, so pass None
            emotion = chunk.reaction.face_emotion if chunk.reaction.face_emotion != "Neutral" else None
        else:
            emotion = None
        dag = self.input_buffer[request_id]["dag"]
        if dag.status == DAGStatus.RUNNING:
            current_seq_number = self.input_buffer[request_id]["chunk_received_from_upstream"]
            asyncio.create_task(self._send_audio_task(request_id, audio_io, current_seq_number, emotion))
            self.input_buffer[request_id]["chunk_received_from_upstream"] += 1
        else:
            current_seq_number = self.input_buffer[request_id]["chunk_received_from_upstream"]
            msg = f"DAG not running (status: {dag.status}) for request {request_id}. Body chunk {current_seq_number} discarded."
            self.input_buffer[request_id]["chunk_received_from_upstream"] += 1
            self.logger.warning(msg)

    async def _handle_end(self, chunk: Union[AudioChunkEnd, AudioWithReactionChunkEnd], cur_time: float) -> None:
        """Handle end chunks from audio streams.

        Signals the end of audio stream processing and triggers cleanup tasks.

        Args:
            chunk (Union[AudioChunkEnd, AudioWithReactionChunkEnd]):
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
        asyncio.create_task(self._send_stream_end_task(request_id))

    async def _start_stream_task(
        self,
        request_id: str,
    ) -> None:
        """Start WebSocket stream for audio2face processing.

        Establishes WebSocket connection and sends initial configuration
        to the audio2face server, then starts receiving responses.

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
        chunk_start = a2f_pb2.Audio2FaceBlendshapeV1Request()
        chunk_start.class_name = "StreamingAudio2FaceV1ChunkStart"
        chunk_start.request_id = request_id
        audio_src_sample_width = self.input_buffer[request_id]["audio_src_sample_width"]
        if audio_src_sample_width != self.__class__.SAMPLE_WIDTH:
            msg = f"Audio source sample width {audio_src_sample_width} does not match expected sample width {self.__class__.SAMPLE_WIDTH}"
            self.logger.error(msg)
            raise ValueError(msg)
        audio_src_n_channels = self.input_buffer[request_id]["audio_src_n_channels"]
        if audio_src_n_channels != self.__class__.N_CHANNELS:
            msg = f"Audio source n_channels {audio_src_n_channels} does not match expected n_channels {self.__class__.N_CHANNELS}"
            self.logger.error(msg)
            raise ValueError(msg)
        chunk_start.sample_rate = self.__class__.FRAME_RATE
        chunk_start.sample_width = audio_src_sample_width
        chunk_start.n_channels = audio_src_n_channels
        chunk_start.profile_name = self.input_buffer[request_id]["profile_name"]
        chunk_start.response_chunk_n_frames = self.input_buffer[request_id]["response_chunk_n_frames"]
        pb_bytes = await loop.run_in_executor(self.executor, chunk_start.SerializeToString)
        await ws_client.send(pb_bytes)
        self.input_buffer[request_id]["ws_client"] = ws_client
        asyncio.create_task(self._receive_stream_task(request_id))

    async def _send_audio_task(
        self,
        request_id: str,
        audio_io: io.BytesIO,
        seq_number: int,
        emotion: Union[str, None] = None,
    ) -> None:
        """Send audio data to WebSocket server for processing.

        Handles both PCM and WAV audio formats, extracts audio parameters
        for WAV files, and sends audio chunks with optional emotion data.

        Args:
            request_id (str):
                Unique identifier for the request.
            audio_io (io.BytesIO):
                Audio data in BytesIO format.
            seq_number (int):
                Sequence number for ordering audio chunks.
            emotion (Union[str, None], optional):
                Emotion information to apply to the face animation.
                Defaults to None.
        """
        if seq_number == 0:
            self.input_buffer[request_id]["first_chunk_handle_time"] = time.time()
        try:
            audio_src_type = self.input_buffer[request_id]["audio_src_type"]
            # If audio source type is wav, need to get audio parameters first to establish stream
            if audio_src_type == "wav" and self.input_buffer[request_id]["audio_src_n_channels"] is None:
                with wave.open(audio_io, "rb") as wf:
                    n_channels = wf.getnchannels()
                    sample_width = wf.getsampwidth()
                    frame_rate = wf.getframerate()
                if n_channels != self.__class__.N_CHANNELS:
                    msg = f"Audio source n_channels {n_channels} does not match expected n_channels {self.__class__.N_CHANNELS}"
                    self.logger.error(msg)
                    raise ValueError(msg)
                if sample_width != self.__class__.SAMPLE_WIDTH:
                    msg = f"Audio source sample width {sample_width} does not match expected sample width {self.__class__.SAMPLE_WIDTH}"
                    self.logger.error(msg)
                    raise ValueError(msg)
                self.input_buffer[request_id]["audio_src_n_channels"] = n_channels
                self.input_buffer[request_id]["audio_src_sample_width"] = sample_width
                self.input_buffer[request_id]["audio_src_frame_rate"] = frame_rate
                await self._start_stream_task(request_id)
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
            chunk_body = a2f_pb2.Audio2FaceBlendshapeV1Request()
            chunk_body.class_name = "StreamingAudio2FaceV1ChunkBody"
            audio_src_frame_rate = self.input_buffer[request_id]["audio_src_frame_rate"]
            if audio_src_type == "pcm":
                pcm_bytes = audio_io.getvalue()
            else:
                audio_io.seek(0)
                with wave.open(audio_io, "rb") as wf:
                    n_frames = wf.getnframes()
                    pcm_bytes = wf.readframes(n_frames)
            if audio_src_frame_rate != self.__class__.FRAME_RATE:
                loop = asyncio.get_running_loop()
                pcm_bytes = await loop.run_in_executor(
                    self.executor,
                    resample_pcm,
                    pcm_bytes,
                    audio_src_frame_rate,
                    self.__class__.FRAME_RATE,
                )
            chunk_body.pcm_bytes = pcm_bytes
            if emotion is not None:
                # TODO: A2F offset_name provides 1,2,3,4 suffixes
                # Can select different suffixes based on emotion intensity in the future
                chunk_body.offset_name = f"{emotion.lower()}_1"
            loop = asyncio.get_running_loop()
            data = await loop.run_in_executor(self.executor, chunk_body.SerializeToString)
            await ws_client.send(data)
            self.input_buffer[request_id]["chunk_sent_to_server"] += 1
        except Exception as e:
            msg = f"Error in streaming audio2face generation: {e}"
            msg = msg + f" for request {request_id}"
            traceback_str = traceback.format_exc()
            msg += f"\n{traceback_str}"
            self.logger.error(msg)
            dag = self.input_buffer[request_id]["dag"]
            dag.set_status(DAGStatus.FAILED)

    async def _send_stream_end_task(self, request_id: str) -> None:
        """Send stream end signal to WebSocket server.

        Waits for all pending audio chunks to be sent, then sends
        the end signal to complete the audio stream.

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
                msg = msg + f" for request {request_id}"
                self.logger.warning(msg)
                return
        chunk_end = a2f_pb2.Audio2FaceBlendshapeV1Request()
        chunk_end.class_name = "StreamingAudio2FaceV1ChunkEnd"
        loop = asyncio.get_running_loop()
        data = await loop.run_in_executor(self.executor, chunk_end.SerializeToString)
        await ws_client.send(data)

    async def _receive_stream_task(self, request_id: str) -> None:
        """Receive face animation data from WebSocket server.

        Continuously receives blendshape data from the server and forwards
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
                resp = a2f_pb2.Audio2FaceBlendshapeV1Response()
                await loop.run_in_executor(self.executor, resp.ParseFromString, response)
                if resp.class_name == "Audio2FaceV1ResponseChunkStart":
                    for node in downstream_nodes:
                        next_node_name = node.name
                        payload = node.payload
                        start_trunk = FaceChunkStart(
                            request_id=request_id,
                            blendshape_names=resp.blendshape_names,
                            dtype=resp.dtype,
                            node_name=next_node_name,
                            dag=dag,
                        )
                        coroutines.append(payload.feed_stream(start_trunk))
                elif resp.class_name == "Audio2FaceV1ResponseChunkBody":
                    blendshape_data = resp.data
                    seq_number = self.input_buffer[request_id]["chunk_received_from_server"]
                    for node in downstream_nodes:
                        payload = node.payload
                        body_chunk = FaceChunkBody(
                            request_id=request_id,
                            data=blendshape_data,
                            seq_number=seq_number,
                        )
                        coroutines.append(payload.feed_stream(body_chunk))
                    self.input_buffer[request_id]["chunk_received_from_server"] += 1
                    first_chunk_handle_time = self.input_buffer[request_id]["first_chunk_handle_time"]
                    if seq_number == 0 and first_chunk_handle_time is not None:
                        cur_time = time.time()
                        latency = cur_time - first_chunk_handle_time
                        msg = (
                            f"Request {request_id} first FaceChunkBody delay {latency:.2f}s "
                            + "from sending first AudioChunkBody."
                        )
                        if dag_start_time is not None:
                            latency = cur_time - dag_start_time
                            msg = msg[:-1] + f", delay {latency:.2f}s from dag start."
                        self.logger.debug(msg)
                elif resp.class_name == "Audio2FaceV1ResponseChunkEnd":
                    chunk_end_received = True
                    for node in downstream_nodes:
                        payload = node.payload
                        end_chunk = FaceChunkEnd(
                            request_id=request_id,
                        )
                        coroutines.append(payload.feed_stream(end_chunk))
                    first_chunk_handle_time = self.input_buffer[request_id]["first_chunk_handle_time"]
                    if first_chunk_handle_time is not None:
                        cur_time = time.time()
                        latency = cur_time - first_chunk_handle_time
                        msg = (
                            f"Request {request_id} FaceChunkEnd delay {latency:.2f}s "
                            + "from sending first AudioChunkBody."
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
                msg = f"Error in receiving stream: {e}"
                self.logger.error(msg)
                dag.set_status(DAGStatus.FAILED)
        # Pop request_id from input buffer if chunk_end_received
        # If not chunk_end_received, DAG is interrupted,
        # keep request_id to receive data from upstream
        if chunk_end_received:
            self.input_buffer.pop(request_id)
        await ws_client.close()
