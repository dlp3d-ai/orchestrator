import asyncio
from abc import abstractmethod
from asyncio import QueueEmpty
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Union

import websockets

from ...data_structures.audio_chunk import AudioChunkBody, AudioChunkEnd, AudioChunkStart
from ...data_structures.process_flow import DAGStatus
from ...utils.exception import MissingAPIKeyException, failure_callback
from ...utils.log import setup_logger
from ...utils.streamable import ChunkWithoutStartError, Streamable


class AutomaticSpeechRecognitionAdapter(Streamable):
    """Automatic speech recognition adapter.

    This is the base class for all automatic speech recognition adapters. It
    provides common functionality for handling audio streams and managing
    WebSocket connections for real-time speech recognition.
    """

    FRAME_RATE: int = 16000
    AVAILABLE_FOR_STREAM = False
    N_CHANNELS: int = 1
    SAMPLE_WIDTH: int = 2

    def __init__(
        self,
        name: str,
        queue_size: int = 100,
        sleep_time: float = 0.01,
        clean_interval: float = 10.0,
        expire_time: float = 120.0,
        max_workers: int = 1,
        thread_pool_executor: ThreadPoolExecutor | None = None,
        logger_cfg: Union[None, Dict[str, Any]] = None,
    ):
        """Initialize the automatic speech recognition adapter.

        Args:
            name (str):
                The name of the automatic speech recognition adapter.
                Logger's name will be set to this name.
            queue_size (int, optional):
                The size of the queue.
                Defaults to 100.
            sleep_time (float, optional):
                The sleep time.
                Defaults to 0.01.
            clean_interval (float, optional):
                The interval to clean the expired requests.
                Defaults to 10.0.
            expire_time (float, optional):
                The time to expire the request.
                Defaults to 120.0.
            max_workers (int, optional):
                The number of workers for the thread pool executor.
                Defaults to 1.
            thread_pool_executor (ThreadPoolExecutor | None, optional):
                External thread pool executor. Defaults to None.
            logger_cfg (Union[None, Dict[str, Any]], optional):
                Logger configuration. Defaults to None.
        """
        Streamable.__init__(
            self,
            queue_size=queue_size,
            sleep_time=sleep_time,
            clean_interval=clean_interval,
            expire_time=expire_time,
            logger_cfg=logger_cfg,
        )
        self.name = name
        self.logger_cfg["logger_name"] = name
        self.logger = setup_logger(**self.logger_cfg)
        self.executor = (
            thread_pool_executor if thread_pool_executor is not None else ThreadPoolExecutor(max_workers=max_workers)
        )
        self.executor_external = True if thread_pool_executor is not None else False

    def __del__(self) -> None:
        """Destructor, cleanup thread pool executor."""
        if not self.executor_external:
            self.executor.shutdown(wait=True)

    async def _one_loop(self, cur_time: float) -> bool:
        """One loop of the streamable object.

        Args:
            cur_time (float):
                The current time.

        Returns:
            bool:
                Whether to continue the loop.
                For class Streamable, it's always True.
        """
        try:
            input_chunk = self.queue.get_nowait()
        except QueueEmpty:
            # Clean expired requests
            if cur_time - self.last_clean_time > self.clean_interval:
                keys = list(self.input_buffer.keys())
                keys_to_remove = []
                for key in keys:
                    last_update_time = self.input_buffer[key]["last_update_time"]
                    if cur_time - last_update_time > self.expire_time:
                        keys_to_remove.append(key)
                if len(keys_to_remove) > 0:
                    msg = f"Cleaned {len(keys_to_remove)} expired requests: {keys_to_remove}"
                    self.logger.warning(msg)
                    for key in keys_to_remove:
                        if "ws_client" in self.input_buffer[key]:
                            try:
                                if self.input_buffer[key]["ws_client"].protocol.state is not websockets.protocol.CLOSED:
                                    await self.input_buffer[key]["ws_client"].close()
                            except Exception as e:
                                self.logger.error(f"Failed to close websocket connection for request {key}: {e}")
                        self.input_buffer.pop(key)
                self.last_clean_time = cur_time
            else:
                await asyncio.sleep(self.sleep_time)
            return True
        if isinstance(input_chunk, dict):
            if "chunk_type" not in input_chunk:
                msg = f"Missing chunk type in input: {input_chunk}"
                self.logger.error(msg)
                return True
            type_str = input_chunk["chunk_type"]
        else:
            type_str = input_chunk.chunk_type
        if type_str == "start":
            await self._handle_start(input_chunk, cur_time)  # type: ignore
            return True
        elif type_str == "body":
            await self._handle_body(input_chunk, cur_time)  # type: ignore
            return True
        elif type_str == "end":
            await self._handle_end(input_chunk, cur_time)  # type: ignore
            return True
        else:
            msg = f"Received an unknown chunk type: {type_str}"
            self.logger.error(msg)
            return True

    async def _handle_start(self, chunk: AudioChunkStart, cur_time: float) -> None:
        """Handle the start chunk.

        Args:
            chunk (AudioChunkStart):
                The audio chunk start message containing audio metadata.
            cur_time (float):
                The current timestamp for tracking request timing.
        """
        request_id = chunk.request_id
        if chunk.audio_type == "wav":
            msg = f"WAV format is not supported for sensetime streaming ASR, request_id: {request_id}"
            self.logger.error(msg)
            raise NotImplementedError(msg)
        elif chunk.audio_type == "pcm":
            if chunk.n_channels != self.__class__.N_CHANNELS:
                msg = (
                    "Only mono audio is supported for sensetime streaming ASR, "
                    + f"but got {chunk.n_channels} channels for request {request_id}."
                )
                self.logger.error(msg)
                raise NotImplementedError(msg)
            if chunk.sample_width != self.__class__.SAMPLE_WIDTH:
                msg = (
                    "Only 16-bit audio is supported for sensetime streaming ASR, "
                    + f"but got {chunk.sample_width} bytes for request {request_id}."
                )
                self.logger.error(msg)
                raise NotImplementedError(msg)

        dag = chunk.dag
        dag_start_time = dag.conf.get("start_time", None)
        api_keys = dag.conf.get("user_settings", {})
        language = dag.conf.get("language", "zh")
        callback_bytes_fn = chunk.dag.conf.get("callback_bytes_fn", None)
        self.input_buffer[request_id] = {
            "dag_start_time": dag_start_time,
            "start_time": cur_time,
            "last_update_time": cur_time,
            "dag": dag,
            "node_name": chunk.node_name,
            "api_keys": api_keys,
            "audio_type": chunk.audio_type,
            "language": language,
            "n_channels": chunk.n_channels,
            "sample_width": chunk.sample_width,
            "frame_rate": chunk.frame_rate,
            "ws_client": None,
            "chunk_received_from_upstream": 0,
            "chunk_sent_to_server": 0,
            "connection_failed": False,
            "callback_bytes_fn": callback_bytes_fn,
        }
        task = asyncio.create_task(self._create_connection(request_id, cur_time))
        task.add_done_callback(lambda t: self._handle_task_exception(t, request_id))

    def _handle_task_exception(self, task: asyncio.Task, request_id: str) -> None:
        """Handle exceptions from the initialization task.

        Args:
            task (asyncio.Task): The completed task.
            request_id (str): The request ID associated with the task.
        """
        exception = task.exception()
        if exception is not None:
            if isinstance(exception, MissingAPIKeyException):
                msg = f"Missing API key during ASR client initialization: {exception}"
                self.logger.error(msg)
                # Create an async task to handle the failure callback
                asyncio.create_task(self._send_failure_callback(msg, request_id))
            else:
                msg = f"Unexpected error during ASR client task: {exception}"
                self.logger.error(msg)
                # Create an async task to handle the failure callback for other exceptions too
                asyncio.create_task(self._send_failure_callback(f"Unexpected error: {exception}", request_id))
            dag = self.input_buffer[request_id]["dag"]
            if dag is not None:
                dag.set_status(DAGStatus.FAILED)

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

    async def _handle_body(self, chunk: AudioChunkBody, cur_time: float) -> None:
        """Handle the body chunk.

        Args:
            chunk (AudioChunkBody):
                The audio chunk body message containing audio data.
            cur_time (float):
                The current timestamp for tracking request timing.
        """
        request_id = chunk.request_id
        if request_id not in self.input_buffer:
            msg = f"Request {request_id} not found in input buffer, but received a body message."
            self.logger.error(msg)
            raise ChunkWithoutStartError(msg)
        if self.input_buffer[request_id]["audio_type"] == "wav":
            raise NotImplementedError("WAV format is not supported for streaming ASR.")
        pcm_io = chunk.audio_io
        dag = self.input_buffer[request_id]["dag"]
        if dag.status == DAGStatus.RUNNING:
            seq_number = self.input_buffer[request_id]["chunk_received_from_upstream"]
            self.input_buffer[request_id]["chunk_received_from_upstream"] += 1
            task = asyncio.create_task(self._send_pcm_task(request_id, pcm_io.getvalue(), seq_number))
            task.add_done_callback(lambda t: self._handle_task_exception(t, request_id))
        self.input_buffer[request_id]["last_update_time"] = cur_time

    async def _handle_end(self, chunk: AudioChunkEnd, cur_time: float) -> None:
        """Handle the end chunk.

        Args:
            chunk (AudioChunkEnd):
                The audio chunk end message signaling the end of audio stream.
            cur_time (float):
                The current timestamp for tracking request timing.
        """
        request_id = chunk.request_id
        if request_id not in self.input_buffer:
            msg = f"Request {request_id} not found in input buffer, but received an end message."
            self.logger.error(msg)
            raise ChunkWithoutStartError(msg)
        self.input_buffer[request_id]["last_update_time"] = cur_time
        dag = self.input_buffer[request_id]["dag"]
        if dag.status == DAGStatus.RUNNING:
            task = asyncio.create_task(self._send_to_downstream_and_clean(request_id))
            task.add_done_callback(lambda t: self._handle_task_exception(t, request_id))

    @abstractmethod
    async def _create_connection(self, request_id: str, cur_time: float) -> None:
        """Create a connection to the server.

        Args:
            request_id (str): The request ID.
            cur_time (float): The current time.
        """
        raise NotImplementedError

    @abstractmethod
    async def _send_pcm_task(self, request_id: str, pcm_bytes: bytes, seq_number: int) -> None:
        """Send PCM bytes to the server.

        Args:
            request_id (str): The request ID.
            pcm_bytes (bytes): The PCM audio data.
            seq_number (int): The sequence number.
        """
        raise NotImplementedError

    @abstractmethod
    async def _send_to_downstream_and_clean(self, request_id: str) -> None:
        """Send result to downstream and clean up.

        Args:
            request_id (str): The request ID.
        """
        raise NotImplementedError
