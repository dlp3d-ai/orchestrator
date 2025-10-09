import asyncio
import base64
import json
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Union

import websockets

from ...data_structures.process_flow import DAGStatus
from ...data_structures.text_chunk import TextChunkBody, TextChunkEnd, TextChunkStart
from ...utils.audio import resample_pcm
from ...utils.exception import MissingAPIKeyException
from ...utils.executor_registry import ExecutorRegistry
from .asr_adapter import AutomaticSpeechRecognitionAdapter


class OpenAIRealtimeASRClient(AutomaticSpeechRecognitionAdapter):
    """OpenAI realtime automatic speech recognition client.

    This client provides real-time speech recognition using OpenAI's realtime
    API through WebSocket connections. It supports streaming audio input and
    provides continuous transcription results.
    """

    AVAILABLE_FOR_STREAM = True
    FRAME_RATE: int = 24000

    CHUNK_DURATION = 0.04
    CHUNK_N_BYTES = 1280
    ExecutorRegistry.register_class("OpenAIRealtimeASRClient")

    def __init__(
        self,
        name: str,
        wss_url: str,
        openai_model_name: str = "gpt-4o-mini-transcribe",
        proxy_url: Union[None, str] = None,
        request_timeout: float = 20.0,
        queue_size: int = 100,
        sleep_time: float = 0.01,
        clean_interval: float = 10.0,
        expire_time: float = 120.0,
        commit_timeout: float = 2.0,
        max_workers: int = 1,
        thread_pool_executor: ThreadPoolExecutor | None = None,
        logger_cfg: Union[None, Dict[str, Any]] = None,
    ):
        """Initialize the OpenAI realtime automatic speech recognition client.

        Args:
            name (str):
                The name of the client.
                Logger's name will be set to this name.
            wss_url (str):
                The WebSocket URL for the OpenAI API.
            openai_model_name (str):
                The model name to use for the OpenAI API.
            proxy_url (Union[None, str]):
                The proxy URL to use for the OpenAI API.
            request_timeout (float):
                The timeout for the OpenAI API request.
            queue_size (int):
                The size of the queue for the OpenAI API request.
            sleep_time (float):
                The sleep time for the OpenAI API request.
            clean_interval (float):
                The interval for cleaning the input buffer.
            expire_time (float):
                The time to live for the input buffer.
            commit_timeout (float):
                The timeout for waiting for ASR response after commit.
                Defaults to 10.0 seconds.
            max_workers (int):
                The number of workers for the thread pool executor.
            thread_pool_executor:
                External thread pool executor. Defaults to None.
            logger_cfg (Union[None, Dict[str, Any]]):
                The configuration for the logger.
        """
        super().__init__(
            name=name,
            queue_size=queue_size,
            sleep_time=sleep_time,
            clean_interval=clean_interval,
            expire_time=expire_time,
            max_workers=max_workers,
            thread_pool_executor=thread_pool_executor,
            logger_cfg=logger_cfg,
        )
        self.wss_url = wss_url
        self.openai_model_name = openai_model_name
        self.proxy_url = proxy_url
        self.request_timeout = request_timeout
        self.commit_timeout = commit_timeout

    def _encode_pcm(self, pcm_bytes: bytes) -> str:
        """Encode the PCM bytes to the target format.

        Args:
            pcm_bytes (bytes):
                The raw PCM audio data to encode.

        Returns:
            str:
                Base64 encoded PCM data for transmission.
        """
        encoded_pcm = base64.b64encode(pcm_bytes).decode("utf-8")
        return encoded_pcm

    async def _create_connection(self, request_id: str, cur_time: float) -> None:
        """Create a connection to the server.

        Args:
            request_id (str):
                The request ID for tracking the connection.
            cur_time (float):
                The current timestamp for connection timing.
        """
        openai_api_key = self.input_buffer[request_id].get("api_keys", {}).get("openai_api_key", "")
        if not openai_api_key:
            msg = "OpenAI API key is not found in the API keys."
            self.logger.error(msg)
            raise MissingAPIKeyException(msg)
        language = self.input_buffer[request_id].get("language", "zh")
        additional_headers = {
            "Authorization": f"Bearer {openai_api_key}",
            "OpenAI-Beta": "realtime=v1",
        }
        try:
            ws = await websockets.connect(
                self.wss_url,
                proxy=self.proxy_url,
                additional_headers=additional_headers,
                ping_interval=self.request_timeout,
                ping_timeout=self.request_timeout,
            )
            data = {
                "type": "transcription_session.update",
                "session": {
                    "input_audio_transcription": {
                        "model": self.openai_model_name,
                        "language": language,
                    },
                    "turn_detection": None,
                },
            }
            loop = asyncio.get_event_loop()
            data = await loop.run_in_executor(
                self.executor,
                json.dumps,
                data,
            )
            await ws.send(data)
            self.input_buffer[request_id]["ws_client"] = ws
            self.input_buffer[request_id]["commit_time"] = None
        except Exception as e:
            self.input_buffer[request_id]["connection_failed"] = True
            self.logger.error(f"Failed to create connection to OpenAI realtime ASR server: {e}")
            raise e

    async def _send_pcm_task(self, request_id: str, pcm_bytes: bytes, seq_number: int) -> None:
        """Send PCM bytes to the server.

        Args:
            request_id (str):
                The request ID for tracking the audio chunk.
            pcm_bytes (bytes):
                The PCM audio data to send.
            seq_number (int):
                The sequence number of the audio chunk.
        """
        dag = self.input_buffer[request_id]["dag"]
        ws_client = self.input_buffer[request_id].get("ws_client", None)
        while ws_client is None:
            if self.input_buffer[request_id]["connection_failed"]:
                return
            await asyncio.sleep(self.sleep_time)
            ws_client = self.input_buffer[request_id].get("ws_client", None)
        while self.input_buffer[request_id]["chunk_sent_to_server"] < seq_number:
            if dag.status == DAGStatus.RUNNING:
                await asyncio.sleep(self.sleep_time)
            else:
                msg = f"Streaming audio sending interrupted by DAG status {dag.status}"
                msg = msg + f" for request {request_id}"
                self.logger.warning(msg)
                return

        # Resample audio to 24000Hz if needed
        frame_rate = self.input_buffer[request_id]["frame_rate"]
        if frame_rate != self.__class__.FRAME_RATE:
            loop = asyncio.get_event_loop()
            pcm_bytes = await loop.run_in_executor(
                self.executor,
                resample_pcm,
                pcm_bytes,
                frame_rate,
                self.__class__.FRAME_RATE,
            )

        # Encode audio data
        loop = asyncio.get_event_loop()
        audio_chunk = await loop.run_in_executor(
            self.executor,
            self._encode_pcm,
            pcm_bytes,
        )

        data = {
            "type": "input_audio_buffer.append",
            "audio": audio_chunk,
        }
        data = await loop.run_in_executor(
            self.executor,
            json.dumps,
            data,
        )
        await ws_client.send(data)
        self.input_buffer[request_id]["chunk_sent_to_server"] += 1

    async def _send_to_downstream_and_clean(self, request_id: str) -> None:
        """Send result to downstream and clean up.

        Args:
            request_id (str):
                The request ID for tracking the completion.
        """
        dag = self.input_buffer[request_id]["dag"]
        ws_client = self.input_buffer[request_id].get("ws_client", None)
        while ws_client is None:
            if self.input_buffer[request_id]["connection_failed"]:
                self.input_buffer.pop(request_id, None)
                return
            await asyncio.sleep(self.sleep_time)
            ws_client = self.input_buffer[request_id].get("ws_client", None)
        while (
            self.input_buffer[request_id]["chunk_sent_to_server"]
            < self.input_buffer[request_id]["chunk_received_from_upstream"]
        ):
            if dag.status == DAGStatus.RUNNING:
                await asyncio.sleep(self.sleep_time)
            else:
                msg = f"Streaming audio sending interrupted by DAG status {dag.status}"
            await asyncio.sleep(self.sleep_time)
        start_time = time.time()
        loop = asyncio.get_event_loop()
        data = await loop.run_in_executor(
            self.executor,
            json.dumps,
            {"type": "input_audio_buffer.commit"},
        )
        await ws_client.send(data)
        self.input_buffer[request_id]["commit_time"] = time.time()
        self.logger.debug(f"sent asr commit message to server for request {request_id}")
        asr_text = ""
        dag = self.input_buffer[request_id]["dag"]
        node_name = self.input_buffer[request_id]["node_name"]
        dag_node = dag.get_node(node_name)
        downstream_nodes = dag_node.downstreams
        if len(downstream_nodes) == 0:
            self.logger.warning(f"Request {request_id} has no downstreams, so the result is discarded.")
        for node in downstream_nodes:
            next_node_name = node.name
            payload = node.payload
            start_trunk = TextChunkStart(
                request_id=request_id,
                node_name=next_node_name,
                dag=dag,
            )
            await payload.feed_stream(start_trunk)

        commit_time = self.input_buffer[request_id]["commit_time"]
        while True:
            current_time = time.time()
            if current_time - commit_time > self.commit_timeout:
                break
            try:
                remaining_time = self.commit_timeout - (current_time - commit_time)
                if remaining_time <= 0:
                    break
                message = await asyncio.wait_for(ws_client.recv(), timeout=remaining_time)
                if isinstance(message, str):
                    msg = json.loads(message)
                    if msg["type"] == "conversation.item.input_audio_transcription.delta":
                        asr_text += msg["delta"]
                        for node in downstream_nodes:
                            next_node_name = node.name
                            payload = node.payload
                            body_trunk = TextChunkBody(
                                request_id=request_id,
                                text_segment=msg["delta"],
                            )
                            await payload.feed_stream(body_trunk)
                    elif msg["type"] == "conversation.item.input_audio_transcription.completed":
                        break
                    elif msg["type"] == "error":
                        raise Exception(msg["error"])
            except (asyncio.TimeoutError, Exception) as e:
                self.logger.error(f"ASR error for request {request_id}: {e}")
                break
        if not asr_text:
            self.logger.warning(f"ASR text is empty for request {request_id}, use default text.")
            asr_text = ""
            for node in downstream_nodes:
                next_node_name = node.name
                payload = node.payload
                body_trunk = TextChunkBody(
                    request_id=request_id,
                    text_segment=asr_text,
                )
                await payload.feed_stream(body_trunk)

        for node in downstream_nodes:
            next_node_name = node.name
            payload = node.payload
            end_trunk = TextChunkEnd(
                request_id=request_id,
            )
            await payload.feed_stream(end_trunk)
        end_time = time.time()
        time_diff = end_time - start_time
        msg = f"Streaming ASR generation finished in {time_diff:.3f} seconds from audio stream end"
        dag_start_time = self.input_buffer[request_id].get("dag_start_time", None)
        if dag_start_time is not None:
            from_dag_start = end_time - dag_start_time
            msg = msg + f", in {from_dag_start:.3f} seconds from DAG start"
        msg = msg + f", for request {request_id}, speech_text: {asr_text}"
        self.logger.debug(msg)
        self.input_buffer.pop(request_id)
        await ws_client.close()
