import asyncio
import io
import json
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Union

import websockets
from typing_extensions import Buffer

from ...data_structures.process_flow import DAGStatus
from ...data_structures.text_chunk import TextChunkBody, TextChunkEnd, TextChunkStart
from ...utils.audio import resample_pcm
from ...utils.executor_registry import ExecutorRegistry
from .asr_adapter import AutomaticSpeechRecognitionAdapter


class SensetimeASRClient(AutomaticSpeechRecognitionAdapter):
    """Sensetime automatic speech recognition client.

    This client provides real-time speech recognition using Sensetime's ASR
    service through WebSocket connections. It supports both streaming and non-
    streaming audio input with configurable parameters.
    """

    AVAILABLE_FOR_STREAM = True

    CHUNK_DURATION = 0.04
    CHUNK_N_BYTES = 1280
    _DEFAULT_PARAMS = dict(
        continuous_decoding=False,
        speech_pause_time=300,
        server_vad=False,
        punctuation=2,
        tradition=False,
        product_id="test",
        app_id="app_maltose_asr",
        device_id="test",
    )
    ExecutorRegistry.register_class("SensetimeASRClient")

    def __init__(
        self,
        name: str,
        ws_url: str,
        queue_size: int = 100,
        sleep_time: float = 0.01,
        clean_interval: float = 10.0,
        expire_time: float = 120.0,
        max_workers: int = 1,
        thread_pool_executor: ThreadPoolExecutor | None = None,
        logger_cfg: Union[None, Dict[str, Any]] = None,
    ):
        """Initialize the sensetime text to speech client.

        Args:
            name (str):
                The name of the text to speech client.
                Logger's name will be set to this name.
            ws_url (str):
                The URL of the ASR service.
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
            thread_pool_executor:
                External thread pool executor. Defaults to None.
            logger_cfg (Union[None, Dict[str, Any]], optional):
                Logger configuration. Defaults to None.
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
        self.ws_url = ws_url

    async def _create_connection(self, request_id: str, cur_time: float) -> None:
        """Create a connection to the server.

        Args:
            request_id (str):
                The request ID for tracking the connection.
            cur_time (float):
                The current timestamp for connection timing.
        """
        try:
            ws_client = await websockets.connect(self.ws_url)
            params = self._DEFAULT_PARAMS.copy()
            params["signal"] = "start"
            loop = asyncio.get_event_loop()
            start_message = await loop.run_in_executor(self.executor, json.dumps, params)
            await ws_client.send(start_message)
            self.input_buffer[request_id]["ws_client"] = ws_client
        except Exception as e:
            self.input_buffer[request_id]["connection_failed"] = True
            self.logger.error(f"Failed to create connection to Sensetime ASR server: {e}")
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
        loop = asyncio.get_event_loop()
        frame_rate = self.input_buffer[request_id]["frame_rate"]
        if frame_rate != self.__class__.FRAME_RATE:
            pcm_bytes = await loop.run_in_executor(
                self.executor,
                resample_pcm,
                pcm_bytes,
                frame_rate,
                self.__class__.FRAME_RATE,
            )
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
        await ws_client.send(pcm_bytes)
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
        end_message = json.dumps({"signal": "end"})
        start_time = time.time()
        await ws_client.send(end_message)
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
        loop = asyncio.get_event_loop()
        while True:
            message = await ws_client.recv()
            if isinstance(message, str):
                msg = await loop.run_in_executor(self.executor, json.loads, message)
                if msg["status"] == "ok":
                    if msg["type"] == "server_ready":
                        pass
                    elif msg["type"] == "partial_result" or msg["type"] == "final_result":
                        asr_text += msg["result"]
                        for node in downstream_nodes:
                            next_node_name = node.name
                            payload = node.payload
                            body_trunk = TextChunkBody(
                                request_id=request_id,
                                text_segment=msg["result"],
                            )
                            await payload.feed_stream(body_trunk)
                    elif msg["type"] == "speech_end":
                        break
                    else:
                        msg = f"Get unexpected ASR result type: {msg}"
                        self.logger.error(msg)
                        raise ValueError(msg)
                else:
                    msg = f"Get unexpected ASR status: {msg}"
                    self.logger.error(msg)
                    raise ValueError(msg)
            else:
                msg = f"Get unexpected ASR message type: {type(message)}"
                self.logger.error(msg)
                raise ValueError(msg)
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

    async def _send_audio(
        self,
        websocket: websockets.ClientConnection,
        audio_data: bytes,
        streaming: bool,
    ) -> None:
        """Send audio data to the WebSocket connection.

        Args:
            websocket (websockets.ClientConnection):
                The WebSocket connection to send audio data to.
            audio_data (bytes):
                The raw audio data to send.
            streaming (bool):
                Whether to send data in streaming mode with delays.
        """
        try:
            for i in range(0, len(audio_data), self.__class__.CHUNK_N_BYTES):
                end_idx = min(i + self.__class__.CHUNK_N_BYTES, len(audio_data))
                if end_idx > i:
                    frames = audio_data[i:end_idx]
                    await websocket.send(frames)
                    if streaming:
                        await asyncio.sleep(self.__class__.CHUNK_DURATION)
            end_message = json.dumps({"signal": "end"})
            await websocket.send(end_message)
        except websockets.exceptions.ConnectionClosed as e:
            msg = "Connection closed unexpectedly."
            self.logger.info(msg)
            raise e

    async def _on_message(
        self,
        message: Union[str, Buffer],
        text_results: List[str],
        subtitle_results: List[Dict[str, Any]],
        start_time: float = 0.0,
    ) -> bool:
        """Process the message from the websocket.

        Args:
            message (Union[str, Buffer]):
                The message from the websocket.
            text_result (List[str]):
                The text result.
            subtitle_result (List[Dict[str, Any]]):
                The subtitle result.
            start_time (float):
                The start time of the query.
                Defaults to 0.0.

        Returns:
            bool:
                Whether to continue the loop.
        """
        try:
            # if msg_type == websocket.ABNF.OPCODE_TEXT:
            if isinstance(message, str):
                msg = json.loads(message)
                if msg["status"] == "ok":
                    if msg["type"] == "server_ready":
                        return True
                    elif msg["type"] == "partial_result":
                        text_results.append(msg["result"])
                    elif msg["type"] == "final_result":
                        text_results.append(msg["result"])
                        for subtitle in msg["result_pieces"]:
                            if "start" in subtitle:
                                subtitle["start"] = subtitle["start"] + start_time
                            if "end" in subtitle:
                                subtitle["end"] = subtitle["end"] + start_time
                        subtitle_results.extend(msg["result_pieces"])
                    elif msg["type"] == "speech_end":
                        return False
                    else:
                        msg = f"Get unexpected ASR result type: {msg}"
                        self.logger.error(msg)
                        raise ValueError(msg)
                else:
                    msg = f"Get unexpected ASR status: {msg}"
                    self.logger.error(msg)
                    raise ValueError(msg)
            else:
                msg = f"Get unexpected ASR message type: {type(message)}"
                self.logger.error(msg)
                raise ValueError(msg)
        except Exception as e:
            msg = f"Error on_data: {e}"
            self.logger.error(msg)
            return False
        return True
