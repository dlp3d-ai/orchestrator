import asyncio
import hashlib
import io
import json
import socket
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Union

import httpx
import websockets

from ...data_structures.process_flow import DAGStatus
from ...data_structures.text_chunk import TextChunkBody, TextChunkEnd, TextChunkStart
from ...utils.audio import resample_pcm
from ...utils.executor_registry import ExecutorRegistry
from .asr_adapter import AutomaticSpeechRecognitionAdapter


class SoftSugarASRClient(AutomaticSpeechRecognitionAdapter):
    """SoftSugar automatic speech recognition client.

    This client provides real-time speech recognition using SoftSugar's ASR
    service through WebSocket connections. It includes token-based
    authentication and supports both streaming and non-streaming audio input.
    """

    AVAILABLE_FOR_STREAM = True

    CHUNK_DURATION = 0.04
    CHUNK_N_BYTES = 1280
    ExecutorRegistry.register_class("SoftSugarASRClient")

    def __init__(
        self,
        name: str,
        ws_url: str,
        softsugar_api: str,
        softsugar_refresh_interval: float = 4 * 60 * 60,
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
            softsugar_api (str):
                The API of the softsugar service.
            softsugar_refresh_interval (float, optional):
                The interval to refresh the softsugar token.
                Defaults to 4 * 60 * 60.
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
        self.softsugar_api = softsugar_api
        self.softsugar_refresh_interval = softsugar_refresh_interval

        self.last_refresh_time: Union[float, None] = None
        self.access_token: Union[str, None] = None
        self.access_token_expire_time: Union[float, None] = None
        self.refresh_token: Union[str, None] = None
        self.refresh_token_expire_time: Union[float, None] = None

        self.hostname = socket.gethostname()

    def _create_token_request_data(
        self,
        softsugar_app_id: str,
        softsugar_app_key: str,
        cur_time: Union[float, None] = None,
    ) -> Dict[str, Any]:
        """Create the token request data.

        Args:
            softsugar_app_id (str):
                The app id of the softsugar service.
            softsugar_app_key (str):
                The app key of the softsugar service.
            cur_time (Union[float, None], optional):
                The current timestamp for token generation.
                Defaults to None, uses current time.

        Returns:
            Dict[str, Any]:
                The token request data including app ID, timestamp, and signature.
        """
        if cur_time is None:
            cur_time = time.time()
        timestamp = str(int(cur_time * 1000))
        raw_string = f"{softsugar_app_id}{timestamp}{softsugar_app_key}"
        sign = hashlib.md5(raw_string.encode()).hexdigest()
        request_data = {"appId": softsugar_app_id, "timestamp": timestamp, "sign": sign, "grantType": "sign"}
        return request_data

    async def _ensure_token_valid(
        self,
        softsugar_app_id: str,
        softsugar_app_key: str,
        cur_time: Union[float, None] = None,
    ):
        """Ensure the access token is valid and refresh if necessary.

        Args:
            softsugar_app_id (str):
                The app id of the softsugar service.
            softsugar_app_key (str):
                The app key of the softsugar service.
            cur_time (Union[float, None], optional):
                The current timestamp for token validation.
                Defaults to None, uses current time.
        """
        if cur_time is None:
            cur_time = time.time()
        loop = asyncio.get_event_loop()
        if (
            self.last_refresh_time is None
            or self.access_token_expire_time is None
            or cur_time >= self.access_token_expire_time
        ):
            request_dict = await loop.run_in_executor(
                self.executor,
                self._create_token_request_data,
                softsugar_app_id,
                softsugar_app_key,
                cur_time,
            )
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.softsugar_api}/api/uc/v1/access/api/token",
                    headers={"Content-Type": "application/json"},
                    json=request_dict,
                )
                response.raise_for_status()  # Raise an exception for HTTP errors
                response_dict = await loop.run_in_executor(self.executor, response.json)
                self.last_refresh_time = cur_time
                self.access_token_expire_time = cur_time + int(response_dict["data"]["expiresIn"])
                self.refresh_token_expire_time = cur_time + int(response_dict["data"]["refreshTokenExpiresIn"])
                self.access_token = response_dict["data"]["accessToken"]
                self.refresh_token = response_dict["data"]["refreshToken"]

    async def _create_connection(self, request_id: str, cur_time: float) -> None:
        """Create a connection to the server.

        Args:
            request_id (str):
                The request ID for tracking the connection.
            cur_time (float):
                The current timestamp for connection timing.
        """
        softsugar_app_id = self.input_buffer[request_id].get("api_keys", {}).get("softsugar_app_id", "")
        softsugar_app_key = self.input_buffer[request_id].get("api_keys", {}).get("softsugar_app_key", "")
        if not softsugar_app_id or not softsugar_app_key:
            msg = "SoftSugar app ID or app key is not found in the API keys."
            self.logger.error(msg)
            raise ValueError(msg)
        language = self.input_buffer[request_id].get("language", "zh")
        try:
            await self._ensure_token_valid(softsugar_app_id, softsugar_app_key, cur_time)
            endpoint = self.ws_url + "?Authorization=Bearer%20" + (self.access_token or "")
            ws_client = await websockets.connect(endpoint)
            starter_dict = dict(
                type="ASR5",
                device=self.hostname,
                session=request_id,
                asr=dict(
                    language="zh-CN" if language == "zh" else "en-US",
                    word_time=True,
                ),
            )
            loop = asyncio.get_event_loop()
            data = await loop.run_in_executor(self.executor, json.dumps, starter_dict)
            await ws_client.send(data)
            response_json = await ws_client.recv()
            response_dict = json.loads(response_json)
            if response_dict["status"] != "ok":
                msg = f"Get unexpected ASR status on starting streaming ASR: {response_dict}"
                self.logger.error(msg)
                raise ValueError(msg)
            self.input_buffer[request_id]["ws_client"] = ws_client
        except Exception as e:
            self.input_buffer[request_id]["connection_failed"] = True
            self.logger.error(f"Failed to create connection to SoftSugar ASR server: {e}")
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
        frame_rate = self.input_buffer[request_id]["frame_rate"]
        loop = asyncio.get_event_loop()
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
        eof_dict = dict(
            signal="eof",
            trace=request_id,
        )
        start_time = time.time()
        await ws_client.send(json.dumps(eof_dict))
        self.logger.debug(f"sent asr commit message to server for request {request_id}")
        asr_text = ""
        receive_count = 0
        # Typically, the server returns 2 messages after the eof signal
        # The first message is the ASR result,
        # the second message is the EOF response
        while receive_count <= 2:
            message = await ws_client.recv()
            receive_count += 1
            response_json = json.loads(message)
            if response_json["status"] == "ok" and "asr" in response_json:
                type_str = response_json["asr"]["type"]
                if type_str == "eof":
                    break
                else:
                    asr_text = response_json["asr"]["text"]
            else:
                msg = f"Get unexpected ASR result: {response_json}"
                self.logger.error(msg)
                raise ValueError(msg)
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
            body_trunk = TextChunkBody(
                request_id=request_id,
                text_segment=asr_text,
            )
            await payload.feed_stream(body_trunk)
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
