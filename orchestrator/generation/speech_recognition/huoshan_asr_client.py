import asyncio
import gzip
import json
import socket
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Union

import websockets

from ...data_structures.process_flow import DAGStatus
from ...data_structures.text_chunk import TextChunkBody, TextChunkEnd, TextChunkStart
from ...utils.audio import resample_pcm
from ...utils.exception import MissingAPIKeyException
from ...utils.executor_registry import ExecutorRegistry
from .asr_adapter import AutomaticSpeechRecognitionAdapter


class HuoshanASRClientError(Exception):
    """Huoshan ASR client error."""

    pass


class HuoshanASRClient(AutomaticSpeechRecognitionAdapter):
    """Huoshan realtime automatic speech recognition client.

    This client provides real-time speech recognition using Huoshan's streaming
    ASR API through WebSocket connections. It supports streaming audio input
    and provides continuous transcription results.
    """

    AVAILABLE_FOR_STREAM = True
    FRAME_RATE: int = 16000
    CHUNK_DURATION = 0.04
    CHUNK_N_BYTES = 1280

    HOSTNAME_FULL = socket.gethostname()

    CLIENT_FULL_REQUEST_TYPE = 0b0001
    CLIENT_AUDIO_ONLY_REQUEST_TYPE = 0b0010
    SERVER_FULL_RESPONSE_TYPE = 0b1001
    SERVER_ACK_TYPE = 0b1011
    SERVER_ERROR_RESPONSE_TYPE = 0b1111

    NO_SEQUENCE_FLAG = 0b0000
    NEG_SEQUENCE_FLAG = 0b0010

    NO_SERIALIZATION = 0b0000
    JSON_SERIALIZATION = 0b0001
    GZIP_COMPRESSION = 0b0001
    ExecutorRegistry.register_class("HuoshanASRClient")

    def __init__(
        self,
        name: str,
        wss_url: str,
        cluster_id: str = "volcengine_streaming_common",
        default_language: str = "zh-CN",
        request_timeout: float = 20.0,
        queue_size: int = 100,
        sleep_time: float = 0.01,
        clean_interval: float = 10.0,
        expire_time: float = 120.0,
        commit_timeout: float = 10.0,
        max_workers: int = 1,
        thread_pool_executor: ThreadPoolExecutor | None = None,
        logger_cfg: Union[None, Dict[str, Any]] = None,
    ):
        """Initialize the Huoshan ASR client.

        Args:
            name (str):
                The name of the client. Logger's name will be set to this name.
            wss_url (str):
                The WebSocket URL for the Huoshan ASR API.
            cluster_id (str, optional):
                The cluster ID for the Huoshan ASR service.
                Defaults to 'volcengine_streaming_common'.
            default_language (str, optional):
                The default language code for speech recognition.
                Defaults to 'zh-CN'.
            request_timeout (float, optional):
                The timeout for the Huoshan ASR API request in seconds.
                Defaults to 20.0.
            queue_size (int, optional):
                The size of the queue for the audio input buffer.
                Defaults to 100.
            sleep_time (float, optional):
                The sleep time in seconds between polling operations.
                Defaults to 0.01.
            clean_interval (float, optional):
                The interval in seconds for cleaning expired entries in the
                input buffer. Defaults to 10.0.
            expire_time (float, optional):
                The time to live in seconds for entries in the input buffer.
                Defaults to 120.0.
            commit_timeout (float, optional):
                The timeout in seconds for waiting for ASR response after
                sending commit message. Defaults to 10.0.
            max_workers (int, optional):
                The number of workers for the thread pool executor.
                Defaults to 1.
            thread_pool_executor (ThreadPoolExecutor | None, optional):
                External thread pool executor. If None, a new executor will
                be created. Defaults to None.
            logger_cfg (Union[None, Dict[str, Any]], optional):
                Logger configuration, see `setup_logger` for detailed
                description. Logger name will use the class name.
                Defaults to None.
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
        self.cluster_id = cluster_id
        self.default_language = default_language
        self.request_timeout = request_timeout
        self.commit_timeout = commit_timeout

    def _construct_request(self, request_id: str, appid: str, token: str, language: str) -> dict[str, Any]:
        """Construct the request dictionary for Huoshan ASR API.

        Args:
            request_id (str):
                The unique request ID for tracking the ASR request.
            appid (str):
                The Huoshan application ID.
            token (str):
                The authentication token for Huoshan ASR service.
            language (str):
                The language code for speech recognition.

        Returns:
            dict[str, Any]:
                The request dictionary containing app, user, request, and
                audio configuration.
        """
        req_dict = {
            "app": {
                "appid": appid,
                "cluster": self.cluster_id,
                "token": token,
            },
            "user": {"uid": self.__class__.HOSTNAME_FULL},
            "request": {
                "reqid": request_id,
                "show_utterances": False,
                "result_type": "single",
                "sequence": 1,
                "workflow": "audio_in,resample,partition,vad,fe,decode,itn,nlu_punctuate",
            },
            "audio": {
                "format": "raw",
                "rate": self.__class__.FRAME_RATE,
                "language": language,
                "bits": 16,
                "channel": 1,
            },
        }
        return req_dict

    async def _create_connection(self, request_id: str, cur_time: float) -> None:
        """Create a connection to the server.

        Args:
            request_id (str):
                The request ID for tracking the connection.
            cur_time (float):
                The current timestamp for connection timing.
        """
        app_id = self.input_buffer[request_id].get("api_keys", {}).get("huoshan_app_id", "")
        token = self.input_buffer[request_id].get("api_keys", {}).get("huoshan_token", "")
        if not app_id or not token:
            msg = "Huoshan app ID or token is not found in the API keys."
            self.logger.error(msg)
            raise MissingAPIKeyException(msg)
        language = self.input_buffer[request_id].get("language", self.default_language)
        request_dict = self._construct_request(request_id, app_id, token, language)
        loop = asyncio.get_event_loop()
        request_bytes = await loop.run_in_executor(
            self.executor,
            _encode_request,
            request_dict,
        )
        header = _generate_header(self.__class__.CLIENT_FULL_REQUEST_TYPE, self.__class__.NO_SEQUENCE_FLAG)
        full_client_request = bytearray(header)
        # payload size(4 bytes)
        full_client_request.extend((len(request_bytes)).to_bytes(4, "big"))
        # payload
        full_client_request.extend(request_bytes)
        token_header = {"Authorization": "Bearer; {}".format(token)}
        try:
            ws = await websockets.connect(self.wss_url, additional_headers=token_header, max_size=100 * 1024 * 1024)
            await ws.send(full_client_request)
            res = await ws.recv()
            result = self._parse_response(res)
            if "payload_msg" in result and result["payload_msg"]["code"] != 1000:
                msg = f"Huoshan ASR server returned error {result['payload_msg']} for request {request_id}"
                self.logger.error(msg)
                raise HuoshanASRClientError(msg)
            self.input_buffer[request_id]["ws_client"] = ws
            self.input_buffer[request_id]["commit_time"] = None
        except Exception as e:
            self.input_buffer[request_id]["connection_failed"] = True
            traceback_str = traceback.format_exc()
            self.logger.error(f"Failed to create connection to Huoshan ASR server: {e}\n{traceback_str}")
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
        # Resample audio if needed
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
        payload_bytes = await loop.run_in_executor(
            self.executor,
            gzip.compress,
            pcm_bytes,
        )
        header = _generate_header(self.__class__.CLIENT_AUDIO_ONLY_REQUEST_TYPE, self.__class__.NO_SEQUENCE_FLAG)
        audio_only_request = bytearray(header)
        audio_only_request.extend((len(payload_bytes)).to_bytes(4, "big"))
        audio_only_request.extend(payload_bytes)
        # Send audio-only client request
        await ws_client.send(audio_only_request)
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
        mute_chunk_bytes = bytes(self.__class__.CHUNK_N_BYTES)
        payload_bytes = await loop.run_in_executor(
            self.executor,
            gzip.compress,
            mute_chunk_bytes,
        )
        header = _generate_header(self.__class__.CLIENT_AUDIO_ONLY_REQUEST_TYPE, self.__class__.NEG_SEQUENCE_FLAG)
        audio_only_request = bytearray(header)
        audio_only_request.extend((len(payload_bytes)).to_bytes(4, "big"))
        audio_only_request.extend(payload_bytes)
        await ws_client.send(audio_only_request)
        commit_time = time.time()
        self.input_buffer[request_id]["commit_time"] = commit_time
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
        while True:
            current_time = time.time()
            if current_time - commit_time > self.commit_timeout:
                break
            try:
                remaining_time = self.commit_timeout - (current_time - commit_time)
                if remaining_time <= 0:
                    break
                message = await asyncio.wait_for(ws_client.recv(), timeout=remaining_time)
                result = self._parse_response(message)
                if "payload_msg" in result and result["payload_msg"]["code"] != 1000:
                    msg = f"Huoshan ASR server returned error {result['payload_msg']} for request {request_id}"
                    self.logger.error(msg)
                    break
                if "payload_msg" in result and result["payload_msg"]["sequence"] < 0:
                    asr_text = result["payload_msg"]["result"][0]["text"]
                    break
            except asyncio.TimeoutError:
                self.logger.error(
                    f"ASR timeout for request {request_id}: "
                    + f"spent {current_time - commit_time:.3f} seconds, "
                    + f"timeout {self.commit_timeout:.3f} seconds"
                )
                break
            except Exception as e:
                self.logger.error(f"ASR error for request {request_id}: {e}")
                break
        if not asr_text:
            self.logger.warning(f"ASR text is empty for request {request_id}, use default text.")
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

    def _parse_response(self, res: bytes) -> dict[str, Any]:
        """Parse the response message from Huoshan ASR server.

        The response message format consists of:
        - protocol_version (4 bits), header_size (4 bits)
        - message_type (4 bits), message_type_specific_flags (4 bits)
        - serialization_method (4 bits), message_compression (4 bits)
        - reserved (8 bits) - reserved field
        - header_extensions - extension header with size equal to
          8 * 4 * (header_size - 1) bytes
        - payload - similar to HTTP request body

        Args:
            res (bytes):
                The raw response bytes from the server.

        Returns:
            dict[str, Any]:
                Parsed response dictionary containing:
                - 'seq' (int, optional): Sequence number for ACK messages
                - 'code' (int, optional): Error code for error messages
                - 'payload_msg' (dict | str, optional): Parsed payload message
                - 'payload_size' (int): Size of the payload in bytes
        """
        # protocol_version
        _ = res[0] >> 4
        header_size = res[0] & 0x0F
        message_type = res[1] >> 4
        # message_type_specific_flags
        _ = res[1] & 0x0F
        serialization_method = res[2] >> 4
        message_compression = res[2] & 0x0F
        # reserved
        _ = res[3]
        # header_extensions
        _ = res[4 : header_size * 4]
        payload = res[header_size * 4 :]
        result = {}
        payload_msg = None
        payload_size = 0
        if message_type == self.__class__.SERVER_FULL_RESPONSE_TYPE:
            payload_size = int.from_bytes(payload[:4], "big", signed=True)
            payload_msg = payload[4:]
        elif message_type == self.__class__.SERVER_ACK_TYPE:
            seq = int.from_bytes(payload[:4], "big", signed=True)
            result["seq"] = seq
            if len(payload) >= 8:
                payload_size = int.from_bytes(payload[4:8], "big", signed=False)
                payload_msg = payload[8:]
        elif message_type == self.__class__.SERVER_ERROR_RESPONSE_TYPE:
            code = int.from_bytes(payload[:4], "big", signed=False)
            result["code"] = code
            payload_size = int.from_bytes(payload[4:8], "big", signed=False)
            payload_msg = payload[8:]
        if payload_msg is None:
            return result
        if message_compression == self.__class__.GZIP_COMPRESSION:
            payload_msg = gzip.decompress(payload_msg)
        if serialization_method == self.__class__.JSON_SERIALIZATION:
            payload_msg = json.loads(str(payload_msg, "utf-8"))
        elif serialization_method != self.__class__.NO_SERIALIZATION:
            payload_msg = str(payload_msg, "utf-8")
        result["payload_msg"] = payload_msg
        result["payload_size"] = payload_size
        return result


def _encode_request(request_dict: dict[str, Any]) -> bytes:
    """Encode and compress the request dictionary.

    Args:
        request_dict (dict[str, Any]):
            The request dictionary to encode.

    Returns:
        bytes:
            The compressed JSON-encoded request bytes.
    """
    payload_bytes = str.encode(json.dumps(request_dict))
    payload_bytes = gzip.compress(payload_bytes)
    return payload_bytes


def _generate_header(
    message_type: int, message_type_specific_flags: int, extension_header: bytes | None = None
) -> bytearray:
    """Generate the protocol header for Huoshan ASR messages.

    The header format consists of:
    - protocol_version (4 bits), header_size (4 bits)
    - message_type (4 bits), message_type_specific_flags (4 bits)
    - serialization_method (4 bits), message_compression (4 bits)
    - reserved (8 bits) - reserved field
    - header_extensions - extension header with size equal to
      8 * 4 * (header_size - 1) bytes

    Args:
        message_type (int):
            The message type identifier (e.g., CLIENT_FULL_REQUEST_TYPE).
        message_type_specific_flags (int):
            The message type specific flags (e.g., NO_SEQUENCE_FLAG).
        extension_header (bytes | None, optional):
            Optional extension header bytes. Defaults to None.

    Returns:
        bytearray:
            The generated protocol header as a bytearray.
    """
    header = bytearray()
    extension_header_len = 0 if extension_header is None else len(extension_header)
    header_size = int(extension_header_len / 4) + 1
    # PROTOCOL_VERSION | header_size
    header.append((0b0001 << 4) | header_size)
    # message_type | message_type_specific_flags
    header.append((message_type << 4) | message_type_specific_flags)
    # JSON | GZIP
    header.append((0b0001 << 4) | 0b0001)
    # reserved_data
    header.append(0x00)
    if extension_header is not None:
        header.extend(extension_header)
    return header
