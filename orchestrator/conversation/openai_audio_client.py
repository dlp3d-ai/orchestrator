import asyncio
import base64
import io
import json
import time
import traceback
import wave
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Union

import websockets
from prometheus_client import Histogram

from ..data_structures.audio_chunk import (
    AudioWithSubtitleChunkBody,
    AudioWithSubtitleChunkEnd,
    AudioWithSubtitleChunkStart,
)
from ..data_structures.process_flow import DAGStatus
from ..utils.audio import resample_pcm
from ..utils.exception import MissingAPIKeyException
from ..utils.executor_registry import ExecutorRegistry
from .audio_conversation_adapter import AudioConversationAdapter


class OpenAIAudioClient(AudioConversationAdapter):
    """OpenAI realtime audio conversation client.

    This client handles real-time audio conversations using OpenAI's Realtime
    API. It supports streaming audio input and output with WebSocket
    connections. The client processes PCM audio data and provides real-time
    transcription and audio generation capabilities.
    """

    AVAILABLE_FOR_STREAM = True
    N_CHANNELS: int = 1
    SAMPLE_WIDTH: int = 2
    FRAME_RATE: int = 24000
    ExecutorRegistry.register_class("OpenAIAudioClient")
    CHUNK_DURATION = 0.04

    def __init__(
        self,
        name: str,
        agent_prompts_file: str,
        wss_url: str,
        openai_model_name: str = "gpt-4o-mini-realtime-preview-2024-12-17",
        proxy_url: Union[None, str] = None,
        request_timeout: float = 20.0,
        queue_size: int = 100,
        sleep_time: float = 0.01,
        clean_interval: float = 10.0,
        expire_time: float = 120.0,
        max_workers: int = 1,
        thread_pool_executor: ThreadPoolExecutor | None = None,
        latency_histogram: Histogram | None = None,
        input_token_number_histogram: Histogram | None = None,
        output_token_number_histogram: Histogram | None = None,
        token_number_histogram: Histogram | None = None,
        logger_cfg: Union[None, Dict[str, Any]] = None,
    ):
        """Initialize the OpenAI realtime audio conversation client.

        Args:
            name (str):
                The name of the conversation adapter.
            agent_prompts_file (str):
                The path to the agent prompts file containing conversation instructions.
            wss_url (str):
                The WebSocket URL for the OpenAI realtime API connection.
            openai_model_name (str, optional):
                The OpenAI model name to use for realtime conversations.
                Defaults to "gpt-4o-mini-realtime-preview-2024-12-17".
            proxy_url (Union[None, str], optional):
                The proxy URL for the WebSocket connection.
                Defaults to None.
            request_timeout (float, optional):
                The timeout for WebSocket operations in seconds.
                Defaults to 20.0.
            queue_size (int, optional):
                The maximum size of the input queue for audio chunks.
                Defaults to 100.
            sleep_time (float, optional):
                The sleep interval between operations in seconds.
                Defaults to 0.01.
            clean_interval (float, optional):
                The interval to clean expired requests in seconds.
                Defaults to 10.0.
            expire_time (float, optional):
                The time after which requests expire in seconds.
                Defaults to 120.0.
            max_workers (int, optional):
                The maximum number of worker threads for the thread pool.
                Defaults to 1.
            thread_pool_executor (ThreadPoolExecutor | None, optional):
                External thread pool executor to use. If None, creates a new one.
                Defaults to None.
            latency_histogram (Histogram | None, optional):
                Prometheus Histogram metric for recording request latency distribution
                in seconds. If provided, latency metrics will be collected for monitoring
                purposes. Defaults to None.
            input_token_number_histogram (Histogram | None, optional):
                Prometheus Histogram metric for recording input token count distribution
                per request. If provided, input token usage metrics will be collected for
                monitoring purposes. Defaults to None.
            output_token_number_histogram (Histogram | None, optional):
                Prometheus Histogram metric for recording output token count distribution
                per request. If provided, output token usage metrics will be collected for
                monitoring purposes. Defaults to None.
            logger_cfg (Union[None, Dict[str, Any]], optional):
                Logger configuration dictionary. Defaults to None.
        """
        AudioConversationAdapter.__init__(
            self,
            name=name,
            agent_prompts_file=agent_prompts_file,
            proxy_url=proxy_url,
            request_timeout=request_timeout,
            queue_size=queue_size,
            sleep_time=sleep_time,
            clean_interval=clean_interval,
            expire_time=expire_time,
            latency_histogram=latency_histogram,
            input_token_number_histogram=input_token_number_histogram,
            output_token_number_histogram=output_token_number_histogram,
            logger_cfg=logger_cfg,
        )
        self.wss_url = wss_url
        self.openai_model_name = openai_model_name
        self.executor = (
            thread_pool_executor if thread_pool_executor is not None else ThreadPoolExecutor(max_workers=max_workers)
        )
        self.executor_external = True if thread_pool_executor is not None else False

    def __del__(self) -> None:
        """Destructor for cleanup of thread pool executor.

        Ensures that the thread pool executor is properly shutdown when the
        client is destroyed, but only if it was created internally.
        """
        if not self.executor_external:
            self.executor.shutdown(wait=True)

    async def _create_session(
        self,
        request_id: str,
        cascade_memories: Union[None, Dict[str, Any]],
        voice_name: str = "alloy",
    ) -> None:
        """Create a WebSocket session with OpenAI's realtime API.

        Establishes a WebSocket connection to OpenAI's realtime API and
        initializes the session with conversation history and instructions.

        Args:
            request_id (str):
                Unique identifier for the conversation request.
            cascade_memories (Union[None, Dict[str, Any]]):
                Memory context from previous conversations, or None if no context.
            voice_name (str, optional):
                The voice to use for audio generation. Defaults to "alloy".
        """
        conversation_model_override = self.input_buffer[request_id]["conversation_model_override"]
        openai_model_name = (
            conversation_model_override if conversation_model_override is not None else self.openai_model_name
        )
        wss_url = self.wss_url + "?model=" + openai_model_name

        openai_api_key = self.input_buffer[request_id]["api_keys"].get("openai_api_key", "")
        if not openai_api_key:
            msg = "OpenAI API key is not found in the API keys."
            self.logger.error(msg)
            raise MissingAPIKeyException(msg)

        headers = {
            "Authorization": f"Bearer {openai_api_key}",
            "OpenAI-Beta": "realtime=v1",
        }
        ws = await websockets.connect(
            wss_url,
            proxy=self.proxy_url,
            additional_headers=headers,
            ping_interval=self.request_timeout,
            ping_timeout=self.request_timeout,
        )
        self.input_buffer[request_id]["ws_client"] = ws
        self.logger.info(f"OpenAI realtime client created for request {request_id}")
        memory_adapter = self.input_buffer[request_id]["memory_adapter"]
        conversation_history = await memory_adapter.build_chat_history(cascade_memories)
        loop = asyncio.get_event_loop()
        conversation_history = await loop.run_in_executor(
            self.executor,
            _json_dumps_not_ensure_ascii,
            {
                "CHAT_HISTORY": conversation_history,
            },
        )
        conversation_prompt = self.input_buffer[request_id]["user_prompt"]
        prompt = conversation_prompt + "\n" + conversation_history
        data = {
            "type": "session.update",
            "session": {
                "turn_detection": None,
                "instructions": prompt,
                "voice": voice_name,
                "input_audio_transcription": {"model": "gpt-4o-mini-transcribe"},
            },
        }
        data = await loop.run_in_executor(self.executor, json.dumps, data)
        await ws.send(data)
        await self._send_stream_start_task(request_id)
        asyncio.create_task(self._receive_pcm(request_id))

    async def _send_audio(self, request_id: str, audio_bytes: bytes, seq_number: int) -> None:
        """Send audio data to the OpenAI realtime API.

        Processes and sends audio chunks to the WebSocket connection.
        Handles both PCM and WAV audio formats with automatic resampling.

        Args:
            request_id (str):
                Unique identifier for the conversation request.
            audio_bytes (bytes):
                Raw audio data bytes to send.
            seq_number (int):
                Sequence number of the audio chunk for ordering.
        """
        loop = asyncio.get_event_loop()
        start_time = time.time()
        audio_type = self.input_buffer[request_id]["audio_type"]
        if audio_type == "pcm":
            pcm_bytes = audio_bytes
            frame_rate = self.input_buffer[request_id]["frame_rate"]
        elif audio_type == "wav":
            with wave.open(io.BytesIO(audio_bytes), "rb") as wav_file:
                pcm_bytes = wav_file.readframes(wav_file.getnframes())
                frame_rate = wav_file.getframerate()
        else:
            msg = f"Unknown audio type: {audio_type} for request {request_id}"
            self.logger.error(msg)
            dag = self.input_buffer[request_id]["dag"]
            dag.set_status(DAGStatus.FAILED)
            return
        while "ws_client" not in self.input_buffer[request_id]:
            await asyncio.sleep(self.sleep_time)
            if time.time() - start_time > self.request_timeout:
                msg = (
                    f"OpenAI realtime client not connected after {self.request_timeout} seconds"
                    + f" for request {request_id}"
                )
                self.logger.error(msg)
                return
        ws = self.input_buffer[request_id]["ws_client"]
        if frame_rate != self.__class__.FRAME_RATE:
            pcm_bytes = await loop.run_in_executor(
                self.executor,
                resample_pcm,
                pcm_bytes,
                frame_rate,
                self.__class__.FRAME_RATE,
            )
        audio_buffer = await loop.run_in_executor(
            self.executor,
            self._encode_pcm,
            pcm_bytes,
        )
        message = await loop.run_in_executor(
            self.executor, json.dumps, {"type": "input_audio_buffer.append", "audio": audio_buffer}
        )
        while self.input_buffer[request_id]["input_chunk_sent"] < seq_number:
            await asyncio.sleep(self.sleep_time)
        await ws.send(message)
        self.input_buffer[request_id]["input_chunk_sent"] += 1

    async def _commit_audio(self, request_id: str) -> None:
        """Commit audio input to the OpenAI realtime API.

        Signals the end of audio input and triggers the AI response generation.
        Waits for all audio chunks to be sent before committing.

        Args:
            request_id (str):
                Unique identifier for the conversation request.
        """
        start_time = time.time()
        self.input_buffer[request_id]["commit_time"] = start_time
        dag = self.input_buffer[request_id]["dag"]
        while "ws_client" not in self.input_buffer[request_id]:
            await asyncio.sleep(self.sleep_time)
            if time.time() - start_time > self.request_timeout:
                msg = (
                    f"OpenAI realtime client not connected after {self.request_timeout} seconds"
                    + f" for request {request_id}"
                )
                self.logger.error(msg)
                return
        ws = self.input_buffer[request_id]["ws_client"]
        while self.input_buffer[request_id]["input_chunk_sent"] < self.input_buffer[request_id]["input_chunk_received"]:
            if self.input_buffer[request_id]["dag"].status == DAGStatus.RUNNING:
                await asyncio.sleep(self.sleep_time)
            else:
                msg = f"Audio conversation sending commit audio interrupted by DAG status {dag.status}"
                msg = msg + f" for request {request_id}"
                self.logger.warning(msg)
                return
        # too short to call thread pool executor
        await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
        await ws.send(json.dumps({"type": "response.create"}))

    async def _receive_pcm(self, request_id: str) -> None:
        """Receive audio response from the OpenAI realtime API.

        Handles incoming WebSocket messages and processes audio responses.
        Manages conversation state, transcription, and memory updates.

        Args:
            request_id (str):
                Unique identifier for the conversation request.
        """
        ws = self.input_buffer[request_id]["ws_client"]
        dag_start_time = self.input_buffer[request_id]["dag_start_time"]
        user_input = ""
        assistant_output = ""
        user_input_done = False
        assistant_output_done = False
        history_done = False
        response_done = False
        loop = asyncio.get_event_loop()
        input_token_number = 0
        output_token_number = 0
        while True:
            try:
                message = await ws.recv()
                cur_time = time.time()
                self.input_buffer[request_id]["last_update_time"] = cur_time
                message = json.loads(message)
                event_type = message["type"]
                if event_type == "input_audio_buffer.committed":
                    self.input_buffer[request_id]["committed_time"] = time.time()
                if event_type != "response.audio.delta":
                    self.logger.debug(f"{event_type}: {message}")
                if event_type == "response.audio.delta":
                    seq_number = self.input_buffer[request_id]["chunk_received"]
                    self.input_buffer[request_id]["chunk_received"] += 1
                    pcm_bytes = await loop.run_in_executor(self.executor, self._decode_pcm, message["delta"])
                    ret_dict = await loop.run_in_executor(self.executor, self.convert_tts_time, pcm_bytes)
                    audio_io = io.BytesIO(pcm_bytes)
                    audio_io.seek(0)
                    ret_dict["audio"] = audio_io
                    if seq_number == 0:
                        start_time = self.input_buffer[request_id]["commit_time"]
                        cur_time = time.time()
                        latency = cur_time - start_time
                        msg = f"request {request_id} audio LLM first chunk latency from commit: {latency:.2f} seconds"
                        if dag_start_time is not None:
                            time_diff = cur_time - dag_start_time
                            msg += f", from dag start: {time_diff:.2f} seconds"
                        if self.input_buffer[request_id]["committed_time"] is not None:
                            time_diff = cur_time - self.input_buffer[request_id]["committed_time"]
                            msg += f", from received committed: {time_diff:.2f} seconds"
                        self.logger.debug(msg)
                        if self.latency_histogram:
                            user_id = self.input_buffer[request_id]["user_id"]
                            self.latency_histogram.labels(adapter=self.name, user_id=user_id).observe(latency)
                    await self._send_audio_to_downstream(request_id, ret_dict, seq_number)
                # elif event_type == "response.audio_transcript.delta":
                #     self.logger.debug(f"response.audio_transcript.delta: {message['delta']}")
                elif event_type == "conversation.item.input_audio_transcription.completed":
                    user_input = message["transcript"]
                    user_input_done = True
                elif event_type == "response.audio_transcript.done":
                    assistant_output = message["transcript"]
                    assistant_output_done = True
                elif event_type == "response.done":
                    input_token_number = message["response"]["usage"]["input_tokens"]
                    output_token_number = message["response"]["usage"]["output_tokens"]
                    response_done = True
                elif event_type == "error":
                    error_msg = message["error"]
                    self.logger.error(f"Error: {error_msg}")
                    return
                if user_input_done and assistant_output_done:
                    if not history_done:
                        memory_db_client = self.input_buffer[request_id]["memory_db_client"]
                        timezone = self.input_buffer[request_id]["timezone"]
                        if memory_db_client is not None:
                            cur_time = time.time()
                            await memory_db_client.append_chat_history(
                                character_id=self.input_buffer[request_id]["character_id"],
                                unix_timestamp=cur_time,
                                role="user",
                                content=user_input,
                                relationship=self.input_buffer[request_id]["relationship"],
                                timezone=timezone,
                            )
                            await memory_db_client.append_chat_history(
                                character_id=self.input_buffer[request_id]["character_id"],
                                unix_timestamp=cur_time,
                                role="assistant",
                                content=assistant_output,
                                timezone=timezone,
                                **self.input_buffer[request_id]["emotion"],
                            )
                        history_done = True
                    if response_done:
                        start_time = self.input_buffer[request_id]["commit_time"]
                        end_time = time.time()
                        msg = f"Streaming audio conversation with the OpenAI Realtime API took {end_time - start_time} seconds"
                        msg = msg + f" for request {request_id}"
                        self.logger.info(msg)
                        if self.input_token_number_histogram:
                            self.input_token_number_histogram.labels(adapter=self.name, user_id=user_id).observe(
                                input_token_number
                            )
                        if self.output_token_number_histogram:
                            self.output_token_number_histogram.labels(adapter=self.name, user_id=user_id).observe(
                                output_token_number
                            )
                        await self._close_session(request_id)
                        break
            except Exception as e:
                msg = f"Error in streaming audio conversation: {e}"
                msg = msg + f" for request {request_id}"
                traceback_str = traceback.format_exc()
                msg += f"\n{traceback_str}"
                self.logger.error(msg)
                return

    async def _close_session(self, request_id: str) -> None:
        """Close the WebSocket session with OpenAI's realtime API.

        Closes the WebSocket connection and sends stream end task to downstream nodes.

        Args:
            request_id (str):
                Unique identifier for the conversation request.
        """
        ws = self.input_buffer[request_id]["ws_client"]
        await ws.close()
        await self._send_stream_end_task(request_id)

    async def _send_audio_to_downstream(
        self,
        request_id: str,
        ret_dict: Dict[str, Any],
        seq_number: int,
    ) -> None:
        """Send processed audio data to downstream processing nodes.

        Forwards audio chunks with timing and transcription information
        to the next nodes in the processing pipeline.

        Args:
            request_id (str):
                Unique identifier for the conversation request.
            ret_dict (Dict[str, Any]):
                Dictionary containing audio data, duration, and speech timing.
            seq_number (int):
                Sequence number of the audio chunk for ordering.
        """
        try:
            dag = self.input_buffer[request_id]["dag"]
            while self.input_buffer[request_id]["chunk_sent"] < seq_number:
                if dag.status == DAGStatus.RUNNING:
                    await asyncio.sleep(self.sleep_time)
                else:
                    msg = f"Streaming audio conversation interrupted by DAG status {dag.status}"
                    msg = msg + f" for request {request_id}"
                    msg += f", seq_number: {seq_number}"
                    self.logger.warning(msg)
                    return
            # prepare downstream
            node_name = self.input_buffer[request_id]["node_name"]
            dag_node = dag.get_node(node_name)
            downstream_nodes = dag_node.downstreams
            downstream_warned = self.input_buffer[request_id]["downstream_warned"]
            if len(downstream_nodes) == 0 and not downstream_warned:
                self.logger.warning(f"Request {request_id} has no downstreams, so the result is discarded.")
                self.input_buffer[request_id]["downstream_warned"] = True
                return
            coroutines = list()
            for node in downstream_nodes:
                payload = node.payload
                body_trunk = AudioWithSubtitleChunkBody(
                    request_id=request_id,
                    duration=ret_dict["duration"],
                    audio_io=ret_dict["audio"],
                    seq_number=seq_number,
                    speech_text=ret_dict["speech_text"],
                    speech_time=ret_dict["speech_time"],
                )
                coroutines.append(payload.feed_stream(body_trunk))
            asyncio.gather(*coroutines)
            self.input_buffer[request_id]["chunk_sent"] += 1
        except Exception as e:
            msg = f"Error in streaming audio conversation: {e}"
            msg = msg + f" for request {request_id}"
            traceback_str = traceback.format_exc()
            msg += f"\n{traceback_str}"
            self.logger.error(msg)
            dag = self.input_buffer[request_id]["dag"]
            dag.set_status(DAGStatus.FAILED)
            return

    def convert_tts_time(self, pcm_bytes: bytes) -> Dict[str, Any]:
        """Convert PCM audio data to timing information for speech synthesis.

        Calculates duration and creates timing metadata for audio chunks.
        Note: OpenAI's text and audio are not aligned, so random motion is used.

        Args:
            pcm_bytes (bytes):
                Raw PCM audio data bytes.

        Returns:
            Dict[str, Any]:
                Dictionary containing speech_text, duration, and speech_time.
                speech_text is set to "-" due to alignment issues.
                speech_time contains timing information for the audio chunk.
        """
        duration = len(pcm_bytes) / (
            self.__class__.N_CHANNELS * self.__class__.SAMPLE_WIDTH * self.__class__.FRAME_RATE
        )
        # NOTE: OpenAI's text and audio are not aligned, so random motion is used.
        speech_text = "-"
        speech_time = [(0, 0)]
        return {
            "speech_text": speech_text,
            "duration": duration,
            "speech_time": speech_time,
        }

    def _encode_pcm(self, pcm_bytes: bytes) -> str:
        """Encode PCM audio data to base64 format for WebSocket transmission.

        Args:
            pcm_bytes (bytes):
                Raw PCM audio data bytes.

        Returns:
            str:
                Base64 encoded string representation of the PCM data.
        """
        encoded_pcm = base64.b64encode(pcm_bytes).decode("utf-8")
        return encoded_pcm

    def _decode_pcm(self, encoded_pcm: str) -> bytes:
        """Decode base64 encoded PCM data back to raw bytes.

        Args:
            encoded_pcm (str):
                Base64 encoded string representation of PCM data.

        Returns:
            bytes:
                Raw PCM audio data bytes.
        """
        pcm_bytes = base64.b64decode(encoded_pcm)
        return pcm_bytes

    async def _send_stream_start_task(self, request_id: str) -> None:
        """Send stream start signal to downstream processing nodes.

        Notifies downstream nodes that audio streaming has begun and
        provides audio format information.

        Args:
            request_id (str):
                Unique identifier for the conversation request.
        """
        # prepare downstream
        dag = self.input_buffer[request_id]["dag"]
        node_name = self.input_buffer[request_id]["node_name"]
        dag_node = dag.get_node(node_name)
        downstream_nodes = dag_node.downstreams
        downstream_warned = self.input_buffer[request_id]["downstream_warned"]
        if len(downstream_nodes) == 0 and not downstream_warned:
            self.logger.warning(f"Request {request_id} has no downstreams, so the result is discarded.")
            self.input_buffer[request_id]["downstream_warned"] = True
            return
        coroutines = list()
        for node in downstream_nodes:
            next_node_name = node.name
            payload = node.payload
            start_trunk = AudioWithSubtitleChunkStart(
                request_id=request_id,
                audio_type="pcm",
                n_channels=self.__class__.N_CHANNELS,
                sample_width=self.__class__.SAMPLE_WIDTH,
                frame_rate=self.__class__.FRAME_RATE,
                node_name=next_node_name,
                dag=dag,
            )
            coroutines.append(payload.feed_stream(start_trunk))
        asyncio.gather(*coroutines)

    async def _send_stream_end_task(self, request_id: str) -> None:
        """Send stream end signal to downstream processing nodes.

        Notifies downstream nodes that audio streaming has completed and
        cleans up the request from the input buffer.

        Args:
            request_id (str):
                Unique identifier for the conversation request.
        """
        try:
            dag = self.input_buffer[request_id]["dag"]
        except KeyError:
            msg = f"Request {request_id} not found in input buffer, skip sending stream end task."
            self.logger.warning(msg)
            return
        while self.input_buffer[request_id]["chunk_received"] > self.input_buffer[request_id]["chunk_sent"]:
            if dag.status == DAGStatus.RUNNING:
                await asyncio.sleep(self.sleep_time)
            else:
                msg = f"Audio conversation sending stream end task interrupted by DAG status {dag.status}"
                msg = msg + f" for request {request_id}"
                self.logger.warning(msg)
                return
        # prepare downstream
        dag = self.input_buffer[request_id]["dag"]
        node_name = self.input_buffer[request_id]["node_name"]
        dag_node = dag.get_node(node_name)
        downstream_nodes = dag_node.downstreams
        downstream_warned = self.input_buffer[request_id]["downstream_warned"]
        if len(downstream_nodes) == 0 and not downstream_warned:
            self.logger.warning(f"Request {request_id} has no downstreams, so the result is discarded.")
            self.input_buffer[request_id]["downstream_warned"] = True
            return
        coroutines = list()
        for node in downstream_nodes:
            payload = node.payload
            end_trunk = AudioWithSubtitleChunkEnd(
                request_id=request_id,
            )
            coroutines.append(payload.feed_stream(end_trunk))
        asyncio.gather(*coroutines)
        self.input_buffer.pop(request_id)


def _json_dumps_not_ensure_ascii(obj: Any, **kwargs: Any) -> str:
    """Serialize object to JSON string without ASCII encoding.

    Ensures that Unicode characters are properly encoded in the JSON output
    by setting ensure_ascii=False.

    Args:
        obj (Any):
            The object to serialize to JSON.
        **kwargs (Any):
            Additional keyword arguments for json.dumps.

    Returns:
        str:
            JSON string representation of the object with Unicode support.
    """
    kwargs = kwargs.copy()
    if "ensure_ascii" in kwargs:
        kwargs.pop("ensure_ascii")
    return json.dumps(obj, ensure_ascii=False, **kwargs)
