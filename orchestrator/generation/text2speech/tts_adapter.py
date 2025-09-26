import asyncio
import io
import re
import time
import traceback
from abc import abstractmethod
from typing import Any, Dict, Union

from ...data_structures.audio_chunk import (
    AudioWithSubtitleChunkBody,
    AudioWithSubtitleChunkEnd,
    AudioWithSubtitleChunkStart,
)
from ...data_structures.process_flow import DAGStatus
from ...data_structures.text_chunk import TextChunkBody, TextChunkEnd, TextChunkStart
from ...utils.log import setup_logger
from ...utils.sentence_splitter import SentenceSplitter
from ...utils.streamable import ChunkWithoutStartError, Streamable


class TextToSpeechAdapter(Streamable):
    """Base text-to-speech adapter for streaming audio generation.

    This abstract base class provides the core functionality for converting
    text chunks to speech audio streams. It handles text segmentation,
    streaming processing, and downstream audio distribution. Subclasses must
    implement the specific TTS provider integration.
    """

    AVAILABLE_FOR_STREAM = False
    DOWNSTREAM_FRAME_RATE: int = 16000

    def __init__(
        self,
        name: str,
        queue_size: int = 100,
        sleep_time: float = 0.01,
        clean_interval: float = 10.0,
        expire_time: float = 120.0,
        logger_cfg: Union[None, Dict[str, Any]] = None,
    ):
        """Initialize the text-to-speech adapter.

        Args:
            name (str):
                Unique name for this TTS adapter instance.
                Used as the logger name for identification.
            queue_size (int, optional):
                Maximum number of requests to queue for processing.
                Defaults to 100.
            sleep_time (float, optional):
                Sleep duration between processing iterations in seconds.
                Defaults to 0.01.
            clean_interval (float, optional):
                Interval in seconds for cleaning expired requests.
                Defaults to 10.0.
            expire_time (float, optional):
                Time in seconds after which requests expire.
                Defaults to 120.0.
            logger_cfg (Union[None, Dict[str, Any]], optional):
                Logger configuration dictionary. If None, default
                logging configuration is used. Defaults to None.
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
        self.sentence_splitter = SentenceSplitter(logger=self.logger)

    @abstractmethod
    async def _generate_tts(
        self,
        request_id: str,
        text: str,
        voice_name: str,
        voice_speed: float = 1.0,
        voice_style: Union[None, str] = None,
        language: str = "zh",
        start_time: float = 0.0,
    ) -> Dict[str, Any]:
        """Generate TTS audio from text using the specific provider.

        Args:
            request_id (str):
                Unique identifier for the TTS request.
            text (str):
                Input text to convert to speech.
            voice_name (str):
                Voice identifier for the TTS provider.
            voice_speed (float, optional):
                Speech rate multiplier. 1.0 is normal speed.
                Defaults to 1.0.
            voice_style (Union[None, str], optional):
                Voice style or emotion parameter. Defaults to None.
            language (str, optional):
                Language code for the text. Defaults to "zh".
            start_time (float, optional):
                Start time offset for generation. Defaults to 0.0.

        Returns:
            Dict[str, Any]:
                Dictionary containing TTS results with keys:
                - `audio` (BytesIO): Audio data in WAV format
                - `speech_text` (str): Processed text used for synthesis
                - `speech_time` (List[Tuple[int, float]]): Character timing
                - `duration` (float): Audio duration in seconds
        """
        raise NotImplementedError

    @abstractmethod
    async def get_voice_names(self, **kwargs: Any) -> Dict[str, Any]:
        """Get available voice names and their descriptions.

        Returns:
            Dict[str, Any]:
                Dictionary mapping voice identifiers to their
                human-readable names or descriptions.
        """
        raise NotImplementedError

    async def _handle_start(self, chunk: TextChunkStart, cur_time: float) -> None:
        """Handle the start chunk for a new TTS request.

        Args:
            chunk (TextChunkStart):
                The start chunk containing DAG configuration and request metadata.
            cur_time (float):
                Current timestamp for request tracking.
        """
        dag = chunk.dag
        conf = dag.conf
        request_id = chunk.request_id
        dag_start_time = conf.get("start_time", None)
        voice_name = conf.get("voice_name", None)
        if voice_name is None:
            voice_options = await self.get_voice_names()
            voice_name = list(voice_options.keys())[0]
            self.logger.warning(f"voice_name is not set in dag conf for {request_id}, use {voice_name} as default")
        voice_speed = conf.get("voice_speed", 1.0)
        language = conf.get("language", "zh")
        api_keys = conf.get("user_settings", {})
        chunk_n_char_lowerbound = conf.get("chunk_n_char_lowerbound", 10)
        chunk_n_char_lowerbound_en = conf.get("chunk_n_char_lowerbound_en", 25)
        request_id = chunk.request_id
        # Initialize buffer state using sentence splitter
        buffer_state = self.sentence_splitter.create_buffer_state()
        self.input_buffer[request_id] = {
            "dag_start_time": dag_start_time,
            "start_time": cur_time,
            "last_update_time": cur_time,
            "dag": dag,
            "voice_name": voice_name,
            "voice_speed": voice_speed,
            "voice_style": None,
            "language": language,
            "api_keys": api_keys,
            "node_name": chunk.node_name,
            "chunk_sent": 0,
            "downstream_warned": False,
            "chunk_n_char_lowerbound": chunk_n_char_lowerbound,
            "chunk_n_char_lowerbound_en": chunk_n_char_lowerbound_en,
            **buffer_state,  # Merge sentence splitter buffer state
        }
        asyncio.create_task(self._send_stream_start_task(request_id))

    async def _handle_body(self, chunk: TextChunkBody, cur_time: float) -> None:
        """Handle text body chunks and trigger TTS generation when appropriate.

        Args:
            chunk (TextChunkBody):
                The body chunk containing text segment and style information.
            cur_time (float):
                Current timestamp for request tracking.
        """
        request_id = chunk.request_id
        text_segment = chunk.text_segment
        style = chunk.style
        self.input_buffer[request_id]["voice_style"] = style
        if request_id not in self.input_buffer:
            msg = f"Request {request_id} not found in input buffer, but received a body message."
            self.logger.error(msg)
            raise ChunkWithoutStartError(msg)
        self.input_buffer[request_id]["last_update_time"] = cur_time
        self.input_buffer[request_id]["first_chunk_handle_time"] = cur_time

        # Define generation callback for sentence splitter
        def generation_callback(segment_text: str, seq_number: int) -> None:
            """Callback function for when a text segment is ready for TTS
            generation."""
            dag = self.input_buffer[request_id]["dag"]
            if dag.status == DAGStatus.RUNNING:
                speech_text_chunk = self.sentence_splitter.filter_text(segment_text)
                asyncio.create_task(self._stream_generate_task(request_id, speech_text_chunk, seq_number))

        # Use sentence splitter to process the text segment
        await self.sentence_splitter.process_text_segment(
            text_segment=text_segment,
            buffer_state=self.input_buffer[request_id],
            chunk_n_char_lowerbound=self.input_buffer[request_id]["chunk_n_char_lowerbound"],
            chunk_n_char_lowerbound_en=self.input_buffer[request_id]["chunk_n_char_lowerbound_en"],
            generation_callback=generation_callback,
        )

    async def _handle_end(self, chunk: TextChunkEnd, cur_time: float) -> None:
        """Handle the end chunk and finalize TTS processing.

        Args:
            chunk (TextChunkEnd):
                The end chunk signaling completion of text input.
            cur_time (float):
                Current timestamp for request tracking.
        """
        request_id = chunk.request_id
        if request_id not in self.input_buffer:
            msg = f"Request {request_id} not found in input buffer, but received an end message."
            self.logger.error(msg)
            raise ChunkWithoutStartError(msg)
        self.input_buffer[request_id]["last_update_time"] = cur_time
        chunk_for_generation = self.input_buffer[request_id]["text_segments"]
        split_marks = self.sentence_splitter.SPLIT_MARKS
        if not all(c in split_marks for c in chunk_for_generation) and len(chunk_for_generation) > 0:
            seq_number = self.input_buffer[request_id]["chunk_received"]
            self.input_buffer[request_id]["chunk_received"] += 1
            speech_text_chunk = self.sentence_splitter.filter_text(chunk_for_generation)
            asyncio.create_task(self._stream_generate_task(request_id, speech_text_chunk, seq_number))
        asyncio.create_task(self._send_stream_end_task(request_id))

    async def _send_stream_start_task(
        self,
        request_id: str,
    ) -> None:
        """Send stream start task.

        Args:
            request_id (str):
                The request id.
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
                audio_type="wav",
                node_name=next_node_name,
                dag=dag,
            )
            coroutines.append(payload.feed_stream(start_trunk))
        asyncio.gather(*coroutines)

    async def _stream_generate_task(
        self,
        request_id: str,
        text_segment: str,
        seq_number: int,
    ) -> None:
        """Stream generate task."""
        # check if the text contains meaningful content
        if not self.contains_meaningful_text(text_segment):
            msg = f"Skip processing text segment without meaningful content: '{text_segment}'"
            msg = msg + f" for request {request_id}"
            msg += f", seq_number: {seq_number}"
            self.logger.debug(msg)
            # directly increase the count of sent chunks and return
            self.input_buffer[request_id]["chunk_sent"] += 1
            return

        try:
            generation_start_time = time.time()
            ret_dict = await self._generate_tts(
                request_id=request_id,
                text=text_segment,
                voice_name=self.input_buffer[request_id]["voice_name"],
                voice_speed=self.input_buffer[request_id]["voice_speed"],
                voice_style=self.input_buffer[request_id]["voice_style"],
                language=self.input_buffer[request_id]["language"],
            )
            duration = ret_dict["duration"]
            generation_end_time = time.time()
            msg = f"Streaming TTS generation spent {generation_end_time - generation_start_time:.3f} seconds"
            msg = msg + f" for request {request_id}"
            msg += f", seq_number: {seq_number}"
            self.logger.debug(msg)
            dag = self.input_buffer[request_id]["dag"]
            while self.input_buffer[request_id]["chunk_sent"] < seq_number:
                if dag.status == DAGStatus.RUNNING:
                    await asyncio.sleep(self.sleep_time)
                else:
                    msg = f"Streaming TTS generation interrupted by DAG status {dag.status}"
                    msg = msg + f" for request {request_id}"
                    msg += f", seq_number: {seq_number}"
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
                audio_io = io.BytesIO()
                audio_io.write(ret_dict["audio"].getvalue())
                audio_io.seek(0)
                body_trunk = AudioWithSubtitleChunkBody(
                    request_id=request_id,
                    duration=duration,
                    audio_io=audio_io,
                    seq_number=seq_number,
                    speech_text=text_segment,
                    speech_time=ret_dict["speech_time"],
                )
                coroutines.append(payload.feed_stream(body_trunk))
            asyncio.gather(*coroutines)
            self.input_buffer[request_id]["chunk_sent"] += 1
            first_chunk_handle_time = self.input_buffer[request_id].get("first_chunk_handle_time", None)
            dag_start_time = dag.conf.get("start_time", None)
            if seq_number == 0 and first_chunk_handle_time is not None:
                cur_time = time.time()
                latency = cur_time - first_chunk_handle_time
                msg = (
                    f"Request {request_id} first AudioWithSubtitleChunkBody delay {latency:.2f}s "
                    + "from receiving first TextChunkBody."
                )
                if dag_start_time is not None:
                    latency = cur_time - dag_start_time
                    msg = msg[:-1] + f", delay {latency:.2f}s from dag start."
                self.logger.debug(msg)
        except Exception as e:
            msg = f"Error in streaming TTS: {e}"
            msg = msg + f" for request {request_id}"
            traceback_str = traceback.format_exc()
            msg += f"\n{traceback_str}"
            self.logger.error(msg)
            dag = self.input_buffer[request_id]["dag"]
            dag.set_status(DAGStatus.FAILED)
            return

    async def _send_stream_end_task(self, request_id: str) -> None:
        """Stream end task."""
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
                msg = f"TTS sending stream end task interrupted by DAG status {dag.status}"
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
        first_chunk_handle_time = self.input_buffer[request_id].get("first_chunk_handle_time", None)
        dag_start_time = dag.conf.get("start_time", None)
        if first_chunk_handle_time is not None:
            cur_time = time.time()
            latency = cur_time - first_chunk_handle_time
            msg = (
                f"Request {request_id} AudioWithSubtitleChunkEnd delay {latency:.2f}s "
                + "from receiving first TextChunkBody."
            )
            if dag_start_time is not None:
                latency = cur_time - dag_start_time
                msg = msg[:-1] + f", delay {latency:.2f}s from dag start."
            self.logger.debug(msg)
        self.input_buffer.pop(request_id)

    def contains_meaningful_text(self, text: str) -> bool:
        """Check if the text contains meaningful characters (Chinese or
        English).

        Args:
            text (str):
                The text to check for meaningful content.

        Returns:
            bool:
                True if the text contains Chinese characters or English letters,
                False if the text only contains punctuation, spaces, or other
                non-meaningful characters.
        """
        for char in text:
            # check Chinese characters
            if "\u4e00" <= char <= "\u9fff":
                return True
            # check English letters
            if char.isalpha():
                return True
        return False
