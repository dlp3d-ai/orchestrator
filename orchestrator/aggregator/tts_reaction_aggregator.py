import asyncio
import io
import re
import time
import traceback
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import List, Union

from ..data_structures.audio_chunk import (
    AudioWithReactionChunkBody,
    AudioWithReactionChunkEnd,
    AudioWithReactionChunkStart,
    AudioWithSubtitleChunkBody,
    AudioWithSubtitleChunkEnd,
    AudioWithSubtitleChunkStart,
)
from ..data_structures.process_flow import DAGStatus
from ..data_structures.reaction import Reaction, ReactionChunkBody, ReactionChunkEnd, ReactionChunkStart
from ..utils.executor_registry import ExecutorRegistry
from ..utils.streamable import ChunkWithoutStartError, Streamable


class TTSReactionAggregator(Streamable):
    """Aggregator for combining TTS audio and reaction data into synchronized
    streams.

    This class receives streaming chunks from TTS and Reaction nodes,
    synchronizes them based on text matching, and aggregates them into
    AudioWithReactionChunk objects. It handles complex text alignment between
    TTS speech and reaction responses, ensuring proper timing and sequence for
    downstream processing.
    """

    ExecutorRegistry.register_class("TTSReactionAggregator")

    def __init__(
        self,
        *args,
        max_workers: int = 1,
        thread_pool_executor: ThreadPoolExecutor | None = None,
        **kwargs,
    ):
        """Initialize the TTS reaction aggregator.

        Args:
            *args:
                Variable length argument list passed to parent class.
            max_workers (int, optional):
                Maximum number of worker threads for text processing.
                Defaults to 1.
            thread_pool_executor (ThreadPoolExecutor | None, optional):
                External thread pool executor to use. If None, creates a new one.
                Defaults to None.
            **kwargs:
                Additional keyword arguments passed to parent class.
        """
        super().__init__(*args, **kwargs)
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

    async def _handle_start(
        self,
        chunk: Union[AudioWithSubtitleChunkStart, ReactionChunkStart],
        cur_time: float,
    ) -> None:
        """Handle start chunks from TTS or reaction nodes.

        Initializes the input buffer for a new request and sets up tracking
        for both TTS and reaction streams. Sends stream start to downstream
        nodes when audio type is available.

        Args:
            chunk (Union[AudioWithSubtitleChunkStart, ReactionChunkStart]):
                Start chunk from either TTS or reaction node.
            cur_time (float):
                Current timestamp for tracking.
        """
        request_id = chunk.request_id
        if request_id not in self.input_buffer:
            self.input_buffer[request_id] = dict(
                start_chunk_classes=set(),
                end_chunk_classes=set(),
                last_update_time=cur_time,
                dag=chunk.dag,
                node_name=chunk.node_name,
                chunk_aggregated=0,
                chunk_sent=0,
                is_tts_end=False,
                is_reaction_end=False,
                downstream_warned=False,
                tts_chunks=deque(),
                reaction_chunks=deque(),
                lock=asyncio.Lock(),
                audio_type=None,  # Initialize as None
                stream_start_initiated=False,  # Flag to prevent duplicate sending
                active_tasks=set(),  # Track active tasks to prevent race conditions
            )
        chunk_class_str = chunk.__class__.__name__
        self.input_buffer[request_id]["start_chunk_classes"].add(chunk_class_str)
        if isinstance(chunk, AudioWithSubtitleChunkStart):
            self.input_buffer[request_id]["audio_type"] = chunk.audio_type

        # Send stream start only when audio_type is available and not yet sent
        if (
            self.input_buffer[request_id]["audio_type"] is not None
            and not self.input_buffer[request_id]["stream_start_initiated"]
        ):
            self.input_buffer[request_id]["stream_start_initiated"] = True
            asyncio.create_task(self._send_stream_start_task(request_id))

    async def _handle_body(
        self,
        chunk: Union[AudioWithReactionChunkBody, ReactionChunkBody],
        cur_time: float,
    ) -> None:
        """Handle body chunks from TTS or reaction nodes.

        Validates the chunk against existing start chunks and adds it to the
        appropriate buffer. Triggers aggregation task if valid.

        Args:
            chunk (Union[AudioWithReactionChunkBody, ReactionChunkBody]):
                Body chunk from either TTS or reaction node.
            cur_time (float):
                Current timestamp for tracking.
        """
        request_id = chunk.request_id
        chunk_class_str = chunk.__class__.__name__
        if request_id not in self.input_buffer:
            msg = (
                f"Request {request_id} not found in input buffer, "
                + f"but received a body message of class {chunk_class_str}."
            )
            self.logger.error(msg)
            raise ChunkWithoutStartError(msg)
        start_chunk_class_str = chunk_class_str.replace("ChunkBody", "ChunkStart")
        if start_chunk_class_str not in self.input_buffer[request_id]["start_chunk_classes"]:
            msg = (
                f"Start chunk {start_chunk_class_str} for request {request_id} not found in input buffer, "
                + "but received a body message."
            )
            self.logger.error(msg)
            raise ChunkWithoutStartError(msg)
        dag = self.input_buffer[request_id]["dag"]
        self.input_buffer[request_id]["last_update_time"] = cur_time
        if isinstance(chunk, AudioWithSubtitleChunkBody):
            self.input_buffer[request_id]["tts_chunks"].append(chunk)
        elif isinstance(chunk, ReactionChunkBody):
            self.input_buffer[request_id]["reaction_chunks"].append(chunk)
        else:
            dag.set_status(DAGStatus.FAILED)
            msg = f"Received unknown body message of class {chunk_class_str}, request id: {request_id}"
            self.logger.error(msg)
            raise ValueError(msg)
        asyncio.create_task(self._stream_aggregate_task(request_id))

    async def _handle_end(
        self,
        chunk: Union[AudioWithSubtitleChunkEnd, ReactionChunkEnd],
        cur_time: float,
    ) -> None:
        """Handle end chunks from TTS or reaction nodes.

        Marks the corresponding stream as ended and triggers final aggregation
        when both TTS and reaction streams are complete.

        Args:
            chunk (Union[AudioWithSubtitleChunkEnd, ReactionChunkEnd]):
                End chunk from either TTS or reaction node.
            cur_time (float):
                Current timestamp for tracking.
        """
        request_id = chunk.request_id
        if request_id not in self.input_buffer:
            msg = f"Request {request_id} not found in input buffer, but received an end message."
            self.logger.error(msg)
            raise ChunkWithoutStartError(msg)
        self.input_buffer[request_id]["last_update_time"] = cur_time
        if isinstance(chunk, AudioWithSubtitleChunkEnd):
            self.input_buffer[request_id]["is_tts_end"] = True
        elif isinstance(chunk, ReactionChunkEnd):
            self.input_buffer[request_id]["is_reaction_end"] = True
        else:
            dag = self.input_buffer[request_id]["dag"]
            dag.set_status(DAGStatus.FAILED)
            msg = f"Received unknown end message of class {chunk.__class__.__name__}, request id: {request_id}"
            self.logger.error(msg)
            raise ValueError(msg)
        asyncio.create_task(self._stream_aggregate_task(request_id))
        if self.input_buffer[request_id]["is_tts_end"] and self.input_buffer[request_id]["is_reaction_end"]:
            asyncio.create_task(self._send_stream_end_task(request_id))

    async def _send_stream_start_task(self, request_id: str) -> None:
        """Send stream start to downstream nodes.

        Creates and sends AudioWithReactionChunkStart to all downstream nodes
        for the specified request.

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
            return
        coroutines = list()
        for node in downstream_nodes:
            next_node_name = node.name
            payload = node.payload
            start_trunk = AudioWithReactionChunkStart(
                request_id=request_id,
                node_name=next_node_name,
                dag=dag,
                audio_type=self.input_buffer[request_id]["audio_type"],
            )
            coroutines.append(payload.feed_stream(start_trunk))
        asyncio.gather(*coroutines)

    async def _send_stream_body_task(
        self,
        request_id: str,
        seq_number: int,
        tts_aggregated: List[AudioWithSubtitleChunkBody],
        reaction_aggregated: List[ReactionChunkBody],
    ) -> None:
        """Send aggregated stream body to downstream nodes.

        Combines TTS audio and reaction data, performs text alignment and timing
        synchronization, then sends the result to downstream nodes.

        Args:
            request_id (str):
                Unique identifier for the request.
            seq_number (int):
                Sequence number for ordering chunks.
            tts_aggregated (List[AudioWithSubtitleChunkBody]):
                List of TTS audio chunks to aggregate.
            reaction_aggregated (List[ReactionChunkBody]):
                List of reaction chunks to aggregate.
        """
        # Register this task to prevent race conditions
        if request_id not in self.input_buffer:
            self.logger.warning(f"Request {request_id} not found in input buffer, skipping send stream body task.")
            return

        task_id = f"body_task_{seq_number}"
        self.input_buffer[request_id]["active_tasks"].add(task_id)

        try:
            generation_start_time = time.time()

            # Get audio information
            duration = sum([chunk.duration for chunk in tts_aggregated])
            audio_io = io.BytesIO()
            for chunk in tts_aggregated:
                audio_io.write(chunk.audio_io.read())
            audio_io.seek(0)

            # Build complete TTS text and speech_time mapping
            tts_full_text = ""
            tts_speech_time_map = {}  # {tts_char_index: (time, duration)}
            start_time = 0.0
            start_index = 0
            for speech_chunk in tts_aggregated:
                for index, speech_time_chunk in speech_chunk.speech_time:
                    tts_speech_time_map[start_index + index] = (
                        speech_time_chunk + start_time,
                        0,
                    )  # duration temporarily set to 0
                tts_full_text += speech_chunk.speech_text
                start_time += speech_chunk.duration
                start_index += len(speech_chunk.speech_text)

            # Get reaction information
            speech_text = "".join([chunk.reaction.speech_text for chunk in reaction_aggregated])

            # Align TTS and reaction text, calculate speech_time based on reaction
            speech_time = []
            reaction_index = 0
            tts_index = 0

            while reaction_index < len(speech_text) and tts_index < len(tts_full_text):
                reaction_char = speech_text[reaction_index]
                tts_char = tts_full_text[tts_index]

                if reaction_char == tts_char:
                    # Character match, record speech_time (based on reaction index)
                    if tts_index in tts_speech_time_map:
                        speech_time.append((reaction_index, tts_speech_time_map[tts_index][0]))
                    reaction_index += 1
                    tts_index += 1
                elif reaction_char in tts_full_text[tts_index:]:
                    # Reaction character exists later in TTS, skip TTS characters
                    tts_index += 1
                else:
                    # Reaction character does not exist in TTS, skip reaction character
                    reaction_index += 1

            # Process motion_keywords
            if all([chunk.reaction.motion_keywords is None for chunk in reaction_aggregated]):
                motion_keywords = None
            else:
                motion_keywords = []
                start_index = 0
                for reaction_chunk in reaction_aggregated:
                    motion_keywords_chunk = reaction_chunk.reaction.motion_keywords
                    if motion_keywords_chunk:
                        for index, motion_keyword in motion_keywords_chunk:
                            motion_keywords.append((start_index + index, motion_keyword))
                    start_index += len(reaction_chunk.reaction.speech_text)
            reaction = Reaction(
                speech_text=speech_text,
                label_expression=reaction_aggregated[-1].reaction.label_expression,
                motion_keywords=motion_keywords,
                face_emotion=reaction_aggregated[-1].reaction.face_emotion,
            )

            generation_end_time = time.time()
            msg = f"Streaming TTS reaction aggregator spent {generation_end_time - generation_start_time:.3f} seconds"
            msg = msg + f" for request {request_id}, seq_number: {seq_number}"
            self.logger.debug(msg)
            dag = self.input_buffer[request_id]["dag"]
            while self.input_buffer[request_id]["chunk_sent"] < seq_number:
                if dag.status == DAGStatus.RUNNING:
                    await asyncio.sleep(self.sleep_time)
                else:
                    msg = f"Streaming TTS reaction aggregator interrupted by DAG status {dag.status}"
                    msg = msg + f" for request {request_id}, seq_number: {seq_number}"
                    self.logger.warning(msg)
                    return
            # Prepare downstream
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
                body_trunk = AudioWithReactionChunkBody(
                    request_id=request_id,
                    duration=duration,
                    audio_io=audio_io,
                    seq_number=seq_number,
                    speech_text=speech_text,
                    speech_time=speech_time,
                    reaction=reaction,
                )
                coroutines.append(payload.feed_stream(body_trunk))
            asyncio.gather(*coroutines)
            self.input_buffer[request_id]["chunk_sent"] += 1
        except Exception as e:
            msg = f"Error in streaming TTS reaction aggregator: {e} for request {request_id}"
            traceback_str = traceback.format_exc()
            msg += f"\n{traceback_str}"
            self.logger.error(msg)
            if request_id in self.input_buffer:
                dag = self.input_buffer[request_id]["dag"]
                dag.set_status(DAGStatus.FAILED)
        finally:
            # Always clean up the task registration
            if request_id in self.input_buffer:
                self.input_buffer[request_id]["active_tasks"].discard(task_id)

    async def _send_stream_end_task(self, request_id: str) -> None:
        """Send stream end to downstream nodes.

        Waits for all pending chunks to be sent, then sends end signal to
        downstream nodes and cleans up the request buffer.

        Args:
            request_id (str):
                Unique identifier for the request.
        """
        try:
            dag = self.input_buffer[request_id]["dag"]
        except KeyError:
            msg = f"Request {request_id} not found in input buffer, skip sending stream end task."
            self.logger.warning(msg)
            return
        # Wait for all chunks to be aggregated and sent
        while self.input_buffer[request_id]["chunk_aggregated"] > self.input_buffer[request_id]["chunk_sent"]:
            if dag.status == DAGStatus.RUNNING:
                await asyncio.sleep(self.sleep_time)
            else:
                msg = f"TTS reaction aggregator sending stream end task interrupted by DAG status {dag.status}"
                msg = msg + f" for request {request_id}"
                self.logger.warning(msg)
                return
        # Wait for all active tasks to complete to prevent race conditions
        while self.input_buffer[request_id]["active_tasks"]:
            if dag.status == DAGStatus.RUNNING:
                await asyncio.sleep(self.sleep_time)
            else:
                msg = f"TTS reaction aggregator waiting for active tasks interrupted by DAG status {dag.status}"
                msg = msg + f" for request {request_id}"
                self.logger.warning(msg)
                return
        # Prepare downstream
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
            end_trunk = AudioWithReactionChunkEnd(
                request_id=request_id,
            )
            coroutines.append(payload.feed_stream(end_trunk))
        asyncio.gather(*coroutines)
        self.input_buffer.pop(request_id)

    async def _stream_aggregate_task(self, request_id: str) -> None:
        """Main aggregation task for synchronizing TTS and reaction streams.

        Performs intelligent text matching between TTS and reaction chunks,
        aggregates matching segments, and sends them to downstream nodes.
        Uses normalized text comparison for robust matching.

        Args:
            request_id (str):
                Unique identifier for the request.
        """
        buffer = self.input_buffer[request_id]
        async with buffer["lock"]:
            reaction_chunks = buffer["reaction_chunks"]
            tts_chunks = buffer["tts_chunks"]

            # No work to do if either is empty
            if not reaction_chunks or not tts_chunks:
                return

            # if tts_chunk and reaction_chunk are both end
            if buffer["is_tts_end"] and buffer["is_reaction_end"]:
                seq_number = buffer["chunk_aggregated"]
                buffer["chunk_aggregated"] += 1
                tts_aggregated = list(tts_chunks)
                reaction_aggregated = list(reaction_chunks)
                asyncio.create_task(
                    self._send_stream_body_task(request_id, seq_number, tts_aggregated, reaction_aggregated)
                )
                tts_chunks.clear()
                reaction_chunks.clear()
                return

            # Precompute normalized text for all chunks to avoid repeated extraction
            tts_normalized = []
            tts_texts = []
            cumulative_tts_normalized = ""

            loop = asyncio.get_running_loop()

            for chunk in tts_chunks:
                text = chunk.speech_text
                tts_texts.append(text)
                normalized = await loop.run_in_executor(self.executor, self.extract_meaningful_text, text)
                tts_normalized.append(normalized)
                cumulative_tts_normalized += normalized

            # Process reaction chunks and match with TTS
            while reaction_chunks:
                # Build reaction text incrementally
                reaction_text = ""
                normalized_reaction_text = ""
                matched = False

                # Try to match with increasing number of reaction chunks
                for j in range(len(reaction_chunks)):
                    reaction_text += reaction_chunks[j].reaction.speech_text
                    normalized_reaction = await loop.run_in_executor(
                        self.executor, self.extract_meaningful_text, reaction_chunks[j].reaction.speech_text
                    )
                    normalized_reaction_text += normalized_reaction

                    # Skip empty reaction text
                    if not normalized_reaction_text:
                        continue

                    # Check if this reaction text can be matched in TTS
                    if normalized_reaction_text in cumulative_tts_normalized:
                        # Find the ending position in TTS
                        tts_idx = -1  # Initialize to invalid value
                        match_found = False

                        # Find which TTS chunks contain the matched text
                        current_tts_text = ""
                        current_tts_normalized = ""

                        for i in range(len(tts_chunks)):
                            current_tts_normalized += tts_normalized[i]
                            current_tts_text += tts_texts[i]

                            # If we've accumulated enough characters to match reaction text
                            if current_tts_normalized.endswith(normalized_reaction_text):
                                tts_idx = i
                                match_found = True
                                break

                        # Process the match if found
                        if match_found:
                            self.logger.debug(f"Match found in TTS and reaction: {current_tts_text}")
                            seq_number = buffer["chunk_aggregated"]
                            buffer["chunk_aggregated"] += 1

                            # Create lists of chunks to send
                            tts_aggregated = [tts_chunks[i] for i in range(tts_idx + 1)]
                            reaction_aggregated = [reaction_chunks[i] for i in range(j + 1)]

                            # Send the task
                            asyncio.create_task(
                                self._send_stream_body_task(request_id, seq_number, tts_aggregated, reaction_aggregated)
                            )

                            # Remove processed chunks
                            for _ in range(j + 1):
                                reaction_chunks.popleft()
                            for _ in range(tts_idx + 1):
                                tts_chunks.popleft()

                            # Update our precomputed data for next iteration
                            tts_normalized = tts_normalized[tts_idx + 1 :]
                            tts_texts = tts_texts[tts_idx + 1 :]
                            cumulative_tts_normalized = "".join(tts_normalized)

                            matched = True
                            break

                # If no match was found with any combination, we're done
                if not matched:
                    break

    def extract_meaningful_text(self, s: str) -> str:
        """Extract meaningful characters from text for matching purposes.

        Removes HTML tags, parentheses, brackets and other formatting,
        keeping only Chinese characters and English letters for text comparison.

        Args:
            s (str):
                Input text to process.

        Returns:
            str:
                Normalized text containing only meaningful characters in lowercase.
        """
        # Remove all <tag>...</tag> tags
        s = re.sub(r"<[^<>]+?>.*?</[^<>]+?>", "", s, flags=re.DOTALL)

        # Remove parentheses content: Chinese （）
        s = re.sub(r"（.*?）", "", s)

        # Remove English ()
        s = re.sub(r"\(.*?\)", "", s)

        # Remove English []
        s = re.sub(r"\[.*?\]", "", s)

        # Extract Chinese characters and English letters, convert to lowercase
        text = "".join(re.findall(r"[\u4e00-\u9fffa-zA-Z]", s)).lower()
        return text
