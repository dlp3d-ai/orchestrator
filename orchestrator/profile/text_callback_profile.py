import json
from typing import Callable

from ..data_structures.process_flow import DAGStatus
from ..data_structures.text_chunk import TextChunkBody, TextChunkEnd, TextChunkStart
from ..utils.streamable import ChunkWithoutStartError
from .text_stream_profile import TextStreamProfile


class TextCallbackProfile(TextStreamProfile):
    """A class for text profile with callback function."""

    def __init__(self, *args, **kwargs):
        """Initialize the text callback profile.

        Args:
            mark_status_on_end (bool, optional):
                Whether to mark the status of the DAG on
                receiving the end chunk.
                Defaults to False.
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
            logger_cfg (Union[None, Dict[str, Any]], optional):
                Logger configuration. Defaults to None.
        """
        super().__init__(*args, **kwargs)

    async def _handle_start(self, chunk: TextChunkStart, cur_time: float) -> None:
        """Handle the start chunk.

        Args:
            chunk (TextChunkStart):
                The text start chunk containing request information.
            cur_time (float):
                Current timestamp.
        """
        msg = f"Received start chunk for request {chunk.request_id}, dag progress: {chunk.node_name}."
        self.logger.debug(msg)
        self.input_buffer[chunk.request_id] = dict(
            dag=chunk.dag,
            callback_instances=chunk.dag.conf["callback_instances"],
            callback_str_fn=chunk.dag.conf["callback_str_fn"],
            last_update_time=cur_time,
        )

    async def _handle_body(self, chunk: TextChunkBody, cur_time: float) -> None:
        """Handle the body chunk.

        Args:
            chunk (TextChunkBody):
                The text body chunk containing text segments.
            cur_time (float):
                Current timestamp.
        """
        if chunk.request_id not in self.input_buffer:
            msg = f"Request {chunk.request_id} not found in input buffer, but received a body message."
            self.logger.error(msg)
            raise ChunkWithoutStartError(msg)
        dag = self.input_buffer[chunk.request_id]["dag"]
        if dag.status != DAGStatus.RUNNING:
            msg = f"DAG {dag.name} is not running."
            self.logger.error(msg)
            raise ChunkWithoutStartError(msg)
        self.input_buffer[chunk.request_id]["last_update_time"] = cur_time
        msg = f"Received body chunk for request {chunk.request_id}, text segment: {chunk.text_segment}."
        self.logger.debug(msg)
        callback_str_fn: Callable = self.input_buffer[chunk.request_id]["callback_str_fn"]
        callback_dict = dict(
            request_id=chunk.request_id,
            text_segment=chunk.text_segment,
        )
        await callback_str_fn(json.dumps(callback_dict, ensure_ascii=False))

    async def _handle_end(self, chunk: TextChunkEnd, cur_time: float) -> None:
        """Handle the end chunk.

        Args:
            chunk (TextChunkEnd):
                The text end chunk signaling completion.
            cur_time (float):
                Current timestamp.
        """
        msg = f"Received end chunk for request {chunk.request_id}."
        self.logger.debug(msg)
        if chunk.request_id not in self.input_buffer:
            msg = f"Request {chunk.request_id} not found in input buffer, but received an end message."
            self.logger.error(msg)
            raise ChunkWithoutStartError(msg)
        self.input_buffer[chunk.request_id]["last_update_time"] = cur_time
        callback_str_fn: Callable = self.input_buffer[chunk.request_id]["callback_str_fn"]
        callback_dict = dict(
            request_id=chunk.request_id,
            signal="eof",
        )
        await callback_str_fn(json.dumps(callback_dict, ensure_ascii=False))
        await self._handle_status(chunk.request_id)
        self.input_buffer.pop(chunk.request_id)
