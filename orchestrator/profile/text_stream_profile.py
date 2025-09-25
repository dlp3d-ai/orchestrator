from ..data_structures.process_flow import DAGStatus
from ..data_structures.text_chunk import TextChunkBody, TextChunkEnd, TextChunkStart
from ..utils.streamable import ChunkWithoutStartError
from .stream_profile import StreamProfile


class TextStreamProfile(StreamProfile):
    """Text stream profile."""

    def __init__(self, *args, **kwargs):
        """Initialize the text stream profile.

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
        self.logger.info(msg)
        self.input_buffer[chunk.request_id] = dict(
            dag=chunk.dag,
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
        msg = f"Received body chunk for request {chunk.request_id}, text segment: {chunk.text_segment}."
        self.logger.info(msg)
        if chunk.request_id not in self.input_buffer:
            msg = f"Request {chunk.request_id} not found in input buffer, but received a body message."
            self.logger.error(msg)
            raise ChunkWithoutStartError(msg)
        dag = self.input_buffer[chunk.request_id]["dag"]
        self.input_buffer[chunk.request_id]["last_update_time"] = cur_time
        if dag.status != DAGStatus.RUNNING:
            msg = f"DAG {dag.name} is not running."
            self.logger.error(msg)
            raise ChunkWithoutStartError(msg)

    async def _handle_end(self, chunk: TextChunkEnd, cur_time: float) -> None:
        """Handle the end chunk.

        Args:
            chunk (TextChunkEnd):
                The text end chunk signaling completion.
            cur_time (float):
                Current timestamp.
        """
        msg = f"Received end chunk for request {chunk.request_id}."
        self.logger.info(msg)
        if chunk.request_id not in self.input_buffer:
            msg = f"Request {chunk.request_id} not found in input buffer, but received an end message."
            self.logger.error(msg)
            raise ChunkWithoutStartError(msg)
        await self._handle_status(chunk.request_id)
        self.input_buffer[chunk.request_id]["last_update_time"] = cur_time
        self.input_buffer.pop(chunk.request_id)
