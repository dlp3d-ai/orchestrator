import numpy as np

from ..data_structures.face_chunk import FaceChunkBody, FaceChunkEnd, FaceChunkStart
from ..data_structures.process_flow import DAGStatus
from ..utils.streamable import ChunkWithoutStartError
from .stream_profile import StreamProfile


class FaceStreamProfile(StreamProfile):
    """Face stream profile."""

    def __init__(self, *args, **kwargs):
        """Initialize the face stream profile.

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

    async def _handle_start(self, chunk: FaceChunkStart, cur_time: float) -> None:
        """Handle the start chunk.

        Args:
            chunk (FaceChunkStart):
                The face start chunk containing blendshape information.
            cur_time (float):
                Current timestamp.
        """
        msg = f"Received start chunk for request {chunk.request_id}, dag progress: {chunk.node_name}, "
        msg += f"n_blendshape: {len(chunk.blendshape_names)}, dtype: {chunk.dtype}."
        self.logger.info(msg)
        if "float16" in chunk.dtype:
            dtype = np.float16
        elif "float32" in chunk.dtype:
            dtype = np.float32
        else:
            raise ValueError(f"Unsupported dtype: {chunk.dtype}")
        self.input_buffer[chunk.request_id] = dict(
            body_count=0,
            dag=chunk.dag,
            last_update_time=cur_time,
            blendshape_names=chunk.blendshape_names,
            n_blendshape=len(chunk.blendshape_names),
            dtype_str=chunk.dtype,
            dtype=dtype,
        )

    async def _handle_body(self, chunk: FaceChunkBody, cur_time: float) -> None:
        """Handle the body chunk.

        Args:
            chunk (FaceChunkBody):
                The face body chunk containing blendshape data.
            cur_time (float):
                Current timestamp.
        """
        if chunk.request_id not in self.input_buffer:
            msg = f"Request {chunk.request_id} not found in input buffer, but received a body message."
            self.logger.error(msg)
            raise ChunkWithoutStartError(msg)
        self.input_buffer[chunk.request_id]["last_update_time"] = cur_time
        dag = self.input_buffer[chunk.request_id]["dag"]
        if dag.status != DAGStatus.RUNNING:
            msg = f"DAG {dag.name} is not running."
            self.logger.error(msg)
            raise ChunkWithoutStartError(msg)
        dtype = self.input_buffer[chunk.request_id]["dtype"]
        n_blendshape = self.input_buffer[chunk.request_id]["n_blendshape"]
        blendshape_data = np.frombuffer(chunk.data, dtype=dtype)
        blendshape_data = blendshape_data.reshape(-1, n_blendshape)
        n_frames = blendshape_data.shape[0]
        msg = f"Received body chunk for request {chunk.request_id}, seq_number: {chunk.seq_number}, "
        msg += f"n_frames: {n_frames}, data size: {len(chunk.data)}."
        self.logger.info(msg)
        self.input_buffer[chunk.request_id]["body_count"] += 1

    async def _handle_end(self, chunk: FaceChunkEnd, cur_time: float) -> None:
        """Handle the end chunk.

        Args:
            chunk (FaceChunkEnd):
                The face end chunk signaling completion.
            cur_time (float):
                Current timestamp.
        """
        msg = f"Received end chunk for request {chunk.request_id}."
        self.logger.info(msg)
        if chunk.request_id not in self.input_buffer:
            msg = f"Request {chunk.request_id} not found in input buffer, but received an end message."
            self.logger.error(msg)
            raise ChunkWithoutStartError(msg)
        self.input_buffer[chunk.request_id]["last_update_time"] = cur_time
        await self._handle_status(chunk.request_id)
        self.input_buffer.pop(chunk.request_id)
