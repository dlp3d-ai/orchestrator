from typing import Union

from ..data_structures.face_chunk import FaceChunkBody, FaceChunkEnd, FaceChunkStart
from ..data_structures.motion_chunk import MotionChunkBody, MotionChunkEnd, MotionChunkStart
from ..data_structures.process_flow import DAGStatus
from ..utils.streamable import ChunkWithoutStartError
from .stream_profile import StreamProfile


class MotionFaceStreamProfile(StreamProfile):
    """Motion and face stream profile for handling combined motion and face
    chunks."""

    def __init__(self, *args, **kwargs):
        """Initialize the motion face stream profile.

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

    async def _handle_start(self, chunk: Union[MotionChunkStart, FaceChunkStart], cur_time: float) -> None:
        """Handle the start chunk.

        Args:
            chunk (Union[MotionChunkStart, FaceChunkStart]):
                The motion or face start chunk containing request information.
            cur_time (float):
                Current timestamp.
        """
        self.input_buffer[chunk.request_id] = dict(
            body_count=0, dag=chunk.dag, last_update_time=cur_time, end_received=set()
        )
        msg = f"Received {chunk.__class__.__name__} for request {chunk.request_id}, dag progress: {chunk.node_name},"
        if isinstance(chunk, MotionChunkStart):
            msg += f" n_joints: {len(chunk.joint_names)}, restpose_name: {chunk.restpose_name},"
            msg += f" blendshape_names: {chunk.blendshape_names},"
            msg += f" dtype: {chunk.dtype}, timeline_start_idx: {chunk.timeline_start_idx}"
        elif isinstance(chunk, FaceChunkStart):
            msg += f" n_blendshape: {len(chunk.blendshape_names)}, dtype: {chunk.dtype}, timeline_start_idx: {chunk.timeline_start_idx}"
        else:
            raise ValueError(f"Unknown start chunk type: {chunk.__class__.__name__}")
        self.logger.info(msg)

    async def _handle_body(self, chunk: Union[MotionChunkBody, FaceChunkBody], cur_time: float) -> None:
        """Handle the body chunk.

        Args:
            chunk (Union[MotionChunkBody, FaceChunkBody]):
                The motion or face body chunk containing data.
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
        msg = f"Received {chunk.__class__.__name__} for request {chunk.request_id}, seq_number: {chunk.seq_number}, "
        msg += f"data size: {len(chunk.data)}."
        self.logger.info(msg)
        self.input_buffer[chunk.request_id]["body_count"] += 1

    async def _handle_end(self, chunk: Union[MotionChunkEnd, FaceChunkEnd], cur_time: float) -> None:
        """Handle the end chunk.

        Args:
            chunk (Union[MotionChunkEnd, FaceChunkEnd]):
                The motion or face end chunk signaling completion.
            cur_time (float):
                Current timestamp.
        """
        msg = f"Received {chunk.__class__.__name__} for request {chunk.request_id}."
        self.logger.info(msg)
        if chunk.request_id not in self.input_buffer:
            msg = f"Request {chunk.request_id} not found in input buffer, but received an end message."
            self.logger.error(msg)
            raise ChunkWithoutStartError(msg)
        self.input_buffer[chunk.request_id]["end_received"].add(chunk.__class__.__name__)
        if len(self.input_buffer[chunk.request_id]["end_received"]) == 2:
            await self._handle_status(chunk.request_id)
            self.input_buffer.pop(chunk.request_id)
