import asyncio
import json
import traceback
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import Any, Dict, List, Literal, Union

import numpy as np

from ..data_structures.face_chunk import FaceChunkBody, FaceChunkEnd, FaceChunkStart
from ..data_structures.motion_chunk import MotionChunkBody, MotionChunkEnd, MotionChunkStart
from ..data_structures.process_flow import DAGStatus
from ..utils.executor_registry import ExecutorRegistry
from ..utils.streamable import ChunkWithoutStartError, Streamable


class BlendshapesAggregator(Streamable):
    """Blendshapes aggregator for extracting blendshapes from motion and face
    stream, aggregating them as one face stream and sending to downstreams.

    This class aggregates blendshape data from motion and face sources,
    handling synchronization and data combination according to specified
    configuration. It processes streaming data chunks and outputs aggregated
    blendshape information to downstream components.
    """

    ExecutorRegistry.register_class("BlendshapesAggregator")

    def __init__(
        self,
        motion_first_blendshape_names: Union[List[str], str, None] = None,
        add_blendshape_names: Union[List[str], str, None] = None,
        thread_pool_executor: Union[None, ThreadPoolExecutor] = None,
        max_workers: int = 4,
        queue_size: int = 100,
        sleep_time: float = 0.01,
        clean_interval: float = 10.0,
        expire_time: float = 120.0,
        logger_cfg: Union[None, Dict[str, Any]] = None,
    ):
        """Initialize the blendshapes aggregator.

        Args:
            motion_first_blendshape_names (Union[List[str], str, None], optional):
                List of blendshape names to prioritize from motion data,
                or path to JSON file containing the list. Defaults to None.
            add_blendshape_names (Union[List[str], str, None], optional):
                List of blendshape names to add from motion to face data,
                or path to JSON file containing the list. Defaults to None.
            thread_pool_executor (Union[None, ThreadPoolExecutor], optional):
                External thread pool executor to use. If None, creates
                a new one with max_workers. Defaults to None.
            max_workers (int, optional):
                Maximum number of worker threads for internal executor.
                Defaults to 4.
            queue_size (int, optional):
                Maximum size of the input queue. Defaults to 100.
            sleep_time (float, optional):
                Sleep time in seconds between processing iterations.
                Defaults to 0.01.
            clean_interval (float, optional):
                Interval in seconds for cleaning expired requests.
                Defaults to 10.0.
            expire_time (float, optional):
                Time in seconds after which requests expire.
                Defaults to 120.0.
            logger_cfg (Union[None, Dict[str, Any]], optional):
                Logger configuration dictionary. Defaults to None.
        """
        Streamable.__init__(
            self,
            queue_size=queue_size,
            sleep_time=sleep_time,
            clean_interval=clean_interval,
            expire_time=expire_time,
            logger_cfg=logger_cfg,
        )
        if thread_pool_executor is None:
            self.thread_pool_executor = ThreadPoolExecutor(max_workers=max_workers)
        else:
            self.thread_pool_executor = thread_pool_executor
        self.executor_external = True if thread_pool_executor is not None else False
        if isinstance(motion_first_blendshape_names, str):
            with open(motion_first_blendshape_names, "r") as f:
                self.motion_first_blendshape_names = json.load(f)
        else:
            self.motion_first_blendshape_names = motion_first_blendshape_names
        if isinstance(add_blendshape_names, str):
            with open(add_blendshape_names, "r") as f:
                self.add_blendshape_names = json.load(f)
        else:
            self.add_blendshape_names = add_blendshape_names

    def __del__(self):
        """Clean up resources on object destruction.

        Shuts down the thread pool executor if it was created internally.
        """
        if not self.executor_external:
            self.thread_pool_executor.shutdown(wait=False)

    async def _handle_start(
        self,
        chunk: Union[MotionChunkStart, FaceChunkStart],
        cur_time: float,
    ) -> None:
        """Handle start chunk processing.

        Args:
            chunk (Union[MotionChunkStart, FaceChunkStart]):
                Start chunk containing initialization data.
            cur_time (float):
                Current timestamp for request tracking.
        """
        request_id = chunk.request_id
        if request_id not in self.input_buffer:
            blendshape_aggregate_init = (
                None
                if self.add_blendshape_names is not None or self.motion_first_blendshape_names is not None
                else False
            )
            self.input_buffer[request_id] = dict(
                last_update_time=cur_time,
                dag=chunk.dag,
                node_name=chunk.node_name,
                body_received_counter=dict(),
                body_sent_counter=dict(),
                start_sent=set(),
                end_sent=set(),
                blendshape_aggregate=blendshape_aggregate_init,
                motion_n_joints=None,
                motion_dtype=None,
                motion_timeline_start_idx=None,
                motion_blendshape_names=None,
                motion_blendshape_name_mapping=None,
                motion_buffer=b"",
                face_blendshape_names=None,
                face_blendshape_name_mapping=None,
                face_dtype=None,
                face_buffer=b"",
                buffer_frame_idx=None,
                blendshape_lock=Lock(),
            )
        chunk_class_str = chunk.__class__.__name__
        chunk_type = chunk_class_str.split("Chunk")[0]
        self.input_buffer[request_id]["body_received_counter"][chunk_type] = 0
        self.input_buffer[request_id]["body_sent_counter"][chunk_type] = 0
        asyncio.create_task(self._send_chunk_start_task(request_id, chunk))

    async def _handle_body(
        self,
        chunk: Union[MotionChunkBody, FaceChunkBody],
        cur_time: float,
    ) -> None:
        """Handle body chunk processing.

        Args:
            chunk (Union[MotionChunkBody, FaceChunkBody]):
                Body chunk containing data to process.
            cur_time (float):
                Current timestamp for request tracking.
        """
        request_id = chunk.request_id
        chunk_class_str = chunk.__class__.__name__
        chunk_type = chunk_class_str.split("Chunk")[0]
        if request_id not in self.input_buffer:
            msg = (
                f"Request {request_id} not found in input buffer, "
                + f"but received a body message of class {chunk_class_str}."
            )
            self.logger.error(msg)
            raise ChunkWithoutStartError(msg)
        self.input_buffer[request_id]["last_update_time"] = cur_time
        dag = self.input_buffer[request_id]["dag"]
        seq_number = self.input_buffer[request_id]["body_received_counter"][chunk_type]
        self.input_buffer[request_id]["body_received_counter"][chunk_type] += 1
        if dag.status == DAGStatus.RUNNING:
            asyncio.create_task(
                self._send_chunk_body_task(
                    request_id=request_id,
                    chunk=chunk,
                    seq_number=seq_number,
                )
            )

    async def _handle_end(
        self,
        chunk: Union[MotionChunkEnd, FaceChunkEnd],
        cur_time: float,
    ) -> None:
        """Handle end chunk processing.

        Args:
            chunk (Union[MotionChunkEnd, FaceChunkEnd]):
                End chunk signaling completion of data stream.
            cur_time (float):
                Current timestamp for request tracking.
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
        self.input_buffer[request_id]["last_update_time"] = cur_time
        # send end signal to downstreams
        dag = self.input_buffer[request_id]["dag"]
        if dag.status == DAGStatus.RUNNING:
            asyncio.create_task(
                self._send_chunk_end_task(
                    request_id,
                    chunk,
                )
            )

    async def _send_chunk_start_task(self, request_id: str, chunk: Union[MotionChunkStart, FaceChunkStart]) -> None:
        """Send start chunk to downstream components.

        Args:
            request_id (str):
                Unique identifier for the request.
            chunk (Union[MotionChunkStart, FaceChunkStart]):
                Start chunk to send downstream.
        """
        class_name = chunk.__class__.__name__
        chunk_type = class_name.split("Chunk")[0]
        dag = self.input_buffer[request_id]["dag"]
        dag_node = dag.get_node(self.input_buffer[request_id]["node_name"])
        # classification is not a stream, skip sending start signal
        try:
            if isinstance(chunk, MotionChunkStart):
                self.input_buffer[request_id]["motion_n_joints"] = len(chunk.joint_names)
                self.input_buffer[request_id]["motion_dtype"] = chunk.dtype
                self.input_buffer[request_id]["motion_timeline_start_idx"] = chunk.timeline_start_idx
                bit_width = chunk.dtype.rsplit("float", 1)[1]
                try:
                    bit_width = int(bit_width)
                    width = bit_width // 8
                except ValueError:
                    msg = f"Unknown dtype {chunk.dtype} for motion chunk, request id: {request_id}"
                    self.logger.error(msg)
                    raise ValueError(msg)
                self.input_buffer[request_id]["motion_byte_rate"] = (len(chunk.joint_names) * 9 + 3 + 3) * width
                if chunk.blendshape_names is not None:
                    if self.input_buffer[request_id]["blendshape_aggregate"] is None:
                        self.logger.debug(f"blendshape_aggregate is set to True for request {request_id}")
                        self.input_buffer[request_id]["blendshape_aggregate"] = True
                    self.input_buffer[request_id]["motion_blendshape_names"] = chunk.blendshape_names
                    self.input_buffer[request_id]["motion_blendshape_name_mapping"] = {
                        name: i for i, name in enumerate(chunk.blendshape_names)
                    }
                    self.input_buffer[request_id]["motion_buffer"] = b""
                    motion_chunk_start = MotionChunkStart(
                        request_id=chunk.request_id,
                        node_name=chunk.node_name,
                        dag=dag,
                        joint_names=chunk.joint_names,
                        restpose_name=chunk.restpose_name,
                        dtype=chunk.dtype,
                        blendshape_names=None,
                        timeline_start_idx=chunk.timeline_start_idx,
                    )
                    if chunk.timeline_start_idx is not None:
                        self.input_buffer[request_id]["buffer_frame_idx"] = chunk.timeline_start_idx
                    else:
                        self.input_buffer[request_id]["buffer_frame_idx"] = 0
                else:
                    self.input_buffer[request_id]["blendshape_aggregate"] = False
                    self.logger.debug(f"blendshape_aggregate is set to False for request {request_id}")
                    motion_chunk_start = chunk
                for node in dag_node.downstreams:
                    payload = node.payload
                    await payload.feed_stream(motion_chunk_start)
                self.logger.debug(f"Sent motion chunk start for request {request_id}")
                self.input_buffer[request_id]["start_sent"].add(chunk_type)
            elif isinstance(chunk, FaceChunkStart):
                # chunk.blendshape_names never be None
                self.input_buffer[request_id]["face_blendshape_names"] = chunk.blendshape_names
                bit_width = chunk.dtype.rsplit("float", 1)[1]
                try:
                    bit_width = int(bit_width)
                    width = bit_width // 8
                except ValueError:
                    msg = f"Unknown dtype {chunk.dtype} for face chunk, request id: {request_id}"
                    self.logger.error(msg)
                    raise ValueError(msg)
                self.input_buffer[request_id]["face_byte_rate"] = len(chunk.blendshape_names) * width
                self.input_buffer[request_id]["face_blendshape_name_mapping"] = {
                    name: i for i, name in enumerate(chunk.blendshape_names)
                }
                self.input_buffer[request_id]["face_dtype"] = chunk.dtype
                self.input_buffer[request_id]["face_buffer"] = b""
                while self.input_buffer[request_id]["motion_timeline_start_idx"] is None:
                    if dag.status == DAGStatus.RUNNING:
                        await asyncio.sleep(self.sleep_time)
                    else:
                        msg = f"Streaming start sending interrupted by DAG status {dag.status}"
                        msg += f" for request {request_id}"
                        self.logger.warning(msg)
                        return
                if self.input_buffer[request_id]["blendshape_aggregate"]:
                    face_chunk_start = FaceChunkStart(
                        request_id=chunk.request_id,
                        blendshape_names=chunk.blendshape_names,
                        dtype=chunk.dtype,
                        node_name=chunk.node_name,
                        dag=dag,
                        timeline_start_idx=self.input_buffer[request_id]["motion_timeline_start_idx"],
                    )
                else:
                    face_chunk_start = chunk
                for node in dag_node.downstreams:
                    payload = node.payload
                    await payload.feed_stream(face_chunk_start)
                self.logger.debug(f"Sent face chunk start for request {request_id}")
                self.input_buffer[request_id]["start_sent"].add(chunk_type)
            else:
                msg = f"Received unknown start chunk of class {chunk.__class__.__name__}, request id: {request_id}"
                self.logger.error(msg)
                raise TypeError(msg)
        except Exception as e:
            self.logger.error(f"Error sending start chunk to downstreams: {e}")
            dag = self.input_buffer[request_id]["dag"]
            dag.set_status(DAGStatus.FAILED)
            raise e

    async def _send_chunk_body_task(
        self,
        request_id: str,
        chunk: Union[MotionChunkBody, FaceChunkBody],
        seq_number: int,
    ) -> None:
        """Send body chunk to downstream components.

        Args:
            request_id (str):
                Unique identifier for the request.
            chunk (Union[MotionChunkBody, FaceChunkBody]):
                Body chunk to process and send.
            seq_number (int):
                Sequence number of the chunk.
        """
        class_name = chunk.__class__.__name__
        chunk_type = class_name.split("Chunk")[0]
        dag = self.input_buffer[request_id]["dag"]
        dag_node = dag.get_node(self.input_buffer[request_id]["node_name"])
        loop = asyncio.get_running_loop()
        try:
            if not isinstance(chunk, MotionChunkBody) and not isinstance(chunk, FaceChunkBody):
                msg = f"Received unknown body chunk of class {chunk.__class__.__name__}, request id: {request_id}"
                self.logger.error(msg)
                raise TypeError(msg)
            # waiting for start chunk sent
            while (
                self.input_buffer[request_id]["body_sent_counter"][chunk_type] < seq_number
                or chunk_type not in self.input_buffer[request_id]["start_sent"]
            ):
                if dag.status == DAGStatus.RUNNING:
                    await asyncio.sleep(self.sleep_time)
                else:
                    msg = f"Streaming body sending interrupted by DAG status {dag.status}"
                    msg += f" for request {request_id}"
                    self.logger.warning(msg)
                    return
            # waiting for blendshapes confirmed
            while (
                self.input_buffer[request_id]["blendshape_aggregate"] is None
                or self.input_buffer[request_id]["blendshape_aggregate"]
                and (
                    self.input_buffer[request_id]["face_blendshape_names"] is None
                    or self.input_buffer[request_id]["motion_blendshape_names"] is None
                )
            ):
                if dag.status == DAGStatus.RUNNING:
                    await asyncio.sleep(self.sleep_time)
                else:
                    msg = f"Streaming motion/face body sending interrupted by DAG status {dag.status}"
                    msg += f" for request {request_id}"
                    self.logger.warning(msg)
                    return
            if self.input_buffer[request_id]["blendshape_aggregate"]:
                # append blendshape buffer
                if isinstance(chunk, FaceChunkBody):
                    await loop.run_in_executor(
                        self.thread_pool_executor,
                        self._append_blendshape_buffer,
                        request_id,
                        "face_buffer",
                        chunk.data,
                    )
                else:
                    await loop.run_in_executor(
                        self.thread_pool_executor,
                        self._append_blendshape_buffer,
                        request_id,
                        "motion_buffer",
                        chunk.data,
                    )
                    # motion sending is not influenced by aggregate
                    motion_data_wo_bs = await loop.run_in_executor(
                        self.thread_pool_executor, self._remove_blendshape_from_motion, request_id, chunk.data
                    )
                    motion_chunk_body = MotionChunkBody(
                        request_id=chunk.request_id,
                        seq_number=chunk.seq_number,
                        data=motion_data_wo_bs,
                    )
                    for node in dag_node.downstreams:
                        payload = node.payload
                        await payload.feed_stream(motion_chunk_body)
                    self.logger.debug(f"Sent motion chunk body {motion_chunk_body.seq_number} for request {request_id}")
                face_bytes = await loop.run_in_executor(
                    self.thread_pool_executor, self._aggregate_blendshape, request_id
                )
                if face_bytes is not None:
                    face_chunk_body = FaceChunkBody(
                        request_id=chunk.request_id,
                        seq_number=chunk.seq_number,
                        data=face_bytes,
                    )
                    for node in dag_node.downstreams:
                        payload = node.payload
                        await payload.feed_stream(face_chunk_body)
                    self.logger.debug(f"Sent face chunk body {face_chunk_body.seq_number} for request {request_id}")
            else:
                if isinstance(chunk, FaceChunkBody):
                    for node in dag_node.downstreams:
                        payload = node.payload
                        await payload.feed_stream(chunk)
                    self.logger.debug(
                        f"Sent face chunk body {chunk.seq_number} without aggregate for request {request_id}"
                    )
                else:
                    if self.input_buffer[request_id]["motion_blendshape_names"] is not None:
                        motion_data_wo_bs = await loop.run_in_executor(
                            self.thread_pool_executor, self._remove_blendshape_from_motion, request_id, chunk.data
                        )
                        motion_chunk_body = MotionChunkBody(
                            request_id=chunk.request_id,
                            seq_number=chunk.seq_number,
                            data=motion_data_wo_bs,
                        )
                    else:
                        motion_chunk_body = chunk
                    for node in dag_node.downstreams:
                        payload = node.payload
                        await payload.feed_stream(motion_chunk_body)
                    self.logger.debug(
                        f"Sent motion chunk body {motion_chunk_body.seq_number} without aggregate for request {request_id}"
                    )
            self.input_buffer[request_id]["body_sent_counter"][chunk_type] += 1
        except Exception as e:
            traceback_str = traceback.format_exc()
            self.logger.error(f"Error sending body chunk to downstreams: {e}\nTraceback:\n{traceback_str}")
            dag.set_status(DAGStatus.FAILED)
            raise e

    async def _send_chunk_end_task(
        self,
        request_id: str,
        chunk: Union[MotionChunkEnd, FaceChunkEnd],
    ) -> None:
        """Send end chunk to downstream components.

        Args:
            request_id (str):
                Unique identifier for the request.
            chunk (Union[MotionChunkEnd, FaceChunkEnd]):
                End chunk to send downstream.
        """
        class_name = chunk.__class__.__name__
        chunk_type = class_name.split("Chunk")[0]
        dag = self.input_buffer[request_id]["dag"]
        dag_node = dag.get_node(self.input_buffer[request_id]["node_name"])
        try:
            loop = asyncio.get_running_loop()
            if isinstance(chunk, MotionChunkEnd):
                pass
            elif isinstance(chunk, FaceChunkEnd):
                # wait for motion end and aggregate blendshape if needed
                if self.input_buffer[request_id]["blendshape_aggregate"]:
                    # wait until motion end is sent
                    while "Motion" not in self.input_buffer[request_id]["end_sent"] or (
                        self.input_buffer[request_id]["body_sent_counter"][chunk_type]
                        < self.input_buffer[request_id]["body_received_counter"][chunk_type]
                    ):
                        if dag.status == DAGStatus.RUNNING:
                            await asyncio.sleep(self.sleep_time)
                        else:
                            msg = f"Streaming face end sending interrupted by DAG status {dag.status}"
                            msg += f" for request {request_id}"
                            self.logger.warning(msg)
                            return
                    face_bytes = await loop.run_in_executor(
                        self.thread_pool_executor, self._aggregate_blendshape, request_id
                    )
                    seq_number = self.input_buffer[request_id]["body_sent_counter"][chunk_type]
                    if face_bytes is not None:
                        seq_number += 1
                        face_chunk_body = FaceChunkBody(
                            request_id=chunk.request_id,
                            seq_number=seq_number,
                            data=face_bytes,
                        )
                        for node in dag_node.downstreams:
                            payload = node.payload
                            await payload.feed_stream(face_chunk_body)
                        self.logger.debug(f"Sent face chunk body {face_chunk_body.seq_number} for request {request_id}")
                    face_bytes = await loop.run_in_executor(
                        self.thread_pool_executor, self._get_rest_motion_blendshape, request_id
                    )
                    if face_bytes is not None:
                        seq_number += 1
                        face_chunk_body = FaceChunkBody(
                            request_id=chunk.request_id,
                            seq_number=seq_number,
                            data=face_bytes,
                        )
                        for node in dag_node.downstreams:
                            payload = node.payload
                            await payload.feed_stream(face_chunk_body)
                        self.logger.debug(f"Sent face chunk body {face_chunk_body.seq_number} for request {request_id}")
            else:
                msg = f"Received unknown end chunk of class {chunk.__class__.__name__}, request id: {request_id}"
                self.logger.error(msg)
                raise TypeError(msg)
            while (
                self.input_buffer[request_id]["body_sent_counter"][chunk_type]
                < self.input_buffer[request_id]["body_received_counter"][chunk_type]
            ):
                if dag.status == DAGStatus.RUNNING:
                    await asyncio.sleep(self.sleep_time)
                else:
                    msg = f"Streaming end sending interrupted by DAG status {dag.status}"
                    msg += f" for request {request_id}"
                    self.logger.warning(msg)
                    return
            # send chunk end
            for node in dag_node.downstreams:
                payload = node.payload
                await payload.feed_stream(chunk)
            self.logger.debug(f"Sent {chunk_type} end chunk for request {request_id}")
            self.input_buffer[request_id]["end_sent"].add(chunk_type)
            if len(self.input_buffer[request_id]["end_sent"]) == 2:
                self.input_buffer.pop(request_id)
        except Exception as e:
            self.logger.error(f"Error sending end chunk to downstreams: {e}")
            dag.set_status(DAGStatus.FAILED)
            raise e

    def _append_blendshape_buffer(
        self, request_id: str, key: Literal["motion_buffer", "face_buffer"], buffer: bytes
    ) -> None:
        """Append data to the specified blendshape buffer.

        Args:
            request_id (str):
                Unique identifier for the request.
            key (Literal["motion_buffer", "face_buffer"]):
                Buffer key to append data to.
            buffer (bytes):
                Data bytes to append to the buffer.
        """
        with self.input_buffer[request_id]["blendshape_lock"]:
            self.input_buffer[request_id][key] += buffer

    def _aggregate_blendshape(self, request_id: str) -> Union[bytes, None]:
        """Aggregate blendshape data from motion and face sources.

        Args:
            request_id (str):
                Unique identifier for the request.

        Returns:
            Union[bytes, None]:
                Aggregated blendshape data as bytes, or None if no data
                is available for aggregation.
        """
        with self.input_buffer[request_id]["blendshape_lock"]:
            motion_buffer = self.input_buffer[request_id]["motion_buffer"]
            buffer_frame_idx = self.input_buffer[request_id]["buffer_frame_idx"]
            # before 0 frame, send blendshape values from only motion source
            if buffer_frame_idx < 0 and len(motion_buffer) > 0:
                motion_blendshape_names = self.input_buffer[request_id]["motion_blendshape_names"]
                motion_blendshape_name_mapping = self.input_buffer[request_id]["motion_blendshape_name_mapping"]
                motion_n_joints = self.input_buffer[request_id]["motion_n_joints"]
                motion_dtype = self.input_buffer[request_id]["motion_dtype"]
                motion_ndarray = np.frombuffer(motion_buffer, dtype=motion_dtype)
                one_frame_dim = motion_n_joints * 9 + 3 + 3 + len(motion_blendshape_names)
                mframe_motion = motion_ndarray.reshape(-1, one_frame_dim)
                n_frames = mframe_motion.shape[0]
                src_bs_values = mframe_motion[:, -len(motion_blendshape_names) :]
                if buffer_frame_idx + n_frames > 0:
                    n_frames_to_send = -buffer_frame_idx
                    n_frames_to_keep = n_frames - n_frames_to_send
                else:
                    n_frames_to_send = n_frames
                    n_frames_to_keep = 0
                face_blendshape_names = self.input_buffer[request_id]["face_blendshape_names"]
                face_dtype = self.input_buffer[request_id]["face_dtype"]
                dst_bs_values = np.zeros((n_frames_to_send, len(face_blendshape_names)), dtype=face_dtype)
                for dst_idx, bs_name in enumerate(face_blendshape_names):
                    if bs_name in motion_blendshape_name_mapping:
                        src_idx = motion_blendshape_name_mapping[bs_name]
                        dst_bs_values[:, dst_idx] = src_bs_values[:n_frames_to_send, src_idx]
                self.input_buffer[request_id]["buffer_frame_idx"] += n_frames_to_send
                if n_frames_to_keep > 0:
                    self.input_buffer[request_id]["motion_buffer"] = mframe_motion[n_frames_to_send:].tobytes()
                else:
                    self.input_buffer[request_id]["motion_buffer"] = b""
                return dst_bs_values.astype(face_dtype).tobytes()
            # after 0 frame, send blendshape values from both motion and face sources
            else:
                face_buffer = self.input_buffer[request_id]["face_buffer"]
                # either motion or face buffer is empty, or both are empty, return None
                if len(face_buffer) == 0 or len(motion_buffer) == 0:
                    return None
                # parse motion buffer
                motion_blendshape_names = self.input_buffer[request_id]["motion_blendshape_names"]
                motion_blendshape_name_mapping = self.input_buffer[request_id]["motion_blendshape_name_mapping"]
                motion_n_joints = self.input_buffer[request_id]["motion_n_joints"]
                motion_dtype = self.input_buffer[request_id]["motion_dtype"]
                motion_ndarray = np.frombuffer(motion_buffer, dtype=motion_dtype)
                bs_base_idx = motion_n_joints * 9 + 3 + 3
                one_frame_dim = bs_base_idx + len(motion_blendshape_names)
                mframe_motion = motion_ndarray.reshape(-1, one_frame_dim)
                n_frames_motion = mframe_motion.shape[0]
                # parse face buffer
                face_blendshape_names = self.input_buffer[request_id]["face_blendshape_names"]
                face_blendshape_name_mapping = self.input_buffer[request_id]["face_blendshape_name_mapping"]
                face_dtype = self.input_buffer[request_id]["face_dtype"]
                face_ndarray = np.frombuffer(face_buffer, dtype=face_dtype)
                one_frame_dim = len(face_blendshape_names)
                mframe_face = face_ndarray.reshape(-1, one_frame_dim)
                n_frames_face = mframe_face.shape[0]
                # aggregate mask
                n_frames_to_send = min(n_frames_face, n_frames_motion)
                mframe_motion_abs = np.abs(mframe_motion[:n_frames_to_send, -len(motion_blendshape_names) :])
                motion_bs_values_sum = np.sum(
                    mframe_motion_abs,
                    axis=1,
                    keepdims=False,
                )
                aggregate_mask = (motion_bs_values_sum != 0.0).astype(np.int8)
                dst_bs_values = mframe_face[:n_frames_to_send, :].copy()
                n_frames_motion_to_keep = n_frames_motion - n_frames_to_send
                n_frames_face_to_keep = n_frames_face - n_frames_to_send
                if np.any(aggregate_mask):
                    # aggregate blendshape values
                    if self.add_blendshape_names is not None:
                        for bs_name in self.add_blendshape_names:
                            if bs_name in motion_blendshape_name_mapping and bs_name in face_blendshape_name_mapping:
                                dst_idx = face_blendshape_name_mapping[bs_name]
                                motion_bs_idx = motion_blendshape_name_mapping[bs_name]
                                aggregate_value = (
                                    dst_bs_values[:, dst_idx]
                                    + mframe_motion[:n_frames_to_send, bs_base_idx + motion_bs_idx]
                                )
                                dst_bs_values[:, dst_idx] = aggregate_value * aggregate_mask + dst_bs_values[
                                    :, dst_idx
                                ] * (1 - aggregate_mask)
                    if self.motion_first_blendshape_names is not None:
                        for bs_name in self.motion_first_blendshape_names:
                            if bs_name not in face_blendshape_name_mapping:
                                continue
                            dst_idx = face_blendshape_name_mapping[bs_name]
                            if bs_name in motion_blendshape_name_mapping:
                                motion_bs_idx = motion_blendshape_name_mapping[bs_name]
                                aggregate_value = mframe_motion[:n_frames_to_send, bs_base_idx + motion_bs_idx]
                            else:
                                aggregate_value = 0.0
                            dst_bs_values[:, dst_idx] = aggregate_value * aggregate_mask + dst_bs_values[:, dst_idx] * (
                                1 - aggregate_mask
                            )
                # update buffer
                self.input_buffer[request_id]["buffer_frame_idx"] += n_frames_to_send
                if n_frames_motion_to_keep > 0:
                    self.input_buffer[request_id]["motion_buffer"] = mframe_motion[n_frames_to_send:].tobytes()
                else:
                    self.input_buffer[request_id]["motion_buffer"] = b""
                if n_frames_face_to_keep > 0:
                    self.input_buffer[request_id]["face_buffer"] = mframe_face[n_frames_to_send:].tobytes()
                else:
                    self.input_buffer[request_id]["face_buffer"] = b""
                return dst_bs_values.astype(face_dtype).tobytes()

    def _remove_blendshape_from_motion(self, request_id: str, motion_bytes_with_blendshape: bytes) -> bytes:
        """Remove blendshape data from motion data.

        Args:
            request_id (str):
                Unique identifier for the request.
            motion_bytes_with_blendshape (bytes):
                Motion data bytes containing blendshape information.

        Returns:
            bytes:
                Motion data bytes without blendshape information.
        """
        motion_blendshape_names = self.input_buffer[request_id]["motion_blendshape_names"]
        motion_n_joints = self.input_buffer[request_id]["motion_n_joints"]
        motion_dtype = self.input_buffer[request_id]["motion_dtype"]
        motion_ndarray = np.frombuffer(motion_bytes_with_blendshape, dtype=motion_dtype)
        one_frame_dim = motion_n_joints * 9 + 3 + 3 + len(motion_blendshape_names)
        mframe_motion = motion_ndarray.reshape(-1, one_frame_dim)
        values_wo_bs = mframe_motion[:, : -len(motion_blendshape_names)]
        return values_wo_bs.astype(motion_dtype).tobytes()

    def _get_rest_motion_blendshape(self, request_id: str) -> Union[bytes, None]:
        """Get remaining motion blendshape data from buffer.

        Args:
            request_id (str):
                Unique identifier for the request.

        Returns:
            Union[bytes, None]:
                Remaining motion blendshape data as bytes, or None if
                no data is available in the buffer.
        """
        with self.input_buffer[request_id]["blendshape_lock"]:
            motion_buffer = self.input_buffer[request_id]["motion_buffer"]
            if len(motion_buffer) == 0:
                return None
            motion_blendshape_names = self.input_buffer[request_id]["motion_blendshape_names"]
            motion_blendshape_name_mapping = self.input_buffer[request_id]["motion_blendshape_name_mapping"]
            motion_n_joints = self.input_buffer[request_id]["motion_n_joints"]
            motion_dtype = self.input_buffer[request_id]["motion_dtype"]
            motion_ndarray = np.frombuffer(motion_buffer, dtype=motion_dtype)
            bs_base_idx = motion_n_joints * 9 + 3 + 3
            one_frame_dim = bs_base_idx + len(motion_blendshape_names)
            mframe_motion = motion_ndarray.reshape(-1, one_frame_dim)
            n_frames_motion = mframe_motion.shape[0]
            face_blendshape_names = self.input_buffer[request_id]["face_blendshape_names"]
            face_dtype = self.input_buffer[request_id]["face_dtype"]
            dst_bs_values = np.zeros((n_frames_motion, len(face_blendshape_names)), dtype=face_dtype)
            for dst_idx, bs_name in enumerate(face_blendshape_names):
                if bs_name in motion_blendshape_name_mapping:
                    src_idx = motion_blendshape_name_mapping[bs_name]
                    dst_bs_values[:, dst_idx] = mframe_motion[:, bs_base_idx + src_idx]
            self.input_buffer[request_id]["buffer_frame_idx"] += n_frames_motion
            self.input_buffer[request_id]["motion_buffer"] = b""
            return dst_bs_values.astype(face_dtype).tobytes()
