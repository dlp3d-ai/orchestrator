import asyncio
import traceback
import wave
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Union

from pydantic import BaseModel

from ..data_structures import orchestrator_v4_pb2 as orchestrator_pb2
from ..data_structures.audio_chunk import AudioChunkBody, AudioChunkEnd, AudioChunkStart
from ..data_structures.classification import (
    ClassificationChunkBody,
    ClassificationChunkEnd,
    ClassificationChunkStart,
    ClassificationType,
)
from ..data_structures.face_chunk import FaceChunkBody, FaceChunkEnd, FaceChunkStart
from ..data_structures.motion_chunk import MotionChunkBody, MotionChunkEnd, MotionChunkStart
from ..data_structures.process_flow import DAGStatus
from ..utils.audio import resample_pcm
from ..utils.executor_registry import ExecutorRegistry
from ..utils.streamable import ChunkWithoutStartError, Streamable


class LeaveResponse(BaseModel):
    pass


class NormalResponse(BaseModel):
    pass


class FailedResponse(BaseModel):
    message: str


class CallbackAggregator(Streamable):
    """Callback aggregator for streaming data from multiple sources.

    This class aggregates streaming data from multiple sources (classification,
    audio, motion, face) and sends them to callback functions. It handles
    protobuf serialization and manages streaming data flow through callbacks.
    """

    FRAME_RATE: int = 16000
    N_CHANNELS: int = 1
    SAMPLE_WIDTH: int = 2

    ExecutorRegistry.register_class("CallbackAggregator")

    def __init__(
        self,
        thread_pool_executor: Union[None, ThreadPoolExecutor] = None,
        max_workers: int = 4,
        queue_size: int = 100,
        sleep_time: float = 0.01,
        clean_interval: float = 10.0,
        expire_time: float = 120.0,
        verbose: bool = False,
        logger_cfg: Union[None, Dict[str, Any]] = None,
    ):
        """Initialize the callback aggregator.

        Args:
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
            verbose (bool, optional):
                Enable verbose logging. Defaults to False.
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
        self.verbose = verbose

    def __del__(self):
        """Clean up resources on object destruction.

        Shuts down the thread pool executor if it was created internally.
        """
        if not self.executor_external:
            self.thread_pool_executor.shutdown(wait=False)

    async def _handle_start(
        self,
        chunk: Union[ClassificationChunkStart, AudioChunkStart, MotionChunkStart, FaceChunkStart],
        cur_time: float,
    ) -> None:
        """Handle start chunk processing.

        Args:
            chunk (Union[ClassificationChunkStart, AudioChunkStart, MotionChunkStart, FaceChunkStart]):
                Start chunk containing initialization data.
            cur_time (float):
                Current timestamp for request tracking.
        """
        request_id = chunk.request_id
        if request_id not in self.input_buffer:
            chunk_type_required = chunk.dag.conf["chunk_type_required"]
            self.input_buffer[request_id] = dict(
                chunk_type_required=chunk_type_required,
                last_update_time=cur_time,
                dag=chunk.dag,
                node_name=chunk.node_name,
                callback_instances=chunk.dag.conf["callback_instances"],
                callback_bytes_fn=chunk.dag.conf["callback_bytes_fn"],
                body_received_counter=dict(),
                body_sent_counter=dict(),
                start_sent=set(),
                end_sent=set(),
                motion_byte_rate=None,
                face_byte_rate=None,
            )
        chunk_class_str = chunk.__class__.__name__
        chunk_type = chunk_class_str.split("Chunk")[0]
        self.input_buffer[request_id]["body_received_counter"][chunk_type] = 0
        self.input_buffer[request_id]["body_sent_counter"][chunk_type] = 0
        if isinstance(chunk, AudioChunkStart):
            self.input_buffer[request_id]["audio_type"] = chunk.audio_type
            if chunk.audio_type == "pcm":
                self.input_buffer[request_id]["audio_n_channels"] = chunk.n_channels
                self.input_buffer[request_id]["audio_sample_width"] = chunk.sample_width
                self.input_buffer[request_id]["audio_frame_rate"] = chunk.frame_rate
                self.input_buffer[request_id]["audio_byte_rate"] = (
                    chunk.frame_rate * chunk.sample_width * chunk.n_channels
                )
        asyncio.create_task(self._send_chunk_start_task(request_id, chunk))

    async def _handle_body(
        self,
        chunk: Union[ClassificationChunkBody, AudioChunkBody, MotionChunkBody, FaceChunkBody],
        cur_time: float,
    ) -> None:
        """Handle body chunk processing.

        For classification chunks, sends response to callback. For audio,
        motion, and face chunks, sends binary data to callback after
        protobuf serialization.

        Args:
            chunk (Union[ClassificationChunkBody, AudioChunkBody, MotionChunkBody, FaceChunkBody]):
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
        chunk: Union[ClassificationChunkEnd, AudioChunkEnd, MotionChunkEnd, FaceChunkEnd],
        cur_time: float,
    ) -> None:
        """Handle end chunk processing.

        Sends end signal to callback. If all expected end chunks are
        received, sets DAG status to COMPLETED and removes request from
        input buffer.

        Args:
            chunk (Union[ClassificationChunkEnd, AudioChunkEnd, MotionChunkEnd, FaceChunkEnd]):
                End chunk signaling completion of data stream.
            cur_time (float):
                Current timestamp for request tracking.
        """
        request_id = chunk.request_id
        chunk_class_str = chunk.__class__.__name__
        if request_id not in self.input_buffer:
            msg = (
                f"Request {request_id} not found in input buffer, "
                + f"but received an end message of class {chunk_class_str}."
            )
            self.logger.error(msg)
            raise ChunkWithoutStartError(msg)
        self.input_buffer[request_id]["last_update_time"] = cur_time
        # send end signal to callback
        dag = self.input_buffer[request_id]["dag"]
        if dag.status == DAGStatus.RUNNING:
            asyncio.create_task(
                self._send_chunk_end_task(
                    request_id,
                    chunk,
                )
            )

    async def _send_chunk_start_task(
        self, request_id: str, chunk: Union[ClassificationChunkStart, AudioChunkStart, MotionChunkStart, FaceChunkStart]
    ) -> None:
        """Send start chunk to callback.

        Args:
            request_id (str):
                Unique identifier for the request.
            chunk (Union[ClassificationChunkStart, AudioChunkStart, MotionChunkStart, FaceChunkStart]):
                Start chunk to send to callback.
        """
        class_name = chunk.__class__.__name__
        chunk_type = class_name.split("Chunk")[0]
        loop = asyncio.get_running_loop()
        # classification is not a stream, skip sending start signal
        try:
            if isinstance(chunk, ClassificationChunkStart):
                return
            elif isinstance(chunk, AudioChunkStart):
                if chunk.audio_type == "pcm":
                    callback_bytes_fn = self.input_buffer[request_id]["callback_bytes_fn"]
                    pb_response = orchestrator_pb2.OrchestratorV4Response()
                    pb_response.class_name = "AudioChunkStart"
                    pb_response.audio_n_channels = self.__class__.N_CHANNELS
                    pb_response.audio_sample_width = self.__class__.SAMPLE_WIDTH
                    pb_response.audio_frame_rate = self.__class__.FRAME_RATE
                    pb_response_bytes = await loop.run_in_executor(
                        self.thread_pool_executor, pb_response.SerializeToString
                    )
                    await callback_bytes_fn(pb_response_bytes)
                    self.input_buffer[request_id]["start_sent"].add(chunk_type)
                # audio parameters unknown, skip sending start signal
                elif chunk.audio_type == "wav":
                    return
            elif isinstance(chunk, MotionChunkStart):
                callback_bytes_fn = self.input_buffer[request_id]["callback_bytes_fn"]
                pb_response = orchestrator_pb2.OrchestratorV4Response()
                pb_response.class_name = "MotionChunkStart"
                pb_response.dtype = chunk.dtype
                pb_response.motion_joint_names.extend(chunk.joint_names)
                pb_response.motion_restpose_name = chunk.restpose_name
                bit_width = chunk.dtype.rsplit("float", 1)[1]
                try:
                    bit_width = int(bit_width)
                    width = bit_width // 8
                except ValueError:
                    msg = f"Unknown dtype {chunk.dtype} for motion chunk, request id: {request_id}"
                    self.logger.error(msg)
                    raise ValueError(msg)
                self.input_buffer[request_id]["motion_byte_rate"] = (len(chunk.joint_names) * 9 + 3 + 3) * width
                pb_response_bytes = await loop.run_in_executor(self.thread_pool_executor, pb_response.SerializeToString)
                await callback_bytes_fn(pb_response_bytes)
                self.input_buffer[request_id]["start_sent"].add(chunk_type)
            elif isinstance(chunk, FaceChunkStart):
                callback_bytes_fn = self.input_buffer[request_id]["callback_bytes_fn"]
                pb_response = orchestrator_pb2.OrchestratorV4Response()
                pb_response.class_name = "FaceChunkStart"
                pb_response.dtype = chunk.dtype
                pb_response.face_blendshape_names.extend(chunk.blendshape_names)
                bit_width = chunk.dtype.rsplit("float", 1)[1]
                try:
                    bit_width = int(bit_width)
                    width = bit_width // 8
                except ValueError:
                    msg = f"Unknown dtype {chunk.dtype} for face chunk, request id: {request_id}"
                    self.logger.error(msg)
                    raise ValueError(msg)
                self.input_buffer[request_id]["face_byte_rate"] = len(chunk.blendshape_names) * width
                pb_response_bytes = await loop.run_in_executor(self.thread_pool_executor, pb_response.SerializeToString)
                await callback_bytes_fn(pb_response_bytes)
                self.input_buffer[request_id]["start_sent"].add(chunk_type)
            else:
                msg = f"Received unknown start chunk of class {chunk.__class__.__name__}, request id: {request_id}"
                self.logger.error(msg)
                raise TypeError(msg)
        except Exception as e:
            self.logger.error(f"Error sending start chunk to callback: {e}")
            dag = self.input_buffer[request_id]["dag"]
            dag.set_status(DAGStatus.FAILED)
            raise e

    async def _send_chunk_body_task(
        self,
        request_id: str,
        chunk: Union[ClassificationChunkBody, AudioChunkBody, MotionChunkBody, FaceChunkBody],
        seq_number: int,
    ) -> None:
        """Send body chunk to callback.

        Args:
            request_id (str):
                Unique identifier for the request.
            chunk (Union[ClassificationChunkBody, AudioChunkBody, MotionChunkBody, FaceChunkBody]):
                Body chunk to process and send.
            seq_number (int):
                Sequence number of the chunk.
        """
        class_name = chunk.__class__.__name__
        chunk_type = class_name.split("Chunk")[0]
        callback_bytes_fn = self.input_buffer[request_id]["callback_bytes_fn"]
        dag = self.input_buffer[request_id]["dag"]
        loop = asyncio.get_running_loop()
        try:
            if isinstance(chunk, ClassificationChunkBody):
                self.input_buffer[request_id]["start_sent"].add(chunk_type)
                if chunk.classification_result == ClassificationType.REJECT:
                    msg = f"Received reject signal from classification, request id: {request_id}"
                    self.logger.info(msg)
                    # response = NormalResponse()
                    pb_response = orchestrator_pb2.OrchestratorV4Response()
                    pb_response.class_name = "NormalResponse"
                    pb_response_bytes = await loop.run_in_executor(
                        self.thread_pool_executor, pb_response.SerializeToString
                    )
                    verbose_msg_suffix = f"class_name={pb_response.class_name}"
                elif chunk.classification_result == ClassificationType.LEAVE:
                    msg = f"Received leave signal from classification, request id: {request_id}"
                    self.logger.info(msg)
                    dag.set_status(DAGStatus.COMPLETED)
                    dag_name = self.input_buffer[request_id]["dag"].name
                    msg = f"DAG {dag_name} for request {request_id} completed"
                    self.logger.info(msg)
                    # response = LeaveResponse()
                    pb_response = orchestrator_pb2.OrchestratorV4Response()
                    pb_response.class_name = "LeaveResponse"
                    pb_response_bytes = await loop.run_in_executor(
                        self.thread_pool_executor, pb_response.SerializeToString
                    )
                    verbose_msg_suffix = f"class_name={pb_response.class_name}"
                elif chunk.classification_result == ClassificationType.ACCEPT:
                    msg = f"Received accept signal from classification, request id: {request_id}"
                    self.logger.info(msg)
                    # response = NormalResponse()
                    pb_response = orchestrator_pb2.OrchestratorV4Response()
                    pb_response.class_name = "NormalResponse"
                    pb_response_bytes = await loop.run_in_executor(
                        self.thread_pool_executor, pb_response.SerializeToString
                    )
                    verbose_msg_suffix = f"class_name={pb_response.class_name}"
                else:
                    dag.set_status(DAGStatus.FAILED)
                    msg = (
                        f"Received unknown classification result={chunk.classification_result} "
                        + f"from classification, request id: {request_id}"
                    )
                    self.logger.error(msg)
                    # response = FailedResponse(message=msg)
                    pb_response = orchestrator_pb2.OrchestratorV4Response()
                    pb_response.class_name = "FailedResponse"
                    pb_response.message = msg
                    pb_response_bytes = await loop.run_in_executor(
                        self.thread_pool_executor, pb_response.SerializeToString
                    )
                    verbose_msg_suffix = f"class_name={pb_response.class_name}"
            elif isinstance(chunk, AudioChunkBody):
                audio_type = self.input_buffer[request_id]["audio_type"]
                if audio_type == "wav":
                    if seq_number == 0:
                        with wave.open(chunk.audio_io, "rb") as wf:
                            n_channels = wf.getnchannels()
                            sample_width = wf.getsampwidth()
                            frame_rate = wf.getframerate()
                            n_frames = wf.getnframes()
                            pcm_bytes = wf.readframes(n_frames)
                        self.input_buffer[request_id]["audio_n_channels"] = n_channels
                        self.input_buffer[request_id]["audio_sample_width"] = sample_width
                        self.input_buffer[request_id]["audio_frame_rate"] = frame_rate
                        self.input_buffer[request_id]["audio_byte_rate"] = frame_rate * sample_width * n_channels
                        pb_response = orchestrator_pb2.OrchestratorV4Response()
                        pb_response.class_name = "AudioChunkStart"
                        pb_response.audio_n_channels = self.__class__.N_CHANNELS
                        pb_response.audio_sample_width = self.__class__.SAMPLE_WIDTH
                        pb_response.audio_frame_rate = self.__class__.FRAME_RATE
                        pb_response_bytes = await loop.run_in_executor(
                            self.thread_pool_executor, pb_response.SerializeToString
                        )
                        await callback_bytes_fn(pb_response_bytes)
                        self.input_buffer[request_id]["start_sent"].add(chunk_type)
                    else:
                        with wave.open(chunk.audio_io, "rb") as wf:
                            n_frames = wf.getnframes()
                            pcm_bytes = wf.readframes(n_frames)
                else:  # pcm
                    pcm_bytes = chunk.audio_io.getvalue()
                frame_rate = self.input_buffer[request_id]["audio_frame_rate"]
                if frame_rate != self.__class__.FRAME_RATE:
                    pcm_bytes = await loop.run_in_executor(
                        self.thread_pool_executor,
                        resample_pcm,
                        pcm_bytes,
                        frame_rate,
                        self.__class__.FRAME_RATE,
                    )
                pb_response = orchestrator_pb2.OrchestratorV4Response()
                pb_response.class_name = "AudioChunkBody"
                pb_response.data = pcm_bytes
                pb_response_bytes = await loop.run_in_executor(self.thread_pool_executor, pb_response.SerializeToString)
                duration = len(pcm_bytes) / self.input_buffer[request_id]["audio_byte_rate"]
                verbose_msg_suffix = f"class_name={pb_response.class_name}, duration={duration}"
            elif isinstance(chunk, MotionChunkBody):
                pb_response = orchestrator_pb2.OrchestratorV4Response()
                pb_response.class_name = "MotionChunkBody"
                pb_response.data = chunk.data
                pb_response_bytes = await loop.run_in_executor(self.thread_pool_executor, pb_response.SerializeToString)
                n_frames = len(pb_response.data) / self.input_buffer[request_id]["motion_byte_rate"]
                verbose_msg_suffix = f"class_name={pb_response.class_name}, n_frames={n_frames}"
            elif isinstance(chunk, FaceChunkBody):
                pb_response = orchestrator_pb2.OrchestratorV4Response()
                pb_response.class_name = "FaceChunkBody"
                pb_response.data = chunk.data
                pb_response_bytes = await loop.run_in_executor(self.thread_pool_executor, pb_response.SerializeToString)
                n_frames = len(pb_response.data) / self.input_buffer[request_id]["face_byte_rate"]
                verbose_msg_suffix = f"class_name={pb_response.class_name}, n_frames={n_frames}"
            else:
                msg = f"Received unknown body chunk of class {chunk.__class__.__name__}, request id: {request_id}"
                self.logger.error(msg)
                raise TypeError(msg)
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
            if self.verbose and len(verbose_msg_suffix) > 0:
                verbose_msg = f"Sending body chunk to callback, request id: {request_id}, " + verbose_msg_suffix
                self.logger.debug(verbose_msg)
            await callback_bytes_fn(pb_response_bytes)
            self.input_buffer[request_id]["body_sent_counter"][chunk_type] += 1
        except Exception as e:
            traceback_str = traceback.format_exc()
            self.logger.error(f"Error sending body chunk to callback: {e}\nTraceback:\n{traceback_str}")
            dag.set_status(DAGStatus.FAILED)
            raise e

    async def _send_chunk_end_task(
        self,
        request_id: str,
        chunk: Union[ClassificationChunkEnd, AudioChunkEnd, MotionChunkEnd, FaceChunkEnd],
    ) -> None:
        """Send end chunk to callback.

        Args:
            request_id (str):
                Unique identifier for the request.
            chunk (Union[ClassificationChunkEnd, AudioChunkEnd, MotionChunkEnd, FaceChunkEnd]):
                End chunk to send to callback.
        """
        class_name = chunk.__class__.__name__
        chunk_type = class_name.split("Chunk")[0]
        callback_bytes_fn = self.input_buffer[request_id]["callback_bytes_fn"]
        dag = self.input_buffer[request_id]["dag"]
        try:
            loop = asyncio.get_running_loop()
            # classification is not a stream, skip sending end signal
            if isinstance(chunk, ClassificationChunkEnd):
                pb_response_bytes = None
            elif isinstance(chunk, AudioChunkEnd):
                pb_response = orchestrator_pb2.OrchestratorV4Response()
                pb_response.class_name = "AudioChunkEnd"
                pb_response_bytes = await loop.run_in_executor(self.thread_pool_executor, pb_response.SerializeToString)
            elif isinstance(chunk, MotionChunkEnd):
                pb_response = orchestrator_pb2.OrchestratorV4Response()
                pb_response.class_name = "MotionChunkEnd"
                pb_response_bytes = await loop.run_in_executor(self.thread_pool_executor, pb_response.SerializeToString)
            elif isinstance(chunk, FaceChunkEnd):
                pb_response = orchestrator_pb2.OrchestratorV4Response()
                pb_response.class_name = "FaceChunkEnd"
                pb_response_bytes = await loop.run_in_executor(self.thread_pool_executor, pb_response.SerializeToString)
            else:
                msg = f"Received unknown end chunk of class {chunk.__class__.__name__}, request id: {request_id}"
                self.logger.error(msg)
                raise TypeError(msg)
            if pb_response_bytes is not None:
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
                await callback_bytes_fn(pb_response_bytes)
            self.input_buffer[request_id]["end_sent"].add(chunk_type)
            if len(self.input_buffer[request_id]["end_sent"]) == self.input_buffer[request_id]["chunk_type_required"]:
                dag.set_status(DAGStatus.COMPLETED)
                dag_name = self.input_buffer[request_id]["dag"].name
                msg = f"DAG {dag_name} for request {request_id} completed"
                self.logger.info(msg)
                self.input_buffer.pop(request_id)
        except Exception as e:
            self.logger.error(f"Error sending end chunk to callback: {e}")
            dag.set_status(DAGStatus.FAILED)
            raise e
