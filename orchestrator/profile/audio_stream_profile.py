import os
import wave
from typing import Union

from ..data_structures.audio_chunk import (
    AudioChunkBody,
    AudioChunkEnd,
    AudioChunkStart,
    AudioWithSubtitleChunkBody,
    AudioWithSubtitleChunkEnd,
    AudioWithSubtitleChunkStart,
)
from ..data_structures.process_flow import DAGStatus
from ..utils.streamable import ChunkWithoutStartError
from .stream_profile import StreamProfile


class AudioStreamProfile(StreamProfile):
    """Audio stream profile."""

    def __init__(self, *args, save_dir: Union[str, None] = None, **kwargs):
        """Initialize the streamable object.

        Args:
            save_dir (Union[str, None], optional):
                The directory to save the audio.
                Defaults to None.
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
        self.save_dir = save_dir

    async def _handle_start(self, chunk: Union[AudioChunkStart, AudioWithSubtitleChunkStart], cur_time: float) -> None:
        """Handle the start chunk.

        Args:
            chunk (Union[AudioChunkStart, AudioWithSubtitleChunkStart]):
                The audio start chunk containing request information.
            cur_time (float):
                Current timestamp.
        """
        msg = (
            f"Received start chunk for request {chunk.request_id}, "
            + f"dag progress: {chunk.node_name}, "
            + f"audio type: {chunk.audio_type}."
        )
        self.input_buffer[chunk.request_id] = dict(
            body_count=0,
            audio_type=chunk.audio_type,
            dag=chunk.dag,
            node_name=chunk.node_name,
            last_update_time=cur_time,
        )
        self.logger.info(msg)

    async def _handle_body(self, chunk: Union[AudioChunkBody, AudioWithSubtitleChunkBody], cur_time: float) -> None:
        """Handle the body chunk.

        Args:
            chunk (Union[AudioChunkBody, AudioWithSubtitleChunkBody]):
                The audio body chunk containing audio data and optional subtitle information.
            cur_time (float):
                Current timestamp.
        """
        duration = chunk.duration
        audio_io = chunk.audio_io
        seq_number = chunk.seq_number
        msg = f"Received body chunk for request {chunk.request_id}, duration: {duration}, seq_number: {seq_number}"
        if isinstance(chunk, AudioWithSubtitleChunkBody):
            speech_time = chunk.speech_time
            speech_text = chunk.speech_text
            msg += f", speech text: {speech_text}, speech time: {speech_time}."
        else:
            msg += "."
        self.logger.info(msg)
        self.input_buffer[chunk.request_id]["last_update_time"] = cur_time
        if self.save_dir is not None:
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
            cur_count = self.input_buffer[chunk.request_id]["body_count"]
            audio_type = self.input_buffer[chunk.request_id]["audio_type"]
            save_path = os.path.join(self.save_dir, f"{chunk.request_id}_{cur_count:03d}.{audio_type}")
            with open(save_path, "wb") as f:
                f.write(audio_io.getvalue())
            msg = f"Saved audio to {save_path}."
            self.logger.info(msg)
        if chunk.request_id not in self.input_buffer:
            msg = f"Request {chunk.request_id} not found in input buffer, but received a body message."
            self.logger.error(msg)
            raise ChunkWithoutStartError(msg)
        if self.input_buffer[chunk.request_id]["audio_type"] == "wav":
            with wave.open(audio_io, "rb") as f:
                rate = f.getframerate()
                channels = f.getnchannels()
                sample_width = f.getsampwidth()
                msg = f"Received wav audio, rate: {rate}, channels: {channels}, sample width: {sample_width}."
                self.logger.info(msg)
        dag = self.input_buffer[chunk.request_id]["dag"]
        if dag.status != DAGStatus.RUNNING:
            msg = f"DAG {dag.name} is not running."
            self.logger.error(msg)
            raise ChunkWithoutStartError(msg)
        self.input_buffer[chunk.request_id]["body_count"] += 1

    async def _handle_end(self, chunk: Union[AudioChunkEnd, AudioWithSubtitleChunkEnd], cur_time: float) -> None:
        """Handle the end chunk.

        Args:
            chunk (Union[AudioChunkEnd, AudioWithSubtitleChunkEnd]):
                The audio end chunk signaling completion.
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
