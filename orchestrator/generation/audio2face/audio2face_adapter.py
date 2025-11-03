import io
from abc import abstractmethod
from asyncio import QueueFull
from typing import Any, Dict, Union

from prometheus_client import Histogram

from ...data_structures.audio_chunk import (
    AudioChunkBody,
    AudioChunkEnd,
    AudioChunkStart,
    AudioWithReactionChunkBody,
    AudioWithReactionChunkEnd,
    AudioWithReactionChunkStart,
)
from ...utils.streamable import Streamable


class Audio2FaceAdapter(Streamable):
    """Audio to face adapter for generating facial animation from audio data.

    Abstract base class for audio-to-face generation adapters. Provides the
    interface for converting audio input into facial animation data (blendshapes).
    Supports both streaming and non-streaming modes of operation.

    Attributes:
        AVAILABLE_FOR_STREAM (bool):
            Whether this adapter supports streaming mode. Defaults to False.
        N_CHANNELS (int):
            Number of audio channels expected. Defaults to 1 (mono).
        SAMPLE_WIDTH (int):
            Sample width in bytes. Defaults to 2 (16-bit).
        FRAME_RATE (int):
            Audio frame rate in Hz. Defaults to 16000.
    """

    AVAILABLE_FOR_STREAM = False
    N_CHANNELS: int = 1
    SAMPLE_WIDTH: int = 2
    FRAME_RATE: int = 16000

    def __init__(
        self,
        queue_size: int = 100,
        sleep_time: float = 0.01,
        clean_interval: float = 10.0,
        expire_time: float = 120.0,
        latency_histogram: Histogram | None = None,
        logger_cfg: Union[None, Dict[str, Any]] = None,
    ):
        """Initialize the audio2face adapter.

        Sets up the adapter with configuration for queue management, timing
        parameters, and logging. Inherits from Streamable to provide streaming
        capabilities for audio processing.

        Args:
            queue_size (int, optional):
                Maximum size of the internal processing queue for audio chunks.
                Defaults to 100.
            sleep_time (float, optional):
                Sleep interval in seconds between processing cycles.
                Defaults to 0.01.
            clean_interval (float, optional):
                Interval in seconds for cleaning expired requests from memory.
                Defaults to 10.0.
            expire_time (float, optional):
                Time in seconds after which requests are considered expired.
                Defaults to 120.0.
            latency_histogram (Histogram | None, optional):
                Prometheus Histogram metric for recording request latency distribution
                in seconds. If provided, latency metrics will be collected for monitoring
                purposes. Defaults to None.
            logger_cfg (Union[None, Dict[str, Any]], optional):
                Logger configuration dictionary. If None, default logging is used.
                Defaults to None.
        """
        Streamable.__init__(
            self,
            queue_size=queue_size,
            sleep_time=sleep_time,
            clean_interval=clean_interval,
            expire_time=expire_time,
            logger_cfg=logger_cfg,
        )
        self.latency_histogram = latency_histogram

    @abstractmethod
    async def generate_audio2face(
        self,
        audio: Union[str, io.BytesIO],
        request_id: Union[str, None] = None,
        **kwargs: Any,
    ) -> io.BytesIO:
        """Generate facial animation data from audio input.

        Abstract method that must be implemented by subclasses to convert
        audio data into facial animation blendshapes. The implementation
        should handle the specific audio-to-face generation logic.

        Args:
            audio (Union[str, io.BytesIO]):
                Audio data to generate facial animation from. Can be a file path
                (str) or audio data in BytesIO format.
            request_id (Union[str, None], optional):
                Unique identifier for request tracking and logging purposes.
                Defaults to None, making log messages non-identifiable.
            **kwargs (Any):
                Additional keyword arguments specific to the implementation.

        Returns:
            io.BytesIO:
                Generated facial animation data in BytesIO format, typically
                containing blendshape values or other facial animation parameters.

        Raises:
            NotImplementedError:
                This method must be implemented by subclasses.
        """
        raise NotImplementedError

    async def feed_stream(
        self,
        chunk: Union[
            AudioChunkStart,
            AudioChunkBody,
            AudioChunkEnd,
            AudioWithReactionChunkStart,
            AudioWithReactionChunkBody,
            AudioWithReactionChunkEnd,
        ],
    ) -> None:
        """Feed audio chunks to the audio2face adapter for processing.

        Adds audio chunks to the internal processing queue for streaming
        audio-to-face generation. Supports both basic audio chunks and
        audio chunks with reaction/emotion information.

        Args:
            chunk (Union[AudioChunkStart, AudioChunkBody, AudioChunkEnd,
                         AudioWithReactionChunkStart, AudioWithReactionChunkBody,
                         AudioWithReactionChunkEnd]):
                Audio chunk to be processed. Can be start, body, or end chunks,
                with or without reaction information.

        Raises:
            QueueFull:
                If the internal processing queue is full and cannot accept
                more chunks.
        """
        try:
            self.queue.put_nowait(chunk)
        except QueueFull as e:
            msg = "The queue is full"
            self.logger.error(msg)
            raise e
