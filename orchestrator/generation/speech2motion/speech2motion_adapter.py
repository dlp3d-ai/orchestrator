import io
from abc import abstractmethod
from typing import Any, Union

from prometheus_client import Histogram

from ...utils.streamable import Streamable


class Speech2MotionAdapter(Streamable):
    """Speech to motion adapter for generating motion animation from speech
    text.

    Abstract base class for speech-to-motion generation adapters. Provides the
    interface for converting speech text input into motion animation data.
    Supports both streaming and non-streaming modes of operation.

    Attributes:
        AVAILABLE_FOR_STREAM (bool):
            Whether this adapter supports streaming mode. Defaults to False.
    """

    AVAILABLE_FOR_STREAM = False

    def __init__(
        self,
        *args,
        latency_histogram: Histogram | None = None,
        **kwargs,
    ):
        """Initialize the speech2motion adapter.

        Sets up the adapter with configuration for queue management, timing
        parameters, and logging. Inherits from Streamable to provide streaming
        capabilities for speech processing.

        Args:
            *args:
                Variable length argument list passed to the parent Streamable class.
            latency_histogram (Histogram | None, optional):
                Prometheus Histogram metric for recording request latency distribution
                in seconds. If provided, latency metrics will be collected for monitoring
                purposes. Defaults to None.
            **kwargs:
                Arbitrary keyword arguments passed to the parent Streamable class.
        """
        super().__init__(*args, **kwargs)
        self.latency_histogram = latency_histogram

    @abstractmethod
    async def generate_speech2motion(
        self,
        *args: Any,
        request_id: Union[str, None] = None,
        **kwargs: Any,
    ) -> io.BytesIO:
        """Generate motion animation data from speech text input.

        Abstract method that must be implemented by subclasses to convert
        speech text into motion animation data. The implementation should
        handle the specific speech-to-motion generation logic.

        Args:
            *args (Any):
                Variable length argument list specific to the implementation.
                Typically includes speech text, duration, avatar information, etc.
            request_id (Union[str, None], optional):
                Unique identifier for request tracking and logging purposes.
                Defaults to None, making log messages non-identifiable.
            **kwargs (Any):
                Additional keyword arguments specific to the implementation.
                May include motion keywords, timing information, user preferences, etc.

        Returns:
            io.BytesIO:
                Generated motion animation data in BytesIO format, typically
                containing joint transformations, root bone translations, and
                other motion parameters.

        Raises:
            NotImplementedError:
                This method must be implemented by subclasses.
        """
        raise NotImplementedError
