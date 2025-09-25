from abc import ABC, abstractmethod
from typing import Any

from orchestrator.utils.streamable import Streamable

from ..data_structures.process_flow import DAGStatus


class StreamProfile(Streamable):
    """Super class for all stream profiles."""

    def __init__(self, *args, mark_status_on_end: bool = False, **kwargs):
        """Initialize the streamable object.

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
        self.mark_status_on_end = mark_status_on_end

    @abstractmethod
    async def _handle_start(self, chunk: Any, cur_time: float) -> None:
        """Handle the start chunk."""
        pass

    @abstractmethod
    async def _handle_body(self, chunk: Any, cur_time: float) -> None:
        """Handle the body chunk."""
        pass

    @abstractmethod
    async def _handle_end(self, chunk: Any, cur_time: float) -> None:
        """Handle the end chunk."""
        pass

    async def _handle_status(self, request_id: str) -> None:
        """Handle the status of the request.

        Args:
            request_id (str): The request id.
        """
        if self.mark_status_on_end:
            dag = self.input_buffer[request_id]["dag"]
            dag.set_status(DAGStatus.COMPLETED)
