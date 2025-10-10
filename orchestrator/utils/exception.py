from typing import Callable

from ..data_structures import orchestrator_v4_pb2 as orchestrator_pb2


class MissingAPIKeyException(Exception):
    """Exception raised when a required API key is not found.

    This exception is raised when a required API key is not found in the
    configuration.
    """

    pass


async def failure_callback(msg: str, callback_bytes_fn: Callable) -> None:
    """Send a failure response message through the callback function.

    This function creates a protobuf failure response with the provided error message
    and sends it through the specified callback function. It is typically used to
    notify clients about errors that occurred during request processing.

    Args:
        msg (str):
            The error message to include in the failure response.
        callback_bytes_fn (Callable):
            The callback function that will send the protobuf response bytes.
            This function should be awaitable and accept bytes as its argument.
    """
    pb_response = orchestrator_pb2.OrchestratorV4Response()
    pb_response.class_name = "FailedResponse"
    pb_response.message = msg
    pb_response_bytes = pb_response.SerializeToString()
    await callback_bytes_fn(pb_response_bytes)
