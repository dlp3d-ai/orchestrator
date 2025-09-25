from .speech2motion_streaming_client import Speech2MotionStreamingClient

_S2M_ADAPTERS = dict(
    Speech2MotionStreamingClient=Speech2MotionStreamingClient,
)


def build_speech2motion_adapter(cfg: dict) -> Speech2MotionStreamingClient:
    """Build a speech2motion adapter instance from a configuration dictionary.

    Creates and returns a speech2motion adapter instance based on the provided
    configuration. The configuration must include a 'type' field specifying
    which adapter class to instantiate, along with any required parameters
    for that adapter.

    Args:
        cfg (dict):
            Configuration dictionary containing:
            - type (str): The adapter class name to instantiate
            - Additional parameters specific to the chosen adapter type

    Returns:
        Speech2MotionStreamingClient:
            An instance of the specified speech2motion adapter class.

    Raises:
        TypeError:
            If the specified adapter type is not found in the available adapters.
    """
    cfg = cfg.copy()
    cls_name = cfg.pop("type")
    if cls_name not in _S2M_ADAPTERS:
        msg = f"Unknown speech2motion adapter type: {cls_name}"
        raise TypeError(msg)
    ret_inst = _S2M_ADAPTERS[cls_name](**cfg)
    return ret_inst
