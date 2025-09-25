from .audio2face_streaming_client import Audio2FaceStreamingClient

_A2F_ADAPTERS = dict(
    Audio2FaceStreamingClient=Audio2FaceStreamingClient,
)


def build_audio2face_adapter(cfg: dict) -> Audio2FaceStreamingClient:
    """Build an audio2face adapter instance from a configuration dictionary.

    Creates and returns an audio2face adapter instance based on the provided
    configuration. The configuration must include a 'type' field specifying
    which adapter class to instantiate, along with any required parameters
    for that adapter.

    Args:
        cfg (dict):
            Configuration dictionary containing:
            - type (str): The adapter class name to instantiate
            - Additional parameters specific to the chosen adapter type

    Returns:
        Audio2FaceStreamingClient:
            An instance of the specified audio2face adapter class.

    Raises:
        TypeError:
            If the specified adapter type is not found in the available adapters.
    """
    cfg = cfg.copy()
    cls_name = cfg.pop("type")
    if cls_name not in _A2F_ADAPTERS:
        msg = f"Unknown audio2face adapter type: {cls_name}"
        raise TypeError(msg)
    ret_inst = _A2F_ADAPTERS[cls_name](**cfg)
    return ret_inst
