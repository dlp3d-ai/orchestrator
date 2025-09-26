from .asr_adapter import AutomaticSpeechRecognitionAdapter
from .openai_realtime_asr_client import OpenAIRealtimeASRClient
from .sensetime_asr_client import SensetimeASRClient
from .softsugar_asr_client import SoftSugarASRClient

_ASR_ADAPTERS = dict(
    SensetimeASRClient=SensetimeASRClient,
    SoftSugarASRClient=SoftSugarASRClient,
    OpenAIRealtimeASRClient=OpenAIRealtimeASRClient,
)


def build_asr_adapter(cfg: dict) -> AutomaticSpeechRecognitionAdapter:
    """Build an ASR adapter instance from a configuration dictionary.

    Args:
        cfg (dict):
            Configuration dictionary containing the adapter type and parameters.
            Must include 'type' key specifying the adapter class name.

    Returns:
        AutomaticSpeechRecognitionAdapter:
            An instance of the specified ASR adapter class.

    Raises:
        TypeError:
            If the specified adapter type is not supported.
    """
    cfg = cfg.copy()
    cls_name = cfg.pop("type")
    if cls_name not in _ASR_ADAPTERS:
        msg = f"Unknown asr adapter type: {cls_name}"
        raise TypeError(msg)
    ret_inst = _ASR_ADAPTERS[cls_name](**cfg)
    return ret_inst
