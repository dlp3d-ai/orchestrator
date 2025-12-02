from .chatterbox_tts_client import ChatterboxTTSClient
from .elevenlabs_tts_client import ElevenLabsTTSClient
from .huoshan_tts_client import HuoshanTTSClient
from .sensenova_tts_client import SensenovaTTSClient
from .sensetime_tts_client import SensetimeTTSClient
from .softsugar_tts_client import SoftSugarTTSClient
from .tts_adapter import TextToSpeechAdapter

_TTS_ADAPTERS = dict(
    SensetimeTTSClient=SensetimeTTSClient,
    HuoshanTTSClient=HuoshanTTSClient,
    SoftSugarTTSClient=SoftSugarTTSClient,
    SensenovaTTSClient=SensenovaTTSClient,
    ElevenLabsTTSClient=ElevenLabsTTSClient,
    ChatterboxTTSClient=ChatterboxTTSClient,
)


def build_tts_adapter(
    cfg: dict,
) -> TextToSpeechAdapter:
    """Build a TTS adapter instance from a configuration dictionary.

    Args:
        cfg (dict):
            Configuration dictionary containing:
            - `type` (str): TTS adapter class name (e.g., 'ElevenLabsTTSClient')
            - Additional parameters specific to the chosen adapter type

    Returns:
        TextToSpeechAdapter:
            Configured TTS adapter instance of the specified type.

    Raises:
        TypeError:
            If the specified adapter type is not found in the registry.
    """
    cfg = cfg.copy()
    cls_name = cfg.pop("type")
    if cls_name not in _TTS_ADAPTERS:
        msg = f"Unknown text2speech adapter type: {cls_name}"
        raise TypeError(msg)
    ret_inst = _TTS_ADAPTERS[cls_name](**cfg)
    return ret_inst
