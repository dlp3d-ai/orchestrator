from .classification_adapter import ClassificationAdapter
from .deepseek_classification_client import DeepSeekClassificationClient
from .gemini_classification_client import GeminiClassificationClient
from .openai_classification_client import OpenAIClassificationClient
from .sensechat_classification_client import SenseChatClassificationClient
from .sensenova_classification_client import SenseNovaClassificationClient
from .sensenova_omni_classification_client import SenseNovaOmniClassificationClient
from .xai_classification_client import XAIClassificationClient

_CLASSIFICATION_ADAPTERS = dict(
    OpenAIClassificationClient=OpenAIClassificationClient,
    XAIClassificationClient=XAIClassificationClient,
    GeminiClassificationClient=GeminiClassificationClient,
    SenseNovaOmniClassificationClient=SenseNovaOmniClassificationClient,
    SenseNovaClassificationClient=SenseNovaClassificationClient,
    SenseChatClassificationClient=SenseChatClassificationClient,
    DeepSeekClassificationClient=DeepSeekClassificationClient,
)


def build_classification_adapter(cfg: dict) -> ClassificationAdapter:
    """Build a classification adapter instance from a configuration dictionary.

    Args:
        cfg (dict):
            Configuration dictionary containing the adapter type and parameters.
            Must include a 'type' key specifying the adapter class name.

    Returns:
        ClassificationAdapter:
            Instance of the specified classification adapter.

    Raises:
        TypeError:
            If the specified adapter type is not supported.
    """
    cfg = cfg.copy()
    cls_name = cfg.pop("type")
    if cls_name not in _CLASSIFICATION_ADAPTERS:
        msg = f"Unknown classification adapter type: {cls_name}"
        raise TypeError(msg)
    ret_inst = _CLASSIFICATION_ADAPTERS[cls_name](**cfg)
    return ret_inst
