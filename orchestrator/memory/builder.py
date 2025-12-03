from .deepseek_memory_client import DeepSeekMemoryClient
from .gemini_memory_client import GeminiMemoryClient
from .memory_adapter import BaseMemoryAdapter
from .openai_memory_client import OpenAIMemoryClient
from .sensechat_memory_client import SenseChatMemoryClient
from .sensenova_memory_client import SenseNovaMemoryClient
from .sensenova_omni_memory_client import SenseNovaOmniMemoryClient
from .xai_memory_client import XAIMemoryClient

_MEMORY_ADAPTERS = dict(
    OpenAIMemoryClient=OpenAIMemoryClient,
    SenseNovaOmniMemoryClient=SenseNovaOmniMemoryClient,
    XAIMemoryClient=XAIMemoryClient,
    SenseNovaMemoryClient=SenseNovaMemoryClient,
    SenseChatMemoryClient=SenseChatMemoryClient,
    GeminiMemoryClient=GeminiMemoryClient,
    DeepSeekMemoryClient=DeepSeekMemoryClient,
)


def build_memory_adapter(cfg: dict) -> BaseMemoryAdapter:
    """Build a memory adapter instance from a configuration dictionary.

    Args:
        cfg (dict):
            Configuration dictionary containing adapter type and parameters.

    Returns:
        BaseMemoryAdapter:
            Configured memory adapter instance.

    Raises:
        TypeError:
            If the specified adapter type is not supported.
    """
    cfg = cfg.copy()
    cls_name = cfg.pop("type")
    if cls_name not in _MEMORY_ADAPTERS:
        msg = f"Unknown memory adapter type: {cls_name}"
        raise TypeError(msg)
    ret_inst = _MEMORY_ADAPTERS[cls_name](**cfg)
    return ret_inst
