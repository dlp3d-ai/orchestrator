from .memory_adapter import BaseMemoryAdapter
from .sensenova_omni_memory_client import SenseNovaOmniMemoryClient
from .xai_memory_client import XAIMemoryClient

_MEMORY_ADAPTERS = dict(
    SenseNovaOmniMemoryClient=SenseNovaOmniMemoryClient,
    XAIMemoryClient=XAIMemoryClient,
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
