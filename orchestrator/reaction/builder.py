from .dummy_reaction_client import DummyReactionClient
from .gemini_reaction_client import GeminiReactionClient
from .openai_reaction_client import OpenAIReactionClient
from .reaction_adapter import ReactionAdapter
from .sensenova_omni_reaction_client import SenseNovaOmniReactionClient
from .xai_reaction_client import XAIReactionClient

_REACTION_ADAPTERS = dict(
    GeminiReactionClient=GeminiReactionClient,
    OpenAIReactionClient=OpenAIReactionClient,
    SenseNovaOmniReactionClient=SenseNovaOmniReactionClient,
    DummyReactionClient=DummyReactionClient,
    XAIReactionClient=XAIReactionClient,
)


def build_reaction_adapter(cfg: dict) -> ReactionAdapter:
    """Build a reaction adapter instance from a configuration dictionary.

    Args:
        cfg (dict):
            Configuration dictionary containing the adapter type and parameters.
            Must include 'type' key specifying the adapter class name.

    Returns:
        ReactionAdapter:
            An instance of the specified reaction adapter class.

    Raises:
        TypeError:
            If the specified adapter type is not found in the available adapters.
    """
    cfg = cfg.copy()
    cls_name = cfg.pop("type")
    if cls_name not in _REACTION_ADAPTERS:
        msg = f"Unknown reaction adapter type: {cls_name}"
        raise TypeError(msg)
    ret_inst = _REACTION_ADAPTERS[cls_name](**cfg)
    return ret_inst
