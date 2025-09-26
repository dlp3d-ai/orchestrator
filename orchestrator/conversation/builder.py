from .anthropic_conversation_client import AnthropicConversationClient
from .conversation_adapter import ConversationAdapter
from .deepseek_conversation_client import DeepSeekConversationClient
from .gemini_conversation_client import GeminiConversationClient
from .openai_audio_client import OpenAIAudioClient
from .openai_conversation_client import OpenAIConversationClient
from .sensenova_omni_conversation_client import SenseNovaOmniConversationClient
from .xai_conversation_client import XAIConversationClient

_CONVERSATION_ADAPTERS = dict(
    AnthropicConversationClient=AnthropicConversationClient,
    OpenAIConversationClient=OpenAIConversationClient,
    DeepSeekConversationClient=DeepSeekConversationClient,
    XAIConversationClient=XAIConversationClient,
    GeminiConversationClient=GeminiConversationClient,
    OpenAIAudioClient=OpenAIAudioClient,
    SenseNovaOmniConversationClient=SenseNovaOmniConversationClient,
)


def build_conversation_adapter(cfg: dict) -> ConversationAdapter:
    """Build a conversation adapter instance from a configuration dictionary.

    Args:
        cfg (dict):
            Configuration dictionary containing adapter type and parameters.

    Returns:
        ConversationAdapter:
            The instantiated conversation adapter.

    Raises:
        TypeError:
            If the specified adapter type is not supported.
    """
    cfg = cfg.copy()
    cls_name = cfg.pop("type")
    if cls_name not in _CONVERSATION_ADAPTERS:
        msg = f"Unknown conversation adapter type: {cls_name}"
        raise TypeError(msg)
    ret_inst = _CONVERSATION_ADAPTERS[cls_name](**cfg)
    return ret_inst
