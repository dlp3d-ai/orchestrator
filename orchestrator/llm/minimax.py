from typing import Optional

from .openai_chat import OpenAIChatProviderConfig

MINIMAX_DEFAULT_MODEL = "MiniMax-M2.7"
MINIMAX_DEFAULT_BASE_URL = "https://api.minimaxi.com/v1"
MINIMAX_API_KEY_FIELD = "minimax_api_key"


def build_minimax_config(
    *,
    model_name: str,
    base_url: Optional[str],
    timeout: Optional[float],
    proxy_url: Optional[str],
) -> OpenAIChatProviderConfig:
    return OpenAIChatProviderConfig(
        provider_name="MiniMax",
        api_key_field=MINIMAX_API_KEY_FIELD,
        model_name=model_name,
        base_url=base_url,
        timeout=timeout,
        proxy_url=proxy_url,
    )
