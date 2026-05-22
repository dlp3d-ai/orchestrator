from typing import Any, Dict, Optional

from .openai_chat import OpenAIChatProviderConfig

SENSENOVA_DEFAULT_MODEL = "sensenova-6.7-flash-lite"
SENSENOVA_DEFAULT_BASE_URL = "https://token.sensenova.cn/v1"
SENSENOVA_API_KEY_FIELD = "sensenova_api_key"
SENSENOVA_EXTRA_BODY: Dict[str, Any] = {"thinking": {"type": "disabled"}}


def build_sensenova_config(
    *,
    model_name: str,
    base_url: Optional[str],
    timeout: Optional[float],
    proxy_url: Optional[str],
) -> OpenAIChatProviderConfig:
    return OpenAIChatProviderConfig(
        provider_name="SenseNova",
        api_key_field=SENSENOVA_API_KEY_FIELD,
        model_name=model_name,
        base_url=base_url,
        timeout=timeout,
        proxy_url=proxy_url,
    )
