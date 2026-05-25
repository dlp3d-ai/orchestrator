import asyncio
from typing import Any, Dict, Optional, Union

from prometheus_client import Histogram

from ..data_structures.classification import ClassificationType
from ..llm.minimax import (
    MINIMAX_DEFAULT_BASE_URL,
    MINIMAX_DEFAULT_MODEL,
    MINIMAX_EXTRA_BODY,
    build_minimax_config,
    strip_minimax_thinking,
)
from ..llm.openai_chat import complete, create_client
from .classification_adapter import ClassificationAdapter


class MiniMaxClassificationClient(ClassificationAdapter):
    """Classification client for MiniMax API using OpenAI-compatible interface.

    This client provides text classification functionality through MiniMax's
    API using the OpenAI-compatible interface. It supports motion keyword-based
    classification and uses MiniMax models for text analysis.
    """

    def __init__(
        self,
        name: str,
        motion_keywords: Union[str, list[str], None],
        minimax_model_name: str = MINIMAX_DEFAULT_MODEL,
        minimax_url: str = MINIMAX_DEFAULT_BASE_URL,
        proxy_url: Union[None, str] = None,
        timeout: float = 20.0,
        latency_histogram: Histogram | None = None,
        logger_cfg: Union[None, Dict[str, Any]] = None,
    ):
        """Initialize the MiniMax classification client.

        Args:
            name (str):
                The name of the classification client.
            motion_keywords (Union[str, list[str], None]):
                The motion keywords.
            minimax_model_name (str, optional):
                The name of the MiniMax model to use.
                Defaults to MINIMAX_DEFAULT_MODEL.
            proxy_url (Union[None, str], optional):
                The proxy URL for the MiniMax API.
                Defaults to None, use no proxy.
            timeout (float, optional):
                The timeout for the MiniMax API.
                Defaults to 20.0.
            latency_histogram (Histogram | None, optional):
                Prometheus Histogram metric for recording request latency distribution
                in seconds. If provided, latency metrics will be collected for monitoring
                purposes. Defaults to None.
            logger_cfg (Union[None, Dict[str, Any]], optional):
                The logger configuration. Defaults to None.
        """
        super().__init__(
            name=name,
            motion_keywords=motion_keywords,
            proxy_url=proxy_url,
            latency_histogram=latency_histogram,
            logger_cfg=logger_cfg,
        )
        self.minimax_model_name = minimax_model_name
        self.minimax_url = minimax_url
        self.timeout = timeout
        self.llm_provider_config = build_minimax_config(
            model_name=minimax_model_name,
            base_url=minimax_url,
            timeout=timeout,
            proxy_url=proxy_url,
        )

    async def _init_llm_client(self, request_id: str) -> None:
        """Initialize the LLM client.

        Args:
            request_id (str):
                The request id.
        """
        self.input_buffer[request_id]["llm_client"] = create_client(
            self.input_buffer[request_id]["api_keys"],
            self.llm_provider_config,
        )

    async def classify(
        self,
        request_id: str,
        prompt: str,
        text: str,
        response_format: Optional[Dict[str, Any]] = None,
        tag_prompt: Optional[str] = None,
    ) -> ClassificationType:
        """Classify the required response type according to user's text input,
        based on LLM.

        Args:
            request_id (str):
                The request id.
            prompt (str):
                The prompt to classify the text.
            text (str):
                The text to classify.
            response_format (Optional[Dict[str, Any]], optional):
                Response format specification. Defaults to None.
            tag_prompt (Optional[str], optional):
                Tag prompt for the LLM. Defaults to None.

        Returns:
            ClassificationType: The classification type.
        """
        llm_client = self.input_buffer[request_id].get("llm_client", None)
        while llm_client is None:
            await asyncio.sleep(self.sleep_time)
            llm_client = self.input_buffer[request_id].get("llm_client", None)

        model_name_override = self.input_buffer[request_id]["classification_model_override"]
        minimax_model_name = model_name_override if model_name_override else self.minimax_model_name
        system_content = prompt + "\n" + tag_prompt if tag_prompt else prompt
        try:
            response = await complete(
                client=llm_client,
                api_keys=None,
                config=self.llm_provider_config,
                model_override=minimax_model_name,
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": f"<user_input>: {text}"},
                ],
                temperature=1,
                max_tokens=1000,
                response_format=response_format,
                extra_body=MINIMAX_EXTRA_BODY,
            )
            response_text = strip_minimax_thinking(response.content)

            if "reject" in response_text.lower():
                classification_result = "reject"
            elif "leave" in response_text.lower():
                classification_result = "leave"
            else:
                classification_result = "accept"
            self.logger.debug(f"Classification response: {classification_result}")
            return ClassificationType(classification_result)
        except Exception as e:
            self.logger.error(f"Classification error: {e}")
            raise e
