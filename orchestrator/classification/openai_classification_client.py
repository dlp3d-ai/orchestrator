import asyncio
import json
from typing import Any, Dict, Optional, Union

import httpx
import openai
from prometheus_client import Histogram

from ..data_structures.classification import ClassificationType
from ..utils.exception import MissingAPIKeyException
from .classification_adapter import ClassificationAdapter


class OpenAIClassificationClient(ClassificationAdapter):
    """Classification client for OpenAI API.

    This client provides text classification functionality through OpenAI's
    API. It supports motion keyword-based classification and uses OpenAI models
    for text analysis with configurable timeout and proxy settings.
    """

    def __init__(
        self,
        name: str,
        motion_keywords: Union[str, list[str], None],
        openai_model_name: str = "gpt-4.1-mini-2025-04-14",
        proxy_url: Union[None, str] = None,
        timeout: float = 2.0,
        latency_histogram: Histogram | None = None,
        logger_cfg: Union[None, Dict[str, Any]] = None,
    ):
        """Initialize the OpenAI classification client.

        Args:
            name (str):
                The name of the classification client.
            motion_keywords (Union[str, list[str], None]):
                The motion keywords.
            openai_model_name (str, optional):
                The name of the OpenAI model to use.
                Defaults to "gpt-4.1-mini-2025-04-14".
            proxy_url (Union[None, str], optional):
                The proxy URL for the OpenAI API.
                Defaults to None, use no proxy.
            timeout (float, optional):
                The timeout for the OpenAI API.
                Defaults to 2.0.
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
        self.openai_model_name = openai_model_name
        self.timeout = timeout

        if self.proxy_url is not None:
            self.http_client = httpx.AsyncClient(proxy=self.proxy_url)
        else:
            self.http_client = None

    async def _init_llm_client(self, request_id: str) -> None:
        """Initialize the LLM client.

        Args:
            request_id (str):
                The request id.
        """
        openai_api_key = self.input_buffer[request_id]["api_keys"].get("openai_api_key", "")
        if not openai_api_key:
            msg = "OpenAI API key is not found in the API keys."
            self.logger.error(msg)
            raise MissingAPIKeyException(msg)
        self.input_buffer[request_id]["llm_client"] = openai.AsyncOpenAI(
            api_key=openai_api_key,
            http_client=self.http_client,
            timeout=self.timeout,
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
        openai_model_name = model_name_override if model_name_override else self.openai_model_name
        try:
            response = await llm_client.chat.completions.create(
                model=openai_model_name,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": f"<user_input>: {text}"},
                ],
                temperature=1,
                max_tokens=1000,
                response_format=response_format,  # type: ignore
            )
            response = json.loads(response.choices[0].message.content)["type"]  # type: ignore
            self.logger.debug(f"Classification response: {response}")
            return ClassificationType(response)
        except Exception as e:
            self.logger.error(f"Classification error: {e}")
            raise e
