import asyncio
import json
from typing import Any, Dict, Optional, Union

import httpx
import openai

from ..data_structures.classification import ClassificationType
from ..utils.exception import MissingAPIKeyException
from .classification_adapter import ClassificationAdapter


class GeminiClassificationClient(ClassificationAdapter):
    """Classification client for Google Gemini API using OpenAI-compatible
    interface.

    This client provides text classification functionality through Google's
    Gemini API using the OpenAI-compatible interface. It supports motion
    keyword-based classification and uses Gemini models for text analysis.
    """

    def __init__(
        self,
        name: str,
        motion_keywords: Union[str, list[str], None],
        gemini_model_name: str = "gemini-2.5-flash-lite",
        proxy_url: Union[None, str] = None,
        logger_cfg: Union[None, Dict[str, Any]] = None,
    ):
        """Initialize the Gemini classification client.

        Args:
            name (str):
                The name of the classification client.
            motion_keywords (Union[str, list[str], None]):
                The motion keywords.
            gemini_model_name (str, optional):
                The name of the Gemini model to use.
                Defaults to "gemini-2.5-flash-lite".
            proxy_url (Union[None, str], optional):
                The proxy URL for the Gemini API.
                Defaults to None, use no proxy.
            logger_cfg (Union[None, Dict[str, Any]], optional):
                The logger configuration. Defaults to None.
        """
        super().__init__(
            name=name,
            motion_keywords=motion_keywords,
            proxy_url=proxy_url,
            logger_cfg=logger_cfg,
        )
        self.gemini_model_name = gemini_model_name
        self.gemini_base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"

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
        gemini_api_key = self.input_buffer[request_id]["api_keys"].get("gemini_api_key", "")
        if not gemini_api_key:
            msg = "Gemini API key is not found in the API keys."
            self.logger.error(msg)
            raise MissingAPIKeyException(msg)

        self.input_buffer[request_id]["llm_client"] = openai.AsyncOpenAI(
            api_key=gemini_api_key,
            http_client=self.http_client,
            base_url=self.gemini_base_url,
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
        gemini_model_name = model_name_override if model_name_override else self.gemini_model_name
        try:
            response = await llm_client.chat.completions.create(
                model=gemini_model_name,
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
