import asyncio
import json
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional, Union

import httpx
import jwt
from prometheus_client import Histogram

from ..data_structures.classification import ClassificationType
from ..utils.exception import MissingAPIKeyException
from ..utils.executor_registry import ExecutorRegistry
from .classification_adapter import ClassificationAdapter


class SenseChatClassificationClient(ClassificationAdapter):
    """Classification client for SenseChat API.

    This client provides text classification functionality through SenseChat
    API. It supports motion keyword-based classification and handles
    authentication via JWT tokens.
    """

    ExecutorRegistry.register_class("SenseChatClassificationClient")

    def __init__(
        self,
        name: str,
        motion_keywords: Union[str, list[str], None],
        sensechat_model_name: str = "SenseChat-5-1202",
        sensechat_url: str = "https://api.sensenova.cn/v1/llm/chat-completions",
        proxy_url: Union[None, str] = None,
        timeout: float = 2.0,
        max_workers: int = 1,
        thread_pool_executor: ThreadPoolExecutor | None = None,
        latency_histogram: Histogram | None = None,
        logger_cfg: Union[None, Dict[str, Any]] = None,
    ):
        """Initialize the SenseChat classification client.

        Args:
            name (str):
                The name of the classification client.
            motion_keywords (Union[str, list[str], None]):
                The motion keywords.
            sensechat_model_name (str, optional):
                The name of the SenseChat model to use.
                Defaults to "SenseChat-5-1202".
            sensechat_url (str, optional):
                The SenseChat API URL.
                Defaults to "https://api.sensenova.cn/v1/llm/chat-completions".
            proxy_url (Union[None, str], optional):
                The proxy URL for the SenseChat API.
                Defaults to None, use no proxy.
            timeout (float, optional):
                The timeout for requests in seconds.
                Defaults to 2.0.
            max_workers (int, optional):
                Maximum number of worker threads. Defaults to 1.
            thread_pool_executor (ThreadPoolExecutor | None, optional):
                Thread pool executor.
                If None, a new thread pool executor will be created based on
                max_workers. Defaults to None.
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
        self.sensechat_model_name = sensechat_model_name
        self.sensechat_url = sensechat_url
        self.timeout = timeout

        if self.proxy_url is not None:
            self.http_client = httpx.AsyncClient(proxy=self.proxy_url, timeout=self.timeout)
        else:
            self.http_client = httpx.AsyncClient(timeout=self.timeout)

        self.executor = (
            thread_pool_executor if thread_pool_executor is not None else ThreadPoolExecutor(max_workers=max_workers)
        )
        self.executor_external = True if thread_pool_executor is not None else False

    def __del__(self) -> None:
        """Destructor, cleanup thread pool executor."""
        if not self.executor_external:
            self.executor.shutdown(wait=True)

    async def _init_llm_client(self, request_id: str) -> None:
        """Initialize the LLM client.

        Args:
            request_id (str):
                The request id.
        """
        pass

    def _gen_token(self, api_keys: dict) -> str:
        """Generate a JWT token for the SenseChat API.

        Args:
            api_keys (dict):
                The API keys dictionary.

        Returns:
            str:
                JWT token string for API authentication.
        """
        ak = api_keys.get("sensechat_ak", "")
        sk = api_keys.get("sensechat_sk", "")
        if not ak or not sk:
            msg = "SenseChat API key is not found in the API keys."
            self.logger.error(msg)
            raise MissingAPIKeyException(msg)

        headers = {"alg": "HS256", "typ": "JWT"}
        payload = {
            "iss": ak,
            "exp": int(time.time()) + 1800,
            "nbf": int(time.time()) - 5,
        }
        return jwt.encode(payload, sk, headers=headers)

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
                Response format specification. Not supported by SenseChat API.
                Defaults to None.
            tag_prompt (Optional[str], optional):
                Tag prompt for the LLM. Defaults to None.

        Returns:
            ClassificationType: The classification type.
        """
        api_keys = self.input_buffer[request_id]["api_keys"]
        if not api_keys.get("sensechat_ak") or not api_keys.get("sensechat_sk"):
            msg = "SenseChat API key or secret key is not found in the API keys."
            self.logger.error(msg)
            raise MissingAPIKeyException(msg)

        try:
            loop = asyncio.get_running_loop()
            jwt_token = await loop.run_in_executor(self.executor, self._gen_token, api_keys)

            model_name_override = self.input_buffer[request_id]["classification_model_override"]
            model_name = model_name_override if model_name_override else self.sensechat_model_name

            system_content = prompt + "\n" + tag_prompt if tag_prompt else prompt

            messages = [
                {
                    "role": "system",
                    "content": system_content,
                },
                {
                    "role": "user",
                    "content": f"<user_input>: {text}",
                },
            ]

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {jwt_token}",
            }

            data = {
                "max_new_tokens": 1000,
                "messages": messages,
                "model": model_name,
                "stream": False,
            }

            response = await self.http_client.post(self.sensechat_url, headers=headers, json=data)
            response.raise_for_status()

            response_data = response.json()
            data = response_data.get("data", {})
            choices = data.get("choices", [])
            response_text = choices[0].get("message", "")

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
