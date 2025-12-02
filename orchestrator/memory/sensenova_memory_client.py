import asyncio
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional, Union

import httpx
import jwt
from prometheus_client import Histogram

from ..io.memory.database_memory_client import DatabaseMemoryClient
from ..utils.exception import MissingAPIKeyException
from ..utils.executor_registry import ExecutorRegistry
from .memory_adapter import BaseMemoryAdapter


class SenseNovaMemoryClient(BaseMemoryAdapter):
    """SenseNova memory client that implements memory management based on
    SenseNova API.

    This class provides memory management functionality using the SenseNova API
    for LLM calls and memory operations.
    """

    ExecutorRegistry.register_class("SenseNovaMemoryClient")

    def __init__(
        self,
        name: str,
        db_client: DatabaseMemoryClient,
        sensenova_model_name: str = "SenseNova-V6-5-Pro",
        sensenova_url: str = "https://api.sensenova.cn/v1/llm/chat-completions",
        proxy_url: Union[None, str] = None,
        timeout: float = 10.0,
        conversation_char_threshold: int = 10000,
        conversation_char_target: int = 8000,
        short_term_length_threshold: int = 20,
        short_term_target_size: int = 10,
        medium_term_length_threshold: int = 10,
        max_workers: int = 1,
        thread_pool_executor: ThreadPoolExecutor | None = None,
        input_token_number_histogram: Histogram | None = None,
        output_token_number_histogram: Histogram | None = None,
        logger_cfg: Union[None, Dict[str, Any]] = None,
    ):
        """Initialize the SenseNova memory client.

        Args:
            name (str):
                Name of the memory client.
            db_client (DatabaseMemoryClient):
                Database client for memory operations.
            sensenova_model_name (str, optional):
                Default SenseNova model name to use. Defaults to "SenseNova-V6-5-Pro".
            sensenova_url (str, optional):
                SenseNova API URL. Defaults to "https://api.sensenova.cn/v1/llm/chat-completions".
            proxy_url (Union[None, str], optional):
                Proxy URL for API requests. Defaults to None.
            timeout (float, optional):
                Request timeout in seconds. Defaults to 10.0.
            conversation_char_threshold (int, optional):
                Character threshold for conversation compression. Defaults to 10000.
            conversation_char_target (int, optional):
                Target character count for conversation compression. Defaults to 8000.
            short_term_length_threshold (int, optional):
                Length threshold for short-term memory compression. Defaults to 20.
            short_term_target_size (int, optional):
                Target size for short-term memory compression. Defaults to 10.
            medium_term_length_threshold (int, optional):
                Length threshold for medium-term memory compression. Defaults to 10.
            max_workers (int, optional):
                Maximum number of worker threads. Defaults to 1.
            thread_pool_executor (ThreadPoolExecutor | None, optional):
                Thread pool executor.
                If None, a new thread pool executor will be created based on
                max_workers. Defaults to None.
            input_token_number_histogram (Histogram | None, optional):
                Prometheus Histogram metric for recording input token count distribution
                per request. If provided, input token usage metrics will be collected for
                monitoring purposes. Defaults to None.
            output_token_number_histogram (Histogram | None, optional):
                Prometheus Histogram metric for recording output token count distribution
                per request. If provided, output token usage metrics will be collected for
                monitoring purposes. Defaults to None.
            logger_cfg (Union[None, Dict[str, Any]], optional):
                Logger configuration. Defaults to None.
        """
        super().__init__(
            name=name,
            db_client=db_client,
            conversation_char_threshold=conversation_char_threshold,
            conversation_char_target=conversation_char_target,
            short_term_length_threshold=short_term_length_threshold,
            short_term_target_size=short_term_target_size,
            medium_term_length_threshold=medium_term_length_threshold,
            input_token_number_histogram=input_token_number_histogram,
            output_token_number_histogram=output_token_number_histogram,
            logger_cfg=logger_cfg,
        )

        self.sensenova_model_name = sensenova_model_name
        self.sensenova_url = sensenova_url
        self.proxy_url = proxy_url
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

    def _gen_token(self, api_keys: dict) -> str:
        """Generate a JWT token for the SenseNova API authentication.

        Args:
            api_keys (dict):
                The API keys dictionary.

        Returns:
            str:
                JWT token string for API authentication.
        """
        ak = api_keys.get("sensenova_ak", "")
        sk = api_keys.get("sensenova_sk", "")
        if not ak or not sk:
            msg = "SenseNova API key is not found in the API keys."
            self.logger.error(msg)
            raise MissingAPIKeyException(msg)

        headers = {"alg": "HS256", "typ": "JWT"}
        payload = {
            "iss": ak,
            "exp": int(time.time()) + 1800,
            "nbf": int(time.time()) - 5,
        }
        return jwt.encode(payload, sk, headers=headers)

    async def call_llm(
        self,
        system_prompt: str,
        user_input: str,
        max_tokens: int,
        response_format: Optional[Dict[str, Any]] = None,
        tag_prompt: Optional[str] = None,
        api_keys: Optional[Dict[str, Any]] = None,
        model_override: Optional[str] = None,
    ) -> str:
        """Call SenseNova LLM for text generation.

        Args:
            system_prompt (str):
                System prompt for the LLM.
            user_input (str):
                User input for the LLM.
            max_tokens (int):
                Maximum number of tokens to generate.
            response_format (Optional[Dict[str, Any]], optional):
                Response format specification. Defaults to None.
            tag_prompt (Optional[str], optional):
                Tag prompt for the LLM. Defaults to None.
            api_keys (Optional[Dict[str, Any]], optional):
                API keys for the LLM. Defaults to None.
            model_override (Optional[str], optional):
                Model name override. Defaults to None.

        Returns:
            str:
                Generated text content from the SenseNova LLM.
        """
        try:
            if not api_keys:
                raise ValueError("api_keys is required for SenseNova LLM calls")
            sensenova_ak = api_keys.get("sensenova_ak", "")
            sensenova_sk = api_keys.get("sensenova_sk", "")
            if not sensenova_ak or not sensenova_sk:
                msg = "SenseNova API key is not found in the API keys."
                self.logger.error(msg)
                raise MissingAPIKeyException(msg)

            loop = asyncio.get_running_loop()
            jwt_token = await loop.run_in_executor(self.executor, self._gen_token, api_keys)

            sensenova_model_name = model_override if model_override else self.sensenova_model_name

            system_content = system_prompt + "\n" + tag_prompt if tag_prompt else system_prompt

            messages = [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": system_content},
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_input},
                    ],
                },
            ]

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {jwt_token}",
            }

            data = {
                "max_new_tokens": max_tokens,
                "messages": messages,
                "model": sensenova_model_name,
                "stream": False,
                "thinking": {"enabled": False},
            }

            response = await self.http_client.post(self.sensenova_url, headers=headers, json=data)
            response.raise_for_status()

            response_data = response.json()
            data = response_data.get("data", {})
            choices = data.get("choices", [])
            content = choices[0].get("message", "")

            if not content:
                raise ValueError("LLM returned empty content")

            if self.input_token_number_histogram:
                input_token_number = response_data.get("usage", {}).get("prompt_tokens", 0)
                self.input_token_number_histogram.labels(adapter=self.name).observe(input_token_number)
            if self.output_token_number_histogram:
                output_token_number = response_data.get("usage", {}).get("completion_tokens", 0)
                self.output_token_number_histogram.labels(adapter=self.name).observe(output_token_number)

            match = re.search(r"<output>(.*?)</output>", content, re.DOTALL)
            if match:
                output = match.group(1)
            else:
                self.logger.warning(f"Failed to extract <output> tag from content: {content}")
                output = content
            return output
        except Exception as e:
            exception_type = type(e).__name__
            error_msg = f"SenseNova LLM call failed: {exception_type}: {e}"
            if "response" in locals() and response is not None:
                try:
                    response_text = response.text if hasattr(response, "text") else None
                    if response_text:
                        error_msg += f" | LLM response content: {response_text[:500]}"
                except Exception:
                    pass
            elif "response_data" in locals() and response_data is not None:
                try:
                    response_str = str(response_data)[:500]
                    if response_str:
                        error_msg += f" | LLM response data: {response_str}"
                except Exception:
                    pass
            self.logger.error(error_msg)
            raise e
