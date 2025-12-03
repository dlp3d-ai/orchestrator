import json
from typing import Any, Dict, Optional, Union

import httpx
import openai
from prometheus_client import Histogram

from ..io.memory.database_memory_client import DatabaseMemoryClient
from ..utils.exception import MissingAPIKeyException
from .memory_adapter import BaseMemoryAdapter


class OpenAIMemoryClient(BaseMemoryAdapter):
    """OpenAI memory client that implements memory management based on OpenAI
    API.

    This class provides memory management functionality using the OpenAI API
    for LLM calls and memory operations.
    """

    def __init__(
        self,
        name: str,
        db_client: DatabaseMemoryClient,
        openai_model_name: str = "gpt-4.1-mini-2025-04-14",
        proxy_url: Union[None, str] = None,
        timeout: float = 10.0,
        conversation_char_threshold: int = 10000,
        conversation_char_target: int = 8000,
        short_term_length_threshold: int = 20,
        short_term_target_size: int = 10,
        medium_term_length_threshold: int = 10,
        input_token_number_histogram: Histogram | None = None,
        output_token_number_histogram: Histogram | None = None,
        logger_cfg: Union[None, Dict[str, Any]] = None,
    ):
        """Initialize the OpenAI memory client.

        Args:
            name (str):
                Name of the memory client.
            db_client (DatabaseMemoryClient):
                Database client for memory operations.
            openai_model_name (str, optional):
                Default OpenAI model name to use. Defaults to "gpt-4o".
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

        self.openai_model_name = openai_model_name
        self.proxy_url = proxy_url
        self.timeout = timeout

        if self.proxy_url is not None:
            self.http_client = httpx.AsyncClient(proxy=self.proxy_url)
        else:
            self.http_client = None

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
        """Call OpenAI LLM for text generation.

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
                Generated text content from the OpenAI LLM.
        """
        try:
            if not api_keys:
                raise ValueError("api_keys is required for OpenAI LLM calls")
            openai_api_key = api_keys.get("openai_api_key", "")
            if not openai_api_key:
                msg = "OpenAI API key is not found in the API keys."
                self.logger.error(msg)
                raise MissingAPIKeyException(msg)

            openai_client = openai.AsyncOpenAI(
                api_key=openai_api_key,
                http_client=self.http_client,
                timeout=self.timeout,
            )

            openai_model_name = model_override if model_override else self.openai_model_name
            response = await openai_client.chat.completions.create(
                model=openai_model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_input},
                ],
                max_tokens=max_tokens,
                response_format=response_format,  # type: ignore
            )
            content = response.choices[0].message.content
            if content is None:
                raise ValueError("LLM returned None content")
            if self.input_token_number_histogram:
                input_token_number = response.usage.prompt_tokens if response.usage else 0
                self.input_token_number_histogram.labels(adapter=self.name).observe(input_token_number)
            if self.output_token_number_histogram:
                output_token_number = response.usage.completion_tokens if response.usage else 0
                self.output_token_number_histogram.labels(adapter=self.name).observe(output_token_number)
            output = json.loads(content)["output"]
            return output
        except Exception as e:
            exception_type = type(e).__name__
            error_msg = f"OpenAI LLM call failed: {exception_type}: {e}"
            if "response" in locals() and response is not None:
                try:
                    response_content = response.choices[0].message.content if response.choices else None
                    if response_content:
                        error_msg += f" | LLM response content: {response_content[:500]}"
                except Exception:
                    pass
            self.logger.error(error_msg)
            raise e
