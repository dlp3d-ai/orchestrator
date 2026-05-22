import re
from typing import Any, Dict, Optional, Union

from prometheus_client import Histogram

from ..io.memory.database_memory_client import DatabaseMemoryClient
from ..utils.executor_registry import ExecutorRegistry
from ..llm.minimax import MINIMAX_DEFAULT_BASE_URL, MINIMAX_DEFAULT_MODEL, build_minimax_config
from ..llm.openai_chat import complete
from .memory_adapter import BaseMemoryAdapter


class MiniMaxMemoryClient(BaseMemoryAdapter):
    """MiniMax memory client that implements memory management based on
    MiniMax API.

    This class provides memory management functionality using the MiniMax API
    for LLM calls and memory operations.
    """

    def __init__(
        self,
        name: str,
        db_client: DatabaseMemoryClient,
        minimax_model_name: str = MINIMAX_DEFAULT_MODEL,
        minimax_url: str = MINIMAX_DEFAULT_BASE_URL,
        proxy_url: Union[None, str] = None,
        timeout: float = 20.0,
        conversation_char_threshold: int = 10000,
        conversation_char_target: int = 8000,
        short_term_length_threshold: int = 20,
        short_term_target_size: int = 10,
        medium_term_length_threshold: int = 10,
        input_token_number_histogram: Histogram | None = None,
        output_token_number_histogram: Histogram | None = None,
        logger_cfg: Union[None, Dict[str, Any]] = None,
    ):
        """Initialize the MiniMax memory client.

        Args:
            name (str):
                Name of the memory client.
            db_client (DatabaseMemoryClient):
                Database client for memory operations.
            minimax_model_name (str, optional):
                Default MiniMax model name to use. Defaults to MINIMAX_DEFAULT_MODEL.
            proxy_url (Union[None, str], optional):
                Proxy URL for API requests. Defaults to None.
            timeout (float, optional):
                Request timeout in seconds. Defaults to 20.0.
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

        self.minimax_model_name = minimax_model_name
        self.minimax_url = minimax_url
        self.proxy_url = proxy_url
        self.timeout = timeout
        self.llm_provider_config = build_minimax_config(
            model_name=minimax_model_name,
            base_url=minimax_url,
            timeout=timeout,
            proxy_url=proxy_url,
        )

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
        """Call MiniMax LLM for text generation.

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
                Generated text content from the MiniMax LLM.
        """
        try:
            minimax_model_name = model_override if model_override else self.minimax_model_name

            system_content = system_prompt + "\n" + tag_prompt if tag_prompt else system_prompt

            response = await complete(
                api_keys=api_keys,
                config=self.llm_provider_config,
                model_override=minimax_model_name,
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_input},
                ],
                temperature=1,
                max_tokens=max_tokens,
            )

            if self.input_token_number_histogram:
                self.input_token_number_histogram.labels(adapter=self.name).observe(response.usage.prompt_tokens)
            if self.output_token_number_histogram:
                self.output_token_number_histogram.labels(adapter=self.name).observe(response.usage.completion_tokens)

            match = re.search(r"<output>(.*?)</output>", response.content, re.DOTALL)
            if match:
                output = match.group(1)
            else:
                self.logger.warning(f"Failed to extract <output> tag from content: {response.content}")
                output = response.content
            return output
        except Exception as e:
            exception_type = type(e).__name__
            error_msg = f"MiniMax LLM call failed: {exception_type}: {e}"
            self.logger.error(error_msg)
            raise e
