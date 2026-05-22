import json
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional, Union

from prometheus_client import Histogram

from ..io.memory.database_memory_client import DatabaseMemoryClient
from ..llm.openai_chat import complete
from ..llm.sensenova import (
    SENSENOVA_DEFAULT_BASE_URL,
    SENSENOVA_DEFAULT_MODEL,
    SENSENOVA_EXTRA_BODY,
    build_sensenova_config,
)
from ..utils.executor_registry import ExecutorRegistry
from .memory_adapter import BaseMemoryAdapter


class SenseNovaMemoryClient(BaseMemoryAdapter):
    """SenseNova memory client using the OpenAI-compatible API."""

    ExecutorRegistry.register_class("SenseNovaMemoryClient")

    def __init__(
        self,
        name: str,
        db_client: DatabaseMemoryClient,
        sensenova_model_name: str = SENSENOVA_DEFAULT_MODEL,
        sensenova_url: str = SENSENOVA_DEFAULT_BASE_URL,
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
        self.llm_provider_config = build_sensenova_config(
            model_name=sensenova_model_name,
            base_url=sensenova_url,
            timeout=timeout,
            proxy_url=proxy_url,
        )
        self.executor = (
            thread_pool_executor if thread_pool_executor is not None else ThreadPoolExecutor(max_workers=max_workers)
        )
        self.executor_external = True if thread_pool_executor is not None else False

    def __del__(self) -> None:
        if not self.executor_external:
            self.executor.shutdown(wait=True)

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
        try:
            model_name = model_override if model_override else self.sensenova_model_name
            system_content = system_prompt + "\n" + tag_prompt if tag_prompt else system_prompt
            response = await complete(
                api_keys=api_keys,
                config=self.llm_provider_config,
                model_override=model_name,
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_input},
                ],
                temperature=1,
                max_tokens=max_tokens,
                response_format=response_format,
                extra_body=SENSENOVA_EXTRA_BODY,
            )

            if self.input_token_number_histogram:
                self.input_token_number_histogram.labels(adapter=self.name).observe(response.usage.prompt_tokens)
            if self.output_token_number_histogram:
                self.output_token_number_histogram.labels(adapter=self.name).observe(response.usage.completion_tokens)

            content = response.content
            if response_format:
                try:
                    parsed = json.loads(content)
                    if isinstance(parsed, dict) and "output" in parsed:
                        return parsed["output"]
                except json.JSONDecodeError:
                    pass

            match = re.search(r"<output>(.*?)</output>", content, re.DOTALL)
            if match:
                return match.group(1)

            self.logger.warning(f"Failed to extract <output> tag from content: {content}")
            return content
        except Exception as e:
            self.logger.error(f"SenseNova LLM call failed: {type(e).__name__}: {e}")
            raise e
