import asyncio
from typing import Any, Dict, Optional, Union

from prometheus_client import Histogram

from ..data_structures.classification import ClassificationType
from ..llm.openai_chat import complete, create_client
from ..llm.sensenova import (
    SENSENOVA_DEFAULT_BASE_URL,
    SENSENOVA_DEFAULT_MODEL,
    SENSENOVA_EXTRA_BODY,
    build_sensenova_config,
)
from ..utils.executor_registry import ExecutorRegistry
from .classification_adapter import ClassificationAdapter


class SenseNovaClassificationClient(ClassificationAdapter):
    """Classification client for SenseNova OpenAI-compatible API."""

    ExecutorRegistry.register_class("SenseNovaClassificationClient")

    def __init__(
        self,
        name: str,
        motion_keywords: Union[str, list[str], None],
        sensenova_model_name: str = SENSENOVA_DEFAULT_MODEL,
        sensenova_url: str = SENSENOVA_DEFAULT_BASE_URL,
        proxy_url: Union[None, str] = None,
        timeout: float = 10.0,
        max_workers: int = 1,
        thread_pool_executor: Any | None = None,
        latency_histogram: Histogram | None = None,
        logger_cfg: Union[None, Dict[str, Any]] = None,
    ):
        super().__init__(
            name=name,
            motion_keywords=motion_keywords,
            proxy_url=proxy_url,
            latency_histogram=latency_histogram,
            logger_cfg=logger_cfg,
        )
        self.sensenova_model_name = sensenova_model_name
        self.sensenova_url = sensenova_url
        self.timeout = timeout
        self.llm_provider_config = build_sensenova_config(
            model_name=sensenova_model_name,
            base_url=sensenova_url,
            timeout=timeout,
            proxy_url=proxy_url,
        )

    async def _init_llm_client(self, request_id: str) -> None:
        """Initialize the SenseNova LLM client for a request.

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
        """Classify the response type according to the user's text input.

        Args:
            request_id (str):
                The request id.
            prompt (str):
                System prompt used for classification.
            text (str):
                User text to classify.
            response_format (Optional[Dict[str, Any]], optional):
                Optional structured-output format. Defaults to None.
            tag_prompt (Optional[str], optional):
                Extra tag prompt appended to the system prompt. Defaults to
                None.

        Returns:
            ClassificationType:
                Classification result parsed as accept, reject, or leave.
        """
        llm_client = self.input_buffer[request_id].get("llm_client", None)
        while llm_client is None:
            await asyncio.sleep(self.sleep_time)
            llm_client = self.input_buffer[request_id].get("llm_client", None)

        model_name_override = self.input_buffer[request_id]["classification_model_override"]
        model_name = model_name_override if model_name_override else self.sensenova_model_name
        system_content = prompt + "\n" + tag_prompt if tag_prompt else prompt
        try:
            response = await complete(
                client=llm_client,
                api_keys=None,
                config=self.llm_provider_config,
                model_override=model_name,
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": f"<user_input>: {text}"},
                ],
                temperature=1,
                max_tokens=1000,
                response_format=response_format,
                extra_body=SENSENOVA_EXTRA_BODY,
            )
            response_text = response.content
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
