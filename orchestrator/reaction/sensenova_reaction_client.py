import asyncio
import re
import time
from typing import Any, Dict, Optional, Union

from prometheus_client import Histogram

from ..data_structures.reaction import ReactionDelta
from ..llm.openai_chat import complete, create_client
from ..llm.sensenova import (
    SENSENOVA_DEFAULT_BASE_URL,
    SENSENOVA_DEFAULT_MODEL,
    SENSENOVA_EXTRA_BODY,
    build_sensenova_config,
)
from ..utils.executor_registry import ExecutorRegistry
from .reaction_adapter import ReactionAdapter


class SenseNovaReactionClient(ReactionAdapter):
    """SenseNova reaction client using the OpenAI-compatible API."""

    ExecutorRegistry.register_class("SenseNovaReactionClient")

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

    async def get_reaction_delta(
        self,
        request_id: str,
        prompt: str,
        text: str,
        tag: str,
        user_input: str,
        current_emotion: Dict[str, int] | None = None,
        current_relationship: Dict[str, Any] | None = None,
        response_format: Optional[Dict[str, Any]] = None,
        tag_prompt: Optional[str] = None,
    ) -> ReactionDelta:
        """Get the reaction delta according to user and agent text.

        Args:
            request_id (str):
                The request id.
            prompt (str):
                System prompt for reaction analysis.
            text (str):
                Agent response text.
            tag (str):
                Response tag.
            user_input (str):
                User input text.
            current_emotion (Dict[str, int] | None, optional):
                Current emotion state. Defaults to None.
            current_relationship (Dict[str, Any] | None, optional):
                Current relationship state. Defaults to None.
            response_format (Optional[Dict[str, Any]], optional):
                Optional structured-output format. Defaults to None.
            tag_prompt (Optional[str], optional):
                Extra tag prompt appended to the system prompt. Defaults to
                None.

        Returns:
            ReactionDelta:
                Parsed emotion, relationship, and motion deltas.
        """
        llm_client = self.input_buffer[request_id].get("llm_client", None)
        while llm_client is None:
            await asyncio.sleep(self.sleep_time)
            llm_client = self.input_buffer[request_id].get("llm_client", None)

        try:
            user_message_parts = [f"<user_input>: {user_input}", f"<agent_response>: {text}"]
            if current_relationship:
                relationship_str = ", ".join([f"{k}: {v}" for k, v in current_relationship.items()])
                user_message_parts.append(f"<current_relationship>: {relationship_str}")
            if current_emotion:
                emotion_str = ", ".join([f"{k}: {v}" for k, v in current_emotion.items()])
                user_message_parts.append(f"<current_emotion>: {emotion_str}")
            if tag:
                user_message_parts.append(f"<tag>: {tag}")

            user_message = "\n".join(user_message_parts)
            model_name_override = self.input_buffer[request_id]["reaction_model_override"]
            model_name = model_name_override if model_name_override else self.sensenova_model_name
            system_content = prompt + "\n" + tag_prompt if tag_prompt else prompt

            start_time = time.time()
            response = await complete(
                client=llm_client,
                api_keys=None,
                config=self.llm_provider_config,
                model_override=model_name,
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_message},
                ],
                temperature=1,
                response_format=response_format,
                extra_body=SENSENOVA_EXTRA_BODY,
            )
            response_text = response.content

            def extract_value(pattern, default=0):
                match = re.search(pattern, response_text)
                if match:
                    try:
                        return int(match.group(1))
                    except ValueError:
                        return default
                return default

            def extract_text(pattern, default=""):
                match = re.search(pattern, response_text)
                if match:
                    return match.group(1)
                return default

            emotion_delta = {
                "happiness_delta": extract_value(r"<happiness_delta>(-?\d+)</happiness_delta>"),
                "sadness_delta": extract_value(r"<sadness_delta>(-?\d+)</sadness_delta>"),
                "fear_delta": extract_value(r"<fear_delta>(-?\d+)</fear_delta>"),
                "anger_delta": extract_value(r"<anger_delta>(-?\d+)</anger_delta>"),
                "disgust_delta": extract_value(r"<disgust_delta>(-?\d+)</disgust_delta>"),
                "surprise_delta": extract_value(r"<surprise_delta>(-?\d+)</surprise_delta>"),
                "shyness_delta": extract_value(r"<shyness_delta>(-?\d+)</shyness_delta>"),
            }
            relationship_delta = extract_value(r"<relationship_delta>(-?\d+)</relationship_delta>")
            speech_keywords = extract_text(r"<speech_keywords>(.*?)</speech_keywords>")
            motion_keywords = extract_text(r"<motion_keywords>(.*?)</motion_keywords>")

            motion = []
            if speech_keywords and motion_keywords:
                motion.append({"speech_keywords": speech_keywords, "motion_keywords": motion_keywords})

            response_delta = {
                "emotion_delta": emotion_delta,
                "relationship_delta": relationship_delta,
                "motion": motion,
                "speech_text": text,
            }

            self.logger.debug(
                f"SenseNova spent {time.time() - start_time} seconds to get reaction delta: {response_delta}"
            )
            return ReactionDelta(**response_delta)
        except Exception as e:
            self.logger.error(f"Reaction error: {e}")
            raise e
