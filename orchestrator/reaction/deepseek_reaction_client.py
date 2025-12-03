import asyncio
import re
import time
from typing import Any, Dict, Optional, Union

import httpx
import openai
from prometheus_client import Histogram

from ..data_structures.reaction import ReactionDelta
from ..utils.exception import MissingAPIKeyException
from ..utils.executor_registry import ExecutorRegistry
from .reaction_adapter import ReactionAdapter


class DeepSeekReactionClient(ReactionAdapter):
    """DeepSeek reaction client using DeepSeek API for emotion and motion
    analysis.

    This client uses DeepSeek API to analyze conversation context and generate
    appropriate emotional reactions and motion suggestions for animated agents.
    """

    def __init__(
        self,
        name: str,
        motion_keywords: Union[str, list[str], None],
        deepseek_model_name: str = "deepseek-chat",
        proxy_url: Union[None, str] = None,
        timeout: float = 20.0,
        latency_histogram: Histogram | None = None,
        logger_cfg: Union[None, Dict[str, Any]] = None,
    ):
        """Initialize the DeepSeek reaction client.

        Args:
            name (str):
                The name of the reaction client.
            motion_keywords (Union[str, list[str], None]):
                The motion keywords.
            deepseek_model_name (str, optional):
                The name of the DeepSeek model to use.
                Defaults to "deepseek-chat".
            proxy_url (Union[None, str], optional):
                The proxy URL for the DeepSeek API.
                Defaults to None, use no proxy.
            timeout (float, optional):
                The timeout for the DeepSeek API requests.
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
        self.deepseek_model_name = deepseek_model_name
        self.deepseek_base_url = "https://api.deepseek.com"
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
        deepseek_api_key = self.input_buffer[request_id]["api_keys"].get("deepseek_api_key", "")
        if not deepseek_api_key:
            msg = "DeepSeek API key is not found in the API keys."
            self.logger.error(msg)
            raise MissingAPIKeyException(msg)

        self.input_buffer[request_id]["llm_client"] = openai.AsyncOpenAI(
            api_key=deepseek_api_key,
            base_url=self.deepseek_base_url,
            http_client=self.http_client,
            timeout=self.timeout,
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
        """Get the reaction delta according to user's text input using DeepSeek
        LLM.

        Args:
            request_id (str):
                The request id.
            prompt (str):
                The system prompt for reaction analysis.
            text (str):
                The agent response text to analyze.
            tag (str):
                The tag associated with the response.
            user_input (str):
                The user input text that triggered the agent response.
            current_emotion (Dict[str, int] | None, optional):
                Current emotion state with emotion names as keys and values as integers.
                Defaults to None.
            current_relationship (Dict[str, Any] | None, optional):
                Current relationship state between user and agent.
                Defaults to None.
            response_format (Optional[Dict[str, Any]], optional):
                JSON schema format for structured response. Not supported by DeepSeek API.
                Defaults to None.
            tag_prompt (Optional[str], optional):
                Additional prompt specific to the tag.
                Defaults to None.

        Returns:
            ReactionDelta:
                Reaction delta containing emotion changes, relationship changes,
                and motion suggestions.
        """
        llm_client = self.input_buffer[request_id].get("llm_client", None)
        while llm_client is None:
            await asyncio.sleep(self.sleep_time)
            llm_client = self.input_buffer[request_id].get("llm_client", None)

        model_name_override = self.input_buffer[request_id]["reaction_model_override"]
        deepseek_model_name = model_name_override if model_name_override else self.deepseek_model_name
        system_content = prompt + "\n" + tag_prompt if tag_prompt else prompt

        try:
            user_message_parts = [f"<user_input>: {user_input}"]
            user_message_parts.append(f"<agent_response>: {text}")

            if current_relationship:
                relationship_str = ", ".join([f"{k}: {v}" for k, v in current_relationship.items()])
                user_message_parts.append(f"<current_relationship>: {relationship_str}")

            if current_emotion:
                emotion_str = ", ".join([f"{k}: {v}" for k, v in current_emotion.items()])
                user_message_parts.append(f"<current_emotion>: {emotion_str}")

            if tag:
                user_message_parts.append(f"<tag>: {tag}")

            user_message = "\n".join(user_message_parts)

            start_time = time.time()
            response = await llm_client.chat.completions.create(
                model=deepseek_model_name,
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_message},
                ],
                temperature=1,
                max_tokens=2000,
            )
            response_text = response.choices[0].message.content or ""  # type: ignore

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

            happiness_delta = extract_value(r"<happiness_delta>(-?\d+)</happiness_delta>")
            sadness_delta = extract_value(r"<sadness_delta>(-?\d+)</sadness_delta>")
            fear_delta = extract_value(r"<fear_delta>(-?\d+)</fear_delta>")
            anger_delta = extract_value(r"<anger_delta>(-?\d+)</anger_delta>")
            disgust_delta = extract_value(r"<disgust_delta>(-?\d+)</disgust_delta>")
            surprise_delta = extract_value(r"<surprise_delta>(-?\d+)</surprise_delta>")
            shyness_delta = extract_value(r"<shyness_delta>(-?\d+)</shyness_delta>")
            relationship_delta = extract_value(r"<relationship_delta>(-?\d+)</relationship_delta>")
            speech_keywords = extract_text(r"<speech_keywords>(.*?)</speech_keywords>")
            motion_keywords = extract_text(r"<motion_keywords>(.*?)</motion_keywords>")

            emotion_delta = {
                "happiness_delta": happiness_delta,
                "sadness_delta": sadness_delta,
                "fear_delta": fear_delta,
                "anger_delta": anger_delta,
                "disgust_delta": disgust_delta,
                "surprise_delta": surprise_delta,
                "shyness_delta": shyness_delta,
            }

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
                f"DeepSeek spent {time.time() - start_time} seconds to get reaction delta: {response_delta}"
            )
            return ReactionDelta(**response_delta)

        except Exception as e:
            self.logger.error(f"Reaction error: {e}")
            raise e
