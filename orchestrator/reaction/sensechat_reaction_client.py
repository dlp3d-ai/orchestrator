import asyncio
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional, Union

import httpx
import jwt
from prometheus_client import Histogram

from ..data_structures.reaction import ReactionDelta
from ..utils.exception import MissingAPIKeyException
from ..utils.executor_registry import ExecutorRegistry
from .reaction_adapter import ReactionAdapter


class SenseChatReactionClient(ReactionAdapter):
    """SenseChat reaction client using SenseChat API for emotion and motion
    analysis.

    This client uses SenseChat API to analyze conversation context and generate
    appropriate emotional reactions and motion suggestions for animated agents.
    """

    ExecutorRegistry.register_class("SenseChatReactionClient")

    def __init__(
        self,
        name: str,
        motion_keywords: Union[str, list[str], None],
        sensechat_model_name: str = "SenseChat-5-1202",
        sensechat_url: str = "https://api.sensenova.cn/v1/llm/chat-completions",
        proxy_url: Union[None, str] = None,
        timeout: float = 10.0,
        max_workers: int = 1,
        thread_pool_executor: ThreadPoolExecutor | None = None,
        latency_histogram: Histogram | None = None,
        logger_cfg: Union[None, Dict[str, Any]] = None,
    ):
        """Initialize the SenseChat reaction client.

        Args:
            name (str):
                The name of the reaction client.
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
                Defaults to 10.0.
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
        """Get the reaction delta according to user's text input using
        SenseChat LLM.

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
                Response format specification. Not supported by SenseChat API.
                Defaults to None.
            tag_prompt (Optional[str], optional):
                Additional prompt specific to the tag.
                Defaults to None.

        Returns:
            ReactionDelta:
                Reaction delta containing emotion changes, relationship changes,
                and motion suggestions.
        """
        api_keys = self.input_buffer[request_id]["api_keys"]
        if not api_keys.get("sensechat_ak") or not api_keys.get("sensechat_sk"):
            msg = "SenseChat API key or secret key is not found in the API keys."
            self.logger.error(msg)
            raise MissingAPIKeyException(msg)

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
            loop = asyncio.get_running_loop()
            jwt_token = await loop.run_in_executor(self.executor, self._gen_token, api_keys)

            model_name_override = self.input_buffer[request_id]["reaction_model_override"]
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
                "max_new_tokens": 2000,
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
                f"SenseChat spent {time.time() - start_time} seconds to get reaction delta: {response_delta}"
            )
            return ReactionDelta(**response_delta)
        except Exception as e:
            self.logger.error(f"Reaction error: {e}")
            raise e
