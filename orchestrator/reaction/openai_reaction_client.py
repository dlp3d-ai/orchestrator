import asyncio
import json
import time
from typing import Any, Dict, Optional, Union

from prometheus_client import Histogram

from ..data_structures.reaction import ReactionDelta
from ..llm.openai_chat import OpenAIChatProviderConfig, complete, create_client
from .reaction_adapter import ReactionAdapter


class OpenAIReactionClient(ReactionAdapter):
    """Reaction client using OpenAI's API for emotion and motion analysis.

    This client uses OpenAI's language models to analyze conversation context
    and generate appropriate emotional reactions and motion suggestions for
    animated agents.
    """

    def __init__(
        self,
        name: str,
        motion_keywords: Union[str, list[str], None],
        openai_model_name: str = "gpt-4.1-mini-2025-04-14",
        proxy_url: Union[None, str] = None,
        timeout: float = 10.0,
        latency_histogram: Histogram | None = None,
        logger_cfg: Union[None, Dict[str, Any]] = None,
    ):
        """Initialize the OpenAI reaction client.

        Args:
            name (str):
                The name of the reaction client.
            motion_keywords (Union[str, list[str], None]):
                The motion keywords.
            openai_model_name (str, optional):
                The name of the OpenAI model to use.
                Defaults to "gpt-4.1-mini-2025-04-14".
            proxy_url (Union[None, str], optional):
                The proxy URL for the OpenAI API.
                Defaults to None, use no proxy.
            timeout (float, optional):
                The timeout for the OpenAI API requests.
                Defaults to 10.0.
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
        self.openai_model_name = openai_model_name
        self.timeout = timeout
        self.llm_provider_config = OpenAIChatProviderConfig(
            provider_name="OpenAI",
            api_key_field="openai_api_key",
            model_name=openai_model_name,
            timeout=timeout,
            proxy_url=proxy_url,
        )

    async def _init_llm_client(self, request_id: str) -> None:
        """Initialize the LLM client.

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
        """Get the reaction delta according to user's text input using OpenAI
        LLM.

        Analyzes the conversation context including user input, agent response,
        current emotion state, and relationship status to generate appropriate
        emotional reactions and motion suggestions.

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
                JSON schema format for structured response.
                Defaults to None.
            tag_prompt (Optional[str], optional):
                Additional prompt specific to the tag.
                Defaults to None.

        Returns:
            ReactionDelta:
                Reaction delta containing emotion changes, relationship changes,
                and motion suggestions.

        Raises:
            Exception:
                If the OpenAI API request fails or response parsing fails.
        """
        llm_client = self.input_buffer[request_id].get("llm_client", None)
        while llm_client is None:
            await asyncio.sleep(self.sleep_time)
            llm_client = self.input_buffer[request_id].get("llm_client", None)

        model_name_override = self.input_buffer[request_id]["reaction_model_override"]
        openai_model_name = model_name_override if model_name_override else self.openai_model_name
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
            response = await complete(
                client=llm_client,
                api_keys=None,
                config=self.llm_provider_config,
                model_override=openai_model_name,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=1,
                response_format=response_format,  # type: ignore
            )
            response_delta = json.loads(response.content)
            response_delta["speech_text"] = text
            self.logger.debug(
                f"openai spent {time.time() - start_time} seconds to get reaction delta: {response_delta}"
            )
            return ReactionDelta(**response_delta)

        except Exception as e:
            self.logger.error(f"Reaction error: {e}")
            raise e
