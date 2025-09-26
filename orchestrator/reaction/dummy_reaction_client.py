from typing import Any, Dict, Optional, Union

from ..data_structures.reaction import EmotionDelta, ReactionDelta
from .reaction_adapter import ReactionAdapter


class DummyReactionClient(ReactionAdapter):
    """Dummy reaction client that returns empty reaction deltas.

    This client is used for testing or when no actual reaction processing is
    needed. It always returns a ReactionDelta with empty emotion delta, zero
    relationship delta, and empty motion list.
    """

    def __init__(
        self,
        name: str,
        motion_keywords: Union[str, list[str], None],
        proxy_url: Union[None, str] = None,
        logger_cfg: Union[None, Dict[str, Any]] = None,
    ):
        """Initialize the dummy reaction client.

        Args:
            name (str):
                The name of the reaction client.
            motion_keywords (Union[str, list[str], None]):
                The motion keywords.
            proxy_url (Union[None, str], optional):
                The proxy URL for the reaction.
                Defaults to None.
            logger_cfg (Union[None, Dict[str, Any]], optional):
                The logger configuration. Defaults to None.
        """
        super().__init__(
            name=name,
            motion_keywords=motion_keywords,
            proxy_url=proxy_url,
            logger_cfg=logger_cfg,
        )

    async def _init_llm_client(self, request_id: str) -> None:
        """Initialize the LLM client.

        Args:
            request_id (str):
                The request id.
        """
        pass

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
        """Get the reaction delta according to user's text input.

        This dummy implementation always returns an empty reaction delta
        with no emotion changes, zero relationship delta, and empty motion list.

        Args:
            request_id (str):
                The request id.
            prompt (str):
                The system prompt for reaction analysis (unused).
            text (str):
                The agent response text (unused).
            tag (str):
                The tag associated with the response (unused).
            user_input (str):
                The user input text (unused).
            current_emotion (Dict[str, int] | None, optional):
                Current emotion state (unused). Defaults to None.
            current_relationship (Dict[str, Any] | None, optional):
                Current relationship state (unused). Defaults to None.
            response_format (Optional[Dict[str, Any]], optional):
                Response format specification (unused). Defaults to None.
            tag_prompt (Optional[str], optional):
                Tag-specific prompt (unused). Defaults to None.

        Returns:
            ReactionDelta:
                Empty reaction delta with no changes.
        """
        response = {"emotion_delta": EmotionDelta(), "relationship_delta": 0, "motion": []}
        return ReactionDelta(**response)
