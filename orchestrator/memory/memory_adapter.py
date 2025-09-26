from abc import abstractmethod
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union

from ..io.memory.database_memory_client import DatabaseMemoryClient
from ..utils.super import Super

if TYPE_CHECKING:
    from .memory_manager import MemoryManager

INITIAL_EMOTION_STATE = {
    "happiness": 15,
    "sadness": 14,
    "fear": 14,
    "anger": 14,
    "disgust": 14,
    "surprise": 15,
    "shyness": 14,
}

INITIAL_RELATIONSHIP_STATE = {
    "stage": "Stranger",
    "value": 0,
}


class BaseMemoryAdapter(Super):
    """Base memory adapter that defines common methods for all memory-related
    operations.

    This class provides the foundation for memory management, including
    building chat history, context, and handling conversations with different
    types of memory data.
    """

    def __init__(
        self,
        name: str,
        db_client: DatabaseMemoryClient,
        conversation_char_threshold: int = 10000,
        conversation_char_target: int = 8000,
        short_term_length_threshold: int = 20,
        short_term_target_size: int = 10,
        medium_term_length_threshold: int = 10,
        logger_cfg: Union[None, Dict[str, Any]] = None,
    ):
        """Initialize the base memory adapter.

        Args:
            name (str):
                Name of the memory adapter.
            db_client (DatabaseMemoryClient):
                Database client for memory operations.
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
            logger_cfg (Union[None, Dict[str, Any]], optional):
                Logger configuration. Defaults to None.
        """
        Super.__init__(self, logger_cfg=logger_cfg)
        self.db_client = db_client
        self.name = name
        self.conversation_char_threshold = conversation_char_threshold
        self.conversation_char_target = conversation_char_target
        self.short_term_length_threshold = short_term_length_threshold
        self.short_term_target_size = short_term_target_size
        self.medium_term_length_threshold = medium_term_length_threshold
        self.logger_cfg = logger_cfg

        self.memory_manager: Optional["MemoryManager"] = None

    def set_memory_manager(self, memory_manager: "MemoryManager") -> None:
        """Set the memory manager instance.

        Args:
            memory_manager (MemoryManager):
                Memory manager instance to set.
        """
        self.memory_manager = memory_manager

    def get_memory_manager(self) -> "MemoryManager":
        """Get the memory manager instance.

        Returns:
            MemoryManager:
                Memory manager instance.

        Raises:
            RuntimeError:
                If memory manager is not initialized.
        """
        if self.memory_manager is None:
            raise RuntimeError("Memory manager not initialized. Call set_memory_manager() first.")
        return self.memory_manager

    async def build_chat_history(self, cascade_memories: Union[None, Dict[str, Any]]):
        """Build chat history from cascade memories.

        Args:
            cascade_memories (Union[None, Dict[str, Any]]):
                Cascade memories containing short-term memories.

        Returns:
            List[Dict[str, str]]:
                Formatted chat history with role and content.
        """
        chat_history = []
        if cascade_memories is None:
            return chat_history
        short_term_memories = cascade_memories.get("short_term_memories", [])
        for entry in short_term_memories:
            if entry["role"] == "user" and "relationship" in entry:
                content = f"{entry['content']} [relationship_stage: {entry['relationship']}]"
            else:
                content = entry["content"]
            chat_history.append({"role": entry["role"], "content": content})
        return chat_history

    async def build_chat_context(
        self, profile_memory: Union[None, Dict[str, Any]], cascade_memories: Union[None, Dict[str, Any]]
    ):
        """Build chat context from profile and cascade memories.

        Args:
            profile_memory (Union[None, Dict[str, Any]]):
                User profile memory data.
            cascade_memories (Union[None, Dict[str, Any]]):
                Cascade memories containing long-term and medium-term memories.

        Returns:
            str:
                Formatted chat context string.
        """
        chat_context = ""
        # User profile
        if profile_memory is not None:
            profile_memory_content = profile_memory.get("content", "")
            chat_context += f"<user_profile>: {profile_memory_content}\n"
        # Long-term + medium-term memories
        if cascade_memories is not None:
            long_term_memory = cascade_memories.get("long_term_memory", {})
            medium_term_memory = cascade_memories.get("medium_term_memory", [])
            long_term_memory_content = long_term_memory.get("content", "")
            medium_term_memory_content = "\n".join([entry.get("content", "") for entry in medium_term_memory])
            chat_context += f"<long_term_memory>: {long_term_memory_content}\n"
            chat_context += f"<medium_term_memory>: {medium_term_memory_content}\n"
        return chat_context

    async def build_user_message(self, message: str, start_time: float, relationship_stage: str):
        """Build formatted user message with timestamp and relationship stage.

        Args:
            message (str):
                User input message.
            start_time (float):
                Unix timestamp of the message.
            relationship_stage (str):
                Current relationship stage.

        Returns:
            str:
                Formatted user message with timestamp and relationship info.
        """
        # Convert timestamp to Beijing time
        beijing_tz = timezone(timedelta(hours=8))
        dt = datetime.fromtimestamp(start_time, tz=beijing_tz)
        time_str = dt.strftime("%Y-%m-%d %H:%M %A")

        user_message = f"user_input: {message} [relationship_stage: {relationship_stage}] [time: {time_str}]"
        return user_message

    @abstractmethod
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
        """Call LLM for text generation.

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
                Generated text content from the LLM.
        """
        raise NotImplementedError

    async def handle_conversation(
        self,
        character_id: str,
        user_input: str,
        profile_memory: Optional[Dict[str, Any]] = None,
        cascade_memories: Optional[Dict[str, Any]] = None,
        relationship: Optional[Tuple[str, int]] = None,
        api_keys: Optional[Dict[str, Any]] = None,
        memory_model_override: Optional[str] = None,
    ) -> None:
        """Handle conversation based on input type and call appropriate
        processing logic.

        Args:
            character_id (str):
                Character identifier.
            user_input (str):
                User input message.
            profile_memory (Optional[Dict[str, Any]], optional):
                User profile memory data. Defaults to None.
            cascade_memories (Optional[Dict[str, Any]], optional):
                Cascade memories data. Defaults to None.
            relationship (Optional[Tuple[str, int]], optional):
                Current relationship stage and value. Defaults to None.
            api_keys (Optional[Dict[str, Any]], optional):
                API keys for the LLM. Defaults to None.
            memory_model_override (Optional[str], optional):
                Memory model override. Defaults to None.
        """
        memory_manager = self.get_memory_manager()
        if memory_manager.is_user_entry(user_input):
            await memory_manager.handle_user_entry(
                character_id, profile_memory, cascade_memories, api_keys, memory_model_override
            )
        else:
            await memory_manager.handle_normal_conversation(
                character_id=character_id,
                user_input=user_input,
                profile_memory=profile_memory,
                cascade_memories=cascade_memories,
                relationship=relationship,
                api_keys=api_keys,
                memory_model_override=memory_model_override,
            )
