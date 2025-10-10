import asyncio
import traceback
from typing import Any, Dict, List, Optional, Union

from ..io.memory.database_memory_client import DatabaseMemoryClient
from ..utils.exception import MissingAPIKeyException, failure_callback
from ..utils.super import Super
from .memory_adapter import BaseMemoryAdapter
from .prompt import (
    LONG_AND_MEDIUM_TERM_SUMMARY_PROMPT,
    LONG_TERM_MEMORY_CONTENT_FORMAT,
    MEDIUM_TERM_MEMORY_CONTENT_FORMAT,
    PROFILE_MEMORY_UPDATE_FORMAT,
    PROFILE_MEMORY_UPDATE_PROMPT,
    SHORT_AND_MEDIUM_TERM_SUMMARY_PROMPT,
    TAG_PROMPT,
)
from .task_manager import MemoryTask, TaskManager


class MemoryProcessor(Super):
    """Memory processor responsible for executing various memory-related tasks.

    This class handles memory compression, summarization, and updates across
    different memory levels (short-term, medium-term, and long-term).
    """

    def __init__(
        self,
        db_client: DatabaseMemoryClient,
        task_manager: TaskManager,
        memory_adapter: BaseMemoryAdapter,
        conversation_char_threshold: int = 10000,
        conversation_char_target: int = 8000,
        short_term_length_threshold: int = 20,
        short_term_target_size: int = 10,
        medium_term_length_threshold: int = 10,
        medium_term_char_threshold: int = 100,
        long_term_char_threshold: int = 100,
        profile_memory_char_threshold: int = 500,
        logger_cfg: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the memory processor.

        Args:
            db_client (DatabaseMemoryClient):
                Database client for memory operations.
            task_manager (TaskManager):
                Task manager for handling memory tasks.
            memory_adapter (BaseMemoryAdapter):
                Memory adapter for LLM calls.
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
            medium_term_char_threshold (int, optional):
                Character threshold for medium-term memory content. Defaults to 100.
            long_term_char_threshold (int, optional):
                Character threshold for long-term memory content. Defaults to 100.
            profile_memory_char_threshold (int, optional):
                Character threshold for profile memory content. Defaults to 500.
            logger_cfg (Optional[Dict[str, Any]], optional):
                Logger configuration. Defaults to None.
        """
        Super.__init__(self, logger_cfg=logger_cfg)
        self.db_client = db_client
        self.task_manager = task_manager
        self.memory_adapter = memory_adapter

        self.conversation_char_threshold = conversation_char_threshold
        self.conversation_char_target = conversation_char_target
        self.short_term_length_threshold = short_term_length_threshold
        self.short_term_target_size = short_term_target_size
        self.medium_term_length_threshold = medium_term_length_threshold
        self.medium_term_char_threshold = medium_term_char_threshold
        self.long_term_char_threshold = long_term_char_threshold
        self.profile_memory_char_threshold = profile_memory_char_threshold

    async def process_turn_level_summary(self, task: MemoryTask) -> bool:
        """Process turn-level summary task.

        Args:
            task (MemoryTask):
                Memory task to process.

        Returns:
            bool:
                True if task completed successfully, False otherwise.
        """
        try:
            task.start()
            character_id = task.character_id
            params = task.params

            api_keys = params.get("api_keys")
            model_override = params.get("model_override")

            cascade_memories = params.get("cascade_memories")
            if cascade_memories is None:
                task.complete("No cascade memories to summarize")
                return True

            short_term_memories = cascade_memories.get("short_term_memories", [])
            medium_term_memories = cascade_memories.get("medium_term_memories", [])

            if not short_term_memories:
                task.complete("No short term memories to summarize")
                return True

            # Get latest medium-term memory
            latest_medium_term_memory = medium_term_memories[-1] if medium_term_memories else None

            # LLM merge all short-term memories and latest medium-term memory
            new_latest_medium_term_content = await self._merge_short_and_medium_term(
                short_term_memories, latest_medium_term_memory, api_keys, model_override
            )

            # Update previous round medium-term memory
            if latest_medium_term_memory:
                await self.db_client.update_medium_term_memory(
                    character_id=character_id,
                    start_timestamp=latest_medium_term_memory["start_timestamp"],
                    content=new_latest_medium_term_content,
                    last_short_term_timestamp=short_term_memories[-1]["chat_timestamp"],
                )
            else:
                self.logger.warning("No latest medium term memory to update")

            task.complete("Medium term summary completed")
            return True

        except Exception as e:
            error_msg = f"Failed to process turn level summary: {str(e)}"
            traceback_str = traceback.format_exc()
            self.logger.error(error_msg + "\n" + traceback_str)

            # Handle MissingAPIKeyException - send failure callback immediately
            if isinstance(e, MissingAPIKeyException):
                msg = f"Missing API key during memory processing: {e}"
                self.logger.error(msg)
                # If callback function exists, send failure callback immediately
                if task.callback_bytes_fn:
                    await self._send_failure_callback(msg, task.callback_bytes_fn)
                # Mark task as failed and don't retry for API key errors
                task.fail(msg)
                return False

            task.fail(error_msg + "\n" + traceback_str)
            return False

    async def process_create_medium_term_memory(self, task: MemoryTask) -> bool:
        """Process create medium-term memory space task.

        Args:
            task (MemoryTask):
                Memory task to process.

        Returns:
            bool:
                True if task completed successfully, False otherwise.
        """
        try:
            task.start()
            character_id = task.character_id
            params = task.params
            unix_timestamp = params.get("unix_timestamp")

            # Create medium-term memory space for this round
            current_timestamp = self.db_client.convert_unix_timestamp_to_str(unix_timestamp)
            await self.db_client.append_medium_term_memory(character_id=character_id, start_timestamp=current_timestamp)

            task.complete("Medium term memory space created")
            return True

        except Exception as e:
            error_msg = f"Failed to create medium term memory space: {str(e)}"
            traceback_str = traceback.format_exc()
            self.logger.error(error_msg + "\n" + traceback_str)
            task.fail(error_msg + "\n" + traceback_str)
            return False

    async def process_profile_memory_update(self, task: MemoryTask) -> bool:
        """Process user profile memory update task.

        Args:
            task (MemoryTask):
                Memory task to process.

        Returns:
            bool:
                True if task completed successfully, False otherwise.
        """
        try:
            task.start()
            character_id = task.character_id
            params = task.params

            api_keys = params.get("api_keys")
            model_override = params.get("model_override")

            profile_memory = params.get("profile_memory")

            # Get unsummarized chat records
            last_profile_timestamp = profile_memory.get("last_short_term_timestamp") if profile_memory else None
            new_short_term_memories = await self.db_client.get_chat_histories_after(
                character_id, last_profile_timestamp
            )

            if not new_short_term_memories:
                task.complete("No new short term memories to update profile")
                return True

            # LLM merge newly acquired chat records with current user profile memory
            new_profile_memory_content = await self._update_profile_memory(
                profile_memory, new_short_term_memories, api_keys, model_override
            )

            # Update profile memory
            last_short_term_timestamp = str(new_short_term_memories[-1]["chat_timestamp"])
            await self.db_client.set_profile_memory(
                character_id=character_id,
                content=new_profile_memory_content,
                last_short_term_timestamp=last_short_term_timestamp,
            )

            task.complete("Profile memory update completed")
            return True

        except Exception as e:
            error_msg = f"Failed to process profile memory update: {str(e)}"
            traceback_str = traceback.format_exc()
            self.logger.error(error_msg + "\n" + traceback_str)

            # Handle MissingAPIKeyException - send failure callback immediately
            if isinstance(e, MissingAPIKeyException):
                msg = f"Missing API key during memory processing: {e}"
                self.logger.error(msg)
                # If callback function exists, send failure callback immediately
                if task.callback_bytes_fn:
                    await self._send_failure_callback(msg, task.callback_bytes_fn)
                # Mark task as failed and don't retry for API key errors
                task.fail(msg)
                return False

            task.fail(error_msg + "\n" + traceback_str)
            return False

    async def need_medium_term_compression(self, cascade_memories: Union[Dict[str, Any], None]) -> bool:
        """Check if medium-term memory compression is needed.

        Args:
            cascade_memories (Union[Dict[str, Any], None]):
                Cascade memories data.

        Returns:
            bool:
                True if compression is needed, False otherwise.
        """
        if cascade_memories is None:
            return False
        medium_term_memories = cascade_memories.get("medium_term_memories", [])
        return len(medium_term_memories) > self.medium_term_length_threshold

    async def process_medium_term_compression(self, task: MemoryTask) -> bool:
        """Process medium-term memory compression task.

        Args:
            task (MemoryTask):
                Memory task to process.

        Returns:
            bool:
                True if task completed successfully, False otherwise.
        """
        try:
            task.start()
            character_id = task.character_id
            params = task.params

            api_keys = params.get("api_keys")
            model_override = params.get("model_override")

            cascade_memories = params.get("cascade_memories", {})

            medium_term_memories = cascade_memories.get("medium_term_memories", [])
            long_term_memory = cascade_memories.get("long_term_memory", {})
            long_term_content = long_term_memory.get("content", "") if long_term_memory else ""

            # LLM merge earliest medium-term memory and long-term memory
            earliest_medium_term = medium_term_memories[0]
            earliest_medium_term_content = earliest_medium_term.get("content", "")
            new_long_term_content = await self._merge_long_term_and_medium_term(
                long_term_content, earliest_medium_term_content, api_keys, model_override
            )

            # Update long-term memory
            last_short_term_timestamp = earliest_medium_term.get("last_short_term_timestamp", "")
            last_medium_term_timestamp = earliest_medium_term.get("start_timestamp", "")
            await self.db_client.set_long_term_memory(
                character_id=character_id,
                content=new_long_term_content,
                last_short_term_timestamp=last_short_term_timestamp,
                last_medium_term_timestamp=last_medium_term_timestamp,
            )

            # Delete earliest medium-term memory
            await self.db_client.remove_medium_term_memory_not_after(
                character_id=character_id, not_after_timestamp=earliest_medium_term["start_timestamp"]
            )

            task.complete("Medium term compression completed")
            return True

        except Exception as e:
            error_msg = f"Failed to process medium term compression: {str(e)}"
            traceback_str = traceback.format_exc()
            self.logger.error(error_msg + "\n" + traceback_str)

            # Handle MissingAPIKeyException - send failure callback immediately
            if isinstance(e, MissingAPIKeyException):
                msg = f"Missing API key during memory processing: {e}"
                self.logger.error(msg)
                # If callback function exists, send failure callback immediately
                if task.callback_bytes_fn:
                    await self._send_failure_callback(msg, task.callback_bytes_fn)
                # Mark task as failed and don't retry for API key errors
                task.fail(msg)
                return False

            task.fail(error_msg + "\n" + traceback_str)
            return False

    async def need_short_term_compression(self, cascade_memories: Union[Dict[str, Any], None]) -> bool:
        """Check if short-term memory compression is needed.

        Args:
            cascade_memories (Union[Dict[str, Any], None]):
                Cascade memories data.

        Returns:
            bool:
                True if compression is needed, False otherwise.
        """
        if cascade_memories is None:
            return False
        short_term_memories = cascade_memories.get("short_term_memories", [])
        return len(short_term_memories) > self.short_term_length_threshold

    async def process_short_term_compression(self, task: MemoryTask) -> bool:
        """Process short-term memory count compression task.

        Args:
            task (MemoryTask):
                Memory task to process.

        Returns:
            bool:
                True if task completed successfully, False otherwise.
        """
        try:
            task.start()
            character_id = task.character_id
            params = task.params

            api_keys = params.get("api_keys")
            model_override = params.get("model_override")

            cascade_memories = params.get("cascade_memories", {})

            short_term_memories = cascade_memories.get("short_term_memories", [])
            medium_term_memories = cascade_memories.get("medium_term_memories", [])

            # Short-term memories that need compression
            count = self.short_term_length_threshold - self.short_term_target_size
            short_term_memories_to_compress = short_term_memories[:count]

            # Get latest medium-term memory
            latest_medium_term = medium_term_memories[-1] if medium_term_memories else None

            # LLM merge short-term memories that need compression and latest medium-term memory
            new_latest_medium_term_content = await self._merge_short_and_medium_term(
                short_term_memories_to_compress, latest_medium_term, api_keys, model_override
            )

            if latest_medium_term:
                # Update medium-term memory
                await self.db_client.update_medium_term_memory(
                    character_id=character_id,
                    start_timestamp=latest_medium_term["start_timestamp"],
                    content=new_latest_medium_term_content,
                    last_short_term_timestamp=short_term_memories_to_compress[-1]["chat_timestamp"],
                )
            else:
                self.logger.warning("No latest medium term memory to update")

            task.complete("Short term compression completed")
            return True

        except Exception as e:
            error_msg = f"Failed to process short term compression: {str(e)}"
            traceback_str = traceback.format_exc()
            self.logger.error(error_msg + "\n" + traceback_str)

            # Handle MissingAPIKeyException - send failure callback immediately
            if isinstance(e, MissingAPIKeyException):
                msg = f"Missing API key during memory processing: {e}"
                self.logger.error(msg)
                # If callback function exists, send failure callback immediately
                if task.callback_bytes_fn:
                    await self._send_failure_callback(msg, task.callback_bytes_fn)
                # Mark task as failed and don't retry for API key errors
                task.fail(msg)
                return False

            task.fail(error_msg + "\n" + traceback_str)
            return False

    async def need_short_term_char_compression(
        self,
        cascade_memories: Union[Dict[str, Any], None],
        profile_memory: Union[Dict[str, Any], None],
        user_input: Optional[str] = None,
        relationship: Optional[str] = None,
    ) -> bool:
        """Check if short-term memory character compression is needed.

        Args:
            cascade_memories (Union[Dict[str, Any], None]):
                Cascade memories data.
            profile_memory (Union[Dict[str, Any], None]):
                User profile memory data.
            user_input (Optional[str], optional):
                User input message. Defaults to None.
            relationship (Optional[str], optional):
                Current relationship stage. Defaults to None.

        Returns:
            bool:
                True if compression is needed, False otherwise.
        """
        if cascade_memories is None:
            return False
        return (
            self._calculate_total_chars(cascade_memories, profile_memory, user_input, relationship)
            > self.conversation_char_threshold
        )

    async def process_short_term_char_compression(self, task: MemoryTask) -> bool:
        """Process short-term memory character compression task.

        Args:
            task (MemoryTask):
                Memory task to process.

        Returns:
            bool:
                True if task completed successfully, False otherwise.
        """
        try:
            task.start()
            character_id = task.character_id
            params = task.params

            api_keys = params.get("api_keys")
            model_override = params.get("model_override")

            cascade_memories = params.get("cascade_memories", {})

            profile_memory = params.get("profile_memory")
            user_input = params.get("user_input")
            relationship = params.get("relationship")

            short_term_memories = cascade_memories.get("short_term_memories", [])
            medium_term_memories = cascade_memories.get("medium_term_memories", [])

            # Calculate current total character count
            total_chars = self._calculate_total_chars(cascade_memories, profile_memory, user_input, relationship)

            # If total character count does not exceed threshold, no compression needed
            if total_chars <= self.conversation_char_threshold:
                task.complete("No compression needed")
                return True

            # Calculate characters to reduce, compress to target value instead of threshold, leave growth space
            medium_chars_added = self.medium_term_char_threshold - len(
                medium_term_memories[-1]["content"] if medium_term_memories else ""
            )
            chars_to_reduce = (total_chars - self.conversation_char_target) + medium_chars_added

            # Start compression from earliest short-term memory until target character count is reached
            count = 0
            current_chars_reduced = 0

            # Calculate number of short-term memories to compress
            for i, memory in enumerate(short_term_memories):
                # Calculate character count of this short-term memory
                role = memory.get("role", "")
                content = memory.get("content", "")
                memory_chars = len(f"{role}: {content}")

                count += 1
                current_chars_reduced += memory_chars

                # If enough characters have been reduced, stop adding
                if current_chars_reduced >= chars_to_reduce:
                    break

            # Short-term memories that need compression
            short_term_memories_to_compress = short_term_memories[:count]

            # Get latest medium-term memory
            latest_medium_term = medium_term_memories[-1] if medium_term_memories else None

            # LLM merge short-term memories that need compression and latest medium-term memory
            new_latest_medium_term_content = await self._merge_short_and_medium_term(
                short_term_memories_to_compress, latest_medium_term, api_keys, model_override
            )

            if latest_medium_term:
                # Update medium-term memory
                await self.db_client.update_medium_term_memory(
                    character_id=character_id,
                    start_timestamp=latest_medium_term["start_timestamp"],
                    content=new_latest_medium_term_content,
                    last_short_term_timestamp=short_term_memories_to_compress[-1]["chat_timestamp"],
                )
            else:
                self.logger.warning("No latest medium term memory to update")

            task.complete("Short term char compression completed")
            return True

        except Exception as e:
            error_msg = f"Failed to process short term char compression: {str(e)}"
            traceback_str = traceback.format_exc()
            self.logger.error(error_msg + "\n" + traceback_str)

            # Handle MissingAPIKeyException - send failure callback immediately
            if isinstance(e, MissingAPIKeyException):
                msg = f"Missing API key during memory processing: {e}"
                self.logger.error(msg)
                # If callback function exists, send failure callback immediately
                if task.callback_bytes_fn:
                    await self._send_failure_callback(msg, task.callback_bytes_fn)
                # Mark task as failed and don't retry for API key errors
                task.fail(msg)
                return False

            task.fail(error_msg + "\n" + traceback_str)
            return False

    async def _merge_short_and_medium_term(
        self,
        short_term_memories: List[Dict[str, Any]],
        latest_medium_term_memory: Union[Dict[str, Any], None],
        api_keys: Optional[Dict[str, Any]] = None,
        model_override: Optional[str] = None,
    ) -> str:
        """Merge short-term memories with the latest medium-term memory.

        Args:
            short_term_memories (List[Dict[str, Any]]):
                List of short-term memories to merge.
            latest_medium_term_memory (Union[Dict[str, Any], None]):
                Latest medium-term memory to merge with.
            api_keys (Optional[Dict[str, Any]], optional):
                API keys for the LLM. Defaults to None.
            model_override (Optional[str], optional):
                Model name override. Defaults to None.

        Returns:
            str:
                Merged medium-term memory content.
        """
        medium_memory_content = latest_medium_term_memory.get("content", "") if latest_medium_term_memory else ""
        short_memory_content = []
        for entry in short_term_memories:
            if entry["role"] == "user" and "relationship" in entry:
                content = f"{entry['content']} [relationship_stage: {entry['relationship']}]"
            else:
                content = entry["content"]
            short_memory_content.append({"role": entry["role"], "content": content})

        user_input = f"""
        <medium_memory_content>: {medium_memory_content}
        <short_memory_content>: {short_memory_content}
        <summary_max_length>: {self.medium_term_char_threshold}
        """

        response = await self.memory_adapter.call_llm(
            system_prompt=SHORT_AND_MEDIUM_TERM_SUMMARY_PROMPT,
            user_input=user_input,
            max_tokens=self.medium_term_char_threshold,
            response_format=MEDIUM_TERM_MEMORY_CONTENT_FORMAT,
            tag_prompt=TAG_PROMPT,
            api_keys=api_keys,
            model_override=model_override,
        )
        return response

    async def _update_profile_memory(
        self,
        profile_memory: Union[Dict[str, Any], None],
        new_short_term_memories: List[Dict[str, Any]],
        api_keys: Optional[Dict[str, Any]] = None,
        model_override: Optional[str] = None,
    ) -> str:
        """Update user profile memory.

        Args:
            profile_memory (Union[Dict[str, Any], None]):
                Current profile memory data.
            new_short_term_memories (List[Dict[str, Any]]):
                New short-term memories to incorporate.
            api_keys (Optional[Dict[str, Any]], optional):
                API keys for the LLM. Defaults to None.
            model_override (Optional[str], optional):
                Model name override. Defaults to None.

        Returns:
            str:
                Updated profile memory content.
        """
        short_memory_content = [
            {"role": entry["role"], "content": entry["content"]} for entry in new_short_term_memories
        ]
        profile_memory_content = profile_memory.get("content", "") if profile_memory else ""

        user_input = f"""
        <profile_memory_content>: {profile_memory_content}
        <short_memory_content>: {short_memory_content}
        <summary_max_length>: {self.profile_memory_char_threshold}
        """

        response = await self.memory_adapter.call_llm(
            system_prompt=PROFILE_MEMORY_UPDATE_PROMPT,
            user_input=user_input,
            max_tokens=self.profile_memory_char_threshold,
            response_format=PROFILE_MEMORY_UPDATE_FORMAT,
            tag_prompt=TAG_PROMPT,
            api_keys=api_keys,
            model_override=model_override,
        )
        return response

    async def _merge_long_term_and_medium_term(
        self,
        long_term_content: str,
        medium_term_content: str,
        api_keys: Optional[Dict[str, Any]] = None,
        model_override: Optional[str] = None,
    ) -> str:
        """Merge long-term memory with the earliest medium-term memory.

        Args:
            long_term_content (str):
                Current long-term memory content.
            medium_term_content (str):
                Earliest medium-term memory content to merge.
            api_keys (Optional[Dict[str, Any]], optional):
                API keys for the LLM. Defaults to None.
            model_override (Optional[str], optional):
                Model name override. Defaults to None.

        Returns:
            str:
                Merged long-term memory content.
        """
        user_input = f"""
        <long_term_content>: {long_term_content}
        <medium_term_content>: {medium_term_content}
        <summary_max_length>: {self.long_term_char_threshold}
        """
        response = await self.memory_adapter.call_llm(
            system_prompt=LONG_AND_MEDIUM_TERM_SUMMARY_PROMPT,
            user_input=user_input,
            max_tokens=self.long_term_char_threshold,
            response_format=LONG_TERM_MEMORY_CONTENT_FORMAT,
            tag_prompt=TAG_PROMPT,
            api_keys=api_keys,
            model_override=model_override,
        )
        return response

    def _calculate_total_chars(
        self,
        cascade_memories: Dict[str, Any],
        profile_memory: Optional[Dict[str, Any]] = None,
        user_input: Optional[str] = None,
        relationship: Optional[str] = None,
    ) -> int:
        """Calculate total character count across all memory levels.

        Args:
            cascade_memories (Dict[str, Any]):
                Cascade memories data.
            profile_memory (Optional[Dict[str, Any]], optional):
                User profile memory data. Defaults to None.
            user_input (Optional[str], optional):
                User input message. Defaults to None.
            relationship (Optional[str], optional):
                Current relationship stage. Defaults to None.

        Returns:
            int:
                Total character count across all memory levels.
        """
        total_chars = 0

        # Long-term memory
        long_term_memory = cascade_memories.get("long_term_memory", {})
        if long_term_memory.get("content"):
            total_chars += len(long_term_memory["content"])

        # Medium-term memory
        medium_term_memories = cascade_memories.get("medium_term_memories", [])
        for memory in medium_term_memories:
            if memory.get("content"):
                total_chars += len(memory["content"])

        # Short-term memory
        short_term_memories = cascade_memories.get("short_term_memories", [])
        for memory in short_term_memories:
            role = memory.get("role", "")
            content = memory.get("content", "")
            total_chars += len(f"{role}: {content}")

        # User profile
        if profile_memory:
            profile_memory_content = profile_memory.get("content", "")
            total_chars += len(f"<user_profile>: {profile_memory_content}\n")

        # User input
        user_message = ""
        if user_input:
            user_message += f"user_input: {user_input} "
        if relationship:
            user_message += f"[relationship_stage: {relationship}] "
        if len(user_message) > 0:
            user_message += "[time: 2024-08-20 18:40 Wednesday]"
            total_chars += len(user_message)

        return total_chars

    async def _send_failure_callback(self, msg: str, callback_bytes_fn) -> None:
        """Send failure callback asynchronously.

        Args:
            msg (str): The error message to send.
            callback_bytes_fn: The callback function that will send the protobuf response bytes.
        """
        try:
            await failure_callback(msg, callback_bytes_fn)
        except Exception as e:
            self.logger.error(f"Failed to send failure callback: {e}")
